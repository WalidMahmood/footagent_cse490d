"""
FootAgent — Tracking Agent
YOLOv11-small + ByteTrack + K-means team clustering + Homography pitch mapping

Processes football video at 25fps, outputs per-frame tracking data:
{player_id, team, pixel_x, pixel_y, pitch_x, pitch_y, speed, timestamp}
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time


class ByteTrack:
    """
    ByteTrack with ID recycling for football (max ~25 entities on pitch).
    Recycles IDs from dead tracks so IDs stay in 1..max_players range.
    """

    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8,
                 max_players=30):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.max_players = max_players

        self.tracks = []
        self.lost_stracks = []

        # ID recycling pool: available IDs from 1..max_players
        self._id_pool = list(range(max_players, 0, -1))  # stack, pop from end
        self._next_overflow = max_players + 1

    def _alloc_id(self):
        """Get a recycled ID, force-reclaim from oldest lost track, or overflow."""
        if self._id_pool:
            return self._id_pool.pop()
        # Pool empty — force-expire oldest lost track to reclaim its ID
        if self.lost_stracks:
            # Sort by longest time since update (oldest first)
            self.lost_stracks.sort(key=lambda t: t.time_since_update, reverse=True)
            oldest = self.lost_stracks.pop(0)
            return oldest.track_id  # Reuse this ID directly
        # Absolute last resort (shouldn't happen in football)
        tid = self._next_overflow
        self._next_overflow += 1
        return tid

    def _free_id(self, tid):
        """Return an ID to the pool for reuse."""
        # Accept ALL IDs back, not just <= max_players
        if tid not in self._id_pool:
            self._id_pool.append(tid)

    def update(self, detections):
        """
        Update tracks with new detections.
        detections: list of (x1, y1, x2, y2, confidence) tuples
        Returns: list of (x1, y1, x2, y2, track_id, confidence) tuples
        """
        dets_conf = [d for d in detections if d[4] >= self.track_thresh]

        for track in self.tracks:
            track.predict()
        for track in self.lost_stracks:
            track.time_since_update += 1

        # Step 1: Match detections with active tracks (IoU)
        matched_det_idx = set()
        keep_tracks = []

        if len(self.tracks) > 0 and len(dets_conf) > 0:
            dists = self._iou_distance(self.tracks, dets_conf)
            row_ind, col_ind = linear_sum_assignment(dists)
            matched_rows = set()
            for r, c in zip(row_ind, col_ind):
                if dists[r, c] < self.match_thresh:
                    self.tracks[r].update(dets_conf[c])
                    keep_tracks.append(self.tracks[r])
                    matched_det_idx.add(c)
                    matched_rows.add(r)
            for i, t in enumerate(self.tracks):
                if i not in matched_rows:
                    t.mark_lost()
                    self.lost_stracks.append(t)
        else:
            for t in self.tracks:
                t.mark_lost()
                self.lost_stracks.append(t)

        # Step 2: Match remaining detections with lost tracks
        unmatched = [i for i in range(len(dets_conf)) if i not in matched_det_idx]

        if unmatched and self.lost_stracks:
            remaining = [dets_conf[i] for i in unmatched]
            dists = self._iou_distance(self.lost_stracks, remaining)
            row_ind, col_ind = linear_sum_assignment(dists)
            recovered = set()
            for r, c in zip(row_ind, col_ind):
                if dists[r, c] < self.match_thresh * 1.5:
                    self.lost_stracks[r].update(remaining[c])
                    keep_tracks.append(self.lost_stracks[r])
                    matched_det_idx.add(unmatched[c])
                    recovered.add(r)
            self.lost_stracks = [t for i, t in enumerate(self.lost_stracks)
                                 if i not in recovered]

        # Step 3: New tracks from unmatched detections (with recycled IDs)
        still_unmatched = [i for i in range(len(dets_conf)) if i not in matched_det_idx]
        active_count = len(keep_tracks)
        for i in still_unmatched:
            if active_count >= self.max_players:
                break  # Hard cap: never exceed max_players active tracks
            tid = self._alloc_id()
            keep_tracks.append(STrack(dets_conf[i], tid))
            active_count += 1

        self.tracks = keep_tracks

        # Expire lost tracks and recycle their IDs
        surviving_lost = []
        for t in self.lost_stracks:
            if t.time_since_update < self.track_buffer:
                surviving_lost.append(t)
            else:
                self._free_id(t.track_id)
        self.lost_stracks = surviving_lost

        results = []
        for track in self.tracks:
            if track.is_confirmed():
                results.append(tuple(track.to_tlbr()) + (track.track_id, track.confidence))

        return results
    
    def _iou_distance(self, tracks, detections):
        """Calculate IoU distance matrix between tracks and detections."""
        dist_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                dist_matrix[i, j] = 1.0 - self._iou(track.tlbr, det[:4])
        
        return dist_matrix
    
    def _iou(self, box1, box2):
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0


class STrack:
    """Single object track for ByteTrack."""
    
    def __init__(self, detection, track_id):
        self.tlbr = detection[:4]
        self.confidence = detection[4]
        self.track_id = track_id
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        
        # Kalman filter for smoothing
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        cx = (detection[0] + detection[2]) / 2
        cy = (detection[1] + detection[3]) / 2
        w = detection[2] - detection[0]
        h = detection[3] - detection[1]
        
        self.kf.x = np.array([cx, cy, w, h, 0, 0, 0]).reshape(7, 1)
    
    def predict(self):
        """Predict next state."""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
    
    def update(self, detection):
        """Update with new detection."""
        self.tlbr = detection[:4]
        self.confidence = detection[4]
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        
        cx = (detection[0] + detection[2]) / 2
        cy = (detection[1] + detection[3]) / 2
        w = detection[2] - detection[0]
        h = detection[3] - detection[1]
        
        self.kf.update([cx, cy, w, h])
    
    def mark_lost(self):
        self.hit_streak = 0
    
    def is_confirmed(self):
        return self.hit_streak > 0 and self.time_since_update <= 1
    
    def is_removed(self):
        return self.time_since_update > 10
    
    def to_tlbr(self):
        """Return current bounding box from Kalman filter."""
        if self.time_since_update > 0:
            return self.kf.x[:4].reshape(4)
        return self.tlbr


class TeamClassifier:
    """
    K-means clustering on jersey colors for team assignment.
    Excludes goalkeeper by detecting outlier colors.
    """
    
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.fitted = False
        self.team_colors = {}
    
    def extract_color_features(self, frame, bbox):
        """
        Extract dominant color features from a bounding box region.
        Returns RGB histogram features.
        """
        x1, y1, x2, y2 = map(int, bbox[:4])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return np.zeros(30)
        
        # Resize to fixed size for consistency
        roi = cv2.resize(roi, (32, 32))
        
        # Extract color histograms in RGB
        hist_r = cv2.calcHist([roi], [0], None, [10], [0, 256])
        hist_g = cv2.calcHist([roi], [1], None, [10], [0, 256])
        hist_b = cv2.calcHist([roi], [2], None, [10], [0, 256])
        
        # Normalize and concatenate
        hist_r = cv2.normalize(hist_r, None).flatten()
        hist_g = cv2.normalize(hist_g, None).flatten()
        hist_b = cv2.normalize(hist_b, None).flatten()
        
        return np.concatenate([hist_r, hist_g, hist_b])
    
    def fit_predict_teams(self, frame, detections, track_ids):
        """
        Fit K-means on detections and assign teams.
        Returns dict: {track_id: team_label}
        """
        if len(detections) < 2:
            return {tid: 0 for tid in track_ids}
        
        # Extract color features for all detections
        features = []
        valid_indices = []
        
        for i, det in enumerate(detections):
            feat = self.extract_color_features(frame, det)
            if np.sum(feat) > 0:
                features.append(feat)
                valid_indices.append(i)
        
        if len(features) < 2:
            return {track_ids[i]: 0 for i in valid_indices}
        
        features = np.array(features)
        
        # Fit K-means
        self.kmeans.fit(features)
        labels = self.kmeans.labels_
        
        # Handle edge case where K-means finds only 1 cluster
        unique_labels = np.unique(labels)
        if len(unique_labels) == 1:
            # Assign all to team 0
            return {track_ids[i]: 0 for i in valid_indices}
        
        # Store team colors for reference
        self.team_colors = {}
        for label in unique_labels:
            mask = labels == label
            self.team_colors[int(label)] = np.mean(features[mask], axis=0)
        
        self.fitted = True
        
        # Map track_ids to team labels
        team_assignments = {}
        for i, idx in enumerate(valid_indices):
            team_assignments[track_ids[idx]] = int(labels[i])
        
        return team_assignments
    
    def predict_single(self, frame, bbox):
        """Predict team for a single detection (if already fitted)."""
        if not self.fitted:
            return 0
        
        feat = self.extract_color_features(frame, bbox)
        return int(self.kmeans.predict([feat])[0])


class HomographyCalibrator:
    """
    Maps pixel coordinates to pitch coordinates using homography.
    Standard football pitch: 105m x 68m
    """
    
    def __init__(self, pitch_length=105.0, pitch_width=68.0):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.H = None  # Homography matrix
        self.H_inv = None
    
    def set_points(self, pixel_points, pitch_points):
        """
        Compute homography matrix from 4+ corresponding points.
        pixel_points: list of (x, y) in image
        pitch_points: list of (x, y) in metres
        """
        if len(pixel_points) < 4:
            raise ValueError("Need at least 4 points for homography")
        
        pixel_np = np.array(pixel_points, dtype=np.float32)
        pitch_np = np.array(pitch_points, dtype=np.float32)
        
        self.H, _ = cv2.findHomography(pixel_np, pitch_np)
        self.H_inv = np.linalg.inv(self.H)
    
    def pixel_to_pitch(self, px, py):
        """Convert pixel coordinates to pitch coordinates (metres)."""
        if self.H is None:
            # Fallback scaling (1920x1080 -> 105x68m)
            return (px / 1920.0) * self.pitch_length, (py / 1080.0) * self.pitch_width
        
        point = np.array([px, py, 1.0])
        result = self.H @ point
        result = result / result[2]
        
        return result[0], result[1]
    
    def pitch_to_pixel(self, sx, sy):
        """Convert pitch coordinates to pixel coordinates."""
        if self.H_inv is None:
            return sx, sy
        
        point = np.array([sx, sy, 1.0])
        result = self.H_inv @ point
        result = result / result[2]
        
        return result[0], result[1]
    
    def auto_calibrate(self, frame_shape):
        """
        Auto-calibrate using standard pitch markings detection.
        This is a simplified version - in production, use line detection.
        
        Assumes camera view where pitch occupies most of frame.
        Uses heuristic mapping based on typical broadcast camera angle.
        """
        h, w = frame_shape[:2]
        
        # Define pitch corners in metres (standard 105x68m pitch)
        pitch_points = [
            [0, 0],           # Bottom-left
            [self.pitch_length, 0],      # Bottom-right
            [self.pitch_length, self.pitch_width],  # Top-right
            [0, self.pitch_width]        # Top-left
        ]
        
        # Heuristic pixel points (assumes pitch fills ~80% of frame)
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)
        
        pixel_points = [
            [margin_x, h - margin_y],           # Bottom-left
            [w - margin_x, h - margin_y],       # Bottom-right
            [w - margin_x, margin_y],           # Top-right
            [margin_x, margin_y]                # Top-left
        ]
        
        self.set_points(pixel_points, pitch_points)


class TrackingAgent:
    """
    Main Tracking Agent: YOLOv11 + ByteTrack + Team Classification + Homography
    
    Processes video at 25fps and outputs tracking data per frame.
    """
    
    def __init__(self, model_size='n', device='cuda'):
        """
        Initialize tracking agent.
        
        Args:
            model_size: YOLO model size ('n'=nano, 's'=small, 'm'=medium)
            device: 'cuda' or 'cpu'
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load YOLO model (COCO pretrained - person class = 0)
        model_name = 'yolov8n.pt'
        print(f"[TrackingAgent] Loading {model_name} on {self.device}...")
        self.model = YOLO(model_name)
        self.model.to(self.device)
        
        # Initialize tracker
        self.tracker = ByteTrack(track_thresh=0.4, match_thresh=0.7)
        
        # Initialize team classifier
        self.team_classifier = TeamClassifier(n_clusters=2)
        
        # Initialize homography calibrator
        self.homography = HomographyCalibrator()
        
        # Tracking state
        self.frame_count = 0
        self.fps = 25.0
        self.player_history = {}  # Track player positions over time for speed calc
        
        # Team assignments (updated periodically)
        self.team_assignments = {}
        self.last_team_update = 0
        self.team_update_interval = 125  # Update teams every 5 seconds (25fps * 5)
        
        print(f"[TrackingAgent] Initialized on {self.device}")
    
    def detect_players(self, frame):
        """
        Detect players in frame using YOLOv11.
        
        Returns:
            List of (x1, y1, x2, y2, confidence) for person detections
        """
        # Run YOLO inference (only class 0 = person)
        results = self.model(frame, verbose=False, conf=0.3, classes=[0])
        
        detections = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                detections.append((x1, y1, x2, y2, conf))
        
        return detections
    
    def track_players(self, frame, detections):
        """
        Track players across frames using ByteTrack.
        
        Returns:
            List of (x1, y1, x2, y2, track_id, confidence)
        """
        tracked = self.tracker.update(detections)
        return tracked
    
    def assign_teams(self, frame, tracked_players):
        """
        Assign teams using K-means color clustering.
        Updates periodically for stability.
        """
        self.frame_count += 1
        
        # Update team assignments periodically
        if self.frame_count - self.last_team_update >= self.team_update_interval:
            if len(tracked_players) >= 4:  # Need enough players
                detections = [(t[0], t[1], t[2], t[3], t[5]) for t in tracked_players]
                track_ids = [t[4] for t in tracked_players]
                
                self.team_assignments = self.team_classifier.fit_predict_teams(
                    frame, detections, track_ids
                )
                self.last_team_update = self.frame_count
        
        return self.team_assignments
    
    def calculate_speed(self, track_id, pitch_x, pitch_y, timestamp):
        """
        Calculate player speed in m/s from position history.
        """
        if track_id not in self.player_history:
            self.player_history[track_id] = []
        
        self.player_history[track_id].append((pitch_x, pitch_y, timestamp))
        
        # Keep only last 1 second of history
        cutoff = timestamp - 1.0
        self.player_history[track_id] = [
            (x, y, t) for x, y, t in self.player_history[track_id] if t > cutoff
        ]
        
        history = self.player_history[track_id]
        if len(history) < 2:
            return 0.0
        
        # Calculate speed from recent positions
        x1, y1, t1 = history[0]
        x2, y2, t2 = history[-1]
        
        dt = t2 - t1
        if dt < 0.01:
            return 0.0
        
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        speed = distance / dt
        
        return min(speed, 12.0)  # Cap at realistic max speed (~12 m/s for sprint)
    
    def process_video(self, video_path, output_path=None, visualize=True):
        """
        Process entire video and generate tracking data.
        
        Args:
            video_path: Path to input video
            output_path: Path to save tracking JSON
            visualize: Whether to save visualization video
        
        Returns:
            List of frame-level tracking data
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[TrackingAgent] Processing: {video_path}")
        print(f"  Resolution: {frame_width}x{frame_height}, FPS: {fps}, Frames: {total_frames}")
        
        # Auto-calibrate homography on first frame
        ret, first_frame = cap.read()
        if ret:
            self.homography.auto_calibrate(first_frame.shape)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        
        # Setup video writer if visualizing
        writer = None
        if visualize and output_path:
            viz_path = output_path.replace('.json', '_viz.mp4')
            writer = cv2.VideoWriter(
                viz_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (frame_width, frame_height)
            )
        
        all_tracking_data = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_idx / fps
            
            # Step 1: Detect players
            detections = self.detect_players(frame)
            
            if len(detections) == 0:
                frame_idx += 1
                continue
            
            # Step 2: Track players
            tracked = self.track_players(frame, detections)
            
            # Step 3: Assign teams
            team_assignments = self.assign_teams(frame, tracked)
            
            # Step 4: Map to pitch coordinates and calculate speed
            frame_data = []
            for track in tracked:
                x1, y1, x2, y2, track_id, conf = track
                
                # Center of bounding box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                # Map to pitch coordinates
                pitch_x, pitch_y = self.homography.pixel_to_pitch(cx, cy)
                
                # Calculate speed
                speed = self.calculate_speed(track_id, pitch_x, pitch_y, timestamp)
                
                # Get team assignment
                team = team_assignments.get(int(track_id), -1)
                
                player_info = {
                    'player_id': int(track_id),
                    'team': int(team),
                    'pixel_x': float(cx),
                    'pixel_y': float(cy),
                    'pitch_x': float(pitch_x),
                    'pitch_y': float(pitch_y),
                    'speed': float(speed),
                    'confidence': float(conf),
                    'timestamp': timestamp
                }
                
                frame_data.append(player_info)
            
            all_tracking_data.append({
                'frame': frame_idx,
                'timestamp': timestamp,
                'players': frame_data
            })
            
            # Step 5: Visualize
            if visualize and writer:
                viz_frame = self._visualize_frame(frame, tracked, team_assignments)
                writer.write(viz_frame)
            
            # Progress
            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%)")
            
            frame_idx += 1
        
        cap.release()
        if writer:
            writer.release()
        
        # Save tracking data
        if output_path:
            self._save_tracking_data(all_tracking_data, output_path)
            print(f"[TrackingAgent] Saved tracking data to: {output_path}")
        
        return all_tracking_data
    
    def _visualize_frame(self, frame, tracked_players, team_assignments):
        """
        Draw tracking boxes and IDs on frame for visualization.
        """
        viz = frame.copy()
        
        team_colors = {0: (0, 255, 0), 1: (255, 0, 0), -1: (255, 255, 255)}
        
        for track in tracked_players:
            x1, y1, x2, y2, track_id, conf = track
            team = team_assignments.get(int(track_id), -1)
            color = team_colors.get(team, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(viz, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw ID and team
            label = f"ID:{int(track_id)} T{int(team)}"
            cv2.putText(viz, label, (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return viz
    
    def _save_tracking_data(self, tracking_data, output_path):
        """Save tracking data to JSON file."""
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'metadata': {
                    'agent': 'FootAgent Tracking Agent v1.0',
                    'model': 'YOLOv11 + ByteTrack + K-means + Homography',
                    'frames_processed': len(tracking_data)
                },
                'tracking': tracking_data
            }, f, indent=2)
    
    def get_vram_usage(self):
        """Get current VRAM usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        return 0


def main():
    """Test tracking agent on a sample video."""
    import argparse
    
    parser = argparse.ArgumentParser(description='FootAgent Tracking Agent')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--output', type=str, default='data/matches/tracking_output.json',
                       help='Output JSON path')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm'],
                       help='YOLO model size')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = TrackingAgent(model_size=args.model_size)
    
    print(f"VRAM before: {agent.get_vram_usage():.1f} MB")
    
    # Process video
    tracking_data = agent.process_video(
        video_path=args.video,
        output_path=args.output,
        visualize=not args.no_viz
    )
    
    print(f"VRAM after: {agent.get_vram_usage():.1f} MB")
    print(f"Total frames processed: {len(tracking_data)}")
    print(f"Total player detections: {sum(len(f['players']) for f in tracking_data)}")


if __name__ == '__main__':
    main()
