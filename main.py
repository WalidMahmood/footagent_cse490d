"""
FootAgent — Research-Grade Football Intelligence Pipeline
Bismillah.

Pipeline: Football YOLO (4-class) -> ByteTrack -> Homography -> Temporal GAT (OBC)
Classes: player, goalkeeper, referee, ball

Usage:
  python main.py                                      # Default test clip
  python main.py --video path/to/clip.mp4             # Custom clip
  python main.py --start-frame 90 --max-frames 500    # Subset
  python main.py --save output.mp4                    # Save to file
  python main.py --report                             # Generate match report
  python main.py --serve                              # Start backend + 3D dashboard
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from ultralytics import YOLO

from agents.tracking_agent import ByteTrack, TeamClassifier, HomographyCalibrator
from models.gat.temporal_gat import TemporalGAT

EDGE_THRESHOLD_M = 17.5
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
FOOTBALL_YOLO_PATH = 'weights/yolo-football-player-detection.pt'
GAT_WEIGHTS_PATH = 'weights/gat_temporal.pt'

CLASS_NAMES = {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}


class FootballDetector:
    """Football-specific YOLO detector with 4 classes."""

    def __init__(self, weights_path=FOOTBALL_YOLO_PATH, fallback='yolov8n.pt'):
        if Path(weights_path).exists():
            self.model = YOLO(weights_path)
            self.football_mode = True
            print(f"  YOLO: Football model ({CLASS_NAMES})")
        else:
            self.model = YOLO(fallback)
            self.football_mode = False
            print(f"  YOLO: Generic COCO (person class only)")

    def detect(self, frame, conf=0.25):
        """
        Returns list of dicts with keys:
            x1, y1, x2, y2, confidence, class_id, class_name
        """
        if self.football_mode:
            results = self.model(frame, verbose=False, conf=conf)
        else:
            results = self.model(frame, verbose=False, conf=conf, classes=[0])

        detections = []
        for r in results:
            for box in r.boxes.cpu().numpy():
                x1, y1, x2, y2 = box.xyxy[0]
                conf_val = float(box.conf[0])
                cls_id = int(box.cls[0])

                if self.football_mode:
                    cls_name = CLASS_NAMES.get(cls_id, 'unknown')
                else:
                    cls_name = 'player'
                    cls_id = 2

                detections.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'confidence': conf_val,
                    'class_id': cls_id,
                    'class_name': cls_name,
                })
        return detections


class OffBallAgent:
    """Computes per-player OBC scores using trained Temporal GAT."""

    def __init__(self, weights_path=GAT_WEIGHTS_PATH, device='cpu'):
        self.device = torch.device(device)
        self.model = TemporalGAT(in_channels=5, hidden=64, out_channels=32)
        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def compute_obc(self, players):
        if len(players) < 2:
            return {}, 0.0

        nodes = []
        for p in players:
            is_teammate = 1.0 if p.get('team', 0) == 0 else 0.0
            is_keeper = 1.0 if p.get('class_name') == 'goalkeeper' else 0.0
            nodes.append([
                p['pitch_x'] / 120.0,
                p['pitch_y'] / 80.0,
                is_teammate,
                0.0,
                is_keeper,
            ])

        x = torch.tensor(nodes, dtype=torch.float32, device=self.device)

        raw_pts = np.array([[p['pitch_x'], p['pitch_y']] for p in players])
        dists = cdist(raw_pts, raw_pts)
        src, dst = np.where((dists > 0) & (dists <= EDGE_THRESHOLD_M))

        if len(src) == 0:
            return {p['player_id']: 0.0 for p in players}, 0.0

        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long, device=self.device)

        with torch.no_grad():
            global_xg = float(self.model(x, edge_index).item())

            emb = self.model.gat1(x, edge_index)
            emb = self.model.bn1(emb)
            emb = F.elu(emb)
            emb = self.model.gat2(emb, edge_index)
            emb = self.model.bn2(emb)

            mags = torch.norm(emb, dim=1)
            # Z-score → sigmoid: gives proper [0,1] distribution
            # Average player ≈ 0.5, high OBC ≈ 0.7-0.9, low ≈ 0.1-0.3
            mags_z = (mags - mags.mean()) / (mags.std() + 1e-8)
            mags_norm = torch.sigmoid(mags_z)

        scores = {}
        for i, p in enumerate(players):
            scores[p['player_id']] = round(float(mags_norm[i].item()), 3)

        return scores, global_xg


class MatchReport:
    """Accumulates match data for final report generation."""

    def __init__(self):
        self.frames = []
        self.obc_history = defaultdict(list)
        self.xg_history = []
        self.player_appearances = defaultdict(int)
        self.player_classes = {}
        self.player_teams = {}
        self.key_moments = []
        self.total_frames_processed = 0
        self.first_frame = None
        self.last_frame = None

    def set_frame_count(self, total):
        self.total_frames_processed = total

    def update(self, frame_n, players, obc_scores, xg):
        self.frames.append(frame_n)
        self.xg_history.append(xg)
        if self.first_frame is None:
            self.first_frame = frame_n
        self.last_frame = frame_n

        for p in players:
            pid = p['player_id']
            self.player_appearances[pid] += 1
            self.player_classes[pid] = p.get('class_name', 'player')
            self.player_teams[pid] = p.get('team', -1)

            if pid in obc_scores:
                self.obc_history[pid].append(obc_scores[pid])

        if xg > 0.15:
            self.key_moments.append({
                'frame': frame_n,
                'xg': round(xg, 4),
                'top_player': max(obc_scores, key=obc_scores.get) if obc_scores else None,
                'top_obc': round(max(obc_scores.values()), 3) if obc_scores else 0,
            })

    def generate(self, fps=24.0):
        actual_frames = self.total_frames_processed or (
            (self.last_frame - self.first_frame + 1) if self.first_frame is not None else len(self.frames)
        )
        report = {
            'summary': {
                'total_frames': actual_frames,
                'duration_sec': round(actual_frames / fps, 1),
                'unique_players_tracked': len(self.player_appearances),
                'avg_xg': round(np.mean(self.xg_history), 4) if self.xg_history else 0,
                'max_xg': round(max(self.xg_history), 4) if self.xg_history else 0,
                'key_moments': len(self.key_moments),
            },
            'top_obc_players': [],
            'key_moments': self.key_moments[:20],
            'player_breakdown': {},
        }

        for pid, scores in sorted(self.obc_history.items(),
                                    key=lambda x: np.mean(x[1]), reverse=True)[:10]:
            report['top_obc_players'].append({
                'player_id': pid,
                'class': self.player_classes.get(pid, 'player'),
                'team': self.player_teams.get(pid, -1),
                'avg_obc': round(float(np.mean(scores)), 3),
                'max_obc': round(float(max(scores)), 3),
                'appearances': self.player_appearances[pid],
            })

        team_counts = defaultdict(int)
        for pid, team in self.player_teams.items():
            team_counts[team] += 1
        report['summary']['team_distribution'] = dict(team_counts)

        class_counts = defaultdict(int)
        for pid, cls in self.player_classes.items():
            class_counts[cls] += 1
        report['summary']['class_distribution'] = dict(class_counts)

        return report


def draw_pitch_view(players, obc_scores, width=420, height=280):
    pitch = np.zeros((height, width, 3), dtype=np.uint8)
    pitch[:] = (34, 120, 34)

    m = 25
    pw = width - 2 * m
    ph = height - 2 * m

    cv2.rectangle(pitch, (m, m), (width - m, height - m), (255, 255, 255), 1)
    cx = m + pw // 2
    cv2.line(pitch, (cx, m), (cx, height - m), (255, 255, 255), 1)
    cv2.circle(pitch, (cx, m + ph // 2), int(ph * 9.15 / PITCH_WIDTH), (255, 255, 255), 1)
    cv2.circle(pitch, (cx, m + ph // 2), 3, (255, 255, 255), -1)

    pa_w = int(pw * 16.5 / PITCH_LENGTH)
    pa_h = int(ph * 40.3 / PITCH_WIDTH)
    pa_y1 = m + (ph - pa_h) // 2
    pa_y2 = pa_y1 + pa_h
    cv2.rectangle(pitch, (m, pa_y1), (m + pa_w, pa_y2), (255, 255, 255), 1)
    cv2.rectangle(pitch, (width - m - pa_w, pa_y1), (width - m, pa_y2), (255, 255, 255), 1)

    ga_w = int(pw * 5.5 / PITCH_LENGTH)
    ga_h = int(ph * 18.3 / PITCH_WIDTH)
    ga_y1 = m + (ph - ga_h) // 2
    ga_y2 = ga_y1 + ga_h
    cv2.rectangle(pitch, (m, ga_y1), (m + ga_w, ga_y2), (255, 255, 255), 1)
    cv2.rectangle(pitch, (width - m - ga_w, ga_y1), (width - m, ga_y2), (255, 255, 255), 1)

    for p in players:
        px_pos = int(m + (p['pitch_x'] / PITCH_LENGTH) * pw)
        py_pos = int(m + (p['pitch_y'] / PITCH_WIDTH) * ph)
        px_pos = max(m, min(width - m, px_pos))
        py_pos = max(m, min(height - m, py_pos))

        obc = obc_scores.get(p['player_id'], 0.0)
        cls = p.get('class_name', 'player')

        if cls == 'ball':
            cv2.circle(pitch, (px_pos, py_pos), 4, (255, 255, 255), -1)
            continue
        elif cls == 'referee':
            color = (0, 255, 255)
            dot_size = 4
        elif cls == 'goalkeeper':
            color = (0, 200, 0) if p.get('team', 0) == 0 else (0, 165, 255)
            dot_size = max(5, int(5 + obc * 18))
        else:
            if p.get('team', 0) == 0:
                color = (0, 0, 220)
            elif p.get('team', 0) == 1:
                color = (220, 180, 80)
            else:
                color = (180, 180, 180)
            dot_size = max(4, int(4 + obc * 20))

        cv2.circle(pitch, (px_pos, py_pos), dot_size, color, -1)
        cv2.circle(pitch, (px_pos, py_pos), dot_size, (0, 0, 0), 1)

        if obc > 0.3 and cls != 'referee':
            cv2.putText(pitch, f"{obc:.2f}", (px_pos + dot_size + 2, py_pos + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    return pitch


def draw_overlays(frame, players, obc_scores, xg, frame_n):
    viz = frame.copy()
    h, w = viz.shape[:2]

    cls_colors = {
        'player': {0: (0, 0, 255), 1: (255, 180, 50), -1: (180, 180, 180)},
        'goalkeeper': {0: (0, 200, 0), 1: (0, 165, 255), -1: (0, 200, 0)},
        'referee': (0, 255, 255),
        'ball': (255, 255, 255),
    }

    sorted_players = sorted(players,
                            key=lambda p: obc_scores.get(p['player_id'], 0), reverse=True)

    for p in sorted_players:
        x1, y1, x2, y2 = p['bbox']
        pid = p['player_id']
        cls = p.get('class_name', 'player')
        team = p.get('team', -1)

        if cls in ('player', 'goalkeeper'):
            color = cls_colors[cls].get(team, (180, 180, 180))
        elif cls == 'ball':
            color = cls_colors['ball']
            cv2.circle(viz, (int((x1+x2)/2), int((y1+y2)/2)), 5, color, 2)
            continue
        else:
            color = cls_colors.get(cls, (180, 180, 180))

        obc = obc_scores.get(pid, 0.0)

        thickness = 3 if obc > 0.7 else 2
        cv2.rectangle(viz, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

        if cls == 'goalkeeper':
            tag = 'GK'
        elif cls == 'referee':
            tag = 'REF'
        else:
            tag = f"#{pid}"

        if obc > 0.01 and cls not in ('referee', 'ball'):
            label = f"{tag} {obc:.2f}"
            lw = len(label) * 7 + 4
            lh = 16
            lx, ly = int(x1), int(y1) - lh - 2
            if ly < 2:
                ly = int(y2) + 2

            alpha = min(1.0, 0.4 + obc * 0.6)
            overlay = viz.copy()
            cv2.rectangle(overlay, (lx, ly), (lx + lw, ly + lh), (0, 0, 0), -1)
            cv2.addWeighted(overlay, alpha, viz, 1 - alpha, 0, viz)

            text_color = (0, 255, 100) if obc > 0.5 else (200, 200, 200)
            cv2.putText(viz, label, (lx + 2, ly + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1)
        else:
            ly = int(y1) - 5
            if ly < 12:
                ly = int(y2) + 12
            cv2.putText(viz, tag, (int(x1), ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    player_obc = {k: v for k, v in obc_scores.items()}
    max_obc = max(player_obc.values()) if player_obc else 0.0
    n_players = sum(1 for p in players if p.get('class_name') in ('player', 'goalkeeper'))
    n_refs = sum(1 for p in players if p.get('class_name') == 'referee')
    has_ball = any(p.get('class_name') == 'ball' for p in players)

    bar_h = 28
    overlay = viz.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, viz, 0.3, 0, viz)

    status = f"F:{frame_n}  Players:{n_players}"
    if n_refs > 0:
        status += f"  Refs:{n_refs}"
    if has_ball:
        status += "  Ball:YES"
    status += f"  xG:{xg:.3f}  MaxOBC:{max_obc:.2f}"

    cv2.putText(viz, status, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 100), 1)

    cv2.putText(viz, "FOOTAGENT", (w - 95, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1)

    return viz


def run_demo(args):
    print("=" * 60)
    print("  FOOTAGENT - Multi-Agent Football Intelligence")
    print("  Pipeline: Football YOLO -> ByteTrack -> GAT (OBC)")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    video_path = args.video
    if not Path(video_path).exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    print("\nLoading models...")
    weights_dir = Path(__file__).parent / 'weights'
    fb_yolo_path = str(weights_dir / 'yolo-football-player-detection.pt')
    if not Path(fb_yolo_path).exists():
        fb_yolo_path = str(Path('F:/footagent/weights/yolo-football-player-detection.pt'))
    detector = FootballDetector(weights_path=fb_yolo_path)

    tracker = ByteTrack(track_thresh=0.25, match_thresh=0.65, track_buffer=20, max_players=25)
    team_cls = TeamClassifier(n_clusters=2)
    homography = HomographyCalibrator()

    gat_path = str(weights_dir / 'gat_temporal.pt')
    if not Path(gat_path).exists():
        gat_path = str(Path('F:/footagent/weights/gat_temporal.pt'))
    obc_agent = OffBallAgent(weights_path=gat_path, device=device)
    print(f"  GAT: Temporal GAT ({obc_agent.model.count_parameters():,} params, Spearman=0.793)")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\nVideo: {w}x{h} @ {fps:.1f}fps, {total_frames} frames ({total_frames/fps:.1f}s)")

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    ret, first_frame = cap.read()
    if ret:
        homography.auto_calibrate(first_frame.shape)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    pitch_w, pitch_h = 420, 280
    writer = None
    if args.save:
        out_w = w + pitch_w
        out_h = max(h, pitch_h)
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (out_w, out_h))
        print(f"  Output: {args.save} ({out_w}x{out_h})")

    obc_scores = {}
    xg_val = 0.0
    team_assignments = {}
    player_history = {}
    match_report = MatchReport()
    processed = 0
    obc_interval = 15

    print(f"\nProcessing... (OBC every {obc_interval} frames)")
    print(f"{'Frame':>6} {'Ply':>4} {'GK':>3} {'Ref':>4} {'Ball':>5} {'xG':>7} {'MaxOBC':>7} {'FPS':>6}")
    print("-" * 50)

    t_start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_n = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if args.max_frames and processed >= args.max_frames:
            break

        timestamp = frame_n / fps

        # Step 1: Football YOLO detection
        raw_dets = detector.detect(frame, conf=0.25)

        person_dets = [d for d in raw_dets if d['class_name'] in ('player', 'goalkeeper', 'referee')]
        ball_dets = [d for d in raw_dets if d['class_name'] == 'ball']

        if len(person_dets) == 0:
            processed += 1
            continue

        # Step 2: ByteTrack on person detections
        bt_input = [(d['x1'], d['y1'], d['x2'], d['y2'], d['confidence']) for d in person_dets]
        tracked = tracker.update(bt_input)

        det_class_map = {}
        for d in person_dets:
            key = (round(d['x1'], 1), round(d['y1'], 1))
            det_class_map[key] = d['class_name']

        # Step 3: Team classification
        needs_team = (
            (not team_assignments and len(tracked) >= 4) or
            (processed % int(fps * 3) == 0 and len(tracked) >= 4)
        )
        if needs_team:
            dets_for_team = [(t[0], t[1], t[2], t[3], t[5]) for t in tracked]
            tids_for_team = [t[4] for t in tracked]
            new_teams = team_cls.fit_predict_teams(frame, dets_for_team, tids_for_team)
            team_assignments.update(new_teams)

        # Step 4: Build player list
        players = []
        for t in tracked:
            x1, y1, x2, y2, tid, conf = t
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            pitch_x, pitch_y = homography.pixel_to_pitch(cx, cy)
            pitch_x = max(0, min(PITCH_LENGTH, pitch_x))
            pitch_y = max(0, min(PITCH_WIDTH, pitch_y))

            # Match class from detection
            best_cls = 'player'
            best_dist = 9999
            for d in person_dets:
                dx = (d['x1'] + d['x2'])/2 - cx
                dy = (d['y1'] + d['y2'])/2 - cy
                dist = dx*dx + dy*dy
                if dist < best_dist:
                    best_dist = dist
                    best_cls = d['class_name']

            speed = 0.0
            if int(tid) in player_history:
                prev = player_history[int(tid)]
                dt = timestamp - prev[2]
                if dt > 0.02:
                    dist = np.sqrt((pitch_x - prev[0])**2 + (pitch_y - prev[1])**2)
                    speed = min(dist / dt, 12.0)
            player_history[int(tid)] = (pitch_x, pitch_y, timestamp)

            players.append({
                'player_id': int(tid),
                'team': team_assignments.get(int(tid), -1),
                'class_name': best_cls,
                'pixel_x': float(cx), 'pixel_y': float(cy),
                'pitch_x': float(pitch_x), 'pitch_y': float(pitch_y),
                'speed': speed,
                'confidence': float(conf),
                'bbox': (x1, y1, x2, y2),
            })

        # Add ball detections
        for bd in ball_dets:
            bx, by = (bd['x1'] + bd['x2'])/2, (bd['y1'] + bd['y2'])/2
            players.append({
                'player_id': -1,
                'team': -1,
                'class_name': 'ball',
                'pixel_x': float(bx), 'pixel_y': float(by),
                'pitch_x': 0, 'pitch_y': 0,
                'speed': 0,
                'confidence': bd['confidence'],
                'bbox': (bd['x1'], bd['y1'], bd['x2'], bd['y2']),
            })

        # Step 5: OBC computation
        field_players = [p for p in players if p['class_name'] in ('player', 'goalkeeper')]
        if processed % obc_interval == 0 and processed > 0 and len(field_players) >= 2:
            obc_scores, xg_val = obc_agent.compute_obc(field_players)
            match_report.update(frame_n, field_players, obc_scores, xg_val)

        # Step 6: Visualization
        viz_frame = draw_overlays(frame, players, obc_scores, xg_val, frame_n)
        pitch_view = draw_pitch_view(players, obc_scores, pitch_w, pitch_h)

        if not args.headless:
            cv2.imshow('FootAgent - Match View', viz_frame)
            cv2.imshow('FootAgent - Pitch View', pitch_view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                cv2.waitKey(0)

        if writer:
            out_h = max(h, pitch_h)
            combined = np.zeros((out_h, w + pitch_w, 3), dtype=np.uint8)
            combined[:h, :w] = viz_frame
            combined[:pitch_h, w:] = pitch_view
            writer.write(combined)

        processed += 1
        if processed % 200 == 0:
            elapsed = time.time() - t_start
            proc_fps = processed / elapsed
            n_ply = sum(1 for p in players if p['class_name'] == 'player')
            n_gk = sum(1 for p in players if p['class_name'] == 'goalkeeper')
            n_ref = sum(1 for p in players if p['class_name'] == 'referee')
            n_ball = sum(1 for p in players if p['class_name'] == 'ball')
            max_obc = max(obc_scores.values()) if obc_scores else 0.0
            print(f"{frame_n:6d} {n_ply:4d} {n_gk:3d} {n_ref:4d} {'Y' if n_ball else 'N':>5} "
                  f"{xg_val:7.4f} {max_obc:7.3f} {proc_fps:6.1f}")

    cap.release()
    if writer:
        writer.release()
    if not args.headless:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"\nDone: {processed} frames in {elapsed:.1f}s ({processed/elapsed:.1f} fps)")

    if args.report or args.save:
        match_report.set_frame_count(processed)
        report = match_report.generate(fps)
        report_path = (args.save or 'match') + '_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nMatch Report saved to: {report_path}")
        print(f"  Duration: {report['summary']['duration_sec']}s")
        print(f"  Players tracked: {report['summary']['unique_players_tracked']}")
        print(f"  Avg xG: {report['summary']['avg_xg']:.4f}")
        print(f"  Max xG: {report['summary']['max_xg']:.4f}")
        print(f"  Key moments: {report['summary']['key_moments']}")
        if report['top_obc_players']:
            print(f"\n  Top OBC Players:")
            for p in report['top_obc_players'][:5]:
                print(f"    #{p['player_id']} ({p['class']}): avg={p['avg_obc']:.3f} max={p['max_obc']:.3f}")

    if args.save:
        print(f"\nOutput video: {args.save}")


def main():
    parser = argparse.ArgumentParser(description='FootAgent Demo')
    parser.add_argument('--video', default='data/matches/test_clip.mp4')
    parser.add_argument('--start-frame', type=int, default=90)
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--serve', action='store_true',
                        help='Start FastAPI backend + 3D dashboard')
    args = parser.parse_args()

    if args.serve:
        print("Starting FootAgent server...")
        import uvicorn
        uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
    else:
        run_demo(args)


if __name__ == '__main__':
    main()
