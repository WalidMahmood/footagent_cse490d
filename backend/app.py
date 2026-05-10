"""
FootAgent Backend — serves the 3D dashboard and runs the pipeline.

Endpoints:
  POST /upload           — upload a video file
  POST /match/start      — start processing a video
  GET  /match/status      — poll current processing state
  GET  /match/memory/{id} — get accumulated match data (frames, incidents, stats)
  GET  /match/report/{id} — get final match report JSON
"""
import json
import os
import shutil
import threading
import time
import traceback
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="FootAgent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Phase states: idle → loading → processing → done | error
MATCH_STATUS = {
    "phase": "idle",       # idle | loading | processing | done | error
    "active": False,
    "frame_idx": 0,
    "match_id": None,
    "total_frames": 0,
    "fps": 0,
    "players_count": 0,
    "xg": 0.0,
    "max_obc": 0.0,
    "progress": 0,         # 0-100
    "error": None,
}

MATCH_MEMORY = {}
MATCH_REPORTS = {}

_pipeline_lock = threading.Lock()


def _run_pipeline(video_path: str, match_id: str):
    global MATCH_STATUS

    MATCH_STATUS["phase"] = "loading"
    print(f"[Backend] Loading models...")

    try:
        from main import FootballDetector, OffBallAgent, MatchReport
        from agents.tracking_agent import ByteTrack, TeamClassifier, HomographyCalibrator
    except ImportError as e:
        print(f"[Backend] Import error: {e}")
        MATCH_STATUS["phase"] = "error"
        MATCH_STATUS["active"] = False
        MATCH_STATUS["error"] = f"Import error: {e}"
        return

    try:
        memory = {"frames": [], "incidents": [], "stats": {}}
        MATCH_MEMORY[match_id] = memory

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[Backend] Device: {device}")

        weights_dir = Path(__file__).parent.parent / 'weights'
        fb_path = str(weights_dir / 'yolo-football-player-detection.pt')
        if not Path(fb_path).exists():
            fb_path = str(Path('F:/footagent/weights/yolo-football-player-detection.pt'))

        detector = FootballDetector(weights_path=fb_path)
        tracker = ByteTrack(track_thresh=0.25, match_thresh=0.65, track_buffer=20, max_players=25)
        team_cls = TeamClassifier(n_clusters=2)
        homography = HomographyCalibrator()

        gat_path = str(weights_dir / 'gat_temporal.pt')
        if not Path(gat_path).exists():
            gat_path = str(Path('F:/footagent/weights/gat_temporal.pt'))
        obc_agent = OffBallAgent(weights_path=gat_path, device=device)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Backend] Cannot open video: {video_path}")
            MATCH_STATUS["phase"] = "error"
            MATCH_STATUS["active"] = False
            MATCH_STATUS["error"] = f"Cannot open video: {video_path}"
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        MATCH_STATUS["fps"] = fps
        MATCH_STATUS["total_frames"] = total_frames

        ret, first_frame = cap.read()
        if ret:
            homography.auto_calibrate(first_frame.shape)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # NOW we're actually processing
        MATCH_STATUS["phase"] = "processing"
        print(f"[Backend] Models loaded. Starting frame processing.")

        obc_scores = {}
        xg_val = 0.0
        team_assignments = {}
        player_history = {}
        match_report = MatchReport()
        processed = 0
        obc_interval = 15
        start_frame = 90

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"[Backend] Processing {video_path} ({total_frames} frames @ {fps:.0f}fps)")

        while cap.isOpened() and MATCH_STATUS["active"]:
            ret, frame = cap.read()
            if not ret:
                break

            frame_n = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            timestamp = frame_n / fps

            raw_dets = detector.detect(frame, conf=0.25)
            person_dets = [d for d in raw_dets if d['class_name'] in ('player', 'goalkeeper', 'referee')]
            ball_dets = [d for d in raw_dets if d['class_name'] == 'ball']

            if len(person_dets) == 0:
                MATCH_STATUS["frame_idx"] = frame_n
                MATCH_STATUS["progress"] = min(99, int((frame_n / max(total_frames, 1)) * 100))
                processed += 1
                continue

            bt_input = [(d['x1'], d['y1'], d['x2'], d['y2'], d['confidence']) for d in person_dets]
            tracked = tracker.update(bt_input)

            needs_team = (
                (not team_assignments and len(tracked) >= 4) or
                (processed % int(fps * 3) == 0 and len(tracked) >= 4)
            )
            if needs_team:
                dets_for_team = [(t[0], t[1], t[2], t[3], t[5]) for t in tracked]
                tids_for_team = [t[4] for t in tracked]
                new_teams = team_cls.fit_predict_teams(frame, dets_for_team, tids_for_team)
                team_assignments.update(new_teams)

            players = []
            for t in tracked:
                x1, y1, x2, y2, tid, conf = t
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                pitch_x, pitch_y = homography.pixel_to_pitch(cx, cy)
                pitch_x = max(0, min(105.0, pitch_x))
                pitch_y = max(0, min(68.0, pitch_y))

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
                        d = np.sqrt((pitch_x - prev[0])**2 + (pitch_y - prev[1])**2)
                        speed = min(d / dt, 12.0)
                player_history[int(tid)] = (pitch_x, pitch_y, timestamp)

                players.append({
                    'player_id': int(tid),
                    'team': team_assignments.get(int(tid), -1),
                    'class_name': best_cls,
                    'pitch_x': round(float(pitch_x), 2),
                    'pitch_y': round(float(pitch_y), 2),
                    'speed': round(speed, 2),
                    'confidence': round(float(conf), 3),
                })

            for bd in ball_dets:
                bx, by = (bd['x1'] + bd['x2'])/2, (bd['y1'] + bd['y2'])/2
                bpx, bpy = homography.pixel_to_pitch(bx, by)
                players.append({
                    'player_id': -1,
                    'team': -1,
                    'class_name': 'ball',
                    'pitch_x': round(float(max(0, min(105, bpx))), 2),
                    'pitch_y': round(float(max(0, min(68, bpy))), 2),
                    'speed': 0,
                    'confidence': round(bd['confidence'], 3),
                })

            field_players = [p for p in players if p['class_name'] in ('player', 'goalkeeper')]
            if processed % obc_interval == 0 and processed > 0 and len(field_players) >= 2:
                obc_scores, xg_val = obc_agent.compute_obc(field_players)
                match_report.update(frame_n, field_players, obc_scores, xg_val)

            max_obc = max(obc_scores.values()) if obc_scores else 0.0

            frame_data = {
                'frame_idx': frame_n,
                'players': players,
                'off_ball': {
                    'player_scores': {str(k): v for k, v in obc_scores.items()},
                    'possession_prob': round(xg_val, 4),
                },
                'vram_usage': torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            }

            if len(memory["frames"]) > 300:
                memory["frames"] = memory["frames"][-200:]
            memory["frames"].append(frame_data)

            if xg_val > 0.01:
                memory["incidents"].append({
                    'frame': frame_n,
                    'xg': round(xg_val, 4),
                    'type': 'high_xg' if xg_val > 0.05 else 'buildup',
                })

            MATCH_STATUS["frame_idx"] = frame_n
            MATCH_STATUS["players_count"] = len(players)
            MATCH_STATUS["xg"] = round(xg_val, 4)
            MATCH_STATUS["max_obc"] = round(max_obc, 3)
            MATCH_STATUS["progress"] = min(99, int((frame_n / max(total_frames, 1)) * 100))

            processed += 1
            if processed % 100 == 0:
                pct = int((frame_n / max(total_frames, 1)) * 100)
                print(f"[Backend] Frame {frame_n}/{total_frames} ({pct}%) | {len(players)} players | xG={xg_val:.4f} | maxOBC={max_obc:.3f}")

        cap.release()

        match_report.set_frame_count(processed)
        report = match_report.generate(fps)
        MATCH_REPORTS[match_id] = report
        memory["stats"] = report.get("summary", {})

        report_path = f"data/matches/{match_id}_report.json"
        os.makedirs("data/matches", exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        MATCH_STATUS["phase"] = "done"
        MATCH_STATUS["active"] = False
        MATCH_STATUS["progress"] = 100
        print(f"[Backend] Done: {processed} frames processed for {match_id}")

    except Exception as e:
        traceback.print_exc()
        MATCH_STATUS["phase"] = "error"
        MATCH_STATUS["active"] = False
        MATCH_STATUS["error"] = str(e)
        print(f"[Backend] Pipeline error: {e}")


# --- Routes ---

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    os.makedirs("data/uploads", exist_ok=True)
    file_path = f"data/uploads/{file.filename}"
    with open(file_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)
    return {"status": "success", "file_path": file_path, "filename": file.filename}


@app.post("/match/start")
async def start_match(video_path: str, match_id: str):
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    if MATCH_STATUS["active"]:
        return {"status": "error", "message": "A match is already being processed"}

    # Set state BEFORE thread starts — prevents poll race condition
    MATCH_STATUS["active"] = True
    MATCH_STATUS["phase"] = "loading"
    MATCH_STATUS["match_id"] = match_id
    MATCH_STATUS["frame_idx"] = 0
    MATCH_STATUS["total_frames"] = 0
    MATCH_STATUS["players_count"] = 0
    MATCH_STATUS["xg"] = 0.0
    MATCH_STATUS["max_obc"] = 0.0
    MATCH_STATUS["progress"] = 0
    MATCH_STATUS["error"] = None

    thread = threading.Thread(target=_run_pipeline, args=(video_path, match_id), daemon=True)
    thread.start()
    return {"status": "success", "match_id": match_id}


@app.get("/match/status")
async def get_status():
    return MATCH_STATUS


@app.get("/match/memory/{match_id}")
async def get_memory(match_id: str):
    if match_id not in MATCH_MEMORY:
        return {"frames": [], "incidents": [], "stats": {}}
    return MATCH_MEMORY[match_id]


@app.get("/match/report/{match_id}")
async def get_report(match_id: str):
    if match_id in MATCH_REPORTS:
        return MATCH_REPORTS[match_id]
    path = f"data/matches/{match_id}_report.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Report not found")


dashboard_dir = Path(__file__).parent.parent / "dashboard"
if dashboard_dir.exists():
    app.mount("/dashboard", StaticFiles(directory=str(dashboard_dir), html=True), name="dashboard")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
