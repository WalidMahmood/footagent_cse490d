from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import shutil
import cv2
import threading
from typing import Optional
from pipeline.graph import create_footagent_graph
from pipeline.nodes import PipelineNodes
from pipeline.model_loaders import ModelLoader
from pipeline.vram_scheduler import VRAMScheduler

app = FastAPI(title="FootAgent API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for dashboard
os.makedirs("dashboard", exist_ok=True)
app.mount("/dashboard", StaticFiles(directory="dashboard", html=True), name="dashboard")

# Shared state
MATCH_STATUS = {"active": False, "frame_idx": 0, "match_id": None}
PIPELINE = None

def init_pipeline():
    global PIPELINE
    loader = ModelLoader()
    scheduler = VRAMScheduler(loader)
    nodes = PipelineNodes(loader, scheduler)
    PIPELINE = create_footagent_graph(nodes)

def run_pipeline_thread(video_path: str, match_id: str):
    global MATCH_STATUS
    print(f"[Backend] Starting pipeline thread for match: {match_id}")
    MATCH_STATUS["active"] = True
    MATCH_STATUS["match_id"] = match_id
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Backend] ERROR: Could not open video: {video_path}")
            MATCH_STATUS["active"] = False
            return

        frame_idx = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        print(f"[Backend] Video opened: {video_path} at {fps} FPS")
        
        while cap.isOpened() and MATCH_STATUS["active"]:
            ret, frame = cap.read()
            if not ret:
                print("[Backend] End of video reached.")
                break
                
            # Run LangGraph pipeline
            state = {
                "frame_idx": frame_idx,
                "timestamp": frame_idx / fps,
                "raw_frame": frame,
                "match_id": match_id,
                "players": [],
                "is_incident": False,
                "event_type": "normal",
                "foul_analysis": None,
                "off_ball_metrics": None,
                "vram_usage": 0.0,
                "active_models": [],
                "match_memory_path": f"data/matches/{match_id}_memory.json"
            }
            
            # Invoke graph
            if PIPELINE:
                PIPELINE.invoke(state)
            
            MATCH_STATUS["frame_idx"] = frame_idx
            if frame_idx % 10 == 0:
                print(f"[Backend] Processed frame {frame_idx}")
            frame_idx += 1
            
        cap.release()
    except Exception as e:
        print(f"[Backend] CRITICAL ERROR in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        MATCH_STATUS["active"] = False
        print(f"[Backend] Pipeline thread finished for match: {match_id}")

@app.on_event("startup")
async def startup_event():
    init_pipeline()

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    os.makedirs("data/uploads", exist_ok=True)
    file_path = f"data/uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"status": "success", "file_path": file_path, "filename": file.filename}

@app.post("/match/start")
async def start_match(video_path: str, match_id: str, background_tasks: BackgroundTasks):
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
        
    if MATCH_STATUS["active"]:
        return {"status": "error", "message": "A match is already being processed"}
        
    background_tasks.add_task(run_pipeline_thread, video_path, match_id)
    return {"status": "success", "match_id": match_id}

@app.get("/match/status")
async def get_status():
    return MATCH_STATUS

@app.get("/match/memory/{match_id}")
async def get_memory(match_id: str):
    path = f"data/matches/{match_id}_memory.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Match memory not found")
    with open(path, 'r') as f:
        import json
        return json.load(f)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
