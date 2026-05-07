from typing import Annotated, Dict, List, Optional, TypedDict, Union
from typing_extensions import TypedDict
import numpy as np

class AgentState(TypedDict):
    """
    Main state object for the FootAgent LangGraph pipeline.
    """
    # Current frame information
    frame_idx: int
    timestamp: float
    raw_frame: Optional[np.ndarray]  # Current frame pixels
    
    # Tracking data
    players: List[Dict]  # List of player info (id, team, x, y, speed)
    ball: Optional[Dict]  # Ball position and velocity
    
    # Scene context
    pitch_mapping: Optional[Dict]  # Homography data
    is_incident: bool  # Whether a potential foul is detected
    event_type: str  # 'normal', 'incident', 'chance'
    
    # Agent outputs
    foul_analysis: Optional[Dict]  # Output from ViT/VAR agent
    off_ball_metrics: Optional[Dict]  # Output from GAT agent
    
    # System state
    vram_usage: float
    active_models: List[str]
    match_id: str
    
    # Match History
    match_memory_path: str
