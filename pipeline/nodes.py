import cv2
import torch
import numpy as np
from typing import Dict, List, Optional
from agents.tracking_agent import TrackingAgent
from .state import AgentState
from .match_memory import MatchMemory
from .model_loaders import ModelLoader
from .vram_scheduler import VRAMScheduler


class PipelineNodes:
    """
    Contains node implementations for the LangGraph FootAgent pipeline.
    """

    def __init__(self, loader: ModelLoader, scheduler: VRAMScheduler):
        self.loader = loader
        self.scheduler = scheduler
        self.tracking_agent = TrackingAgent(model_size='n')
        self.memory = None  # Initialized per match

    def tracking_node(self, state: AgentState) -> Dict:
        """
        Node: Detect and track players in the current frame.
        """
        print(f"--- [Node] Tracking (Frame {state['frame_idx']}) ---")
        frame = state["raw_frame"]
        
        # Step 1: Run tracking
        detections = self.tracking_agent.detect_players(frame)
        tracked = self.tracking_agent.track_players(frame, detections)
        team_assignments = self.tracking_agent.assign_teams(frame, tracked)
        
        # Step 2: Format output
        players_data = []
        for t in tracked:
            x1, y1, x2, y2, tid, conf = t
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            pitch_x, pitch_y = self.tracking_agent.homography.pixel_to_pitch(cx, cy)
            
            players_data.append({
                "player_id": int(tid),
                "team": int(team_assignments.get(int(tid), -1)),
                "pixel_x": float(cx),
                "pixel_y": float(cy),
                "pitch_x": float(pitch_x),
                "pitch_y": float(pitch_y),
                "confidence": float(conf)
            })
            
        return {
            "players": players_data,
            "vram_usage": self.scheduler.get_current_vram_gb(),
            "active_models": self.loader.get_active_models()
        }

    def trigger_node(self, state: AgentState) -> Dict:
        """
        Node: Analyze tracking data to detect incidents (fouls/chances).
        """
        # Simplified trigger logic: Check for player proximity or high speed near goal
        # In production, this would use a small heuristic or shallow model
        is_incident = False
        event_type = "normal"
        
        # Heuristic: If multiple players are very close together
        # (This is just a placeholder for the actual trigger logic)
        if len(state["players"]) > 1:
             # Check distances...
             pass
             
        return {
            "is_incident": is_incident,
            "event_type": event_type
        }

    def off_ball_node(self, state: AgentState) -> Dict:
        """
        Node: Run Temporal GAT for off-ball positioning analysis.
        Calculates Off-Ball Contribution (OBC) scores per player.
        """
        player_list = state["players"]
        if len(player_list) < 2:
            return {"off_ball_metrics": {"possession_prob": 0.0, "threat_level": "low", "player_scores": {}}}
            
        print(f"--- [Node] Off-Ball Analysis (GAT) - {len(player_list)} players ---")
        self.scheduler.check_and_optimize("gat")
        gat_model = self.loader.load_gat("checkpoints/gat/temporal_gat_20260506_152134/best_model.pt")
        
        # 1. Prepare Node Features: [x, y, teammate, actor, keeper]
        nodes = []
        for p in player_list:
            nodes.append([
                p["pitch_x"] / 105.0, 
                p["pitch_y"] / 68.0,  
                1.0 if p["team"] == 0 else 0.0,
                0.0, 0.0
            ])
        
        x = torch.tensor(nodes, dtype=torch.float32).to(self.loader.device)
        
        # 2. Build Edges
        edge_index = []
        for i, p1 in enumerate(player_list):
            for j, p2 in enumerate(player_list):
                if i == j: continue
                dist = np.sqrt((p1["pitch_x"] - p2["pitch_x"])**2 + (p1["pitch_y"] - p2["pitch_y"])**2)
                if dist < 40.0: # Increased threshold for better connectivity in scattered scenes
                    edge_index.append([i, j])
        
        if not edge_index:
            return {"off_ball_metrics": {"possession_prob": 0.0, "threat_level": "low", "player_scores": {}}}
            
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.loader.device)
        
        # 3. Run Inference
        with torch.no_grad():
            # Get the global score
            score = gat_model(x, edge_index)
            global_obc = float(score.cpu().item())
            
            # Heuristic: Extract relative player threat from GAT features (magnitude of node embeddings)
            # In a research-grade setup, we'd use attention weights, but this provides a strong visual indicator
            # We run the first two layers to get node-level embeddings
            node_embeddings = gat_model.gat1(x, edge_index)
            node_embeddings = torch.nn.functional.elu(node_embeddings)
            node_embeddings = gat_model.gat2(node_embeddings, edge_index)
            
            # Relative score based on embedding magnitude normalized by global score
            player_magnitudes = torch.norm(node_embeddings, dim=1)
            player_magnitudes = (player_magnitudes / (player_magnitudes.max() + 1e-6)) * global_obc
            
            player_scores = {}
            for i, p in enumerate(player_list):
                player_scores[p["player_id"]] = float(player_magnitudes[i].cpu().item())
        
        metrics = {
            "possession_prob": global_obc,
            "threat_level": "high" if global_obc > 0.7 else ("medium" if global_obc > 0.4 else "low"),
            "player_scores": player_scores
        }
        
        print(f"  Global OBC: {global_obc:.4f} | Max Player Threat: {max(player_scores.values()) if player_scores else 0:.4f}")
        
        return {"off_ball_metrics": metrics}

    def var_node(self, state: AgentState) -> Dict:
        """
        Node: Run ViT Foul Classifier if an incident is detected.
        """
        if not state["is_incident"]:
            return {}
            
        print("--- [Node] VAR Analysis (ViT) ---")
        self.scheduler.check_and_optimize("vit")
        vit_model = self.loader.load_vit()
        
        # Run foul classification on recent frame buffer...
        analysis = {"foul_type": "none", "severity": 0, "confidence": 0.0}
        
        return {"foul_analysis": analysis}

    def memory_node(self, state: AgentState) -> Dict:
        """
        Node: Update match memory and logging.
        """
        if self.memory is None or self.memory.match_id != state["match_id"]:
            self.memory = MatchMemory(state["match_id"])
            
        self.memory.update_frame(
            state["frame_idx"], 
            state["timestamp"], 
            state["players"], 
            event_type=state["event_type"],
            off_ball=state["off_ball_metrics"]
        )
        
        if state["foul_analysis"]:
            self.memory.log_incident(state["frame_idx"], state["timestamp"], state["foul_analysis"])
            
        return {}
