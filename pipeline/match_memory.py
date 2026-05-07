import json
import os
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional


class MatchMemory:
    """
    JSON-based match state logging and memory system.
    Stores events, player stats, and analysis results during a match.
    """

    def __init__(self, match_id: str, log_dir: str = "data/matches"):
        self.match_id = match_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        
        self.file_path = self.log_dir / f"{match_id}_memory.json"
        self.memory = {
            "metadata": {
                "match_id": match_id,
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "version": "1.0"
            },
            "frames": [],
            "incidents": [],
            "stats": {
                "total_frames": 0,
                "team_possession": {0: 0.0, 1: 0.0}
            }
        }
        
        # Initial save
        self._save()

    def update_frame(self, frame_idx: int, timestamp: float, players: List[Dict], 
                     ball: Optional[Dict] = None, event_type: str = "normal",
                     off_ball: Optional[Dict] = None):
        """Append frame-level tracking data."""
        frame_data = {
            "frame": frame_idx,
            "timestamp": timestamp,
            "event_type": event_type,
            "players": players,
            "ball": ball,
            "off_ball": off_ball
        }
        self.memory["frames"].append(frame_data)
        self.memory["stats"]["total_frames"] += 1
        
        self._save()

    def log_incident(self, frame_idx: int, timestamp: float, 
                     analysis: Dict, frames_path: Optional[str] = None):
        """Log a specialized incident (foul, card, etc.)."""
        incident_data = {
            "id": f"inc_{len(self.memory['incidents'])}",
            "frame": frame_idx,
            "timestamp": timestamp,
            "analysis": analysis,
            "frames_path": frames_path
        }
        self.memory["incidents"].append(incident_data)
        self._save()

    def get_recent_history(self, n_frames: int = 50) -> List[Dict]:
        """Get the last N frames of tracking data."""
        return self.memory["frames"][-n_frames:]

    def generate_report(self) -> Dict:
        """Aggregate data into a summary report."""
        return {
            "match_summary": {
                "total_duration": self.memory["frames"][-1]["timestamp"] if self.memory["frames"] else 0,
                "incidents_count": len(self.memory["incidents"]),
                "possession": self.memory["stats"]["team_possession"]
            },
            "key_incidents": self.memory["incidents"]
        }

    def _save(self):
        """Atomic save to JSON with thread safety and retry logic."""
        with self.lock:
            temp_path = self.file_path.with_suffix(".tmp")
            try:
                with open(temp_path, 'w') as f:
                    json.dump(self.memory, f, indent=2)
                
                # Atomic swap with retry for Windows file locking
                max_retries = 5
                for i in range(max_retries):
                    try:
                        if os.path.exists(self.file_path):
                            os.remove(self.file_path)
                        os.rename(temp_path, self.file_path)
                        break
                    except (PermissionError, OSError):
                        if i == max_retries - 1:
                            print(f"[MatchMemory] Warning: Could not save memory after {max_retries} attempts.")
                        time.sleep(0.01)
            except Exception as e:
                print(f"[MatchMemory] Critical Error saving: {e}")


if __name__ == "__main__":
    # Test
    mem = MatchMemory("test_match")
    mem.update_frame(1, 0.04, [{"id": 1, "team": 0, "x": 100, "y": 200}])
    print(f"Memory saved to {mem.file_path}")
