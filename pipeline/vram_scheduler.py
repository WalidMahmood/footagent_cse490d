import torch
import time
from typing import Dict, List, Optional
from .model_loaders import ModelLoader


class VRAMScheduler:
    """
    Manages VRAM budget by orchestrating model loads/unloads.
    Target budget: 8GB VRAM.
    """

    def __init__(self, loader: ModelLoader, vram_limit_gb: float = 7.5):
        self.loader = loader
        self.vram_limit = vram_limit_gb
        self.priorities = {
            "yolo": 1,  # Always on
            "gat": 2,   # Frequent
            "vit": 3    # Triggered (highest load)
        }

    def check_and_optimize(self, requested_model: str):
        """
        Ensure requested model can be loaded within budget.
        Unloads lower priority models if needed.
        """
        current_vram = self.get_current_vram_gb()
        
        if current_vram > self.vram_limit:
            print(f"[VRAMScheduler] Warning: VRAM at {current_vram:.2f}GB. Optimizing...")
            self._evict_models(requested_model)

    def _evict_models(self, keep_model: str):
        """Evict models to make room."""
        active = self.loader.get_active_models()
        # Sort by priority (higher value = less important for constant runtime)
        to_check = sorted([m for m in active if m != keep_model], 
                          key=lambda x: self.priorities.get(x, 0), 
                          reverse=True)
        
        for model in to_check:
            if self.get_current_vram_gb() < self.vram_limit * 0.8:
                break
            self.loader.unload_model(model)

    def get_current_vram_gb(self) -> float:
        """Get current VRAM usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0

    def monitor_step(self):
        """Standard heartbeat to check health."""
        vram = self.get_current_vram_gb()
        if vram > self.vram_limit * 0.95:
             print(f"[VRAMScheduler] CRITICAL: VRAM at {vram:.2f}GB")
             # Emergency eviction
             active = self.loader.get_active_models()
             if "vit" in active:
                 self.loader.unload_model("vit")
