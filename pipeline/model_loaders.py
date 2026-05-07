from typing import Dict, List, Optional
import torch
from ultralytics import YOLO
from models.gat.temporal_gat import TemporalGAT
from models.vit_foul_classifier import ViTFoulClassifier
import os


class ModelLoader:
    """
    Handles lazy loading and VRAM management for all models in the FootAgent ecosystem.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.models = {}
        print(f"[ModelLoader] Initialized on {self.device}")

    def load_yolo(self, model_size: str = "n"):
        """Load YOLO detection model."""
        if "yolo" not in self.models:
            model_path = "yolov8n.pt"
            
            print(f"[ModelLoader] Loading YOLO ({model_size})...")
            self.models["yolo"] = YOLO(model_path).to(self.device)
        return self.models["yolo"]

    def load_gat(self, checkpoint_path: str):
        """Load Temporal GAT model from checkpoint."""
        if "gat" not in self.models:
            print(f"[ModelLoader] Loading Temporal GAT from {checkpoint_path}...")
            model = TemporalGAT(in_channels=5, hidden=64, out_channels=32)
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device)
            model.eval()
            self.models["gat"] = model
        return self.models["gat"]

    def load_vit(self, checkpoint_path: Optional[str] = None):
        """Load ViT Foul Classifier."""
        if "vit" not in self.models:
            print(f"[ModelLoader] Loading ViT Foul Classifier...")
            model = ViTFoulClassifier(n_classes=9)
            if checkpoint_path and os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device)
            model.eval()
            self.models["vit"] = model
        return self.models["vit"]

    def unload_model(self, model_name: str):
        """Unload a model to free VRAM."""
        if model_name in self.models:
            print(f"[ModelLoader] Unloading {model_name}...")
            del self.models[model_name]
            torch.cuda.empty_cache()
            return True
        return False

    def get_active_models(self) -> List[str]:
        return list(self.models.keys())
