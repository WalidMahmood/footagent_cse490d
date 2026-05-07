import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights


class R2Plus1DFoulClassifier(nn.Module):
    """
    R(2+1)D-18 Foul Classifier.
    Specialized in temporal actions by decomposing 3D convolutions.
    """

    def __init__(self, n_classes=9, dropout=0.5):
        super().__init__()
        
        # Load pretrained R(2+1)D-18
        self.backbone = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        
        # Replace head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, n_classes)
        )
        
        print(f"[R2Plus1D] Initialized with {n_classes} action classes.")

    def forward(self, x):
        """
        Input: [B, C, T, H, W] — e.g., [B, 3, 16, 112, 112]
        """
        return self.backbone(x)


if __name__ == "__main__":
    model = R2Plus1DFoulClassifier()
    # Test pass
    dummy = torch.randn(2, 3, 16, 112, 112)
    output = model(dummy)
    print(f"Input: {dummy.shape}")
    print(f"Output: {output.shape}")
