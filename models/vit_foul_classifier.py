"""
ViT-B/16 Foul Classifier for MVFouls.
Bismillah.

Architecture:
  - Backbone: ViT-B/16 pretrained on ImageNet (timm)
  - Temporal: Average pooling across 16 frames
  - Head: Linear classifier for 9 action classes
  - Auxiliary: Severity prediction (multi-task)
"""
import torch
import torch.nn as nn
import timm


class ViTFoulClassifier(nn.Module):
    """
    ViT-B/16 classifier for foul action recognition.

    Input: [B, 16, 3, 224, 224] — 16 frames per clip
    Output: action_logits [B, n_classes], severity_logits [B, 4]
    """

    def __init__(self, n_classes=9, n_severity=4, dropout=0.3,
                 freeze_backbone=False):
        super().__init__()

        # Load pretrained ViT-B/16
        self.backbone = timm.create_model('vit_base_patch16_224',
                                           pretrained=True,
                                           num_classes=0)  # Remove head
        self.feat_dim = self.backbone.num_features  # 768 for ViT-B

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Classification heads
        self.action_head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

        self.severity_head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, 64),
            nn.GELU(),
            nn.Linear(64, n_severity),
        )

    def forward(self, frames):
        """
        Args:
            frames: [B, T, 3, H, W] — batch of clip frame sequences
        Returns:
            action_logits: [B, n_classes]
            severity_logits: [B, n_severity]
        """
        B, T, C, H, W = frames.shape

        # Flatten batch and time dimensions
        x = frames.reshape(B * T, C, H, W)

        # Extract features for each frame
        feats = self.backbone(x)  # [B*T, feat_dim]

        # Reshape back and temporal average pool
        feats = feats.reshape(B, T, -1)  # [B, T, feat_dim]
        feats = feats.mean(dim=1)  # [B, feat_dim]

        # Classify
        action_logits = self.action_head(feats)
        severity_logits = self.severity_head(feats)

        return action_logits, severity_logits

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


class ViTFoulClassifierFrozen(ViTFoulClassifier):
    """ViT-B with frozen backbone — only trains classification heads."""

    def __init__(self, **kwargs):
        kwargs['freeze_backbone'] = True
        super().__init__(**kwargs)


if __name__ == '__main__':
    model = ViTFoulClassifier(n_classes=9)
    total, trainable = model.count_parameters()
    print(f"ViT-B Foul Classifier:")
    print(f"  Total params: {total:,}")
    print(f"  Trainable params: {trainable:,}")

    # Test forward pass
    dummy = torch.randn(2, 16, 3, 224, 224)
    act, sev = model(dummy)
    print(f"  Input: {dummy.shape}")
    print(f"  Action logits: {act.shape}")
    print(f"  Severity logits: {sev.shape}")
