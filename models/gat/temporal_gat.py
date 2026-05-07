"""
Temporal GAT Architecture for Off-Ball Credit Attribution.
Bismillah.

Architecture:
  - Input: Graph per freeze-frame (N players, 5 features each)
  - GATConv layers with multi-head attention
  - Global mean pooling -> MLP -> xG prediction (scalar)

Ablation variants:
  - TemporalGAT: Main model (GAT + temporal aggregator)
  - GATM: GAT-only baseline (no temporal component)
  - GCNBaseline: GCN baseline for comparison
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch


class TemporalGAT(nn.Module):
    """
    Main research model: Graph Attention Network for xG prediction.

    Architecture:
      GAT(5 -> 64, 4 heads) -> GAT(64 -> 32, 4 heads) -> GlobalPool -> MLP -> 1

    Node features: [x, y, teammate, actor, keeper]
    Edge construction: distance-based (17.5m threshold from EDA)
    """

    def __init__(self, in_channels=5, hidden=64, out_channels=32,
                 heads=4, dropout=0.3):
        super().__init__()

        self.dropout = dropout

        # GAT layers
        self.gat1 = GATConv(in_channels, hidden, heads=heads,
                            dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden * heads, out_channels, heads=heads,
                            dropout=dropout, concat=False)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden * heads)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # MLP head for xG prediction
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # xG is in [0, 1]
        )

    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x: [N_total, 5] node features
            edge_index: [2, E_total] edge indices
            batch: [N_total] batch assignment vector
        Returns:
            pred: [B, 1] xG predictions
        """
        # GAT layer 1
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GAT layer 2
        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)

        # Global pooling (graph-level representation)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = global_mean_pool(x, batch)

        # Predict xG
        out = self.mlp(x)
        return out.squeeze(-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GATMBaseline(nn.Module):
    """
    Ablation: Single GAT layer (simpler, fewer parameters).
    Tests whether multi-layer attention provides benefit.
    """

    def __init__(self, in_channels=5, hidden=64, heads=4, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.gat = GATConv(in_channels, hidden, heads=heads,
                           dropout=dropout, concat=False)
        self.bn = nn.BatchNorm1d(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, edge_index, batch=None):
        x = self.gat(x, edge_index)
        x = self.bn(x)
        x = F.elu(x)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = global_mean_pool(x, batch)
        return self.mlp(x).squeeze(-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GCNBaseline(nn.Module):
    """
    Ablation: GCN baseline (no attention mechanism).
    Tests whether attention is needed vs simple message passing.
    """

    def __init__(self, in_channels=5, hidden=64, out_channels=32, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.gcn1 = GCNConv(in_channels, hidden)
        self.gcn2 = GCNConv(hidden, out_channels)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, edge_index, batch=None):
        x = self.gcn1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = global_mean_pool(x, batch)
        return self.mlp(x).squeeze(-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model(name='temporal_gat', **kwargs):
    """Factory function to create model by name."""
    models = {
        'temporal_gat': TemporalGAT,
        'gat_m': GATMBaseline,
        'gcn': GCNBaseline,
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Options: {list(models.keys())}")
    return models[name](**kwargs)


if __name__ == '__main__':
    # Quick sanity check
    print("Model Parameter Counts:")
    for name in ['temporal_gat', 'gat_m', 'gcn']:
        model = get_model(name)
        print(f"  {name}: {model.count_parameters():,} parameters")

    # Test forward pass with dummy data
    model = get_model('temporal_gat')
    x = torch.randn(22, 5)  # 22 players, 5 features
    edge_index = torch.randint(0, 22, (2, 80))  # 80 edges
    out = model(x, edge_index)
    print(f"\nForward pass test: input {x.shape} -> output {out.shape}, value={out.item():.4f}")
