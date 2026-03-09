"""
========================================================================================
    models/gnn_model.py
    Graph Attention Network (GATv2) for Rare Variant Pathogenicity Prediction

    Architecture:
        Raw node features
            → FeatureEncoder (MLP)
            → L × GATv2Block  (all output hidden_channels — uniform for JK)
            → JumpingKnowledge aggregation
            → MLP Classifier
            → Pathogenicity score ∈ [0, 1]
========================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv, JumpingKnowledge,
    LayerNorm,
)
from torch_geometric.utils import dropout_edge, add_self_loops, remove_self_loops
from typing import Optional, List, Tuple
import numpy as np


# ─── Focal Loss ───────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss — down-weights easy (well-classified) examples so training
    focuses on hard borderline variants.
    FL(p) = -alpha * (1-p)^gamma * log(p)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce   = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        probs = torch.sigmoid(logits)
        p_t   = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss  = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()


# ─── Feature Encoder ──────────────────────────────────────────────────────────
class FeatureEncoder(nn.Module):
    """Project raw variant features into a fixed-size embedding."""

    def __init__(self, in_channels: int, hidden_channels: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels * 2),
            nn.LayerNorm(hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── GATv2 Block ──────────────────────────────────────────────────────────────
class GATBlock(nn.Module):
    """
    Single GATv2 message-passing layer.
    Always uses concat=True with `heads` heads → output dim = hidden * heads.
    A projection then reduces it back to `hidden` for uniform JK dimensions.
    """

    def __init__(
        self,
        in_channels:  int,
        hidden:       int,
        heads:        int   = 4,
        dropout:      float = 0.3,
        edge_dropout: float = 0.1,
    ):
        super().__init__()
        self.edge_dropout = edge_dropout

        # GATv2: concat=True → output is hidden * heads
        self.conv = GATv2Conv(
            in_channels, hidden,
            heads=heads,
            dropout=dropout,
            concat=True,
            add_self_loops=True,
        )
        # Project back to `hidden` so every layer emits the same dimension
        self.proj = nn.Linear(hidden * heads, hidden, bias=False)
        self.norm = LayerNorm(hidden)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)

        # Residual: project input to `hidden` if sizes differ
        self.res = (
            nn.Linear(in_channels, hidden, bias=False)
            if in_channels != hidden else nn.Identity()
        )

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        training:   bool = False,
    ) -> torch.Tensor:
        residual = self.res(x)

        if training and self.edge_dropout > 0:
            edge_index, _ = dropout_edge(edge_index, p=self.edge_dropout,
                                         training=training)

        out = self.conv(x, edge_index)       # (N, hidden * heads)
        out = self.proj(out)                 # (N, hidden)
        out = self.norm(out)
        out = self.act(out)
        out = self.drop(out)
        return out + residual                # (N, hidden)


# ─── Main GNN Model ───────────────────────────────────────────────────────────
class RareVariantGNN(nn.Module):
    """
    GATv2-based node classifier for rare variant pathogenicity prediction.

    Every GATv2 layer emits tensors of shape (N, hidden_channels).
    JumpingKnowledge then aggregates across layers:
        - 'cat'  → (N, hidden * num_layers)   fed into classifier
        - 'max'  → (N, hidden)
        - 'last' → (N, hidden)

    Args:
        in_channels:     Raw node feature dimension (from combined feature matrix)
        hidden_channels: Uniform hidden dimension throughout the network
        num_layers:      Number of GATv2 message-passing steps
        heads:           Attention heads per layer (internal; projected away)
        dropout:         Node/weight dropout probability
        edge_dropout:    Edge dropout probability during training
        jk_mode:         JumpingKnowledge aggregation: 'cat', 'max', or 'last'
    """

    def __init__(
        self,
        in_channels:      int,
        hidden_channels:  int   = 128,
        num_layers:       int   = 3,
        heads:            int   = 4,
        dropout:          float = 0.3,
        edge_dropout:     float = 0.1,
        jk_mode:          str   = 'cat',
    ):
        super().__init__()
        self.num_layers = num_layers
        self.jk_mode    = jk_mode
        self.hidden     = hidden_channels

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoder = FeatureEncoder(in_channels, hidden_channels, dropout)

        # ── GATv2 layers — all emit (N, hidden_channels) ─────────────────────
        self.gat_layers = nn.ModuleList([
            GATBlock(
                in_channels  = hidden_channels,   # every layer same size after encoder
                hidden       = hidden_channels,
                heads        = heads,
                dropout      = dropout,
                edge_dropout = edge_dropout,
            )
            for _ in range(num_layers)
        ])

        # ── Jumping Knowledge ─────────────────────────────────────────────────
        # All layer outputs are (N, hidden) → JK works uniformly.
        # 'cat' mode: channels arg = per-layer dim; output = hidden * num_layers
        # 'max'/'last' mode: output = hidden
        self.jk = JumpingKnowledge(
            mode       = jk_mode,
            channels   = hidden_channels,
            num_layers = num_layers,
        )

        # Compute classifier input size exactly
        if jk_mode == 'cat':
            clf_in = hidden_channels * num_layers
        else:
            clf_in = hidden_channels

        # ── Classifier MLP ───────────────────────────────────────────────────
        clf_mid = max(hidden_channels // 2, 32)
        self.classifier = nn.Sequential(
            nn.Linear(clf_in, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, clf_mid),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(clf_mid, 1),
        )

        # ── Loss ─────────────────────────────────────────────────────────────
        self.criterion = FocalLoss(alpha=0.25, gamma=2.0)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        batch:      Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits : (N,) raw scores
            probs  : (N,) pathogenicity probabilities via sigmoid
        """
        x = self.encoder(x)                  # (N, hidden)

        xs: List[torch.Tensor] = []
        for layer in self.gat_layers:
            x = layer(x, edge_index, training=self.training)
            xs.append(x)                     # each (N, hidden)

        x = self.jk(xs)                      # (N, hidden*L) or (N, hidden)
        logits = self.classifier(x).squeeze(-1)
        probs  = torch.sigmoid(logits)
        return logits, probs

    def loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            logits = logits[mask]
            labels = labels[mask]
        return self.criterion(logits, labels.float())

    @torch.no_grad()
    def predict(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        logits, probs = self.forward(x, edge_index)
        return (probs > 0.5).long(), probs


# ─── Ensemble ─────────────────────────────────────────────────────────────────
class EnsembleGNN(nn.Module):
    def __init__(self, models: List[RareVariantGNN]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x, edge_index):
        probs_list = []
        for m in self.models:
            _, p = m(x, edge_index)
            probs_list.append(p.unsqueeze(0))
        mean_probs = torch.cat(probs_list, dim=0).mean(dim=0)
        logits = torch.logit(mean_probs.clamp(1e-7, 1 - 1e-7))
        return logits, mean_probs


# ─── Factory ──────────────────────────────────────────────────────────────────
def build_model(
    in_channels:     int,
    hidden_channels: int   = 128,
    num_layers:      int   = 3,
    heads:           int   = 4,
    dropout:         float = 0.3,
    jk_mode:         str   = 'cat',
) -> RareVariantGNN:
    return RareVariantGNN(
        in_channels     = in_channels,
        hidden_channels = hidden_channels,
        num_layers      = num_layers,
        heads           = heads,
        dropout         = dropout,
        jk_mode         = jk_mode,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    configs = [
        dict(hidden_channels=64,  num_layers=2, heads=4, jk_mode='cat'),
        dict(hidden_channels=128, num_layers=3, heads=4, jk_mode='cat'),
        dict(hidden_channels=64,  num_layers=3, heads=8, jk_mode='max'),
        dict(hidden_channels=32,  num_layers=2, heads=2, jk_mode='last'),
    ]
    N, F, E = 230, 48, 7328
    x          = torch.randn(N, F)
    edge_index = torch.randint(0, N, (2, E))

    for cfg in configs:
        model  = build_model(in_channels=F, **cfg)
        logits, probs = model(x, edge_index)
        assert logits.shape == (N,), f"Bad logits shape: {logits.shape}"
        assert probs.shape  == (N,), f"Bad probs shape: {probs.shape}"
        print(f"  hidden={cfg['hidden_channels']:3d} layers={cfg['num_layers']} "
              f"heads={cfg['heads']} jk={cfg['jk_mode']:4s} "
              f"→ params={count_parameters(model):,}  ✓")

    print("\nAll smoke tests passed ✓")
