"""Graph encoders over music structure graphs (Task 2).

Implements a configurable GraphSAGE / GAT stack with residual connections,
batch normalisation and a mean+max readout, plus a standalone classifier head
for the GNN-only experiments and the Task 3 ablation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, global_max_pool, global_mean_pool


class GNNEncoder(nn.Module):
    """Message-passing encoder producing a fixed-size graph embedding.

    GraphSAGE update (matches the project spec):
        h_i^{l+1} = sigma( W^{l} . CONCAT( h_i^{l}, MEAN_{j in N(i)} h_j^{l} ) )

    PyG's ``SAGEConv`` implements exactly this concat-of-self-and-mean form.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        conv: str = "sage",
        dropout: float = 0.2,
        heads: int = 4,
        readout: str = "mean_max",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.conv_type = conv.lower()
        self.readout = readout
        self.dropout = dropout

        # Project raw acoustic features to hidden width first, so every graph
        # layer (and the residual connections) operate at a constant dimension.
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(self._make_conv(hidden_dim, hidden_dim, heads))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.out_dim = hidden_dim * (2 if readout == "mean_max" else 1)

    def _make_conv(self, in_dim: int, out_dim: int, heads: int) -> nn.Module:
        if self.conv_type == "sage":
            return SAGEConv(in_dim, out_dim, aggr="mean")
        if self.conv_type == "gat":
            # Concatenated heads must recombine to out_dim so residuals line up.
            if out_dim % heads != 0:
                raise ValueError(f"hidden_dim {out_dim} must be divisible by heads {heads}")
            return GATConv(in_dim, out_dim // heads, heads=heads, concat=True, dropout=self.dropout)
        raise ValueError(f"Unknown conv type {self.conv_type!r}; expected 'sage' or 'gat'")

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (graph_embedding (B, out_dim), node_embeddings (N, hidden))."""
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            residual = h
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + residual  # stabilises deeper stacks on small graphs
        return self._pool(h, batch), h

    def _pool(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.readout == "mean":
            return global_mean_pool(h, batch)
        if self.readout == "max":
            return global_max_pool(h, batch)
        if self.readout == "mean_max":
            # Mean captures overall texture; max captures the single most
            # distinctive segment, which matters for sparse instrument tags.
            return torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)
        raise ValueError(f"Unknown readout {self.readout!r}")


class GNNTagClassifier(nn.Module):
    """Task 2: GNN encoder + MLP head for multi-label tagging."""

    def __init__(
        self,
        in_dim: int,
        num_tags: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        conv: str = "sage",
        dropout: float = 0.2,
        heads: int = 4,
        readout: str = "mean_max",
    ) -> None:
        super().__init__()
        self.encoder = GNNEncoder(in_dim, hidden_dim, num_layers, conv, dropout, heads, readout)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tags),
        )

    def forward(self, data) -> torch.Tensor:
        graph_embedding, _ = self.encoder(data.x, data.edge_index, data.batch)
        return self.head(graph_embedding)


class MelCNNBaseline(nn.Module):
    """Baseline B2: a plain 2-D CNN over the log-mel spectrogram.

    Deliberately no graph and no text - this is the comparison the spec asks
    for, isolating what the graph structure actually contributes.
    """

    def __init__(self, num_tags: int, n_mels: int = 128, channels: int = 64, dropout: float = 0.3) -> None:
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(1, channels),
            block(channels, channels),
            block(channels, channels * 2),
            block(channels * 2, channels * 2),
            block(channels * 2, channels * 4),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, num_tags),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, n_mels, frames) -> logits (B, num_tags)."""
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)
        return self.head(self.pool(self.features(mel)))
