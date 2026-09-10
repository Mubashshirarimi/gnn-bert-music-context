"""Task 3: GNN-BERT fusion for multi-context understanding.

Four fusion modes are implemented behind one interface so the ablation table in
the report is an apples-to-apples comparison - same heads, same optimiser, same
schedule, only the fusion path changes:

  * ``bert_only``       - text branch alone (reproduces Task 1 inside this model)
  * ``gnn_only``        - graph branch alone (reproduces Task 2)
  * ``concat``          - early concatenation of the two pooled vectors
  * ``cross_attention`` - graph embedding attends over BERT token states
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .bert_encoder import BertTextEncoder
from .gnn_model import GNNEncoder

FUSION_MODES = ("cross_attention", "concat", "gnn_only", "bert_only")


class CrossAttentionFusion(nn.Module):
    """Graph-to-text cross-attention.

    The graph embedding g forms a single query; BERT's token states form keys
    and values. The attended text vector is concatenated with g:

        A = softmax(Q K^T / sqrt(d)),  Q = g W_Q,  K = H_text W_K
        z = CONCAT(g, A H_text)

    Attention weights are returned so the report's case studies can show which
    words a track's structure attends to.
    """

    def __init__(self, graph_dim: int, text_dim: int, hidden_dim: int, num_heads: int = 8, dropout: float = 0.3) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}")
        self.query_proj = nn.Linear(graph_dim, hidden_dim)
        self.kv_proj = nn.Linear(text_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_dim = graph_dim + hidden_dim

    def forward(
        self, graph_embedding: torch.Tensor, text_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query_proj(graph_embedding).unsqueeze(1)   # (B, 1, H)
        keys = self.kv_proj(text_states)                        # (B, L, H)
        # MultiheadAttention masks positions where the pad mask is True.
        attended, weights = self.attention(
            query, keys, keys, key_padding_mask=(attention_mask == 0), need_weights=True
        )
        attended = self.norm(attended.squeeze(1))
        return torch.cat([graph_embedding, attended], dim=1), weights.squeeze(1)


class GNNBertFusion(nn.Module):
    """End-to-end fusion model with a tag head and optional emotion heads."""

    def __init__(
        self,
        node_in_dim: int,
        num_tags: int,
        gnn_cfg: dict,
        fusion_cfg: dict,
        text_cfg: dict,
        emotion_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.mode = fusion_cfg.get("mode", "cross_attention")
        if self.mode not in FUSION_MODES:
            raise ValueError(f"Unknown fusion mode {self.mode!r}; expected one of {FUSION_MODES}")
        self.emotion_enabled = emotion_enabled

        self.gnn: GNNEncoder | None = None
        self.text: BertTextEncoder | None = None
        self.cross: CrossAttentionFusion | None = None

        if self.mode != "bert_only":
            self.gnn = GNNEncoder(
                in_dim=node_in_dim,
                hidden_dim=gnn_cfg.get("hidden_dim", 256),
                num_layers=gnn_cfg.get("num_layers", 3),
                conv=gnn_cfg.get("conv", "sage"),
                dropout=gnn_cfg.get("dropout", 0.2),
                heads=gnn_cfg.get("heads", 4),
                readout=gnn_cfg.get("readout", "mean_max"),
            )
        if self.mode != "gnn_only":
            self.text = BertTextEncoder(
                model_name=text_cfg.get("model_name", "bert-base-uncased"),
                freeze=text_cfg.get("freeze_bert", False),
                unfreeze_last_n=text_cfg.get("unfreeze_last_n", 4),
            )

        hidden_dim = fusion_cfg.get("hidden_dim", 512)
        dropout = fusion_cfg.get("dropout", 0.3)

        if self.mode == "cross_attention":
            self.cross = CrossAttentionFusion(
                graph_dim=self.gnn.out_dim,
                text_dim=self.text.hidden_size,
                hidden_dim=hidden_dim,
                num_heads=fusion_cfg.get("num_heads", 8),
                dropout=dropout,
            )
            fused_dim = self.cross.out_dim
        elif self.mode == "concat":
            fused_dim = self.gnn.out_dim + self.text.hidden_size
        elif self.mode == "gnn_only":
            fused_dim = self.gnn.out_dim
        else:  # bert_only
            fused_dim = self.text.hidden_size

        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.tag_head = nn.Linear(hidden_dim, num_tags)
        if emotion_enabled:
            # Valence and arousal as two scalar regressions (DEAM targets).
            self.emotion_head = nn.Linear(hidden_dim, 2)

    def forward(self, batch) -> dict[str, torch.Tensor]:
        graph_embedding = text_pooled = None
        attention_weights = None

        if self.gnn is not None:
            graph_embedding, _ = self.gnn(batch.x, batch.edge_index, batch.batch)
        if self.text is not None:
            text_pooled, text_states = self.text(batch.input_ids, batch.attention_mask)

        if self.mode == "cross_attention":
            fused, attention_weights = self.cross(graph_embedding, text_states, batch.attention_mask)
        elif self.mode == "concat":
            fused = torch.cat([graph_embedding, text_pooled], dim=1)
        elif self.mode == "gnn_only":
            fused = graph_embedding
        else:
            fused = text_pooled

        hidden = self.trunk(fused)
        outputs = {"tag_logits": self.tag_head(hidden), "fused": hidden}
        if self.emotion_enabled:
            outputs["emotion"] = self.emotion_head(hidden)
        if attention_weights is not None:
            outputs["text_attention"] = attention_weights
        return outputs


class MultiTaskLoss(nn.Module):
    """L = L_tags + lambda * (MSE_valence + MSE_arousal).

    ``pos_weight`` compensates MTAT's heavy class imbalance: even among the top
    50 tags, positives are roughly 1-10% of clips, and unweighted BCE collapses
    to predicting all-zeros for the rarer tags.
    """

    def __init__(self, emotion_weight: float = 0.5, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.tag_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.emotion_loss = nn.MSELoss()
        self.emotion_weight = emotion_weight

    def forward(self, outputs: dict, targets: dict) -> tuple[torch.Tensor, dict[str, float]]:
        loss = self.tag_loss(outputs["tag_logits"], targets["tags"])
        parts = {"tag_loss": float(loss.detach())}

        if "emotion" in outputs and targets.get("emotion") is not None:
            emotion = self.emotion_loss(outputs["emotion"], targets["emotion"])
            loss = loss + self.emotion_weight * emotion
            parts["emotion_loss"] = float(emotion.detach())

        parts["total_loss"] = float(loss.detach())
        return loss, parts
