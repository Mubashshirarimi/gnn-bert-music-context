"""Task 4: contrastive GNN-BERT dual encoder and retrieval evaluation.

Learns a shared embedding space between music structure graphs and their
natural-language context strings with a symmetric InfoNCE objective, then
evaluates caption->audio and audio->caption retrieval at R@K.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bert_encoder import BertTextEncoder
from .gnn_model import GNNEncoder


class DualEncoder(nn.Module):
    """Projects graphs and text into a shared L2-normalised embedding space."""

    def __init__(
        self,
        node_in_dim: int,
        gnn_cfg: dict,
        text_cfg: dict,
        embed_dim: int = 256,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
    ) -> None:
        super().__init__()
        self.gnn = GNNEncoder(
            in_dim=node_in_dim,
            hidden_dim=gnn_cfg.get("hidden_dim", 256),
            num_layers=gnn_cfg.get("num_layers", 3),
            conv=gnn_cfg.get("conv", "sage"),
            dropout=gnn_cfg.get("dropout", 0.2),
            heads=gnn_cfg.get("heads", 4),
            readout=gnn_cfg.get("readout", "mean_max"),
        )
        self.text = BertTextEncoder(
            model_name=text_cfg.get("model_name", "bert-base-uncased"),
            freeze=text_cfg.get("freeze_bert", False),
            unfreeze_last_n=text_cfg.get("unfreeze_last_n", 4),
        )
        self.graph_proj = nn.Sequential(
            nn.Linear(self.gnn.out_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(self.text.hidden_size, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # Optimise log-temperature (CLIP convention): keeps tau positive and
        # makes the effective learning rate on it scale-free.
        log_temp = torch.tensor(float(np.log(temperature)))
        self.log_temperature = nn.Parameter(log_temp) if learnable_temperature else log_temp

    @property
    def temperature(self) -> torch.Tensor:
        # Clamped to stop the logit scale running away early in training.
        return self.log_temperature.clamp(min=float(np.log(0.01)), max=float(np.log(0.5))).exp()

    def encode_graph(self, batch) -> torch.Tensor:
        embedding, _ = self.gnn(batch.x, batch.edge_index, batch.batch)
        return F.normalize(self.graph_proj(embedding), dim=1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.text(input_ids, attention_mask)
        return F.normalize(self.text_proj(pooled), dim=1)

    def forward(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_graph(batch), self.encode_text(batch.input_ids, batch.attention_mask)


def info_nce_loss(
    graph_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    temperature: torch.Tensor | float,
    duplicate_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Symmetric InfoNCE over an in-batch similarity matrix.

    ``duplicate_mask[i, j] = True`` marks non-diagonal pairs that should not be
    treated as negatives. MTAT contains several 29 s clips cut from the same
    track, which share an identical metadata string; scoring them as negatives
    would penalise the model for being right.
    """
    logits = graph_embeddings @ text_embeddings.T / temperature
    if duplicate_mask is not None:
        logits = logits.masked_fill(duplicate_mask, float("-inf"))

    targets = torch.arange(logits.shape[0], device=logits.device)
    loss_g2t = F.cross_entropy(logits, targets)
    loss_t2g = F.cross_entropy(logits.T, targets)
    loss = 0.5 * (loss_g2t + loss_t2g)
    return loss, {
        "loss_graph_to_text": float(loss_g2t.detach()),
        "loss_text_to_graph": float(loss_t2g.detach()),
        "total_loss": float(loss.detach()),
    }


def build_duplicate_mask(group_ids: list[str] | np.ndarray) -> torch.Tensor:
    """True for off-diagonal entries whose rows share a group id (same track)."""
    ids = np.asarray(group_ids)
    same = torch.from_numpy(ids[:, None] == ids[None, :])
    same.fill_diagonal_(False)
    return same


@torch.no_grad()
def retrieval_metrics(
    graph_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
    group_ids: np.ndarray | None = None,
) -> dict[str, float]:
    """Recall@K and median rank in both retrieval directions.

    A hit counts if the retrieved item is the true pair *or* belongs to the same
    source track, mirroring the duplicate handling used during training.
    """
    similarity = graph_embeddings @ text_embeddings.T
    n = similarity.shape[0]
    device = similarity.device

    if group_ids is not None:
        ids = np.asarray(group_ids)
        relevant = torch.from_numpy(ids[:, None] == ids[None, :]).to(device)
    else:
        relevant = torch.eye(n, dtype=torch.bool, device=device)

    results: dict[str, float] = {}
    for name, matrix, relevance in (
        ("caption_to_audio", similarity.T, relevant.T),
        ("audio_to_caption", similarity, relevant),
    ):
        order = matrix.argsort(dim=1, descending=True)
        hits = torch.gather(relevance, 1, order)          # (n, n) bool, ranked
        first_hit = hits.float().argmax(dim=1)            # rank of best true match
        for k in ks:
            results[f"{name}_R@{k}"] = float(hits[:, :k].any(dim=1).float().mean())
        results[f"{name}_median_rank"] = float(first_hit.median() + 1)
        results[f"{name}_mean_rank"] = float(first_hit.float().mean() + 1)
    return results


@torch.no_grad()
def qualitative_retrieval(
    graph_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    captions: list[str],
    clip_ids: list[str],
    num_queries: int = 10,
    top_k: int = 3,
    seed: int = 42,
) -> list[dict]:
    """Sample caption queries and return their top-k retrieved clips."""
    rng = np.random.default_rng(seed)
    n = graph_embeddings.shape[0]
    queries = rng.choice(n, size=min(num_queries, n), replace=False)

    similarity = text_embeddings[queries] @ graph_embeddings.T
    scores, indices = similarity.topk(min(top_k, n), dim=1)

    examples = []
    for row, query in enumerate(queries):
        examples.append(
            {
                "query_caption": captions[query],
                "ground_truth_clip": clip_ids[query],
                "retrieved": [
                    {
                        "clip_id": clip_ids[int(idx)],
                        "caption": captions[int(idx)],
                        "score": float(score),
                        "correct": clip_ids[int(idx)] == clip_ids[query],
                    }
                    for score, idx in zip(scores[row], indices[row])
                ],
            }
        )
    return examples
