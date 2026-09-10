"""Evaluation metrics: multi-label tagging, emotion regression, graph coherence."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def _to_numpy(array) -> np.ndarray:
    return array.detach().cpu().numpy() if isinstance(array, torch.Tensor) else np.asarray(array)


def _valid_tag_columns(y_true: np.ndarray) -> np.ndarray:
    """Columns with at least one positive AND one negative.

    Tags that are all-zero (or all-one) in a split have undefined AUC and a
    degenerate F1; averaging over them silently drags the macro score toward
    zero, so they are excluded and their count is reported instead.
    """
    positives = y_true.sum(axis=0)
    return (positives > 0) & (positives < y_true.shape[0])


def search_thresholds(y_true, y_score, grid: np.ndarray | None = None) -> np.ndarray:
    """Per-tag decision threshold maximising F1 on the given (validation) split.

    Must only ever be fitted on validation data and then applied unchanged to
    test, otherwise the test F1 is optimistically biased.
    """
    y_true, y_score = _to_numpy(y_true), _to_numpy(y_score)
    grid = np.arange(0.05, 0.96, 0.05) if grid is None else grid

    thresholds = np.full(y_true.shape[1], 0.5)
    for tag in range(y_true.shape[1]):
        truth = y_true[:, tag]
        if truth.sum() == 0:
            continue
        scores = y_score[:, tag]
        best_f1, best_threshold = -1.0, 0.5
        for threshold in grid:
            f1 = f1_score(truth, (scores >= threshold).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_threshold = f1, float(threshold)
        thresholds[tag] = best_threshold
    return thresholds


def tagging_metrics(
    y_true, y_score, thresholds: np.ndarray | float = 0.5, tag_names: list[str] | None = None
) -> dict:
    """Macro/micro F1, mean AUC-PR and mean ROC-AUC for multi-label tagging."""
    y_true, y_score = _to_numpy(y_true), _to_numpy(y_score)
    if np.isscalar(thresholds):
        thresholds = np.full(y_true.shape[1], float(thresholds))
    y_pred = (y_score >= thresholds[None, :]).astype(int)

    valid = _valid_tag_columns(y_true)
    n_valid = int(valid.sum())

    results = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "num_tags_evaluated": n_valid,
        "num_tags_total": int(y_true.shape[1]),
        "num_samples": int(y_true.shape[0]),
    }

    if n_valid:
        results["auc_pr_macro"] = float(
            average_precision_score(y_true[:, valid], y_score[:, valid], average="macro")
        )
        results["auc_pr_micro"] = float(
            average_precision_score(y_true[:, valid], y_score[:, valid], average="micro")
        )
        results["roc_auc_macro"] = float(
            roc_auc_score(y_true[:, valid], y_score[:, valid], average="macro")
        )
    else:
        results.update({"auc_pr_macro": float("nan"), "auc_pr_micro": float("nan"), "roc_auc_macro": float("nan")})

    if tag_names is not None:
        per_tag = {}
        for i, name in enumerate(tag_names):
            if not valid[i]:
                continue
            per_tag[name] = {
                "f1": float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)),
                "auc_pr": float(average_precision_score(y_true[:, i], y_score[:, i])),
                "support": int(y_true[:, i].sum()),
                "threshold": float(thresholds[i]),
            }
        results["per_tag"] = per_tag

    return results


def emotion_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE and R^2 for valence/arousal regression (DEAM)."""
    y_true, y_pred = _to_numpy(y_true), _to_numpy(y_pred)
    if y_true.ndim == 1:
        y_true, y_pred = y_true[:, None], y_pred[:, None]

    results: dict[str, float] = {}
    for i, name in enumerate(["valence", "arousal"][: y_true.shape[1]]):
        truth, pred = y_true[:, i], y_pred[:, i]
        residual = ((truth - pred) ** 2).sum()
        total = ((truth - truth.mean()) ** 2).sum()
        results[f"mae_{name}"] = float(np.abs(truth - pred).mean())
        results[f"rmse_{name}"] = float(np.sqrt(((truth - pred) ** 2).mean()))
        results[f"r2_{name}"] = float(1.0 - residual / total) if total > 0 else float("nan")

    mae_keys = [k for k in results if k.startswith("mae_")]
    results["mae_mean"] = float(np.mean([results[k] for k in mae_keys]))
    return results


@torch.no_grad()
def graph_coherence_score(
    node_embeddings: torch.Tensor, edge_index: torch.Tensor, threshold: float = 0.5
) -> float:
    """Fraction of edges whose endpoint embeddings exceed a cosine threshold.

        S_graph = (1 / |E|) * sum_{(i,j) in E} 1[cos(h_i, h_j) > tau]

    A high score means message passing has pulled structurally linked segments
    (repeats, shared chords) into agreement rather than smoothing everything
    uniformly.
    """
    if edge_index.numel() == 0:
        return float("nan")
    normed = torch.nn.functional.normalize(node_embeddings, dim=1)
    src, dst = edge_index[0], edge_index[1]
    # Ignore self-loops, whose cosine similarity is trivially 1.
    keep = src != dst
    if keep.sum() == 0:
        return float("nan")
    cosine = (normed[src[keep]] * normed[dst[keep]]).sum(dim=1)
    return float((cosine > threshold).float().mean())


def random_baseline_metrics(y_true, seed: int = 42, strategy: str = "prior") -> dict:
    """Baseline B1: random / prior-frequency tag predictor.

    ``prior`` scores every clip with each tag's training frequency, which is a
    genuinely hard-to-beat trivial baseline on AUC-PR for imbalanced tags.
    """
    y_true = _to_numpy(y_true)
    rng = np.random.default_rng(seed)
    if strategy == "prior":
        prior = y_true.mean(axis=0)
        y_score = np.tile(prior, (y_true.shape[0], 1))
        # Break ties randomly so ranking metrics are not degenerate.
        y_score = y_score + rng.normal(0, 1e-6, size=y_score.shape)
    else:
        y_score = rng.random(y_true.shape)
    return tagging_metrics(y_true, y_score, thresholds=0.5)
