"""Evaluation, plots and result tables for every task.

    python -m src.evaluate --task 1
    python -m src.evaluate --task 3 --run-name task3_concat
    python -m src.evaluate --task 4
    python -m src.evaluate --baselines          # random / prior predictors only

Thresholds are fitted on the validation split and applied unchanged to test, so
reported test F1 is not optimistically biased.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader

from .bert_encoder import build_tokenizer
from .contrastive import qualitative_retrieval, retrieval_metrics
from .dataset import TextTagDataset, load_vocabulary, rebuild_texts
from .metrics import (
    emotion_metrics,
    graph_coherence_score,
    random_baseline_metrics,
    search_thresholds,
    tagging_metrics,
)
from .train import build_model, load_graph_split, load_mel_split
from .utils import (
    get_device,
    load_checkpoint,
    load_config,
    resolve,
    set_seed,
    setup_logging,
    update_metrics_json,
)

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def build_eval_loader(cfg, task: str, split: str):
    batch_size = cfg["contrastive"]["batch_size"] if task == "4" else cfg["train"]["batch_size"]

    if task == "1":
        tokenizer = build_tokenizer(cfg["text"]["model_name"])
        processed = resolve(cfg, "processed_dir")
        payload = torch.load(processed / f"graphs_{split}.pt", map_location="cpu", weights_only=False)
        labels = np.load(processed / f"labels_{split}.npy")
        texts = rebuild_texts(cfg, processed, payload["clip_ids"])
        dataset = TextTagDataset(texts, labels, tokenizer, cfg["text"]["max_length"])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return loader, {"texts": texts, "clip_ids": payload["clip_ids"]}

    if task == "cnn":
        dataset = load_mel_split(cfg, split)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False), {}

    needs_text = task in ("3", "4") and cfg["model"]["fusion"]["mode"] != "gnn_only"
    tokenizer = build_tokenizer(cfg["text"]["model_name"]) if needs_text else None
    dataset = load_graph_split(cfg, split, tokenizer)
    loader = GeoDataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader, {"dataset": dataset, "texts": dataset.texts, "clip_ids": dataset.clip_ids}


@torch.no_grad()
def collect_predictions(model, loader, task: str, device) -> dict:
    """Run inference over a split and gather everything the metrics need."""
    model.eval()
    scores, targets = [], []
    graph_embeddings, text_embeddings, groups = [], [], []
    coherence_samples: list[float] = []
    fused_vectors = []

    for batch in loader:
        if isinstance(batch, (tuple, list)):
            batch = tuple(item.to(device) for item in batch)
        elif isinstance(batch, dict):
            batch = {key: value.to(device) for key, value in batch.items()}
        else:
            batch = batch.to(device)

        if task == "1":
            logits = model(batch["input_ids"], batch["attention_mask"])
            targets.append(batch["labels"].cpu())
        elif task == "cnn":
            mel, y = batch
            logits = model(mel)
            targets.append(y.cpu())
        elif task == "2":
            logits = model(batch)
            targets.append(batch.y.view(logits.shape).cpu())
            _, node_embeddings = model.encoder(batch.x, batch.edge_index, batch.batch)
            coherence_samples.append(graph_coherence_score(node_embeddings, batch.edge_index))
        elif task == "3":
            outputs = model(batch)
            logits = outputs["tag_logits"]
            targets.append(batch.y.view(logits.shape).cpu())
            fused_vectors.append(outputs["fused"].float().cpu())
            if model.gnn is not None:
                _, node_embeddings = model.gnn(batch.x, batch.edge_index, batch.batch)
                coherence_samples.append(graph_coherence_score(node_embeddings, batch.edge_index))
        else:  # task 4
            graph_emb, text_emb = model(batch)
            graph_embeddings.append(graph_emb.float().cpu())
            text_embeddings.append(text_emb.float().cpu())
            if hasattr(batch, "track_group"):
                groups.extend(batch.track_group.cpu().numpy().tolist())
            continue

        scores.append(torch.sigmoid(logits.float()).cpu())

    output: dict = {}
    if scores:
        output["scores"] = torch.cat(scores).numpy()
        output["targets"] = torch.cat(targets).numpy()
    if graph_embeddings:
        output["graph_embeddings"] = torch.cat(graph_embeddings)
        output["text_embeddings"] = torch.cat(text_embeddings)
        output["groups"] = np.array(groups) if groups else None
    if fused_vectors:
        output["fused"] = torch.cat(fused_vectors).numpy()
    if coherence_samples:
        finite = [v for v in coherence_samples if np.isfinite(v)]
        output["graph_coherence"] = float(np.mean(finite)) if finite else float("nan")
    return output


def load_trained_model(cfg, task: str, run_name: str, device, num_tags: int, node_dim: int | None):
    checkpoint = resolve(cfg, "results_dir") / "checkpoints" / f"{run_name}.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"No checkpoint at {checkpoint}; train the task first")
    model = build_model(cfg, task, num_tags, node_dim).to(device)
    meta = load_checkpoint(checkpoint, model, device)
    return model, meta


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_training_curves(history_path: Path, out_path: Path, run_name: str) -> None:
    if not history_path.exists():
        return
    history = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = [h["epoch"] for h in history]

    metric_keys = [k for k in ("val_macro_f1", "val_micro_f1", "val_recall") if k in history[0]]
    fig, axes = plt.subplots(1, 2 if metric_keys else 1, figsize=(11, 4), squeeze=False)

    axes[0][0].plot(epochs, [h["train_loss"] for h in history], label="train", color=PALETTE[0])
    axes[0][0].plot(epochs, [h["val_loss"] for h in history], label="validation", color=PALETTE[1])
    axes[0][0].set_xlabel("epoch"); axes[0][0].set_ylabel("loss")
    axes[0][0].set_title(f"{run_name}: loss"); axes[0][0].legend(); axes[0][0].grid(alpha=0.3)

    if metric_keys:
        for i, key in enumerate(metric_keys):
            axes[0][1].plot(epochs, [h.get(key, np.nan) for h in history],
                            label=key.replace("val_", ""), color=PALETTE[i % len(PALETTE)])
        axes[0][1].set_xlabel("epoch"); axes[0][1].set_ylabel("score")
        axes[0][1].set_title(f"{run_name}: validation metrics")
        axes[0][1].legend(); axes[0][1].grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_tag_f1(metrics: dict, out_path: Path, run_name: str, top_n: int = 50) -> None:
    per_tag = metrics.get("per_tag")
    if not per_tag:
        return
    items = sorted(per_tag.items(), key=lambda kv: kv[1]["support"], reverse=True)[:top_n]
    names = [name for name, _ in items]
    f1s = [values["f1"] for _, values in items]
    aucs = [values["auc_pr"] for _, values in items]

    fig, ax = plt.subplots(figsize=(max(9, len(names) * 0.28), 4.5))
    x = np.arange(len(names))
    ax.bar(x - 0.2, f1s, width=0.4, label="F1", color=PALETTE[0])
    ax.bar(x + 0.2, aucs, width=0.4, label="AUC-PR", color=PALETTE[2])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("score")
    ax.set_title(f"{run_name}: per-tag performance (ordered by support)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_tsne(embeddings: np.ndarray, labels: np.ndarray, tag_names: list[str], out_path: Path,
              run_name: str, max_samples: int = 2000, perplexity: int = 30, seed: int = 42) -> None:
    """t-SNE of the fused representation, coloured by dominant genre and by mood."""
    if embeddings.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(embeddings.shape[0], max_samples, replace=False)
        embeddings, labels = embeddings[idx], labels[idx]

    projected = TSNE(
        n_components=2, perplexity=min(perplexity, max(5, embeddings.shape[0] // 4)),
        init="pca", random_state=seed,
    ).fit_transform(embeddings)

    genre_like = ["classical", "rock", "electro", "jazz", "opera", "ambient", "techno", "pop", "country", "metal"]
    mood_like = ["quiet", "loud", "fast", "slow", "soft", "hard", "weird", "mellow", "dark", "harpsichord"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, keywords, title in ((axes[0], genre_like, "genre-like tags"), (axes[1], mood_like, "mood/texture tags")):
        available = [t for t in keywords if t in tag_names][:6]
        ax.scatter(projected[:, 0], projected[:, 1], s=4, c="#d9d9d9", label="other")
        for i, tag in enumerate(available):
            mask = labels[:, tag_names.index(tag)] > 0
            if mask.sum() == 0:
                continue
            ax.scatter(projected[mask, 0], projected[mask, 1], s=6,
                       color=PALETTE[i % len(PALETTE)], label=tag, alpha=0.75)
        ax.set_title(f"{run_name}: {title}")
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(markerscale=2, fontsize=8, loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_model_comparison(metrics_path: Path, out_path: Path) -> None:
    """Bar chart of Macro-F1 / AUC-PR across every evaluated model."""
    if not metrics_path.exists():
        return
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = [(name, block["test"]) for name, block in data.items()
            if isinstance(block, dict) and isinstance(block.get("test"), dict)
            and "macro_f1" in block["test"]]
    if not rows:
        return
    rows.sort(key=lambda kv: kv[1]["macro_f1"])

    names = [name for name, _ in rows]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(names) + 2))
    y = np.arange(len(names))
    ax.barh(y - 0.2, [m["macro_f1"] for _, m in rows], height=0.4, label="Macro-F1", color=PALETTE[0])
    ax.barh(y + 0.2, [m.get("auc_pr_macro", np.nan) for _, m in rows], height=0.4, label="AUC-PR", color=PALETTE[2])
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlabel("score"); ax.set_title("Model comparison on the MTAT test split")
    ax.legend(); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Task evaluation
# --------------------------------------------------------------------------- #
def evaluate_tagging(cfg, task: str, run_name: str, device, logger) -> dict:
    label_tags, _ = load_vocabulary(resolve(cfg, "processed_dir"))
    plots_dir = resolve(cfg, "plots_dir")

    val_loader, _ = build_eval_loader(cfg, task, "val")
    test_loader, test_extras = build_eval_loader(cfg, task, "test")

    node_dim = None
    if task in ("2", "3"):
        node_dim = int(test_extras["dataset"].data.x.shape[1])

    model, meta = load_trained_model(cfg, task, run_name, device, len(label_tags), node_dim)
    logger.info("Loaded %s (best epoch %s, %s=%.4f)", run_name, meta.get("epoch"),
                meta.get("monitor"), meta.get("score", float("nan")))

    val_out = collect_predictions(model, val_loader, task, device)
    test_out = collect_predictions(model, test_loader, task, device)

    # Fit thresholds on validation only, then freeze them for test.
    if cfg["eval"]["threshold"] == "search":
        thresholds = search_thresholds(val_out["targets"], val_out["scores"])
    else:
        thresholds = float(cfg["eval"]["threshold"])

    val_metrics = tagging_metrics(val_out["targets"], val_out["scores"], thresholds, label_tags)
    test_metrics = tagging_metrics(test_out["targets"], test_out["scores"], thresholds, label_tags)

    for name, out in (("val", val_out), ("test", test_out)):
        if "graph_coherence" in out:
            (val_metrics if name == "val" else test_metrics)["graph_coherence"] = out["graph_coherence"]

    logger.info(
        "%s TEST: macro-F1 %.4f | micro-F1 %.4f | AUC-PR %.4f | ROC-AUC %.4f",
        run_name, test_metrics["macro_f1"], test_metrics["micro_f1"],
        test_metrics["auc_pr_macro"], test_metrics["roc_auc_macro"],
    )

    plot_training_curves(
        resolve(cfg, "results_dir") / "history" / f"{run_name}.json",
        plots_dir / f"{run_name}_curves.png", run_name,
    )
    plot_per_tag_f1(test_metrics, plots_dir / f"{run_name}_per_tag.png", run_name)

    if task == "3" and "fused" in test_out:
        plot_tsne(
            test_out["fused"], test_out["targets"], label_tags,
            plots_dir / f"{run_name}_tsne.png", run_name,
            max_samples=cfg["eval"]["tsne_samples"], perplexity=cfg["eval"]["tsne_perplexity"],
            seed=cfg["seed"],
        )

    payload = {
        "run_name": run_name,
        "task": task,
        "best_epoch": meta.get("epoch"),
        "thresholds": (thresholds.tolist() if isinstance(thresholds, np.ndarray) else thresholds),
        "val": {k: v for k, v in val_metrics.items() if k != "per_tag"},
        "test": test_metrics,
    }
    if cfg["model"]["emotion"]["enabled"] and "emotion_true" in test_out:
        payload["test"]["emotion"] = emotion_metrics(test_out["emotion_true"], test_out["emotion_pred"])
    return payload


def evaluate_contrastive(cfg, run_name: str, device, logger) -> dict:
    label_tags, _ = load_vocabulary(resolve(cfg, "processed_dir"))
    test_loader, extras = build_eval_loader(cfg, "4", "test")
    node_dim = int(extras["dataset"].data.x.shape[1])

    model, meta = load_trained_model(cfg, "4", run_name, device, len(label_tags), node_dim)
    out = collect_predictions(model, test_loader, "4", device)

    ks = tuple(cfg["contrastive"]["recall_at_k"])
    metrics = retrieval_metrics(
        out["graph_embeddings"].to(device), out["text_embeddings"].to(device),
        ks=ks, group_ids=out.get("groups"),
    )
    logger.info(
        "%s TEST retrieval: caption->audio R@1 %.4f R@5 %.4f R@10 %.4f",
        run_name, metrics["caption_to_audio_R@1"], metrics["caption_to_audio_R@5"],
        metrics["caption_to_audio_R@10"],
    )

    examples = qualitative_retrieval(
        out["graph_embeddings"], out["text_embeddings"],
        captions=extras["texts"], clip_ids=[str(c) for c in extras["clip_ids"]],
        num_queries=10, top_k=3, seed=cfg["seed"],
    )
    retrieval_dir = resolve(cfg, "retrieval_dir")
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    (retrieval_dir / f"{run_name}_examples.json").write_text(
        json.dumps(examples, indent=2), encoding="utf-8"
    )

    plot_training_curves(
        resolve(cfg, "results_dir") / "history" / f"{run_name}.json",
        resolve(cfg, "plots_dir") / f"{run_name}_curves.png", run_name,
    )
    return {"run_name": run_name, "task": "4", "best_epoch": meta.get("epoch"), "test": metrics,
            "num_qualitative_examples": len(examples)}


def evaluate_baselines(cfg, logger) -> dict:
    """Baseline B1: random and prior-frequency tag predictors on the test split."""
    processed = resolve(cfg, "processed_dir")
    label_tags, _ = load_vocabulary(processed)
    test_labels = np.load(processed / "labels_test.npy")

    results = {}
    for strategy in ("random", "prior"):
        metrics = random_baseline_metrics(test_labels, seed=cfg["seed"], strategy=strategy)
        results[strategy] = metrics
        logger.info("baseline %-6s: macro-F1 %.4f | AUC-PR %.4f", strategy,
                    metrics["macro_f1"], metrics["auc_pr_macro"])
    return {"run_name": "baseline_random", "task": "B1", "test": results["prior"], "variants": results,
            "num_tags": len(label_tags)}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["1", "2", "3", "4", "cnn"], default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--baselines", action="store_true", help="evaluate trivial baselines only")
    parser.add_argument("--comparison-plot", action="store_true", help="redraw the cross-model comparison chart")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    set_seed(cfg["seed"])
    results_dir = resolve(cfg, "results_dir")
    logger = setup_logging(results_dir / "logs" / "evaluate.log")
    device = get_device(cfg["device"])

    if args.baselines:
        payload = evaluate_baselines(cfg, logger)
        update_metrics_json(results_dir, "baseline_random", payload)
    elif args.task:
        run_name = args.run_name or f"task{args.task}"
        if args.task == "4":
            payload = evaluate_contrastive(cfg, run_name, device, logger)
        else:
            payload = evaluate_tagging(cfg, args.task, run_name, device, logger)
        update_metrics_json(results_dir, run_name, payload)
        logger.info("Wrote results/metrics.json section %r", run_name)

    plot_model_comparison(results_dir / "metrics.json", resolve(cfg, "plots_dir") / "model_comparison.png")


if __name__ == "__main__":
    main()
