"""Unified training entry point for all four tasks.

    python -m src.train --task 1                      # BERT tag classifier
    python -m src.train --task 2                      # GNN on structure graphs
    python -m src.train --task 3                      # GNN-BERT fusion
    python -m src.train --task 3 --set model.fusion.mode=concat --run-name concat
    python -m src.train --task 4                      # contrastive dual encoder
    python -m src.train --task cnn                    # baseline B2 (mel CNN)

Checkpoints go to results/checkpoints/<run_name>.pt and per-epoch history to
results/history/<run_name>.json. Final metric tables are produced by
``python -m src.evaluate``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader

from .bert_encoder import BertTagClassifier, build_tokenizer
from .contrastive import DualEncoder, build_duplicate_mask, info_nce_loss, retrieval_metrics
from .dataset import (
    MelSpectrogramDataset,
    MusicGraphDataset,
    TextTagDataset,
    compute_pos_weight,
    load_vocabulary,
    rebuild_texts,
)
from .fusion_model import GNNBertFusion, MultiTaskLoss
from .gnn_model import GNNTagClassifier, MelCNNBaseline
from .metrics import tagging_metrics
from .utils import (
    count_parameters,
    get_device,
    load_config,
    resolve,
    save_checkpoint,
    set_seed,
    setup_logging,
)

TASKS = ("1", "2", "3", "4", "cnn")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_graph_split(cfg, split: str, tokenizer=None) -> MusicGraphDataset:
    path = resolve(cfg, "processed_dir") / f"graphs_{split}.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run scripts/preprocess.py first")
    return MusicGraphDataset(
        path,
        tokenizer=tokenizer,
        max_length=cfg["text"]["max_length"],
        attach_text=tokenizer is not None,
        cfg=cfg,
    )


def load_mel_split(cfg, split: str) -> MelSpectrogramDataset:
    processed = resolve(cfg, "processed_dir")
    meta = json.loads((processed / f"mel_{split}_meta.json").read_text(encoding="utf-8"))
    labels = np.load(processed / f"labels_{split}.npy")
    return MelSpectrogramDataset(processed / f"mel_{split}.f16", labels, tuple(meta["shape"]))


def build_loaders(cfg, task: str, logger):
    """Returns (train_loader, val_loader, num_tags, node_dim, extras)."""
    batch_size = cfg["contrastive"]["batch_size"] if task == "4" else cfg["train"]["batch_size"]
    # Windows spawns worker processes, which would each copy the in-memory
    # graph store; with ~4 GB free RAM that is far more costly than the
    # collation it saves, so graph tasks stay single-process.
    workers = 0 if task in ("1", "2", "3", "4") else cfg["train"]["num_workers"]
    label_tags, _ = load_vocabulary(resolve(cfg, "processed_dir"))
    num_tags = len(label_tags)
    extras: dict = {"label_tags": label_tags}

    if task == "1":
        tokenizer = build_tokenizer(cfg["text"]["model_name"])
        processed = resolve(cfg, "processed_dir")
        loaders = []
        for split in ("train", "val"):
            payload = torch.load(processed / f"graphs_{split}.pt", map_location="cpu", weights_only=False)
            labels = np.load(processed / f"labels_{split}.npy")
            texts = rebuild_texts(cfg, processed, payload["clip_ids"])
            dataset = TextTagDataset(texts, labels, tokenizer, cfg["text"]["max_length"])
            loaders.append(
                DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=workers)
            )
            if split == "train":
                extras["pos_weight"] = compute_pos_weight(labels)
        return loaders[0], loaders[1], num_tags, None, extras

    if task == "cnn":
        train_ds, val_ds = load_mel_split(cfg, "train"), load_mel_split(cfg, "val")
        extras["pos_weight"] = compute_pos_weight(train_ds.labels.numpy())
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers),
            num_tags,
            None,
            extras,
        )

    # Graph-based tasks 2, 3 and 4.
    needs_text = task in ("3", "4") and cfg["model"]["fusion"]["mode"] != "gnn_only"
    tokenizer = build_tokenizer(cfg["text"]["model_name"]) if needs_text else None
    train_ds = load_graph_split(cfg, "train", tokenizer)
    val_ds = load_graph_split(cfg, "val", tokenizer)
    node_dim = int(train_ds.data.x.shape[1])
    extras["pos_weight"] = compute_pos_weight(train_ds.labels)
    extras["train_dataset"], extras["val_dataset"] = train_ds, val_ds
    logger.info("Graphs: %d train / %d val | node feature dim %d", len(train_ds), len(val_ds), node_dim)
    return (
        GeoDataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers),
        GeoDataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers),
        num_tags,
        node_dim,
        extras,
    )


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def build_model(cfg, task: str, num_tags: int, node_dim: int | None) -> nn.Module:
    if task == "1":
        return BertTagClassifier(
            num_tags=num_tags,
            model_name=cfg["text"]["model_name"],
            freeze=cfg["text"]["freeze_bert"],
            unfreeze_last_n=cfg["text"]["unfreeze_last_n"],
        )
    if task == "2":
        gnn = cfg["model"]["gnn"]
        return GNNTagClassifier(
            in_dim=node_dim,
            num_tags=num_tags,
            hidden_dim=gnn["hidden_dim"],
            num_layers=gnn["num_layers"],
            conv=gnn["conv"],
            dropout=gnn["dropout"],
            heads=gnn["heads"],
            readout=gnn["readout"],
        )
    if task == "3":
        return GNNBertFusion(
            node_in_dim=node_dim,
            num_tags=num_tags,
            gnn_cfg=cfg["model"]["gnn"],
            fusion_cfg=cfg["model"]["fusion"],
            text_cfg=cfg["text"],
            emotion_enabled=cfg["model"]["emotion"]["enabled"],
        )
    if task == "4":
        return DualEncoder(
            node_in_dim=node_dim,
            gnn_cfg=cfg["model"]["gnn"],
            text_cfg=cfg["text"],
            embed_dim=cfg["contrastive"]["embed_dim"],
            temperature=cfg["contrastive"]["temperature"],
        )
    if task == "cnn":
        return MelCNNBaseline(num_tags=num_tags, n_mels=cfg["audio"]["n_mels"])
    raise ValueError(f"Unknown task {task!r}")


def build_optimizer(cfg, model: nn.Module, task: str):
    """Two parameter groups: pretrained BERT weights get a much smaller LR."""
    base_lr = cfg["contrastive"]["lr"] if task == "4" else cfg["train"]["lr"]
    bert_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (bert_params if ".bert." in name or name.startswith("bert.") else other_params).append(param)

    groups = [{"params": other_params, "lr": base_lr}]
    if bert_params:
        groups.append({"params": bert_params, "lr": cfg["train"]["bert_lr"]})
    return torch.optim.AdamW(groups, weight_decay=cfg["train"]["weight_decay"])


def build_scheduler(optimizer, cfg, steps_per_epoch: int, epochs: int):
    """Linear warmup then cosine decay, applied per optimiser step."""
    total = max(1, steps_per_epoch * epochs)
    warmup = max(1, int(total * cfg["train"]["warmup_ratio"]))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------------- #
# Epoch loops
# --------------------------------------------------------------------------- #
def _move_batch(batch, device):
    """Move a PyG Batch, a dict batch, or a (tensor, tensor) tuple onto the device."""
    if isinstance(batch, (tuple, list)):
        return tuple(item.to(device) for item in batch)
    if isinstance(batch, dict):
        return {key: value.to(device) for key, value in batch.items()}
    return batch.to(device)


def run_epoch(model, loader, device, task, criterion, optimizer=None, scheduler=None, scaler=None, cfg=None):
    """One pass over ``loader``. Training when ``optimizer`` is given."""
    training = optimizer is not None
    model.train(training)
    autocast_enabled = bool(cfg["train"]["amp"]) and device.type == "cuda"

    total_loss, n_batches = 0.0, 0
    scores, targets, groups = [], [], []

    for batch in loader:
        batch = _move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
            if task == "1":
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss = criterion(logits, batch["labels"])
                y = batch["labels"]
            elif task == "cnn":
                mel, y = batch
                logits = model(mel)
                loss = criterion(logits, y)
            elif task == "2":
                logits = model(batch)
                y = batch.y.view(logits.shape)
                loss = criterion(logits, y)
            elif task == "3":
                outputs = model(batch)
                logits = outputs["tag_logits"]
                y = batch.y.view(logits.shape)
                loss, _ = criterion(outputs, {"tags": y})
            else:  # task 4
                graph_emb, text_emb = model(batch)
                # Clips cut from the same source track share a context string;
                # excluding them as negatives avoids punishing correct matches.
                mask = None
                if hasattr(batch, "track_group"):
                    mask = build_duplicate_mask(batch.track_group.cpu().numpy()).to(device)
                loss, _ = info_nce_loss(graph_emb, text_emb, model.temperature, mask)
                logits = y = None

        if training:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += float(loss.detach())
        n_batches += 1
        if logits is not None and not training:
            scores.append(torch.sigmoid(logits.float()).detach().cpu())
            targets.append(y.float().detach().cpu())
        elif task == "4" and not training:
            scores.append(graph_emb.float().detach().cpu())
            targets.append(text_emb.float().detach().cpu())
            if hasattr(batch, "track_group"):
                groups.extend(batch.track_group.cpu().numpy().tolist())

    output: dict = {"loss": total_loss / max(1, n_batches)}
    if scores and not training:
        output["scores"] = torch.cat(scores)
        output["targets"] = torch.cat(targets)
        if groups:
            output["groups"] = np.array(groups)
    return output


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    set_seed(cfg["seed"])
    task = args.task
    run_name = args.run_name or f"task{task}"

    results_dir = resolve(cfg, "results_dir")
    logger = setup_logging(results_dir / "logs" / f"{run_name}.log")
    device = get_device(cfg["device"])
    logger.info("Run %s | task %s | device %s", run_name, task, device)

    train_loader, val_loader, num_tags, node_dim, extras = build_loaders(cfg, task, logger)
    model = build_model(cfg, task, num_tags, node_dim).to(device)
    total, trainable = count_parameters(model)
    logger.info("Model parameters: %.2fM total, %.2fM trainable", total / 1e6, trainable / 1e6)

    if task == "4":
        criterion = None
    elif task == "3":
        criterion = MultiTaskLoss(
            emotion_weight=cfg["model"]["emotion"]["loss_weight"],
            pos_weight=extras["pos_weight"].to(device),
        )
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=extras["pos_weight"].to(device))

    epochs = args.epochs or (cfg["contrastive"]["epochs"] if task == "4" else cfg["train"]["epochs"])
    optimizer = build_optimizer(cfg, model, task)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader), epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["train"]["amp"]) and device.type == "cuda")

    monitor = "val_recall" if task == "4" else cfg["train"]["monitor"]
    best_score, best_epoch, patience = -math.inf, -1, cfg["train"]["early_stopping_patience"]
    history: list[dict] = []
    checkpoint_path = results_dir / "checkpoints" / f"{run_name}.pt"

    for epoch in range(1, epochs + 1):
        started = time.time()
        train_out = run_epoch(model, train_loader, device, task, criterion, optimizer, scheduler, scaler, cfg)
        val_out = run_epoch(model, val_loader, device, task, criterion, cfg=cfg)

        record = {
            "epoch": epoch,
            "train_loss": train_out["loss"],
            "val_loss": val_out["loss"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - started, 1),
        }

        if task == "4":
            metrics = retrieval_metrics(
                val_out["scores"].to(device),
                val_out["targets"].to(device),
                ks=tuple(cfg["contrastive"]["recall_at_k"]),
                group_ids=val_out.get("groups"),
            )
            record.update({f"val_{k}": v for k, v in metrics.items()})
            score = metrics["caption_to_audio_R@5"]
            record["val_recall"] = score
        else:
            metrics = tagging_metrics(val_out["targets"], val_out["scores"], thresholds=0.5)
            record.update({f"val_{k}": v for k, v in metrics.items() if k != "per_tag"})
            score = record.get(monitor, record["val_macro_f1"])

        history.append(record)
        logger.info(
            "epoch %2d | train %.4f | val %.4f | %s %.4f | %.0fs",
            epoch, record["train_loss"], record["val_loss"], monitor, score, record["seconds"],
        )

        if score > best_score:
            best_score, best_epoch = score, epoch
            save_checkpoint(
                checkpoint_path,
                model,
                extra={
                    "epoch": epoch,
                    "score": score,
                    "monitor": monitor,
                    "config": dict(cfg),
                    "task": task,
                    "num_tags": num_tags,
                    "node_dim": node_dim,
                    "run_name": run_name,
                },
            )
        elif epoch - best_epoch >= patience:
            logger.info("Early stopping: no improvement for %d epochs", patience)
            break

    history_path = results_dir / "history" / f"{run_name}.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    logger.info("Best %s = %.4f at epoch %d -> %s", monitor, best_score, best_epoch, checkpoint_path)


if __name__ == "__main__":
    main()
