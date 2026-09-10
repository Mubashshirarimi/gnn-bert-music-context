"""Shared helpers: config loading, seeding, logging, checkpoint I/O."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class Config(dict):
    """dict with attribute access, so cfg.train.epochs works as well as cfg["train"]["epochs"]."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load config.yaml and apply ``section.key=value`` override strings."""
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        cfg = Config(yaml.safe_load(fh))

    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Malformed override {override!r}; expected section.key=value")
        dotted, raw = override.split("=", 1)
        node: dict = cfg
        keys = dotted.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = yaml.safe_load(raw)  # parses ints/floats/bools/lists

    return cfg


def resolve(cfg: Config, key: str) -> Path:
    """Resolve a path from cfg.paths into an absolute path under the project root."""
    return PROJECT_ROOT / cfg["paths"][key]


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        # Slower, but makes GPU reductions reproducible run to run.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device(preference: str = "cuda") -> torch.device:
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "cuda":
        logging.warning("CUDA requested but unavailable; falling back to CPU.")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("gnn-bert")


# --------------------------------------------------------------------------- #
# Checkpoints and results
# --------------------------------------------------------------------------- #
def save_checkpoint(path: Path, model: torch.nn.Module, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **(extra or {})}, path)


def load_checkpoint(path: Path, model: torch.nn.Module, device: torch.device) -> dict:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    return {k: v for k, v in payload.items() if k != "state_dict"}


def update_metrics_json(results_dir: Path, section: str, payload: dict) -> Path:
    """Merge a result block into results/metrics.json without clobbering other tasks."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "metrics.json"
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logging.warning("metrics.json was unreadable; starting a fresh one.")
    data[section] = payload
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
