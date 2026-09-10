"""Non-neural and shallow baselines required by the project spec.

  * **B1** random / prior-frequency tag predictor  -> ``src.metrics.random_baseline_metrics``
  * **B2** CNN on mel-spectrogram (no graph, no text) -> ``src.gnn_model.MelCNNBaseline``,
    trained via ``python -m src.train --task cnn``
  * **B3** BERT-only                                -> Task 1
  * **B4** PCA + MLP on hand-crafted audio features -> implemented here

Run B4 with:

    python -m src.baselines --pca-mlp
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import load_vocabulary
from .metrics import search_thresholds, tagging_metrics
from .utils import load_config, resolve, set_seed, setup_logging, update_metrics_json


def handcrafted_features(memmap_path, shape) -> np.ndarray:
    """Summarise each clip's log-mel patch into time-pooled statistics.

    For every mel band: mean, standard deviation, and the mean of the first-order
    temporal difference (a cheap proxy for onset density / rhythmic activity).
    This yields 3 x n_mels features per clip with no learned parameters.
    """
    mel = np.memmap(memmap_path, dtype=np.float16, mode="r", shape=tuple(shape))
    chunks = []
    # Chunked so the full corpus never has to sit in RAM at once.
    for start in range(0, shape[0], 512):
        block = np.asarray(mel[start : start + 512], dtype=np.float32)
        delta = np.abs(np.diff(block, axis=2))
        chunks.append(
            np.concatenate(
                [block.mean(axis=2), block.std(axis=2), delta.mean(axis=2)], axis=1
            )
        )
    return np.concatenate(chunks, axis=0)


def run_pca_mlp(cfg, logger) -> dict:
    processed = resolve(cfg, "processed_dir")
    label_tags, _ = load_vocabulary(processed)

    features, labels = {}, {}
    for split in ("train", "val", "test"):
        meta = json.loads((processed / f"mel_{split}_meta.json").read_text(encoding="utf-8"))
        features[split] = handcrafted_features(processed / f"mel_{split}.f16", meta["shape"])
        labels[split] = np.load(processed / f"labels_{split}.npy")
        logger.info("%s features: %s", split, features[split].shape)

    # 128 components retains the great majority of mel-statistic variance while
    # keeping the MLP small enough to fit in seconds. PCA cannot produce more
    # components than it has samples or features, which matters on --limit runs.
    n_components = min(128, *features["train"].shape)
    if n_components < 128:
        logger.warning("Reducing PCA components to %d for this data size", n_components)

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=cfg["seed"])),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(512, 256),
                    activation="relu",
                    alpha=1e-4,
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=60,
                    early_stopping=True,
                    n_iter_no_change=6,
                    random_state=cfg["seed"],
                    verbose=False,
                ),
            ),
        ]
    )

    logger.info("Fitting PCA + MLP on %d training clips...", features["train"].shape[0])
    pipeline.fit(features["train"], labels["train"])

    def predict_proba(split: str) -> np.ndarray:
        raw = pipeline.predict_proba(features[split])
        # MLPClassifier returns a list of per-label arrays when some label is
        # constant in the training split; normalise both shapes to (N, K).
        if isinstance(raw, list):
            return np.column_stack([column[:, -1] if column.ndim == 2 else column for column in raw])
        return raw

    val_scores, test_scores = predict_proba("val"), predict_proba("test")
    thresholds = search_thresholds(labels["val"], val_scores)
    metrics = tagging_metrics(labels["test"], test_scores, thresholds, label_tags)
    logger.info(
        "PCA+MLP TEST: macro-F1 %.4f | micro-F1 %.4f | AUC-PR %.4f",
        metrics["macro_f1"], metrics["micro_f1"], metrics["auc_pr_macro"],
    )
    return {"run_name": "baseline_pca_mlp", "task": "B4", "test": metrics,
            "val": tagging_metrics(labels["val"], val_scores, thresholds)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pca-mlp", action="store_true")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    set_seed(cfg["seed"])
    results_dir = resolve(cfg, "results_dir")
    logger = setup_logging(results_dir / "logs" / "baselines.log")

    if args.pca_mlp:
        payload = run_pca_mlp(cfg, logger)
        update_metrics_json(results_dir, "baseline_pca_mlp", payload)
        logger.info("Wrote results/metrics.json section 'baseline_pca_mlp'")
    else:
        parser.error("nothing to do: pass --pca-mlp")


if __name__ == "__main__":
    main()
