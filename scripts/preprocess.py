"""Build music structure graphs and mel patches from raw MagnaTagATune audio.

Usage
-----
    python scripts/preprocess.py                 # full corpus
    python scripts/preprocess.py --limit 500     # quick smoke run
    python scripts/preprocess.py --workers 8

Outputs (under data/processed/):
    tag_vocabulary.json      label + auxiliary tag names
    manifest.csv             one row per usable clip
    graphs_{split}.pt        pre-collated PyG graphs for the split
    mel_{split}.f16          float16 memmap of (N, n_mels, 256) log-mel patches
    mel_{split}_meta.json    memmap shape
    labels_{split}.npy       (N, K) binary tag matrix
    examples/                20 individual graphs exported for the submission
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio_features import FeatureExtractor, load_audio  # noqa: E402
from src.dataset import build_manifest, save_vocabulary  # noqa: E402
from src.graph_builder import GraphConfig, build_track_graph, graph_summary  # noqa: E402
from src.utils import PROJECT_ROOT, load_config, resolve, set_seed, setup_logging  # noqa: E402

MEL_FRAMES = 256  # fixed time resolution for the CNN baseline patches

# Per-process globals, initialised once per worker (see _init_worker).
_EXTRACTOR: FeatureExtractor | None = None
_AUDIO_CFG: dict | None = None
_GRAPH_CFG: GraphConfig | None = None
_AUDIO_ROOT: Path | None = None


def _init_worker(audio_cfg: dict, graph_cfg_dict: dict, audio_root: str) -> None:
    global _EXTRACTOR, _AUDIO_CFG, _GRAPH_CFG, _AUDIO_ROOT
    _AUDIO_CFG = audio_cfg
    _GRAPH_CFG = GraphConfig(**graph_cfg_dict)
    _AUDIO_ROOT = Path(audio_root)
    torch.set_num_threads(1)  # workers must not each spawn a thread pool
    _EXTRACTOR = FeatureExtractor(
        sample_rate=audio_cfg["sample_rate"],
        n_fft=audio_cfg["n_fft"],
        hop_length=audio_cfg["hop_length"],
        n_mels=audio_cfg["n_mels"],
        n_chroma=audio_cfg["n_chroma"],
        fmin=audio_cfg["fmin"],
        fmax=audio_cfg["fmax"],
        device="cpu",
    )


def _process_clip(task: tuple[int, str]) -> tuple[int, object, np.ndarray | None, str | None]:
    """Worker entry point: returns (clip_id, Data|None, mel_patch|None, error)."""
    clip_id, mp3_path = task
    path = _AUDIO_ROOT / mp3_path
    try:
        waveform = load_audio(path, _AUDIO_CFG["sample_rate"])
        duration = waveform.shape[0] / _AUDIO_CFG["sample_rate"]
        if duration < _AUDIO_CFG.get("min_clip_seconds", 0.0):
            return clip_id, None, None, f"too short ({duration:.1f}s)"

        graph = build_track_graph(waveform, _EXTRACTOR, _AUDIO_CFG, _GRAPH_CFG)

        # Fixed-size mel patch for the CNN baseline: adaptive-pool the time axis
        # so every clip yields exactly MEL_FRAMES columns.
        log_mel = _EXTRACTOR.log_mel(waveform)
        patch = F.adaptive_avg_pool1d(log_mel.unsqueeze(0), MEL_FRAMES).squeeze(0)
        return clip_id, graph, patch.numpy().astype(np.float16), None
    except Exception as exc:  # a handful of MTAT mp3s are zero-byte or truncated
        return clip_id, None, None, f"{type(exc).__name__}: {exc}"


def extract_audio_archive(raw_dir: Path, audio_root: Path, logger) -> None:
    """Unzip mp3.zip (single archive) or mp3.zip.001..003 (split archive)."""
    if audio_root.exists() and any(audio_root.glob("*/*.mp3")):
        logger.info("Audio already extracted at %s", audio_root)
        return

    audio_root.mkdir(parents=True, exist_ok=True)
    single = raw_dir / "mp3.zip"
    parts = sorted(raw_dir.glob("mp3.zip.0*"))

    if single.exists() and single.stat().st_size > 2_000_000_000:
        archive = single
    elif parts:
        # The original mirror ships a split zip; concatenate before extracting.
        archive = raw_dir / "mp3_joined.zip"
        if not archive.exists():
            logger.info("Joining %d split archive parts...", len(parts))
            with open(archive, "wb") as out:
                for part in parts:
                    out.write(part.read_bytes())
    else:
        raise FileNotFoundError(f"No usable mp3 archive in {raw_dir}")

    logger.info("Extracting %s -> %s (this takes a few minutes)", archive.name, audio_root)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(audio_root)
    logger.info("Extraction complete: %d mp3 files", sum(1 for _ in audio_root.glob("*/*.mp3")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--limit", type=int, default=None, help="process only the first N clips (smoke test)")
    # Leave a couple of cores free so the machine stays usable during the run.
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    set_seed(cfg["seed"])
    logger = setup_logging(resolve(cfg, "results_dir") / "preprocess.log")

    raw_dir = resolve(cfg, "raw_dir")
    processed_dir = resolve(cfg, "processed_dir")
    processed_dir.mkdir(parents=True, exist_ok=True)
    audio_root = raw_dir / "audio"

    if not args.skip_extract:
        extract_audio_archive(raw_dir, audio_root, logger)

    logger.info("Building manifest from annotations...")
    manifest, label_tags, auxiliary_tags = build_manifest(raw_dir, cfg)
    save_vocabulary(processed_dir, label_tags, auxiliary_tags)
    logger.info(
        "Manifest: %d clips | %d label tags | %d auxiliary tags",
        len(manifest), len(label_tags), len(auxiliary_tags),
    )
    logger.info("Split sizes: %s", manifest["split"].value_counts().to_dict())

    # Keep only clips whose audio file is actually present on disk.
    present = manifest["mp3_path"].map(lambda p: (audio_root / p).exists())
    missing = int((~present).sum())
    if missing:
        logger.warning("%d clips have no audio file on disk and are dropped", missing)
    manifest = manifest[present].reset_index(drop=True)

    if args.limit:
        manifest = manifest.groupby("split", group_keys=False).head(max(1, args.limit // 3)).reset_index(drop=True)
        logger.info("--limit active: %d clips retained", len(manifest))

    audio_cfg = dict(cfg["audio"])
    audio_cfg["min_clip_seconds"] = cfg["data"]["min_clip_seconds"]
    graph_cfg_dict = dict(cfg["graph"])

    n_mels = cfg["audio"]["n_mels"]
    examples_dir = processed_dir / "examples"
    examples_dir.mkdir(exist_ok=True)

    all_failures: list[tuple[int, str]] = []
    kept_clip_ids: list[int] = []
    example_summaries: list[dict] = []
    node_dim: int | None = None

    # One split at a time. Holding all ~21k graphs *and* their mel patches in
    # memory at once peaks around 2 GB, which is more than this machine has
    # free; per-split processing with the mel written straight into its memmap
    # keeps the resident set to a few hundred MB.
    for split in ("train", "val", "test"):
        rows = manifest[manifest["split"] == split].reset_index(drop=True)
        if rows.empty:
            logger.warning("Split %s is empty; skipping", split)
            continue

        tasks = list(zip(rows["clip_id"].tolist(), rows["mp3_path"].tolist()))
        logger.info("Split %s: processing %d clips with %d workers...", split, len(tasks), args.workers)

        mel_path = processed_dir / f"mel_{split}.f16"
        mel_memmap = np.memmap(
            mel_path, dtype=np.float16, mode="w+", shape=(len(tasks), n_mels, MEL_FRAMES)
        )

        graphs, ordered_ids, written = [], [], 0
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(audio_cfg, graph_cfg_dict, str(audio_root)),
        ) as pool:
            for clip_id, graph, mel, error in tqdm(
                pool.map(_process_clip, tasks, chunksize=16), total=len(tasks), desc=f"{split} clips"
            ):
                if error is not None:
                    all_failures.append((clip_id, error))
                    continue
                mel_memmap[written] = mel
                written += 1
                graphs.append(graph)
                ordered_ids.append(int(clip_id))

        # Reindex the manifest rows to the clips that actually succeeded, in the
        # same order the mel rows were written.
        rows = rows.set_index("clip_id").loc[ordered_ids].reset_index()
        labels = rows[label_tags].to_numpy(dtype=np.float32)
        np.save(processed_dir / f"labels_{split}.npy", labels)

        # Integer code per source track, so the contrastive loss can mask
        # same-track clips as non-negatives after PyG batching.
        group_codes = rows["track_group"].astype("category").cat.codes.to_numpy()
        for i, graph in enumerate(graphs):
            graph.y = torch.from_numpy(labels[i]).unsqueeze(0)
            graph.clip_id = ordered_ids[i]
            graph.track_group = torch.tensor([int(group_codes[i])], dtype=torch.long)

        # Trim the memmap to the rows actually written (failures leave gaps).
        mel_memmap.flush()
        del mel_memmap
        if written != len(tasks):
            full = np.memmap(mel_path, dtype=np.float16, mode="r", shape=(len(tasks), n_mels, MEL_FRAMES))
            trimmed = np.array(full[:written])
            del full
            trimmed.tofile(mel_path)
            logger.info("  trimmed mel memmap from %d to %d rows", len(tasks), written)

        (processed_dir / f"mel_{split}_meta.json").write_text(
            json.dumps({"shape": [written, n_mels, MEL_FRAMES], "dtype": "float16"}), encoding="utf-8"
        )

        data, slices = InMemoryDataset.collate(graphs)
        torch.save(
            {
                "data": data,
                "slices": slices,
                "clip_ids": rows["clip_id"].tolist(),
                "texts": rows["context_text"].tolist(),
                "track_groups": rows["track_group"].tolist(),
                "label_tags": label_tags,
            },
            processed_dir / f"graphs_{split}.pt",
        )
        logger.info("Saved %s: %d graphs, mel %s", split, len(graphs), (written, n_mels, MEL_FRAMES))

        # Export the 20 example graphs (submission requirement) from the train split.
        if split == "train" and not example_summaries:
            for graph in graphs[:20]:
                clip_id = int(graph.clip_id)
                torch.save(graph, examples_dir / f"graph_clip{clip_id}.pt")
                summary = graph_summary(graph)
                summary["clip_id"] = clip_id
                example_summaries.append(summary)
            (examples_dir / "example_graphs.json").write_text(
                json.dumps(example_summaries, indent=2), encoding="utf-8"
            )
            logger.info("Exported %d example graphs to %s", len(example_summaries), examples_dir)

        node_dim = int(graphs[0].x.shape[1])
        kept_clip_ids.extend(ordered_ids)
        del graphs, data, slices

    if all_failures:
        pd.DataFrame(all_failures, columns=["clip_id", "error"]).to_csv(
            processed_dir / "failed_clips.csv", index=False
        )
        logger.warning("%d clips failed; see failed_clips.csv", len(all_failures))
        for clip_id, error in all_failures[:5]:
            logger.warning("  clip %s: %s", clip_id, error)

    manifest = manifest[manifest["clip_id"].isin(set(kept_clip_ids))].reset_index(drop=True)
    manifest.to_csv(processed_dir / "manifest.csv", index=False)
    logger.info("Done. %d clips kept | node feature dim = %s", len(manifest), node_dim)


if __name__ == "__main__":
    main()
