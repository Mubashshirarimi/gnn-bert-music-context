"""MagnaTagATune loading: tag vocabulary, splits, text context, and torch datasets.

Design notes
------------
* **Labels** are rebuilt from ``annotations_final.csv`` with the standard
  synonym merge, then reduced to the top-K most frequent tags. The shipped
  ``*_gt_mtt.tsv`` files are used only for split membership, so the label
  ordering is defined in one place and is verifiable (see ``scripts/verify_labels.py``).
* **Splits** follow the Won et al. convention that MTAT's ``0``-``f`` directories
  partition the corpus by album folder, which is what prevents artist leakage.
* **Memory**: this machine has limited free RAM, so mel spectrograms live in an
  on-disk float16 memmap and graphs are stored in one pre-collated tensor per
  split rather than as 25k individual files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset

from .bert_encoder import build_context_string

# Standard MagnaTagATune synonym merge (Won et al., "Evaluation of CNN-based
# Automatic Music Tagging Models", 2020). Each group collapses to its first entry.
TAG_SYNONYMS: list[list[str]] = [
    ["beat", "beats"],
    ["chant", "chanting"],
    ["choir", "choral"],
    ["classical", "clasical", "classic"],
    ["drum", "drums"],
    ["electro", "electronic", "electronica", "electric"],
    ["fast", "fast beat", "quick"],
    ["female", "female singer", "female singing", "female vocals", "female voice", "woman", "woman singing", "women"],
    ["flute", "flutes"],
    ["guitar", "guitars"],
    ["hard", "hard rock"],
    ["harpsichord", "harpsicord"],
    ["heavy", "heavy metal", "metal"],
    ["horn", "horns"],
    ["india", "indian"],
    ["jazz", "jazzy"],
    ["male", "male singer", "male vocal", "male vocals", "male voice", "man", "man singing", "men"],
    ["no beat", "no drums", "no percussion", "no singer", "no singing", "no vocal", "no vocals", "no voice", "no voices", "instrumental"],
    ["opera", "operatic"],
    ["orchestra", "orchestral"],
    ["quiet", "silence"],
    ["singer", "singing"],
    ["space", "spacey"],
    ["string", "strings"],
    ["synth", "synthesizer"],
    ["violin", "violins"],
    ["vocal", "vocals", "voice", "voices"],
    ["strange", "weird"],
]


# --------------------------------------------------------------------------- #
# Annotation loading
# --------------------------------------------------------------------------- #
def load_annotations(raw_dir: Path) -> pd.DataFrame:
    """Read annotations_final.csv and collapse synonymous tag columns."""
    path = raw_dir / "annotations_final.csv"
    frame = pd.read_csv(path, sep="\t")
    frame.columns = [c.strip().strip('"') for c in frame.columns]

    tag_columns = [c for c in frame.columns if c not in ("clip_id", "mp3_path")]
    for group in TAG_SYNONYMS:
        present = [tag for tag in group if tag in frame.columns]
        if len(present) < 2:
            continue
        canonical, others = present[0], present[1:]
        # Logical OR across synonyms, then drop the redundant columns.
        frame[canonical] = frame[present].max(axis=1)
        frame = frame.drop(columns=others)
        tag_columns = [c for c in tag_columns if c not in others]

    frame["clip_id"] = frame["clip_id"].astype(int)
    return frame


def select_top_tags(annotations: pd.DataFrame, top_k: int = 50) -> tuple[list[str], list[str]]:
    """Split the merged vocabulary into (top-K label tags, remaining auxiliary tags).

    Ordered by descending corpus frequency, ties broken alphabetically so the
    vocabulary is deterministic across machines.
    """
    tag_columns = [c for c in annotations.columns if c not in ("clip_id", "mp3_path")]
    counts = annotations[tag_columns].sum().sort_values(ascending=False)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    names = [name for name, _ in ordered]
    return names[:top_k], names[top_k:]


def load_clip_metadata(raw_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(raw_dir / "clip_info_final.csv", sep="\t")
    frame.columns = [c.strip().strip('"') for c in frame.columns]
    frame["clip_id"] = frame["clip_id"].astype(int)
    return frame[["clip_id", "title", "artist", "album", "mp3_path"]]


def load_official_splits(raw_dir: Path) -> dict[str, set[int]]:
    """Clip ids per split from the shipped *_gt_mtt.tsv files."""
    splits: dict[str, set[int]] = {}
    for split, filename in (("train", "train_gt_mtt.tsv"), ("val", "val_gt_mtt.tsv"), ("test", "test_gt_mtt.tsv")):
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing split file {path}")
        ids = pd.read_csv(path, sep="\t", header=None, usecols=[0]).iloc[:, 0]
        splits[split] = set(int(v) for v in ids)
    return splits


def split_by_folder(mp3_paths: pd.Series, cfg) -> pd.Series:
    """Fallback split assignment from the leading hex directory of mp3_path."""
    folder = mp3_paths.astype(str).str.split("/").str[0]
    mapping = {}
    for name, key in (("train", "train_folders"), ("val", "val_folders"), ("test", "test_folders")):
        for value in cfg["data"][key]:
            mapping[value] = name
    return folder.map(mapping)


def build_manifest(raw_dir: Path, cfg) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Join annotations + metadata + splits into one table, one row per clip.

    Returns (manifest, label_tags, auxiliary_tags).
    """
    annotations = load_annotations(raw_dir)
    label_tags, auxiliary_tags = select_top_tags(annotations, cfg["data"]["top_k_tags"])
    metadata = load_clip_metadata(raw_dir)

    manifest = annotations.merge(metadata, on="clip_id", how="inner", suffixes=("", "_meta"))
    path_column = "mp3_path_meta" if "mp3_path_meta" in manifest.columns else "mp3_path"
    manifest["mp3_path"] = manifest[path_column]

    if cfg["data"]["split_by"] == "folder":
        try:
            official = load_official_splits(raw_dir)
            lookup = {clip: split for split, ids in official.items() for clip in ids}
            manifest["split"] = manifest["clip_id"].map(lookup)
        except FileNotFoundError:
            manifest["split"] = split_by_folder(manifest["mp3_path"], cfg)
    else:
        manifest["split"] = split_by_folder(manifest["mp3_path"], cfg)

    # Clips absent from the official split lists carry no usable partition.
    manifest = manifest[manifest["split"].notna()].reset_index(drop=True)

    # Drop clips with no positive label among the top-K tags: they carry no
    # supervision signal and inflate the trivial all-zero solution.
    manifest["n_positive"] = manifest[label_tags].sum(axis=1)
    manifest = manifest[manifest["n_positive"] > 0].reset_index(drop=True)

    manifest["context_text"] = [
        build_context_string(
            title=row.get("title"),
            artist=row.get("artist"),
            album=row.get("album"),
            auxiliary_tags=[tag for tag in auxiliary_tags if row.get(tag, 0) == 1],
            use_metadata=cfg["text"]["use_metadata"],
            use_auxiliary_tags=cfg["text"]["use_auxiliary_tags"],
        )
        for _, row in manifest.iterrows()
    ]
    # Clips cut from the same source track share a metadata string; this id lets
    # the contrastive loss avoid treating them as negatives.
    manifest["track_group"] = (
        manifest["artist"].astype(str) + "||" + manifest["album"].astype(str) + "||" + manifest["title"].astype(str)
    )
    return manifest, label_tags, auxiliary_tags


def save_vocabulary(processed_dir: Path, label_tags: list[str], auxiliary_tags: list[str]) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "tag_vocabulary.json").write_text(
        json.dumps({"label_tags": label_tags, "auxiliary_tags": auxiliary_tags}, indent=2),
        encoding="utf-8",
    )


def load_vocabulary(processed_dir: Path) -> tuple[list[str], list[str]]:
    payload = json.loads((processed_dir / "tag_vocabulary.json").read_text(encoding="utf-8"))
    return payload["label_tags"], payload["auxiliary_tags"]


def rebuild_texts(cfg, processed_dir: Path, clip_ids: list[int]) -> list[str]:
    """Regenerate context strings for ``clip_ids`` under the *current* config.

    Preprocessing bakes one context string per clip into ``graphs_{split}.pt``.
    Rebuilding here from ``manifest.csv`` instead means text-side switches such
    as ``text.use_auxiliary_tags=false`` take effect at train time, without
    re-running the 13-minute audio pipeline. Returns strings in ``clip_ids``
    order, so they line up with the stored graphs.
    """
    manifest = pd.read_csv(processed_dir / "manifest.csv")
    _, auxiliary_tags = load_vocabulary(processed_dir)
    manifest = manifest.set_index("clip_id").reindex(list(clip_ids))

    present = [tag for tag in auxiliary_tags if tag in manifest.columns]
    use_metadata = cfg["text"]["use_metadata"]
    use_auxiliary = cfg["text"]["use_auxiliary_tags"]

    texts = []
    for row in manifest.itertuples(index=False):
        values = row._asdict()
        texts.append(
            build_context_string(
                title=values.get("title"),
                artist=values.get("artist"),
                album=values.get("album"),
                auxiliary_tags=[tag for tag in present if values.get(tag, 0) == 1],
                use_metadata=use_metadata,
                use_auxiliary_tags=use_auxiliary,
            )
        )
    return texts


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
class MusicGraphDataset(InMemoryDataset):
    """Pre-built music structure graphs, with text tokenised at load time.

    Graphs are stored once by ``scripts/preprocess.py`` without any tokenizer
    dependency; the context strings are tokenised here instead, so switching
    text encoders (BERT -> DistilBERT, different max_length) needs no
    re-preprocessing of the 25k graphs.

    Each ``Data`` carries ``input_ids`` / ``attention_mask`` shaped (1, L) so
    PyG's default concatenation along dim 0 yields (B, L) after batching.
    """

    def __init__(
        self,
        processed_path: Path,
        tokenizer=None,
        max_length: int = 128,
        attach_text: bool = True,
        cfg=None,
    ) -> None:
        super().__init__(root=str(processed_path.parent))
        payload = torch.load(processed_path, map_location="cpu", weights_only=False)
        self.data, self.slices = payload["data"], payload["slices"]
        self.clip_ids: list[int] = payload["clip_ids"]
        self.texts: list[str] = payload["texts"]
        self.track_groups: list[str] = payload["track_groups"]

        # Honour text-side config overrides made after preprocessing.
        if cfg is not None:
            self.texts = rebuild_texts(cfg, processed_path.parent, self.clip_ids)

        self.attach_text = attach_text and tokenizer is not None
        if self.attach_text:
            encoded = tokenizer(
                self.texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            self._input_ids = encoded["input_ids"]
            self._attention_mask = encoded["attention_mask"]

    def _download(self) -> None:  # InMemoryDataset hooks we do not use
        pass

    def _process(self) -> None:
        pass

    def get(self, idx: int) -> Data:
        data = super().get(idx)
        if self.attach_text:
            data.input_ids = self._input_ids[idx].unsqueeze(0)
            data.attention_mask = self._attention_mask[idx].unsqueeze(0)
        return data

    def get_group_ids(self, indices) -> list[str]:
        return [self.track_groups[int(i)] for i in indices]

    @property
    def labels(self) -> np.ndarray:
        """(N, num_tags) label matrix, materialised from the collated store."""
        return self.data.y.view(len(self), -1).numpy()


class MelSpectrogramDataset(Dataset):
    """Lazy float16 memmap of fixed-size log-mel patches for the CNN baseline."""

    def __init__(self, memmap_path: Path, labels: np.ndarray, shape: tuple[int, int, int]) -> None:
        self.memmap_path = memmap_path
        self.shape = shape
        self.labels = torch.from_numpy(labels.astype(np.float32))
        self._memmap: np.memmap | None = None

    def _ensure_open(self) -> np.memmap:
        # Opened lazily so DataLoader worker processes each get their own handle.
        if self._memmap is None:
            self._memmap = np.memmap(self.memmap_path, dtype=np.float16, mode="r", shape=self.shape)
        return self._memmap

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        mel = np.asarray(self._ensure_open()[index], dtype=np.float32)
        return torch.from_numpy(mel), self.labels[index]


class TextTagDataset(Dataset):
    """Tokenised context strings + tag targets for the Task 1 BERT baseline."""

    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer, max_length: int = 128) -> None:
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.from_numpy(labels.astype(np.float32))

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][index],
            "attention_mask": self.encodings["attention_mask"][index],
            "labels": self.labels[index],
        }


def compute_pos_weight(labels: np.ndarray, cap: float = 20.0) -> torch.Tensor:
    """BCE ``pos_weight`` = (#negatives / #positives) per tag, clipped.

    Uncapped weights explode for the rarest tags (ratios above 100) and make
    training unstable, so the ratio is clipped at ``cap``.
    """
    positives = labels.sum(axis=0)
    negatives = labels.shape[0] - positives
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.where(positives > 0, negatives / np.maximum(positives, 1), 1.0)
    return torch.from_numpy(np.clip(weight, 1.0, cap).astype(np.float32))
