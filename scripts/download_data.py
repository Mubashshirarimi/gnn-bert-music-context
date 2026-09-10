"""Download MagnaTagATune (annotations, metadata, official splits, audio archive).

    python scripts/download_data.py
    python scripts/download_data.py --source official   # City University mirror

Two mirrors are supported. The Hugging Face mirror is the default because the
original City University server serves at roughly 60 KB/s, which puts the 3 GB
audio archive at well over ten hours; the same files come off Hugging Face at
1-2 MB/s. Downloads resume, so an interrupted run can simply be restarted.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_config, resolve, setup_logging  # noqa: E402

MIRRORS = {
    "hf": {
        "base": "https://huggingface.co/datasets/confit/magnatagatune/resolve/main",
        "files": [
            "annotations_final.csv",
            "clip_info_final.csv",
            "train_gt_mtt.tsv",
            "val_gt_mtt.tsv",
            "test_gt_mtt.tsv",
            "mp3.zip",
        ],
    },
    "official": {
        "base": "https://mirg.city.ac.uk/datasets/magnatagatune",
        "files": [
            "annotations_final.csv",
            "clip_info_final.csv",
            "mp3.zip.001",
            "mp3.zip.002",
            "mp3.zip.003",
        ],
    },
}

EXPECTED_SIZES = {
    "annotations_final.csv": 21_517_373,
    "clip_info_final.csv": 8_608_397,
    "mp3.zip": 2_972_769_864,
}


def remote_size(url: str) -> int | None:
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=30) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:
        return None


def download(url: str, destination: Path, logger, chunk: int = 1 << 20) -> None:
    """Resumable download with a byte-range request when a partial file exists."""
    expected = remote_size(url)
    existing = destination.stat().st_size if destination.exists() else 0

    if expected is not None and existing == expected:
        logger.info("%-24s already complete (%.1f MB)", destination.name, existing / 1e6)
        return

    headers = {}
    mode = "wb"
    if existing and expected is not None and existing < expected:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
        logger.info("%-24s resuming at %.1f MB", destination.name, existing / 1e6)

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response, open(destination, mode) as out:
        downloaded = existing
        while True:
            block = response.read(chunk)
            if not block:
                break
            out.write(block)
            downloaded += len(block)
            if expected:
                percent = 100.0 * downloaded / expected
                print(f"\r  {destination.name}: {downloaded/1e6:8.1f} / {expected/1e6:.1f} MB "
                      f"({percent:5.1f}%)", end="", flush=True)
    print()
    logger.info("%-24s done (%.1f MB)", destination.name, destination.stat().st_size / 1e6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=list(MIRRORS), default="hf")
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-audio", action="store_true", help="metadata and splits only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging()
    raw_dir = resolve(cfg, "raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)

    mirror = MIRRORS[args.source]
    logger.info("Downloading MagnaTagATune from the %r mirror into %s", args.source, raw_dir)

    for filename in mirror["files"]:
        if args.skip_audio and filename.startswith("mp3"):
            continue
        try:
            download(f"{mirror['base']}/{filename}", raw_dir / filename, logger)
        except Exception as exc:
            logger.error("Failed to download %s: %s", filename, exc)
            if not filename.startswith("mp3"):
                raise

    for filename, expected in EXPECTED_SIZES.items():
        path = raw_dir / filename
        if path.exists() and path.stat().st_size != expected:
            logger.warning(
                "%s is %d bytes, expected %d - re-run this script to resume",
                filename, path.stat().st_size, expected,
            )

    logger.info("Next: python scripts/preprocess.py")


if __name__ == "__main__":
    main()
