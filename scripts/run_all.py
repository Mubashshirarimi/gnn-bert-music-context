"""Run the full experiment suite: all four tasks, both baselines, the ablation sweep.

    python scripts/run_all.py                  # everything
    python scripts/run_all.py --only 1 2       # selected stages
    python scripts/run_all.py --epochs 3       # short smoke run over the real data
    python scripts/run_all.py --list           # show stages without running

Each stage is a separate subprocess so a failure in one does not lose the
others; a summary table is printed at the end and results accumulate in
results/metrics.json.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logging  # noqa: E402


def stage(name: str, description: str, commands: list[list[str]]) -> dict:
    return {"name": name, "description": description, "commands": commands}


def build_stages(epochs: int | None) -> list[dict]:
    """Every stage, in dependency order."""
    epoch_override = ["--epochs", str(epochs)] if epochs else []

    def train(task: str, run_name: str, *overrides: str) -> list[str]:
        cmd = [sys.executable, "-m", "src.train", "--task", task, "--run-name", run_name, *epoch_override]
        if overrides:
            cmd += ["--set", *overrides]
        return cmd

    def evaluate(task: str, run_name: str, *overrides: str) -> list[str]:
        cmd = [sys.executable, "-m", "src.evaluate", "--task", task, "--run-name", run_name]
        if overrides:
            cmd += ["--set", *overrides]
        return cmd

    return [
        stage("baselines_trivial", "B1: random / prior-frequency predictors",
              [[sys.executable, "-m", "src.evaluate", "--baselines"]]),
        stage("baseline_pca_mlp", "B4: PCA + MLP on hand-crafted features",
              [[sys.executable, "-m", "src.baselines", "--pca-mlp"]]),
        stage("baseline_cnn", "B2: CNN on mel-spectrogram",
              [train("cnn", "baseline_cnn"), evaluate("cnn", "baseline_cnn")]),
        stage("task1_bert", "Task 1: BERT multi-label tag classifier",
              [train("1", "task1_bert"), evaluate("1", "task1_bert")]),
        stage("task1_bert_metadata_only", "Task 1 variant: metadata text only (no auxiliary tags)",
              [train("1", "task1_bert_metadata_only", "text.use_auxiliary_tags=false"),
               evaluate("1", "task1_bert_metadata_only", "text.use_auxiliary_tags=false")]),
        stage("task2_gnn_sage", "Task 2: GraphSAGE on music structure graphs",
              [train("2", "task2_gnn_sage"), evaluate("2", "task2_gnn_sage")]),
        stage("task2_gnn_gat", "Task 2 variant: GAT encoder",
              [train("2", "task2_gnn_gat", "model.gnn.conv=gat"),
               evaluate("2", "task2_gnn_gat", "model.gnn.conv=gat")]),
        stage("task3_cross_attention", "Task 3: GNN-BERT cross-attention fusion",
              [train("3", "task3_cross_attention", "model.fusion.mode=cross_attention"),
               evaluate("3", "task3_cross_attention", "model.fusion.mode=cross_attention")]),
        stage("task3_concat", "Task 3 ablation: early concatenation",
              [train("3", "task3_concat", "model.fusion.mode=concat"),
               evaluate("3", "task3_concat", "model.fusion.mode=concat")]),
        stage("task3_gnn_only", "Task 3 ablation: graph branch only",
              [train("3", "task3_gnn_only", "model.fusion.mode=gnn_only"),
               evaluate("3", "task3_gnn_only", "model.fusion.mode=gnn_only")]),
        stage("task3_bert_only", "Task 3 ablation: text branch only",
              [train("3", "task3_bert_only", "model.fusion.mode=bert_only"),
               evaluate("3", "task3_bert_only", "model.fusion.mode=bert_only")]),
        stage("task4_contrastive", "Task 4: contrastive dual encoder + retrieval",
              [train("4", "task4_contrastive"), evaluate("4", "task4_contrastive")]),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", default=None, help="stage names or task numbers to run")
    parser.add_argument("--skip", nargs="*", default=[], help="stage names to skip")
    parser.add_argument("--epochs", type=int, default=None, help="override epochs for every training stage")
    parser.add_argument("--list", action="store_true", help="print the stage list and exit")
    args = parser.parse_args()

    logger = setup_logging(PROJECT_ROOT / "results" / "logs" / "run_all.log")
    stages = build_stages(args.epochs)

    if args.list:
        for item in stages:
            print(f"  {item['name']:32s} {item['description']}")
        return

    if args.only:
        wanted = set(args.only)
        stages = [s for s in stages if s["name"] in wanted or any(f"task{w}" in s["name"] for w in wanted)]
    stages = [s for s in stages if s["name"] not in set(args.skip)]

    if not stages:
        logger.error("No stages selected. Use --list to see the available names.")
        sys.exit(1)

    logger.info("Running %d stages: %s", len(stages), ", ".join(s["name"] for s in stages))
    summary = []

    for item in stages:
        logger.info("=" * 70)
        logger.info("STAGE %s - %s", item["name"], item["description"])
        logger.info("=" * 70)
        started = time.time()
        status = "ok"

        for command in item["commands"]:
            logger.info("$ %s", " ".join(command[1:]))
            result = subprocess.run(command, cwd=PROJECT_ROOT)
            if result.returncode != 0:
                logger.error("Stage %s failed (exit %d)", item["name"], result.returncode)
                status = f"FAILED (exit {result.returncode})"
                break

        elapsed = time.time() - started
        summary.append((item["name"], status, elapsed))
        logger.info("Stage %s: %s in %.1f min", item["name"], status, elapsed / 60)

    logger.info("=" * 70)
    logger.info("SUMMARY")
    for name, status, elapsed in summary:
        logger.info("  %-32s %-20s %6.1f min", name, status, elapsed / 60)
    total = sum(item[2] for item in summary)
    logger.info("Total wall time: %.1f min", total / 60)

    failures = [name for name, status, _ in summary if status != "ok"]
    if failures:
        logger.error("Failed stages: %s", ", ".join(failures))
        sys.exit(1)
    logger.info("All stages completed. See results/metrics.json and results/plots/")


if __name__ == "__main__":
    main()
