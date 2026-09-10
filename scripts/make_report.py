"""Turn results/metrics.json into LaTeX tables and a Markdown summary for the report.

    python scripts/make_report.py

Writes:
    report/results_tables.tex   \\input-ed by final_report.tex
    report/results_summary.md   human-readable version of the same numbers

Keeping the tables generated means the report can never drift from the metrics
that were actually measured.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import PROJECT_ROOT, setup_logging  # noqa: E402

# Display name and ordering for every run the pipeline can produce.
RUN_LABELS: list[tuple[str, str]] = [
    ("baseline_random", "B1: prior-frequency predictor"),
    ("baseline_pca_mlp", "B4: PCA + MLP (hand-crafted)"),
    ("baseline_cnn", "B2: CNN on mel-spectrogram"),
    ("task1_bert_metadata_only", "Task 1: BERT (metadata only)"),
    ("task1_bert", "Task 1: BERT (metadata + aux tags)"),
    ("task2_gnn_sage", "Task 2: GraphSAGE"),
    ("task2_gnn_gat", "Task 2: GAT"),
    ("task3_bert_only", "Task 3 ablation: BERT only"),
    ("task3_gnn_only", "Task 3 ablation: GNN only"),
    ("task3_concat", "Task 3 ablation: early concat"),
    ("task3_cross_attention", "Task 3: GNN-BERT cross-attention"),
]

ABLATION_RUNS = ["task3_bert_only", "task3_gnn_only", "task3_concat", "task3_cross_attention"]


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "--"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "--" if number != number else f"{number:.{digits}f}"  # NaN check


def escape(text: str) -> str:
    return text.replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def main_table(metrics: dict) -> tuple[str, list[list[str]]]:
    rows = []
    for key, label in RUN_LABELS:
        block = metrics.get(key)
        if not isinstance(block, dict) or not isinstance(block.get("test"), dict):
            continue
        test = block["test"]
        rows.append([
            label,
            fmt(test.get("macro_f1")),
            fmt(test.get("micro_f1")),
            fmt(test.get("auc_pr_macro")),
            fmt(test.get("roc_auc_macro")),
        ])

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Multi-label tagging performance on the MagnaTagATune test split "
        r"(5{,}329 clips, top-50 tags). Per-tag decision thresholds are fitted on the "
        r"validation split and frozen for test.}",
        r"  \label{tab:main-results}",
        r"  \begin{tabular}{lcccc}",
        r"    \toprule",
        r"    Model & Macro-F1 & Micro-F1 & AUC-PR & ROC-AUC \\",
        r"    \midrule",
    ]
    for row in rows:
        lines.append("    " + " & ".join([escape(row[0])] + row[1:]) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    return "\n".join(lines), rows


def ablation_table(metrics: dict) -> tuple[str, list[list[str]]]:
    rows = []
    for key in ABLATION_RUNS:
        block = metrics.get(key)
        if not isinstance(block, dict) or not isinstance(block.get("test"), dict):
            continue
        label = dict(RUN_LABELS).get(key, key)
        test = block["test"]
        rows.append([
            label.split(": ", 1)[-1],
            fmt(test.get("macro_f1")),
            fmt(test.get("auc_pr_macro")),
            fmt(test.get("graph_coherence")),
        ])

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Fusion ablation. All four variants share the same heads, optimiser "
        r"and schedule; only the fusion path differs.}",
        r"  \label{tab:ablation}",
        r"  \begin{tabular}{lccc}",
        r"    \toprule",
        r"    Fusion variant & Macro-F1 & AUC-PR & Graph coherence \\",
        r"    \midrule",
    ]
    for row in rows:
        lines.append("    " + " & ".join([escape(row[0])] + row[1:]) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    return "\n".join(lines), rows


def retrieval_table(metrics: dict) -> tuple[str, list[list[str]]]:
    block = metrics.get("task4_contrastive")
    rows = []
    if isinstance(block, dict) and isinstance(block.get("test"), dict):
        test = block["test"]
        for direction, label in (("caption_to_audio", "Caption $\\rightarrow$ Audio"),
                                 ("audio_to_caption", "Audio $\\rightarrow$ Caption")):
            rows.append([
                label,
                fmt(test.get(f"{direction}_R@1")),
                fmt(test.get(f"{direction}_R@5")),
                fmt(test.get(f"{direction}_R@10")),
                fmt(test.get(f"{direction}_median_rank"), 1),
            ])

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Cross-modal retrieval on the test split (Task 4). Clips cut from the "
        r"same source track count as correct matches, mirroring the training-time "
        r"duplicate masking.}",
        r"  \label{tab:retrieval}",
        r"  \begin{tabular}{lcccc}",
        r"    \toprule",
        r"    Direction & R@1 & R@5 & R@10 & Median rank \\",
        r"    \midrule",
    ]
    for row in rows:
        lines.append("    " + " & ".join(row) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    return "\n".join(lines), rows


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_(no results yet)_\n"
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c).replace("$\\rightarrow$", "->") for c in row) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    logger = setup_logging()
    metrics_path = PROJECT_ROOT / "results" / "metrics.json"
    if not metrics_path.exists():
        logger.error("No results/metrics.json - run scripts/run_all.py first")
        sys.exit(1)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    main_tex, main_rows = main_table(metrics)
    ablation_tex, ablation_rows = ablation_table(metrics)
    retrieval_tex, retrieval_rows = retrieval_table(metrics)

    report_dir = PROJECT_ROOT / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "results_tables.tex").write_text(
        "% Auto-generated by scripts/make_report.py - do not edit by hand.\n\n"
        + main_tex + "\n" + ablation_tex + "\n" + retrieval_tex,
        encoding="utf-8",
    )

    summary = [
        "# Results summary",
        "",
        "Auto-generated from `results/metrics.json` by `scripts/make_report.py`.",
        "",
        "## Multi-label tagging (MTAT test split, top-50 tags)",
        "",
        markdown_table(["Model", "Macro-F1", "Micro-F1", "AUC-PR", "ROC-AUC"], main_rows),
        "## Fusion ablation (Task 3)",
        "",
        markdown_table(["Fusion variant", "Macro-F1", "AUC-PR", "Graph coherence"], ablation_rows),
        "## Cross-modal retrieval (Task 4)",
        "",
        markdown_table(["Direction", "R@1", "R@5", "R@10", "Median rank"], retrieval_rows),
    ]
    (report_dir / "results_summary.md").write_text("\n".join(summary), encoding="utf-8")

    logger.info("Wrote report/results_tables.tex (%d main, %d ablation, %d retrieval rows)",
                len(main_rows), len(ablation_rows), len(retrieval_rows))
    logger.info("Wrote report/results_summary.md")
    if not main_rows:
        logger.warning("No completed runs found in metrics.json yet.")


if __name__ == "__main__":
    main()
