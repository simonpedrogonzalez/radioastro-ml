"""Compare saved QA, parity-logistic, and optional parity-CNN predictions."""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ml.report import (
    _comparison_lines,
    _metric_lines,
    _plot_confusion_matrices,
    _plot_strength,
)
from ml.task_evaluation import evaluate_task
from scripts.reporting import QuartoReporter, export_detached_report


LABELS = (0, 1, 2)
PROBABILITY_COLUMNS = ("probability_none", "probability_amp", "probability_phase")


def validation_rows(run):
    with (run / "predictions.csv").open(newline="") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if row["split"] in {"val", "validation"}
        ]
    if not rows:
        raise ValueError(f"No validation predictions in {run}")
    required = {*PROBABILITY_COLUMNS, "sample_kind", "corruption_snr_target"}
    missing = required - rows[0].keys()
    if missing:
        raise ValueError(f"{run} cannot be re-evaluated; missing columns {sorted(missing)}")
    return {row["sample_id"]: row for row in rows}


def _evaluate(rows, ids):
    return evaluate_task(
        [int(rows[key]["true_class"]) for key in ids],
        np.asarray([
            [float(rows[key][column]) for column in PROBABILITY_COLUMNS]
            for key in ids
        ]),
        LABELS,
        [
            {
                "corruption_snr_target": float(rows[key]["corruption_snr_target"]),
                "sample_kind": rows[key]["sample_kind"],
            }
            for key in ids
        ],
        clean_label=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--cnn", type=Path)
    parser.add_argument("--output", type=Path, default=Path("ml/runs"))
    parser.add_argument("--export", type=Path, required=True)
    args = parser.parse_args()

    runs = {
        "Four QA metrics": validation_rows(args.baseline),
        "Two parity metrics": validation_rows(args.parity),
    }
    if args.cnn:
        config = json.loads((args.cnn / "config.json").read_text())
        if config["input_mode"] != "parity":
            raise ValueError("The CNN must use parity input")
        runs[f'Parity CNN (seed {config["seed"]})'] = validation_rows(args.cnn)
    baseline = runs["Four QA metrics"]

    def identity(rows):
        return {
            key: (
                row["true_class"], row["source_id"],
                row["corruption_snr_target"], row["sample_kind"],
            )
            for key, row in rows.items()
        }

    if any(identity(baseline) != identity(rows) for rows in runs.values()):
        raise ValueError("Runs do not have identical validation cohorts")
    ids = sorted(baseline)
    metrics = {name: _evaluate(rows, ids) for name, rows in runs.items()}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output.resolve() / f"{stamp}_model_comparison"
    output.mkdir(parents=True, exist_ok=False)
    _plot_strength(metrics, output / "by_snr.png")
    _plot_confusion_matrices(metrics, output / "confusion_matrices.png")

    title = (
        "Four-metric vs parity logistic regression vs parity CNN"
        if args.cnn else "Four-metric vs parity logistic regression"
    )
    lines = [
        "---", f'title: "{title}"', "format:", "  html:",
        "    page-layout: full", "    embed-resources: true", "---", "",
        "## Overall validation", "", *_comparison_lines(metrics), "",
        "![Confusion matrices](confusion_matrices.png)", "",
        "## Metrics by corruption strength", "",
        "![Five metrics for all three evaluation views](by_snr.png)", "",
        "## Model details", "",
    ]
    for name, result in metrics.items():
        lines += _metric_lines(name, result)
    qmd = output / "report.qmd"
    qmd.write_text("\n".join(lines), encoding="utf-8")
    QuartoReporter(qmd).finish()
    if not qmd.with_suffix(".html").is_file():
        raise RuntimeError("Report rendering failed")
    exported = export_detached_report(qmd, args.export)
    print(f"Report: {qmd.with_suffix('.html')}\nExport: {exported.html_path}")


if __name__ == "__main__":
    main()
