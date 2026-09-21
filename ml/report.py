"""Small, self-contained run reports."""

from __future__ import annotations

import csv
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _metric_lines(name: str, metrics: dict[str, Any]) -> list[str]:
    return [
        f"### {name}",
        "",
        (
            "| Samples | Accuracy | Balanced accuracy | Macro precision | "
            "Macro recall | Macro F1 |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        "| "
        + " | ".join(
            [
                str(metrics["sample_count"]),
                f'{metrics["accuracy"]:.4f}',
                f'{metrics["balanced_accuracy"]:.4f}',
                f'{metrics["macro_precision"]:.4f}',
                f'{metrics["macro_recall"]:.4f}',
                f'{metrics["macro_f1"]:.4f}',
            ]
        )
        + " |",
        "",
        "Confusion matrix (rows=true, columns=predicted; classes 0/1/2):",
        "",
        "```text",
        *[" ".join(str(value) for value in row) for row in metrics["confusion_matrix"]],
        "```",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "| ---: | ---: | ---: | ---: | ---: |",
        *[
            f'| {label} | {values["precision"]:.4f} | '
            f'{values["recall"]:.4f} | {values["f1"]:.4f} | '
            f'{values["support"]} |'
            for label, values in metrics["per_class"].items()
        ],
        "",
        (
            "| Target SNR | Samples | Accuracy | Balanced accuracy | Macro "
            "precision | Macro recall | Macro F1 | FPR | Detection recall | "
            "Type accuracy |"
        ),
        (
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: |"
        ),
        *[
            "| "
            + " | ".join(
                [
                    level,
                    str(values["sample_count"]),
                    f'{values["accuracy"]:.4f}',
                    f'{values["balanced_accuracy"]:.4f}',
                    f'{values["macro_precision"]:.4f}',
                    f'{values["macro_recall"]:.4f}',
                    f'{values["macro_f1"]:.4f}',
                    _optional(values.get("false_positive_rate")),
                    _optional(values.get("detection_recall")),
                    _optional(values.get("corruption_type_accuracy")),
                ]
            )
            + " |"
            for level, values in metrics["by_corruption_snr_target"].items()
        ],
        "",
    ]


def _optional(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def write_run(
    output_root: str | Path,
    *,
    dataset_path: Path,
    model: Any,
    prediction_rows: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    feature_names: tuple[str, ...],
    split_counts: dict[str, dict[str, int]],
    versions: dict[str, str],
) -> Path:
    """Write one model, prediction table, and Markdown report."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_root).expanduser().resolve() / f"{timestamp}_metric_logreg"
    run_dir.mkdir(parents=True, exist_ok=False)

    with (run_dir / "model.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    with (run_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)

    lines = [
        "# Metric logistic-regression run",
        "",
        f"- Dataset: `{dataset_path}`",
        "- Labels: `none=0`, `amp=1`, `phase=2`",
        "- Label criterion: `constant_antenna_type`",
        (
            "- Model: `StandardScaler -> LogisticRegression(C=1.0, "
            "class_weight=balanced, solver=lbfgs, max_iter=1000)`"
        ),
        f"- Test evaluated: `{'yes' if 'test' in results else 'no'}`",
        "",
        "## Environment",
        "",
        *[f"- {name}: `{version}`" for name, version in versions.items()],
        "",
        "## Split sizes",
        "",
        "| Split | Samples | Sources |",
        "| --- | ---: | ---: |",
        *[
            f'| {split} | {counts["samples"]} | {counts["sources"]} |'
            for split, counts in split_counts.items()
        ],
        "",
        "## Features",
        "",
        *[f"- `{name}`" for name in feature_names],
        "",
        "## Results",
        "",
    ]
    for split, metrics in results.items():
        lines.extend(_metric_lines(split.title(), metrics))
    lines.extend(
        [
            (
                "Corruption reports and their `SNR_corr` values were used only "
                "for labels and grouped evaluation; they were not model inputs."
            ),
            "",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return run_dir


__all__ = ["write_run"]
