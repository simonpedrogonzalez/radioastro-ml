"""Small, self-contained run reports."""

from __future__ import annotations

import csv
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.reporting import QuartoReporter


CLASS_NAMES = {0: "none", 1: "amp", 2: "phase"}


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
        *_severity_lines(metrics),
        "",
    ]


def _conditional_type(values: dict[str, Any]) -> float | None:
    if "type_accuracy_among_detected" in values:
        return values["type_accuracy_among_detected"]
    detected = values.get("detection_recall")
    return values["corruption_type_accuracy"] / detected if detected else None


def _severity_lines(metrics: dict[str, Any]) -> list[str]:
    levels = metrics["by_corruption_snr_target"]
    clean = levels.get("0")

    def row(level: str, values: dict[str, Any]) -> str:
        total = values["sample_count"]
        detected = round(values["detection_recall"] * total)
        correct = round(values["corruption_type_accuracy"] * total)
        conditional = _conditional_type(values)
        type_score = f"{conditional:.4f} ({correct}/{detected})" if conditional is not None else "—"
        return (f'| {level} | {total} | {values["detection_recall"]:.4f} '
                f'({detected}/{total}) | {type_score} |')

    lines = [
        "| Target SNR | Samples | Detected as corrupted | Correct type among detections |",
        "| ---: | ---: | ---: | ---: |",
        *[
            row(level, values)
            for level, values in levels.items() if values.get("detection_recall") is not None
        ],
    ]
    if clean is not None:
        lines += ["", f'Clean false alarms (SNR 0): '
                         f'{round(clean["false_positive_rate"] * clean["sample_count"])}'
                         f'/{clean["sample_count"]} ({clean["false_positive_rate"]:.4f}).']
    lines += ["", "Detection uses all corrupted samples at that level; type correctness "
                  "uses only those detected. If none are detected, type correctness is undefined (—)."]
    return lines


def _plot_severity(metrics: dict[str, Any], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = [
        (float(level), values)
        for level, values in metrics["by_corruption_snr_target"].items()
        if values.get("detection_recall") is not None
    ]
    levels.sort()
    x = [level for level, _ in levels]
    detection = [values["detection_recall"] for _, values in levels]
    correct_type = [_conditional_type(values) for _, values in levels]

    figure, axis = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    axis.plot(x, detection, marker="o", linewidth=2, label="detected as corrupted")
    axis.plot(x, correct_type, marker="o", linewidth=2, label="correct type among detections")
    axis.set(xlabel="Target corruption SNR", ylabel="Fraction", ylim=(-0.02, 1.02))
    axis.set_xticks(x)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _plot_feature_scatter(
    model: Any,
    rows: list[dict[str, Any]],
    feature_names: tuple[str, ...],
    destination: Path,
) -> tuple[str, str]:
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(feature_names) < 2:
        raise ValueError("At least two features are required for the feature scatter")
    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["logisticregression"]
    importance = np.max(np.abs(np.asarray(classifier.coef_, dtype=float)), axis=0)
    selected = np.argsort(importance)[-2:][::-1]
    selected_names = tuple(feature_names[int(index)] for index in selected)
    validation = [row for row in rows if row["split"] == "validation"]
    values = np.asarray(
        [[row[name] for name in feature_names] for row in validation], dtype=float
    )
    standardized = scaler.transform(values)
    truth = np.asarray([row["true_class"] for row in validation], dtype=int)

    figure, axis = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
    colors = {0: "#4c78a8", 1: "#f58518", 2: "#54a24b"}
    for class_id in CLASS_NAMES:
        chosen = truth == class_id
        axis.scatter(
            standardized[chosen, selected[0]],
            standardized[chosen, selected[1]],
            s=28,
            alpha=0.7,
            color=colors[class_id],
            label=CLASS_NAMES[class_id],
        )
    axis.axhline(0.0, color="0.75", linewidth=0.8)
    axis.axvline(0.0, color="0.75", linewidth=0.8)
    axis.set(
        xlabel=f"{selected_names[0]} (standardized)",
        ylabel=f"{selected_names[1]} (standardized)",
    )
    axis.grid(alpha=0.15)
    axis.legend(title="True class", frameon=False)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return selected_names


def _feature_importance(
    model: Any, feature_names: tuple[str, ...]
) -> list[dict[str, Any]]:
    import numpy as np

    classifier = model.named_steps["logisticregression"]
    coefficients = np.asarray(classifier.coef_, dtype=float)
    classes = [CLASS_NAMES[int(class_id)] for class_id in classifier.classes_]
    return sorted(
        [
            {
                "feature": feature,
                "coefficients": {
                    class_name: float(coefficients[class_index, feature_index])
                    for class_index, class_name in enumerate(classes)
                },
                "importance": float(
                    np.max(np.abs(coefficients[:, feature_index]))
                ),
            }
            for feature_index, feature in enumerate(feature_names)
        ],
        key=lambda row: row["importance"],
        reverse=True,
    )


def _plot_feature_importance(
    importance_rows: list[dict[str, Any]], destination: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = list(reversed(importance_rows))
    figure, axis = plt.subplots(figsize=(8.2, 4.2), constrained_layout=True)
    axis.barh(
        [row["feature"] for row in ordered],
        [row["importance"] for row in ordered],
        color="#4c78a8",
    )
    axis.set_xlabel("Maximum absolute standardized coefficient")
    axis.grid(axis="x", alpha=0.25)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


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
    """Write one model, prediction table, and Quarto HTML report."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_root).expanduser().resolve() / f"{timestamp}_metric_logreg"
    run_dir.mkdir(parents=True, exist_ok=False)

    with (run_dir / "model.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    with (run_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    severity_plot = run_dir / "severity_performance.png"
    scatter_plot = run_dir / "feature_scatter.png"
    importance_plot = run_dir / "feature_importance.png"
    _plot_severity(results["validation"], severity_plot)
    scatter_features = _plot_feature_scatter(
        model, prediction_rows, feature_names, scatter_plot
    )
    importance_rows = _feature_importance(model, feature_names)
    _plot_feature_importance(importance_rows, importance_plot)

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
            "## Feature importance",
            "",
            (
                "Coefficients are fitted to standardized features. Their signs are "
                "class-specific; the overall importance shown here is the maximum "
                "absolute coefficient across the three classes."
            ),
            "",
            "| Feature | None coefficient | Amp coefficient | Phase coefficient | Overall importance |",
            "| --- | ---: | ---: | ---: | ---: |",
            *[
                f'| `{row["feature"]}` | '
                f'{row["coefficients"]["none"]:.4f} | '
                f'{row["coefficients"]["amp"]:.4f} | '
                f'{row["coefficients"]["phase"]:.4f} | '
                f'{row["importance"]:.4f} |'
                for row in importance_rows
            ],
            "",
            "![Feature importance](feature_importance.png)",
            "",
            "## Diagnostic plots",
            "",
            "### Performance by corruption level",
            "",
            (
                "Detection uses all corrupted samples; type correctness uses only "
                "the ones detected as corrupted."
            ),
            "",
            "![Performance by corruption level](severity_performance.png)",
            "",
            "### Most influential feature pair",
            "",
            (
                "The two features were selected by the largest absolute fitted "
                "coefficient across classes after standardization: "
                f"`{scatter_features[0]}` and `{scatter_features[1]}`. Points are "
                "held-out validation samples colored by true class."
            ),
            "",
            "![True classes in the most influential feature pair](feature_scatter.png)",
            "",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    quarto_lines = [
        "---",
        'title: "Metric logistic-regression run"',
        "format:",
        "  html:",
        "    toc: true",
        "    embed-resources: true",
        "---",
        "",
        *lines[2:],
    ]
    quarto_path = run_dir / "report.qmd"
    quarto_path.write_text("\n".join(quarto_lines), encoding="utf-8")
    QuartoReporter(quarto_path, every=1).finish()
    if not quarto_path.with_suffix(".html").is_file():
        raise RuntimeError("Quarto did not produce report.html")
    return run_dir


__all__ = ["write_run"]
