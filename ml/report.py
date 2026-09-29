"""Small, self-contained run reports."""

from __future__ import annotations

import csv
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ml.evaluate import METRICS
from scripts.reporting import QuartoReporter


CLASS_NAMES = {0: "none", 1: "amp", 2: "phase"}


def _format_metric(value: float | None, *, bold: bool = False) -> str:
    if value is None:
        return "—"
    rendered = f"{value:.4f}"
    return f"**{rendered}**" if bold else rendered


def _comparison_lines(results: dict[str, dict[str, Any]]) -> list[str]:
    main = {name: result["overall"]["main"] for name, result in results.items()}
    defined = {
        metric: [value[metric] for value in main.values() if value[metric] is not None]
        for metric in METRICS
    }
    maxima = {metric: max(values) if values else None for metric, values in defined.items()}
    lines = [
        "| Model | " + " | ".join(metric.upper() for metric in METRICS) + " |",
        "| --- | " + " | ".join("---:" for _ in METRICS) + " |",
    ]
    for name, values in main.items():
        cells = [
            _format_metric(
                values[metric],
                bold=(len(main) > 1 and maxima[metric] is not None
                      and values[metric] == maxima[metric]),
            )
            for metric in METRICS
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def _metric_lines(name: str, result: dict[str, Any]) -> list[str]:
    if result["evaluation_kind"] == "increased_noise_controls":
        lines = [f"### {name}", "", "| Noise target | Samples | Correct rejection | FPR | Predicted counts |", "| ---: | ---: | ---: | ---: | --- |"]
        for level, values in result["by_noise_snr_target"].items():
            counts = ", ".join(
                f"{label}: {count}" for label, count in values["predicted_counts"].items()
            )
            lines.append(
                f'| {level} | {values["sample_count"]} | '
                f'{values["correct_rejection_rate"]:.4f} | '
                f'{values["false_positive_rate"]:.4f} | {counts} |'
            )
        return [*lines, ""]
    views = result["overall"]
    main = views["main"]
    lines = [
        f"### {name}",
        "",
        "| View | Samples | " + " | ".join(metric.upper() for metric in METRICS) + " |",
        "| --- | ---: | " + " | ".join("---:" for _ in METRICS) + " |",
    ]
    for label, key in (("Main (macro)", "main"), ("Detection", "detection"),
                       ("Error identification (macro)", "error_identification")):
        values = views[key]
        lines.append(
            f'| {label} | {values["sample_count"]} | '
            + " | ".join(_format_metric(values[metric]) for metric in METRICS)
            + " |"
        )
    error = views["error_identification"]
    lines += [
        "",
        f'Error-identification coverage: {error["detected_corruptions"]}/'
        f'{error["eligible_corruptions"]} '
        f'({_format_metric(error["detection_coverage"])}).',
        "",
        "| Class | Precision | Recall | F1 | AUROC | AUPRC | Support |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in main["labels"]:
        values = main["per_class"][str(label)]
        lines.append(
            f"| {label} | "
            + " | ".join(_format_metric(values[metric]) for metric in METRICS)
            + f' | {values["support"]} |'
        )
    return [*lines, ""]


def _plot_strength(
    results: dict[str, dict[str, Any]], destination: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    view_names = (("main", "Main"), ("detection", "Detection"),
                  ("error_identification", "Error identification"))
    figure, axes = plt.subplots(3, 5, figsize=(16, 9), sharex=True, sharey=True,
                               constrained_layout=True)
    for row, (view, title) in enumerate(view_names):
        for column, metric in enumerate(METRICS):
            axis = axes[row, column]
            for name, result in results.items():
                levels = sorted(
                    (float(level), values[view][metric])
                    for level, values in result["by_corruption_snr_target"].items()
                )
                axis.plot([level for level, _ in levels],
                          [value for _, value in levels], marker="o", label=name)
            if row == 0:
                axis.set_title(metric.upper())
            if column == 0:
                axis.set_ylabel(title)
            if row == 2:
                axis.set_xlabel("Target corruption SNR")
            axis.set_ylim(-0.02, 1.02)
            axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=max(1, len(labels)),
                  frameon=False)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _plot_confusion_matrices(
    results: dict[str, dict[str, Any]], destination: Path
) -> None:
    import matplotlib
    import numpy as np
    from sklearn.metrics import ConfusionMatrixDisplay

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(results)
    mains = [results[name]["overall"]["main"] for name in models]
    totals = {main["sample_count"] for main in mains}
    labels = {tuple(main["labels"]) for main in mains}
    if len(totals) != 1 or len(labels) != 1:
        raise ValueError("Confusion-matrix comparison requires identical cohorts and labels")
    figure, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.4),
                               constrained_layout=True, squeeze=False)
    displays = []
    for axis, name, main in zip(axes[0], models, mains, strict=True):
        display = ConfusionMatrixDisplay(
            np.asarray(main["confusion_matrix"]), display_labels=main["labels"]
        )
        display.plot(ax=axis, colorbar=False, values_format="d",
                     im_kw={"vmin": 0, "vmax": main["sample_count"]})
        axis.set_title(name)
        displays.append(display)
    figure.colorbar(displays[-1].im_, ax=list(axes[0]), label="Samples", shrink=0.8)
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
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    severity_plot = run_dir / "severity_performance.png"
    confusion_plot = run_dir / "confusion_matrix.png"
    scatter_plot = run_dir / "feature_scatter.png"
    importance_plot = run_dir / "feature_importance.png"
    _plot_strength({"Logistic regression": results["validation"]}, severity_plot)
    _plot_confusion_matrices(
        {"Logistic regression": results["validation"]}, confusion_plot
    )
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
            "### Confusion matrix",
            "",
            "![Validation confusion matrix](confusion_matrix.png)",
            "",
            "### Performance by corruption level",
            "",
            (
                "Each strength cohort contains the clean baselines and every error "
                "class at that strength. Error identification is conditional on "
                "binary detection."
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


__all__ = [
    "_comparison_lines",
    "_metric_lines",
    "_plot_confusion_matrices",
    "_plot_strength",
    "write_run",
]
