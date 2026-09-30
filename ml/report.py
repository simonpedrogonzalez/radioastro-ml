"""Small, self-contained run reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml.evaluate import METRICS
from scripts.reporting import QuartoReporter


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


def _noise_control_lines(results: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| Model | Noise target | Samples | Correct rejection | FPR | Matched baseline FPR | Excess FPR | Predicted counts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, result in results.items():
        cohorts = [("all", result["overall"]), *result["by_noise_snr_target"].items()]
        for level, values in cohorts:
            counts = ", ".join(
                f"{label}: {count}" for label, count in values["predicted_counts"].items()
            )
            lines.append(
                f"| {name} | {level} | {values['sample_count']} | "
                f"{values['correct_rejection_rate']:.4f} | "
                f"{values['false_positive_rate']:.4f} | "
                f"{values['matched_baseline_false_positive_rate']:.4f} | "
                f"{values['excess_false_positive_rate']:+.4f} | {counts} |"
            )
    return lines


def _plot_noise_controls(
    results: dict[str, dict[str, Any]], destination: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    for name, result in results.items():
        pairs = sorted(
            (float(level), values)
            for level, values in result["by_noise_snr_target"].items()
        )
        levels = [level for level, _ in pairs]
        values = [values for _, values in pairs]
        line, = axis.plot(
            levels, [value["false_positive_rate"] for value in values], marker="o", label=name
        )
        axis.plot(
            levels,
            [value["matched_baseline_false_positive_rate"] for value in values],
            linestyle="--", color=line.get_color(), alpha=.65,
        )
    axis.set(xlabel="Injected-noise SNR target", ylabel="False-positive rate", ylim=(-.02, 1.02))
    axis.grid(alpha=.25)
    axis.legend(frameon=False)
    axis.text(.01, .01, "solid: noisy control; dashed: matched clean baseline",
              transform=axis.transAxes, fontsize=9)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def write_tabular_report(
    run_dir: str | Path,
    results: dict[str, dict[str, Any]],
    tuning: dict[str, dict[str, Any]],
    importance: dict[str, list[dict[str, Any]]],
) -> Path:
    """Render one canonical comparison report for tuned tabular models."""

    run_dir = Path(run_dir)
    validation = {name: value["validation"] for name, value in results.items()}
    controls = {name: value["noise_controls"] for name, value in results.items()}
    _plot_strength(validation, run_dir / "strength.png")
    _plot_confusion_matrices(validation, run_dir / "confusion_matrices.png")
    _plot_noise_controls(controls, run_dir / "noise_controls.png")
    lines = [
        "---", 'title: "Tuned tabular corruption-detection models"', "format:", "  html:",
        "    page-layout: full", "    toc: true", "    embed-resources: true", "---", "",
        "# Validation comparison", "", *_comparison_lines(validation), "",
        "![Validation confusion matrices](confusion_matrices.png)", "",
        "# Metrics by corruption strength", "",
        "![Five metrics for all evaluation views](strength.png)", "",
        "# Increased-noise controls", "",
        "All controls are true clean examples and were excluded from fitting and model selection.", "",
        *_noise_control_lines(controls), "", "![Noise-control false positives](noise_controls.png)", "",
        "# Hyperparameter selection", "",
        "| Model | CV macro F1 | CV macro recall | Selected parameters |", "| --- | ---: | ---: | --- |",
    ]
    for name, values in tuning.items():
        parameters = json.dumps(values["best_params"], sort_keys=True).replace("|", "\\|")
        lines.append(
            f"| {name} | {values['mean_f1']:.4f} ± {values['std_f1']:.4f} | "
            f"{values['mean_recall']:.4f} ± {values['std_recall']:.4f} | `{parameters}` |"
        )
    lines += ["", "# Model details", ""]
    for name, result in validation.items():
        lines += _metric_lines(name, result)
        lines += ["#### Feature importance", "", "| Feature | Importance |", "| --- | ---: |"]
        lines += [f"| `{row['feature']}` | {row['importance']:.6g} |"
                  for row in importance[name]]
        lines.append("")
    qmd = run_dir / "report.qmd"
    qmd.write_text("\n".join(lines) + "\n", encoding="utf-8")
    QuartoReporter(qmd, every=1).finish()
    if not qmd.with_suffix(".html").is_file():
        raise RuntimeError("Quarto did not produce report.html")
    return qmd


__all__ = [
    "_comparison_lines",
    "_metric_lines",
    "_plot_confusion_matrices",
    "_noise_control_lines",
    "_plot_noise_controls",
    "_plot_strength",
    "write_tabular_report",
]
