"""Small, self-contained run reports."""

from __future__ import annotations

import json
import warnings
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from typing import Any

from ml.evaluate import METRICS
from ml.task_evaluation import (
    STRENGTH_SATURATION_TARGET,
    corruption_strength_weights,
    strength_weighted_confusion,
)
from scripts.reporting import QuartoReporter


def _hard_sample_style() -> list[str]:
    """Compact the shared six-column sample gallery without shrinking other tables."""

    return [
        "<style>",
        ".hard-samples table { font-size: .62rem; line-height: 1.15; "
        "table-layout: fixed; width: 100%; }",
        ".hard-samples th, .hard-samples td { padding: .2rem; vertical-align: top; }",
        ".hard-samples th:nth-child(1), .hard-samples td:nth-child(1) { width: 9%; "
        "overflow-wrap: anywhere; }",
        ".hard-samples th:nth-child(n+2):nth-child(-n+5), "
        ".hard-samples td:nth-child(n+2):nth-child(-n+5) { width: 16%; }",
        ".hard-samples th:nth-child(6), .hard-samples td:nth-child(6) { width: 27%; }",
        ".hard-samples code { white-space: normal; overflow-wrap: anywhere; font-size: .58rem; }",
        ".hard-samples img { width: 100%; height: auto; max-width: none; }",
        "</style>",
    ]


def _class_name(label: Any) -> str:
    return {"0": "No corruption (0)", "1": "Amplitude (1)",
            "2": "Phase (2)"}.get(str(label), f"Class {label}")


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
    if result["evaluation_kind"] != "classification":
        raise ValueError("Noise controls must use the shared overall control tables")
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


def _ensure_strength_weighting(
    results: dict[str, dict[str, Any]], prediction_rows: list[dict[str, Any]]
) -> None:
    """Populate new weighted-confusion fields when reading older artifacts."""

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        if row.get("split") == "validation":
            by_model[str(row["model"])].append(row)
    for name, result in results.items():
        main = result["overall"]["main"]
        rows = by_model.get(name, [])
        if not rows:
            raise ValueError(f"No validation predictions found for model {name!r}")
        labels = tuple(main["labels"])
        label_by_text = {str(label): label for label in labels}
        try:
            truth = [label_by_text[str(row["true_label"])] for row in rows]
            predicted = [label_by_text[str(row["predicted_label"])] for row in rows]
        except KeyError as exc:
            raise ValueError(f"Prediction label absent from result labels: {exc.args[0]}") from exc
        targets = [float(row["corruption_snr_target"]) for row in rows]
        if "strength_weighted_confusion_matrix" not in main:
            main["strength_weighted_confusion_matrix"] = strength_weighted_confusion(
                truth, predicted, labels, targets
            )
        if "strength_weighting" not in result:
            result["strength_weighting"] = {
                "formula": "clean=1; corrupted=min(1, log1p(target)/log1p(30))",
                "saturation_target": STRENGTH_SATURATION_TARGET,
                "by_target": {
                    f"{level:g}": float(corruption_strength_weights([level])[0])
                    for level in sorted(set(targets))
                },
            }


def _plot_confusion_matrices(
    results: dict[str, dict[str, Any]], destination: Path, *,
    raw_label: str = "Raw counts", raw_format: str = "d",
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
    weighted = [main.get("strength_weighted_confusion_matrix") for main in mains]
    if any(matrix is None for matrix in weighted):
        raise ValueError("Validation result is missing strength-weighted confusion matrix")
    raw_arrays = [np.asarray(main["confusion_matrix"]) for main in mains]
    weighted_arrays = [np.asarray(matrix, dtype=float) for matrix in weighted]
    maxima = (max(float(matrix.max()) for matrix in raw_arrays),
              max(float(matrix.max()) for matrix in weighted_arrays))
    figure, axes = plt.subplots(2, len(models), figsize=(5 * len(models), 8.6),
                               constrained_layout=True, squeeze=False)
    for row, (matrices, row_label, value_format) in enumerate((
        (raw_arrays, raw_label, raw_format),
        (weighted_arrays, "Strength-weighted counts", ".1f"),
    )):
        displays = []
        for column, (axis, name, main, matrix) in enumerate(
            zip(axes[row], models, mains, matrices, strict=True)
        ):
            display = ConfusionMatrixDisplay(matrix, display_labels=main["labels"])
            display.plot(ax=axis, colorbar=False, values_format=value_format,
                         im_kw={"vmin": 0, "vmax": maxima[row]})
            axis.set_title(name if row == 0 else "")
            if column == 0:
                axis.set_ylabel(f"{row_label}\nTrue label")
            displays.append(display)
        figure.colorbar(displays[-1].im_, ax=list(axes[row]), label=row_label, shrink=0.75)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _plot_strength_weighting(
    results: dict[str, dict[str, Any]], destination: Path
) -> None:
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    definitions = [result["strength_weighting"] for result in results.values()]
    if any(definition != definitions[0] for definition in definitions[1:]):
        raise ValueError("Models use different corruption-strength weighting")
    definition = definitions[0]
    saturation = float(definition["saturation_target"])
    observed = sorted(
        (float(target), float(weight))
        for target, weight in definition["by_target"].items() if float(target) > 0
    )
    maximum = max(target for target, _ in observed)
    grid = np.linspace(.001, maximum, 1000)
    curve = np.minimum(1.0, np.log1p(grid) / np.log1p(saturation))
    figure, axis = plt.subplots(figsize=(7.5, 4.3), constrained_layout=True)
    axis.plot(grid, curve, color="#4c78a8", label="Corrupted-sample weight")
    axis.scatter([target for target, _ in observed], [weight for _, weight in observed],
                 color="#4c78a8", zorder=3, label="Observed targets")
    axis.scatter([0], [1], marker="D", color="#e45756", zorder=4,
                 label="Clean baseline (special weight)")
    axis.set(xlabel="Target corruption SNR", ylabel="Confusion-matrix sample weight",
             ylim=(-.02, 1.05), xlim=(-maximum * .02, maximum * 1.02))
    axis.grid(alpha=.25); axis.legend(frameon=False)
    axis.text(.99, .05, definition["formula"], transform=axis.transAxes,
              ha="right", fontsize=9)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _noise_control_lines(results: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| Model | Original FPR | Increased-noise FPR | Excess FPR |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, result in results.items():
        values = result["overall"]
        lines.append(
            f"| {name} | {values['matched_baseline_false_positive_rate']:.4f} | "
            f"{values['false_positive_rate']:.4f} | "
            f"{values['excess_false_positive_rate']:+.4f} |"
        )
    return lines


def _noise_control_count_lines(results: dict[str, dict[str, Any]]) -> list[str]:
    first = next(iter(results.values()))["overall"]["predicted_counts"]
    labels = list(first)
    if any(list(result["overall"]["predicted_counts"]) != labels
           for result in results.values()):
        raise ValueError("Noise-control models use different predicted classes")
    lines = [
        "| Model | " + " | ".join(f"Predicted {_class_name(label)}" for label in labels) + " |",
        "| --- | " + " | ".join("---:" for _ in labels) + " |",
    ]
    for name, result in results.items():
        counts = result["overall"]["predicted_counts"]
        cells = []
        for label in labels:
            value = float(counts[label])
            cells.append(str(int(value)) if value.is_integer() else f"{value:.1f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
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


def _select_hard_samples(
    prediction_rows: list[dict[str, Any]], limit: int = 10
) -> list[dict[str, Any]]:
    """Return validation samples that every represented model gets wrong."""

    if limit <= 0:
        raise ValueError("hard-sample limit must be positive")
    validation = [row for row in prediction_rows if row.get("split") == "validation"]
    if not validation:
        return []
    by_model: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in validation:
        model, sample = str(row.get("model", "")), str(row.get("sample_id", ""))
        if not model or not sample or sample in by_model[model]:
            raise ValueError("Predictions require one model/sample validation row")
        by_model[model][sample] = row
    sample_sets = [set(rows) for rows in by_model.values()]
    if any(samples != sample_sets[0] for samples in sample_sets[1:]):
        raise ValueError("Hard-sample comparison requires identical validation cohorts")
    hard = []
    for sample in sample_sets[0]:
        rows = [model_rows[sample] for model_rows in by_model.values()]
        truth = str(rows[0]["true_label"])
        target = float(rows[0]["corruption_snr_target"])
        kind = str(rows[0]["sample_kind"])
        if any(str(row["true_label"]) != truth
               or float(row["corruption_snr_target"]) != target
               or str(row["sample_kind"]) != kind for row in rows[1:]):
            raise ValueError(f"Models disagree on truth metadata for {sample!r}")
        if all(str(row["predicted_label"]) != truth for row in rows):
            hard.append({
                **rows[0],
                "model_count": len(rows),
                "prediction_counts": dict(sorted(Counter(
                    str(row["predicted_label"]) for row in rows
                ).items())),
            })
    hard.sort(key=lambda row: (
        0 if row["sample_kind"] == "baseline" else 1,
        0 if row["sample_kind"] == "baseline" else -float(row["corruption_snr_target"]),
        str(row["sample_id"]),
    ))
    return hard[:limit]


def _hard_sample_lines(
    run_dir: Path,
    dataset_path: Path,
    prediction_rows: list[dict[str, Any]],
) -> list[str]:
    from astropy.wcs import FITSFixedWarning

    from scripts.imaging.plot_utils import casa_image_to_png
    from scripts.preprocessing.schema import load_dataset_manifest, load_sample_manifest

    selected = _select_hard_samples(prediction_rows)
    model_count = len({str(row["model"]) for row in prediction_rows
                       if row.get("split") == "validation"})
    lines = [
        f"A sample appears here only when all **{model_count}** included model runs "
        "classified it incorrectly. Clean false positives come first; remaining "
        "rows are ordered by decreasing corruption strength.", "",
    ]
    if not selected:
        return [*lines, "No shared hard samples were found.", ""]
    needed = {str(row["sample_id"]) for row in selected}
    manifests = {}
    for path in load_dataset_manifest(dataset_path, require_samples=False).samples:
        payload = json.loads(path.read_text(encoding="utf-8"))
        sample_id = payload.get("sample_id")
        if sample_id in needed:
            manifests[sample_id] = load_sample_manifest(
                path, require_files=True, verify_integrity=False
            )
    missing = needed - set(manifests)
    if missing:
        raise ValueError(f"Hard samples are absent from the dataset index: {sorted(missing)}")
    lines += [
        "::: {.hard-samples}",
        "| ID | Dirty | Clean | Residual | PSF | Metrics |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    image_root = run_dir / "hard_samples"
    for row in selected:
        sample_id = str(row["sample_id"]); manifest = manifests[sample_id]
        images = []
        for channel in ("dirty", "clean", "residual", "psf"):
            destination = image_root / sample_id / f"{channel}.png"
            if not destination.is_file():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FITSFixedWarning)
                    casa_image_to_png(manifest.products[channel], destination,
                                      title=f"{sample_id}: {channel}", draw_beam=True)
            relative = destination.relative_to(run_dir).as_posix()
            images.append(f'<img src="{relative}" alt="{escape(sample_id)} {channel}" width="260">')
        qa = json.loads(manifest.imaging_qa.read_text(encoding="utf-8"))["metrics"]
        residual = qa["residual"]
        measured = row.get("corruption_snr")
        measured_text = "—" if measured in (None, "", "None") else f"{float(measured):.4g}"
        votes = ", ".join(
            f"{_class_name(label)}: {count}"
            for label, count in row["prediction_counts"].items()
        )
        metrics = "<br>".join((
            f"target SNR: {float(row['corruption_snr_target']):g}",
            f"measured SNR: {measured_text}",
            f"dynamic range/MAD: {float(qa['dynamic_range_scaled_mad']):.5g}",
            f"residual MAD: {float(residual['scaled_mad_jy_per_beam']):.5g} Jy/beam",
            f"peak/MAD: {float(residual['peak_over_scaled_mad']):.5g}",
            f"p99.5/MAD: {float(residual['p99_5_over_scaled_mad']):.5g}",
            f"predicted votes: {escape(votes)}",
        ))
        identity = (f"<code>{escape(sample_id)}</code><br>"
                    f"{escape(manifest.label_name)}")
        lines.append("| " + " | ".join((identity, *images, metrics)) + " |")
    return [*lines, ":::", ""]


def write_tabular_report(
    run_dir: str | Path,
    results: dict[str, dict[str, Any]],
    tuning: dict[str, dict[str, Any]],
    importance: dict[str, list[dict[str, Any]]],
    *,
    dataset_path: str | Path,
    prediction_rows: list[dict[str, Any]],
) -> Path:
    """Render one canonical comparison report for tuned tabular models."""

    run_dir = Path(run_dir)
    validation = {name: value["validation"] for name, value in results.items()}
    controls = {name: value["noise_controls"] for name, value in results.items()}
    _ensure_strength_weighting(validation, prediction_rows)
    _plot_strength(validation, run_dir / "strength.png")
    _plot_confusion_matrices(validation, run_dir / "confusion_matrices.png")
    _plot_strength_weighting(validation, run_dir / "strength_weighting.png")
    _plot_noise_controls(controls, run_dir / "noise_controls.png")
    hard_lines = _hard_sample_lines(
        run_dir, Path(dataset_path).expanduser().resolve(), prediction_rows
    )
    lines = [
        "---", 'title: "Tuned tabular corruption-detection models"', "format:", "  html:",
        "    page-layout: full", "    toc: true", "    embed-resources: true", "---", "",
        *_hard_sample_style(), "",
        "# Validation comparison", "", *_comparison_lines(validation), "",
        "# Confusion matrices", "",
        "The top row contains ordinary counts. The bottom row discounts weak "
        "corruptions using the plotted logarithmic weighting; clean samples retain weight 1.", "",
        "![Raw and strength-weighted validation confusion matrices](confusion_matrices.png)", "",
        "![Corruption-strength weighting](strength_weighting.png)", "",
        "# Metrics by corruption strength", "",
        "![Five metrics for all evaluation views](strength.png)", "",
        "# Increased-noise controls", "",
        "All controls are true clean examples and were excluded from fitting and model selection.", "",
        *_noise_control_lines(controls), "", "## Increased-noise predicted counts", "",
        *_noise_control_count_lines(controls), "", "![Noise-control false positives](noise_controls.png)", "",
        "# Hard samples shared by every model", "", *hard_lines,
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
    "_ensure_strength_weighting",
    "_hard_sample_style",
    "_hard_sample_lines",
    "_metric_lines",
    "_plot_confusion_matrices",
    "_noise_control_count_lines",
    "_noise_control_lines",
    "_plot_noise_controls",
    "_plot_strength",
    "_plot_strength_weighting",
    "_select_hard_samples",
    "write_tabular_report",
]
