"""Quarto report for V1 regression experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from urllib.parse import quote

import numpy as np

from ml.logreg import ALL_FEATURES, FEATURE_VIEWS, FeatureSet
from ml.regression import (TARGETS, RegressionData, concatenate_data, cross_talk,
                           describe_data, evaluate_data, evaluation_definition, write_json)
from scripts.reporting import QuartoReporter

# Match the three XGBoost rows in the existing six-model classification plots.
COLORS = dict(zip(FEATURE_VIEWS, ("C3", "C4", "C5"), strict=True))
KIND_COLORS = {"amp": "#0072B2", "phase": "#D55E00", "noise": "#CC79A7", "clean": "#009E73"}


def fmt(value):
    if value is None:
        return "—"
    if value != 0 and abs(value) < .001:
        return f"{value:.2e}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def comparison(results):
    columns = [(target, metric) for target in TARGETS for metric in ("mae", "rmse")]
    lines = ["| Model | " + " | ".join(f"{target} {metric.upper()}" for target, metric in columns) + " |",
             "| --- | " + " | ".join(["---:"] * len(columns)) + " |"]
    best = {}
    for target, metric in columns:
        values = [result["overall"]["main"][target][metric] for result in results.values()]
        defined = [value for value in values if value is not None]
        best[target, metric] = min(defined) if defined else None
    for name, result in results.items():
        cells = []
        for target, metric in columns:
            value = result["overall"]["main"][target][metric]
            text = fmt(value)
            cells.append(f"**{text}**" if value is not None and value == best[target, metric] else text)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def _save(plt, figure, run, name):
    figure.savefig(run / name, dpi=150)
    plt.close(figure)


def plots(run, data, predictions, evaluation, matrices, *, median_prediction_lines=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    model_names = list(next(iter(predictions.values())))
    colors = {name: COLORS.get(name, f"C{i + 3}") for i, name in enumerate(model_names)}
    for split, results in evaluation.items():
        for view in ("main", "total_mixture"):
            row_names = TARGETS if view == "main" else ("total", "mixture")
            columns = (("mae", False), ("rmse", False), ("mae", True), ("rmse", True))
            fig, axes = plt.subplots(len(row_names), 4, figsize=(15, 2.8 * len(row_names)), layout="constrained", squeeze=False)
            for row, key in enumerate(row_names):
                for col, (metric, relative) in enumerate(columns):
                    axis = axes[row, col]
                    has_values = False
                    for name in model_names:
                        bins = results[name]["by_strength"]
                        sections = [b["relative"] if relative else b for b in bins]
                        values = [(b["main"][key] if view == "main" else b[key])[metric] for b in sections]
                        has_values |= any(value is not None for value in values)
                        axis.plot([b["strength"] for b in bins],
                                  [np.nan if v is None else v * (100 if relative else 1) for v in values],
                                  marker="o", markersize=4, color=colors[name], label=name)
                    if not has_values:
                        axis.text(.5, .5, "No eligible samples", ha="center",
                                  va="center", transform=axis.transAxes, fontsize=10)
                        if bins:
                            levels = [b["strength"] for b in bins]
                            axis.set_xlim(min(levels) - 1, max(levels) + 1)
                    if row == 0:
                        axis.set_title(f"Relative {metric.upper()} (%)" if relative else metric.upper())
                    axis.set_xlabel("Injected SNR_corr")
                    axis.set_ylim(bottom=0)
                    axis.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
                    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
                    if col == 0:
                        label = {"total": "Detection\nTotal", "mixture": "Identification\nMixture"}.get(key, key.capitalize())
                        axis.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=18,
                                        fontsize=12, fontweight="bold")
                    axis.grid(alpha=.25)
                    if row == col == 0:
                        axis.legend(fontsize=8)
            _save(plt, fig, run, f"{split}_{view}.png")
    fig, axes = plt.subplots(len(model_names), 3, figsize=(12, 10 * len(model_names) / 3),
                             layout="constrained", squeeze=False)
    markers = {"amp": "o", "phase": "^", "noise": "x", "clean": "s"}
    for row, name in enumerate(model_names):
        for col, target in enumerate(TARGETS):
            axis = axes[row, col]
            all_true, all_pred = [], []
            for split in evaluation:
                values = data[split]
                for kind, marker in markers.items():
                    mask = values.kinds == kind
                    if mask.any():
                        axis.scatter(values.targets[mask, col], predictions[split][name][mask, col],
                                     marker=marker, color=KIND_COLORS[kind], alpha=.6, s=14, label=kind)
                all_true.extend(values.targets[:, col]); all_pred.extend(predictions[split][name][:, col])
            if median_prediction_lines:
                # Each injected level contains many sources with slightly different
                # measured targets. Group by requested level and take both medians.
                lines = []
                for split in evaluation:
                    values = data[split]
                    mask = values.kinds == target
                    for strength in sorted(set(values.strengths[mask])):
                        selected = mask & (values.strengths == strength)
                        lines.append((float(np.median(values.targets[selected, col])),
                                      float(np.median(predictions[split][name][selected, col]))))
                if lines:
                    axis.plot(*np.asarray(lines).T, color="black", marker="o", markersize=3,
                              linewidth=1.4, label="Median by injected level")
            low, high = min(all_true + all_pred), max(all_true + all_pred)
            padding = max((high - low) * .03, 1e-7)
            axis.plot([low - padding, high + padding], [low - padding, high + padding], "k--", lw=.8)
            axis.ticklabel_format(useOffset=False)
            axis.set(xlabel="True", ylabel="Predicted")
            if row == 0:
                axis.set_title(target.capitalize())
            if col == 0:
                axis.set_ylabel(f"{name}\n\nPredicted", rotation=0, ha="right", va="center",
                                labelpad=15, fontsize=12, fontweight="bold")
            if row == col == 0:
                axis.legend(fontsize=8)
    _save(plt, fig, run, "scatter.png")
    fig, axes = plt.subplots(1, len(matrices), figsize=(4 * len(matrices), 4),
                             layout="constrained", squeeze=False)
    axes = axes[0]
    maximum = max([v for rows in matrices.values() for row in rows for v in row["values"] if v is not None] + [1.])
    for axis, (name, rows) in zip(axes, matrices.items(), strict=True):
        values = np.array([[np.nan if v is None else v for v in row["values"]] for row in rows])
        display = axis.imshow(values, vmin=0, vmax=maximum, cmap="Blues", aspect="auto")
        axis.set(title=name, xticks=range(3), xticklabels=("amp", "phase", "noise"),
                 yticks=range(4), yticklabels=("amp", "phase", "noise", "clean mean"))
        for i, row in enumerate(rows):
            for j, value in enumerate(row["values"]):
                axis.text(j, i, fmt(value), ha="center", va="center", fontsize=8,
                          color="white" if value is not None and value > maximum / 2 else "black")
    fig.colorbar(display, ax=axes, shrink=.8, label="Predicted / true SNR; clean row is a mean SNR")
    _save(plt, fig, run, "cross_talk.png")


def hard_samples(run, data, predictions):
    model_names = list(next(iter(predictions.values())))
    records = []
    for split, models in predictions.items():
        values = data[split]
        for i, truth in enumerate(values.targets):
            pred = np.array([models[name][i] for name in model_names])
            errors = np.abs(pred - truth)
            off_kind = np.maximum(0, pred.copy())
            active = np.array([truth[0] > 0, truth[1] > 0, truth[2] > 0])
            off_kind[:, active] = 0
            records.append({"split": split, "i": i, "errors": errors, "prediction": pred,
                            "total": float(errors.sum(1).max()), "leakage": float(off_kind.sum(1).max())})
    lines = ["## Hard samples", "", "Ranked by the largest total absolute error among the models. "
             "Vectors are (amp, phase, noise); cells show prediction and |error|.", ""]
    def section(title, subset, score):
        chosen = sorted(subset, key=lambda row: row[score], reverse=True)[:6]
        lines.extend([f"### {title}", "", "| Sample / source | True | "
                      + " | ".join(f"{name} pred; error" for name in model_names) + " | Products |",
                      "| " + " | ".join(["---"] * (len(model_names) + 3)) + " |"])
        for row in chosen:
            values = data[row["split"]]; i = row["i"]
            vector = lambda v: "(" + ", ".join(fmt(x) for x in v) + ")"
            cells = [f"{vector(p)};<br>{vector(e)}" for p, e in zip(row["prediction"], row["errors"], strict=True)]
            links = [f'<a href="{quote(os.path.relpath(path, run))}">{escape(name)}</a>'
                     for name, path in values.products[i].items()]
            sample = escape(values.features.sample_ids[i]); source = escape(values.features.source_ids[i])
            lines.append(f"| {sample}<br>source: {source}<br>{row['split']} | {vector(values.targets[i])} | "
                         + " | ".join(cells) + " | " + " · ".join(links) + " |")
        sources = Counter(data[row["split"]].features.source_ids[row["i"]] for row in chosen)
        repeated = [f"{name} ({count})" for name, count in sources.items() if count > 1]
        lines.extend(["", "Repeated source variants in this table: " + (", ".join(repeated) or "none") + ".", ""])
    section("Lower strengths (injected SNR ≤ 20), including clean", [r for r in records if data[r["split"]].strengths[r["i"]] <= 20], "total")
    section("Largest off-kind predictions (all validation samples)", records, "leakage")
    return lines


def write_report(run, data, predictions, evaluation, matrices, config):
    model_features = config.get("features", FEATURE_VIEWS)
    model_names = list(model_features)
    plots(run, data, predictions, evaluation, matrices, median_prediction_lines=config.get("median_prediction_lines", False))
    lines = ["---", 'title: "V1 multi-output regression"', "format:", "  html:", "    embed-resources: true",
             "    toc: true", "    page-layout: full", "---", "",
             "Targets: **amp, phase and extra-noise SNR_corr**. Clean = **(0, 0, 0)**; ordinary thermal noise remains present.", ""]
    if config["quick"]:
        lines.extend(["**Smoke run:** 20 trees, fixed parameters, no hyperparameter selection.", ""])
    if config["noise_provisional"]:
        lines.extend(["**Noise output is provisional:** training extra-noise targets are all zero. "
                      "This run has no training examples of extra-noise variation.", ""])
    exclusion = config.get("noise_strength_exclusion")
    if exclusion and exclusion.get("at_or_above") is not None:
        lines.extend([f"Noise samples with injected SNR_corr ≥ {fmt(exclusion['at_or_above'])} are excluded from both training and validation. Other sample types remain included.", ""])
    lines.extend(["## Data and fitting", "", "| Partition | Samples | Sources |", "| --- | ---: | ---: |"])
    for split, values in data.items():
        lines.append(f"| {split} | {len(values.targets)} | {len(set(values.features.source_ids))} |")
    lines.extend(["", "Each model predicts all three fault strengths. Settings are selected by cross-validation within training sources; "
                  "final evaluation pools clean, amp, phase and noise samples from unseen sources. Reserved test data are unused.", "",
                  "| Model | Features | Selected parameters |", "| --- | ---: | --- |"])
    for name, features in model_features.items():
        params = ", ".join(f"{key}={value}" for key, value in config["best_params"][name].items())
        if name in config.get("model_strategies", {}):
            params += f", multi_strategy={config['model_strategies'][name]}"
        lines.append(f"| {name} | {len(features)} | {params} |")
    if config.get("training_rows_per_output"):
        lines.extend(["", "| Model | Amp training rows | Phase training rows | Noise training rows |",
                      "| --- | ---: | ---: | ---: |"])
        for name, counts in config["training_rows_per_output"].items():
            lines.append(f"| {name} | " + " | ".join(str(counts[target]) for target in TARGETS) + " |")
        lines.extend(["", config["training_description"], ""])
    lines.extend(["", "Setup: [config.json](config.json). Scores: [results.json](results.json); "
                  "[CV candidates](cv_results.json); [predictions](predictions.csv).", ""])
    for split, results in evaluation.items():
        first = results[model_names[0]]["overall"]
        lines.extend(["## All types", "",
                      f"{first['n']} samples from {first['sources']} sources. "
                      "Predictions are clipped at zero. "
                      "Baselines predict the training means or (0,0,0).", "",
                      *comparison(results), "", "### Detection / ID", "",
                      "**Total = amp + phase + noise. Mixture = (amp, phase, noise) / Total.** "
                      "Total metrics use all samples. Mixture errors average over the three proportions "
                      "and samples with positive true and predicted Total.", "",
                      "| Model | Total MAE | Total MSE | Total RMSE | Mixture MAE | Mixture MSE | Mixture RMSE |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
        columns = [(view, metric) for view in ("total", "mixture")
                   for metric in ("mae", "mse", "rmse")]
        best = {}
        for view, metric in columns:
            values = [result["overall"][view][metric] for result in results.values()]
            defined = [value for value in values if value is not None]
            best[view, metric] = min(defined) if defined else None
        for name, result in results.items():
            cells = []
            for view, metric in columns:
                value = result["overall"][view][metric]
                text = fmt(value)
                cells.append(f"**{text}**" if value is not None and value == best[view, metric] else text)
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.extend(["", "### Per corruption strength", "",
                      "Each level pools the available fault types; levels above 100 contain noise faults only. Absolute metrics also include the same clean samples at each level. "
                      "Relative amp/phase/noise and Total errors divide by injected SNR, excluding clean samples, and are shown as percentages. "
                      "Mixture is already normalized: its relative columns show the same errors as percentages.", "",
                      f"![]({split}_main.png)", "", f"![]({split}_total_mixture.png)", "",
                      "| Injected SNR | Absolute samples | Relative samples | Sources | "
                      + " | ".join(f"{name} mixture coverage" for name in model_names) + " |",
                      "| ---: | ---: | ---: | ---: | " + " | ".join(["---"] * len(model_names)) + " |"])
        for i, group in enumerate(results[model_names[0]]["by_strength"]):
            cells = []
            for name in model_names:
                c = results[name]["by_strength"][i]["coverage"]
                cells.append(f"{c['covered']}/{c['eligible']}; missed {c['missed']}")
            lines.append(f"| {fmt(group['strength'])} | {group['n']} | {group['relative']['n']} | {group['sources']} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.extend(["## Predicted versus true", "", "Every validation sample appears in each output panel. Color and shape = true fault type; dashed line = perfect prediction. Solid black line joins median true and predicted values at each injected strength.", "", "![](scatter.png)", "",
                  "## Cross-talk", "", "All validation fault types are included. "
                  "Each pure-kind row divides summed predictions by summed true strength of that kind. "
                  "The clean row shows mean predictions. All plots share one linear scale. "
                  "Ratios compare recovered perturbation strengths; sums across outputs are not combined visibility norms.", "",
                  "![](cross_talk.png)", "", "| Model | True kind | N | Denominator | Sum amp | Sum phase | Sum noise |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for name, rows in matrices.items():
        for row in rows:
            lines.append(f"| {name} | {row['kind']} | {row['n']} | {fmt(row['denominator'])} | " + " | ".join(fmt(v) for v in row["sums"]) + " |")
    lines.extend(["", *hard_samples(run, data, predictions)])
    lines.extend(["## Training coverage", "", "| Kind | Injected SNR | Samples | Sources |", "| --- | ---: | ---: | ---: |"])
    training = data["train"]
    for kind in sorted(set(training.kinds)):
        for level in sorted(set(training.strengths[training.kinds == kind])):
            mask = (training.kinds == kind) & (training.strengths == level)
            sources = len(set(np.asarray(training.features.source_ids)[mask]))
            lines.append(f"| {kind} | {fmt(level)} | {int(mask.sum())} | {sources} |")
    qmd = run / "report.qmd"
    qmd.write_text("\n".join(lines) + "\n")
    QuartoReporter(qmd, every=1).finish()
    if not qmd.with_suffix(".html").is_file():
        raise RuntimeError(f"Quarto did not produce {qmd.with_suffix('.html')}; saved results can still be inspected")


def refresh_report(source: Path, output: Path) -> Path:
    """Re-evaluate saved predictions into a new report without fitting or imaging."""
    source = source.expanduser().resolve()
    config = json.loads((source / "config.json").read_text())
    if config.get("target_definition_version") != 2:
        raise ValueError("Report refresh requires saved extra-noise SNR_corr targets")
    model_names = list(config["features"])
    model_files = [filename for files in config.get("model_files", {
        name: [f"{name}.json"] for name in model_names}).values() for filename in files]
    splits = [split for split in ("train", "validation", "noise_controls") if split in config["splits"]]
    artifacts = ["config.json", "cv_results.json", "predictions.csv"]
    artifacts += [f"{split}_features.npz" for split in splits] + model_files
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    fingerprints = {filename: digest(source / filename) for filename in artifacts}
    with (source / "predictions.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    lookup = {(row["split"], row["model"], row["sample_id"]): row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate saved prediction rows")
    data, predictions = {}, {}
    for split in splits:
        with np.load(source / f"{split}_features.npz") as saved:
            sample_ids = tuple(saved["sample_ids"].tolist())
            features = FeatureSet(saved["X"], np.zeros(len(sample_ids)), sample_ids,
                                  tuple(saved["source_ids"].tolist()), tuple({} for _ in sample_ids))
            products = tuple(json.loads(lookup[split, model_names[0], sid]["products"]) for sid in sample_ids) if split != "train" else tuple({} for _ in sample_ids)
            data[split] = RegressionData(features, saved["y"], saved["kinds"], saved["strengths"], products)
        if split == "train":
            continue
        predictions[split] = {}
        for name in model_names:
            selected = [lookup[split, name, sid] for sid in sample_ids]
            truth = np.array([[float(row[f"true_{target}"]) for target in TARGETS] for row in selected])
            np.testing.assert_array_equal(truth, data[split].targets)
            predictions[split][name] = np.array([[float(row[f"pred_{target}"]) for target in TARGETS] for row in selected])
            raw = np.array([[float(row[f"raw_{target}"]) for target in TARGETS] for row in selected])
            np.testing.assert_array_equal(predictions[split][name], np.maximum(0, raw))
    held_out = list(predictions)
    validation = concatenate_data(*(data[split] for split in held_out))
    pooled = {name: np.concatenate([predictions[split][name] for split in held_out]) for name in model_names}
    data = {"train": data["train"], "validation": validation}
    if set(data["train"].features.source_ids) & set(validation.features.source_ids):
        raise ValueError("Training and validation sources overlap")
    expected_rows = len(validation.targets) * len(model_names)
    if len(rows) != expected_rows:
        raise ValueError("Saved predictions do not match the validation population")
    results = {name: evaluate_data(validation, pred) for name, pred in pooled.items()}
    for name, target in (("train_mean", np.mean(data["train"].targets, axis=0)), ("no_error", np.zeros(3))):
        results[name] = evaluate_data(validation, np.broadcast_to(target, validation.targets.shape))
    matrices = {name: cross_talk(validation.targets, pred, validation.kinds) for name, pred in pooled.items()}
    run = output.expanduser().resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_multi_regression_report")
    run.mkdir(parents=True, exist_ok=False)
    for filename in ["cv_results.json", "train_features.npz"] + model_files:
        shutil.copy2(source / filename, run / filename)
    shutil.copy2(source / "config.json", run / "fit_config.json")
    np.savez_compressed(run / "validation_features.npz", X=validation.features.X, y=validation.targets,
                        sample_ids=validation.features.sample_ids, source_ids=validation.features.source_ids,
                        kinds=validation.kinds, strengths=validation.strengths, feature_names=ALL_FEATURES)
    for row in rows:
        row["split"] = "validation"
    with (run / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    config.update(evaluation_definition=evaluation_definition(), report_source_run=str(source),
                  source_artifact_sha256=fingerprints, splits={split: describe_data(values) for split, values in data.items()},
                  report_code_sha256={str(path): digest(path) for path in (Path(__file__), Path(__file__).with_name("regression.py"))})
    write_json(run / "config.json", config)
    write_json(run / "results.json", {"evaluation": {"validation": results}, "cross_talk": matrices})
    write_report(run, data, {"validation": pooled}, {"validation": results}, matrices, config)
    for filename, previous in fingerprints.items():
        if digest(source / filename) != previous:
            raise RuntimeError(f"Source artifact changed during report refresh: {filename}")
    print(f"Report: {run / 'report.html'}", flush=True)
    return run


def main(argv=None):
    parser = argparse.ArgumentParser(description="Refresh regression evaluation and plots from saved predictions; preserve the source report.")
    parser.add_argument("--from-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("ml/runs"))
    args = parser.parse_args(argv)
    refresh_report(args.from_run, args.output)


if __name__ == "__main__":
    main()
