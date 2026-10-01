"""Run and incrementally report the minimal ResNet-18/DINOv2 experiment matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import traceback
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml import dinov2, resnet18
from ml.evaluate import METRICS
from ml.nn_common import (PARITY_CHANNELS, RAW_CHANNELS, RESIDUAL_RRR_CHANNELS,
                          choose_device, load_data, prepare, train_job)
from ml.report import (
    _ensure_strength_weighting,
    _hard_sample_lines,
    _hard_sample_style,
    _metric_lines,
    _noise_control_count_lines,
    _noise_control_lines,
    _plot_confusion_matrices,
    _plot_noise_controls,
    _plot_strength_weighting,
)
from scripts.reporting import QuartoReporter


SEEDS = (42, 43, 44)
FILES = ("checkpoint.pt", "config.json", "history.json", "results.json", "predictions.csv")
TRAINABLE_SCOPES = {
    ("resnet18", "head"): "head_only",
    ("resnet18", "last"): "head_and_last",
    ("resnet18", "all"): "all",
    ("dinov2", "linear"): "head_adapter",
    ("dinov2", "last"): "head_last_adapter",
}


def _experiment(backbone: str, mode: str, channels=RAW_CHANNELS, schedule="base") -> dict[str, Any]:
    if tuple(channels) == PARITY_CHANNELS:
        channel_id = "eo"
    elif tuple(channels) == RESIDUAL_RRR_CHANNELS:
        channel_id = "rrr"
    else:
        channel_id = "-".join(name[0] for name in channels)
    trainable_scope = TRAINABLE_SCOPES[(backbone, mode)]
    experiment_id = f"{backbone}__{trainable_scope}__{channel_id}__{schedule}"
    return {"experiment_id": experiment_id, "backbone": backbone, "mode": mode,
            "trainable_scope": trainable_scope, "channels": tuple(channels),
            "schedule": schedule}


def experiments(models=("resnet18", "dinov2"), phase="all") -> list[dict[str, Any]]:
    """Return the deduplicated scientific experiment matrix (before seeds)."""

    selected = []
    for backbone in models:
        modes = ("head", "last", "all") if backbone == "resnet18" else ("linear", "last")
        reference = "last" if backbone == "resnet18" else "linear"
        if phase in {"residual100", "residual_rrr100"}:
            if backbone == "resnet18":
                channels = (("residual",) if phase == "residual100" else
                            RESIDUAL_RRR_CHANNELS)
                selected += [_experiment(backbone, mode, channels, "residual100")
                             for mode in modes]
            continue
        if phase in {"all", "depth", "smoke"}:
            selected += [_experiment(backbone, mode) for mode in modes]
        if phase in {"all", "schedule"}:
            selected += [_experiment(backbone, reference, schedule=name) for name in
                         ("base", "lr-third", "lr-triple", "warmup5", "epochs20", "epochs40")]
        if phase in {"all", "channels"}:
            channel_sets = [RAW_CHANNELS, *combinations(RAW_CHANNELS, 3), *((name,) for name in RAW_CHANNELS)]
            if backbone == "resnet18":
                channel_sets.append(PARITY_CHANNELS)
            selected += [_experiment(backbone, reference, channels) for channels in channel_sets]
    return list({row["experiment_id"]: row for row in selected}.values())


def jobs(models=("resnet18", "dinov2"), phase="all") -> list[dict[str, Any]]:
    seeds = (SEEDS[0],) if phase == "smoke" else SEEDS
    return [{**experiment, "seed": seed,
             "job_id": f'{experiment["experiment_id"]}__seed{seed}'}
            for experiment in experiments(models, phase) for seed in seeds]


def _schedule(name: str) -> dict[str, Any]:
    values = {"epochs": 30, "patience": 5, "warmup_epochs": 0, "lr_scale": 1.0}
    if name == "lr-third": values["lr_scale"] = 1 / 3
    elif name == "lr-triple": values["lr_scale"] = 3.0
    elif name == "warmup5": values["warmup_epochs"] = 5
    elif name == "epochs20": values["epochs"] = 20
    elif name == "epochs40": values["epochs"] = 40
    elif name == "residual100":
        values.update(epochs=100, patience=None, warmup_epochs=5)
    elif name != "base": raise ValueError(f"Unknown schedule: {name}")
    return values


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _dataset_fingerprint(path: Path) -> str:
    """Hash the index and listed manifests, whose integrity maps identify all products."""

    index = path.read_bytes(); digest = hashlib.sha256(index)
    for reference in json.loads(index)["samples"]:
        digest.update(reference.encode()); digest.update((path.parent / reference).read_bytes())
    return digest.hexdigest()


def _complete(directory: Path, config: dict[str, Any] | None = None) -> bool:
    if not all((directory / name).is_file() for name in FILES):
        return False
    if config is None:
        return True
    try:
        return json.loads((directory / "config.json").read_text()) == config
    except (OSError, json.JSONDecodeError):
        return False


def _display_name(experiment_id: str, *, multiline: bool = False) -> str:
    separator = "\n" if multiline else " · "
    fields = experiment_id.split("__")
    if len(fields) != 4:
        return separator.join(fields)
    backbone, mode, channels, schedule = fields
    parts = [{"resnet18": "ResNet-18", "dinov2": "DINOv2"}.get(backbone, backbone), mode]
    if channels != "d-c-r-p":
        parts.append(channels)
    if schedule != "base":
        parts.append(schedule)
    return separator.join(parts)


def _mean_validation(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Average confusion counts across seeds while retaining one validation cohort."""

    mains = [result["overall"]["main"] for result in results]
    if len({main["sample_count"] for main in mains}) != 1 or \
            len({tuple(main["labels"]) for main in mains}) != 1:
        raise ValueError("Seed aggregation requires identical validation cohorts and labels")
    first = mains[0]
    raw = (first["confusion_matrix"] if len(mains) == 1 else
           np.mean([main["confusion_matrix"] for main in mains], axis=0).tolist())
    return {"overall": {"main": {
        "sample_count": first["sample_count"],
        "labels": first["labels"],
        "confusion_matrix": raw,
        "strength_weighted_confusion_matrix": np.mean(
            [main["strength_weighted_confusion_matrix"] for main in mains], axis=0
        ).tolist(),
    }}}


def _mean_noise_controls(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Average rates and predicted counts across completed seeds of one experiment."""

    levels = list(results[0]["by_noise_snr_target"])
    if any(list(result["by_noise_snr_target"]) != levels for result in results[1:]):
        raise ValueError("Seed aggregation requires identical noise-control targets")

    def average(values: list[dict[str, Any]]) -> dict[str, Any]:
        if len({value["sample_count"] for value in values}) != 1:
            raise ValueError("Seed aggregation requires identical noise-control cohorts")
        labels = list(values[0]["predicted_counts"])
        return {
            "sample_count": values[0]["sample_count"],
            "correct_rejection_rate": float(np.mean([
                value["correct_rejection_rate"] for value in values
            ])),
            "false_positive_rate": float(np.mean([
                value["false_positive_rate"] for value in values
            ])),
            "predicted_counts": {
                label: float(np.mean([value["predicted_counts"][label] for value in values]))
                for label in labels
            },
            "matched_baseline_false_positive_rate": float(np.mean([
                value["matched_baseline_false_positive_rate"] for value in values
            ])),
            "excess_false_positive_rate": float(np.mean([
                value["excess_false_positive_rate"] for value in values
            ])),
        }

    return {
        "evaluation_kind": "noise_robustness",
        "overall": average([result["overall"] for result in results]),
        "by_noise_snr_target": {
            level: average([result["by_noise_snr_target"][level] for result in results])
            for level in levels
        },
    }


def _strength_plot(
    results: dict[str, list[dict[str, Any]]], destination: Path
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    views = (("main", "Main"), ("detection", "Detection"),
             ("error_identification", "Error identification"))
    figure, axes = plt.subplots(3, 5, figsize=(16, 9), sharex=True, sharey=True,
                               constrained_layout=True)
    first = next(iter(results.values()))[0]
    levels = sorted(float(level) for level in first["by_corruption_snr_target"])
    for row, (view, title) in enumerate(views):
        for column, metric in enumerate(METRICS):
            for name, runs in results.items():
                if any(sorted(float(level) for level in run["by_corruption_snr_target"])
                       != levels for run in runs):
                    raise ValueError("Models use different corruption-strength targets")
                values = np.asarray([
                    [run["by_corruption_snr_target"][f"{level:g}"][view][metric]
                     for level in levels]
                    for run in runs
                ], dtype=float)
                counts = np.sum(np.isfinite(values), axis=0)
                mean = np.divide(np.nansum(values, axis=0), counts,
                                 out=np.full(len(levels), np.nan), where=counts > 0)
                line, = axes[row, column].plot(levels, mean, marker="o", label=name)
                if len(runs) > 1:
                    sd = np.asarray([
                        np.std(values[np.isfinite(values[:, index]), index], ddof=1)
                        if counts[index] > 1 else 0.0
                        for index in range(len(levels))
                    ])
                    axes[row, column].fill_between(
                        levels, mean - sd, mean + sd, color=line.get_color(), alpha=.15
                    )
            axes[row, column].set_ylim(-.02, 1.02); axes[row, column].grid(alpha=.25)
            if row == 0: axes[row, column].set_title(metric.upper())
            if column == 0: axes[row, column].set_ylabel(title)
            if row == 2: axes[row, column].set_xlabel("Target corruption SNR")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center",
                  ncol=min(4, len(labels)), frameon=False)
    figure.savefig(destination, dpi=160); plt.close(figure)


def _loss_plot(histories: dict[int, list[dict[str, Any]]], destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    observed_epochs = set()
    for seed, history in histories.items():
        epochs = [row["epoch"] for row in history]
        observed_epochs.update(epochs)
        axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o",
                     label=f"seed {seed}")
        axes[1].plot(epochs, [row["validation_loss"] for row in history], marker="o",
                     label=f"seed {seed}")
    for axis, title in zip(axes, ("Weighted training loss", "Weighted validation loss"), strict=True):
        axis.set(xlabel="Epoch", ylabel="Cross-entropy", title=title); axis.grid(alpha=.25)
        if len(observed_epochs) <= 10:
            axis.set_xticks(sorted(observed_epochs))
    axes[1].legend(frameon=False)
    figure.savefig(destination, dpi=160); plt.close(figure)


def write_report(output: Path, expected: list[dict[str, Any]], *, render: bool = True,
                 failures: dict[str, str] | None = None, dataset_sha: str | None = None) -> Path:
    """Regenerate the partial report from immutable completed-job artifacts."""

    output.mkdir(parents=True, exist_ok=True); figures = output / "figures"; figures.mkdir(exist_ok=True)
    complete = []
    for job in expected:
        directory = output / job["job_id"]
        if not _complete(directory):
            continue
        config = json.loads((directory / "config.json").read_text())
        if dataset_sha is None or config.get("dataset_sha256") == dataset_sha:
            complete.append(job)
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    by_job, prediction_rows, dataset_paths = {}, [], set()
    for job in complete:
        directory = output / job["job_id"]
        result = json.loads((directory / "results.json").read_text())
        by_job[job["job_id"]] = result
        grouped.setdefault(job["experiment_id"], []).append((job, result))
        config = json.loads((directory / "config.json").read_text())
        if config.get("dataset"):
            dataset_paths.add(str(Path(config["dataset"]).expanduser().resolve()))
        with (directory / "predictions.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                row["model"] = job["job_id"]
                prediction_rows.append(row)
    if by_job:
        _ensure_strength_weighting(
            {job_id: result["validation"] for job_id, result in by_job.items()},
            prediction_rows,
        )
    for runs in grouped.values():
        runs.sort(key=lambda pair: pair[0]["seed"])

    version = f"__{dataset_sha[:12]}" if dataset_sha else ""
    assets = {}
    for experiment_id, runs in grouped.items():
        loss = figures / f"{experiment_id}{version}__loss.png"
        _loss_plot({
            job["seed"]: json.loads((output / job["job_id"] / "history.json").read_text())
            for job, _ in runs
        }, loss)
        assets[experiment_id] = {"loss": loss}

    weight_plot = figures / f"strength_weighting{version}__v2.png"
    confusion_plot = figures / f"model_comparison{version}__confusion_v3.png"
    strength_plot = figures / f"model_comparison{version}__strength.png"
    noise_plot = figures / f"model_comparison{version}__noise.png"
    comparison_validations = {
        _display_name(experiment_id, multiline=True): _mean_validation(
            [result["validation"] for _, result in runs]
        )
        for experiment_id, runs in grouped.items()
    }
    comparison_controls = {
        _display_name(experiment_id): _mean_noise_controls(
            [result["noise_controls"] for _, result in runs]
        )
        for experiment_id, runs in grouped.items()
    }
    comparison_strength = {
        _display_name(experiment_id): [result["validation"] for _, result in runs]
        for experiment_id, runs in grouped.items()
    }
    if by_job:
        _plot_strength_weighting(
            {job_id: result["validation"] for job_id, result in by_job.items()}, weight_plot
        )
        multiple_seeds = any(len(runs) > 1 for runs in grouped.values())
        _plot_confusion_matrices(
            comparison_validations, confusion_plot,
            raw_label="Mean count per seed" if multiple_seeds else "Raw counts",
            raw_format=".1f" if multiple_seeds else "d",
        )
        _strength_plot(comparison_strength, strength_plot)
        _plot_noise_controls(comparison_controls, noise_plot)

    failed = len(set(failures or {}) - {job["job_id"] for job in complete})
    lines = ["---", 'title: "Neural corruption-detection experiments"',
             "format:", "  html:", "    page-layout: full", "    toc: true",
             "    embed-resources: true", "---", "", *_hard_sample_style(), "",
             "# Progress", "", f"- Complete: **{len(complete)}/{len(expected)}**",
             f"- Failed this invocation: **{failed}**",
             f"- Pending: **{len(expected) - len(complete) - failed}**", "",
             "# Validation comparison", "",
             "One row is shown for every experiment with at least one completed seed. "
             "A single seed is reported directly; multiple seeds are mean ± sample SD.", "",
             "| Experiment | Seeds | " + " | ".join(name.upper() for name in METRICS) + " |",
             "| --- | ---: | " + " | ".join("---:" for _ in METRICS) + " |"]
    summaries = {}
    for experiment_id, runs in grouped.items():
        mains = [result["validation"]["overall"]["main"] for _, result in runs]
        summaries[experiment_id] = {
            metric: np.asarray([main[metric] for main in mains
                                if main[metric] is not None], dtype=float)
            for metric in METRICS
        }
    maxima = {
        metric: max((values[metric].mean() for values in summaries.values()
                     if len(values[metric])), default=None)
        for metric in METRICS
    }
    for experiment_id, runs in grouped.items():
        cells = []
        for metric in METRICS:
            values = summaries[experiment_id][metric]
            if len(values) != len(runs):
                cells.append("—")
                continue
            rendered = (f"{values[0]:.4f}" if len(values) == 1 else
                        f"{values.mean():.4f} ± {values.std(ddof=1):.4f}")
            if len(grouped) > 1 and maxima[metric] == values.mean():
                rendered = f"**{rendered}**"
            cells.append(rendered)
        seeds = ", ".join(str(job["seed"]) for job, _ in runs)
        lines.append(f"| `{experiment_id}` | {seeds} | " + " | ".join(cells) + " |")

    lines += ["", "## Individual runs", "",
              "| Experiment | Seed | Precision | Recall | F1 | AUROC | AUPRC | "
              "Noise FPR | Best epoch | Parameters | Minutes |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for experiment_id, runs in grouped.items():
        for job, result in runs:
            main = result["validation"]["overall"]["main"]
            cells = ["—" if main[name] is None else f'{main[name]:.4f}' for name in METRICS]
            lines.append(f'| `{experiment_id}` | {job["seed"]} | ' + " | ".join(cells) +
                         f' | {result["noise_controls"]["overall"]["false_positive_rate"]:.4f} | '
                         f'{result["best_epoch"]} | {result["trainable_parameters"]:,} | '
                         f'{result["runtime_seconds"] / 60:.1f} |')

    if by_job:
        lines += ["", "# Confusion matrices", "",
                  "All model variants are columns in one comparison figure. The top row "
                  "shows ordinary counts and the bottom row shows logarithmically "
                  "strength-weighted counts. With multiple completed seeds, each cell is "
                  "the mean count per seed; clean samples retain weight 1.", "",
                  f"![Raw and strength-weighted confusion matrices]"
                  f"({confusion_plot.relative_to(output)})", "",
                  f"![Corruption-strength weighting]({weight_plot.relative_to(output)})", ""]

        lines += ["", "# Metrics by corruption strength", "",
                  "Each panel compares all model variants. Lines are per-model seed means; "
                  "shading shows ±1 sample SD when multiple seeds are complete.", "",
                  f"![All-model metrics by corruption strength]"
                  f"({strength_plot.relative_to(output)})", ""]

        lines += ["", "# Increased-noise controls", "",
                  "All controls are true clean examples and were excluded from fitting "
                  "and model selection. Each row and curve represents one model variant; "
                  "multiple completed seeds are averaged per seed. Identical curves overlap "
                  "exactly, so use the summary tables to confirm ties.", "",
                  *_noise_control_lines(comparison_controls), "",
                  "## Increased-noise predicted counts", "",
                  *_noise_control_count_lines(comparison_controls), "",
                  f"![All-model noise-control comparison]({noise_plot.relative_to(output)})", ""]

    if len(dataset_paths) == 1:
        lines += ["", "# Hard samples shared by every completed run", "",
                  *_hard_sample_lines(output, Path(next(iter(dataset_paths))), prediction_rows)]
    elif prediction_rows:
        lines += ["", "# Hard samples shared by every completed run", "",
                  "Hard-sample images are unavailable because completed jobs do not identify "
                  "one common dataset path.", ""]

    if grouped:
        lines += ["", "# Training loss by epoch", "",
                  "Curves include every currently completed seed. A smoke run has one epoch, "
                  "so each curve contains one point.", ""]
        for experiment_id in grouped:
            loss = assets[experiment_id]["loss"]
            lines += [f"## `{experiment_id}`", "",
                      f"![Training and validation loss]({loss.relative_to(output)})", ""]

        lines += ["", "# Model details", ""]
        for experiment_id, runs in grouped.items():
            for job, result in runs:
                lines += _metric_lines(
                    f"`{experiment_id}` — seed {job['seed']}", result["validation"]
                )

    if failures:
        lines += ["", "# Failures", ""] + [f"- `{job}`: {message}" for job, message in failures.items()]
    qmd = output / "report.qmd"; qmd.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if render: QuartoReporter(qmd, every=1).finish()
    return qmd


def run(args: argparse.Namespace) -> None:
    output, dataset = args.output.resolve(), args.dataset.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = jobs(tuple(args.models), args.phase)
    if args.max_jobs is not None: selected = selected[:args.max_jobs]
    expected = selected if args.phase in {"smoke", "residual100", "residual_rrr100"} else jobs(tuple(args.models), "all")
    control_index = dataset.with_name("dataset_noise_controls.json")
    fingerprints = {"main": _dataset_fingerprint(dataset),
                    "noise_controls": _dataset_fingerprint(control_index)}
    dataset_sha = hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()
    representations = set()
    for job in selected:
        if job["channels"] == RESIDUAL_RRR_CHANNELS:
            representations.add("residual_rrr")
        elif job["channels"] == PARITY_CHANNELS:
            representations.add("parity")
        else:
            representations.add("raw")
    splits, labels = load_data(dataset, sorted(representations))
    if fingerprints != {"main": _dataset_fingerprint(dataset),
                        "noise_controls": _dataset_fingerprint(control_index)}:
        raise RuntimeError("Dataset changed while it was being loaded; retry after generation stops")
    if args.clean_label not in labels: raise ValueError("--clean-label is absent from the dataset")
    device, failures, completed = choose_device(args.device), {}, 0
    state = {"device": device, "current": None, "failures": failures}
    prepared = normalization = cache_key = None
    try:
        for job in selected:
            if cache_key != job["channels"]:
                prepared, normalization = prepare(splits, job["channels"], labels)
                cache_key = job["channels"]
            schedule = _schedule(job["schedule"])
            if args.phase == "smoke": schedule.update(epochs=1, patience=1)
            config = {**job, **schedule, "channels": list(job["channels"]), "seed": job["seed"],
                      "dataset": str(dataset), "dataset_sha256": dataset_sha, "labels": list(labels),
                      "dataset_fingerprints": fingerprints,
                      "clean_label": args.clean_label, "batch_size": args.batch_size,
                      "weight_decay": 1e-4, "normalization": normalization,
                      "backbone_revision": ("IMAGENET1K_V1" if job["backbone"] == "resnet18"
                                             else dinov2.HUB_REVISION)}
            directory = output / job["job_id"]
            if _complete(directory, config):
                print(f'Skipping complete job: {job["job_id"]}', flush=True); continue
            state["current"] = job["job_id"]; _atomic_json(output / "state.json", state)
            try:
                random.seed(job["seed"]); np.random.seed(job["seed"]); torch.manual_seed(job["seed"])
                if job["backbone"] == "resnet18":
                    model, groups, train_mode = resnet18.build(len(labels), len(normalization["slots"]),
                                                               job["mode"], lr_scale=schedule["lr_scale"])
                else:
                    model, groups, train_mode = dinov2.build(len(labels), job["mode"],
                                                             lr_scale=schedule["lr_scale"])
                train_job(model, groups, train_mode, prepared, labels, args.clean_label,
                          config, directory, device)
                completed += 1; state["current"] = None; _atomic_json(output / "state.json", state)
                if completed % args.report_every == 0:
                    write_report(output, expected, failures=failures, dataset_sha=dataset_sha)
                del model
                if device == "mps": torch.mps.empty_cache()
                elif device == "cuda": torch.cuda.empty_cache()
            except Exception as exc:
                failures[job["job_id"]] = f"{type(exc).__name__}: {exc}"
                traceback.print_exc(); state["current"] = None; _atomic_json(output / "state.json", state)
    finally:
        write_report(output, expected, failures=failures, dataset_sha=dataset_sha)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("ml/runs/nn"))
    parser.add_argument("--phase", choices=("all", "depth", "schedule", "channels", "smoke",
                                             "residual100", "residual_rrr100"), default="all")
    parser.add_argument("--models", nargs="+", choices=("resnet18", "dinov2"), default=("resnet18", "dinov2"))
    parser.add_argument("--device"); parser.add_argument("--clean-label", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--report-every", type=int, default=3)
    parser.add_argument("--max-jobs", type=int)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.report_every <= 0: parser.error("batch size and report interval must be positive")
    run(args)


if __name__ == "__main__":
    main()
