"""Run and incrementally report the minimal ResNet-18/DINOv2 experiment matrix."""

from __future__ import annotations

import argparse
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
from ml.nn_common import PARITY_CHANNELS, RAW_CHANNELS, choose_device, load_data, prepare, train_job
from ml.report import _plot_confusion_matrices
from scripts.reporting import QuartoReporter


SEEDS = (42, 43, 44)
FILES = ("checkpoint.pt", "config.json", "history.json", "results.json", "predictions.csv")


def _experiment(backbone: str, mode: str, channels=RAW_CHANNELS, schedule="base") -> dict[str, Any]:
    channel_id = "eo" if tuple(channels) == PARITY_CHANNELS else "-".join(name[0] for name in channels)
    experiment_id = f"{backbone}__{mode}__{channel_id}__{schedule}"
    return {"experiment_id": experiment_id, "backbone": backbone, "mode": mode,
            "channels": tuple(channels), "schedule": schedule}


def experiments(models=("resnet18", "dinov2"), phase="all") -> list[dict[str, Any]]:
    """Return the deduplicated scientific experiment matrix (before seeds)."""

    selected = []
    for backbone in models:
        modes = ("head", "last", "all") if backbone == "resnet18" else ("linear", "last")
        reference = "last" if backbone == "resnet18" else "linear"
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


def _strength_plot(results: list[dict[str, Any]], destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    views = (("main", "Main"), ("detection", "Detection"),
             ("error_identification", "Error identification"))
    figure, axes = plt.subplots(3, 5, figsize=(16, 9), sharex=True, sharey=True,
                               constrained_layout=True)
    levels = sorted(float(level) for level in results[0]["by_corruption_snr_target"])
    for row, (view, title) in enumerate(views):
        for column, metric in enumerate(METRICS):
            values = np.asarray([[run["by_corruption_snr_target"][f"{level:g}"][view][metric]
                                  for level in levels] for run in results], dtype=float)
            mean = np.nanmean(values, axis=0)
            sd = np.nanstd(values, axis=0, ddof=1) if len(values) > 1 else np.zeros(len(levels))
            axes[row, column].plot(levels, mean, marker="o")
            axes[row, column].fill_between(levels, mean - sd, mean + sd, alpha=.2)
            axes[row, column].set_ylim(-.02, 1.02); axes[row, column].grid(alpha=.25)
            if row == 0: axes[row, column].set_title(metric.upper())
            if column == 0: axes[row, column].set_ylabel(title)
            if row == 2: axes[row, column].set_xlabel("Target corruption SNR")
    figure.savefig(destination, dpi=160); plt.close(figure)


def _loss_plot(histories: list[list[dict[str, Any]]], destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for seed, history in zip(SEEDS, histories, strict=True):
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["train_loss"] for row in history], label=f"seed {seed}")
        axes[1].plot(epochs, [row["validation_loss"] for row in history], label=f"seed {seed}")
    for axis, title in zip(axes, ("Weighted training loss", "Validation loss"), strict=True):
        axis.set(xlabel="Epoch", ylabel="Cross-entropy", title=title); axis.grid(alpha=.25)
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
    for job in complete:
        result = json.loads((output / job["job_id"] / "results.json").read_text())
        grouped.setdefault(job["experiment_id"], []).append((job, result))
    failed = len(set(failures or {}) - {job["job_id"] for job in complete})
    lines = ["---", 'title: "Neural corruption-detection experiments"',
             "format:", "  html:", "    toc: true", "    embed-resources: true", "---", "",
             "# Progress", "", f"- Complete: **{len(complete)}/{len(expected)}**",
             f"- Failed this invocation: **{failed}**",
             f"- Pending: **{len(expected) - len(complete) - failed}**", "", "# Validation runs", "",
             "| Experiment | Seed | Precision | Recall | F1 | AUROC | AUPRC | Best epoch | Parameters | Minutes |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for experiment_id, runs in grouped.items():
        for job, result in sorted(runs, key=lambda pair: pair[0]["seed"]):
            main = result["validation"]["overall"]["main"]
            cells = ["—" if main[name] is None else f'{main[name]:.4f}' for name in METRICS]
            lines.append(f'| `{experiment_id}` | {job["seed"]} | ' + " | ".join(cells) +
                         f' | {result["best_epoch"]} | {result["trainable_parameters"]:,} | '
                         f'{result["runtime_seconds"] / 60:.1f} |')
    lines += ["", "# Three-seed comparisons", "",
              "| Experiment | " + " | ".join(name.upper() for name in METRICS) + " |",
              "| --- | " + " | ".join("---:" for _ in METRICS) + " |"]
    details = []
    for experiment_id, runs in grouped.items():
        if len(runs) != 3:
            continue
        runs.sort(key=lambda pair: pair[0]["seed"])
        mains = [result["validation"]["overall"]["main"] for _, result in runs]
        cells = []
        for metric in METRICS:
            values = np.asarray([main[metric] for main in mains if main[metric] is not None])
            cells.append("—" if len(values) != 3 else f"{values.mean():.4f} ± {values.std(ddof=1):.4f}")
        lines.append(f"| `{experiment_id}` | " + " | ".join(cells) + " |")
        version = f"__{dataset_sha[:12]}" if dataset_sha else ""
        strength, confusion, loss = (figures / f"{experiment_id}{version}__{kind}.png"
                                     for kind in ("strength", "confusion", "loss"))
        if not strength.exists():
            _strength_plot([result["validation"] for _, result in runs], strength)
            _plot_confusion_matrices({f'seed {job["seed"]}': result["validation"]
                                      for job, result in runs}, confusion)
            _loss_plot([json.loads((output / job["job_id"] / "history.json").read_text())
                        for job, _ in runs], loss)
        details += [f"## `{experiment_id}`", "", f"![Loss]({loss.relative_to(output)})", "",
                    f"![Strength metrics]({strength.relative_to(output)})", "",
                    f"![Confusion matrices]({confusion.relative_to(output)})", ""]
    if failures:
        lines += ["", "# Failures", ""] + [f"- `{job}`: {message}" for job, message in failures.items()]
    lines += ["", "# Completed comparisons", "", *details]
    qmd = output / "report.qmd"; qmd.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if render: QuartoReporter(qmd, every=1).finish()
    return qmd


def run(args: argparse.Namespace) -> None:
    output, dataset = args.output.resolve(), args.dataset.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = jobs(tuple(args.models), args.phase)
    if args.max_jobs is not None: selected = selected[:args.max_jobs]
    expected = selected if args.phase == "smoke" else jobs(tuple(args.models), "all")
    dataset_sha = _dataset_fingerprint(dataset)
    splits, labels = load_data(dataset)
    if _dataset_fingerprint(dataset) != dataset_sha:
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
    parser.add_argument("--phase", choices=("all", "depth", "schedule", "channels", "smoke"), default="all")
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
