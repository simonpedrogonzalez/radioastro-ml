"""Run the planned channel, parity-feature and head-only comparisons; train/val only."""

import argparse
import csv
import hashlib
import json
import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ml import cnn
from ml.logreg import FEATURE_NAMES, features_from_dataloader, train as train_logreg
from ml.report import (
    _comparison_lines,
    _metric_lines,
    _plot_confusion_matrices,
    _plot_strength,
)
from scripts.preprocessing import source_dataset_id
from scripts.reporting import QuartoReporter


def prediction_rows(split_name, split, probabilities):
    return [dict(sample_id=s, source_id=source_dataset_id(s), split=split_name,
                 true_class=int(split.batch["label"][i]), predicted_class=int(probabilities[i].argmax()),
                 **{f"probability_{label}": float(probabilities[i, j])
                    for j, label in enumerate(("none", "amp", "phase"))},
                 **split.batch["label_metadata"][i])
            for i, s in enumerate(split.batch["sample_id"])]


def read_run(path, splits, dataset_hash, mode, seed, head_only=False):
    config = json.loads((path / "config.json").read_text())
    if (config["dataset_index_sha256"] != dataset_hash or config["input_mode"] != mode
            or config["seed"] != seed or config["preprocessing"] != cnn.PREPROCESSING
            or config["training"].get("head_only", False) != head_only
            or config["weights"] != "ResNet18_Weights.IMAGENET1K_V1"
            or config["training"]["warmup_epochs"] != 5
            or config["training"]["finetune_max_epochs"] != 15):
        raise ValueError(f"Incompatible reference/run configuration: {path}")
    with (path / "predictions.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    expected = {(name, s): (int(split.batch["label"][i]), split.batch["label_metadata"][i]["corruption_snr_target"])
                for name, split in splits.items() for i, s in enumerate(split.batch["sample_id"])}
    actual = {(r["split"], r["sample_id"]): (int(r["true_class"]), float(r["corruption_snr_target"])) for r in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError(f"Sample/label/level mismatch with current train/val cohort: {path}")
    by_key = {(row["split"], row["sample_id"]): row for row in rows}
    probabilities = np.asarray([
        [float(by_key[("val", sample_id)][f"probability_{label}"])
         for label in ("none", "amp", "phase")]
        for sample_id in splits["val"].batch["sample_id"]
    ])
    validation = cnn.score(splits["val"], probabilities)
    history = json.loads((path / "history.json").read_text())
    return dict(group="parity_head_only" if head_only else f"{mode}_finetuned", seed=seed,
                path=str(path.resolve()), validation=validation,
                training_seconds=sum(h["seconds"] for h in history),
                trainable_parameters=max(h["trainable_parameters"] for h in history))


def scalar_baselines(splits, output):
    qa = {name: features_from_dataloader([split.batch], cnn.FULL_FEATURE_NAMES)
          for name, split in splits.items()}
    four = [cnn.FULL_FEATURE_NAMES.index(f) for f in FEATURE_NAMES]
    matrices = {name: {
        "parity_logreg": split.parity_features,
        "qa4_parity_logreg": np.column_stack((qa[name].X[:, four], split.parity_features)),
        "qa4_logreg": qa[name].X[:, four], "qa14_logreg": qa[name].X,
    } for name, split in splits.items()}
    feature_names = {
        "parity_logreg": cnn.PARITY_FEATURE_NAMES,
        "qa4_parity_logreg": FEATURE_NAMES + cnn.PARITY_FEATURE_NAMES,
        "qa4_logreg": FEATURE_NAMES, "qa14_logreg": cnn.FULL_FEATURE_NAMES,
    }
    entries = []
    for name, features in feature_names.items():
        path = output / name
        path.mkdir()
        start = time.perf_counter()
        model = train_logreg(matrices["train"][name], qa["train"].y)
        seconds = time.perf_counter() - start
        results, rows = {}, []
        for partition, split in splits.items():
            probabilities = model.predict_proba(matrices[partition][name])
            results[partition] = cnn.score(split, probabilities)
            rows.extend(prediction_rows(partition, split, probabilities))
        with (path / "model.pkl").open("wb") as stream:
            pickle.dump(model, stream)
        with (path / "predictions.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        np.savez(path / "features.npz", train=matrices["train"][name], val=matrices["val"][name],
                 train_ids=qa["train"].sample_ids, val_ids=qa["val"].sample_ids,
                 train_labels=qa["train"].y, val_labels=qa["val"].y)
        (path / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        (path / "config.json").write_text(json.dumps({"features": features,
            "model": "StandardScaler -> LogisticRegression(C=1, class_weight=balanced, solver=lbfgs, max_iter=1000)",
            "parity_features": "log1p(mean((E/s)^2)), log1p(mean((O/s)^2)); native [1:,1:]; before clipping/asinh/resize",
            "scale": cnn.PREPROCESSING["scale"]}, indent=2) + "\n")
        entries.append(dict(group=name, seed=None, path=str(path.resolve()), validation=results["val"],
                            training_seconds=seconds, trainable_parameters=int(model[-1].coef_.size + model[-1].intercept_.size)))
        main = results["val"]["overall"]["main"]
        print(f"{name}: val macro recall={main['recall']:.4f}, F1={main['f1']:.4f}", flush=True)
    return entries


def source_comparisons(entries):
    """Paired source bootstrap, conditional on the saved models and chosen validation set."""
    distributions, sources, draws = {}, None, None
    for entry in entries:
        with (Path(entry["path"]) / "predictions.csv").open() as stream:
            rows = [r for r in csv.DictReader(stream) if r["split"] == "val"]
        current_sources = sorted({r["source_id"] for r in rows})
        if sources is None:
            sources = current_sources
            draws = np.random.default_rng(2026).integers(0, len(sources), size=(5000, len(sources)))
        if sources != current_sources:
            raise ValueError("Source bootstrap requires identical validation sources")
        counts = np.zeros((len(sources), 2, 3))
        for row in rows:
            i, truth = sources.index(row["source_id"]), int(row["true_class"])
            counts[i, 0, truth] += 1
            counts[i, 1, truth] += int(row["predicted_class"]) == truth
        observed = np.mean(counts[:, 1].sum(0) / counts[:, 0].sum(0))
        reported = entry["validation"]["overall"]["main"]["recall"]
        if not np.isclose(observed, reported):
            raise ValueError("Saved predictions disagree with reported macro recall")
        sampled = counts[draws].sum(axis=1)
        distributions.setdefault(entry["group"], []).append(np.mean(sampled[:, 1] / sampled[:, 0], axis=1))
    comparisons = []
    for a, b in (("parity_finetuned", "residual_finetuned"), ("parity_finetuned", "parity_head_only"),
                 ("parity_logreg", "parity_finetuned"), ("qa4_parity_logreg", "parity_finetuned"),
                 ("qa4_parity_logreg", "parity_logreg")):
        delta = np.mean(distributions[a], axis=0) - np.mean(distributions[b], axis=0)
        mean = lambda g: np.mean([
            e["validation"]["overall"]["main"]["recall"]
            for e in entries if e["group"] == g
        ])
        comparisons.append(dict(comparison=f"{a} minus {b}", difference=float(mean(a)-mean(b)),
                                source_bootstrap_95_interval=np.quantile(delta, [.025, .975]).tolist()))
    return comparisons


def write_summary(output, entries, provenance):
    comparisons = source_comparisons(entries)
    (output / "summary.json").write_text(json.dumps({"provenance": provenance, "experiments": entries,
        "source_comparisons": comparisons, "bootstrap": {"seed": 2026, "replicates": 5000, "unit": "validation source"}}, indent=2) + "\n")
    run_results = {
        entry["group"] + (f" / seed {entry['seed']}" if entry["seed"] is not None else ""):
        entry["validation"] for entry in entries
    }
    _plot_strength(run_results, output / "severity.png")
    _plot_confusion_matrices(run_results, output / "confusion_matrices.png")
    tr, va = provenance["cohorts"]["train"], provenance["cohorts"]["val"]
    lines = ["---", 'title: "CNN ablations: square sources"', "format:", "  html:", "    embed-resources: true", "---", "",
             f"Train: {tr['retained_samples']} samples / {tr['retained_sources']} sources. "
             f"Validation: {va['retained_samples']} samples / {va['retained_sources']} sources. Test not evaluated. "
             "All results use the same eligible samples, labels and target levels; circular sources remain excluded.", "",
             "## Comparison", "", "Each trained seed is shown separately; bold values are tied column maxima.", "",
             *_comparison_lines(run_results), "",
             "![Confusion matrices](confusion_matrices.png)", "",
             "![Metrics by corruption strength](severity.png)", "",
              "## Paired source comparisons", "",
              "Macro-recall differences with descriptive 95% percentile intervals from 5,000 paired source resamples (seed2026). "
              "Each source keeps all variants; seed results are averaged within each draw. Models/seeds are fixed. "
              "These intervals do not account for validation-based model selection or new training data.", "",
              "| Comparison | Difference | Source-bootstrap interval |", "| --- | ---: | --- |",
              *[f"| {c['comparison']} | {c['difference']:.4f} | [{c['source_bootstrap_95_interval'][0]:.4f}, {c['source_bootstrap_95_interval'][1]:.4f}] |" for c in comparisons], "",
              "## Interpretation limits", "", "The head-only control uses the same two-stage schedule, head learning rates, optimizer reset and early stopping as fine-tuning. "
              "Parity statistics use raw native even/odd energies, not simulation truth. "
              "Training times exclude preload and final reporting; CNN epoch time includes validation. "
              "These validation results do not establish performance on new sources or circular images, or that ImageNet pretraining is necessary.", "",
              "## Every run", ""]
    for entry in entries:
        name = entry["group"] + (f" / seed {entry['seed']}" if entry["seed"] is not None else "")
        lines += _metric_lines(name, entry["validation"])
        path = Path(entry["path"])
        artifact = path / ("report.html" if (path / "report.html").exists() else "results.json")
        lines += [f"[Run artifacts]({os.path.relpath(artifact, output)}); "
                  f"trainable parameters: {entry['trainable_parameters']:,}; training seconds: {entry['training_seconds']:.2f}.", ""]
    lines += ["## Provenance", "", "```json", json.dumps(provenance, indent=2), "```", ""]
    qmd = output / "report.qmd"
    qmd.write_text("\n".join(lines))
    QuartoReporter(qmd, every=1).finish()
    if not qmd.with_suffix(".html").is_file():
        raise RuntimeError(f"Summary HTML rendering failed: {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-parity", type=Path, required=True, help="Completed seed42 parity run")
    parser.add_argument("--reference-residual", type=Path, required=True, help="Completed seed42 residual run")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "runs")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    args = parser.parse_args(argv)
    dataset = args.dataset.resolve()
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_ablations")
    output.mkdir(parents=True, exist_ok=False)
    provenance = {"dataset": str(dataset), "dataset_index_sha256": digest, "preprocessing": cnn.PREPROCESSING,
                  "seeds": [42, 43, 44], "test_evaluated": False,
                  "references": [str(args.reference_parity.resolve()), str(args.reference_residual.resolve())]}
    entries = []
    completed = 0
    print(f"Ablation outputs: {output}", flush=True)
    for mode, reference in (("parity", args.reference_parity), ("residual", args.reference_residual)):
        splits = {s: cnn.preload(dataset, s, mode) for s in ("train", "val")}
        entries.append(read_run(reference, splits, digest, mode, 42))
        if mode == "parity":
            provenance["cohorts"] = {s: split.audit for s, split in splits.items()}
            entries.extend(scalar_baselines(splits, output))
        jobs = [(seed, False) for seed in (43, 44)]
        if mode == "parity":
            jobs += [(seed, True) for seed in (42, 43, 44)]
        for seed, head_only in jobs:
            destination = output / f"{mode}_{'head_only' if head_only else 'finetuned'}_seed{seed}"
            print(f"Experiment {completed+1}/7: {destination.name}", flush=True)
            command = ["--dataset", str(dataset), "--input-mode", mode, "--seed", str(seed), "--output", str(destination)]
            if head_only:
                command.append("--head-only")
            if args.device:
                command += ["--device", args.device]
            cnn.main(command, splits=splits)
            path, = destination.glob("*/results.json")
            entries.append(read_run(path.parent, splits, digest, mode, seed, head_only))
            completed += 1
            (output / "summary.json").write_text(json.dumps({"provenance": provenance, "experiments": entries}, indent=2) + "\n")
        del splits
    write_summary(output, entries, provenance)
    print(output / "report.html", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
