"""Compare combined6 XGBoost output strategies using an existing run's data and CV folds."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from ml.logreg import ALL_FEATURES, FeatureSet
from ml.regression import (GRID, TARGETS, RegressionData, _hash, cross_talk, evaluate_data,
                           evaluation_definition, fit_model, write_json)
from ml.regression_report import write_report
from ml.regression_training import fit_type_specific, predict_outputs, training_mask

STRATEGIES = {"one_per_target": "one_output_per_tree", "shared_tree": "multi_output_tree"}


def compare_strategies(source: Path, output: Path, *, jobs: int = 4, include_type_specific=False) -> Path:
    source = source.expanduser().resolve()
    config = json.loads((source / "config.json").read_text())
    reference_name = "combined6" if "combined6" in config["features"] else "one_per_target"
    if (config.get("target_definition_version") != 2 or config["cv_grid"] != GRID
            or config["seed"] != 42 or config["quick"]
            or config["features"].get(reference_name) != list(ALL_FEATURES)
            or set(config["splits"]) != {"train", "validation"}):
        raise ValueError("Use a full combined6 run with pooled validation and the current targets, seed and CV grid")
    fingerprints = {path.name: _hash(path) for path in source.iterdir() if path.is_file()}
    with (source / "predictions.csv").open() as stream:
        source_rows = list(csv.DictReader(stream))
        original_rows = [row for row in source_rows if row["model"] == reference_name]
    lookup = {row["sample_id"]: row for row in original_rows}
    if len(lookup) != len(original_rows):
        raise ValueError("Duplicate combined6 validation predictions")
    data = {}
    for split in ("train", "validation"):
        with np.load(source / f"{split}_features.npz") as saved:
            np.testing.assert_array_equal(saved["feature_names"], ALL_FEATURES)
            ids = tuple(saved["sample_ids"].tolist())
            features = FeatureSet(saved["X"], np.zeros(len(ids)), ids,
                                  tuple(saved["source_ids"].tolist()), tuple({} for _ in ids))
            products = tuple(json.loads(lookup[sid]["products"]) for sid in ids) if split == "validation" else tuple({} for _ in ids)
            data[split] = RegressionData(features, saved["y"], saved["kinds"], saved["strengths"], products)
    train, validation = data["train"], data["validation"]
    if set(train.features.source_ids) & set(validation.features.source_ids):
        raise ValueError("Training and validation sources overlap")
    all_ids = train.features.sample_ids + validation.features.sample_ids
    if len(all_ids) != len(set(all_ids)) or set(lookup) != set(validation.features.sample_ids):
        raise ValueError("Sample IDs are duplicated or saved predictions do not match validation")
    selected = [lookup[sid] for sid in validation.features.sample_ids]
    original_truth = np.array([[float(row[f"true_{target}"]) for target in TARGETS] for row in selected])
    original_prediction = np.array([[float(row[f"pred_{target}"]) for target in TARGETS] for row in selected])
    np.testing.assert_array_equal(original_truth, validation.targets)
    groups = np.asarray(train.features.source_ids)
    folds = []
    validation_counts = np.zeros(len(groups), dtype=int)
    for fold in config["cv_folds"]:
        a = np.flatnonzero(np.isin(groups, fold["train_sources"]))
        b = np.flatnonzero(np.isin(groups, fold["validation_sources"]))
        if (not len(a) or not len(b) or set(groups[a]) & set(groups[b])
                or len(a) + len(b) != len(groups)):
            raise ValueError("Invalid saved CV source split")
        validation_counts[b] += 1
        folds.append((a, b))
    np.testing.assert_array_equal(validation_counts, np.ones(len(groups), dtype=int))
    strategies = dict(STRATEGIES)
    if include_type_specific:
        strategies.update(one_per_type="one_output_per_tree", shared_by_type="multi_output_tree")
    suffix = "_combined6_training" if include_type_specific else "_combined6_strategies"
    run = output.expanduser().resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + suffix)
    run.mkdir(parents=True, exist_ok=False)
    print(f"Run: {run}", flush=True)
    for split in data:
        shutil.copy2(source / f"{split}_features.npz", run / f"{split}_features.npz")
    shutil.copy2(source / "config.json", run / "source_config.json")
    for key in ("report_source_run", "source_artifact_sha256", "report_code_sha256",
                "training_rows_per_output", "training_description", "model_files"):
        config.pop(key, None)
    config.update(experiment="combined6 output strategies", source_run=str(source),
                  source_artifact_sha256=fingerprints, jobs=jobs,
                  features={name: list(ALL_FEATURES) for name in strategies},
                  model_strategies=strategies, model_files={}, best_params={}, evaluation_definition=evaluation_definition(),
                  python=platform.python_version(),
                  versions={name: version(name) for name in config["versions"]},
                  code_sha256={str(path.resolve()): _hash(path) for path in
                               (Path(__file__), Path(__file__).with_name("regression.py"),
                                Path(__file__).with_name("regression_report.py"),
                                Path(__file__).with_name("regression_training.py"))})
    if include_type_specific:
        mask = training_mask(train.kinds)
        config.update(experiment="combined6 output strategies and training populations",
                      training_rows_per_output={name: {
                          target: int(mask[:, j].sum()) if name in ("one_per_type", "shared_by_type") else len(train.targets)
                          for j, target in enumerate(TARGETS)} for name in strategies},
                      training_description=(
                          "one_per_target and shared_tree train every output on all training samples. "
                          "one_per_type fits each output on clean samples and its own fault type. "
                          "shared_by_type shares tree splits and applies squared-error loss only to clean samples "
                          "and each output's own fault type; unrelated outputs have zero gradient and Hessian. "
                          "Its shared splits and feature bins use the union of training samples. "
                          "All setups use the same source CV folds and score every output on all CV validation rows, "
                          "with scales computed from all training-fold targets."))
    write_json(run / "config.json", config)
    predictions, results, matrices, cv_rows, rows = {}, {}, {}, [], []
    for name, strategy in strategies.items():
        print(f"Fitting {name}: {strategy}, {len(folds)} saved source folds, 4 parameter combinations…", flush=True)
        if name in ("one_per_type", "shared_by_type"):
            model, best, candidates = fit_type_specific(train.features.X, train.targets, train.kinds,
                                                        folds, jobs=jobs, multi_strategy=strategy)
        else:
            model, best, candidates = fit_model(train.features.X, train.targets, folds, jobs=jobs, multi_strategy=strategy)
        estimators = model if isinstance(model, list) else [model]
        filenames = [f"{name}_{target}.json" for target in TARGETS] if isinstance(model, list) else [f"{name}.json"]
        config["model_files"][name] = filenames
        restored = []
        for estimator, filename in zip(estimators, filenames, strict=True):
            estimator.save_model(run / filename)
            reloaded = XGBRegressor()
            reloaded.load_model(run / filename)
            restored.append(reloaded)
        raw = predict_outputs(model, validation.features.X).astype(float)
        if not np.isfinite(raw).all():
            raise ValueError(f"Nonfinite predictions from {name}")
        restored_model = restored if isinstance(model, list) else restored[0]
        np.testing.assert_array_equal(raw, predict_outputs(restored_model, validation.features.X))
        prediction = np.maximum(0, raw)
        if name == "one_per_target":
            np.testing.assert_array_equal(prediction, original_prediction)
        elif name == "shared_tree":
            previous = {row["sample_id"]: row for row in source_rows if row["model"] == name}
            if previous:
                expected = np.array([[float(previous[sid][f"pred_{target}"]) for target in TARGETS]
                                     for sid in validation.features.sample_ids])
                np.testing.assert_array_equal(prediction, expected)
        predictions[name] = prediction
        results[name] = evaluate_data(validation, prediction)
        matrices[name] = cross_talk(validation.targets, prediction, validation.kinds)
        config["best_params"][name] = best
        cv_rows.extend({"model": name, **row} for row in candidates)
        for i, sid in enumerate(validation.features.sample_ids):
            row = {"model": name, "split": "validation", "sample_id": sid,
                   "source_id": validation.features.source_ids[i], "kind": validation.kinds[i],
                   "requested_snr": float(validation.strengths[i]), "products": json.dumps(validation.products[i])}
            for j, target in enumerate(TARGETS):
                row.update({f"true_{target}": float(validation.targets[i, j]),
                            f"pred_{target}": float(prediction[i, j]), f"raw_{target}": float(raw[i, j])})
            rows.append(row)
        print(f"  Selected {best}; Total MAE={results[name]['overall']['total']['mae']:.6f}", flush=True)
    for name, target in (("train_mean", train.targets.mean(0)), ("no_error", np.zeros(3))):
        results[name] = evaluate_data(validation, np.broadcast_to(target, validation.targets.shape))
    with (run / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(run / "config.json", config)
    write_json(run / "cv_results.json", cv_rows)
    write_json(run / "results.json", {"evaluation": {"validation": results}, "cross_talk": matrices})
    write_report(run, data, {"validation": predictions}, {"validation": results}, matrices, config)
    for filename, digest in fingerprints.items():
        if _hash(source / filename) != digest:
            raise RuntimeError(f"Source artifact changed: {filename}")
    print(f"Report: {run / 'report.html'}", flush=True)
    return run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("ml/runs"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--include-type-specific", action="store_true",
                        help="Also fit clean-plus-own-type separate models and masked shared trees")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if not shutil.which("quarto"):
        parser.error("Quarto must be installed to render the report")
    compare_strategies(args.from_run, args.output, jobs=args.jobs, include_type_specific=args.include_type_specific)


if __name__ == "__main__":
    main()
