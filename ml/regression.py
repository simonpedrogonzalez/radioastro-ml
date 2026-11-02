"""Source-grouped V1 regression of amplitude, phase and extra-noise SNR_corr."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.model_selection import ParameterGrid

from ml.logreg import ALL_FEATURES, FEATURE_VIEWS, FeatureSet, features_from_dataloader, grouped_folds
from scripts.preprocessing import FitsSimulationDataset, make_simulation_dataloader

TARGETS = ("amp", "phase", "noise")
METRICS = ("mae", "rmse", "r2")
GRID = {"n_estimators": [100, 300], "max_depth": [2, 4], "learning_rate": [.05]}


def _positive(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def target_from_metadata(metadata: dict) -> tuple[list[float], str, float]:
    """Use measured perturbation SNR for every output; requested SNR defines bins."""
    simulation = metadata["simulation"]
    _positive(simulation["noise"]["simplenoise_jy"], "baseline noise sigma")
    reports = metadata["corruptions"]
    control = simulation.get("noise_control")
    if control is not None:
        if reports or control.get("kind") != "increased_noise":
            raise ValueError("Unsupported noise/gain mixture or noise-control kind")
        reference = _positive(control.get("reference_sigma_jy"), "reference sigma")
        total = _positive(control.get("final_total_sigma_jy"), "total sigma")
        if total < reference:
            raise ValueError("Increased-noise total sigma is below its reference")
        measured = _positive(control.get("SNR_corr_measured"), "measured extra-noise SNR")
        return [0., 0., measured], "noise", _positive(control.get("SNR_corr_target"), "requested noise SNR")
    if not reports:
        return [0., 0., 0.], "clean", 0.
    if len(reports) != 1:
        raise ValueError("V1 regression requires one pure gain injection")
    report = reports[0]
    kind = report["solution"]["corruption_type"]
    if kind not in ("amp", "phase"):
        raise ValueError(f"Unsupported gain kind: {kind}")
    target = [0., 0., 0.]
    target[0 if kind == "amp" else 1] = _positive(report["metrics"].get("SNR_corr"), "measured gain SNR")
    strength = _positive(report["solution"].get("SNR_corr_target"), "requested gain SNR")
    return target, kind, strength


@dataclass(frozen=True)
class RegressionData:
    features: FeatureSet
    targets: np.ndarray
    kinds: np.ndarray
    strengths: np.ndarray
    products: tuple[dict[str, str], ...]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_noise_training_index(index: Path, training_sources: set[str]) -> None:
    from scripts.preprocessing import load_dataset_manifest, load_sample_manifest, partition_for_sample, source_dataset_id

    manifest = load_dataset_manifest(index, require_samples=False)
    if not manifest.samples or manifest.labels != {"not_corrupted": 0}:
        raise ValueError("Training-noise index must be nonempty with not_corrupted: 0 labels")
    seen = set()
    for path in manifest.samples:
        sample = load_sample_manifest(path, require_files=False, verify_integrity=False)
        if (partition_for_sample(sample.sample_id) != "train"
                or source_dataset_id(sample.sample_id) not in training_sources):
            raise ValueError(f"Noise sample is outside main training sources: {sample.sample_id}")
        if sample.sample_id in seen or sample.corruptions or (sample.label_name, sample.label_id) != ("not_corrupted", 0):
            raise ValueError(f"Duplicate or invalid training-noise sample: {sample.sample_id}")
        simulation = json.loads(sample.simulation.read_text())
        _, kind, _ = target_from_metadata({"simulation": simulation, "corruptions": []})
        if kind != "noise":
            raise ValueError(f"Training-noise index contains an ordinary sample: {sample.sample_id}")
        seen.add(sample.sample_id)


def append_training_noise(main: RegressionData, extra: RegressionData) -> RegressionData:
    if np.any(extra.kinds != "noise") or not set(extra.features.source_ids) <= set(main.features.source_ids):
        raise ValueError("Extra data must contain noise samples from main training sources")
    return concatenate_data(main, extra)


def concatenate_data(*datasets: RegressionData) -> RegressionData:
    ids = tuple(sid for data in datasets for sid in data.features.sample_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample IDs in combined data")
    features = FeatureSet(np.concatenate([data.features.X for data in datasets]),
                          np.concatenate([data.features.y for data in datasets]), ids,
                          tuple(sid for data in datasets for sid in data.features.source_ids),
                          tuple(meta for data in datasets for meta in data.features.label_metadata))
    return RegressionData(features, np.concatenate([data.targets for data in datasets]),
                          np.concatenate([data.kinds for data in datasets]),
                          np.concatenate([data.strengths for data in datasets]),
                          tuple(product for data in datasets for product in data.products))


def load_data(index: Path, partition: str) -> tuple[RegressionData, dict[str, str]]:
    dataset = FitsSimulationDataset(index.parent, index=index.name, partition=partition,
                                    label_criterion="constant_antenna_type")
    # Only selected manifests and target/QA metadata enter this fingerprint.
    # The dataset validator checks their image products against manifest hashes.
    inputs = {str(index): _hash(index)}
    for sample in dataset.samples:
        paths = [sample.path, sample.simulation, sample.imaging_qa]
        paths += [reference.corruption for reference in sample.corruptions]
        inputs.update({str(path): _hash(path) for path in paths})
    targets, kinds, strengths, products = [], [], [], []

    def batches():
        loader = make_simulation_dataloader(dataset, batch_size=32, shuffle=False)
        for batch in loader:
            for sample_id, metadata, paths in zip(batch["sample_id"], batch["metadata"], batch["paths"], strict=True):
                try:
                    target, kind, strength = target_from_metadata(metadata)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid regression target for {sample_id}: {exc}") from exc
                targets.append(target); kinds.append(kind); strengths.append(strength)
                products.append({name: str(path) for name, path in paths["products"].items()})
            yield batch

    features = features_from_dataloader(batches())
    return RegressionData(features, np.asarray(targets), np.asarray(kinds),
                          np.asarray(strengths), tuple(products)), inputs


def metric_values(truth, prediction, *, include_r2=True) -> dict:
    truth, prediction = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    if not len(truth):
        return dict.fromkeys(("mae", "mse", "rmse", "r2") if include_r2 else ("mae", "mse", "rmse"))
    error = prediction - truth
    mse = float(np.mean(error ** 2))
    result = {"mae": float(np.mean(np.abs(error))), "mse": mse, "rmse": float(np.sqrt(mse))}
    if not include_r2:
        return result
    if truth.ndim == 2:
        # Average per-component R²; MAE/MSE above average samples and components.
        scores = [metric_values(truth[:, j], prediction[:, j])["r2"] for j in range(truth.shape[1])]
        defined = [value for value in scores if value is not None]
        result["r2"] = float(np.mean(defined)) if defined else None
        return result
    ss = float(np.sum((truth - truth.mean()) ** 2))
    # Fixed-SNR samples differ slightly after finite-precision injection.
    # Treat that numerical scatter as constant truth when reporting R².
    constant = np.allclose(truth, truth[0], rtol=1e-7, atol=1e-12)
    result["r2"] = None if len(truth) < 2 or constant else float(1 - np.sum(error ** 2) / ss)
    return result


def evaluate(truth: np.ndarray, prediction: np.ndarray, *, include_r2=True) -> dict:
    prediction = np.maximum(0, prediction)
    total, predicted_total = truth.sum(1), prediction.sum(1)
    eligible = total > 0
    covered = eligible & (predicted_total > 0)
    return {
        "main": {target: metric_values(truth[:, j], prediction[:, j], include_r2=include_r2)
                 for j, target in enumerate(TARGETS)},
        "total": metric_values(total, predicted_total, include_r2=include_r2),
        "mixture": metric_values(truth[covered] / total[covered, None],
                                 prediction[covered] / predicted_total[covered, None], include_r2=include_r2),
        "coverage": {"eligible": int(eligible.sum()), "covered": int(covered.sum()),
                     "missed": int((eligible & ~covered).sum())},
    }


def evaluate_data(data: RegressionData, prediction: np.ndarray) -> dict:
    def section(mask, *, include_r2=True):
        return {"n": int(mask.sum()), "sources": len(set(np.asarray(data.features.source_ids)[mask])),
                **evaluate(data.targets[mask], prediction[mask], include_r2=include_r2)}

    bins = []
    for strength in sorted(set(data.strengths) - {0.}):
        # Pool all fault kinds at this level; reuse clean samples for absolute errors.
        mask = (data.strengths == strength) | (data.kinds == "clean")
        injected = data.strengths == strength
        relative = evaluate(data.targets[injected] / strength, prediction[injected] / strength, include_r2=False)
        bins.append({"strength": float(strength), **section(mask, include_r2=False),
                     "relative": {"n": int(injected.sum()), **relative}})
    return {"overall": section(np.ones(len(data.targets), dtype=bool)), "by_strength": bins}


def describe_data(data: RegressionData) -> dict:
    return {"n": len(data.targets), "sources": sorted(set(data.features.source_ids)),
            "kind_counts": {kind: int(np.sum(data.kinds == kind)) for kind in sorted(set(data.kinds))},
            "strength_counts": {kind: {str(float(level)): int(np.sum((data.kinds == kind) & (data.strengths == level)))
                                       for level in sorted(set(data.strengths[data.kinds == kind]))}
                                for kind in sorted(set(data.kinds))},
            "target_min": data.targets.min(0).tolist(), "target_max": data.targets.max(0).tolist()}


def evaluation_definition() -> dict:
    return {"version": 3, "population": "All held-out clean, amp, phase and extra-noise samples pooled",
            "total": "amp + phase + noise; diagnostic sum, not a combined visibility norm",
            "mixture": "Three proportions: each component / Total; positive true and predicted Total only",
            "mixture_metrics": "MAE/MSE average samples and all three components; RMSE=sqrt(MSE); R2 averages defined component R2",
            "relative_errors": "Divide by requested injected SNR; exclude clean; mixture already normalized",
            "per_strength_r2": False}


def cross_talk(truth: np.ndarray, prediction: np.ndarray, kinds: np.ndarray) -> list[dict]:
    truth, prediction = truth.copy(), np.maximum(0, prediction.copy())
    rows = []
    for kind, column in (("amp", 0), ("phase", 1), ("noise", 2), ("clean", None)):
        mask = kinds == kind
        n = int(mask.sum())
        denominator = float(n if column is None else truth[mask, column].sum())
        sums = prediction[mask].sum(0)
        rows.append({"kind": kind, "n": n, "denominator": denominator, "sums": sums.tolist(),
                     "values": (sums / denominator).tolist() if denominator > 0 else [None] * 3})
    return rows


def fit_model(X, y, folds, *, jobs: int, quick: bool = False, multi_strategy="one_output_per_tree"):
    from xgboost import XGBRegressor

    def estimator(params):
        return XGBRegressor(objective="reg:squarederror", tree_method="hist",
                            multi_strategy=multi_strategy, random_state=42, n_jobs=jobs, **params)

    rows = []
    if quick:
        best = {"n_estimators": 20, "max_depth": 2, "learning_rate": .05}
    else:
        for params in ParameterGrid(GRID):
            scores = []
            for train, validation in folds:
                scale = y[train].std(0)
                active = scale > 0
                if not active.any():
                    raise ValueError("All training-fold regression targets are constant")
                model = estimator(params).fit(X[train], y[train])
                prediction = np.maximum(0, model.predict(X[validation]))
                scores.append(float(np.mean(np.abs(prediction[:, active] - y[validation][:, active]) / scale[active])))
            rows.append({"params": params, "fold_scores": scores, "mean_scaled_mae": float(np.mean(scores))})
        best = min(rows, key=lambda row: row["mean_scaled_mae"])["params"]
    return estimator(best).fit(X, y), best, rows


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--noise-training-index", type=Path, help="Explicit train-only extra-noise index")
    parser.add_argument("--output", type=Path, default=Path("ml/runs"), help="Parent for a new run directory")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--quick", action="store_true", help="20 trees; skip hyperparameter search")
    args = parser.parse_args(argv)
    if args.jobs < 1 or args.cv_folds < 2:
        parser.error("--jobs must be positive and --cv-folds must be at least 2")
    if not shutil.which("quarto"):
        parser.error("Quarto must be installed and on PATH to render the report")
    import torch

    torch.set_num_threads(args.jobs)
    index = args.dataset.expanduser().resolve()
    data, inputs = {}, {}
    for split, path, partition in (("train", index, "train"), ("validation", index, "val"),
                                   ("noise_controls", index.with_name("dataset_noise_controls.json"), "val")):
        if split == "noise_controls" and not path.exists():
            print("No noise-control index; reporting main validation only.", flush=True)
            continue
        print(f"Loading {split} features and targets…", flush=True)
        data[split], fingerprints = load_data(path, partition)
        inputs.update(fingerprints)
        print(f"  {len(data[split].targets)} samples / {len(set(data[split].features.source_ids))} sources", flush=True)
    if np.any(data["train"].kinds == "noise"):
        raise ValueError("Main index must retain its original baseline/gain training population")
    noise_training_index = args.noise_training_index.expanduser().resolve() if args.noise_training_index else None
    if noise_training_index is not None:
        validate_noise_training_index(noise_training_index, set(data["train"].features.source_ids))
        print("Loading explicit training-noise index…", flush=True)
        extra, fingerprints = load_data(noise_training_index, "train")
        inputs.update(fingerprints)
        data["train"] = append_training_noise(data["train"], extra)
        print(f"  {len(extra.targets)} added noise samples; {len(data['train'].targets)} total training samples", flush=True)
    train = data["train"]
    for split, values in data.items():
        if split != "train" and set(train.features.source_ids) & set(values.features.source_ids):
            raise ValueError(f"Training sources overlap {split}")
    all_ids = [sample for values in data.values() for sample in values.features.sample_ids]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Duplicate sample IDs across partitions/indices")
    if "noise_controls" in data and np.any(data["noise_controls"].kinds != "noise"):
        raise ValueError("Noise-control index contains ordinary samples")
    if "noise_controls" in data:
        data["validation"] = concatenate_data(data["validation"], data.pop("noise_controls"))
    folds = grouped_folds(train.features, args.cv_folds)
    run = args.output.expanduser().resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_multi_regression")
    run.mkdir(parents=True, exist_ok=False)
    config = {
        "dataset": str(index), "seed": 42, "quick": args.quick, "jobs": args.jobs,
        "target_definition_version": 2,
        "targets": ["measured amplitude SNR_corr", "measured phase SNR_corr", "measured extra-noise SNR_corr"],
        "noise_training_index": str(noise_training_index) if noise_training_index else None,
        "clean_target": [0, 0, 0], "clipping": "max(0, prediction) for every target",
        "r2_constant_tolerance": {"rtol": 1e-7, "atol": 1e-12},
        "evaluation_definition": evaluation_definition(),
        "features": {name: list(names) for name, names in FEATURE_VIEWS.items()},
        "feature_definitions": "QA and parity definitions reused without changes from ml/logreg.py",
        "noise_provisional": bool(np.all(train.targets[:, 2] == 0)),
        "cv_scoring": "Mean MAE / training-fold target std, nonconstant targets only; minimize",
        "cv_grid": None if args.quick else GRID,
        "versions": {name: version(name) for name in ("numpy", "scikit-learn", "xgboost", "torch", "astropy")},
        "python": platform.python_version(),
        "splits": {split: describe_data(value) for split, value in data.items()},
        "cv_folds": [{"train_sources": sorted(set(np.asarray(train.features.source_ids)[a])),
                      "validation_sources": sorted(set(np.asarray(train.features.source_ids)[b]))} for a, b in folds],
        "input_sha256": inputs,
        "code_sha256": {str(path): _hash(path) for path in (Path(__file__), Path(__file__).with_name("logreg.py"),
                                                          Path(__file__).with_name("regression_report.py"))},
        "best_params": {},
    }
    write_json(run / "config.json", config)
    for split, values in data.items():
        np.savez_compressed(run / f"{split}_features.npz", X=values.features.X, y=values.targets,
                            sample_ids=values.features.sample_ids, source_ids=values.features.source_ids,
                            kinds=values.kinds, strengths=values.strengths, feature_names=ALL_FEATURES)
    evaluation = {split: {} for split in data if split != "train"}
    predictions = {split: {} for split in evaluation}
    rows, cv_rows = [], []
    for name, names in FEATURE_VIEWS.items():
        print(f"Fitting {name}{' (smoke fit)' if args.quick else ' with grouped CV'}…", flush=True)
        model, best, candidates = fit_model(train.features.view(names), train.targets, folds, jobs=args.jobs, quick=args.quick)
        model.save_model(run / f"{name}.json")
        config["best_params"][name] = best
        cv_rows.extend({"model": name, **row} for row in candidates)
        for split in evaluation:
            values = data[split]
            raw = model.predict(values.features.view(names)).astype(float)
            if not np.isfinite(raw).all():
                raise ValueError(f"Nonfinite predictions from {name}")
            prediction = np.maximum(0, raw)
            predictions[split][name] = prediction
            evaluation[split][name] = evaluate_data(values, prediction)
            for i, sample_id in enumerate(values.features.sample_ids):
                row = {"model": name, "split": split, "sample_id": sample_id, "source_id": values.features.source_ids[i],
                       "kind": values.kinds[i], "requested_snr": float(values.strengths[i]),
                       "products": json.dumps(values.products[i])}
                for j, target in enumerate(TARGETS):
                    row.update({f"true_{target}": float(values.targets[i, j]), f"pred_{target}": float(prediction[i, j]),
                                f"raw_{target}": float(raw[i, j])})
                rows.append(row)
    for split in evaluation:
        for name, target in (("train_mean", np.mean(train.targets, axis=0)), ("no_error", np.array([0., 0., 0.]))):
            prediction = np.broadcast_to(target, data[split].targets.shape)
            evaluation[split][name] = evaluate_data(data[split], prediction)
    combined_truth = np.concatenate([data[split].targets for split in evaluation])
    combined_kinds = np.concatenate([data[split].kinds for split in evaluation])
    matrices = {name: cross_talk(combined_truth, np.concatenate([predictions[split][name] for split in evaluation]),
                                 combined_kinds) for name in FEATURE_VIEWS}
    for path, digest in inputs.items():
        if _hash(Path(path)) != digest:
            raise RuntimeError(f"Input changed during run: {path}")
    write_json(run / "config.json", config)
    write_json(run / "cv_results.json", cv_rows)
    write_json(run / "results.json", {"evaluation": evaluation, "cross_talk": matrices})
    with (run / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    from ml.regression_report import write_report

    write_report(run, data, predictions, evaluation, matrices, config)
    print(f"Report: {run / 'report.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
