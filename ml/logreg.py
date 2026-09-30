"""Tune logistic-regression and XGBoost models on QA and parity features."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import platform
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from ml.nn_common import parity_channels
from ml.report import write_tabular_report
from ml.task_evaluation import evaluate_noise_robustness, evaluate_task
from scripts.preprocessing import FitsSimulationDataset, make_simulation_dataloader, source_dataset_id


QA_FEATURES = (
    "metrics.dynamic_range_scaled_mad",
    "metrics.residual.scaled_mad_jy_per_beam",
    "metrics.residual.peak_over_scaled_mad",
    "metrics.residual.p99_5_over_scaled_mad",
)
PARITY_FEATURES = ("log1p_even_energy_over_mad2", "log1p_odd_energy_over_mad2")
ALL_FEATURES = QA_FEATURES + PARITY_FEATURES
FEATURE_VIEWS = {"qa4": QA_FEATURES, "parity2": PARITY_FEATURES, "combined6": ALL_FEATURES}
LOGREG_GRID = {
    "logisticregression__C": [10.0**power for power in range(-4, 5)],
    "logisticregression__l1_ratio": [0.0, 1.0],
    "logisticregression__class_weight": [None, "balanced"],
}
XGBOOST_GRID = {
    "n_estimators": [100, 300], "max_depth": [2, 4],
    "learning_rate": [.03, .1], "min_child_weight": [1, 5],
    "subsample": [.8, 1.0], "colsample_bytree": [.8, 1.0], "reg_lambda": [1, 10],
}
_y, _x = np.indices((256, 256))
_ANNULUS = (np.hypot(_x - 128, _y - 128) >= 32) & (np.hypot(_x - 128, _y - 128) < 72)


@dataclass(frozen=True)
class FeatureSet:
    X: np.ndarray
    y: np.ndarray
    sample_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    label_metadata: tuple[dict[str, Any], ...]

    def view(self, names: tuple[str, ...]) -> np.ndarray:
        return self.X[:, [ALL_FEATURES.index(name) for name in names]]


def _nested(value: Any, keys: tuple[str, ...], *, context: str) -> Any:
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"{context} is missing")
        value = value[key]
    return value


def parity_features(image) -> tuple[float, float]:
    residual = image[2]
    values = residual.numpy()[_ANNULUS]
    scale = float(1.4826 * np.median(np.abs(values - np.median(values))))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Residual annulus has nonpositive/nonfinite MAD scale")
    components = parity_channels(residual)[:, 1:, 1:].double() / scale
    return tuple(float(value) for value in np.log1p(components.square().mean((-2, -1)).numpy()))


def features_from_dataloader(dataloader: Any) -> FeatureSet:
    vectors, labels, sample_ids, source_ids, metadata_rows = [], [], [], [], []
    for batch in dataloader:
        for image, sample_id, label, qa, metadata in zip(
            batch["image"], batch["sample_id"], np.asarray(batch["label"]).tolist(),
            batch["qa"], batch["label_metadata"], strict=True,
        ):
            if not isinstance(qa, dict):
                raise ValueError(f"Sample {sample_id!r} has no imaging QA metadata")
            values = []
            for name in QA_FEATURES:
                keys = tuple(name.split("."))[1:]
                if _nested(qa.get("metric_validity"), keys, context=f"{sample_id!r} validity for {name}") is not True:
                    raise ValueError(f"Sample {sample_id!r} has invalid metric {name}")
                value = _nested(qa.get("metrics"), keys, context=f"{sample_id!r} metric {name}")
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"Sample {sample_id!r} has nonfinite metric {name}")
                values.append(float(value))
            vectors.append([*values, *parity_features(image)])
            labels.append(int(label)); sample_ids.append(sample_id)
            source_ids.append(source_dataset_id(sample_id)); metadata_rows.append(dict(metadata))
    if not vectors:
        raise ValueError("Cannot extract features from an empty dataloader")
    return FeatureSet(np.asarray(vectors), np.asarray(labels), tuple(sample_ids),
                      tuple(source_ids), tuple(metadata_rows))


def _loader(dataset_path: Path, partition: str, index: str = "dataset.json"):
    dataset = FitsSimulationDataset(dataset_path.parent, index=index, partition=partition,
                                    label_criterion="constant_antenna_type")
    return make_simulation_dataloader(dataset, batch_size=32, shuffle=False)


def _fingerprint(index_path: Path) -> str:
    raw = index_path.read_bytes(); digest = hashlib.sha256(raw)
    for reference in json.loads(raw)["samples"]:
        digest.update(reference.encode()); digest.update((index_path.parent / reference).read_bytes())
    return digest.hexdigest()


def grouped_folds(features: FeatureSet, n_splits: int = 5):
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = list(splitter.split(features.X, features.y, features.source_ids))
    for train, validation in folds:
        if set(np.asarray(features.source_ids)[train]) & set(np.asarray(features.source_ids)[validation]):
            raise AssertionError("A source crossed grouped CV folds")
    return folds


def _best_index(results: dict[str, Any]) -> int:
    return max(range(len(results["params"])), key=lambda index: (
        results["mean_test_f1"][index], results["mean_test_recall"][index],
        -results["std_test_f1"][index],
        -results["params"][index].get("logisticregression__C", 0), -index,
    ))


def _estimator(algorithm: str):
    if algorithm == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(
            solver="saga", max_iter=10_000, random_state=42,
        )), LOGREG_GRID
    if algorithm == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(objective="multi:softprob", tree_method="hist",
                             eval_metric="mlogloss", random_state=42, n_jobs=1), XGBOOST_GRID
    raise ValueError(f"Unknown algorithm: {algorithm}")


def _ensure_xgboost_runtime() -> None:
    """Restart the CLI with a user-local macOS OpenMP runtime when necessary."""

    try:
        import xgboost  # noqa: F401
        return
    except Exception:
        candidates = (
            Path("/opt/homebrew/opt/libomp/lib"), Path("/usr/local/opt/libomp/lib"),
            Path.home() / "homebrew/opt/libomp/lib",
        )
        library = next((path for path in candidates if (path / "libomp.dylib").is_file()), None)
        if sys.platform != "darwin" or library is None or os.environ.get("DYLD_LIBRARY_PATH"):
            raise
        environment = dict(os.environ); environment["DYLD_LIBRARY_PATH"] = str(library)
        os.execve(sys.executable, [sys.executable, "-m", "ml.logreg", *sys.argv[1:]], environment)


def tune(algorithm: str, X: np.ndarray, y: np.ndarray, folds, *, n_jobs: int = -1,
         grid: dict[str, list[Any]] | None = None):
    estimator, default_grid = _estimator(algorithm)
    search = GridSearchCV(estimator, grid or default_grid, scoring={"f1": "f1_macro",
                          "recall": "recall_macro"}, refit=_best_index, cv=folds,
                          n_jobs=n_jobs, error_score="raise", return_train_score=False)
    fit_args = {"sample_weight": compute_sample_weight("balanced", y)} if algorithm == "xgboost" else {}
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        search.fit(X, y, **fit_args)
    index = search.best_index_
    summary = {
        "best_params": search.best_params_,
        "mean_f1": float(search.cv_results_["mean_test_f1"][index]),
        "std_f1": float(search.cv_results_["std_test_f1"][index]),
        "mean_recall": float(search.cv_results_["mean_test_recall"][index]),
        "std_recall": float(search.cv_results_["std_test_recall"][index]),
    }
    rows = [{"parameters": json.dumps(params, sort_keys=True),
             "mean_f1": float(search.cv_results_["mean_test_f1"][i]),
             "std_f1": float(search.cv_results_["std_test_f1"][i]),
             "mean_recall": float(search.cv_results_["mean_test_recall"][i]),
             "std_recall": float(search.cv_results_["std_test_recall"][i])}
            for i, params in enumerate(search.cv_results_["params"])]
    return search.best_estimator_, summary, rows


def _prediction_rows(model_name: str, split_name: str, features: FeatureSet,
                     probabilities: np.ndarray, labels: tuple[int, ...]) -> list[dict[str, Any]]:
    predicted = np.asarray(labels)[probabilities.argmax(1)]
    rows = []
    for index, metadata in enumerate(features.label_metadata):
        row = {"model": model_name, "sample_id": features.sample_ids[index],
               "source_id": features.source_ids[index], "split": split_name,
               "true_label": int(features.y[index]), "predicted_label": int(predicted[index]),
               **metadata, **{name: float(features.X[index, column])
                              for column, name in enumerate(ALL_FEATURES)}}
        row.update({f"probability_{label}": float(probabilities[index, column])
                    for column, label in enumerate(labels)})
        rows.append(row)
    return rows


def _importance(algorithm: str, model, names: tuple[str, ...]) -> list[dict[str, Any]]:
    if algorithm == "logreg":
        values = np.max(np.abs(model[-1].coef_), axis=0)
    else:
        values = model.feature_importances_
    return sorted(({"feature": name, "importance": float(value)}
                   for name, value in zip(names, values, strict=True)),
                  key=lambda row: row["importance"], reverse=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("ml/runs"))
    parser.add_argument("--cv-jobs", type=int, default=-1)
    args = parser.parse_args(argv)
    if argv is None:
        _ensure_xgboost_runtime()
    dataset = args.dataset.expanduser().resolve()
    control_index = dataset.with_name("dataset_noise_controls.json")
    before = {"main": _fingerprint(dataset), "noise_controls": _fingerprint(control_index)}
    train = features_from_dataloader(_loader(dataset, "train"))
    validation = features_from_dataloader(_loader(dataset, "val"))
    controls = features_from_dataloader(_loader(dataset, "val", control_index.name))
    after = {"main": _fingerprint(dataset), "noise_controls": _fingerprint(control_index)}
    if before != after:
        raise RuntimeError("Dataset changed while loading; retry after generation finishes")
    if set(train.y) != {0, 1, 2} or set(validation.y) != {0, 1, 2} or set(controls.y) != {0}:
        raise ValueError("Expected main labels 0/1/2 and all-clean noise controls")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = args.output.resolve() / f"{stamp}_tabular_models"; run_dir.mkdir(parents=True)
    folds = grouped_folds(train)
    results, tuning, importance, prediction_rows, cv_rows, selected = {}, {}, {}, [], [], {}
    for algorithm in ("logreg", "xgboost"):
        for view, names in FEATURE_VIEWS.items():
            model_name = f"{algorithm}_{view}"; print(f"Tuning {model_name}", flush=True)
            model, summary, rows = tune(algorithm, train.view(names), train.y, folds,
                                        n_jobs=args.cv_jobs)
            labels = tuple(int(label) for label in model.classes_)
            validation_probabilities = model.predict_proba(validation.view(names))
            control_probabilities = model.predict_proba(controls.view(names))
            results[model_name] = {
                "validation": evaluate_task(validation.y, validation_probabilities, labels,
                                            validation.label_metadata, clean_label=0),
                "noise_controls": evaluate_noise_robustness(
                    controls.y, control_probabilities, labels, controls.label_metadata,
                    controls.source_ids, validation.y, validation_probabilities,
                    validation.label_metadata, validation.source_ids, clean_label=0,
                ),
            }
            tuning[model_name] = summary; importance[model_name] = _importance(algorithm, model, names)
            selected[model_name] = summary["best_params"]
            prediction_rows += _prediction_rows(model_name, "validation", validation,
                                                validation_probabilities, labels)
            prediction_rows += _prediction_rows(model_name, "noise_controls", controls,
                                                control_probabilities, labels)
            cv_rows += [{"model": model_name, **row} for row in rows]
            with (run_dir / f"{model_name}.pkl").open("wb") as handle:
                pickle.dump(model, handle)

    cohort_counts = {
        name: {"samples": len(features.y), "sources": len(set(features.source_ids))}
        for name, features in (("train", train), ("validation", validation),
                               ("noise_controls", controls))
    }
    config = {"dataset": str(dataset), "fingerprints": before,
              "features": {name: list(values) for name, values in FEATURE_VIEWS.items()},
              "grids": {"logreg": LOGREG_GRID, "xgboost": XGBOOST_GRID},
              "selected_parameters": selected,
              "cohort_counts": cohort_counts,
              "fold_validation_sources": [sorted(set(np.asarray(train.source_ids)[indices]))
                                            for _, indices in folds],
              "versions": {package: version(package) for package in
                           ("numpy", "scikit-learn", "xgboost", "torch", "astropy")},
              "python": platform.python_version()}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (run_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    with (run_dir / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0])); writer.writeheader(); writer.writerows(prediction_rows)
    with (run_dir / "cv_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cv_rows[0])); writer.writeheader(); writer.writerows(cv_rows)
    write_tabular_report(
        run_dir, results, tuning, importance,
        dataset_path=dataset, prediction_rows=prediction_rows,
    )
    print(f"Report: {run_dir / 'report.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ALL_FEATURES", "FEATURE_VIEWS", "PARITY_FEATURES", "QA_FEATURES",
           "FeatureSet", "features_from_dataloader", "grouped_folds", "parity_features", "tune"]
