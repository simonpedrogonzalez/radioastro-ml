"""Fixed logistic-regression baseline on image QA metrics."""

from __future__ import annotations

import argparse
import math
import platform
import warnings
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.preprocessing import (
    FitsSimulationDataset,
    make_simulation_dataloader,
    source_dataset_id,
)


FEATURE_NAMES = (
    # "metrics.clean_peak_jy_per_beam",
    # "metrics.dynamic_range_rms",
    "metrics.dynamic_range_scaled_mad",
    # "metrics.residual.n_pixels",
    # "metrics.residual.area_synthesized_beams",
    # "metrics.residual.rms_jy_per_beam",
    "metrics.residual.scaled_mad_jy_per_beam",
    # "metrics.residual.residual_abs_peak_jy_per_beam",
    # "metrics.residual.residual_min_jy_per_beam",
    # "metrics.residual.residual_max_jy_per_beam",
    "metrics.residual.peak_over_scaled_mad",
    # "metrics.residual.p99_over_scaled_mad",
    "metrics.residual.p99_5_over_scaled_mad",
    # "metrics.residual.rms_over_scaled_mad",
)


@dataclass(frozen=True)
class FeatureSet:
    X: np.ndarray
    y: np.ndarray
    sample_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    label_metadata: tuple[dict[str, float | None], ...]


def _nested(value: Any, keys: tuple[str, ...], *, context: str) -> Any:
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"{context} is missing")
        value = value[key]
    return value


def features_from_dataloader(
    dataloader: Any, feature_names: tuple[str, ...] = FEATURE_NAMES
) -> FeatureSet:
    """Extract the predeclared leakage-safe feature vector from each sample."""

    vectors: list[list[float]] = []
    labels: list[int] = []
    sample_ids: list[str] = []
    source_ids: list[str] = []
    label_metadata: list[dict[str, float | None]] = []
    for batch in dataloader:
        batch_labels = np.asarray(batch["label"]).tolist()
        for sample_id, label, qa, metadata in zip(
            batch["sample_id"],
            batch_labels,
            batch["qa"],
            batch["label_metadata"],
            strict=True,
        ):
            if not isinstance(qa, dict):
                raise ValueError(f"Sample {sample_id!r} has no imaging QA metadata")
            vector = []
            for feature_name in feature_names:
                keys = tuple(feature_name.split("."))[1:]
                validity = _nested(
                    qa.get("metric_validity"),
                    keys,
                    context=f"Sample {sample_id!r} validity for {feature_name}",
                )
                if validity is not True:
                    raise ValueError(
                        f"Sample {sample_id!r} has invalid metric {feature_name}"
                    )
                value = _nested(
                    qa.get("metrics"),
                    keys,
                    context=f"Sample {sample_id!r} metric {feature_name}",
                )
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(
                        f"Sample {sample_id!r} has nonfinite metric {feature_name}"
                    )
                vector.append(float(value))
            vectors.append(vector)
            labels.append(int(label))
            sample_ids.append(sample_id)
            source_ids.append(source_dataset_id(sample_id))
            label_metadata.append(dict(metadata))
    if not vectors:
        raise ValueError("Cannot extract features from an empty dataloader")
    return FeatureSet(
        np.asarray(vectors, dtype=float),
        np.asarray(labels, dtype=int),
        tuple(sample_ids),
        tuple(source_ids),
        tuple(label_metadata),
    )


def train(X: np.ndarray, y: np.ndarray):
    """Fit the single predeclared baseline model."""

    classes = set(np.unique(y).tolist())
    if classes != {0, 1, 2}:
        raise ValueError(
            f"Training labels must contain exactly 0, 1, and 2; got {classes}"
        )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        return model.fit(X, y)


def _loader(dataset_path: Path, partition: str):
    dataset = FitsSimulationDataset(
        dataset_path.parent,
        index=dataset_path.name,
        partition=partition,
        label_criterion="constant_antenna_type",
    )
    return make_simulation_dataloader(dataset, batch_size=32, shuffle=False)


def _prediction_rows(
    split: str, features: FeatureSet, predictions: np.ndarray, probabilities: np.ndarray
) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": sample_id,
            "source_id": source_id,
            "split": split,
            "true_class": int(truth),
            "predicted_class": int(prediction),
            "probability_none": float(probability[0]),
            "probability_amp": float(probability[1]),
            "probability_phase": float(probability[2]),
            "corruption_snr": metadata["corruption_snr"],
            "corruption_snr_target": metadata["corruption_snr_target"],
            **{
                name: float(value)
                for name, value in zip(FEATURE_NAMES, vector, strict=True)
            },
        }
        for sample_id, source_id, truth, prediction, probability, metadata, vector in zip(
            features.sample_ids,
            features.source_ids,
            features.y,
            predictions,
            probabilities,
            features.label_metadata,
            features.X,
            strict=True,
        )
    ]


def _counts(features: FeatureSet) -> dict[str, int]:
    return {"samples": len(features.y), "sources": len(set(features.source_ids))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("ml/runs"))
    parser.add_argument("--evaluate-test", action="store_true")
    arguments = parser.parse_args(argv)
    dataset_path = arguments.dataset.expanduser().resolve()

    train_loader = _loader(dataset_path, "train")
    validation_loader = _loader(dataset_path, "val")
    train_features = features_from_dataloader(train_loader)
    validation_features = features_from_dataloader(validation_loader)
    model = train(train_features.X, train_features.y)

    from ml.evaluate import evaluate
    from ml.report import write_run

    validation_predictions = model.predict(validation_features.X)
    validation_probabilities = model.predict_proba(validation_features.X)
    validation_by_id = dict(
        zip(validation_features.sample_ids, validation_predictions, strict=True)
    )
    results = {"validation": evaluate(validation_loader, validation_by_id)}
    rows = _prediction_rows(
        "validation",
        validation_features,
        validation_predictions,
        validation_probabilities,
    )
    counts = {
        "train": _counts(train_features),
        "validation": _counts(validation_features),
    }

    if arguments.evaluate_test:
        test_loader = _loader(dataset_path, "test")
        test_features = features_from_dataloader(test_loader)
        test_predictions = model.predict(test_features.X)
        test_probabilities = model.predict_proba(test_features.X)
        test_by_id = dict(zip(test_features.sample_ids, test_predictions, strict=True))
        results["test"] = evaluate(test_loader, test_by_id)
        rows.extend(
            _prediction_rows(
                "test", test_features, test_predictions, test_probabilities
            )
        )
        counts["test"] = _counts(test_features)

    versions = {
        "python": platform.python_version(),
        **{
            package: version(package)
            for package in (
                "numpy",
                "torch",
                "astropy",
                "scikit-learn",
                "matplotlib",
            )
        },
    }
    run_dir = write_run(
        arguments.output,
        dataset_path=dataset_path,
        model=model,
        prediction_rows=rows,
        results=results,
        feature_names=FEATURE_NAMES,
        split_counts=counts,
        versions=versions,
    )
    validation = results["validation"]
    print(run_dir)
    print(
        f"validation: accuracy={validation['accuracy']:.4f}, "
        f"balanced_accuracy={validation['balanced_accuracy']:.4f}, "
        f"macro_f1={validation['macro_f1']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
