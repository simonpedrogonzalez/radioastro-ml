"""Experiment-specific views built from the generic classification evaluator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix

from ml.evaluate import METRICS, evaluate_classification


STRENGTH_SATURATION_TARGET = 30.0


def corruption_strength_weights(targets: Sequence[float] | np.ndarray) -> np.ndarray:
    """Weight clean as one and logarithmically saturate corrupted examples at 30."""

    values = np.asarray(targets, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("strength targets must be a nonempty finite nonnegative vector")
    weights = np.minimum(
        1.0, np.log1p(values) / np.log1p(STRENGTH_SATURATION_TARGET)
    )
    weights[values == 0] = 1.0
    return weights


def strength_weighted_confusion(
    y_true: Sequence[Any] | np.ndarray,
    y_predicted: Sequence[Any] | np.ndarray,
    labels: Sequence[Any],
    targets: Sequence[float] | np.ndarray,
) -> list[list[float]]:
    truth, predicted = np.asarray(y_true), np.asarray(y_predicted)
    weights = corruption_strength_weights(targets)
    if truth.shape != predicted.shape or truth.shape != weights.shape:
        raise ValueError("truth, predictions, and strength targets must have equal shape")
    return confusion_matrix(
        truth, predicted, labels=tuple(labels), sample_weight=weights
    ).tolist()


def _main_view(truth, scores, labels):
    result = evaluate_classification(truth, scores, labels)
    return {
        "sample_count": result["sample_count"],
        **result["macro"],
        "labels": result["labels"],
        "per_class": result["per_class"],
        "confusion_matrix": result["confusion_matrix"],
    }


def _detection_view(truth, scores, labels, clean_label):
    clean_index = labels.index(clean_label)
    binary_truth = np.where(truth == clean_label, "clean", "corrupted")
    binary_scores = np.column_stack((scores[:, clean_index], 1 - scores[:, clean_index]))
    result = evaluate_classification(
        binary_truth, binary_scores, ("clean", "corrupted")
    )
    corrupted = result["per_class"]["corrupted"]
    count = int(np.count_nonzero(binary_truth == "corrupted"))
    return {
        "sample_count": len(truth),
        **{metric: corrupted[metric] for metric in METRICS},
        "corrupted_count": count,
        "corrupted_prevalence": count / len(truth),
    }


def _error_view(truth, scores, labels, clean_label):
    error_labels = tuple(label for label in labels if label != clean_label)
    eligible = truth != clean_label
    selected = eligible & (scores[:, labels.index(clean_label)] < 0.5)
    detected = int(selected.sum())
    base = {
        "sample_count": detected,
        "eligible_corruptions": int(eligible.sum()),
        "detected_corruptions": detected,
        "detection_coverage": detected / int(eligible.sum()) if eligible.any() else None,
    }
    if not detected:
        return {
            **base,
            **dict.fromkeys(METRICS),
            "labels": list(error_labels),
            "per_class": {},
            "confusion_matrix": [],
        }
    indices = [labels.index(label) for label in error_labels]
    conditional_scores = scores[selected][:, indices]
    conditional_scores /= conditional_scores.sum(axis=1, keepdims=True)
    result = evaluate_classification(truth[selected], conditional_scores, error_labels)
    return {
        **base,
        **result["macro"],
        "labels": result["labels"],
        "per_class": result["per_class"],
        "confusion_matrix": result["confusion_matrix"],
    }


def _views(truth, scores, labels, clean_label):
    return {
        "main": _main_view(truth, scores, labels),
        "detection": _detection_view(truth, scores, labels, clean_label),
        "error_identification": _error_view(truth, scores, labels, clean_label),
    }


def _metadata_arrays(metadata: Sequence[Mapping[str, Any]], count: int):
    if len(metadata) != count:
        raise ValueError("metadata length must match y_true")
    targets, kinds = [], []
    for index, row in enumerate(metadata):
        target = row.get("corruption_snr_target")
        if target is None or not np.isfinite(target):
            raise ValueError(f"metadata row {index} has no finite target SNR")
        kind = row.get("sample_kind")
        if kind not in {"baseline", "gain", "increased_noise"}:
            raise ValueError(f"metadata row {index} has invalid sample_kind {kind!r}")
        targets.append(float(target))
        kinds.append(kind)
    return np.asarray(targets), np.asarray(kinds)


def _noise_controls(truth, scores, labels, clean_label, targets):
    if np.any(truth != clean_label):
        raise ValueError("Increased-noise controls must all use the clean label")
    if np.any(targets <= 0):
        raise ValueError("Increased-noise targets must be positive")
    predicted = np.asarray(labels)[np.argmax(scores, axis=1)]

    def summary(mask):
        values, counts = np.unique(predicted[mask], return_counts=True)
        predicted_counts = {str(label): 0 for label in labels}
        predicted_counts.update(
            {str(label): int(count) for label, count in zip(values, counts)}
        )
        correct = float(np.mean(predicted[mask] == clean_label))
        return {
            "sample_count": int(mask.sum()),
            "correct_rejection_rate": correct,
            "false_positive_rate": 1 - correct,
            "predicted_counts": predicted_counts,
        }

    return {
        "evaluation_kind": "increased_noise_controls",
        "overall": summary(np.ones(len(truth), dtype=bool)),
        "by_noise_snr_target": {
            f"{level:g}": summary(targets == level) for level in sorted(set(targets))
        },
    }


def evaluate_task(
    y_true: Sequence[Any] | np.ndarray,
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[Any],
    metadata: Sequence[Mapping[str, Any]],
    *,
    clean_label: Any,
) -> dict[str, Any]:
    """Build main, detection, error-type, strength, or noise-control views."""

    truth = np.asarray(y_true)
    scores = np.asarray(probabilities, dtype=float)
    labels = tuple(
        label.item() if isinstance(label, np.generic) else label for label in labels
    )
    if clean_label not in labels:
        raise ValueError("clean_label must be present in labels")
    if len(labels) < 3:
        raise ValueError("Task evaluation requires one clean and at least two error labels")
    evaluate_classification(truth, scores, labels)
    targets, kinds = _metadata_arrays(metadata, len(truth))
    noise = kinds == "increased_noise"
    if noise.any():
        if not noise.all():
            raise ValueError("Cannot pool main samples with increased-noise controls")
        return _noise_controls(truth, scores, labels, clean_label, targets)
    if np.any((kinds == "baseline") & (truth != clean_label)):
        raise ValueError("Baseline samples must use the clean label")
    if np.any((kinds == "gain") & (truth == clean_label)):
        raise ValueError("Gain samples must use an error label")
    baseline = kinds == "baseline"
    if not baseline.any() or np.all(baseline):
        raise ValueError("Main evaluation requires baseline and gain samples")
    if np.any(targets[baseline] != 0) or np.any(targets[~baseline] <= 0):
        raise ValueError("Baseline targets must be zero and gain targets positive")
    result = {
        "evaluation_kind": "classification",
        "overall": _views(truth, scores, labels, clean_label),
        "by_corruption_snr_target": {},
    }
    predicted = np.asarray(labels)[np.argmax(scores, axis=1)]
    result["overall"]["main"]["strength_weighted_confusion_matrix"] = (
        strength_weighted_confusion(truth, predicted, labels, targets)
    )
    result["strength_weighting"] = {
        "formula": "clean=1; corrupted=min(1, log1p(target)/log1p(30))",
        "saturation_target": STRENGTH_SATURATION_TARGET,
        "by_target": {
            f"{level:g}": float(corruption_strength_weights([level])[0])
            for level in sorted(set(targets))
        },
    }
    for level in sorted(set(targets[~baseline])):
        selected = baseline | ((kinds == "gain") & (targets == level))
        result["by_corruption_snr_target"][f"{level:g}"] = _views(
            truth[selected], scores[selected], labels, clean_label
        )
    return result


def evaluate_noise_robustness(
    control_truth: Sequence[Any] | np.ndarray,
    control_probabilities: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[Any],
    control_metadata: Sequence[Mapping[str, Any]],
    control_sources: Sequence[str],
    baseline_truth: Sequence[Any] | np.ndarray,
    baseline_probabilities: Sequence[Sequence[float]] | np.ndarray,
    baseline_metadata: Sequence[Mapping[str, Any]],
    baseline_sources: Sequence[str],
    *,
    clean_label: Any,
) -> dict[str, Any]:
    """Compare held-out noisy controls with matched clean source baselines."""

    controls = evaluate_task(
        control_truth, control_probabilities, labels, control_metadata,
        clean_label=clean_label,
    )
    truth = np.asarray(baseline_truth)
    scores = np.asarray(baseline_probabilities, dtype=float)
    if len(baseline_metadata) != len(truth) or len(baseline_sources) != len(truth):
        raise ValueError("Baseline arrays, metadata, and sources must have equal length")
    evaluate_classification(truth, scores, labels)
    baseline_by_source = {}
    for index, (label, metadata, source) in enumerate(
        zip(truth, baseline_metadata, baseline_sources, strict=True)
    ):
        if metadata.get("sample_kind") != "baseline":
            continue
        if label != clean_label or source in baseline_by_source:
            raise ValueError("Each control source must have one clean baseline")
        baseline_by_source[source] = index
    if len(control_sources) != len(control_metadata):
        raise ValueError("Control metadata and sources must have equal length")
    try:
        matched_indices = [baseline_by_source[source] for source in control_sources]
    except KeyError as exc:
        raise ValueError(f"No validation baseline for control source {exc.args[0]!r}") from exc
    targets, _ = _metadata_arrays(control_metadata, len(control_metadata))
    matched_scores = scores[matched_indices]
    evaluate_classification(np.full(len(targets), clean_label), matched_scores, labels)
    matched = _noise_controls(
        np.full(len(targets), clean_label), matched_scores, tuple(labels), clean_label, targets
    )

    def comparison(control, baseline):
        return {
            **control,
            "matched_baseline_false_positive_rate": baseline["false_positive_rate"],
            "excess_false_positive_rate": (
                control["false_positive_rate"] - baseline["false_positive_rate"]
            ),
            "matched_baseline_predicted_counts": baseline["predicted_counts"],
        }

    return {
        "evaluation_kind": "noise_robustness",
        "overall": comparison(controls["overall"], matched["overall"]),
        "by_noise_snr_target": {
            level: comparison(values, matched["by_noise_snr_target"][level])
            for level, values in controls["by_noise_snr_target"].items()
        },
    }


__all__ = [
    "STRENGTH_SATURATION_TARGET",
    "corruption_strength_weights",
    "evaluate_noise_robustness",
    "evaluate_task",
    "strength_weighted_confusion",
]
