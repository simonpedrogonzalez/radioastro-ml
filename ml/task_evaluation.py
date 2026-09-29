"""Experiment-specific views built from the generic classification evaluator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ml.evaluate import METRICS, evaluate_classification


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
    for level in sorted(set(targets[~baseline])):
        selected = baseline | ((kinds == "gain") & (targets == level))
        result["by_corruption_snr_target"][f"{level:g}"] = _views(
            truth[selected], scores[selected], labels, clean_label
        )
    return result


__all__ = ["evaluate_task"]
