"""Small, task-agnostic classification metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


METRICS = ("precision", "recall", "f1", "auroc", "auprc")


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _curve_metrics(
    binary_truth: np.ndarray, score: np.ndarray
) -> tuple[float | None, float | None]:
    positives = int(binary_truth.sum())
    if positives == 0 or positives == len(binary_truth):
        return None, None
    return (
        float(roc_auc_score(binary_truth, score)),
        float(average_precision_score(binary_truth, score)),
    )


def evaluate_classification(
    y_true: Sequence[Any] | np.ndarray,
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[Any],
) -> dict[str, Any]:
    """Evaluate single-label probabilities in the supplied score-column order."""

    truth = np.asarray(y_true)
    scores = np.asarray(probabilities, dtype=float)
    label_values = tuple(_python_scalar(label) for label in labels)
    if truth.ndim != 1 or not len(truth):
        raise ValueError("y_true must be a nonempty one-dimensional array")
    if len(label_values) < 2 or len(set(label_values)) != len(label_values):
        raise ValueError("labels must contain at least two unique values")
    if scores.shape != (len(truth), len(label_values)):
        raise ValueError(
            f"probabilities must have shape {(len(truth), len(label_values))}; "
            f"got {scores.shape}"
        )
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("probabilities must be finite and nonnegative")
    if not np.allclose(scores.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("probability rows must sum to one")
    unknown = sorted(
        {str(_python_scalar(value)) for value in truth if value not in label_values}
    )
    if unknown:
        raise ValueError(f"y_true contains labels absent from labels: {unknown}")

    label_array = np.asarray(label_values)
    predicted = label_array[np.argmax(scores, axis=1)]
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=label_values, average=None, zero_division=0
    )
    per_class: dict[str, dict[str, float | int | None]] = {}
    for index, label in enumerate(label_values):
        auroc, auprc = _curve_metrics(truth == label, scores[:, index])
        per_class[str(label)] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "auroc": auroc,
            "auprc": auprc,
            "support": int(support[index]),
        }
    macro = {
        "precision": float(np.mean(precision)),
        "recall": float(np.mean(recall)),
        "f1": float(np.mean(f1)),
    }
    for metric in ("auroc", "auprc"):
        defined = [
            values[metric]
            for values in per_class.values()
            if values[metric] is not None
        ]
        macro[metric] = float(np.mean(defined)) if defined else None
    return {
        "sample_count": len(truth),
        "labels": list(label_values),
        "macro": macro,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            truth, predicted, labels=label_values
        ).tolist(),
    }


__all__ = ["METRICS", "evaluate_classification"]
