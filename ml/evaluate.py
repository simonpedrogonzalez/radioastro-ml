"""Evaluation for calibration-error classifiers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


CLASSES = (0, 1, 2)


def _summary(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    present = sorted(set(y_true))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=present, average="macro", zero_division=0
    )
    return {
        "sample_count": len(y_true),
        "accuracy": float(np.mean(np.equal(y_true, y_pred))),
        "balanced_accuracy": float(macro_recall),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": {
            str(label): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(CLASSES)
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=CLASSES
        ).tolist(),
    }


def evaluate(
    dataloader: Iterable[Mapping[str, Any]], predictions: Mapping[str, int]
) -> dict[str, Any]:
    """Evaluate sample-ID-keyed predictions overall and by target SNR."""

    rows: list[tuple[str, int, int, float]] = []
    for batch in dataloader:
        labels = np.asarray(batch["label"]).tolist()
        for sample_id, truth, metadata in zip(
            batch["sample_id"], labels, batch["label_metadata"], strict=True
        ):
            if sample_id not in predictions:
                raise ValueError(f"Missing prediction for sample {sample_id!r}")
            target = metadata.get("corruption_snr_target")
            if target is None or not np.isfinite(target):
                raise ValueError(f"Sample {sample_id!r} has no finite target SNR")
            rows.append(
                (sample_id, int(truth), int(predictions[sample_id]), float(target))
            )

    seen = {sample_id for sample_id, _, _, _ in rows}
    extras = set(predictions) - seen
    if extras:
        raise ValueError(f"Predictions contain unknown sample IDs: {sorted(extras)!r}")
    if not rows:
        raise ValueError("Cannot evaluate an empty dataloader")

    result = _summary(
        [truth for _, truth, _, _ in rows],
        [prediction for _, _, prediction, _ in rows],
    )
    result["by_corruption_snr_target"] = {}
    for level in sorted({target for _, _, _, target in rows}):
        selected = [row for row in rows if row[3] == level]
        truth = [row[1] for row in selected]
        predicted = [row[2] for row in selected]
        level_result = _summary(truth, predicted)
        if level == 0:
            level_result["false_positive_rate"] = float(
                np.mean(np.not_equal(predicted, 0))
            )
        else:
            level_result["detection_recall"] = float(
                np.mean(np.not_equal(predicted, 0))
            )
            level_result["corruption_type_accuracy"] = float(
                np.mean(np.equal(truth, predicted))
            )
        result["by_corruption_snr_target"][f"{level:g}"] = level_result
    return result


__all__ = ["CLASSES", "evaluate"]
