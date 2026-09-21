"""Labels derived from retained simulation metadata."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AssignedLabel:
    id: int
    name: str
    corruption_snr: float | None
    corruption_snr_target: float | None


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def assign_label(
    corruption_reports: Sequence[Mapping[str, Any]], criterion: str
) -> AssignedLabel:
    """Assign a training label using one explicit scientific criterion."""

    if criterion != "constant_antenna_type":
        raise ValueError(f"Unknown label criterion {criterion!r}")
    if not corruption_reports:
        return AssignedLabel(0, "none", 0.0, 0.0)
    if len(corruption_reports) != 1:
        raise ValueError(
            "constant_antenna_type requires zero or one corruption report"
        )

    report = _mapping(corruption_reports[0], name="corruption report")
    solution = _mapping(report.get("solution"), name="solution")
    metrics = _mapping(report.get("metrics"), name="metrics")
    corruption_type = solution.get("corruption_type")
    labels = {"amp": 1, "phase": 2}
    if corruption_type not in labels:
        raise ValueError(
            "solution.corruption_type must be exactly 'amp' or 'phase'"
        )
    snr = _positive_float(metrics.get("SNR_corr"), name="metrics.SNR_corr")
    target = _positive_float(
        solution.get("SNR_corr_target"), name="solution.SNR_corr_target"
    )
    return AssignedLabel(labels[corruption_type], corruption_type, snr, target)


__all__ = ["AssignedLabel", "assign_label"]
