"""Time-grid configuration for sampled corruption functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class TimeGrid:
    solint: str | int = "int"
    interp: Literal["linear"] = "linear"

    def __post_init__(self):
        if self.solint == "int":
            self.dt = "int"
            return
        if isinstance(self.solint, int):
            self.dt = float(self.solint)
            return

        value = self.solint.strip().lower()
        if value.endswith("s"):
            seconds = int(value[:-1])
        elif value.endswith("m"):
            seconds = int(value[:-1]) * 60
        else:
            raise ValueError(
                f"Invalid solint '{self.solint}'. Use 'int', '#s', or '#m'."
            )
        self.dt = float(seconds)

    def get_times(self, times: np.ndarray, *, t0: float) -> tuple[np.ndarray, np.ndarray]:
        times = np.asarray(times, dtype=float)
        if self.dt == "int":
            n = times.shape[0]
            return np.arange(n, dtype=np.int64), times.copy()

        dt = float(self.dt)
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        bin_ids = np.floor((times - t0) / dt).astype(np.int64)
        unique_bins = np.unique(bin_ids)
        centers = t0 + unique_bins.astype(float) * dt
        last = unique_bins.max()
        centers = np.concatenate([centers, [t0 + (last + 1.0) * dt]])
        return bin_ids, centers

    def full_grid(self, t0: float, tf: float) -> np.ndarray:
        if self.dt == "int":
            raise ValueError(
                "full_grid requires numeric solint (e.g. '5s'), not solint='int'."
            )
        dt = float(self.dt)
        t0 = float(t0)
        tf = float(tf)
        if tf < t0:
            raise ValueError(f"tf must be >= t0, got {t0}..{tf}")
        n = int(np.floor((tf - t0) / dt))
        return t0 + np.arange(n + 2, dtype=np.int64) * dt

    def to_report_dict(self) -> dict[str, object]:
        return {"type": "time_grid", "solint": self.solint, "interp": self.interp}

    def to_report_text(self) -> str:
        return repr(self)


__all__ = ["TimeGrid"]
