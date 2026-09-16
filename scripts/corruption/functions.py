"""Configurable scalar corruption functions and immutable realizations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


class _ReportableConfiguration:
    def to_report_text(self) -> str:
        return repr(self)


@dataclass(frozen=True)
class MagnitudeSpec(_ReportableConfiguration):
    amp_max_frac: float = 0.0
    phase_max_deg: float = 0.0
    amp_clip_frac: Optional[float] = None
    phase_clip_deg: Optional[float] = None

    def to_report_dict(self) -> dict[str, object]:
        return {
            "type": "magnitude",
            "amp_max_frac": self.amp_max_frac,
            "phase_max_deg": self.phase_max_deg,
            "amp_clip_frac": self.amp_clip_frac,
            "phase_clip_deg": self.phase_clip_deg,
        }


class CorrFnRealization:
    """A sampled corruption function that can be evaluated deterministically."""

    def eval(self, times: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class CorrFn(_ReportableConfiguration):
    """Immutable specification that samples one deterministic realization."""

    def sample(
        self,
        rng: np.random.Generator,
        *,
        times: np.ndarray,
    ) -> CorrFnRealization:
        raise NotImplementedError

    def to_report_dict(self) -> dict[str, object]:
        raise NotImplementedError


@dataclass(frozen=True)
class Constant(CorrFn, CorrFnRealization):
    """A fixed amplitude gain or phase offset."""

    value: float

    def __post_init__(self) -> None:
        value = float(self.value)
        if not np.isfinite(value):
            raise ValueError(f"value must be finite, got {self.value!r}")
        object.__setattr__(self, "value", value)

    def sample(self, rng, *, times) -> "Constant":
        del rng, times
        return self

    def eval(self, times) -> np.ndarray:
        return np.full(np.asarray(times).shape, self.value, dtype=float)

    def to_report_dict(self) -> dict[str, object]:
        return {"type": "constant", "value": self.value}

    def __repr__(self) -> str:
        return f"Constant(value={self.value!r})"


@dataclass(frozen=True)
class _LinearDriftRealization(CorrFnRealization):
    t0: float
    t1: float
    drift: float

    def eval(self, times: np.ndarray) -> np.ndarray:
        t = np.asarray(times, dtype=float)
        if t.size == 0:
            return t
        duration = self.t1 - self.t0
        if duration <= 0.0:
            return np.zeros_like(t)
        return self.drift * (t - self.t0) / duration


@dataclass(frozen=True)
class MaxLinearDrift(CorrFn):
    """Linear drift with a specified end-to-start change."""

    max_drift: float
    direction: Literal["up", "down", "random"] = "random"

    def __post_init__(self) -> None:
        if self.direction not in ("up", "down", "random"):
            raise ValueError("direction must be 'up', 'down', or 'random'")
        if not np.isfinite(self.max_drift):
            raise ValueError("max_drift must be finite")

    def sample(self, rng, *, times) -> CorrFnRealization:
        t = _sample_times(times)
        sign = 1.0
        if self.direction == "down":
            sign = -1.0
        elif self.direction == "random":
            sign = 1.0 if rng.random() < 0.5 else -1.0
        return _LinearDriftRealization(float(t[0]), float(t[-1]), sign * self.max_drift)

    def to_report_dict(self) -> dict[str, object]:
        return {
            "type": "max_linear_drift",
            "max_drift": self.max_drift,
            "direction": self.direction,
        }


@dataclass(frozen=True)
class MaxSineWave(CorrFn, CorrFnRealization):
    """Sine wave with a specified peak amplitude and fixed phase."""

    max_amp: float
    period_s: float
    phase0: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.max_amp):
            raise ValueError("max_amp must be finite")
        if not np.isfinite(self.period_s) or self.period_s <= 0.0:
            raise ValueError(f"period_s must be > 0, got {self.period_s}")
        if not np.isfinite(self.phase0):
            raise ValueError("phase0 must be finite")

    def sample(self, rng, *, times) -> "MaxSineWave":
        del rng, times
        return self

    def eval(self, times: np.ndarray) -> np.ndarray:
        t = np.asarray(times, dtype=float)
        return self.max_amp * np.sin(2.0 * np.pi * t / self.period_s + self.phase0)

    def to_report_dict(self) -> dict[str, object]:
        return {
            "type": "max_sine_wave",
            "max_amp": self.max_amp,
            "period_s": self.period_s,
            "phase0": self.phase0,
        }


@dataclass(frozen=True)
class RandomPhaseMaxSineWave(CorrFn):
    """Sine-wave specification with a sampled global phase offset."""

    max_amp: float
    period_s: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.max_amp):
            raise ValueError("max_amp must be finite")
        if not np.isfinite(self.period_s) or self.period_s <= 0.0:
            raise ValueError(f"period_s must be > 0, got {self.period_s}")

    def sample(self, rng, *, times) -> MaxSineWave:
        del times
        return MaxSineWave(
            max_amp=self.max_amp,
            period_s=self.period_s,
            phase0=float(rng.uniform(0.0, 2.0 * np.pi)),
        )

    def to_report_dict(self) -> dict[str, object]:
        return {
            "type": "random_phase_max_sine_wave",
            "max_amp": self.max_amp,
            "period_s": self.period_s,
        }


@dataclass(frozen=True)
class _InterpolatedRealization(CorrFnRealization):
    t_grid: np.ndarray
    x_grid: np.ndarray

    def eval(self, times: np.ndarray) -> np.ndarray:
        t = np.asarray(times, dtype=float)
        if t.size == 0:
            return t
        return np.interp(
            np.clip(t, self.t_grid[0], self.t_grid[-1]),
            self.t_grid,
            self.x_grid,
        )


@dataclass(frozen=True)
class fBM(CorrFn):
    """Fractional Brownian-motion specification with sampled-path RMS."""

    max_amp: float
    H: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.max_amp) or self.max_amp < 0.0:
            raise ValueError(f"max_amp must be finite and >= 0, got {self.max_amp}")
        if not np.isfinite(self.H) or not 0.0 < self.H < 1.0:
            raise ValueError(f"H must be in (0,1), got {self.H}")

    def sample(self, rng, *, times) -> CorrFnRealization:
        t = _sample_times(times)
        duration = float(t[-1] - t[0])
        t_rel = t - t[0]
        try:
            from stochastic.processes.continuous import FractionalBrownianMotion
        except ImportError as exc:
            raise RuntimeError(
                "The stochastic package is required for fBM corruption"
            ) from exc

        process = FractionalBrownianMotion(hurst=self.H, t=duration, rng=rng)
        if hasattr(process, "sample_at"):
            x = process.sample_at(t_rel)
        else:
            x_uniform = process.sample(len(t_rel) - 1)
            t_uniform = np.linspace(0.0, duration, num=len(x_uniform))
            x = np.interp(t_rel, t_uniform, x_uniform)
        x = np.asarray(x, dtype=float)
        x -= x[0]
        if x.size > 1:
            rms = float(np.sqrt(np.mean(x[1:] ** 2)))
            if rms > 0.0:
                x *= self.max_amp / rms
            else:
                x.fill(0.0)
        t.setflags(write=False)
        x.setflags(write=False)
        return _InterpolatedRealization(t, x)

    def __repr__(self) -> str:
        return f"fBM(max_amp={self.max_amp!r}, H={self.H!r})"

    def to_report_dict(self) -> dict[str, object]:
        return {
            "type": "fractional_brownian_motion",
            "max_amp": self.max_amp,
            "H": self.H,
        }


def _sample_times(times: np.ndarray) -> np.ndarray:
    values = np.asarray(times, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("times must be a 1D array with at least 2 values")
    if not np.all(np.isfinite(values)):
        raise ValueError("times must be finite")
    result = np.unique(np.sort(values))
    if result.size < 2 or result[-1] <= result[0]:
        raise ValueError("times must contain at least two distinct values")
    return result


__all__ = [
    "Constant",
    "CorrFn",
    "CorrFnRealization",
    "MagnitudeSpec",
    "MaxLinearDrift",
    "MaxSineWave",
    "RandomPhaseMaxSineWave",
    "fBM",
]
