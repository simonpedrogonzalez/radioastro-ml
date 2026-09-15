"""Configurable corruption functions used by antenna-gain injection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from .metrics import (
    ConstantGainParameters,
    DetectabilityMetrics,
    _measure_visibility_powers,
)


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


class CorrFn(_ReportableConfiguration):
    """Base class for configurable corruption functions."""

    def sample(self, rng, **kwargs):
        raise NotImplementedError

    def eval(self, t: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def to_report_dict(self) -> dict[str, object]:
        raise NotImplementedError


@dataclass(frozen=True)
class Constant(CorrFn):
    """A fixed amplitude gain or phase offset, optionally solved from detectability."""

    value: float
    metrics: DetectabilityMetrics | None = None

    def __post_init__(self) -> None:
        value = float(self.value)
        if not np.isfinite(value):
            raise ValueError(f"value must be finite, got {self.value!r}")
        object.__setattr__(self, "value", value)

    @classmethod
    def from_detectability(cls, parameters: ConstantGainParameters) -> "Constant":
        if not isinstance(parameters, ConstantGainParameters):
            raise TypeError("parameters must be ConstantGainParameters")
        power = _measure_visibility_powers(parameters)
        unit_gain_detectability = float(
            np.sqrt(power.weighted_affected_signal_power)
        )
        if not np.isfinite(unit_gain_detectability) or unit_gain_detectability <= 0.0:
            raise ValueError(
                "unit_gain_detectability must be finite and positive, got "
                f"{unit_gain_detectability!r}"
            )
        gain_error_magnitude = float(
            parameters.target_rho_corr / unit_gain_detectability
        )
        if not np.isfinite(gain_error_magnitude) or gain_error_magnitude <= 0.0:
            raise ValueError(
                "gain_error_magnitude must be finite and positive, got "
                f"{gain_error_magnitude!r}"
            )

        amplitude_gain: float | None = None
        phase_offset_rad: float | None = None
        phase_offset_deg: float | None = None
        if parameters.corruption_type == "amp":
            amplitude_gain = float(1.0 + parameters.sign * gain_error_magnitude)
            if not np.isfinite(amplitude_gain):
                raise ValueError(
                    f"Amplitude solution must be finite, got {amplitude_gain!r}"
                )
            if amplitude_gain <= 0.0:
                raise ValueError(
                    "Requested negative amplitude branch is non-positive: "
                    f"1 + sign * gain_error_magnitude = {amplitude_gain!r}"
                )
            value = amplitude_gain
        else:
            if gain_error_magnitude > 2.0:
                raise ValueError(
                    "Phase corruption requires gain_error_magnitude <= 2, got "
                    f"{gain_error_magnitude!r}"
                )
            phase_offset_rad = float(
                parameters.sign * 2.0 * np.arcsin(gain_error_magnitude / 2.0)
            )
            phase_offset_deg = float(np.degrees(phase_offset_rad))
            value = phase_offset_rad

        epsilon_vis = float(
            gain_error_magnitude
            * np.sqrt(power.affected_signal_power / power.total_signal_power)
        )
        rho_corr = float(gain_error_magnitude * unit_gain_detectability)
        if not np.isfinite(epsilon_vis) or epsilon_vis <= 0.0:
            raise ValueError(
                f"epsilon_vis must be finite and positive, got {epsilon_vis!r}"
            )
        if not np.isfinite(rho_corr) or rho_corr <= 0.0:
            raise ValueError(f"rho_corr must be finite and positive, got {rho_corr!r}")
        metrics = DetectabilityMetrics(
            corruption_type=parameters.corruption_type,
            antenna_id=parameters.antenna_id,
            sign=parameters.sign,
            gain_error_magnitude=gain_error_magnitude,
            amplitude_gain=amplitude_gain,
            phase_offset_rad=phase_offset_rad,
            phase_offset_deg=phase_offset_deg,
            epsilon_vis=epsilon_vis,
            rho_corr=rho_corr,
            target_rho_corr=parameters.target_rho_corr,
            unit_gain_detectability=unit_gain_detectability,
            thermal_noise_jy=parameters.thermal_noise_jy,
            valid_sample_count=power.valid_sample_count,
            affected_sample_count=power.affected_sample_count,
            total_signal_power=power.total_signal_power,
            affected_signal_power=power.affected_signal_power,
            weighted_affected_signal_power=power.weighted_affected_signal_power,
        )
        return cls(value=value, metrics=metrics)

    def sample(self, rng, **kwargs):
        del rng, kwargs
        return self

    def eval(self, times, *, rng=None) -> np.ndarray:
        del rng
        return np.full(np.asarray(times).shape, self.value, dtype=float)

    def to_report_dict(self) -> dict[str, object]:
        return {"type": "constant", "value": self.value}

    def to_report_text(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return f"Constant(value={self.value!r})"


@dataclass
class MaxLinearDrift(CorrFn):
    """Linear drift with a specified end-to-start drift."""

    max_drift: float
    direction: Literal["up", "down", "random"] = "random"

    def eval(self, t: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        if t.size == 0:
            return t

        t0 = float(np.min(t))
        t1 = float(np.max(t))
        dt = t1 - t0
        if dt <= 0:
            return np.zeros_like(t, dtype=float)

        x = (t - t0) / dt
        sign = 1.0
        if self.direction == "down":
            sign = -1.0
        elif self.direction == "random":
            sign = 1.0 if rng.random() < 0.5 else -1.0
        return sign * self.max_drift * x

    def to_report_dict(self) -> dict[str, object]:
        return {
            "type": "max_linear_drift",
            "max_drift": self.max_drift,
            "direction": self.direction,
        }


@dataclass
class MaxSineWave(CorrFn):
    """Sine wave with a specified peak amplitude."""

    max_amp: float
    period_s: float
    phase0: float = 0.0

    def eval(self, t: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        if t.size == 0:
            return t
        if self.period_s <= 0:
            raise ValueError(f"period_s must be > 0, got {self.period_s}")
        return self.max_amp * np.sin(2 * np.pi * t / self.period_s + self.phase0)

    def to_report_dict(self) -> dict[str, object]:
        return {
            "type": "max_sine_wave",
            "max_amp": self.max_amp,
            "period_s": self.period_s,
            "phase0": self.phase0,
        }


@dataclass
class RandomPhaseMaxSineWave(CorrFn):
    """Sine wave with a random global phase offset sampled per call."""

    max_amp: float
    period_s: float
    phase0: float | None = None

    def sample(self, rng, **kwargs):
        del kwargs
        self.phase0 = rng.uniform(0.0, 2.0 * np.pi)
        return self

    def eval(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        if t.size == 0:
            return t
        if self.phase0 is None:
            raise ValueError("call .sample first")
        if self.period_s <= 0:
            raise ValueError(f"period_s must be > 0, got {self.period_s}")
        return self.max_amp * np.sin(2.0 * np.pi * t / self.period_s + self.phase0)

    def to_report_dict(self) -> dict[str, object]:
        return {
            "type": "random_phase_max_sine_wave",
            "max_amp": self.max_amp,
            "period_s": self.period_s,
            "phase0": self.phase0,
        }


@dataclass
class fBM(CorrFn):
    """Fractional Brownian-motion drift with a target sampled-path RMS."""

    max_amp: float
    H: float
    t_grid: np.ndarray | None = None
    x_grid: np.ndarray | None = None

    def sample(self, rng, *, times: np.ndarray):
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or times.size < 2:
            raise ValueError("times must be a 1D array with at least 2 values")
        if not np.all(np.isfinite(times)):
            raise ValueError("times must be finite")
        if not (0.0 < self.H < 1.0):
            raise ValueError(f"H must be in (0,1), got {self.H}")
        if self.max_amp < 0:
            raise ValueError(f"max_amp must be >= 0, got {self.max_amp}")

        t = np.unique(np.sort(times))
        if t.size < 2:
            raise ValueError("times must contain at least two distinct values")
        t0 = float(t[0])
        duration = float(t[-1]) - t0
        if duration <= 0:
            raise ValueError("times must span a positive duration")
        t_rel = t - t0

        try:
            from stochastic.processes.continuous import FractionalBrownianMotion
        except ImportError as exc:
            raise RuntimeError(
                "The stochastic package is required for fBM corruption"
            ) from exc

        seed = int(rng.integers(0, 2**32 - 1, dtype=np.uint32))
        np.random.seed(seed)
        process = FractionalBrownianMotion(hurst=self.H, t=duration)
        if hasattr(process, "sample_at"):
            x = process.sample_at(t_rel)
        else:
            x_uniform = process.sample(len(t_rel) - 1)
            t_uniform = np.linspace(0.0, duration, num=len(x_uniform))
            x = np.interp(t_rel, t_uniform, x_uniform)

        x = np.asarray(x, dtype=float)
        x = x - x[0]
        if x.size > 1:
            rms = float(np.sqrt(np.mean(x[1:] ** 2)))
            if rms > 0:
                x = x * (self.max_amp / rms)
            else:
                x[:] = 0.0

        self.t_grid = t
        self.x_grid = x
        return self

    def eval(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        if t.size == 0:
            return t
        if self.t_grid is None or self.x_grid is None:
            raise ValueError("call .sample(rng, times=...) first")
        t_clamped = np.clip(t, self.t_grid[0], self.t_grid[-1])
        return np.interp(t_clamped, self.t_grid, self.x_grid)

    def __repr__(self) -> str:
        return f"fBM(max_amp={self.max_amp!r}, H={self.H!r})"

    def to_report_dict(self) -> dict[str, object]:
        return {"type": "fractional_brownian_motion", "max_amp": self.max_amp, "H": self.H}


__all__ = [
    "Constant",
    "CorrFn",
    "MagnitudeSpec",
    "MaxLinearDrift",
    "MaxSineWave",
    "RandomPhaseMaxSineWave",
    "fBM",
]
