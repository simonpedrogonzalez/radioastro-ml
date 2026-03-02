from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np
import fbm

# Maybe to consider later a common strength knob based on RMS
@dataclass(frozen=True)
class MagnitudeSpec:
    amp_max_frac: float = 0.0
    phase_max_deg: float = 0.0
    amp_clip_frac: Optional[float] = None
    phase_clip_deg: Optional[float] = None



class CorrFn:
    """
    Base class for corruption functions.

    Convention:
      - eval(t) returns a corruption series aligned with t.
      - Units are defined by the specific subclass (e.g., degrees for phase, fractional for amp).
    """
    def sample(self, rng, **kwargs):
        raise NotImplementedError

    def eval(self, t: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


@dataclass
class MaxLinearDrift(CorrFn):
    """
    Linear drift with a specified end-to-start drift.

    If direction="up":    starts at 0 and ends at +max_drift
    If direction="down":  starts at 0 and ends at -max_drift
    If direction="random": randomly picks up/down per call.
    """
    max_drift: float
    direction: Literal["up", "down", "random"] = "random"

    def eval(self, t: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        if t.size == 0:
            return t

        t0 = float(np.min(t))
        t1 = float(np.max(t))
        dt = (t1 - t0)
        if dt <= 0:
            # all times equal -> no drift possible, keep at 0
            return np.zeros_like(t, dtype=float)

        x = (t - t0) / dt  # 0..1
        sign = 1.0
        if self.direction == "down":
            sign = -1.0
        elif self.direction == "random":
            sign = 1.0 if rng.random() < 0.5 else -1.0

        return sign * self.max_drift * x  # 0 .. +/- max_drift


@dataclass
class MaxSineWave(CorrFn):
    """
    Sine wave with a specified peak amplitude (crest magnitude).

    Output range is [-max_amp, +max_amp] (up to sampling).
    """
    max_amp: float
    period_s: float
    phase0: float = 0.0  # radians

    def eval(self, t: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        if t.size == 0:
            return t

        if self.period_s <= 0:
            raise ValueError(f"period_s must be > 0, got {self.period_s}")

        return self.max_amp * np.sin(2 * np.pi * t / self.period_s + self.phase0)

@dataclass
class RandomPhaseMaxSineWave(CorrFn):
    """
    Sine wave with fixed amplitude and period,
    but a random global phase offset drawn per eval() call.

    Output range is [-max_amp, +max_amp].
    """
    max_amp: float
    period_s: float
    phase0: float = None

    def sample(self, rng, **kwargs):
        self.phase0 = rng.uniform(0.0, 2.0 * np.pi)
        return self

    def eval(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        if t.size == 0:
            return t

        if self.phase0 is None:
            raise ValueError('call .sample first')

        if self.period_s <= 0:
            raise ValueError(f"period_s must be > 0, got {self.period_s}")

        return self.max_amp * np.sin(2.0 * np.pi * t / self.period_s + self.phase0)

from stochastic.processes.continuous import FractionalBrownianMotion

@dataclass
class fBM(CorrFn):
    """
    Fractional Brownian motion drift using `stochastic`.

    User-facing params:
      - max_amp: target RMS of the sampled path over the interval (approx)
      - H: Hurst exponent in (0,1)

    Usage:
      fn = fBM(max_amp=..., H=...).sample(rng, times=times)
      y  = fn.eval(times_or_any_t)
    """
    max_amp: float
    H: float

    # cached sampled path (knots)
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

        # Use sorted unique times as our evaluation points
        t = np.unique(np.sort(times))
        if t.size < 2:
            raise ValueError("times must contain at least two distinct values")

        t0 = float(t[0])
        t1 = float(t[-1])
        duration = t1 - t0
        if duration <= 0:
            raise ValueError("times must span a positive duration")

        # stochastic supports sampling at arbitrary times; we give it normalized times [0, duration]
        t_rel = t - t0

        # Seed (stochastic uses numpy RNG under the hood)
        seed = int(rng.integers(0, 2**32 - 1, dtype=np.uint32))
        np.random.seed(seed)

        proc = FractionalBrownianMotion(hurst=self.H, t=duration)

        # Prefer sample_at if available in your stochastic version; fall back if not.
        if hasattr(proc, "sample_at"):
            x = proc.sample_at(t_rel)
        else:
            # Fallback: sample on an evenly spaced grid of same length and interpolate to t_rel
            x_u = proc.sample(len(t_rel) - 1)  # stochastic's fBm sample returns n+1 points in many versions
            t_u = np.linspace(0.0, duration, num=len(x_u))
            x = np.interp(t_rel, t_u, x_u)

        x = np.asarray(x, dtype=float)

        # Anchor so x(t0)=0 (typical for fBm)
        x = x - x[0]

        # Scale to target RMS over the interval (skip first which is 0)
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

        # Clamp outside the sampled interval
        t_clamped = np.clip(t, self.t_grid[0], self.t_grid[-1])
        return np.interp(t_clamped, self.t_grid, self.x_grid)
        