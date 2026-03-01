from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np

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
    def sample(self, rng):
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

    def sample(self, rng):
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

