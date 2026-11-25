"""Block `energy` (16 features): scale, C/s and R/s amplitude statistics over Ω, model flux sums over s."""
from __future__ import annotations

import math

import numpy as np

from scripts.dataset_v2_matched_extract import model6

STATS = ("rms", "mean_abs", "pos_ms", "neg_ms", "max_abs")
MODEL5 = ("flux_positive", "flux_negative_magnitude", "flux_absolute", "flux_inside_beam", "flux_outside_beam")
NAMES = ("energy.log10_s", *(f"energy.{c}.{s}" for c in "CR" for s in STATS), *(f"energy.model.{m}_over_s" for m in MODEL5))


def stats(x):
    """rms, mean |x|, mean(x²·[x>0]), mean(x²·[x<0]), max |x| of a 1-D array (zeros for an all-zero plane)."""
    x2 = x * x
    return [math.sqrt(float(x2.mean())), float(np.abs(x).mean()), float((x2 * (x > 0)).mean()),
            float((x2 * (x < 0)).mean()), float(np.abs(x).max())]


def energy(planes):
    omega, s = planes["omega"], planes["s"]
    # model6's first five values are raw-model Jy sums over Ω (inside/outside the FWHM aperture at the phase
    # centre); the restored peak only enters its sixth value, which is not used here.
    sums = model6(planes["model"], omega, planes["header"], float("nan"))[0][:5]
    values = [math.log10(s), *stats(planes["C"][omega]), *stats(planes["R"][omega]), *(v / s for v in sums)]
    return dict(zip(NAMES, values))
