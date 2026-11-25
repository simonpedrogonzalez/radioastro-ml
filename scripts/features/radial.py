"""Block `radial` (25 features, option B: regions differ per channel), in ρ (beam units, centred on the source).

C (convolved model), per region: pos_ms / neg_ms = mean of x²·[x>0] / x²·[x<0] over the region's pixels,
frac = Σ_r x² / Σ_Ω x². R (residual), per region: pos/neg_contrast = mean_r(x²·[x≷0]) / mean_Ω(x²).
An empty region gives NaN; so do frac and the contrasts of a channel with zero total energy.
"""
from __future__ import annotations

import numpy as np

C_REGIONS = ((0, .5), (.5, 1), (1, 2), (2, 4), (4, np.inf))
R_REGIONS = ((0, 2), (2, 4), (4, 8), (8, 16), (16, np.inf))
C_STATS = ("pos_ms", "neg_ms", "frac")
R_STATS = ("pos_contrast", "neg_contrast")
NAMES = (*(f"radial.C.r{k}.{s}" for k in range(len(C_REGIONS)) for s in C_STATS),
         *(f"radial.R.r{k}.{s}" for k in range(len(R_REGIONS)) for s in R_STATS))


def region_masks(rho, omega, regions):
    return [omega & (rho >= lo) & (rho < hi) for lo, hi in regions]


def radial(planes):
    omega, rho = planes["omega"], planes["rho"]
    out = []
    C = planes["C"]; c2 = C * C
    total = float(c2[omega].sum())
    for r in region_masks(rho, omega, C_REGIONS):
        if not r.any():
            out += [np.nan] * 3
            continue
        out += [float((c2 * (C > 0))[r].mean()), float((c2 * (C < 0))[r].mean()),
                float(c2[r].sum()) / total if total > 0 else np.nan]
    R = planes["R"]; r2 = R * R
    field = float(r2[omega].mean())
    for r in region_masks(rho, omega, R_REGIONS):
        if not r.any() or not field > 0:
            out += [np.nan] * 2
            continue
        out += [float((r2 * (R > 0))[r].mean()) / field, float((r2 * (R < 0))[r].mean()) / field]
    return dict(zip(NAMES, out))
