"""Image-noise reference features for the V1 dataset."""

import math
from collections.abc import Mapping


def sigma_na_from_source(source: Mapping) -> float:
    """Flag-adjusted natural-image RMS (Jy/beam) recorded by the precursor."""
    value = source.get("predicted_image_rms_jy_per_beam")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("Source has no finite positive predicted image RMS")
    return float(value)


def sigma_sna_units(residual_mad_jy_per_beam: float, sigma_na_jy_per_beam: float) -> float:
    """Residual scaled MAD divided by predicted natural-image RMS."""
    if not math.isfinite(residual_mad_jy_per_beam) or residual_mad_jy_per_beam <= 0:
        raise ValueError("Residual MAD must be finite and positive")
    if not math.isfinite(sigma_na_jy_per_beam) or sigma_na_jy_per_beam <= 0:
        raise ValueError("sigma_NA must be finite and positive")
    return residual_mad_jy_per_beam / sigma_na_jy_per_beam
