"""FITS validation and two-dimensional tensor-plane loading."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FitsPlane:
    path: Path
    values: Any
    shape: tuple[int, int]
    unit: str
    celestial_header: dict[str, Any]
    beam: tuple[float, float, float]


def _dependencies():
    try:
        import numpy as np
        from astropy.io import fits
        from astropy.wcs import WCS
    except ImportError as exc:
        raise RuntimeError("NumPy and Astropy are required to load retained FITS images") from exc
    return np, fits, WCS


def _normalized_unit(value: Any, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"FITS image has no BUNIT: {path}")
    normalized = value.strip().lower().replace(" ", "")
    aliases = {"jy/beam", "jybeam-1", "jy.beam-1"}
    if normalized not in aliases:
        raise ValueError(f"FITS BUNIT must be Jy/beam, got {value!r}: {path}")
    return "Jy/beam"


def load_fits_plane(
    path: str | Path,
    *,
    invalid_policy: str = "error",
    fill_value: float = 0.0,
) -> FitsPlane:
    np, fits, WCS = _dependencies()
    source = Path(path).expanduser().resolve()
    if invalid_policy not in {"error", "fill"}:
        raise ValueError("invalid_policy must be 'error' or 'fill'")
    with fits.open(source, memmap=False) as hdul:
        if not hdul or hdul[0].data is None:
            raise ValueError(f"FITS primary HDU has no image data: {source}")
        values = np.asarray(hdul[0].data)
        header = hdul[0].header.copy()
    non_singleton = [size for size in values.shape if size != 1]
    if len(non_singleton) != 2:
        raise ValueError(
            f"FITS image must contain exactly one 2-D plane with only singleton extra axes; "
            f"got shape {values.shape}: {source}"
        )
    values = np.squeeze(values)
    if values.ndim != 2:
        raise ValueError(f"FITS image did not reduce to two dimensions: {source}")
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not bool(np.all(finite)):
        if invalid_policy == "error":
            raise ValueError(f"FITS image contains NaN or Inf pixels: {source}")
        if not math.isfinite(fill_value):
            raise ValueError("fill_value must be finite")
        values = values.copy()
        values[~finite] = float(fill_value)
    unit = _normalized_unit(header.get("BUNIT"), source)
    beam_values = (header.get("BMAJ"), header.get("BMIN"), header.get("BPA", 0.0))
    try:
        beam = tuple(float(value) for value in beam_values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"FITS image has invalid beam headers: {source}") from exc
    if not all(math.isfinite(value) for value in beam) or beam[0] <= 0 or beam[1] <= 0:
        raise ValueError(f"FITS image has invalid beam headers: {source}")
    celestial = WCS(header).celestial
    if celestial.pixel_n_dim != 2 or celestial.world_n_dim != 2:
        raise ValueError(f"FITS image has no two-dimensional celestial WCS: {source}")
    celestial_header = dict(celestial.to_header(relax=True))
    return FitsPlane(source, values, tuple(values.shape), unit, celestial_header, beam)


def _wcs_signature(header: dict[str, Any]) -> dict[str, Any]:
    prefixes = ("CTYPE", "CUNIT", "CRPIX", "CRVAL", "CDELT", "PC", "CD")
    return {key: value for key, value in header.items() if key.startswith(prefixes)}


def validate_fits_triplet(
    paths: dict[str, str | Path],
    *,
    invalid_policy: str = "error",
    fill_value: float = 0.0,
) -> dict[str, FitsPlane]:
    if set(paths) != {"dirty", "clean", "residual"}:
        raise ValueError("FITS triplet must contain dirty, clean, and residual paths")
    planes = {
        name: load_fits_plane(path, invalid_policy=invalid_policy, fill_value=fill_value)
        for name, path in paths.items()
    }
    reference = planes["dirty"]
    for name in ("clean", "residual"):
        plane = planes[name]
        if plane.shape != reference.shape:
            raise ValueError(f"FITS shape mismatch: dirty={reference.shape}, {name}={plane.shape}")
        if plane.unit != reference.unit:
            raise ValueError(f"FITS unit mismatch: dirty={reference.unit}, {name}={plane.unit}")
        if _wcs_signature(plane.celestial_header) != _wcs_signature(reference.celestial_header):
            raise ValueError(f"FITS celestial WCS mismatch between dirty and {name}")
    return planes


__all__ = ["FitsPlane", "load_fits_plane", "validate_fits_triplet"]
