"""Atomic export of durable compressed FITS imaging products."""

from __future__ import annotations

import gzip
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Literal

from .models import Beam


def export_casa_fits(
    image_path: str | Path,
    destination: str | Path,
    *,
    fallback_beam: Beam | None = None,
    fallback_unit: str = "Jy/beam",
    invalid_policy: Literal["error", "fill"] = "error",
    fill_value: float = 0.0,
) -> Path:
    """Export one CASA image and atomically install a validated ``.fits.gz``."""
    if invalid_policy not in {"error", "fill"}:
        raise ValueError("invalid_policy must be 'error' or 'fill'")
    if not math.isfinite(fill_value):
        raise ValueError("fill_value must be finite")
    try:
        import numpy as np
        from astropy.io import fits
        from casatasks import exportfits
    except ImportError as exc:  # pragma: no cover - exercised in CASA
        raise RuntimeError("CASA and Astropy are required to export retained FITS images") from exc
    source = Path(image_path).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    if not target.name.endswith(".fits.gz"):
        raise ValueError(f"Retained FITS destination must end in .fits.gz: {target}")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite retained FITS image: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=".fits", dir=target.parent
    )
    os.close(descriptor)
    raw = Path(raw_name)
    compressed = raw.with_suffix(".fits.gz.tmp")
    try:
        exportfits(
            imagename=str(source),
            fitsimage=str(raw),
            overwrite=True,
            dropstokes=False,
            dropdeg=False,
        )
        with fits.open(raw, mode="update") as hdul:
            header = hdul[0].header
            values = hdul[0].data
            finite = np.isfinite(values)
            if not bool(np.all(finite)):
                if invalid_policy == "error":
                    raise ValueError(f"FITS image contains NaN or Inf pixels: {raw}")
                values[~finite] = float(fill_value)
                header.add_history(
                    "Non-finite pixels produced by CASA export were filled with "
                    f"{float(fill_value):.12g}."
                )
            if not str(header.get("BUNIT", "")).strip():
                header["BUNIT"] = fallback_unit
            if fallback_beam is not None:
                header.setdefault("BMAJ", float(fallback_beam.major_arcsec) / 3600.0)
                header.setdefault("BMIN", float(fallback_beam.minor_arcsec) / 3600.0)
                header.setdefault("BPA", float(fallback_beam.position_angle_deg))
            hdul.flush()
        with raw.open("rb") as input_handle, compressed.open("wb") as compressed_handle:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=compressed_handle, mtime=0
            ) as output:
                shutil.copyfileobj(input_handle, output, length=1024 * 1024)
        from scripts.preprocessing.fits import load_fits_plane

        load_fits_plane(compressed)
        os.replace(compressed, target)
    except Exception:
        compressed.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        raise
    finally:
        raw.unlink(missing_ok=True)
    return target


def export_fits_triplet(
    dirty_image: Path,
    clean_image: Path,
    residual_image: Path,
    output_dir: Path,
    *,
    fallback_beam: Beam,
    invalid_policy: Literal["error", "fill"] = "error",
    fill_value: float = 0.0,
) -> dict[str, Path]:
    destinations = {
        "dirty": output_dir / "dirty.fits.gz",
        "clean": output_dir / "clean.fits.gz",
        "residual": output_dir / "residual.fits.gz",
    }
    installed: list[Path] = []
    try:
        for name, source in (
            ("dirty", dirty_image),
            ("clean", clean_image),
            ("residual", residual_image),
        ):
            installed.append(
                export_casa_fits(
                    source,
                    destinations[name],
                    fallback_beam=fallback_beam,
                    invalid_policy=invalid_policy,
                    fill_value=fill_value,
                )
            )
        from scripts.preprocessing.fits import validate_fits_triplet

        validate_fits_triplet(destinations)
    except Exception:
        for path in installed:
            path.unlink(missing_ok=True)
        raise
    return destinations


def export_fits_products(
    dirty_image: Path,
    clean_image: Path,
    residual_image: Path,
    psf_image: Path,
    output_dir: Path,
    *,
    fallback_beam: Beam,
    invalid_policy: Literal["error", "fill"] = "error",
    fill_value: float = 0.0,
) -> dict[str, Path]:
    """Export and validate the four durable imaging products."""
    destinations = {
        "dirty": output_dir / "dirty.fits.gz",
        "clean": output_dir / "clean.fits.gz",
        "residual": output_dir / "residual.fits.gz",
        "psf": output_dir / "psf.fits.gz",
    }
    installed: list[Path] = []
    try:
        for name, source in (
            ("dirty", dirty_image),
            ("clean", clean_image),
            ("residual", residual_image),
            ("psf", psf_image),
        ):
            installed.append(
                export_casa_fits(
                    source,
                    destinations[name],
                    fallback_beam=fallback_beam,
                    fallback_unit="1" if name == "psf" else "Jy/beam",
                    invalid_policy=invalid_policy,
                    fill_value=fill_value,
                )
            )
        from scripts.preprocessing.fits import validate_fits_products

        validate_fits_products(destinations)
    except Exception:
        for path in installed:
            path.unlink(missing_ok=True)
        raise
    return destinations


__all__ = ["export_casa_fits", "export_fits_products", "export_fits_triplet"]
