"""Region-aware image metrics shared by imaging and simulation workflows."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from .config import read_beam
from .models import Beam, BeamRegion, ImageMetrics, RegionMetrics


@dataclass(frozen=True)
class _ImagePlane:
    values: Any
    valid: Any
    cell_arcsec: tuple[float, float]
    beam: Beam
    brightness_unit: str = "Jy/beam"


def _finite_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _angle_arcsec(value: float, unit: str) -> float:
    unit = str(unit).strip().casefold()
    if unit.startswith("rad"):
        return abs(math.degrees(float(value)) * 3600.0)
    if unit.startswith("deg"):
        return abs(float(value) * 3600.0)
    if unit.startswith("arcmin"):
        return abs(float(value) * 60.0)
    if unit.startswith("arcsec") or unit in {"asec", ""}:
        return abs(float(value))
    raise ValueError(f"Unsupported angular unit {unit!r}")


def _first_plane(array: Any, *, image_path: Path, dtype: Any) -> Any:
    import numpy as np

    result = np.squeeze(np.asarray(array))
    while result.ndim > 2:
        result = result[..., 0]
    if result.ndim != 2:
        raise RuntimeError(f"Unexpected image dimensionality {result.ndim} for {image_path}")
    return np.asarray(result, dtype=dtype).T


def _load_image_plane(
    image_path: str | Path,
    *,
    fallback_beam: Beam | None = None,
) -> _ImagePlane:
    try:
        from casatasks import imhead
        from casatools import image
    except ImportError as exc:  # pragma: no cover - exercised outside CASA
        raise RuntimeError("CASA casatasks, casatools, and NumPy are required for image QA") from exc

    path = Path(image_path).expanduser().resolve()
    ia = image()
    ia.open(str(path))
    try:
        values = _first_plane(ia.getchunk(), image_path=path, dtype=float)
        valid = _first_plane(ia.getchunk(getmask=True), image_path=path, dtype=bool)
        brightness_unit = str(ia.brightnessunit())
    finally:
        ia.close()
    info = imhead(imagename=str(path), mode="summary")
    if not isinstance(info, dict):
        raise RuntimeError(f"Could not read image geometry from {path}")
    cell = tuple(
        _angle_arcsec(info["incr"][index], info["axisunits"][index])
        for index in (0, 1)
    )
    if any(not math.isfinite(value) or value <= 0 for value in cell):
        raise RuntimeError(f"Invalid image cell size {cell!r} in {path}")
    try:
        beam = read_beam(path)
    except RuntimeError:
        if fallback_beam is None:
            raise
        beam = fallback_beam
    return _ImagePlane(
        values=values,
        valid=valid,
        cell_arcsec=cell,
        beam=beam,
        brightness_unit=brightness_unit,
    )


def beam_region_mask(
    shape: Sequence[int],
    cell_arcsec: Sequence[float],
    beam_major_arcsec: float,
    region: BeamRegion,
):
    """Build the canonical centered radial mask for a ``BeamRegion``.

    The lower boundary is inclusive and the upper boundary is exclusive.
    Image arrays use NumPy's ``(ny, nx)`` ordering.
    """
    import numpy as np

    if not isinstance(region, BeamRegion):
        raise TypeError(f"region must be BeamRegion, got {type(region).__name__}")
    if len(shape) != 2 or len(cell_arcsec) != 2:
        raise ValueError("shape and cell_arcsec must each contain exactly two values")
    ny, nx = (int(shape[0]), int(shape[1]))
    if nx <= 0 or ny <= 0:
        raise ValueError("image dimensions must be positive")
    cell_y, cell_x = float(cell_arcsec[1]), float(cell_arcsec[0])
    beam_major = float(beam_major_arcsec)
    if any(not math.isfinite(value) or value <= 0 for value in (cell_x, cell_y, beam_major)):
        raise ValueError("cell sizes and beam major axis must be finite and positive")
    yy, xx = np.indices((ny, nx), dtype=float)
    radius_beams = np.hypot((xx - nx // 2) * cell_x, (yy - ny // 2) * cell_y) / beam_major
    selected = np.ones((ny, nx), dtype=bool)
    if region.min_radius_beams is not None:
        selected &= radius_beams >= region.min_radius_beams
    if region.max_radius_beams is not None:
        selected &= radius_beams < region.max_radius_beams
    return selected


def summarize_residual_pixels(
    values: Any,
    *,
    pixel_area_arcsec2: float,
    beam: Beam,
) -> RegionMetrics:
    """Summarize an already-selected one-dimensional residual population."""
    import numpy as np

    pixel_area = _finite_float(pixel_area_arcsec2, name="pixel_area_arcsec2")
    if pixel_area <= 0:
        raise ValueError("pixel_area_arcsec2 must be positive")
    beam_major = _finite_float(beam.major_arcsec, name="beam.major_arcsec")
    beam_minor = _finite_float(beam.minor_arcsec, name="beam.minor_arcsec")
    if beam_major <= 0 or beam_minor <= 0:
        raise ValueError("beam major and minor axes must be positive")

    residual = np.asarray(values, dtype=float).ravel()
    residual = residual[np.isfinite(residual)]
    if not residual.size:
        raise ValueError("Metric region contains no valid finite pixels")
    median = float(np.median(residual))
    scaled_mad = 1.4826 * float(np.median(np.abs(residual - median)))
    rms = float(np.sqrt(np.mean(np.square(residual))))
    absolute = np.abs(residual)
    residual_peak = float(np.max(absolute))
    valid_scale = math.isfinite(scaled_mad) and scaled_mad > 0

    def ratio(value: float) -> float:
        return value / scaled_mad if valid_scale else float("nan")

    beam_area = math.pi * beam_major * beam_minor / (4.0 * math.log(2.0))
    return RegionMetrics(
        n_pixels=int(residual.size),
        area_synthesized_beams=float(residual.size) * pixel_area / beam_area,
        rms_jy_per_beam=rms,
        scaled_mad_jy_per_beam=scaled_mad,
        residual_abs_peak_jy_per_beam=residual_peak,
        residual_min_jy_per_beam=float(np.min(residual)),
        residual_max_jy_per_beam=float(np.max(residual)),
        peak_over_scaled_mad=ratio(residual_peak),
        p99_over_scaled_mad=ratio(float(np.percentile(absolute, 99.0))),
        p99_5_over_scaled_mad=ratio(float(np.percentile(absolute, 99.5))),
        rms_over_scaled_mad=ratio(rms),
    )


def measure_image_metrics(
    clean_image: str | Path,
    residual_image: str | Path,
    *,
    region: BeamRegion = BeamRegion(),
) -> ImageMetrics:
    """Measure a global clean peak and residual statistics in ``region``."""
    import numpy as np

    clean = _load_image_plane(clean_image)
    # Some tclean residual tables omit unit and restoring-beam metadata.  They
    # are still in Jy/beam on the same grid as the paired restored image, so
    # use that required clean-image beam only as the residual fallback.
    residual = _load_image_plane(residual_image, fallback_beam=clean.beam)
    _check_jy_per_beam(clean.brightness_unit, Path(clean_image))
    if residual.brightness_unit.strip():
        _check_jy_per_beam(residual.brightness_unit, Path(residual_image))
    if clean.values.shape != residual.values.shape:
        raise ValueError(
            f"Clean and residual image shapes differ: {clean.values.shape} vs {residual.values.shape}"
        )
    if not all(
        math.isclose(left, right, rel_tol=1e-9, abs_tol=0.0)
        for left, right in zip(clean.cell_arcsec, residual.cell_arcsec)
    ):
        raise ValueError(
            "Clean and residual image cell sizes differ: "
            f"{clean.cell_arcsec} vs {residual.cell_arcsec}"
        )
    if not all(
        math.isclose(left, right, rel_tol=1e-9, abs_tol=0.0)
        for left, right in (
            (clean.beam.major_arcsec, residual.beam.major_arcsec),
            (clean.beam.minor_arcsec, residual.beam.minor_arcsec),
        )
    ):
        raise ValueError(
            "Clean and residual restoring beams differ: "
            f"{clean.beam!r} vs {residual.beam!r}"
        )
    region_pixels = beam_region_mask(
        residual.values.shape,
        residual.cell_arcsec,
        residual.beam.major_arcsec,
        region,
    )
    residual_selection = region_pixels & residual.valid & np.isfinite(residual.values)
    if not np.any(residual_selection):
        ny, nx = residual.values.shape
        furthest_x = max(nx // 2, nx - 1 - nx // 2) * residual.cell_arcsec[0]
        furthest_y = max(ny // 2, ny - 1 - ny // 2) * residual.cell_arcsec[1]
        available = math.hypot(furthest_x, furthest_y) / residual.beam.major_arcsec
        raise ValueError(
            "Metric region contains no valid finite pixels: "
            f"min_radius_beams={region.min_radius_beams!r}, "
            f"max_radius_beams={region.max_radius_beams!r}, "
            f"maximum corner radius={available:.6g} beams"
        )
    selected = residual.values[residual_selection]
    residual_metrics = summarize_residual_pixels(
        selected,
        pixel_area_arcsec2=residual.cell_arcsec[0] * residual.cell_arcsec[1],
        beam=residual.beam,
    )
    clean_selection = clean.valid & np.isfinite(clean.values)
    if not np.any(clean_selection):
        raise RuntimeError(f"No valid finite pixels in clean image {clean_image}")
    clean_peak = float(np.max(clean.values[clean_selection]))

    def divide(scale: float) -> float:
        return clean_peak / scale if math.isfinite(scale) and scale > 0 else float("nan")

    return ImageMetrics(
        region=region,
        clean_peak_jy_per_beam=clean_peak,
        residual=residual_metrics,
        dynamic_range_rms=divide(residual_metrics.rms_jy_per_beam),
        dynamic_range_scaled_mad=divide(residual_metrics.scaled_mad_jy_per_beam),
    )


def _metric_validity(metrics: ImageMetrics) -> dict[str, Any]:
    residual = {
        name: bool(math.isfinite(float(value)))
        for name, value in vars(metrics.residual).items()
    }
    return {
        "clean_peak_jy_per_beam": math.isfinite(metrics.clean_peak_jy_per_beam),
        "residual": residual,
        "dynamic_range_rms": math.isfinite(metrics.dynamic_range_rms),
        "dynamic_range_scaled_mad": math.isfinite(metrics.dynamic_range_scaled_mad),
    }


def _metric_warnings(metrics: ImageMetrics) -> tuple[str, ...]:
    scaled_mad = metrics.residual.scaled_mad_jy_per_beam
    if math.isfinite(scaled_mad) and scaled_mad > 0:
        return ()
    return (
        "Residual scaled MAD is non-finite or zero; normalized residual metrics are invalid.",
    )


METRIC_UNITS: dict[str, Any] = {
    "region": {
        "min_radius_beams": "synthesized beam major FWHM",
        "max_radius_beams": "synthesized beam major FWHM",
    },
    "clean_peak_jy_per_beam": "Jy/beam",
    "residual": {
        "n_pixels": "pixel",
        "area_synthesized_beams": "synthesized beam",
        "rms_jy_per_beam": "Jy/beam",
        "scaled_mad_jy_per_beam": "Jy/beam",
        "residual_abs_peak_jy_per_beam": "Jy/beam",
        "residual_min_jy_per_beam": "Jy/beam",
        "residual_max_jy_per_beam": "Jy/beam",
        "peak_over_scaled_mad": "dimensionless",
        "p99_over_scaled_mad": "dimensionless",
        "p99_5_over_scaled_mad": "dimensionless",
        "rms_over_scaled_mad": "dimensionless",
    },
    "dynamic_range_rms": "dimensionless",
    "dynamic_range_scaled_mad": "dimensionless",
}


def _new_image():
    try:
        from casatools import image
    except ImportError as exc:  # pragma: no cover - exercised outside CASA
        raise RuntimeError("CASA casatools is required to inspect CASA images") from exc
    return image()


def _flat_values(value: Any) -> list[Any]:
    if hasattr(value, "ravel"):
        value = value.ravel().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    result: list[Any] = []

    def visit(item: Any) -> None:
        if isinstance(item, (str, bytes)):
            result.append(item)
        elif isinstance(item, Iterable):
            for child in item:
                visit(child)
        else:
            result.append(item)

    visit(value)
    return result


def _existing_image(path: str | Path, *, name: str) -> Path:
    result = Path(path).expanduser().resolve()
    if not result.is_dir() or not (result / "table.dat").exists():
        raise FileNotFoundError(f"{name} is not a CASA image table: {result}")
    if '"' in str(result):
        raise ValueError(f"{name} path cannot contain a double quote: {result}")
    return result


def _shape(path: Path) -> tuple[int, ...]:
    ia = _new_image()
    ia.open(str(path))
    try:
        return tuple(int(item) for item in _flat_values(ia.shape()))
    finally:
        ia.close()


def _stretch_compatible(image_shape: tuple[int, ...], other_shape: tuple[int, ...]) -> bool:
    length = max(len(image_shape), len(other_shape))
    left = image_shape + (1,) * (length - len(image_shape))
    right = other_shape + (1,) * (length - len(other_shape))
    return all(a == b or a == 1 or b == 1 for a, b in zip(left, right))


def _check_stretch(image_shape: tuple[int, ...], other: Path, *, name: str) -> None:
    other_shape = _shape(other)
    if not _stretch_compatible(image_shape, other_shape):
        raise ValueError(f"{name} shape {other_shape} cannot be stretched to image shape {image_shape}")


def _check_jy_per_beam(unit: Any, image_path: Path) -> None:
    normalized = str(unit).strip().lower().replace(" ", "")
    if normalized not in {"jy/beam", "jybeam-1", "jybeam^-1"}:
        raise ValueError(f"Expected image brightness unit Jy/beam, found {unit!r} in {image_path}")


def _stat_values(result: Any, key: str) -> list[float]:
    if not isinstance(result, dict) or key not in result:
        raise RuntimeError(f"CASA image.statistics() did not return {key!r}: {result!r}")
    values = [float(item) for item in _flat_values(result[key])]
    if not values or any(not math.isfinite(item) for item in values):
        raise RuntimeError(f"CASA statistic {key!r} is empty or non-finite: {values!r}")
    return values


def measure_pb_region(
    image: str | Path,
    pb: str | Path,
    *,
    pb_min: float,
    pb_max: float | None = None,
    exclude_mask: str | Path | None = None,
) -> RegionMetrics:
    """Measure shared residual metrics in one PB-selected image region."""
    import numpy as np

    image_plane = _load_image_plane(image)
    _check_jy_per_beam(image_plane.brightness_unit, Path(image))
    pb_path = _existing_image(pb, name="pb")
    lower = _finite_float(pb_min, name="pb_min")
    upper = None if pb_max is None else _finite_float(pb_max, name="pb_max")
    if lower < 0:
        raise ValueError("pb_min must be non-negative")
    if upper is not None and upper <= lower:
        raise ValueError("pb_max must be greater than pb_min")
    pb_plane, pb_valid = _load_untyped_plane(pb_path)
    if pb_plane.shape != image_plane.values.shape:
        raise ValueError(f"PB shape {pb_plane.shape} does not match image shape {image_plane.values.shape}")
    selected = pb_valid & np.isfinite(pb_plane) & (pb_plane >= lower)
    if upper is not None:
        selected &= pb_plane <= upper
    if exclude_mask is not None:
        mask_plane, mask_valid = _load_untyped_plane(
            _existing_image(exclude_mask, name="exclude_mask")
        )
        if mask_plane.shape != selected.shape:
            raise ValueError("exclude_mask shape does not match image shape")
        selected &= mask_valid & (mask_plane < 0.1)
    selected &= image_plane.valid & np.isfinite(image_plane.values)
    metrics = summarize_residual_pixels(
        image_plane.values[selected],
        pixel_area_arcsec2=image_plane.cell_arcsec[0] * image_plane.cell_arcsec[1],
        beam=image_plane.beam,
    )
    return metrics


def _load_untyped_plane(path: Path):
    try:
        from casatools import image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("CASA casatools and NumPy are required to inspect CASA images") from exc
    ia = image()
    ia.open(str(path))
    try:
        values = _first_plane(ia.getchunk(), image_path=path, dtype=float)
        valid = _first_plane(ia.getchunk(getmask=True), image_path=path, dtype=bool)
        return values, valid
    finally:
        ia.close()


def _pixel(ia: Any, coordinate: list[int], *, getmask: bool = False) -> Any:
    value = ia.getchunk(blc=coordinate, trc=coordinate, getmask=getmask)
    values = _flat_values(value)
    if len(values) != 1:
        raise RuntimeError(f"Expected one PB pixel at {coordinate}, found {len(values)}")
    return values[0]


def _pipeline_pb_limits(pb_path: Path) -> tuple[float, float]:
    lower, upper = 0.2, 0.3
    ia = _new_image()
    ia.open(str(pb_path))
    try:
        shape = tuple(int(item) for item in _flat_values(ia.shape()))
        if len(shape) != 4:
            raise ValueError(f"Pipeline-compatible PB limit heuristic requires a 4D image, found {shape}")
        nx, ny, _, nfreq = shape
        edge = [nx // 2, 0, 0, nfreq // 2]
        if not bool(_pixel(ia, edge, getmask=True)):
            return lower, upper
        pb_edge = 0.0
        edge_index = -1
        while pb_edge == 0.0 and edge_index < ny // 2:
            edge_index += 1
            pb_edge = float(_pixel(ia, [nx // 2, edge_index, 0, nfreq // 2]))
        if pb_edge > 0.2:
            image_index = max(edge_index, int(0.05 * ny))
            cleanmask_index = image_index + int(0.05 * ny)
            if cleanmask_index >= ny:
                raise RuntimeError("PB image is too small for the Pipeline 5% annulus heuristic")
            lower = float(_pixel(ia, [nx // 2, image_index, 0, nfreq // 2]))
            upper = float(_pixel(ia, [nx // 2, cleanmask_index, 0, nfreq // 2]))
    finally:
        ia.close()
    if not all(math.isfinite(item) for item in (lower, upper)) or upper <= lower:
        raise RuntimeError(f"Pipeline PB limits are invalid: {lower}, {upper}")
    return lower, upper


def vla_pipeline_annulus_rms(
    image: str | Path,
    pb: str | Path,
    clean_mask: str | Path | None,
) -> float:
    """Reproduce the Pipeline's specialized PB-annulus Chauvenet RMS."""
    image_path = _existing_image(image, name="image")
    pb_path = _existing_image(pb, name="pb")
    clean_path = None if clean_mask is None else _existing_image(clean_mask, name="clean_mask")
    lower, upper = _pipeline_pb_limits(pb_path)
    ia = _new_image()
    ia.open(str(image_path))
    try:
        image_shape = tuple(int(item) for item in _flat_values(ia.shape()))
        _check_jy_per_beam(ia.brightnessunit(), image_path)
        _check_stretch(image_shape, pb_path, name="pb")
        if clean_path is not None:
            _check_stretch(image_shape, clean_path, name="clean_mask")
        annulus = f'("{pb_path}" > {lower:.17g}) && ("{pb_path}" < {upper:.17g})'
        selection = annulus if clean_path is None else f'("{clean_path}" < 0.1) && {annulus}'
        result = ia.statistics(
            mask=selection,
            robust=True,
            axes=[0, 1, 2],
            algorithm="chauvenet",
            maxiter=5,
            stretch=True,
        )
        points = _stat_values(result, "npts")
        if statistics.median(points) < 10.0 and clean_path is not None:
            result = ia.statistics(
                mask=annulus,
                robust=True,
                axes=[0, 1, 2],
                algorithm="chauvenet",
                maxiter=5,
                stretch=True,
            )
            points = _stat_values(result, "npts")
        if statistics.median(points) < 1.0:
            raise ValueError("Pipeline annulus selection contains no pixels")
        rms_values = _stat_values(result, "rms")
    finally:
        ia.close()
    return float(statistics.median(rms_values))


__all__ = [
    "METRIC_UNITS",
    "beam_region_mask",
    "measure_image_metrics",
    "measure_pb_region",
    "summarize_residual_pixels",
    "vla_pipeline_annulus_rms",
]
