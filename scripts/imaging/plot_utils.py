"""Individual CASA-image plotting with the established WCS conventions."""

from __future__ import annotations

import math
import platform
import tempfile
from pathlib import Path
from typing import Mapping, Optional, Sequence

from .metrics import beam_region_mask
from .models import Beam, BeamRegion


def _fits_beam_geometry(header, fallback_beam: Optional[Beam]):
    """Return FITS beam values in degrees, using measured geometry if absent."""

    def finite(value, *, positive: bool = False):
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(result) or (positive and result <= 0):
            return None
        return result

    bmaj = finite(header.get("BMAJ"), positive=True)
    bmin = finite(header.get("BMIN"), positive=True)
    bpa = finite(header.get("BPA"))
    if fallback_beam is not None:
        if bmaj is None:
            bmaj = float(fallback_beam.major_arcsec) / 3600.0
        if bmin is None:
            bmin = float(fallback_beam.minor_arcsec) / 3600.0
        if bpa is None:
            bpa = float(fallback_beam.position_angle_deg)
    return bmaj, bmin, 0.0 if bpa is None else bpa


def casa_image_to_png(
    image_path: Path,
    png_path: Path,
    *,
    title: Optional[str] = None,
    mask_path: Optional[Path] = None,
    symmetric: bool = True,
    draw_beam: bool = False,
    robust_percentile: float = 99.5,
    metric_region: Optional[BeamRegion] = None,
    fallback_beam: Optional[Beam] = None,
    display_limits_mjy_per_beam: Optional[tuple[float, float]] = None,
) -> dict[str, object]:
    """Render one CASA image or retained FITS plane using the canonical style."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np
        import astropy
        from astropy.io import fits
        from astropy.wcs import WCS
        from casatasks import exportfits
        from matplotlib import patheffects
        from matplotlib.lines import Line2D
        from matplotlib.patches import Ellipse
    except ImportError as exc:
        raise RuntimeError(
            "CASA, Astropy, NumPy, and Matplotlib are required to create image PNGs"
        ) from exc

    png_path.parent.mkdir(parents=True, exist_ok=True)
    image_path = Path(image_path)
    source_is_fits = image_path.is_file() and image_path.name.casefold().endswith(
        (".fits", ".fits.gz", ".fit", ".fit.gz")
    )
    owns_fits_path = not source_is_fits
    if source_is_fits:
        fits_path = image_path
    else:
        temporary = tempfile.NamedTemporaryFile(
            prefix=f".{png_path.stem}-",
            suffix=".fits",
            dir=str(png_path.parent),
            delete=False,
        )
        fits_path = Path(temporary.name)
        temporary.close()
    mask_fits_path = None
    try:
        if owns_fits_path:
            exportfits(
                imagename=str(image_path),
                fitsimage=str(fits_path),
                overwrite=True,
                dropstokes=False,
                dropdeg=True,
            )
        with fits.open(fits_path) as hdul:
            values = np.squeeze(np.asarray(hdul[0].data))
            wcs = WCS(hdul[0].header).celestial
            header = hdul[0].header.copy()
        while values.ndim > 2:
            values = values[0]
        if values.ndim != 2:
            raise RuntimeError(f"Unexpected image dimensionality {values.ndim} for {image_path}")
        finite = np.isfinite(values)
        if not np.any(finite):
            raise RuntimeError(f"No finite pixels in {image_path}")
        display = values * 1e3
        finite_values = display[finite]
        if display_limits_mjy_per_beam is not None:
            if len(display_limits_mjy_per_beam) != 2:
                raise ValueError("display_limits_mjy_per_beam must contain (vmin, vmax)")
            vmin, vmax = (float(value) for value in display_limits_mjy_per_beam)
            if not all(math.isfinite(value) for value in (vmin, vmax)) or vmin >= vmax:
                raise ValueError(
                    "display_limits_mjy_per_beam must be finite with vmin < vmax"
                )
        elif symmetric:
            limit = float(np.percentile(np.abs(finite_values), robust_percentile))
            vmin, vmax = -limit, limit
        else:
            vmin, vmax = (
                float(value)
                for value in np.percentile(
                    finite_values, [100.0 - robust_percentile, robust_percentile]
                )
            )

        figure = plt.figure()
        axis = plt.subplot(projection=wcs)
        artist = axis.imshow(display, origin="lower", vmin=vmin, vmax=vmax, cmap="inferno")
        axis.set_xlabel("RA")
        axis.set_ylabel("Dec")
        axis.set_title(title or image_path.name)
        colorbar = plt.colorbar(artist, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label("mJy/beam")
        bmaj, bmin, bpa = _fits_beam_geometry(header, fallback_beam)
        if mask_path is not None:
            temporary = tempfile.NamedTemporaryFile(
                prefix=f".{png_path.stem}-mask-",
                suffix=".fits",
                dir=str(png_path.parent),
                delete=False,
            )
            mask_fits_path = Path(temporary.name)
            temporary.close()
            exportfits(
                imagename=str(mask_path),
                fitsimage=str(mask_fits_path),
                overwrite=True,
                dropstokes=False,
                dropdeg=True,
            )
            with fits.open(mask_fits_path) as hdul:
                mask = np.squeeze(np.asarray(hdul[0].data))
            while mask.ndim > 2:
                mask = mask[0]
            if np.any(mask > 0.5) and np.any(mask <= 0.5):
                axis.contour(
                    mask,
                    levels=[0.5],
                    colors="cyan",
                    linewidths=0.8,
                    origin="lower",
                )
        metric_boundary_visible = False
        if metric_region is not None and (
            metric_region.min_radius_beams is not None
            or metric_region.max_radius_beams is not None
        ):
            dx = abs(float(header.get("CDELT1", float("nan"))))
            dy = abs(float(header.get("CDELT2", float("nan"))))
            if bmaj is not None and dx > 0 and dy > 0:
                selected = beam_region_mask(
                    display.shape,
                    (dx * 3600.0, dy * 3600.0),
                    float(bmaj) * 3600.0,
                    metric_region,
                )
                if np.any(selected) and np.any(~selected):
                    # A black underlay keeps the white dashed boundary legible
                    # over both inferno highlights and the cyan CLEAN mask.
                    axis.contour(
                        selected.astype(float),
                        levels=[0.5],
                        colors="#000000",
                        linewidths=2.4,
                        linestyles="--",
                        origin="lower",
                    )
                    axis.contour(
                        selected.astype(float),
                        levels=[0.5],
                        colors="#ffffff",
                        linewidths=1.2,
                        linestyles="--",
                        origin="lower",
                    )
                    metric_boundary_visible = True
        if metric_boundary_visible:
            axis.legend(
                handles=[
                    Line2D(
                        [0], [0], color="#ffffff", linestyle="--", linewidth=1.2,
                        path_effects=[
                            patheffects.Stroke(linewidth=2.4, foreground="#000000"),
                            patheffects.Normal(),
                        ],
                        label="metric region",
                    )
                ],
                loc="upper right",
                framealpha=0.7,
                fontsize="small",
            )
        if draw_beam:
            dx = abs(float(header.get("CDELT1", float("nan"))))
            dy = abs(float(header.get("CDELT2", float("nan"))))
            if bmaj is not None and bmin is not None and dx > 0 and dy > 0:
                height, width = display.shape
                axis.add_patch(
                    Ellipse(
                        (0.12 * width, 0.12 * height),
                        width=float(bmin) / dx,
                        height=float(bmaj) / dy,
                        angle=bpa,
                        fill=False,
                        edgecolor="lime",
                        linewidth=1.5,
                    )
                )
        plt.tight_layout()
        figure.savefig(png_path, dpi=180)
        plt.close(figure)
        return {
            "source_plane": (
                "retained FITS image" if source_is_fits else "squeezed CASA image"
            ),
            "display_unit": "mJy/beam",
            "value_multiplier": 1000.0,
            "vmin": vmin,
            "vmax": vmax,
            "robust_percentile": robust_percentile,
            "symmetric": symmetric,
            "colormap": "inferno",
            "origin": "lower",
            "interpolation": None,
            "figure_size_inches": list(figure.get_size_inches()),
            "dpi": 180,
            "title": title or image_path.name,
            "axis_labels": ["RA", "Dec"],
            "colorbar_label": "mJy/beam",
            "beam": {
                "draw": draw_beam,
                "major_deg": bmaj,
                "minor_deg": bmin,
                "position_angle_deg": bpa,
                "edgecolor": "lime",
                "linewidth": 1.5,
            },
            "metric_region": None
            if metric_region is None
            else {
                "min_radius_beams": metric_region.min_radius_beams,
                "max_radius_beams": metric_region.max_radius_beams,
            },
            "mask_contour_was_rendered": mask_path is not None,
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "astropy": astropy.__version__,
                "matplotlib": matplotlib.__version__,
            },
        }
    finally:
        if owns_fits_path:
            fits_path.unlink(missing_ok=True)
        if mask_fits_path is not None:
            mask_fits_path.unlink(missing_ok=True)


def shared_fits_display_limits(
    image_paths: Sequence[Path],
    *,
    symmetric: bool = True,
    robust_percentile: float = 99.5,
) -> tuple[float, float]:
    """Calculate one mJy/beam color scale shared by retained FITS variants."""
    try:
        import numpy as np
        from astropy.io import fits
    except ImportError as exc:
        raise RuntimeError("Astropy and NumPy are required to scale FITS plots") from exc
    if not image_paths:
        raise ValueError("image_paths must not be empty")
    finite_values = []
    for image_path in image_paths:
        with fits.open(image_path) as hdul:
            values = np.squeeze(np.asarray(hdul[0].data, dtype=float))
        while values.ndim > 2:
            values = values[0]
        if values.ndim != 2:
            raise RuntimeError(
                f"Unexpected image dimensionality {values.ndim} for {image_path}"
            )
        selected = values[np.isfinite(values)] * 1e3
        if selected.size:
            finite_values.append(selected.reshape(-1))
    if not finite_values:
        raise RuntimeError("No finite pixels in FITS comparison")
    combined = np.concatenate(finite_values)
    if symmetric:
        limit = float(np.percentile(np.abs(combined), robust_percentile))
        vmin, vmax = -limit, limit
    else:
        vmin, vmax = (
            float(value)
            for value in np.percentile(
                combined, [100.0 - robust_percentile, robust_percentile]
            )
        )
    if not all(math.isfinite(value) for value in (vmin, vmax)) or vmin >= vmax:
        raise RuntimeError(f"Invalid shared FITS display limits: {(vmin, vmax)!r}")
    return vmin, vmax


def write_fits_comparison_plots(
    samples: Sequence[tuple[str, str, Mapping[str, Path]]],
    output_dir: Path,
    *,
    metric_region: Optional[BeamRegion],
    robust_percentile: float = 99.5,
    display_limits_mjy_per_beam: Optional[
        Mapping[str, tuple[float, float]]
    ] = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Plot retained variants through ``casa_image_to_png`` with shared scales.

    Each tuple is ``(sample_key, title, {dirty, clean, residual})``. All
    variants share one color range per image channel; residual plots also show
    the exact metric annulus. Every panel retains its own WCS axes, colorbar,
    and synthesized-beam glyph from the canonical imaging plotter.
    """
    required = ("dirty", "clean", "residual")
    if not samples:
        raise ValueError("samples must not be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    if display_limits_mjy_per_beam is None:
        limits = {
            channel: shared_fits_display_limits(
                [Path(products[channel]) for _, _, products in samples],
                robust_percentile=robust_percentile,
            )
            for channel in required
        }
    else:
        missing = set(required) - set(display_limits_mjy_per_beam)
        if missing:
            raise ValueError(f"Missing shared display limits for: {sorted(missing)}")
        limits = {
            channel: tuple(display_limits_mjy_per_beam[channel])
            for channel in required
        }
    rows: list[dict[str, object]] = []
    recipes: dict[str, object] = {
        "shared_by": "image channel across all variants of one source",
        "limits_mjy_per_beam": {
            channel: list(channel_limits)
            for channel, channel_limits in limits.items()
        },
        "panels": {},
    }
    panel_recipes = recipes["panels"]
    assert isinstance(panel_recipes, dict)
    for sample_key, title, products in samples:
        row: dict[str, object] = {"key": sample_key, "title": title}
        for channel in required:
            destination = output_dir / f"{sample_key}_{channel}.png"
            recipe = casa_image_to_png(
                Path(products[channel]),
                destination,
                title=f"{title}: {channel}",
                draw_beam=True,
                robust_percentile=robust_percentile,
                metric_region=metric_region if channel == "residual" else None,
                display_limits_mjy_per_beam=limits[channel],
            )
            row[channel] = destination
            panel_recipes[f"{sample_key}:{channel}"] = recipe
        rows.append(row)
    return rows, recipes


def write_individual_plots(
    dirty_image: Path,
    clean_image: Path,
    residual_image: Path,
    output_dir: Path,
    *,
    visibility_id: Optional[str],
    mask_image: Optional[Path] = None,
    metric_region: Optional[BeamRegion] = None,
    fallback_beam: Optional[Beam] = None,
) -> tuple[Path, Path, Path]:
    paths, _ = write_individual_plots_with_recipes(
        dirty_image,
        clean_image,
        residual_image,
        output_dir,
        visibility_id=visibility_id,
        mask_image=mask_image,
        metric_region=metric_region,
        fallback_beam=fallback_beam,
    )
    return paths


def write_individual_plots_with_recipes(
    dirty_image: Path,
    clean_image: Path,
    residual_image: Path,
    output_dir: Path,
    *,
    visibility_id: Optional[str],
    mask_image: Optional[Path] = None,
    metric_region: Optional[BeamRegion] = None,
    fallback_beam: Optional[Beam] = None,
) -> tuple[tuple[Path, Path, Path], dict[str, dict[str, object]]]:
    label = visibility_id or "Measurement Set"
    dirty_png = output_dir / "dirty.png"
    clean_png = output_dir / "clean.png"
    residual_png = output_dir / "residual.png"
    recipes = {}
    recipes["dirty"] = casa_image_to_png(
        dirty_image, dirty_png, title=f"{label} dirty", draw_beam=True
    )
    recipes["clean"] = casa_image_to_png(
        clean_image,
        clean_png,
        title=f"{label} clean",
        mask_path=mask_image,
        draw_beam=True,
    )
    recipes["residual"] = casa_image_to_png(
        residual_image,
        residual_png,
        title=f"{label} residual",
        mask_path=mask_image,
        draw_beam=True,
        metric_region=metric_region,
        fallback_beam=fallback_beam,
    )
    return (dirty_png, clean_png, residual_png), recipes


__all__ = [
    "casa_image_to_png",
    "shared_fits_display_limits",
    "write_fits_comparison_plots",
    "write_individual_plots",
    "write_individual_plots_with_recipes",
]
