"""Individual CASA-image plotting with the established WCS conventions."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Optional

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
) -> None:
    """Render one CASA image in mJy/beam with celestial axes."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np
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
    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{png_path.stem}-", suffix=".fits", dir=str(png_path.parent), delete=False
    )
    fits_path = Path(temporary.name)
    temporary.close()
    mask_fits_path = None
    try:
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
        if symmetric:
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
    finally:
        fits_path.unlink(missing_ok=True)
        if mask_fits_path is not None:
            mask_fits_path.unlink(missing_ok=True)


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
    label = visibility_id or "Measurement Set"
    dirty_png = output_dir / "dirty.png"
    clean_png = output_dir / "clean.png"
    residual_png = output_dir / "residual.png"
    casa_image_to_png(dirty_image, dirty_png, title=f"{label} dirty", draw_beam=True)
    casa_image_to_png(
        clean_image,
        clean_png,
        title=f"{label} clean",
        mask_path=mask_image,
        draw_beam=True,
    )
    casa_image_to_png(
        residual_image,
        residual_png,
        title=f"{label} residual",
        mask_path=mask_image,
        draw_beam=True,
        metric_region=metric_region,
        fallback_beam=fallback_beam,
    )
    return dirty_png, clean_png, residual_png


__all__ = ["casa_image_to_png", "write_individual_plots"]
