"""Test one Högbom iteration on the fitted 0.700962 Jy simulation.

The experiment finds the actual dirty-image peak, restricts CLEAN to exactly
that pixel, and runs ``tclean`` with ``niter=1`` and ``gain=1.0``.  A comparison
PNG displays the dirty image, restored image, residual, and numerical metrics.

Run from the repository root inside CASA with::

    execfile('scripts/remove_all_at_once_test.py')
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from casatasks import imstat, tclean

from scripts import image_extracted
from scripts.components import pointsource_0012_399_phasecenter_fromMS
from scripts.experiment_outputs import write_json
from scripts.sim_utils import find_visibility_ms


# Change this once to select another simulated visibility dataset.
VISIBILITY_ID = "0012-399"
SIMULATED_MS_NAME = "simulated_constant_0.7Jy_phasecenter.ms"
PROJECT_LIST = REPO_ROOT / "collect" / "small_subset" / "small_selection.csv"
OUTPUT_ROOT = REPO_ROOT / "experiments" / "remove_all_at_once_test_07"

NITER = 1
LOOP_GAIN = 1.0
THRESHOLD_JY = 0.0
DECONVOLVER = "hogbom"
NTERMS = 1
WEIGHTING = "briggs"
ROBUST = 0.5


def _product_path(image_base: Path, product: str) -> Path:
    """Return the CASA product path, including MT-MFS's Taylor-term suffix."""
    suffix = f".{product}"
    if DECONVOLVER.casefold() == "mtmfs":
        suffix += ".tt0"
    return Path(f"{image_base}{suffix}")


def _first(stats: dict, key: str) -> float:
    values = np.asarray(stats[key]).reshape(-1)
    return float(values[0])


def _image_stats(image_path: Path) -> dict:
    stats = imstat(imagename=str(image_path))
    minimum = _first(stats, "min")
    maximum = _first(stats, "max")
    output = {
        "min": minimum,
        "max": maximum,
        "max_abs": max(abs(minimum), abs(maximum)),
        "rms": _first(stats, "rms"),
        "sigma": _first(stats, "sigma"),
        "sum": _first(stats, "sum"),
    }
    maxpos = np.asarray(stats.get("maxpos", [])).reshape(-1)
    if maxpos.size >= 2:
        output["max_position_xy_pix"] = [int(maxpos[0]), int(maxpos[1])]
    return output


def _summary_value(summary: dict, key: str, default=None):
    value = summary.get(key, default)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _summary_digest(summary) -> dict:
    if not isinstance(summary, dict):
        return {"returned_type": type(summary).__name__}
    keys = (
        "iterdone",
        "nmajordone",
        "stopcode",
        "stopDescription",
        "peakres",
        "modflux",
    )
    return {
        key: _summary_value(summary, key)
        for key in keys
        if key in summary
    }


def _determine_simulation_geometry(original_ms: Path, output_dir: Path) -> dict:
    """Repeat the beam-derived geometry calculation in simulations.py."""
    row = None
    if PROJECT_LIST.exists():
        row = image_extracted.row_for_folder(
            image_extracted.load_projects(PROJECT_LIST), VISIBILITY_ID
        )
    band_info = image_extracted.choose_band_and_frequency(original_ms, row=row)
    config = None if row is None else row.get("gain_array_config")
    first_cell, first_imsize, _ = image_extracted.choose_firstpass_imaging_setup(
        config=config,
        reference_freq_ghz=band_info["ms_freq_ghz"],
        band=band_info["selected_band"],
    )

    firstpass = output_dir / "original_firstpass"
    old_datacolumn = image_extracted.TCLEAN_BASE.get("datacolumn")
    image_extracted.TCLEAN_BASE["datacolumn"] = "data"
    try:
        image_extracted.run_tclean(
            original_ms,
            str(firstpass),
            cell_arcsec=first_cell,
            imsize=first_imsize,
            niter=0,
            use_multiterm_mfs=False,
            apply_multiterm_clean_controls=False,
        )
    finally:
        if old_datacolumn is None:
            image_extracted.TCLEAN_BASE.pop("datacolumn", None)
        else:
            image_extracted.TCLEAN_BASE["datacolumn"] = old_datacolumn

    bmaj, bmin, beam_pa = image_extracted.read_restoring_beam_arcsec(
        firstpass.with_suffix(".image")
    )
    cell_arcsec, imsize, fov_arcsec = image_extracted.choose_final_imaging_setup(
        bmaj, bmin
    )
    return {
        "cell_arcsec": cell_arcsec,
        "imsize": imsize,
        "fov_arcsec": fov_arcsec,
        "firstpass_beam_major_arcsec": bmaj,
        "firstpass_beam_minor_arcsec": bmin,
        "firstpass_beam_pa_deg": beam_pa,
        "selected_band": band_info["selected_band"],
    }


def _run_dirty(
    simulated_ms: Path,
    output_dir: Path,
    geometry: dict,
) -> tuple[Path, dict]:
    dirty_base = output_dir / "one_iteration_dirty"
    image_extracted.remove_casa_products(str(dirty_base))
    cfg = dict(image_extracted.TCLEAN_BASE)
    cfg.update(
        vis=str(simulated_ms),
        imagename=str(dirty_base),
        datacolumn="data",
        cell=f"{float(geometry['cell_arcsec']):.6f}arcsec",
        imsize=int(geometry["imsize"]),
        niter=0,
        uvrange="",
        deconvolver=DECONVOLVER,
        nterms=NTERMS,
        weighting=WEIGHTING,
        robust=ROBUST,
        fullsummary=True,
    )
    print(
        f"[DIRTY] deconvolver={DECONVOLVER} | nterms={NTERMS} | "
        f"weighting={WEIGHTING} | robust={ROBUST}"
    )
    tclean(**cfg)

    dirty_residual = _product_path(dirty_base, "residual")
    return dirty_base, _image_stats(dirty_residual)


def _run_one_iteration(
    simulated_ms: Path,
    output_dir: Path,
    geometry: dict,
    peak_xy: tuple[int, int],
) -> tuple[Path, dict]:
    clean_base = output_dir / "remove_all_at_once"
    image_extracted.remove_casa_products(str(clean_base))
    xpix, ypix = peak_xy
    one_pixel_mask = f"box[[{xpix}pix,{ypix}pix],[{xpix}pix,{ypix}pix]]"

    cfg = dict(image_extracted.TCLEAN_BASE)
    cfg.update(
        vis=str(simulated_ms),
        imagename=str(clean_base),
        datacolumn="data",
        cell=f"{float(geometry['cell_arcsec']):.6f}arcsec",
        imsize=int(geometry["imsize"]),
        niter=NITER,
        gain=LOOP_GAIN,
        threshold=THRESHOLD_JY,
        uvrange="",
        deconvolver=DECONVOLVER,
        nterms=NTERMS,
        weighting=WEIGHTING,
        robust=ROBUST,
        usemask="user",
        mask=one_pixel_mask,
        fullsummary=True,
    )
    print(
        f"[REMOVE ALL AT ONCE] niter={NITER} | gain={LOOP_GAIN} | "
        f"threshold={THRESHOLD_JY} Jy | mask={one_pixel_mask} | "
        f"deconvolver={DECONVOLVER} | nterms={NTERMS} | "
        f"weighting={WEIGHTING} | robust={ROBUST}"
    )
    summary = tclean(**cfg)
    return clean_base, {
        "mask": one_pixel_mask,
        "tclean_summary": _summary_digest(summary),
    }


def _pattern_correlation(image_a: Path, image_b: Path) -> float:
    a = image_extracted.load_image_2d(image_a).reshape(-1)
    b = image_extracted.load_image_2d(image_b).reshape(-1)
    finite = np.isfinite(a) & np.isfinite(b)
    a = a[finite]
    b = b[finite]
    if a.size == 0 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _metric_lines(report: dict) -> list[str]:
    dirty = report["dirty_stats"]
    model = report["model_stats"]
    residual = report["residual_stats"]
    clean = report["restored_image_stats"]
    summary = report["tclean"]
    return [
        "REMOVE ALL AT ONCE",
        "",
        f"niter requested / done : {NITER} / {summary.get('iterdone', 'unknown')}",
        f"loop gain               : {LOOP_GAIN:.1f}",
        f"deconvolver             : {DECONVOLVER}",
        f"nterms                  : {NTERMS}",
        f"weighting / robust      : {WEIGHTING} / {ROBUST}",
        f"mask                     : {report['one_pixel_mask']}",
        f"stop code                : {summary.get('stopcode', 'unknown')}",
        "",
        f"true source flux         : {report['expected_source_flux_jy']:.12g} Jy",
        f"dirty peak               : {dirty['max']:.12g} Jy/beam",
        f"model peak               : {model['max']:.12g} Jy/pixel",
        f"model sum                : {model['sum']:.12g} Jy",
        f"model - true             : {report['model_flux_error_jy']:+.6e} Jy",
        "",
        f"residual max |pixel|     : {residual['max_abs']:.6e} Jy/beam",
        f"residual RMS             : {residual['rms']:.6e} Jy/beam",
        f"residual sigma           : {residual['sigma']:.6e} Jy/beam",
        f"residual / dirty peak    : {report['residual_peak_fraction']:.6e}",
        f"restored-image peak      : {clean['max']:.12g} Jy/beam",
        f"dirty/residual corr.     : {report['dirty_residual_correlation']:.6f}",
    ]


def _write_comparison(report: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    panels = (
        ("dirty_png", "Dirty image (niter=0)"),
        ("restored_png", "Restored image (one iteration)"),
        ("residual_png", "Residual after one 100% component"),
    )
    for axis, (key, title) in zip(axes.flat[:3], panels):
        axis.imshow(mpimg.imread(report[key]))
        axis.set_title(title)
        axis.axis("off")

    metric_axis = axes.flat[3]
    metric_axis.axis("off")
    metric_axis.text(
        0.02,
        0.98,
        "\n".join(_metric_lines(report)),
        transform=metric_axis.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=11,
    )
    fig.suptitle(
        (
            f"{VISIBILITY_ID}: {report['expected_source_flux_jy']:.9g} Jy "
            f"simulation, one {DECONVOLVER} iteration with gain=1.0"
        ),
        fontsize=16,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> dict:
    original_ms = find_visibility_ms(VISIBILITY_ID)
    expected_source_flux_jy = float(
        pointsource_0012_399_phasecenter_fromMS(original_ms)["flux"][0]
    )
    simulated_ms = original_ms.parent / SIMULATED_MS_NAME
    if not simulated_ms.exists():
        raise FileNotFoundError(
            f"Missing simulated MS: {simulated_ms}\n"
            "Run execfile('scripts/simulations.py') first."
        )

    output_dir = OUTPUT_ROOT / VISIBILITY_ID
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry = _determine_simulation_geometry(original_ms, output_dir)
    dirty_base, dirty_stats = _run_dirty(simulated_ms, output_dir, geometry)

    peak_position = dirty_stats.get("max_position_xy_pix")
    if peak_position is None:
        raise RuntimeError("CASA imstat did not return the dirty-peak pixel position")
    peak_xy = (int(peak_position[0]), int(peak_position[1]))
    clean_base, clean_run = _run_one_iteration(
        simulated_ms, output_dir, geometry, peak_xy
    )

    dirty_image = _product_path(dirty_base, "image")
    restored_image = _product_path(clean_base, "image")
    model_image = _product_path(clean_base, "model")
    residual_image = _product_path(clean_base, "residual")
    model_stats = _image_stats(model_image)
    residual_stats = _image_stats(residual_image)
    restored_stats = _image_stats(restored_image)

    dirty_png = output_dir / "one_iteration_dirty.png"
    restored_png = output_dir / "remove_all_at_once_restored.png"
    residual_png = output_dir / "remove_all_at_once_residual.png"
    image_extracted.image_to_png_if_exists(
        dirty_image,
        dirty_png,
        title="Dirty image",
        draw_beam_ellipse=True,
        symmetric=True,
        cmap="inferno",
    )
    image_extracted.image_to_png_if_exists(
        restored_image,
        restored_png,
        title="One iteration, gain=1.0",
        draw_beam_ellipse=True,
        symmetric=True,
        cmap="inferno",
    )
    image_extracted.image_to_png_if_exists(
        residual_image,
        residual_png,
        title="Residual after removing 100% of first component",
        symmetric=True,
        cmap="inferno",
    )

    model_flux_error = model_stats["sum"] - expected_source_flux_jy
    residual_fraction = (
        residual_stats["max_abs"] / abs(dirty_stats["max"])
        if dirty_stats["max"] != 0
        else float("nan")
    )
    report = {
        "visibility_id": VISIBILITY_ID,
        "original_ms": str(original_ms),
        "simulated_ms": str(simulated_ms),
        "expected_source_flux_jy": expected_source_flux_jy,
        "geometry": geometry,
        "niter": NITER,
        "gain": LOOP_GAIN,
        "threshold_jy": THRESHOLD_JY,
        "deconvolver": DECONVOLVER,
        "nterms": NTERMS,
        "weighting": WEIGHTING,
        "robust": ROBUST,
        "one_pixel_mask": clean_run["mask"],
        "tclean": clean_run["tclean_summary"],
        "dirty_image": str(dirty_image),
        "restored_image": str(restored_image),
        "model_image": str(model_image),
        "residual_image": str(residual_image),
        "dirty_stats": dirty_stats,
        "model_stats": model_stats,
        "residual_stats": residual_stats,
        "restored_image_stats": restored_stats,
        "model_flux_error_jy": model_flux_error,
        "residual_peak_fraction": residual_fraction,
        "dirty_residual_correlation": _pattern_correlation(
            dirty_image, residual_image
        ),
        "dirty_png": str(dirty_png),
        "restored_png": str(restored_png),
        "residual_png": str(residual_png),
    }

    comparison_path = output_dir / "remove_all_at_once_comparison.png"
    report_path = output_dir / "remove_all_at_once_report.json"
    report["comparison_png"] = str(comparison_path)
    report["report_json"] = str(report_path)
    _write_comparison(report, comparison_path)
    write_json(report_path, report)

    print("\n[RESULT]")
    for line in _metric_lines(report):
        print(line)
    print(f"\n  comparison: {comparison_path}")
    print(f"  report    : {report_path}")
    print(f"  residual  : {residual_image}")
    return report


if __name__ == "__main__":
    main()
