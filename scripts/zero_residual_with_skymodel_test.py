#!/usr/bin/env python3
"""Test niter=0 tclean with the exact 0.7 Jy simulation sky model.

This differs from visibility-domain ``uvsub``: the exact component definition
used by ``simulations.py`` is rendered into a CASA model image and supplied to
``tclean`` through ``startmodel``.  With ``niter=0``, CLEAN cannot fit any new
components; it can only calculate the residual and restore the supplied model.

Run from CASA with::

    execfile('scripts/zero_residual_with_skymodel_test.py')
"""

from __future__ import annotations

import shutil
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
from casatools import image, table

from scripts import image_extracted
from scripts.components import pointsource_0012_399_phasecenter_fromMS
from scripts.experiment_outputs import write_json
from scripts.sim_utils import _write_component_list, find_visibility_ms


VISIBILITY_ID = "0012-399"
SIMULATED_MS_NAME = "simulated_constant_0.7Jy_phasecenter.ms"
OUTPUT_ROOT = REPO_ROOT / "experiments" / "zero_residual_with_skymodel_test"
PROJECT_LIST = REPO_ROOT / "collect" / "small_subset" / "small_selection.csv"

NITER = 0
DECONVOLVER = "hogbom"
NTERMS = 1
WEIGHTING = "briggs"
ROBUST = 0.5


def _first(stats: dict, key: str) -> float:
    return float(np.asarray(stats[key]).reshape(-1)[0])


def _image_stats(image_path: Path) -> dict:
    stats = imstat(imagename=str(image_path))
    values = image_extracted.load_image_2d(image_path)
    finite = values[np.isfinite(values)]
    minimum = _first(stats, "min")
    maximum = _first(stats, "max")
    return {
        "min": minimum,
        "max": maximum,
        "max_abs": max(abs(minimum), abs(maximum)),
        "rms": _first(stats, "rms"),
        "sigma": _first(stats, "sigma"),
        "sum": _first(stats, "sum"),
        "finite_pixel_count": int(finite.size),
        "nonzero_finite_pixel_count": int(np.count_nonzero(finite)),
        "all_finite_pixels_exactly_zero": bool(
            finite.size and np.all(finite == 0)
        ),
    }


def _summary_value(value):
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
        key: _summary_value(summary[key])
        for key in keys
        if key in summary
    }


def _determine_simulation_geometry(original_ms: Path, output_dir: Path) -> dict:
    """Repeat the beam-derived geometry calculation used by simulations.py."""
    row = None
    if PROJECT_LIST.exists():
        row = image_extracted.row_for_folder(
            image_extracted.load_projects(PROJECT_LIST),
            VISIBILITY_ID,
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
        bmaj,
        bmin,
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


def _tclean_parameters(
    simulated_ms: Path,
    imagename: Path,
    geometry: dict,
) -> dict:
    """Return the standard/Hogbom configuration used by simulations.py."""
    return {
        "vis": str(simulated_ms),
        "imagename": str(imagename),
        "datacolumn": "data",
        "cell": f"{float(geometry['cell_arcsec']):.6f}arcsec",
        "imsize": int(geometry["imsize"]),
        "niter": NITER,
        "threshold": "0.0Jy",
        "uvrange": "",
        "specmode": "mfs",
        "gridder": "standard",
        "stokes": "I",
        "deconvolver": DECONVOLVER,
        "nterms": NTERMS,
        "weighting": WEIGHTING,
        "robust": ROBUST,
        "interactive": False,
        "restoration": True,
        "calcres": True,
        "calcpsf": True,
        "restart": False,
        "savemodel": "none",
        "fullsummary": True,
    }


def _run_dirty(
    simulated_ms: Path,
    output_dir: Path,
    geometry: dict,
) -> tuple[Path, dict, dict]:
    dirty_base = output_dir / "exact_skymodel_dirty"
    image_extracted.remove_casa_products(str(dirty_base))
    parameters = _tclean_parameters(simulated_ms, dirty_base, geometry)
    print(
        f"[DIRTY] niter=0 | cell={parameters['cell']} | "
        f"imsize={parameters['imsize']} | deconvolver={DECONVOLVER}"
    )
    summary = tclean(**parameters)
    return dirty_base, parameters, _summary_digest(summary)


def _make_exact_startmodel(
    template_image: Path,
    fromcomplist_image: Path,
    startmodel: Path,
    component_path: Path,
    source: dict,
) -> dict:
    """Create and materialize the exact component-list image for tclean."""
    for path in (fromcomplist_image, startmodel):
        if path.exists():
            shutil.rmtree(path)

    template_tool = image()
    template_tool.open(str(template_image))
    try:
        shape = [int(value) for value in template_tool.shape()]
        coordinate_tool = template_tool.coordsys()
        try:
            coordinate_record = coordinate_tool.torecord()
        finally:
            coordinate_tool.done()
    finally:
        template_tool.close()

    # This is the same component-list writer and the same source dictionary
    # used by sim_utils.simulate_ms when the 0.7 Jy MS was generated.
    _write_component_list([source], component_path)

    component_image_tool = image()
    try:
        component_image_tool.fromcomplist(
            outfile=str(fromcomplist_image),
            shape=shape,
            cl=str(component_path),
            csys=coordinate_record,
            overwrite=True,
        )
    finally:
        component_image_tool.done()

    # fromcomplist intentionally creates a lazy "Component List" image table.
    # SynthesisImagerVi2 cannot use that table type directly as startmodel, so
    # materialize its evaluated pixels as a traditional CASA Paged image.
    startmodel_tool = image()
    try:
        startmodel_tool.fromimage(
            outfile=str(startmodel),
            infile=str(fromcomplist_image),
            overwrite=True,
        )
    finally:
        startmodel_tool.done()

    table_tool = table()
    table_tool.open(str(startmodel), nomodify=True)
    try:
        startmodel_table_info = table_tool.info()
    finally:
        table_tool.close()
    table_type = str(startmodel_table_info.get("type", ""))
    if table_type.casefold() != "image":
        raise RuntimeError(
            f"Materialized startmodel is not a CASA Image table: "
            f"{startmodel} (type={table_type!r})"
        )
    return {
        "component_list_image": str(fromcomplist_image),
        "materialized_startmodel_image": str(startmodel),
        "materialized_startmodel_table_type": table_type,
    }


def _run_with_exact_skymodel(
    simulated_ms: Path,
    output_dir: Path,
    geometry: dict,
    startmodel: Path,
) -> tuple[Path, dict, dict]:
    clean_base = output_dir / "zero_residual_with_exact_skymodel"
    image_extracted.remove_casa_products(str(clean_base))
    parameters = _tclean_parameters(simulated_ms, clean_base, geometry)
    parameters["startmodel"] = str(startmodel)
    print(f"[STARTMODEL IMAGE] {startmodel.resolve()}")
    print(
        f"[EXACT SKYMODEL] niter=0 | startmodel={startmodel} | "
        "no fitted CLEAN components allowed"
    )
    summary = tclean(**parameters)
    return clean_base, parameters, _summary_digest(summary)


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
    startmodel = report["startmodel_stats"]
    model = report["tclean_model_stats"]
    residual = report["residual_stats"]
    clean = report["restored_image_stats"]
    summary = report["tclean_summary"]
    return [
        "EXACT SKYMODEL, ZERO CLEAN ITERATIONS",
        "",
        f"niter requested / done : {NITER} / {summary.get('iterdone', 'unknown')}",
        f"deconvolver / nterms    : {DECONVOLVER} / {NTERMS}",
        f"weighting / robust      : {WEIGHTING} / {ROBUST}",
        f"fromcomplist image       : {Path(report['fromcomplist_image']).name}",
        f"tclean startmodel image  : {Path(report['startmodel_image']).name}",
        f"startmodel directory     : {Path(report['startmodel_image']).parent}",
        f"stop code                : {summary.get('stopcode', 'unknown')}",
        "",
        f"true source flux         : {report['expected_source_flux_jy']:.12g} Jy",
        f"dirty peak               : {dirty['max']:.12g} Jy/beam",
        f"startmodel peak / sum    : {startmodel['max']:.12g} / {startmodel['sum']:.12g} Jy",
        f"tclean model peak / sum  : {model['max']:.12g} / {model['sum']:.12g} Jy",
        f"restored-image peak      : {clean['max']:.12g} Jy/beam",
        "",
        f"residual min / max       : {residual['min']:+.6e} / {residual['max']:+.6e}",
        f"residual max |pixel|     : {residual['max_abs']:.6e} Jy/beam",
        f"residual RMS             : {residual['rms']:.6e} Jy/beam",
        f"residual sigma           : {residual['sigma']:.6e} Jy/beam",
        f"nonzero residual pixels  : {residual['nonzero_finite_pixel_count']}",
        f"all residual pixels zero : {residual['all_finite_pixels_exactly_zero']}",
        f"residual / dirty peak    : {report['residual_peak_fraction']:.6e}",
        f"dirty/residual corr.     : {report['dirty_residual_correlation']:.6f}",
    ]


def _write_comparison(report: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    panels = (
        ("dirty_png", "Dirty image (niter=0, no model)"),
        ("restored_png", "Restored image (exact startmodel, niter=0)"),
        ("residual_png", "Residual (exact startmodel, niter=0)"),
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
        fontsize=10.2,
    )
    fig.suptitle(
        (
            f"{VISIBILITY_ID}: {report['expected_source_flux_jy']:.9g} Jy simulation, "
            "exact supplied sky model with niter=0"
        ),
        fontsize=16,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> dict:
    original_ms = find_visibility_ms(VISIBILITY_ID)
    simulated_ms = original_ms.parent / SIMULATED_MS_NAME
    if not simulated_ms.exists():
        raise FileNotFoundError(
            f"Missing simulated MS: {simulated_ms}\n"
            "Run execfile('scripts/simulations.py') first."
        )

    source = pointsource_0012_399_phasecenter_fromMS(original_ms)
    expected_source_flux_jy = float(source["flux"][0])
    output_dir = OUTPUT_ROOT / VISIBILITY_ID
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry = _determine_simulation_geometry(original_ms, output_dir)

    dirty_base, dirty_parameters, dirty_summary = _run_dirty(
        simulated_ms,
        output_dir,
        geometry,
    )
    dirty_image = dirty_base.with_suffix(".image")
    dirty_stats = _image_stats(dirty_image)

    component_path = output_dir / "exact_simulation_source.components.cl"
    fromcomplist_image = output_dir / "exact_simulation_skymodel.fromcomplist"
    startmodel = output_dir / "exact_simulation_skymodel.startmodel"
    startmodel_creation = _make_exact_startmodel(
        dirty_image,
        fromcomplist_image,
        startmodel,
        component_path,
        source,
    )
    startmodel_stats = _image_stats(startmodel)

    clean_base, clean_parameters, clean_summary = _run_with_exact_skymodel(
        simulated_ms,
        output_dir,
        geometry,
        startmodel,
    )
    restored_image = clean_base.with_suffix(".image")
    tclean_model = clean_base.with_suffix(".model")
    residual_image = clean_base.with_suffix(".residual")
    restored_stats = _image_stats(restored_image)
    model_stats = _image_stats(tclean_model)
    residual_stats = _image_stats(residual_image)

    dirty_png = output_dir / "exact_skymodel_dirty.png"
    restored_png = output_dir / "exact_skymodel_restored.png"
    residual_png = output_dir / "exact_skymodel_residual.png"
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
        title="Exact supplied sky model; niter=0",
        draw_beam_ellipse=True,
        symmetric=True,
        cmap="inferno",
    )
    image_extracted.image_to_png_if_exists(
        residual_image,
        residual_png,
        title="Residual after exact supplied sky model; niter=0",
        symmetric=True,
        cmap="inferno",
    )

    residual_fraction = (
        residual_stats["max_abs"] / abs(dirty_stats["max"])
        if dirty_stats["max"] != 0
        else float("nan")
    )
    report = {
        "experiment": "zero_residual_with_skymodel_test",
        "visibility_id": VISIBILITY_ID,
        "original_ms": str(original_ms),
        "simulated_ms": str(simulated_ms),
        "exact_source_component": source,
        "component_list": str(component_path),
        "fromcomplist_image": str(fromcomplist_image),
        "startmodel_creation": startmodel_creation,
        "expected_source_flux_jy": expected_source_flux_jy,
        "geometry": geometry,
        "niter": NITER,
        "deconvolver": DECONVOLVER,
        "nterms": NTERMS,
        "weighting": WEIGHTING,
        "robust": ROBUST,
        "dirty_tclean_parameters": dirty_parameters,
        "dirty_tclean_summary": dirty_summary,
        "skymodel_tclean_parameters": clean_parameters,
        "tclean_summary": clean_summary,
        "dirty_image": str(dirty_image),
        "startmodel_image": str(startmodel),
        "tclean_model_image": str(tclean_model),
        "restored_image": str(restored_image),
        "residual_image": str(residual_image),
        "dirty_stats": dirty_stats,
        "startmodel_stats": startmodel_stats,
        "tclean_model_stats": model_stats,
        "restored_image_stats": restored_stats,
        "residual_stats": residual_stats,
        "residual_peak_fraction": residual_fraction,
        "dirty_residual_correlation": _pattern_correlation(
            dirty_image,
            residual_image,
        ),
        "dirty_png": str(dirty_png),
        "restored_png": str(restored_png),
        "residual_png": str(residual_png),
    }

    comparison_path = output_dir / "zero_residual_with_skymodel_comparison.png"
    report_path = output_dir / "zero_residual_with_skymodel_report.json"
    report["comparison_png"] = str(comparison_path)
    report["report_json"] = str(report_path)
    _write_comparison(report, comparison_path)
    write_json(report_path, report)

    print("\n[RESULT]")
    for line in _metric_lines(report):
        print(line)
    print(f"\n  comparison: {comparison_path}")
    print(f"  report    : {report_path}")
    print(f"  startmodel: {startmodel}")
    print(f"  residual  : {residual_image}")
    return report


main()
