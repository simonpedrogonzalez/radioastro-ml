"""Test whether exact visibility-model subtraction images to a zero residual.

This script uses the existing constant 1 Jy phase-centre simulation made by
``simulations.py``.  It performs two controlled experiments with identical
imaging geometry and tclean settings:

1. Re-predict the known component into MODEL_DATA, subtract it from
   CORRECTED_DATA with ``uvsub``, and image the result.
2. Set CORRECTED_DATA to the exact floating-point expression DATA - DATA and
   image that bitwise-zero visibility column as an identity/FFT control.

Each experiment is imaged once with niter=0 (the decisive dirty-image test)
and once with niter=100 (a check that CLEAN has nothing useful left to do).
The working MS is restored to the component-subtracted state before exit.

Run from CASA with::

    execfile('scripts/model_removal_test.py')
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
from casatasks import clearcal, ft, imstat, tclean, uvsub
from casatools import componentlist, table

from scripts import image_extracted
from scripts.components import pointsource_constant_1Jy_phasecenter_fromMS
from scripts.experiment_outputs import write_json
from scripts.io_utils import copy_ms
from scripts.sim_utils import find_visibility_ms


# Change this once to run the experiment for another simulated visibility ID.
VISIBILITY_ID = "0012-399"
SIMULATED_MS_NAME = "simulated_constant_1Jy_phasecenter.ms"

OUTPUT_ROOT = REPO_ROOT / "experiments" / "model_removal_test"
PROJECT_LIST = REPO_ROOT / "collect" / "small_subset" / "small_selection.csv"
CLEAN_NITER = 100
CLEAN_THRESHOLD_JY = 1e-7
VISIBILITY_CHUNK_ROWS = 2048


def _write_exact_component_list(ms_path: Path, component_path: Path) -> None:
    """Recreate the exact source description used by simulations.py."""
    if component_path.exists():
        shutil.rmtree(component_path)

    source = pointsource_constant_1Jy_phasecenter_fromMS(ms_path)
    cl = componentlist()
    try:
        cl.addcomponent(**source)
        cl.rename(str(component_path))
    finally:
        cl.done()


def _column_difference_stats(
    ms_path: Path,
    column_a: str,
    column_b: str | None = None,
) -> dict:
    """Return unflagged complex-column statistics without loading the full MS."""
    count = 0
    sum_abs_squared = 0.0
    max_abs = 0.0
    all_exact_zero = True

    tb = table()
    tb.open(str(ms_path), nomodify=True)
    try:
        columns = set(tb.colnames())
        if column_a not in columns:
            raise RuntimeError(f"{ms_path} has no {column_a} column")
        if column_b is not None and column_b not in columns:
            raise RuntimeError(f"{ms_path} has no {column_b} column")

        for startrow in range(0, tb.nrows(), VISIBILITY_CHUNK_ROWS):
            nrow = min(VISIBILITY_CHUNK_ROWS, tb.nrows() - startrow)
            values = np.asarray(tb.getcol(column_a, startrow=startrow, nrow=nrow))
            if column_b is not None:
                values = values - np.asarray(
                    tb.getcol(column_b, startrow=startrow, nrow=nrow)
                )

            flags = np.asarray(tb.getcol("FLAG", startrow=startrow, nrow=nrow))
            selected = values[~flags]
            finite = selected[np.isfinite(selected)]
            if finite.size == 0:
                continue

            amplitudes = np.abs(finite).astype(np.float64, copy=False)
            count += int(amplitudes.size)
            sum_abs_squared += float(np.sum(amplitudes * amplitudes, dtype=np.float64))
            max_abs = max(max_abs, float(np.max(amplitudes)))
            all_exact_zero = all_exact_zero and bool(np.all(finite == 0))
    finally:
        tb.close()

    return {
        "column_a": column_a,
        "column_b": column_b,
        "unflagged_finite_values": count,
        "max_abs_jy": max_abs if count else float("nan"),
        "rms_abs_jy": (
            float(np.sqrt(sum_abs_squared / count)) if count else float("nan")
        ),
        "all_values_exactly_zero": bool(count and all_exact_zero),
    }


def _set_corrected_to_exact_zero(ms_path: Path) -> None:
    """Set CORRECTED_DATA = DATA - DATA, which must be bitwise zero."""
    clearcal(vis=str(ms_path), addmodel=True)
    tb = table()
    tb.open(str(ms_path), nomodify=False)
    try:
        columns = set(tb.colnames())
        if "DATA" not in columns or "CORRECTED_DATA" not in columns:
            raise RuntimeError(f"DATA/CORRECTED_DATA missing from {ms_path}")
        for startrow in range(0, tb.nrows(), VISIBILITY_CHUNK_ROWS):
            nrow = min(VISIBILITY_CHUNK_ROWS, tb.nrows() - startrow)
            data = np.asarray(tb.getcol("DATA", startrow=startrow, nrow=nrow))
            tb.putcol(
                "CORRECTED_DATA",
                data - data,
                startrow=startrow,
                nrow=nrow,
            )
    finally:
        tb.close()


def _subtract_component_model(ms_path: Path, component_path: Path) -> dict:
    """Predict the exact component, report its mismatch, then run uvsub."""
    # clearcal first guarantees CORRECTED_DATA starts as an unmodified DATA copy.
    clearcal(vis=str(ms_path), addmodel=True)
    ft(
        vis=str(ms_path),
        complist=str(component_path),
        usescratch=True,
        incremental=False,
    )
    prediction_mismatch = _column_difference_stats(ms_path, "DATA", "MODEL_DATA")
    uvsub(vis=str(ms_path), reverse=False)
    corrected_residual = _column_difference_stats(ms_path, "CORRECTED_DATA")
    return {
        "data_minus_repredicted_model": prediction_mismatch,
        "corrected_after_uvsub": corrected_residual,
    }


def _summary_digest(summary) -> dict:
    """Keep the useful scalar convergence fields returned by tclean."""
    if not isinstance(summary, dict):
        return {"returned_type": type(summary).__name__}

    wanted = {
        "stopcode",
        "iterdone",
        "nmajordone",
        "peakres",
        "modflux",
        "iterDone",
        "peakRes",
        "modelFlux",
        "cycleThresh",
    }
    output = {}
    for key, value in summary.items():
        if key not in wanted:
            continue
        if isinstance(value, np.ndarray):
            output[key] = value.tolist()
        elif isinstance(value, np.generic):
            output[key] = value.item()
        else:
            output[key] = value
    output["available_keys"] = sorted(str(key) for key in summary)
    return output


def _run_tclean(
    ms_path: Path,
    imagename: Path,
    *,
    cell_arcsec: float,
    imsize: int,
    niter: int,
    threshold_jy: float | None = None,
    usemask: str = "",
    mask: str = "",
) -> dict:
    """Run the same standard/Hogbom configuration used by simulations.py."""
    image_extracted.remove_casa_products(str(imagename))
    cfg = dict(image_extracted.TCLEAN_BASE)
    cfg.update(
        vis=str(ms_path),
        imagename=str(imagename),
        datacolumn="corrected",
        cell=f"{cell_arcsec:.6f}arcsec",
        imsize=int(imsize),
        niter=int(niter),
        uvrange="",
        deconvolver="hogbom",
        fullsummary=True,
    )
    if threshold_jy is not None:
        cfg["threshold"] = float(threshold_jy)
    if usemask:
        cfg["usemask"] = usemask
    if mask:
        cfg["mask"] = mask

    print(
        f"[TCLEAN] {imagename.name} | niter={niter} | "
        f"threshold={cfg.get('threshold', 'default')} Jy | "
        f"cell={cell_arcsec:.6f}arcsec | imsize={imsize}"
    )
    return _summary_digest(tclean(**cfg))


def _image_stats(image_path: Path) -> dict:
    stats = imstat(imagename=str(image_path))

    def first(key: str) -> float:
        values = np.asarray(stats[key]).reshape(-1)
        return float(values[0])

    minimum = first("min")
    maximum = first("max")
    return {
        "min_jy_per_beam": minimum,
        "max_jy_per_beam": maximum,
        "max_abs_jy_per_beam": max(abs(minimum), abs(maximum)),
        "rms_jy_per_beam": first("rms"),
        "sigma_jy_per_beam": first("sigma"),
    }


def _determine_simulation_geometry(original_ms: Path, output_dir: Path) -> dict:
    """Repeat simulations.py's first-pass beam-derived grid calculation."""
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


def _image_experiment(
    ms_path: Path,
    output_dir: Path,
    label: str,
    geometry: dict,
) -> dict:
    cell_arcsec = float(geometry["cell_arcsec"])
    imsize = int(geometry["imsize"])

    dirty_base = output_dir / f"{label}_dirty"
    dirty_summary = _run_tclean(
        ms_path,
        dirty_base,
        cell_arcsec=cell_arcsec,
        imsize=imsize,
        niter=0,
    )
    dirty_residual = dirty_base.with_suffix(".residual")
    dirty_stats = _image_stats(dirty_residual)
    bmaj, bmin, _ = image_extracted.read_restoring_beam_arcsec(
        dirty_base.with_suffix(".image")
    )
    mask_nbeams = image_extracted.FINAL_CLEAN_BOX_MASK_NBEAMS
    mask = ""
    if mask_nbeams is not None:
        mask = image_extracted.beam_box_mask_string(
            imsize=imsize,
            cell_arcsec=cell_arcsec,
            bmaj_arcsec=bmaj,
            bmin_arcsec=bmin,
            nbeams=float(mask_nbeams),
        )

    clean_base = output_dir / f"{label}_clean"
    clean_summary = _run_tclean(
        ms_path,
        clean_base,
        cell_arcsec=cell_arcsec,
        imsize=imsize,
        niter=CLEAN_NITER,
        threshold_jy=CLEAN_THRESHOLD_JY,
        usemask="user" if mask else "",
        mask=mask,
    )
    clean_residual = clean_base.with_suffix(".residual")
    clean_stats = _image_stats(clean_residual)

    dirty_png = output_dir / f"{label}_dirty_residual.png"
    clean_png = output_dir / f"{label}_clean_residual.png"
    image_extracted.image_to_png_if_exists(
        dirty_residual,
        dirty_png,
        title=f"{label}: niter=0 residual",
        symmetric=True,
        cmap="inferno",
    )
    image_extracted.image_to_png_if_exists(
        clean_residual,
        clean_png,
        title=f"{label}: niter={CLEAN_NITER} residual",
        symmetric=True,
        cmap="inferno",
    )

    return {
        "dirty_residual_image": str(dirty_residual),
        "dirty_residual_png": str(dirty_png),
        "dirty_stats": dirty_stats,
        "dirty_tclean_summary": dirty_summary,
        "clean_residual_image": str(clean_residual),
        "clean_residual_png": str(clean_png),
        "clean_stats": clean_stats,
        "clean_tclean_summary": clean_summary,
        "mask": mask,
    }


def _write_comparison(results: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    rows = (
        ("component_subtraction", "Re-predicted exact component subtraction"),
        ("identity_zero", "Identity control: DATA - DATA"),
    )
    columns = (
        ("dirty_residual_png", "niter=0"),
        ("clean_residual_png", f"niter={CLEAN_NITER}"),
    )
    for row_index, (result_key, row_title) in enumerate(rows):
        for column_index, (png_key, column_title) in enumerate(columns):
            axis = axes[row_index, column_index]
            axis.imshow(mpimg.imread(results[result_key][png_key]))
            axis.set_title(f"{row_title}\n{column_title}")
            axis.axis("off")
    fig.suptitle(f"{VISIBILITY_ID}: exact-model removal controls", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _conclusion(component: dict, identity: dict) -> str:
    identity_vis_zero = identity["visibility_stats"]["all_values_exactly_zero"]
    identity_image_peak = identity["dirty_stats"]["max_abs_jy_per_beam"]
    component_vis_peak = component["visibility_stats"]["max_abs_jy"]
    component_image_peak = component["dirty_stats"]["max_abs_jy_per_beam"]

    if not identity_vis_zero:
        return "Identity control failed: CORRECTED_DATA was not exactly zero."
    if identity_image_peak != 0.0:
        return (
            "Exactly zero visibilities produced a nonzero dirty image; the imaging "
            "transform introduced numerical residuals."
        )
    if component_vis_peak == 0.0 and component_image_peak == 0.0:
        return "The exact component subtraction produced exactly zero visibilities and image."
    return (
        "Imaging exactly zero visibilities produced exactly zero, but re-predicting "
        "and subtracting the exact sky component left a finite visibility residual. "
        "Any component-subtraction image structure therefore comes from finite-precision "
        "forward prediction/subtraction, not from taking an FFT of zero."
    )


def main() -> dict:
    original_ms = find_visibility_ms(VISIBILITY_ID)
    simulated_ms = original_ms.parent / SIMULATED_MS_NAME
    if not simulated_ms.exists():
        raise FileNotFoundError(
            f"Missing simulated MS: {simulated_ms}\n"
            "Run execfile('scripts/simulations.py') first."
        )

    output_dir = OUTPUT_ROOT / VISIBILITY_ID
    output_dir.mkdir(parents=True, exist_ok=True)
    working_ms = output_dir / f"{Path(SIMULATED_MS_NAME).stem}_model_removed.ms"
    component_path = output_dir / "exact_1Jy_phasecentre.components.cl"

    print(f"[INPUT] simulated MS: {simulated_ms}")
    print(f"[OUTPUT] test directory: {output_dir}")
    copy_ms(str(simulated_ms), str(working_ms))
    _write_exact_component_list(simulated_ms, component_path)
    geometry = _determine_simulation_geometry(original_ms, output_dir)

    component_visibility = _subtract_component_model(working_ms, component_path)
    component_images = _image_experiment(
        working_ms,
        output_dir,
        "component_subtraction",
        geometry,
    )

    _set_corrected_to_exact_zero(working_ms)
    identity_visibility = _column_difference_stats(working_ms, "CORRECTED_DATA")
    identity_images = _image_experiment(
        working_ms,
        output_dir,
        "identity_zero",
        geometry,
    )

    # Leave the persistent working MS in the scientifically useful state.
    restored_visibility = _subtract_component_model(working_ms, component_path)

    results = {
        "visibility_id": VISIBILITY_ID,
        "original_ms": str(original_ms),
        "simulated_ms": str(simulated_ms),
        "model_removed_working_ms": str(working_ms),
        "component_list": str(component_path),
        "geometry": geometry,
        "clean_niter": CLEAN_NITER,
        "clean_threshold_jy": CLEAN_THRESHOLD_JY,
        "component_subtraction": {
            "visibility_stats": component_visibility["corrected_after_uvsub"],
            "prediction_mismatch": component_visibility[
                "data_minus_repredicted_model"
            ],
            **component_images,
        },
        "identity_zero": {
            "visibility_stats": identity_visibility,
            **identity_images,
        },
        "restored_component_subtraction": restored_visibility,
    }
    results["conclusion"] = _conclusion(
        results["component_subtraction"], results["identity_zero"]
    )

    report_path = output_dir / "model_removal_report.json"
    comparison_path = output_dir / "model_removal_comparison.png"
    _write_comparison(results, comparison_path)
    results["report_json"] = str(report_path)
    results["comparison_png"] = str(comparison_path)
    write_json(report_path, results)

    print("\n[RESULT]")
    print(results["conclusion"])
    print(f"  report     : {report_path}")
    print(f"  comparison : {comparison_path}")
    print(f"  residual MS: {working_ms}")
    return results


main()
