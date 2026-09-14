"""Compare real MSs with controlled CASA simulations using standard diagnostics."""

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
from casatasks import imstat

from scripts import image_extracted
from scripts.components import pointsource_constant_1Jy_phasecenter_fromMS, pointsource_0012_399_phasecenter_fromMS
from scripts.plot_extracted import make_residual_title, make_title
from scripts.sim_utils import find_visibility_ms, simulate_ms


OUTPUT_MS_NAME = "simulated_constant_0.7Jy_phasecenter.ms"
BATCH_OUTPUT_DIR = REPO_ROOT / "experiments" / "simulation_constant_0.7Jy_phasecenter"
PROJECT_LIST = REPO_ROOT / "collect" / "small_subset" / "small_selection.csv"
CALIBRATOR_BANDS_CSV = REPO_ROOT / "collect" / "vla_calibrators_bands_v2.csv"
SIMULATION_CLEAN_NITER = 10_000_000
SIMULATION_CLEAN_THRESHOLD = "0.0mJy"
PANEL_SPECS = (
    ("clean", "clean"),
    ("dirty", "dirty"),
    ("residual", "residual"),
    ("uv", "uv coverage"),
    ("amp_uvdist", "amp vs uv-dist"),
    ("amp_uvdist_norm", "amp / median(A)"),
    ("spectrum_by_ant", "spectrum"),
    ("spectrum", "spectrum avg baseline"),
)


def _project_row(visibility_id: str):
    if not PROJECT_LIST.exists():
        return None
    return image_extracted.row_for_folder(
        image_extracted.load_projects(PROJECT_LIST), visibility_id
    )


def _catalog_uv_info(visibility_id: str, band: str | None) -> dict | None:
    if not CALIBRATOR_BANDS_CSV.exists():
        return None
    return image_extracted.lookup_calibrator_uv_limits(
        image_extracted.load_calibrator_uv_limits(CALIBRATOR_BANDS_CSV),
        calibrator_name=visibility_id,
        band=band,
    )


def _export_diagnostics(
    ms_path: Path,
    output_dir: Path,
    *,
    source_kind: str,
    sample_index: int,
    visibility_id: str,
    row,
    band_info: dict,
    cell_arcsec: float,
    imsize: int,
    fov_arcsec: float,
    catalog_info: dict | None,
) -> dict:
    """Run imaging and the eight standard image_extracted exports for one MS."""
    prefix = source_kind.lower()
    applied_uvrange = ""
    catalog_uvrange = ""
    uvmin = uvmax = uv_inside = np.nan
    if catalog_info is not None:
        uvmin = catalog_info["uvmin_kl"]
        uvmax = catalog_info["uvmax_kl"]
        catalog_uvrange = image_extracted.make_uvrange_string(uvmin, uvmax)
        uv_inside = image_extracted.compute_uvlimit_coverage_stats(
            ms_path, band_info["ms_freq_ghz"], uvmin, uvmax
        )["uv_fraction_inside_limits"]

    dirty_base = output_dir / f"{prefix}_dirty"
    image_extracted.run_tclean(
        ms_path,
        str(dirty_base),
        cell_arcsec=cell_arcsec,
        imsize=imsize,
        niter=0,
        uvrange=applied_uvrange,
        use_multiterm_mfs=False,
        apply_multiterm_clean_controls=False,
    )
    dirty_image = dirty_base.with_suffix(".image")
    dirty_residual = dirty_base.with_suffix(".residual")
    dirty_bmaj, dirty_bmin, _ = image_extracted.read_restoring_beam_arcsec(dirty_image)

    clean_niter = image_extracted.FINAL_CLEAN_NITER
    clean_threshold_jy = None
    initial_dirty_peak_jy = None
    if source_kind.casefold() == "simulation":
        stats = imstat(imagename=str(dirty_residual))
        initial_dirty_peak_jy = float(stats["max"][0])
        if not np.isfinite(initial_dirty_peak_jy) or initial_dirty_peak_jy <= 0:
            raise RuntimeError(
                f"Invalid simulated dirty-image peak for {dirty_residual}: "
                f"{initial_dirty_peak_jy}"
            )
        clean_niter = SIMULATION_CLEAN_NITER
        clean_threshold_jy = SIMULATION_CLEAN_THRESHOLD
        print(
            f"[SIMULATION CLEAN CONTROLS] peak={initial_dirty_peak_jy:.9g} Jy/beam | "
            f"threshold={clean_threshold_jy} | niter={clean_niter}"
        )

    clean_usemask = ""
    clean_mask = ""
    mask_nbeams = image_extracted.FINAL_CLEAN_BOX_MASK_NBEAMS
    if mask_nbeams is not None:
        clean_usemask = "user"
        clean_mask = image_extracted.beam_box_mask_string(
            imsize=imsize,
            cell_arcsec=cell_arcsec,
            bmaj_arcsec=dirty_bmaj,
            bmin_arcsec=dirty_bmin,
            nbeams=float(mask_nbeams),
        )

    clean_base = output_dir / f"{prefix}_clean"
    image_extracted.run_tclean(
        ms_path,
        str(clean_base),
        cell_arcsec=cell_arcsec,
        imsize=imsize,
        niter=clean_niter,
        uvrange=applied_uvrange,
        use_multiterm_mfs=False,
        usemask=clean_usemask,
        mask=clean_mask,
        threshold=clean_threshold_jy,
    )
    clean_image = clean_base.with_suffix(".image")
    residual_image = clean_base.with_suffix(".residual")
    beam_major, beam_minor, beam_pa = image_extracted.read_restoring_beam_arcsec(clean_image)
    metrics = image_extracted.clean_residual_qa_metrics(clean_image, residual_image)

    panels = {
        "clean": output_dir / f"{prefix}_clean.png",
        "dirty": output_dir / f"{prefix}_dirty.png",
        "residual": output_dir / f"{prefix}_residual.png",
        "uv": output_dir / f"{prefix}_uv.png",
        "amp_uvdist": output_dir / f"{prefix}_amp_vs_uvdist.png",
        "amp_uvdist_norm": output_dir / f"{prefix}_amp_vs_uvdist_norm.png",
        "spectrum_by_ant": output_dir / f"{prefix}_spectrum_by_ant.png",
        "spectrum": output_dir / f"{prefix}_spectrum.png",
    }
    image_extracted.image_to_png_if_exists(
        clean_image, panels["clean"], title=" ", draw_beam_ellipse=True
    )
    image_extracted.image_to_png_if_exists(
        dirty_image, panels["dirty"], title=" ", draw_beam_ellipse=True
    )
    image_extracted.image_to_png_if_exists(
        residual_image,
        panels["residual"],
        title=" ",
        symmetric=True,
        cmap="inferno",
    )

    exports = (
        ("uv coverage", image_extracted.export_uv_png, panels["uv"], {}),
        ("amp vs uv-dist", image_extracted.export_amp_vs_uvdist_png, panels["amp_uvdist"], {}),
        ("amp / median(A)", image_extracted.export_normalized_amp_vs_uvdist_png, panels["amp_uvdist_norm"], {}),
        ("spectrum by antenna", image_extracted.export_spectrum_png, panels["spectrum_by_ant"], {"avgbaseline": False}),
        ("spectrum", image_extracted.export_spectrum_png, panels["spectrum"], {"avgbaseline": True}),
    )
    for label, function, png_path, extra in exports:
        image_extracted.run_optional_export(
            label,
            function,
            ms_path,
            png_path,
            spw="",
            uvrange=applied_uvrange,
            **extra,
        )

    config = "" if row is None else str(row.get("gain_array_config", ""))
    minutes = None if row is None else row.get("extracted_gain_onsource_min")
    return {
        "sample_index": sample_index,
        "source_kind": source_kind,
        "folder": visibility_id,
        "name": visibility_id,
        "minutes": minutes,
        "beam_major_arcsec": beam_major,
        "beam_minor_arcsec": beam_minor,
        "beam_pa_deg": beam_pa,
        "band_used_for_firstpass": band_info["selected_band"],
        "gain_array_config": config,
        "clean_mode": "standard/hogbom",
        "clean_niter": clean_niter,
        "initial_dirty_peak_jy_per_beam": initial_dirty_peak_jy,
        "clean_threshold_jy": clean_threshold_jy,
        "clean_threshold_fraction": None,
        "catalog_uvrange": catalog_uvrange,
        "catalog_uvmin_kl": uvmin,
        "catalog_uvmax_kl": uvmax,
        "applied_uvrange": applied_uvrange,
        "uv_fraction_inside_limits": uv_inside,
        "cell_arcsec": cell_arcsec,
        "imsize": imsize,
        "fov_arcsec": fov_arcsec,
        "fov_in_beams_minor": fov_arcsec / beam_minor,
        "pixels_per_beam_minor": beam_minor / cell_arcsec,
        "final_clean_mask_mode": "beam_box" if mask_nbeams is not None else "none",
        "final_clean_box_mask_nbeams": mask_nbeams,
        "panels": panels,
        **metrics,
    }


def _clean_title(row: dict) -> str:
    return make_title(row["sample_index"], row).replace(
        "source=original", f"source={row['source_kind'].lower()}"
    )


def _write_diagnostic_sheet(rows: list[dict], output: Path, title: str) -> None:
    """Write alternating original/simulation rows with eight diagnostic columns."""
    if not rows:
        return
    fig, axes = plt.subplots(
        len(rows),
        len(PANEL_SPECS),
        figsize=(4.2 * len(PANEL_SPECS), 5.2 * len(rows)),
        squeeze=False,
    )
    for row_index, row in enumerate(rows):
        for column_index, (key, label) in enumerate(PANEL_SPECS):
            axis = axes[row_index, column_index]
            path = row["panels"].get(key)
            if path is not None and Path(path).exists():
                axis.imshow(mpimg.imread(path), aspect="auto")
            else:
                axis.text(0.5, 0.5, f"{label}\nnot available", ha="center", va="center")
            axis.axis("off")
            if key == "clean":
                axis.set_title(_clean_title(row), fontsize=9)
            elif key == "residual":
                axis.set_title(make_residual_title(row), fontsize=9)
            else:
                axis.set_title(label, fontsize=10)
    fig.suptitle(title, fontsize=16, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] diagnostic sheet: {output}")


def _image_original_and_simulated(
    original_ms: Path,
    simulated_ms: Path,
    *,
    visibility_id: str,
    sample_index: int,
    simulation_description: str = "constant 0.7 Jy phase-centre simulation",
) -> tuple[Path, list[dict]]:
    output_dir = simulated_ms.parent / f"{simulated_ms.stem}_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    row = _project_row(visibility_id)
    band_info = image_extracted.choose_band_and_frequency(original_ms, row=row)
    config = None if row is None else row.get("gain_array_config")
    first_cell, first_imsize, _ = image_extracted.choose_firstpass_imaging_setup(
        config=config,
        reference_freq_ghz=band_info["ms_freq_ghz"],
        band=band_info["selected_band"],
    )

    old_datacolumn = image_extracted.TCLEAN_BASE.get("datacolumn")
    image_extracted.TCLEAN_BASE["datacolumn"] = "data"
    try:
        firstpass = output_dir / "original_firstpass"
        image_extracted.run_tclean(
            original_ms,
            str(firstpass),
            cell_arcsec=first_cell,
            imsize=first_imsize,
            niter=0,
            use_multiterm_mfs=False,
            apply_multiterm_clean_controls=False,
        )
        bmaj, bmin, _ = image_extracted.read_restoring_beam_arcsec(
            firstpass.with_suffix(".image")
        )
        cell_arcsec, imsize, fov_arcsec = image_extracted.choose_final_imaging_setup(bmaj, bmin)
        catalog_info = _catalog_uv_info(visibility_id, band_info["selected_band"])
        diagnostics = [
            _export_diagnostics(
                ms_path,
                output_dir,
                source_kind=source_kind,
                sample_index=sample_index,
                visibility_id=visibility_id,
                row=row,
                band_info=band_info,
                cell_arcsec=cell_arcsec,
                imsize=imsize,
                fov_arcsec=fov_arcsec,
                catalog_info=catalog_info,
            )
            for source_kind, ms_path in (("Original", original_ms), ("Simulation", simulated_ms))
        ]
    finally:
        if old_datacolumn is None:
            image_extracted.TCLEAN_BASE.pop("datacolumn", None)
        else:
            image_extracted.TCLEAN_BASE["datacolumn"] = old_datacolumn

    pair_png = output_dir / "original_vs_simulated_diagnostics.png"
    _write_diagnostic_sheet(
        diagnostics,
        pair_png,
        f"{visibility_id}: original and {simulation_description}",
    )
    return pair_png, diagnostics


def process_visibility_id(visibility_id: str, *, sample_index: int = 0) -> dict:
    visibility_id = str(visibility_id).strip()
    if not visibility_id:
        raise ValueError("visibility_id must not be empty")
    original_ms = find_visibility_ms(visibility_id)
    # source = pointsource_constant_1Jy_phasecenter_fromMS(original_ms)
    source = pointsource_0012_399_phasecenter_fromMS(original_ms)
    simulated_ms = simulate_ms(visibility_id, [source], OUTPUT_MS_NAME)
    comparison_png, diagnostic_rows = _image_original_and_simulated(
        original_ms,
        simulated_ms,
        visibility_id=visibility_id,
        sample_index=sample_index,
    )
    return {
        "status": "ok",
        "visibility_id": visibility_id,
        "original_ms": str(original_ms),
        "simulated_ms": str(simulated_ms),
        "comparison_png": str(comparison_png),
        "diagnostic_rows": diagnostic_rows,
    }


def process_visibility_ids(visibility_ids: list[str]) -> dict:
    requested = [str(value).strip() for value in visibility_ids if str(value).strip()]
    if not requested:
        raise ValueError("visibility_ids must contain at least one non-empty ID")
    results = []
    for index, visibility_id in enumerate(requested):
        print(f"\n[BATCH {index + 1}/{len(requested)}] {visibility_id}")
        try:
            results.append(process_visibility_id(visibility_id, sample_index=index))
        except Exception as exc:
            print(f"[ERROR] {visibility_id}: {exc}")
            results.append({
                "status": "error",
                "visibility_id": visibility_id,
                "error": f"{type(exc).__name__}: {exc}",
            })

    successful = [result for result in results if result["status"] == "ok"]
    failed = [result for result in results if result["status"] != "ok"]
    rows = [row for result in successful for row in result["diagnostic_rows"]]
    BATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    contact_sheet = BATCH_OUTPUT_DIR / "original_vs_simulated_diagnostics.png"
    _write_diagnostic_sheet(
        rows,
        contact_sheet,
        "Original and constant 0.7 Jy phase-centre simulations",
    )
    print(f"\n[BATCH DONE]\n  requested : {len(requested)}")
    print(f"  successful: {len(successful)}\n  failed    : {len(failed)}")
    if successful:
        print(f"  comparison: {contact_sheet}")
    return {
        "requested": len(requested),
        "successful": len(successful),
        "failed": len(failed),
        "contact_sheet": str(contact_sheet) if successful else None,
        "results": results,
    }


def process_0012_399() -> dict:
    return process_visibility_id("0012-399", sample_index=0)


def main() -> dict:
    return process_0012_399()


if __name__ in {"__main__", "<run_path>"}:
    main()
