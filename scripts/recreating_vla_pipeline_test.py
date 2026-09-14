#!/usr/bin/env python3
"""Compare a controlled direct CASA CLEAN with the VLA Imaging Pipeline.

The direct branch deliberately uses one dirty ``tclean`` call followed by one
clean ``tclean`` call with fully specified imaging controls.  The VLA Pipeline
branch and comparison plot use the same machinery as
``vla_imaging_pipeline_test.py``.

Run with CASA 6.6.6-18 plus VLA Pipeline 2025.1.0.36::

    casa --pipeline --nogui --nologger -c scripts/recreating_vla_pipeline_test.py
"""

from __future__ import annotations

import json
import importlib
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from casatasks import imstat, tclean

from scripts.experiment_outputs import json_ready, write_json
from scripts import vla_imaging_pipeline_test as comparison

# CASA sessions retain imported modules between execfile() calls.  Reload this
# helper so edits to its plotting API are picked up without restarting CASA.
comparison = importlib.reload(comparison)


VISIBILITY_ID = "0012-399"
SOURCE_MS = (
    ROOT
    / "collect"
    / "extracted"
    / VISIBILITY_ID
    / VISIBILITY_ID
    / "simulated_constant_0.7Jy_phasecenter.ms"
)
EXPERIMENTS = ROOT / "collect" / "experiments"
DIRECT_METHOD_LABEL = "Direct CASA VLA-parameter recreation"

# Direct CASA imaging controls.
IMSIZE = 256
CELL_ARCSEC = 0.15
FOV_ARCSEC = IMSIZE * CELL_ARCSEC
MASK_BEAM_MAJOR_ARCSEC = 3.092
MASK_BEAM_MINOR_ARCSEC = 0.7483
RESTORING_BEAM_PA_DEG = 13.7018
WEIGHTING = "briggs"
ROBUST = 0.5
DECONVOLVER = "mtmfs"
NTERMS = 1
NITER = 10_000
THRESHOLD = "4.51e-05Jy"
THRESHOLD_JY = 4.51e-05
NSIGMA = 4.0
USEMASK = "auto-multithresh"
SIDELOBE_THRESHOLD = 2.0
NOISE_THRESHOLD = 5.0
LOW_NOISE_THRESHOLD = 1.5
MIN_BEAM_FRACTION = 0.3
NEGATIVE_THRESHOLD = 0.0
PLOT_PARAMETER_EXCLUSIONS = {
    "vis",
    "imagename",
    "interactive",
    "parallel",
    "fullsummary",
}


def first_stat(stats: dict, key: str) -> float:
    return float(np.asarray(stats[key]).reshape(-1)[0])


def direct_tclean_base(input_ms: Path, imagename: Path) -> dict[str, Any]:
    """Parameters shared by the direct dirty and clean CASA calls."""
    return {
        "vis": str(input_ms),
        "imagename": str(imagename),
        "datacolumn": "data",
        "specmode": "mfs",
        "gridder": "standard",
        "imsize": [IMSIZE, IMSIZE],
        "cell": [f"{CELL_ARCSEC}arcsec"],
        "stokes": "I",
        "weighting": WEIGHTING,
        "robust": ROBUST,
        "deconvolver": DECONVOLVER,
        "nterms": NTERMS,
        "nsigma": NSIGMA,
        "restoringbeam": [
            f"{MASK_BEAM_MAJOR_ARCSEC}arcsec",
            f"{MASK_BEAM_MINOR_ARCSEC}arcsec",
            f"{RESTORING_BEAM_PA_DEG}deg",
        ],
        "savemodel": "none",
        "interactive": False,
        "parallel": False,
        "fullsummary": True,
    }


def run_direct_recreation(input_ms: Path, work: Path) -> tuple[dict[str, Path], dict]:
    """Create direct CASA dirty/clean products without calling an imaging helper."""
    dirty_base = work / "direct_recreation_dirty"
    dirty_parameters = direct_tclean_base(input_ms, dirty_base)
    dirty_parameters.update(niter=0, threshold="0.0Jy")
    print(
        f"[DIRECT DIRTY] imsize={IMSIZE}x{IMSIZE} | cell={CELL_ARCSEC}arcsec | "
        f"FoV={FOV_ARCSEC}arcsec | weighting={WEIGHTING} | robust={ROBUST} | "
        f"deconvolver={DECONVOLVER} | nterms={NTERMS}"
    )
    tclean(**dirty_parameters)

    dirty_residual = Path(f"{dirty_base}.residual.tt0")
    dirty_stats = imstat(imagename=str(dirty_residual))
    dirty_min = first_stat(dirty_stats, "min")
    dirty_max = first_stat(dirty_stats, "max")
    dirty_peak = max(abs(dirty_min), abs(dirty_max))
    if not np.isfinite(dirty_peak) or dirty_peak <= 0:
        raise RuntimeError(f"Invalid dirty peak {dirty_peak} from {dirty_residual}")
    clean_base = work / "direct_recreation_clean"
    clean_parameters = direct_tclean_base(input_ms, clean_base)
    clean_parameters.update(
        niter=NITER,
        threshold=THRESHOLD,
        usemask=USEMASK,
        sidelobethreshold=SIDELOBE_THRESHOLD,
        noisethreshold=NOISE_THRESHOLD,
        lownoisethreshold=LOW_NOISE_THRESHOLD,
        minbeamfrac=MIN_BEAM_FRACTION,
        negativethreshold=NEGATIVE_THRESHOLD,
    )
    print(
        f"[DIRECT CLEAN] niter={NITER} | dirty peak={dirty_peak:.9g} Jy/beam | "
        f"threshold={THRESHOLD} | nsigma={NSIGMA} | usemask={USEMASK}"
    )
    clean_summary = tclean(**clean_parameters)

    products = {
        "dirty": dirty_residual,
        "clean": Path(f"{clean_base}.image.tt0"),
        "residual": Path(f"{clean_base}.residual.tt0"),
        "mask": Path(f"{clean_base}.mask"),
    }
    missing = [str(path) for path in products.values() if not path.exists()]
    if missing:
        raise RuntimeError("Missing direct CASA products:\n" + "\n".join(missing))

    summary = {}
    if isinstance(clean_summary, dict):
        for key in ("iterdone", "nmajordone", "stopcode", "stopDescription", "peakres", "modflux"):
            if key in clean_summary:
                value = clean_summary[key]
                summary[key] = value.tolist() if isinstance(value, np.ndarray) else value

    details = {
        "requested_geometry": {
            "imsize": [IMSIZE, IMSIZE],
            "cell_arcsec": CELL_ARCSEC,
            "fov_arcsec": FOV_ARCSEC,
        },
        "mask": {
            "usemask": USEMASK,
            "sidelobethreshold": SIDELOBE_THRESHOLD,
            "noisethreshold": NOISE_THRESHOLD,
            "lownoisethreshold": LOW_NOISE_THRESHOLD,
            "minbeamfrac": MIN_BEAM_FRACTION,
            "negativethreshold": NEGATIVE_THRESHOLD,
        },
        "requested_restoring_beam": {
            "major_arcsec": MASK_BEAM_MAJOR_ARCSEC,
            "minor_arcsec": MASK_BEAM_MINOR_ARCSEC,
            "position_angle_deg": RESTORING_BEAM_PA_DEG,
        },
        "dirty_peak_jy_per_beam": dirty_peak,
        "threshold": THRESHOLD,
        "threshold_jy": THRESHOLD_JY,
        "nsigma": NSIGMA,
        "dirty_tclean_parameters": dirty_parameters,
        "clean_tclean_parameters": clean_parameters,
        "clean_tclean_summary": summary,
    }
    return products, details


def parameter_lines(parameters: dict[str, Any], width: int = 76) -> list[str]:
    """Compact all reproducibility-relevant parameters into readable lines."""
    tokens = [
        f"{key}={value!r}"
        for key, value in parameters.items()
        if key not in PLOT_PARAMETER_EXCLUSIONS
    ]
    lines: list[str] = []
    current = ""
    for token in tokens:
        candidate = token if not current else f"{current}  {token}"
        if len(candidate) <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = ""
        wrapped = textwrap.wrap(
            token,
            width=width,
            subsequent_indent="    ",
            break_long_words=False,
            break_on_hyphens=False,
        )
        if len(wrapped) == 1:
            current = wrapped[0]
        else:
            lines.extend(wrapped)
    if current:
        lines.append(current)
    return lines


def method_plot_text(
    name: str,
    data: dict,
    parameters: dict[str, Any],
    *,
    qa: dict | None = None,
) -> str:
    """Metrics plus all displayed final-tclean arguments for one method."""
    lines = [
        comparison.stats_text(name, data, qa=qa),
        "",
        "Final tclean parameters:",
        *parameter_lines(parameters),
    ]
    return "\n".join(lines)


def write_results(
    output: Path,
    work: Path,
    pipeline_dir: Path,
    input_info: dict,
    field: str,
    direct_products: dict[str, Path],
    direct_details: dict,
) -> dict:
    log = pipeline_dir / "html" / "casa_commands.log"
    calls = comparison.parse_tclean_calls(log)
    pipeline_products, dirty_params, clean_params = comparison.pipeline_products(work, calls)

    direct_geometry = comparison.image_geometry(direct_products["clean"])
    pipeline_geometry = comparison.image_geometry(pipeline_products["clean"])
    fov = min(
        direct_geometry["fov_x_arcsec"],
        direct_geometry["fov_y_arcsec"],
        pipeline_geometry["fov_x_arcsec"],
        pipeline_geometry["fov_y_arcsec"],
    )
    direct = comparison.method_data(direct_products, fov)
    pipeline = comparison.method_data(pipeline_products, fov)
    weblog = comparison.weblog_info(pipeline_dir)

    old_label = comparison.CUSTOM_METHOD_LABEL
    old_id = comparison.VISIBILITY_ID
    comparison.CUSTOM_METHOD_LABEL = DIRECT_METHOD_LABEL
    comparison.VISIBILITY_ID = VISIBILITY_ID
    try:
        png = output / f"{VISIBILITY_ID}_recreated_vs_vla_pipeline.png"
        scales = comparison.comparison_png(
            png,
            direct,
            pipeline,
            fov,
            clean_params,
            weblog["image_metrics"],
            figure_title=(
                f"{VISIBILITY_ID}: direct CASA recreation versus "
                "CASA VLA Imaging Pipeline"
            ),
            custom_stats_text=method_plot_text(
                DIRECT_METHOD_LABEL,
                direct,
                direct_details["clean_tclean_parameters"],
            ),
            pipeline_stats_text=method_plot_text(
                "CASA VLA Imaging Pipeline",
                pipeline,
                clean_params,
                qa=weblog["image_metrics"],
            ),
        )
    finally:
        comparison.CUSTOM_METHOD_LABEL = old_label
        comparison.VISIBILITY_ID = old_id

    direct_public = {
        key: direct[key]
        for key in ("products", "geometry", "metrics", "mask_metrics")
    }
    pipeline_public = {
        key: pipeline[key]
        for key in ("products", "geometry", "metrics", "mask_metrics")
    }
    json_report = output / f"{VISIBILITY_ID}_recreation_comparison_report.json"
    text_report = output / f"{VISIBILITY_ID}_recreation_comparison_report.txt"
    report = {
        "experiment": "recreating_vla_pipeline_test",
        "visibility_id": VISIBILITY_ID,
        "created_at": datetime.now().astimezone().isoformat(),
        "input": {
            **input_info,
            "selected_field": field,
            "direct_and_pipeline_use_same_split_ms": True,
            "direct_run_precedes_pipeline_processing": True,
        },
        "comparison": {
            "common_fov_arcsec": fov,
            "display_scale_percentile": comparison.DISPLAY_PERCENTILE,
            "share_color_scale_between_methods": comparison.SHARE_COLOR_SCALE_BETWEEN_METHODS,
            "display_scales": scales,
            "statistics_region": "central common FoV used by all six panels",
        },
        "direct_recreation": {**direct_public, **direct_details},
        "pipeline": {
            **pipeline_public,
            "pipeline_directory": str(pipeline_dir),
            "commands_log": str(log),
            "dirty_tclean_parameters": dirty_params,
            "final_tclean_parameters": clean_params,
            "weblog": weblog,
        },
        "outputs": {
            "output_directory": str(output),
            "comparison_png": str(png),
            "json_report": str(json_report),
            "text_report": str(text_report),
        },
    }

    text = "\n\n".join(
        [
            f"Direct CASA recreation versus VLA Imaging Pipeline: {VISIBILITY_ID}\n" + "=" * 72,
            f"Source MS: {SOURCE_MS}\nShared split input: {input_info['pipeline_input_ms']}\n"
            f"Comparison FoV: {fov:.6g} arcsec\nComparison PNG: {png}",
            comparison.stats_text(DIRECT_METHOD_LABEL, direct),
            comparison.stats_text(
                "CASA VLA Imaging Pipeline",
                pipeline,
                clean_params,
                weblog["image_metrics"],
            ),
            "Direct CASA controls\n" + "-" * 72 + "\n"
            + json.dumps(json_ready(direct_details), indent=2, sort_keys=True),
            "Image and mask products\n" + "-" * 72 + "\n"
            + json.dumps(
                json_ready({"direct_recreation": direct["products"], "pipeline": pipeline["products"]}),
                indent=2,
            ),
            "Exact final Pipeline tclean parameters\n" + "-" * 72 + "\n"
            + json.dumps(json_ready(clean_params), indent=2, sort_keys=True),
        ]
    ) + "\n"
    write_json(json_report, report)
    text_report.write_text(text, encoding="utf-8")
    print(text)
    print(f"[DONE] comparison PNG: {png}")
    print(f"[DONE] text report: {text_report}")
    print(f"[DONE] JSON report: {json_report}")
    return report


def main() -> dict:
    if not SOURCE_MS.exists():
        raise FileNotFoundError(SOURCE_MS)
    comparison.SOURCE_MS = SOURCE_MS
    tasks = comparison.pipeline_tasks()
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output = EXPERIMENTS / f"recreating_vla_pipeline_test_{timestamp}"
    work = output / "working"
    work.mkdir(parents=True)
    input_ms = work / f"{VISIBILITY_ID}_shared_input.ms"
    field, input_info = comparison.prepare_input(input_ms)
    print(f"[DIRECT + PIPELINE] shared input={input_ms}\n[field] {field}\n[work] {work}")

    direct_products, direct_details = run_direct_recreation(input_ms, work)
    pipeline_dir = comparison.run_pipeline(work, input_ms, field, tasks)
    return write_results(
        output,
        work,
        pipeline_dir,
        input_info,
        field,
        direct_products,
        direct_details,
    )


if __name__ in {"__main__", "<run_path>"}:
    main()
