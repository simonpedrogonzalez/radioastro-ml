#!/usr/bin/env python3
"""Compare direct and VLA Pipeline imaging of the 0012-399 0.7 Jy simulation.

Run from the repository root with the pipeline-enabled CASA installation::

    '/Users/u1528314/Applications/CASA 2.app/Contents/MacOS/casa' \
        --pipeline --nogui --nologger -c scripts/image_0012_399.py

Both imaging engines use a 256x256 grid. The VLA Pipeline run can take several
minutes.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _reload_imaging_package_if_loaded() -> None:
    """Refresh package code when CASA execfile reuses its Python process."""
    module_names = (
        "scripts.imaging.models",
        "scripts.imaging.metadata",
        "scripts.imaging.config",
        "scripts.imaging.metrics",
        "scripts.imaging.qa",
        "scripts.imaging.plot_utils",
        "scripts.imaging.imaging",
        "scripts.imaging.vla_pipeline",
        "scripts.imaging",
    )
    if "scripts.imaging" not in sys.modules:
        return
    for module_name in module_names:
        module = sys.modules.get(module_name)
        if module is not None:
            importlib.reload(module)
    print("Reloaded scripts.imaging package for this CASA session")


_reload_imaging_package_if_loaded()

from scripts.imaging import DefaultImagingConfig, ImagingResult, image_ms, image_ms_VLA_pipe


VISIBILITY_ID = "0012-399"
SIMULATION_MS = (
    ROOT
    / "collect"
    / "extracted"
    / VISIBILITY_ID
    / VISIBILITY_ID
    / "simulated_constant_0.7Jy_phasecenter.ms"
)


def _number(value, fallback="none") -> str:
    if value is None:
        return fallback
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


def _parameter_title(result: ImagingResult) -> str:
    parameters = result.effective_imaging_parameters
    geometry = parameters.get("measured_geometry", {}) or {}
    if result.resolved_config is not None:
        grid = result.resolved_config.grid.image_grid
        imsize = f"{grid.imsize[0]}x{grid.imsize[1]}"
        cell = f"{grid.cell_arcsec[0]:.4g}\"/px"
    else:
        size = geometry.get("imsize", parameters.get("imsize", ["?", "?"]))
        cell_values = geometry.get("cell_arcsec", parameters.get("cell", ["?"]))
        imsize = "x".join(str(item) for item in size)
        cell = f"{_number(cell_values[0])}\"/px"
    deconvolution = (
        f"{parameters.get('deconvolver', '?')} (nterms={parameters.get('nterms', 1)})"
    )
    mask = parameters.get("usemask")
    if result.resolved_config is not None:
        mask = (
            f"central circle ({_number(parameters.get('mask_nbeams'))}-beam diameter)"
            if parameters.get("mask_nbeams") is not None
            else "none"
        )
    return (
        f"{deconvolution} | {parameters.get('weighting', '?')}, "
        f"robust={_number(parameters.get('robust'))}\n"
        f"{imsize} | {cell} | mask={mask or 'none'} | "
        f"niter={parameters.get('niter', '?')} | nsigma={_number(parameters.get('nsigma'))}"
    )


def _metrics_title(result: ImagingResult) -> str:
    metrics = result.qa.metrics
    residual = metrics.residual
    return (
        f"sigma={residual.scaled_mad_jy_per_beam:.4g} Jy/beam | "
        f"max/sigma={residual.peak_over_scaled_mad:.3g} | "
        f"p99/sigma={residual.p99_over_scaled_mad:.3g}\n"
        f"p99.5/sigma={residual.p99_5_over_scaled_mad:.3g} | "
        f"DR={metrics.dynamic_range_scaled_mad:.3g}"
    )


def make_comparison_png(
    direct: ImagingResult,
    vla: ImagingResult,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rows = (("Direct package", direct), ("VLA Pipeline", vla))
    figure, axes = plt.subplots(2, 3, figsize=(18, 11))
    for row_index, (label, result) in enumerate(rows):
        panels = (
            ("Dirty", result.dirty_png, ""),
            ("Clean", result.clean_png, _parameter_title(result)),
            ("Residual", result.residual_png, _metrics_title(result)),
        )
        for column_index, (kind, path, details) in enumerate(panels):
            axis = axes[row_index, column_index]
            axis.imshow(plt.imread(path))
            axis.set_axis_off()
            heading = f"{label} — {kind}"
            axis.set_title(
                heading if not details else f"{heading}\n{details}",
                fontsize=10,
                pad=10,
            )

    figure.suptitle(
        "0012-399 simulated 0.7 Jy source: direct imaging vs VLA Imaging Pipeline",
        fontsize=16,
        y=0.985,
    )
    figure.text(
        0.5,
        0.953,
        "Columns show dirty, clean, and residual products; annotations summarize clean settings and residual QA.",
        ha="center",
        va="top",
        fontsize=10,
    )
    figure.subplots_adjust(
        left=0.015,
        right=0.985,
        bottom=0.025,
        top=0.88,
        wspace=0.03,
        hspace=0.24,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> tuple[ImagingResult, ImagingResult, Path]:
    if not SIMULATION_MS.exists():
        raise FileNotFoundError(f"Simulation Measurement Set not found: {SIMULATION_MS}")

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    experiment_dir = (
        ROOT / "collect" / "experiments" / f"image_{VISIBILITY_ID}_simulation_{timestamp}"
    )

    # Keep every package default except the Taylor-term count. The simulation
    # has CORRECTED_DATA, so the default corrected datacolumn remains in use.
    direct_config = replace(DefaultImagingConfig, nterms=1)
    direct_result = image_ms(
        SIMULATION_MS,
        direct_config,
        experiment_dir / "direct",
        imsize=(256, 256),
        keep_intermediate_products=True,
    )
    vla_result = image_ms_VLA_pipe(
        SIMULATION_MS,
        experiment_dir / "vla_pipeline",
        imsize=(256, 256),
        keep_intermediate_products=True,
    )

    comparison_png = experiment_dir / "0012-399_direct_vs_vla_pipeline.png"
    make_comparison_png(direct_result, vla_result, comparison_png)

    print(f"Experiment complete: {experiment_dir}")
    print(f"Comparison image: {comparison_png}")
    print(f"Direct QA: {direct_result.qa_text}")
    print(f"VLA Pipeline QA: {vla_result.qa_text}")
    return direct_result, vla_result, comparison_png


if __name__ in {"__main__", "<run_path>"}:
    main()
