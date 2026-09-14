#!/usr/bin/env python3
"""Compare custom and VLA Pipeline imaging of one simulated MeasurementSet.

Run with CASA 6.6.6-18 plus VLA Pipeline 2025.1.0.36::

    casa --pipeline --nogui --nologger -c scripts/vla_imaging_pipeline_test.py

Change ``VISIBILITY_ID`` below to select a different simulated dataset. The
source MS and existing custom images are never modified.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from casatasks import imhead, split
from casatools import image, table


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_outputs import json_ready, write_json


VISIBILITY_ID = "0012-399"
SOURCE_MS = (
    ROOT
    / "collect"
    / "extracted"
    / VISIBILITY_ID
    / VISIBILITY_ID
    / "simulated_constant_0.7Jy_phasecenter.ms"
)
CUSTOM_PRODUCTS_DIR = SOURCE_MS.parent / f"{SOURCE_MS.stem}_comparison"
CUSTOM_PRODUCTS = {
    "dirty": CUSTOM_PRODUCTS_DIR / "simulation_dirty.image",
    "clean": CUSTOM_PRODUCTS_DIR / "simulation_clean.image",
    "residual": CUSTOM_PRODUCTS_DIR / "simulation_clean.residual",
    "mask": CUSTOM_PRODUCTS_DIR / "simulation_clean.mask",
}
CUSTOM_METHOD_LABEL = "Custom imaging pipeline"
EXPERIMENTS = ROOT / "collect/experiments"
DISPLAY_PERCENTILE = 99.5
SHARE_COLOR_SCALE_BETWEEN_METHODS = False
TASK_NAMES = (
    "h_init",
    "h_save",
    "hifv_importdata",
    "hifv_flagtargetsdata",
    "hif_checkproductsize",
    "hif_makeimlist",
    "hif_makeimages",
)


def pipeline_tasks() -> dict[str, Any]:
    try:
        import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "VLA Pipeline is unavailable. Use CASA 6.6.6-18 with Pipeline "
            "2025.1.0.36 and start CASA with --pipeline."
        ) from exc

    pipeline.initcli()
    main_scope = vars(sys.modules["__main__"])
    tasks = {
        name: globals().get(name) or main_scope.get(name)
        for name in TASK_NAMES
    }
    missing = [name for name, task in tasks.items() if task is None]
    if missing:
        raise RuntimeError(f"Missing Pipeline tasks: {', '.join(missing)}")
    return tasks


def prepare_input(output_ms: Path) -> tuple[str, dict[str, Any]]:
    tb = table()
    tb.open(str(SOURCE_MS))
    source_columns = [str(name) for name in tb.colnames()]
    tb.close()
    datacolumn = "corrected" if "CORRECTED_DATA" in source_columns else "data"
    split(vis=str(SOURCE_MS), outputvis=str(output_ms), datacolumn=datacolumn)

    tb.open(str(output_ms / "FIELD"))
    field = str(tb.getcol("NAME")[0])
    tb.close()

    tb.open(str(output_ms / "STATE"), nomodify=False)
    old_intents = sorted({str(value) for value in tb.getcol("OBS_MODE")})
    tb.putcol(
        "OBS_MODE",
        np.asarray(["OBSERVE_TARGET#ON_SOURCE"] * tb.nrows()),
    )
    tb.close()

    tb.open(str(output_ms))
    output_columns = [str(name) for name in tb.colnames()]
    tb.close()
    return field, {
        "source_ms": str(SOURCE_MS),
        "source_columns": source_columns,
        "source_datacolumn_copied": datacolumn,
        "pipeline_input_ms": str(output_ms),
        "pipeline_input_columns": output_columns,
        "original_intents_in_pipeline_copy": old_intents,
        "pipeline_copy_intent": "OBSERVE_TARGET#ON_SOURCE",
    }


def run_pipeline(work: Path, input_ms: Path, field: str, tasks: dict[str, Any]) -> Path:
    previous = Path.cwd()
    os.chdir(work)
    try:
        context = tasks["h_init"]()
        context.set_state("ProjectSummary", "observatory", "Karl G. Jansky Very Large Array")
        context.set_state("ProjectSummary", "telescope", "EVLA")
        try:
            tasks["hifv_importdata"](
                vis=[input_ms.name],
                datacolumns={"data": "regcal_contline_science"},
                specline_spws="none",
            )
            tasks["hifv_flagtargetsdata"]()
            tasks["hif_checkproductsize"](maximsize=16384)
            # tasks["hif_makeimlist"](specmode="cont", datatype="regcal", field=field)
            tasks["hif_makeimlist"](
                specmode="cont",
                datatype="regcal",
                field=field,
                hm_imsize=[256, 256],
            )
            tasks["hif_makeimages"](hm_cyclefactor=3.0)
        finally:
            tasks["h_save"]()
    finally:
        os.chdir(previous)

    directories = sorted(path for path in work.glob("pipeline-*") if path.is_dir())
    if not directories:
        raise RuntimeError(f"Pipeline did not create a weblog directory in {work}")
    return directories[-1]


def parse_tclean_calls(log: Path) -> list[dict[str, Any]]:
    """Recover the exact heuristic-generated tclean calls from the Pipeline log."""
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    calls: list[dict[str, Any]] = []
    index = 0
    while index < len(lines):
        if not lines[index].lstrip().startswith("tclean("):
            index += 1
            continue
        chunk = [lines[index].strip()]
        depth = lines[index].count("(") - lines[index].count(")")
        index += 1
        while depth > 0:
            chunk.append(lines[index].strip())
            depth += lines[index].count("(") - lines[index].count(")")
            index += 1

        source = " ".join(chunk)
        node = ast.parse(source).body[0].value
        parameters = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in node.keywords
            if keyword.arg
        }
        calls.append({"command": source, "parameters": parameters})
    if not calls:
        raise RuntimeError(f"No tclean calls found in {log}")
    return calls


def pipeline_products(work: Path, calls: list[dict[str, Any]]) -> tuple[dict[str, Path], dict, dict]:
    calls = [call for call in calls if "regcal" in call["parameters"]["imagename"]]
    dirty = next(call for call in calls if call["parameters"]["niter"] == 0)
    clean = max(
        (call for call in calls if call["parameters"]["niter"] > 0),
        key=lambda call: int(re.search(r"\.iter(\d+)$", call["parameters"]["imagename"])[1]),
    )
    dirty_base = work / dirty["parameters"]["imagename"]
    clean_base = work / clean["parameters"]["imagename"]
    products = {
        "dirty": Path(f"{dirty_base}.residual.tt0"),
        "clean": Path(f"{clean_base}.image.tt0"),
        "residual": Path(f"{clean_base}.residual.tt0"),
        "mask": Path(f"{clean_base}.mask"),
    }
    missing = [str(path) for path in products.values() if not path.exists()]
    if missing:
        raise RuntimeError("Missing Pipeline products:\n" + "\n".join(missing))
    return products, dirty["parameters"], clean["parameters"]


def weblog_info(pipeline_dir: Path) -> dict[str, Any]:
    index = (pipeline_dir / "html/index.html").read_text(errors="replace")
    stage = (pipeline_dir / "html/stage5/t2-4m_details.html").read_text(errors="replace")

    def match(text: str, pattern: str) -> str:
        found = re.search(pattern, text, flags=re.DOTALL)
        return found[1].strip() if found else "not found"

    metrics = {
        "QA Score": match(stage, r"QA Score:\s*&nbsp;\s*([^<&]+)"),
        "total number of major cycles done": match(
            stage, r"total number of major cycles done</th>\s*<td[^>]*>([^<]+)"
        ),
        "clean residual peak / scaled MAD": match(
            stage, r"clean residual peak / scaled MAD</th>\s*<td[^>]*>([^<]+)"
        ),
    }
    return {
        "pipeline_version": match(index, r"Pipeline Version</th>\s*<td>([^<\n]+)"),
        "casa_version": match(index, r"CASA Version</th>\s*<td>([^<(]+)"),
        "weblog_index": str(pipeline_dir / "html/index.html"),
        "image_metrics": metrics,
    }


def image_geometry(path: Path) -> dict[str, Any]:
    info = imhead(imagename=str(path), mode="summary")

    def arcsec(value: float, unit: str) -> float:
        unit = unit.lower()
        if unit.startswith("rad"):
            return abs(float(np.rad2deg(value) * 3600.0))
        if unit.startswith("deg"):
            return abs(float(value) * 3600.0)
        if unit.startswith("arcmin"):
            return abs(float(value) * 60.0)
        return abs(float(value))

    cell_x = arcsec(info["incr"][0], info["axisunits"][0])
    cell_y = arcsec(info["incr"][1], info["axisunits"][1])
    beam = info.get("restoringbeam")
    beam_info = {"major_arcsec": None, "minor_arcsec": None, "position_angle_deg": None}
    if beam:
        pa = beam["positionangle"]
        beam_info = {
            "major_arcsec": arcsec(beam["major"]["value"], beam["major"]["unit"]),
            "minor_arcsec": arcsec(beam["minor"]["value"], beam["minor"]["unit"]),
            "position_angle_deg": (
                float(np.rad2deg(pa["value"])) if pa["unit"].startswith("rad") else float(pa["value"])
            ),
        }
    return {
        "path": str(path),
        "imsize_x": int(info["shape"][0]),
        "imsize_y": int(info["shape"][1]),
        "cell_x_arcsec": cell_x,
        "cell_y_arcsec": cell_y,
        "fov_x_arcsec": int(info["shape"][0]) * cell_x,
        "fov_y_arcsec": int(info["shape"][1]) * cell_y,
        "brightness_unit": str(info.get("unit", "Jy/beam")),
        "beam": beam_info,
    }


def load_image(path: Path) -> np.ndarray:
    ia = image()
    ia.open(str(path))
    try:
        values = np.squeeze(np.asarray(ia.getchunk()))
    finally:
        ia.close()
    while values.ndim > 2:
        values = values[..., 0]
    return np.asarray(values, dtype=float).T


def crop(values: np.ndarray, geometry: dict, fov: float) -> tuple[np.ndarray, tuple[float, ...]]:
    ny, nx = values.shape
    width = min(nx, int(fov / geometry["cell_x_arcsec"])) // 2 * 2
    height = min(ny, int(fov / geometry["cell_y_arcsec"])) // 2 * 2
    x0, y0 = (nx - width) // 2, (ny - height) // 2
    values = values[y0 : y0 + height, x0 : x0 + width]
    half_x = width * geometry["cell_x_arcsec"] / 2
    half_y = height * geometry["cell_y_arcsec"] / 2
    return values, (-half_x, half_x, -half_y, half_y)


def method_data(products: dict[str, Path], fov: float) -> dict[str, Any]:
    geometry = {name: image_geometry(products[name]) for name in ("dirty", "clean", "residual")}
    images, extents = {}, {}
    for name in ("dirty", "clean", "residual"):
        images[name], extents[name] = crop(load_image(products[name]), geometry[name], fov)

    mask_full = load_image(products["mask"])
    mask, mask_extent = crop(mask_full, geometry["clean"], fov)
    active_full = np.isfinite(mask_full) & (mask_full > 0.5)
    active = np.isfinite(mask) & (mask > 0.5)
    finite_full, finite = np.isfinite(mask_full), np.isfinite(mask)
    positions = np.argwhere(active_full)
    bbox = None
    if positions.size:
        y0, x0 = positions.min(axis=0)
        y1, x1 = positions.max(axis=0)
        bbox = [int(x0), int(y0), int(x1), int(y1)]

    dirty = images["dirty"][np.isfinite(images["dirty"])]
    clean = images["clean"][np.isfinite(images["clean"])]
    residual = images["residual"][np.isfinite(images["residual"])]
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    sigma = 1.4826 * mad
    absolute = np.abs(residual)
    metrics = {
        "dirty_peak_jy_per_beam": float(np.max(np.abs(dirty))),
        "clean_peak_jy_per_beam": float(np.max(np.abs(clean))),
        "residual_median_jy_per_beam": median,
        "residual_mad_jy_per_beam": mad,
        "residual_robust_sigma_jy_per_beam": sigma,
        "residual_rms_jy_per_beam": float(np.sqrt(np.mean(residual**2))),
        "residual_max_abs_jy_per_beam": float(np.max(absolute)),
        "residual_peak_to_sigma": float(np.max(absolute) / sigma),
        "residual_p99_abs_over_sigma": float(np.percentile(absolute, 99) / sigma),
        "residual_p995_abs_over_sigma": float(np.percentile(absolute, 99.5) / sigma),
        "dynamic_range_clean_peak_over_sigma": float(np.max(np.abs(clean)) / sigma),
    }
    return {
        "products": {name: str(path) for name, path in products.items()},
        "geometry": geometry,
        "images": images,
        "extents": extents,
        "mask": mask,
        "mask_extent": mask_extent,
        "mask_metrics": {
            "active_fraction_full_image": float(active_full.sum() / finite_full.sum()),
            "active_fraction_comparison_fov": float(active.sum() / finite.sum()),
            "active_bbox_pixels_full_image": bbox,
        },
        "metrics": metrics,
    }


def stats_text(name: str, data: dict, clean_params: dict | None = None, qa: dict | None = None) -> str:
    geometry = data["geometry"]["clean"]
    beam = geometry["beam"]
    metrics = data["metrics"]
    lines = [
        name,
        f"native grid = {geometry['imsize_x']}x{geometry['imsize_y']} px",
        f"cell = {geometry['cell_x_arcsec']:.4g} x {geometry['cell_y_arcsec']:.4g} arcsec",
        f"native FoV = {geometry['fov_x_arcsec']:.4g} x {geometry['fov_y_arcsec']:.4g} arcsec",
        f"beam = {beam['major_arcsec']:.4g} x {beam['minor_arcsec']:.4g} arcsec, "
        f"PA={beam['position_angle_deg']:.4g} deg",
        f"mask active = {100 * data['mask_metrics']['active_fraction_comparison_fov']:.3g}% "
        "of comparison FoV",
        "",
        "Metrics on the shared FoV:",
        f"sigma = 1.4826*MAD(residual) = {metrics['residual_robust_sigma_jy_per_beam']:.4g} Jy/beam",
        f"max(|residual|)/sigma = {metrics['residual_peak_to_sigma']:.4g}",
        f"p99(|residual|)/sigma = {metrics['residual_p99_abs_over_sigma']:.4g}",
        f"p99.5(|residual|)/sigma = {metrics['residual_p995_abs_over_sigma']:.4g}",
        f"DR = clean peak/sigma = {metrics['dynamic_range_clean_peak_over_sigma']:.4g}",
    ]
    if clean_params:
        lines += [
            "",
            "Final Pipeline tclean:",
            f"deconvolver={clean_params['deconvolver']}  nterms={clean_params['nterms']}",
            f"weighting={clean_params['weighting']}  robust={clean_params['robust']}",
            f"niter={clean_params['niter']}  nmajor={clean_params['nmajor']}",
            f"threshold={clean_params['threshold']}  nsigma={clean_params['nsigma']}",
            f"usemask={clean_params['usemask']}",
        ]
    if qa:
        lines += [
            "",
            f"Pipeline QA score = {qa['QA Score']}",
            f"major cycles = {qa['total number of major cycles done']}",
            f"Pipeline residual peak/scaled MAD = {qa['clean residual peak / scaled MAD']}",
        ]
    return "\n".join(lines)


def comparison_png(
    path: Path,
    custom: dict,
    pipeline: dict,
    fov: float,
    params: dict,
    qa: dict,
    *,
    figure_title: str | None = None,
    custom_stats_text: str | None = None,
    pipeline_stats_text: str | None = None,
) -> dict:
    methods = ((CUSTOM_METHOD_LABEL, custom), ("CASA VLA Imaging Pipeline", pipeline))
    products = ("dirty", "clean", "residual")
    if SHARE_COLOR_SCALE_BETWEEN_METHODS:
        shared_scales = {
            name: float(
                np.percentile(
                    np.abs(
                        np.concatenate([
                            data["images"][name][np.isfinite(data["images"][name])]
                            for _, data in methods
                        ])
                    ),
                    DISPLAY_PERCENTILE,
                )
            )
            for name in products
        }
        scales = {label: dict(shared_scales) for label, _ in methods}
    else:
        scales = {
            label: {
                name: float(
                    np.percentile(
                        np.abs(data["images"][name][np.isfinite(data["images"][name])]),
                        DISPLAY_PERCENTILE,
                    )
                )
                for name in products
            }
            for label, data in methods
        }

    fig = plt.figure(figsize=(22, 11))
    grid = fig.add_gridspec(2, 4, width_ratios=(1, 1, 1, 1.12), wspace=0.28, hspace=0.24)
    for row, (label, data) in enumerate(methods):
        for column, name in enumerate(products):
            axis = fig.add_subplot(grid[row, column])
            limit = scales[label][name] * 1e3
            plotted = axis.imshow(
                data["images"][name] * 1e3,
                origin="lower",
                extent=data["extents"][name],
                cmap="inferno",
                vmin=-limit,
                vmax=limit,
                interpolation="nearest",
            )
            axis.set(xlim=(fov / 2, -fov / 2), ylim=(-fov / 2, fov / 2), xlabel="RA offset (arcsec)")
            if column == 0:
                axis.set_ylabel("Dec offset (arcsec)")
            scale_label = "shared" if SHARE_COLOR_SCALE_BETWEEN_METHODS else "own"
            axis.set_title(
                f"{label}\n{name.title()} | {scale_label} ±{limit:.3g} mJy/beam",
                fontsize=10,
            )
            fig.colorbar(plotted, ax=axis, fraction=0.046, pad=0.03).set_label("mJy/beam")

            if name != "dirty" and np.any(data["mask"] > 0.5) and np.any(data["mask"] <= 0.5):
                axis.contour(
                    data["mask"],
                    levels=[0.5],
                    colors="cyan",
                    linewidths=0.8,
                    origin="lower",
                    extent=data["mask_extent"],
                )
            beam = data["geometry"][name]["beam"]
            if beam["major_arcsec"]:
                axis.add_patch(
                    Ellipse(
                        (-0.36 * fov, -0.36 * fov),
                        beam["minor_arcsec"],
                        beam["major_arcsec"],
                        angle=beam["position_angle_deg"],
                        fill=False,
                        edgecolor="lime",
                        linewidth=1.3,
                    )
                )

        text_axis = fig.add_subplot(grid[row, 3])
        text_axis.axis("off")
        override_text = custom_stats_text if row == 0 else pipeline_stats_text
        panel_text = override_text or stats_text(
            label,
            data,
            params if row else None,
            qa if row else None,
        )
        text_axis.text(
            0,
            1,
            panel_text,
            ha="left",
            va="top",
            family="monospace",
            fontsize=6.4 if override_text is not None else 9.2,
        )

    title = figure_title or f"{VISIBILITY_ID}: custom imaging versus CASA VLA Imaging Pipeline"
    fig.suptitle(
        f"{title}\n"
        f"All panels and statistics use the central {fov:.2f} arcsec; cyan contours show CLEAN masks",
        fontsize=15,
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        label: {
            f"{name}_abs_limit_jy_per_beam": value
            for name, value in method_scales.items()
        }
        for label, method_scales in scales.items()
    }


def write_results(output: Path, work: Path, pipeline_dir: Path, input_info: dict, field: str) -> dict:
    log = pipeline_dir / "html/casa_commands.log"
    calls = parse_tclean_calls(log)
    products, dirty_params, clean_params = pipeline_products(work, calls)
    custom_geometry = image_geometry(CUSTOM_PRODUCTS["clean"])
    pipeline_geometry = image_geometry(products["clean"])
    fov = min(
        custom_geometry["fov_x_arcsec"],
        custom_geometry["fov_y_arcsec"],
        pipeline_geometry["fov_x_arcsec"],
        pipeline_geometry["fov_y_arcsec"],
    )
    custom = method_data(CUSTOM_PRODUCTS, fov)
    pipeline = method_data(products, fov)
    weblog = weblog_info(pipeline_dir)

    png = output / f"{VISIBILITY_ID}_custom_vs_vla_pipeline.png"
    scales = comparison_png(png, custom, pipeline, fov, clean_params, weblog["image_metrics"])
    public_custom = {key: custom[key] for key in ("products", "geometry", "metrics", "mask_metrics")}
    public_pipeline = {key: pipeline[key] for key in ("products", "geometry", "metrics", "mask_metrics")}
    report = {
        "experiment": "vla_imaging_pipeline_test",
        "visibility_id": VISIBILITY_ID,
        "created_at": datetime.now().astimezone().isoformat(),
        "input": {
            **input_info,
            "selected_field": field,
            "pipeline_datatype": "regcal_contline_science",
            "pipeline_specline_spws": "none",
            "selfcal_run": False,
        },
        "comparison": {
            "common_fov_arcsec": fov,
            "display_scale_percentile": DISPLAY_PERCENTILE,
            "share_color_scale_between_methods": SHARE_COLOR_SCALE_BETWEEN_METHODS,
            "display_scales": scales,
            "statistics_region": "central common FoV used by all six panels",
        },
        "custom": public_custom,
        "pipeline": {
            **public_pipeline,
            "pipeline_directory": str(pipeline_dir),
            "commands_log": str(log),
            "dirty_tclean_parameters": dirty_params,
            "final_tclean_parameters": clean_params,
            "weblog": weblog,
        },
        "outputs": {
            "output_directory": str(output),
            "comparison_png": str(png),
            "json_report": str(output / f"{VISIBILITY_ID}_imaging_comparison_report.json"),
            "text_report": str(output / f"{VISIBILITY_ID}_imaging_comparison_report.txt"),
        },
    }

    text = "\n\n".join(
        [
            f"VLA Imaging Pipeline comparison: {VISIBILITY_ID}\n" + "=" * 72,
            f"Source MS: {SOURCE_MS}\nPipeline input: {input_info['pipeline_input_ms']}\n"
            f"Comparison FoV: {fov:.6g} arcsec\nComparison PNG: {png}",
            stats_text(CUSTOM_METHOD_LABEL, custom),
            stats_text("CASA VLA Imaging Pipeline", pipeline, clean_params, weblog["image_metrics"]),
            "Image and mask products\n" + "-" * 72 + "\n"
            + json.dumps(json_ready({"custom": custom["products"], "pipeline": pipeline["products"]}), indent=2),
            "Exact initial-dirty Pipeline tclean parameters\n" + "-" * 72 + "\n"
            + json.dumps(json_ready(dirty_params), indent=2, sort_keys=True),
            "Exact final Pipeline tclean parameters\n" + "-" * 72 + "\n"
            + json.dumps(json_ready(clean_params), indent=2, sort_keys=True),
            "Pipeline weblog image metrics\n" + "-" * 72 + "\n"
            + json.dumps(json_ready(weblog["image_metrics"]), indent=2, sort_keys=True),
        ]
    ) + "\n"
    write_json(Path(report["outputs"]["json_report"]), report)
    Path(report["outputs"]["text_report"]).write_text(text, encoding="utf-8")
    print(text)
    print(f"[DONE] comparison PNG: {png}")
    print(f"[DONE] text report: {report['outputs']['text_report']}")
    print(f"[DONE] JSON report: {report['outputs']['json_report']}")
    return report


def main() -> dict:
    for path in (SOURCE_MS, *CUSTOM_PRODUCTS.values()):
        if not path.exists():
            raise FileNotFoundError(path)
    tasks = pipeline_tasks()  # Fail before creating output if Pipeline is absent.
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output = EXPERIMENTS / f"vla_imaging_pipeline_test_{timestamp}"
    work = output / "working"
    work.mkdir(parents=True)
    input_ms = work / f"{VISIBILITY_ID}_pipeline_input.ms"
    field, input_info = prepare_input(input_ms)
    print(f"[PIPELINE] input={input_ms}\n[PIPELINE] field={field}\n[PIPELINE] working directory={work}")
    pipeline_dir = run_pipeline(work, input_ms, field, tasks)
    return write_results(output, work, pipeline_dir, input_info, field)


if __name__ == "__main__":
    main()
