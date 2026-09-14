from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from casatasks import tclean
from casatools import table

from scripts.image_extracted import (
    FINAL_CLEAN_NITER,
    TCLEAN_BASE,
    clean_residual_qa_metrics,
    read_restoring_beam_arcsec,
    remove_casa_products,
)
from scripts.img_utils import casa_image_to_png


DEFAULT_VIS = Path(
    "/Users/u1528314/repos/radioastro-ml/collect/extracted/0205+322/selfcal/0205+322_selfcal.ms"
)
DEFAULT_OUTDIR = Path(
    "/Users/u1528314/repos/radioastro-ml/collect/extracted/0205+322/selfcal/reproduction_compare"
)
DEFAULT_OUTPUT_STEM = "0205+322_selfcal_regular_vs_reproduction"

REGULAR_CELL_ARCSEC = 2.793
REGULAR_FOV_ARCSEC = 714.925
REGULAR_BEAM_MAJOR_ARCSEC = 26.567
REGULAR_BEAM_MINOR_ARCSEC = 11.270

REPRODUCTION_CELL_ARCSEC = 1.855
REPRODUCTION_FOV_ARCSEC = 474.788
REPRODUCTION_BEAM_MAJOR_ARCSEC = 26.318
REPRODUCTION_BEAM_MINOR_ARCSEC = 11.240


@dataclass
class VariantResult:
    key: str
    title: str
    image_path: Path
    residual_path: Path
    png_path: Path
    datacolumn: str
    cell_arcsec: float
    fov_arcsec: float
    imsize: int
    beam_major_arcsec: float
    beam_minor_arcsec: float
    beam_pa_deg: float
    qa_metrics: dict[str, float]


def _ms_has_corrected(ms_path: Path) -> bool:
    tb = table()
    tb.open(str(ms_path))
    try:
        return "CORRECTED_DATA" in tb.colnames()
    finally:
        tb.close()


def _best_datacolumn(ms_path: Path) -> str:
    return "corrected" if _ms_has_corrected(ms_path) else "data"


def _imsize_from_cell_and_fov(cell_arcsec: float, fov_arcsec: float) -> int:
    imsize = int(round(fov_arcsec / cell_arcsec))
    if imsize % 2 == 1:
        imsize += 1
    return imsize


def _format_metric_value(x) -> str:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(value):
        return "nan"
    if abs(value) >= 1e-4:
        return f"{value:.4f}"
    return f"{value:.6f}"


def _ratio(after, before) -> float:
    try:
        after = float(after)
        before = float(before)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(after) or not np.isfinite(before):
        return float("nan")
    if before == 0:
        return float("inf") if after > 0 else 1.0
    return float(after / before)


def _print_metrics(label: str, metrics: dict[str, float]) -> None:
    print(f"[QA] {label}")
    print(
        "sigma = 1.4826*MAD(residual) [Jy/bm] = "
        f"{_format_metric_value(metrics.get('residual_robust_sigma_jy_per_beam'))}"
    )
    print(
        "max = max(|residual|)/sigma = "
        f"{_format_metric_value(metrics.get('residual_peak_to_sigma'))}"
    )
    print(
        "p99 = P99(|residual|)/sigma = "
        f"{_format_metric_value(metrics.get('residual_p99_abs_over_sigma'))}"
    )
    print(
        "p995 = P99.5(|residual|)/sigma = "
        f"{_format_metric_value(metrics.get('residual_p995_abs_over_sigma'))}"
    )
    print(
        "DR = max(|clean|)/sigma = "
        f"{_format_metric_value(metrics.get('dynamic_range'))}"
    )


def _print_ratios(reference_label: str, reference: dict[str, float], other_label: str, other: dict[str, float]) -> None:
    print(f"[QA ratios] {other_label} / {reference_label}")
    print(
        "sigma ratio = other_sigma / reference_sigma = "
        f"{_format_metric_value(_ratio(other.get('residual_robust_sigma_jy_per_beam'), reference.get('residual_robust_sigma_jy_per_beam')))}"
    )
    print(
        "max ratio = other_max / reference_max = "
        f"{_format_metric_value(_ratio(other.get('residual_peak_to_sigma'), reference.get('residual_peak_to_sigma')))}"
    )
    print(
        "p99 ratio = other_p99 / reference_p99 = "
        f"{_format_metric_value(_ratio(other.get('residual_p99_abs_over_sigma'), reference.get('residual_p99_abs_over_sigma')))}"
    )
    print(
        "p995 ratio = other_p995 / reference_p995 = "
        f"{_format_metric_value(_ratio(other.get('residual_p995_abs_over_sigma'), reference.get('residual_p995_abs_over_sigma')))}"
    )
    print(
        "DR ratio = other_DR / reference_DR = "
        f"{_format_metric_value(_ratio(other.get('dynamic_range'), reference.get('dynamic_range')))}"
    )


def _run_variant(
    vis: Path,
    outdir: Path,
    *,
    key: str,
    title: str,
    cell_arcsec: float,
    fov_arcsec: float,
    target_beam_major_arcsec: float,
    target_beam_minor_arcsec: float,
) -> VariantResult:
    datacolumn = _best_datacolumn(vis)
    imsize = _imsize_from_cell_and_fov(cell_arcsec, fov_arcsec)
    image_base = outdir / key

    remove_casa_products(str(image_base))

    cfg = dict(TCLEAN_BASE)
    cfg.update(
        vis=str(vis),
        imagename=str(image_base),
        datacolumn=datacolumn,
        imsize=imsize,
        cell=f"{cell_arcsec:.6f}arcsec",
        niter=int(FINAL_CLEAN_NITER),
    )

    print(
        f"[TCLEAN] {key} | datacolumn={datacolumn} | cell={cell_arcsec:.6f}arcsec | "
        f"FoV={fov_arcsec:.3f}\" | imsize={imsize} | niter={FINAL_CLEAN_NITER}"
    )
    tclean(**cfg)

    image_path = image_base.with_suffix(".image")
    residual_path = image_base.with_suffix(".residual")
    png_path = outdir / f"{key}.png"

    bmaj, bmin, bpa = read_restoring_beam_arcsec(image_path)
    plot_title = (
        f"{title} | datacolumn={datacolumn} | "
        f"cell={cell_arcsec:.3f}\" | FoV={fov_arcsec:.3f}\" | "
        f"target beam={target_beam_major_arcsec:.3f}\"x{target_beam_minor_arcsec:.3f}\" | "
        f"actual beam={bmaj:.3f}\"x{bmin:.3f}\""
    )
    casa_image_to_png(
        str(image_path),
        str(png_path),
        title=plot_title,
        draw_beam_ellipse=True,
        symmetric=True,
        cmap="inferno",
    )

    qa_metrics = clean_residual_qa_metrics(image_path, residual_path)
    return VariantResult(
        key=key,
        title=title,
        image_path=image_path,
        residual_path=residual_path,
        png_path=png_path,
        datacolumn=datacolumn,
        cell_arcsec=cell_arcsec,
        fov_arcsec=fov_arcsec,
        imsize=imsize,
        beam_major_arcsec=bmaj,
        beam_minor_arcsec=bmin,
        beam_pa_deg=bpa,
        qa_metrics=qa_metrics,
    )


def _write_comparison_png(
    results: list[VariantResult],
    out_png: Path,
    *,
    vis: Path,
) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(6.0 * len(results), 6.2), squeeze=False)
    axes_row = axes[0]

    for ax, result in zip(axes_row, results):
        ax.imshow(mpimg.imread(result.png_path))
        ax.axis("off")
        ax.set_title(
            (
                f"{result.title}\n"
                f"cell={result.cell_arcsec:.3f}\"  FoV={result.fov_arcsec:.3f}\"\n"
                f"beam={result.beam_major_arcsec:.3f}\"x{result.beam_minor_arcsec:.3f}\""
            ),
            fontsize=10,
        )

    fig.suptitle(
        f"0205+322 selfcal MS | regular vs reproduction\nvis={vis.name}",
        fontsize=13,
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote comparison PNG: {out_png}")


def main(
    vis: str | Path = DEFAULT_VIS,
    *,
    outdir: str | Path = DEFAULT_OUTDIR,
    output_stem: str = DEFAULT_OUTPUT_STEM,
) -> dict:
    vis = Path(vis).expanduser()
    outdir = Path(outdir).expanduser()
    if not vis.exists():
        raise FileNotFoundError(f"MS not found: {vis}")
    outdir.mkdir(parents=True, exist_ok=True)

    regular = _run_variant(
        vis,
        outdir,
        key="regular",
        title="regular",
        cell_arcsec=REGULAR_CELL_ARCSEC,
        fov_arcsec=REGULAR_FOV_ARCSEC,
        target_beam_major_arcsec=REGULAR_BEAM_MAJOR_ARCSEC,
        target_beam_minor_arcsec=REGULAR_BEAM_MINOR_ARCSEC,
    )
    reproduction = _run_variant(
        vis,
        outdir,
        key="reproduction",
        title="reproduction",
        cell_arcsec=REPRODUCTION_CELL_ARCSEC,
        fov_arcsec=REPRODUCTION_FOV_ARCSEC,
        target_beam_major_arcsec=REPRODUCTION_BEAM_MAJOR_ARCSEC,
        target_beam_minor_arcsec=REPRODUCTION_BEAM_MINOR_ARCSEC,
    )

    _print_metrics("regular", regular.qa_metrics)
    _print_metrics("reproduction", reproduction.qa_metrics)
    _print_ratios("regular", regular.qa_metrics, "reproduction", reproduction.qa_metrics)

    comparison_png = outdir / f"{output_stem}.png"
    _write_comparison_png([regular, reproduction], comparison_png, vis=vis)

    print(f"[DONE] regular image      : {regular.image_path}")
    print(f"[DONE] reproduction image : {reproduction.image_path}")
    print(f"[DONE] comparison png     : {comparison_png}")

    return {
        "vis": str(vis),
        "outdir": str(outdir),
        "comparison_png": str(comparison_png),
        "regular": {
            "image": str(regular.image_path),
            "residual": str(regular.residual_path),
            "png": str(regular.png_path),
            "beam_major_arcsec": regular.beam_major_arcsec,
            "beam_minor_arcsec": regular.beam_minor_arcsec,
            "beam_pa_deg": regular.beam_pa_deg,
            **regular.qa_metrics,
        },
        "reproduction": {
            "image": str(reproduction.image_path),
            "residual": str(reproduction.residual_path),
            "png": str(reproduction.png_path),
            "beam_major_arcsec": reproduction.beam_major_arcsec,
            "beam_minor_arcsec": reproduction.beam_minor_arcsec,
            "beam_pa_deg": reproduction.beam_pa_deg,
            **reproduction.qa_metrics,
        },
    }

