from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.image as mpimg
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from casatasks import applycal, clearcal, gaincal, imstat, rmtables, tclean
from casatools import table

from scripts.img_utils import casa_image_to_png
from scripts.image_extracted import (
    CALIBRATOR_BANDS_CSV,
    FINAL_CLEAN_NITER,
    FIRSTPASS_NITER,
    PROJECT_LIST,
    TCLEAN_BASE,
    choose_band_and_frequency,
    clean_residual_qa_metrics,
    choose_final_imaging_setup,
    choose_firstpass_imaging_setup,
    compute_uvlimit_coverage_stats,
    flag_zero_visibilities,
    load_calibrator_uv_limits,
    load_projects,
    lookup_calibrator_uv_limits,
    make_uvrange_string,
    read_restoring_beam_arcsec,
    remove_casa_products,
    row_for_folder,
)
from scripts.io_utils import copy_ms


RUNS_DIR = Path("/Users/u1528314/repos/radioastro-ml/runs")
WORKSPACE_PREFIX = "uvlim_recal"

PHASE_SOLINT = "int"
AP_SOLINT = "inf"
PHASE_MINSNR = 3.0
AP_MINSNR = 3.0
COMPARISON_IMAGE_UVRANGE = ""


@dataclass
class VariantResult:
    key: str
    title: str
    start_from: str
    cal_sequence: str
    clean_image: Path
    residual_image: Path
    clean_png: Path
    peak_jy_per_beam: float
    rms_jy_per_beam: float
    dynrange: float
    qa_metrics: dict[str, float]


def _sanitize_name(text: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "._+-") else "_" for ch in text)


def _fmt_float(x, fmt: str, fallback: str = "?") -> str:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return fallback
    if not np.isfinite(value):
        return fallback
    return format(value, fmt)


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


def _print_variant_metrics(label: str, metrics: dict[str, float]) -> None:
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


def _print_metric_ratios(label: str, before: dict[str, float], after: dict[str, float]) -> None:
    print(f"[QA ratios] {label} / before")
    print(
        "sigma ratio = after_sigma / before_sigma = "
        f"{_format_metric_value(_ratio(after.get('residual_robust_sigma_jy_per_beam'), before.get('residual_robust_sigma_jy_per_beam')))}"
    )
    print(
        "max ratio = after_max / before_max = "
        f"{_format_metric_value(_ratio(after.get('residual_peak_to_sigma'), before.get('residual_peak_to_sigma')))}"
    )
    print(
        "p99 ratio = after_p99 / before_p99 = "
        f"{_format_metric_value(_ratio(after.get('residual_p99_abs_over_sigma'), before.get('residual_p99_abs_over_sigma')))}"
    )
    print(
        "p995 ratio = after_p995 / before_p995 = "
        f"{_format_metric_value(_ratio(after.get('residual_p995_abs_over_sigma'), before.get('residual_p995_abs_over_sigma')))}"
    )
    print(
        "DR ratio = after_DR / before_DR = "
        f"{_format_metric_value(_ratio(after.get('dynamic_range'), before.get('dynamic_range')))}"
    )


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        try:
            path.unlink()
        except OSError:
            pass


def _resolve_run_root(out_root: str | Path | None = None) -> Path:
    if out_root is not None:
        root = Path(out_root)
        root.mkdir(parents=True, exist_ok=True)
        return root

    cwd = Path.cwd()
    if (cwd / "casa-logs").exists() or cwd.parent == RUNS_DIR:
        return cwd

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    root = RUNS_DIR / datetime.now().strftime(f"{WORKSPACE_PREFIX}_%Y-%m-%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _make_workspace(ms_path: Path, out_root: str | Path | None = None) -> Path:
    run_root = _resolve_run_root(out_root)
    folder_name = _sanitize_name(ms_path.stem)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace = run_root / f"{WORKSPACE_PREFIX}_{folder_name}_{stamp}"
    workspace.mkdir(parents=True, exist_ok=False)
    return workspace


def _copy_ms(src_ms: Path, dst_ms: Path) -> Path:
    copy_ms(str(src_ms), str(dst_ms))
    return dst_ms


def _find_first_ms(root: Path) -> Path | None:
    if not root.exists():
        return None
    for hit in sorted(root.glob("*.ms")):
        if hit.is_dir():
            return hit
    for hit in sorted(root.rglob("*.ms")):
        if hit.is_dir():
            return hit
    return None


def _ms_has_corrected(ms_path: Path) -> bool:
    tb = table()
    tb.open(str(ms_path))
    try:
        return "CORRECTED_DATA" in tb.colnames()
    finally:
        tb.close()


def _best_datacolumn(ms_path: Path, *, prefer_corrected: bool = True) -> str:
    if prefer_corrected and _ms_has_corrected(ms_path):
        return "corrected"
    return "data"


def _resolve_datacolumn(ms_path: Path, datacolumn: str) -> str:
    choice = str(datacolumn).strip().lower()
    if choice in ("", "auto", "best"):
        return _best_datacolumn(ms_path, prefer_corrected=True)
    if choice in ("data", "corrected"):
        if choice == "corrected" and not _ms_has_corrected(ms_path):
            raise RuntimeError(f"{ms_path} does not have CORRECTED_DATA")
        return choice
    if choice == "corrected_data":
        if not _ms_has_corrected(ms_path):
            raise RuntimeError(f"{ms_path} does not have CORRECTED_DATA")
        return "corrected"
    raise ValueError(f"Unsupported datacolumn={datacolumn!r}; use 'auto', 'data', or 'corrected'")


def _put_corrected_into_data(ms_path: Path) -> None:
    tb = table()
    tb.open(str(ms_path), nomodify=False)
    try:
        if "CORRECTED_DATA" not in tb.colnames():
            raise RuntimeError(f"{ms_path} is missing CORRECTED_DATA")
        tb.putcol("DATA", tb.getcol("CORRECTED_DATA"))
    finally:
        tb.close()


def _load_image_stats(image_path: Path) -> tuple[float, float, float]:
    stats = imstat(imagename=str(image_path))
    peak = float(np.nanmax(stats["max"])) if "max" in stats else np.nan
    rms = float(np.nanmax(stats["rms"])) if "rms" in stats else np.nan
    dynrange = peak / rms if np.isfinite(peak) and np.isfinite(rms) and rms > 0 else np.nan
    return peak, rms, dynrange


def _run_tclean(
    ms_path: Path,
    imagename: str,
    *,
    cell_arcsec: float,
    imsize: int,
    niter: int,
    uvrange: str = "",
    datacolumn: str = "corrected",
    savemodel: str = "none",
) -> None:
    remove_casa_products(imagename)
    cfg = dict(TCLEAN_BASE)
    cfg.update(
        vis=str(ms_path),
        imagename=imagename,
        cell=f"{cell_arcsec:.6f}arcsec",
        imsize=int(imsize),
        niter=int(niter),
        uvrange=uvrange,
        datacolumn=datacolumn,
        savemodel=savemodel,
    )
    print(
        f"[TCLEAN] {Path(imagename).name} | "
        f"datacolumn={datacolumn} | "
        f"savemodel={savemodel} | "
        f"cell={cell_arcsec:.4f}\" | "
        f"imsize={imsize} | "
        f"niter={niter} | "
        f"uvrange={uvrange or 'all'}"
    )
    tclean(**cfg)


def _export_image_png(image_path: Path, png_path: Path, *, title: str) -> None:
    if png_path.exists():
        png_path.unlink()
    casa_image_to_png(
        str(image_path),
        str(png_path),
        title=title,
        draw_beam_ellipse=True,
        symmetric=True,
        cmap="inferno",
    )


def _infer_folder_name(ms_path: Path) -> str:
    name = ms_path.stem
    for suffix in ("_pipeline_input", "_source_copy"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _row_for_ms(df: pd.DataFrame, ms_path: Path) -> Optional[pd.Series]:
    folder_name = _infer_folder_name(ms_path)
    row = row_for_folder(df, folder_name)
    if row is not None:
        return row

    if "extracted_ms" in df.columns:
        extracted = df["extracted_ms"].astype("string").fillna("").str.strip()
        match = df.loc[extracted == str(ms_path)]
        if not match.empty:
            return match.iloc[0]

    if "name" in df.columns:
        names = df["name"].astype("string").fillna("").str.strip()
        match = df.loc[names == folder_name]
        if not match.empty:
            return match.iloc[0]

    return None


def _resolve_source_ms(
    input_ms: Path,
    row: Optional[pd.Series],
) -> Path:
    if not _ms_has_corrected(input_ms):
        print(
            f"[INFO] {input_ms} lacks CORRECTED_DATA; "
            f"running the data-backed subset of the workflow only"
        )
    return input_ms


def _prepare_starting_ms(src_ms: Path, out_ms: Path, *, start_from: str) -> Path:
    ms = _copy_ms(src_ms, out_ms)
    if start_from == "corrected":
        _put_corrected_into_data(ms)
    clearcal(vis=str(ms))
    flag_zero_visibilities(ms)
    return ms


def _promote_corrected_for_next_round(src_ms: Path, out_ms: Path) -> Path:
    ms = _copy_ms(src_ms, out_ms)
    _put_corrected_into_data(ms)
    clearcal(vis=str(ms))
    flag_zero_visibilities(ms)
    return ms


def _run_selfcal_round(
    ms_path: Path,
    caltable: Path,
    *,
    calmode: str,
    solint: str,
    minsnr: float,
    uvrange: str,
) -> None:
    _remove_path(caltable)
    rmtables(str(caltable))
    gaincal_kwargs = dict(
        vis=str(ms_path),
        caltable=str(caltable),
        gaintype="G",
        calmode=calmode,
        solint=solint,
        minsnr=minsnr,
        refant="",
        uvrange=uvrange,
    )
    if calmode == "ap":
        gaincal_kwargs["solnorm"] = False
    gaincal(**gaincal_kwargs)
    applycal(
        vis=str(ms_path),
        gaintable=[str(caltable)],
        interp=["linear"],
        calwt=False,
        applymode="calonly",
    )


def _make_clean_result(
    ms_path: Path,
    outdir: Path,
    *,
    key: str,
    title: str,
    cell_arcsec: float,
    imsize: int,
    uvrange: str = "",
    datacolumn: str | None = None,
    savemodel: str = "none",
) -> tuple[Path, Path, Path, float, float, float, dict[str, float]]:
    clean_base = outdir / key
    if datacolumn is None:
        datacolumn = _best_datacolumn(ms_path, prefer_corrected=True)
    _run_tclean(
        ms_path,
        str(clean_base),
        cell_arcsec=cell_arcsec,
        imsize=imsize,
        niter=FINAL_CLEAN_NITER,
        uvrange=uvrange,
        datacolumn=datacolumn,
        savemodel=savemodel,
    )
    clean_image = clean_base.with_suffix(".image")
    residual_image = clean_base.with_suffix(".residual")
    clean_png = outdir / f"{key}.png"
    _export_image_png(clean_image, clean_png, title=title)
    peak, rms, dynrange = _load_image_stats(clean_image)
    qa_metrics = clean_residual_qa_metrics(clean_image, residual_image)
    return clean_image, residual_image, clean_png, peak, rms, dynrange, qa_metrics


def _build_model_column(
    ms_path: Path,
    outdir: Path,
    *,
    key: str,
    cell_arcsec: float,
    imsize: int,
    uvrange: str,
    datacolumn: str = "data",
) -> Path:
    model_base = outdir / key
    _run_tclean(
        ms_path,
        str(model_base),
        cell_arcsec=cell_arcsec,
        imsize=imsize,
        niter=FINAL_CLEAN_NITER,
        uvrange=uvrange,
        datacolumn=datacolumn,
        savemodel="modelcolumn",
    )
    return model_base.with_suffix(".image")


def _make_phase_only_variant(
    seed_ms: Path,
    outdir: Path,
    *,
    key: str,
    title: str,
    start_from: str,
    cell_arcsec: float,
    imsize: int,
    uvrange: str,
    imaging_uvrange: str = "",
) -> tuple[VariantResult, Path]:
    phase_ms = _copy_ms(seed_ms, outdir / f"{key}.ms")
    _build_model_column(
        phase_ms,
        outdir,
        key=f"{key}_model",
        cell_arcsec=cell_arcsec,
        imsize=imsize,
        uvrange=uvrange,
        datacolumn="data",
    )
    _run_selfcal_round(
        phase_ms,
        outdir / f"{key}.g",
        calmode="p",
        solint=PHASE_SOLINT,
        minsnr=PHASE_MINSNR,
        uvrange=uvrange,
    )
    clean_image, residual_image, clean_png, peak, rms, dynrange, qa_metrics = _make_clean_result(
        phase_ms,
        outdir,
        key=f"{key}_after",
        title=title,
        cell_arcsec=cell_arcsec,
        imsize=imsize,
        uvrange=imaging_uvrange,
    )
    return (
        VariantResult(
            key=key,
            title=title,
            start_from=start_from,
            cal_sequence="p",
            clean_image=clean_image,
            residual_image=residual_image,
            clean_png=clean_png,
            peak_jy_per_beam=peak,
            rms_jy_per_beam=rms,
            dynrange=dynrange,
            qa_metrics=qa_metrics,
        ),
        phase_ms,
    )


def _make_phase_amp_variant(
    phase_ms: Path,
    outdir: Path,
    *,
    key: str,
    title: str,
    start_from: str,
    cell_arcsec: float,
    imsize: int,
    uvrange: str,
    imaging_uvrange: str = "",
) -> VariantResult:
    ap_ms = _promote_corrected_for_next_round(phase_ms, outdir / f"{key}.ms")
    _build_model_column(
        ap_ms,
        outdir,
        key=f"{key}_model",
        cell_arcsec=cell_arcsec,
        imsize=imsize,
        uvrange=uvrange,
        datacolumn="data",
    )
    _run_selfcal_round(
        ap_ms,
        outdir / f"{key}.g",
        calmode="ap",
        solint=AP_SOLINT,
        minsnr=AP_MINSNR,
        uvrange=uvrange,
    )
    clean_image, residual_image, clean_png, peak, rms, dynrange, qa_metrics = _make_clean_result(
        ap_ms,
        outdir,
        key=f"{key}_after",
        title=title,
        cell_arcsec=cell_arcsec,
        imsize=imsize,
        uvrange=imaging_uvrange,
    )
    return VariantResult(
        key=key,
        title=title,
        start_from=start_from,
        cal_sequence="p+ap",
        clean_image=clean_image,
        residual_image=residual_image,
        clean_png=clean_png,
        peak_jy_per_beam=peak,
        rms_jy_per_beam=rms,
        dynrange=dynrange,
        qa_metrics=qa_metrics,
    )


def _make_summary_png(
    out_png: Path,
    *,
    folder_name: str,
    summary_label: str = "uv-limit recalibration",
    config: str | None,
    band: str | None,
    uvrange: str,
    imaging_uvrange: str,
    uvmin_kl: float | None,
    uvmax_kl: float | None,
    uv_inside_frac: float | None,
    cell_arcsec: float,
    imsize: int,
    before_title: str,
    before_png: Path,
    results: list[VariantResult],
) -> None:
    panels = [(before_title, before_png, "before")] + [(res.title, res.clean_png, res.cal_sequence) for res in results]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5.8), squeeze=False)
    axes_row = axes[0]

    for ax in axes_row:
        ax.axis("off")

    for ax, (title, png_path, subtitle) in zip(axes_row, panels):
        ax.imshow(mpimg.imread(png_path))
        ax.axis("off")
        ax.set_title(f"{title}\n{subtitle}", fontsize=11)

    uv_inside_pct = 100.0 * float(uv_inside_frac) if uv_inside_frac is not None and np.isfinite(uv_inside_frac) else np.nan
    has_uv_limits = (
        (uvmin_kl is not None and np.isfinite(uvmin_kl))
        or (uvmax_kl is not None and np.isfinite(uvmax_kl))
    )
    uv_line = f"cal/model uv={uvrange or 'all'}; image uv={imaging_uvrange or 'all'}"
    if has_uv_limits or np.isfinite(uv_inside_pct):
        uv_line += (
            f"  in={_fmt_float(uv_inside_pct, '.1f', fallback='?')}%  "
            f"[{_fmt_float(uvmin_kl, '.1f', fallback='-')},{_fmt_float(uvmax_kl, '.1f', fallback='-')}] kl"
        )
    fig.suptitle(
        (
            f"{folder_name} {summary_label}\n"
            f"band={band or '?'}  config={config or '?'}  {uv_line}\n"
            f"cell={_fmt_float(cell_arcsec, '.4f')}\"  imsize={imsize}"
        ),
        fontsize=13,
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote summary PNG: {out_png}")


def run_recalibration(
    ms_path: str | Path,
    *,
    out_root: str | Path | None = None,
    project_csv: str | Path = PROJECT_LIST,
    calibrator_csv: str | Path = CALIBRATOR_BANDS_CSV,
    initial_plot_only: bool = False,
    initial_plot_datacolumn: str = "auto",
) -> dict:
    projects_df = load_projects(Path(project_csv))
    input_ms = Path(ms_path)
    if not input_ms.exists():
        raise FileNotFoundError(f"MS not found: {input_ms}")
    if not input_ms.is_dir() or input_ms.suffix != ".ms":
        raise RuntimeError(f"Expected a Measurement Set directory ending in .ms: {input_ms}")

    row = _row_for_ms(projects_df, input_ms)
    src_ms = _resolve_source_ms(input_ms, row)

    workspace = _make_workspace(src_ms, out_root=out_root)
    folder_name = _infer_folder_name(input_ms)

    probe_ms = _copy_ms(src_ms, workspace / f"{folder_name}_before_probe.ms")
    plot_datacolumn = "data"
    plot_datacolumn_label = "DATA"
    if initial_plot_only:
        requested_datacolumn = _resolve_datacolumn(probe_ms, initial_plot_datacolumn)
        if requested_datacolumn == "corrected":
            _put_corrected_into_data(probe_ms)
            plot_datacolumn_label = "CORRECTED_DATA -> DATA"
        else:
            plot_datacolumn_label = "DATA"
    flag_zero_visibilities(probe_ms)

    band_info = choose_band_and_frequency(probe_ms, row)
    gain_array_config = None if row is None else row.get("gain_array_config")
    uv_limit_info = None
    uvrange = ""
    uv_cov = {
        "uv_fraction_inside_limits": np.nan,
    }
    if not initial_plot_only:
        calib_df = load_calibrator_uv_limits(Path(calibrator_csv))
        uv_limit_info = lookup_calibrator_uv_limits(
            calib_df,
            calibrator_name=folder_name if row is None else row.get("name", folder_name),
            band=band_info["selected_band"],
        )
        if uv_limit_info is None:
            raise RuntimeError(
                f"No uv-limit row found for calibrator={folder_name if row is None else row.get('name', folder_name)} "
                f"band={band_info['selected_band']!r}"
            )

        uvrange = make_uvrange_string(uv_limit_info["uvmin_kl"], uv_limit_info["uvmax_kl"])
        uv_cov = compute_uvlimit_coverage_stats(
            probe_ms,
            band_info["ms_freq_ghz"],
            uv_limit_info["uvmin_kl"],
            uv_limit_info["uvmax_kl"],
        )
    comparison_uvrange = COMPARISON_IMAGE_UVRANGE
    firstpass_uvrange = comparison_uvrange

    firstpass_cell, firstpass_imsize, estimated_beam_arcsec = choose_firstpass_imaging_setup(
        gain_array_config,
        band_info["ms_freq_ghz"],
        band_info["selected_band"],
    )
    firstpass_base = workspace / "beam_firstpass_dirty"
    _run_tclean(
        probe_ms,
        str(firstpass_base),
        cell_arcsec=firstpass_cell,
        imsize=firstpass_imsize,
        niter=FIRSTPASS_NITER,
        uvrange=firstpass_uvrange,
        datacolumn=plot_datacolumn,
    )
    bmaj, bmin, bpa = read_restoring_beam_arcsec(firstpass_base.with_suffix(".image"))
    final_cell, final_imsize, final_fov_arcsec = choose_final_imaging_setup(bmaj, bmin)

    # Original corrected-column-only branches are intentionally omitted here.
    # Many extracted MS products in this repo only preserve DATA.
    before_title = f"before ({plot_datacolumn_label})"
    before_key = "before_corrected" if "CORRECTED" in plot_datacolumn_label else "before_data"
    before_image, before_residual, before_png, before_peak, before_rms, before_dynrange, before_qa_metrics = _make_clean_result(
        probe_ms,
        workspace,
        key=f"{before_key}_clean",
        title=before_title,
        cell_arcsec=final_cell,
        imsize=final_imsize,
        uvrange=comparison_uvrange,
        datacolumn=plot_datacolumn,
    )

    results = []
    if not initial_plot_only:
        data_seed = _prepare_starting_ms(src_ms, workspace / f"{folder_name}_from_data_seed.ms", start_from="data")
        data_phase, data_phase_ms = _make_phase_only_variant(
            data_seed,
            workspace,
            key="from_data_phase",
            title="after from DATA (phase)",
            start_from="data",
            cell_arcsec=final_cell,
            imsize=final_imsize,
            uvrange=uvrange,
            imaging_uvrange=comparison_uvrange,
        )
        data_ap = _make_phase_amp_variant(
            data_phase_ms,
            workspace,
            key="from_data_ap",
            title="after from DATA (phase+amp)",
            start_from="data",
            cell_arcsec=final_cell,
            imsize=final_imsize,
            uvrange=uvrange,
            imaging_uvrange=comparison_uvrange,
        )
        results = [data_phase, data_ap]

    output_stem = "initial_plot" if initial_plot_only else "uvlim_recal"
    summary_png = workspace / f"{folder_name}_{output_stem}_row.png"
    _make_summary_png(
        summary_png,
        folder_name=folder_name,
        summary_label="initial image" if initial_plot_only else "uv-limit recalibration",
        config=None if row is None else str(row.get("gain_array_config", "")).strip(),
        band=band_info["selected_band"],
        uvrange=uvrange,
        imaging_uvrange=comparison_uvrange,
        uvmin_kl=None if uv_limit_info is None else uv_limit_info["uvmin_kl"],
        uvmax_kl=None if uv_limit_info is None else uv_limit_info["uvmax_kl"],
        uv_inside_frac=uv_cov["uv_fraction_inside_limits"],
        cell_arcsec=final_cell,
        imsize=final_imsize,
        before_title=before_title,
        before_png=before_png,
        results=results,
    )

    summary_rows = [
        {
            "key": before_key,
            "title": before_title,
            "start_from": plot_datacolumn,
            "cal_sequence": "before",
            "clean_image": str(before_image),
            "residual_image": str(before_residual),
            "clean_png": str(before_png),
            "peak_jy_per_beam": before_peak,
            "rms_jy_per_beam": before_rms,
            "dynrange": before_dynrange,
            **before_qa_metrics,
        }
    ]
    summary_rows.extend(
        {
            "key": res.key,
            "title": res.title,
            "start_from": res.start_from,
            "cal_sequence": res.cal_sequence,
            "clean_image": str(res.clean_image),
            "residual_image": str(res.residual_image),
            "clean_png": str(res.clean_png),
            "peak_jy_per_beam": res.peak_jy_per_beam,
            "rms_jy_per_beam": res.rms_jy_per_beam,
            "dynrange": res.dynrange,
            **res.qa_metrics,
        }
        for res in results
    )
    summary_csv = workspace / f"{folder_name}_{output_stem}_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    meta = {
        "folder": folder_name,
        "ms": str(src_ms),
        "workspace": str(workspace),
        "summary_png": str(summary_png),
        "summary_csv": str(summary_csv),
        "selected_band": band_info["selected_band"],
        "detected_band": band_info["detected_band"],
        "spw_center_ghz_ms": band_info["ms_freq_ghz"],
        "spw_used_for_band_check": band_info["used_spw"],
        "gain_array_config": gain_array_config,
        "estimated_beam_arcsec": estimated_beam_arcsec,
        "firstpass_cell_arcsec": firstpass_cell,
        "firstpass_imsize": firstpass_imsize,
        "firstpass_uvrange": firstpass_uvrange,
        "beam_major_arcsec": bmaj,
        "beam_minor_arcsec": bmin,
        "beam_pa_deg": bpa,
        "cell_arcsec": final_cell,
        "imsize": final_imsize,
        "fov_arcsec": final_fov_arcsec,
        "catalog_uv_receiver": None if uv_limit_info is None else uv_limit_info["receiver"],
        "catalog_uvmin_kl": np.nan if uv_limit_info is None else uv_limit_info["uvmin_kl"],
        "catalog_uvmax_kl": np.nan if uv_limit_info is None else uv_limit_info["uvmax_kl"],
        "catalog_uvrange": uvrange,
        "comparison_image_uvrange": comparison_uvrange,
        "uv_fraction_inside_limits": uv_cov["uv_fraction_inside_limits"],
        "initial_plot_only": initial_plot_only,
        "initial_plot_datacolumn": plot_datacolumn_label,
        "variants": summary_rows,
    }

    _print_variant_metrics(before_title, before_qa_metrics)
    for res in results:
        _print_variant_metrics(res.title, res.qa_metrics)
        _print_metric_ratios(res.title, before_qa_metrics, res.qa_metrics)

    print(f"[DONE] workspace={workspace}")
    print(f"[DONE] summary PNG: {summary_png}")
    print(f"[DONE] summary CSV: {summary_csv}")
    return meta


def main(
    ms_path: str,
    *,
    out_root: str | Path | None = None,
    initial_plot_only: bool = False,
    initial_plot_datacolumn: str = "auto",
) -> dict:
    return run_recalibration(
        ms_path,
        out_root=out_root,
        initial_plot_only=initial_plot_only,
        initial_plot_datacolumn=initial_plot_datacolumn,
    )


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run uv-limit recalibration experiments in a run-local temp workspace.")
#     parser.add_argument("ms_path", help="Path to the source Measurement Set")
#     parser.add_argument(
#         "--out-root",
#         default=None,
#         help="Optional output root. Defaults to the current casa_run folder when available.",
#     )
#     parser.add_argument(
#         "--initial-plot-only",
#         action="store_true",
#         help="Only make the initial before-DATA image/summary; skip uv-limit lookup and selfcal variants.",
#     )
#     parser.add_argument(
#         "--initial-plot-datacolumn",
#         default="auto",
#         choices=("auto", "data", "corrected"),
#         help=(
#             "Datacolumn for --initial-plot-only. 'auto' copies CORRECTED_DATA "
#             "into DATA in the workspace copy when CORRECTED_DATA is present."
#         ),
#     )
#     args = parser.parse_args()
#     main(
#         args.ms_path,
#         out_root=args.out_root,
#         initial_plot_only=args.initial_plot_only,
#         initial_plot_datacolumn=args.initial_plot_datacolumn,
#     )
