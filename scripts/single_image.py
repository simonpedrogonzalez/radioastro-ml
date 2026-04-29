from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from casatasks import imstat, tclean
from casatools import table

from scripts.image_extracted import (
    FINAL_CLEAN_NITER,
    FIRSTPASS_NITER,
    PROJECT_LIST,
    TCLEAN_BASE,
    choose_band_and_frequency,
    choose_final_imaging_setup,
    choose_firstpass_imaging_setup,
    load_projects,
    read_restoring_beam_arcsec,
    remove_casa_products,
    row_for_folder,
)
from scripts.img_utils import casa_image_to_png


RUNS_DIR = Path("/Users/u1528314/repos/radioastro-ml/runs")
WORKSPACE_PREFIX = "single_image"


def _sanitize_name(text: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "._+-") else "_" for ch in text)


def _normalize_extracted_ms_path(ms_path: str | Path) -> Path:
    path = Path(str(ms_path).strip()).expanduser()
    parts = list(path.parts)
    for idx, part in enumerate(parts[:-1]):
        if part == "collect" and parts[idx + 1].startswith("extracted"):
            parts[idx + 1] = "extracted"
            return Path(*parts)
    return path


def _infer_folder_name(ms_path: Path) -> str:
    name = ms_path.stem
    for suffix in ("_pipeline_input", "_source_copy", "_selfcal"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _resolve_run_root(out_root: str | Path | None = None) -> Path:
    if out_root is not None:
        root = Path(out_root).expanduser()
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
    folder_name = _sanitize_name(_infer_folder_name(ms_path))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace = run_root / f"{WORKSPACE_PREFIX}_{folder_name}_{stamp}"
    workspace.mkdir(parents=True, exist_ok=False)
    return workspace


def _row_for_ms(ms_path: Path):
    projects_df = load_projects(PROJECT_LIST)
    folder_name = _infer_folder_name(ms_path)
    row = row_for_folder(projects_df, folder_name)
    if row is not None:
        return row

    if "extracted_ms" in projects_df.columns:
        normalized_ms = str(_normalize_extracted_ms_path(ms_path))
        extracted = projects_df["extracted_ms"].astype("string").fillna("").map(
            lambda value: str(_normalize_extracted_ms_path(value)) if str(value).strip() else ""
        )
        match = projects_df.loc[extracted == normalized_ms]
        if not match.empty:
            return match.iloc[0]

    if "name" in projects_df.columns:
        names = projects_df["name"].astype("string").fillna("").str.strip()
        match = projects_df.loc[names == folder_name]
        if not match.empty:
            return match.iloc[0]

    return None


def _ms_columns(ms_path: Path) -> list[str]:
    tb = table()
    tb.open(str(ms_path))
    try:
        return list(tb.colnames())
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
    imagename: Path,
    *,
    datacolumn: str,
    cell_arcsec: float,
    imsize: int,
    niter: int,
    uvrange: str = "",
) -> None:
    remove_casa_products(str(imagename))
    cfg = dict(TCLEAN_BASE)
    cfg.update(
        vis=str(ms_path),
        imagename=str(imagename),
        cell=f"{cell_arcsec:.6f}arcsec",
        imsize=int(imsize),
        niter=int(niter),
        datacolumn=datacolumn,
        uvrange=uvrange,
    )
    print(
        f"[TCLEAN] vis={ms_path} datacolumn={datacolumn} "
        f"imagename={imagename} cell={cell_arcsec:.6f}arcsec "
        f"imsize={imsize} niter={niter} uvrange={uvrange or 'all'}"
    )
    tclean(**cfg)


def _make_clean_for_column(
    ms_path: Path,
    workspace: Path,
    *,
    datacolumn: str,
    final_cell: float,
    final_imsize: int,
    uvrange: str,
) -> dict:
    label = "corrected" if datacolumn == "corrected" else "data"
    clean_base = workspace / f"direct_{label}_clean"
    _run_tclean(
        ms_path,
        clean_base,
        datacolumn=datacolumn,
        cell_arcsec=final_cell,
        imsize=final_imsize,
        niter=FINAL_CLEAN_NITER,
        uvrange=uvrange,
    )

    clean_image = clean_base.with_suffix(".image")
    clean_png = workspace / f"direct_{label}_clean.png"
    title = f"{ms_path.name} | direct original MS | {datacolumn.upper()}"
    if uvrange:
        title += f" | uv={uvrange}"
    casa_image_to_png(
        str(clean_image),
        str(clean_png),
        title=title,
        draw_beam_ellipse=True,
        symmetric=True,
        cmap="inferno",
    )
    peak, rms, dynrange = _load_image_stats(clean_image)
    return {
        "datacolumn": datacolumn,
        "clean_image": str(clean_image),
        "clean_png": str(clean_png),
        "peak_jy_per_beam": peak,
        "rms_jy_per_beam": rms,
        "dynrange": dynrange,
        "uvrange": uvrange,
    }


def _make_compare_png(rows: list[dict], out_png: Path) -> None:
    fig, axes = plt.subplots(1, len(rows), figsize=(5.4 * len(rows), 5.6), squeeze=False)
    axes_row = axes[0]

    for ax, row in zip(axes_row, rows):
        ax.imshow(mpimg.imread(row["clean_png"]))
        ax.axis("off")
        ax.set_title(
            (
                f"{row['datacolumn'].upper()}\n"
                f"peak={row['peak_jy_per_beam']:.4g}  "
                f"rms={row['rms_jy_per_beam']:.4g}  "
                f"DR={row['dynrange']:.2f}"
            ),
            fontsize=11,
        )

    uv_label = rows[0].get("uvrange") or "all"
    fig.suptitle(
        f"Direct imaging of original MS, no copy, no table edits | uv={uv_label}",
        fontsize=13,
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote comparison PNG: {out_png}")


def main(
    ms_path: str | Path,
    *,
    out_root: str | Path | None = None,
    uvrange: str = "",
) -> dict:
    ms_path = Path(ms_path).expanduser()
    if not ms_path.exists():
        raise FileNotFoundError(f"MS not found: {ms_path}")
    if not ms_path.is_dir() or ms_path.suffix != ".ms":
        raise RuntimeError(f"Expected a Measurement Set directory ending in .ms: {ms_path}")

    cols = _ms_columns(ms_path)
    print(f"[INFO] MS: {ms_path}")
    print(f"[INFO] Columns: {cols}")

    workspace = _make_workspace(ms_path, out_root=out_root)
    row = _row_for_ms(ms_path)
    band_info = choose_band_and_frequency(ms_path, row)
    gain_array_config = None if row is None else row.get("gain_array_config")

    firstpass_cell, firstpass_imsize, estimated_beam_arcsec = choose_firstpass_imaging_setup(
        gain_array_config,
        band_info["ms_freq_ghz"],
        band_info["selected_band"],
    )

    firstpass_base = workspace / "beam_firstpass_direct_data"
    _run_tclean(
        ms_path,
        firstpass_base,
        datacolumn="data",
        cell_arcsec=firstpass_cell,
        imsize=firstpass_imsize,
        niter=FIRSTPASS_NITER,
        uvrange=uvrange,
    )
    bmaj, bmin, bpa = read_restoring_beam_arcsec(firstpass_base.with_suffix(".image"))
    final_cell, final_imsize, final_fov_arcsec = choose_final_imaging_setup(bmaj, bmin)

    rows = [
        _make_clean_for_column(
            ms_path,
            workspace,
            datacolumn="data",
            final_cell=final_cell,
            final_imsize=final_imsize,
            uvrange=uvrange,
        )
    ]
    if "CORRECTED_DATA" in cols:
        rows.append(
            _make_clean_for_column(
                ms_path,
                workspace,
                datacolumn="corrected",
                final_cell=final_cell,
                final_imsize=final_imsize,
                uvrange=uvrange,
            )
        )
    else:
        print("[WARN] CORRECTED_DATA is not present; only DATA was imaged.")

    compare_png = workspace / f"{_infer_folder_name(ms_path)}_direct_columns.png"
    _make_compare_png(rows, compare_png)

    summary_csv = workspace / f"{_infer_folder_name(ms_path)}_direct_columns_summary.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with summary_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "ms": str(ms_path),
        "workspace": str(workspace),
        "compare_png": str(compare_png),
        "summary_csv": str(summary_csv),
        "columns": cols,
        "selected_band": band_info["selected_band"],
        "detected_band": band_info["detected_band"],
        "gain_array_config": gain_array_config,
        "estimated_beam_arcsec": estimated_beam_arcsec,
        "beam_major_arcsec": bmaj,
        "beam_minor_arcsec": bmin,
        "beam_pa_deg": bpa,
        "cell_arcsec": final_cell,
        "imsize": final_imsize,
        "fov_arcsec": final_fov_arcsec,
        "uvrange": uvrange,
        "results": rows,
    }
    print(f"[DONE] workspace: {workspace}")
    print(f"[DONE] compare PNG: {compare_png}")
    print(f"[DONE] summary CSV: {summary_csv}")
    return meta


# if __name__ == "__main__":
#     main("/path/to/input.ms")
