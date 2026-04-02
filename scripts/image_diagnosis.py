from __future__ import annotations

import shutil
from math import ceil
from pathlib import Path
from typing import Optional

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from casatasks import flagdata, imhead, imstat, tclean
from casatools import table
from casaplotms import plotms

from scripts.img_utils import casa_image_to_png
from scripts.image_extracted import (
    PROJECT_LIST,
    choose_band_and_frequency,
    choose_firstpass_imaging_setup,
    row_for_folder,
)


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
EXTRACTED_DIR = Path("/Users/u1528314/repos/radioastro-ml/collect/extracted")
CALIBRATOR_BANDS_CSV = Path("/Users/u1528314/repos/radioastro-ml/collect/vla_calibrators_bands_v2.csv")

FOLDER_NAME = "0739+016"

OUTDIR_NAME = "diagnostics"

MODIFY_MS_IN_PLACE = False

# beam estimate bootstrap
FIRSTPASS_CELL_ARCSEC = 0.5
FIRSTPASS_IMSIZE = 256

# final imaging grid policy
PIXELS_PER_BEAM = 4.0
FOV_IN_BEAMS = 64.0

MIN_CELL_ARCSEC = 0.02
MAX_CELL_ARCSEC = 20.0
MIN_IMSIZE = 128
MAX_IMSIZE = 1024

# imaging defaults
DEFAULT_NITER_CLEAN = 1000
DEFAULT_NITER_DIRTY = 0

SPW_TRY_LIMIT = 3  # top N SPWs by bandwidth / unflagged fraction

PLOT_COLS = 2

TCLEAN_BASE = dict(
    specmode="mfs",
    stokes="I",
    gridder="standard",
    interactive=False,
)


# -------------------------------------------------------------------
# IO / PATHS
# -------------------------------------------------------------------
def find_sample_ms(folder_name: str) -> tuple[Path, Path]:
    sample_top = EXTRACTED_DIR / folder_name
    if not sample_top.exists():
        raise FileNotFoundError(f"Folder not found: {sample_top}")

    expected_ms = sample_top / folder_name / f"{folder_name}.ms"
    if expected_ms.exists():
        print(f"[INFO] using expected MS: {expected_ms}")
        return sample_top, expected_ms

    hits = sorted(sample_top.rglob("*.ms"))
    if not hits:
        raise FileNotFoundError(f"No .ms found under: {sample_top}")

    print("[WARN] expected MS path not found, falling back to first .ms found")
    for p in hits:
        print(f"    candidate: {p}")
    return sample_top, hits[0]


def load_projects(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def normalize_calibrator_name(name: str | None) -> str:
    if name is None:
        return ""
    return str(name).strip().upper()


def band_to_receiver_code(band: str | None) -> str | None:
    mapping = {
        "P": "P",
        "L": "L",
        "S": "S",
        "C": "C",
        "X": "X",
        "KU": "U",
        "K": "K",
        "KA": "A",
        "Q": "Q",
    }
    if band is None:
        return None
    return mapping.get(str(band).strip().upper())


def load_calibrator_uv_limits(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def lookup_calibrator_uv_limits(
    calib_df: pd.DataFrame,
    *,
    calibrator_name: str | None,
    band: str | None,
) -> dict | None:
    name = normalize_calibrator_name(calibrator_name)
    receiver = band_to_receiver_code(band)
    if not name or receiver is None:
        return None

    name_col = calib_df["name"].astype("string").fillna("").str.upper().str.strip()
    recv_col = calib_df["receiver"].astype("string").fillna("").str.upper().str.strip()
    match = calib_df.loc[(name_col == name) & (recv_col == receiver)]
    if match.empty:
        return None

    row = match.iloc[0]
    uvmin = row.get("uvmin_kl")
    uvmax = row.get("uvmax_kl")

    try:
        uvmin = float(uvmin) if pd.notna(uvmin) else np.nan
    except (TypeError, ValueError):
        uvmin = np.nan
    try:
        uvmax = float(uvmax) if pd.notna(uvmax) else np.nan
    except (TypeError, ValueError):
        uvmax = np.nan

    if np.isfinite(uvmin) and np.isfinite(uvmax) and uvmin > uvmax:
        uvmin, uvmax = uvmax, uvmin

    return {
        "calibrator_name": str(row.get("name", calibrator_name)),
        "receiver": str(row.get("receiver", receiver)),
        "uvmin_kl": uvmin,
        "uvmax_kl": uvmax,
    }


def make_uvrange_string(uvmin_kl: float | None, uvmax_kl: float | None) -> str:
    lo = ""
    hi = ""
    if uvmin_kl is not None and np.isfinite(uvmin_kl) and uvmin_kl > 0:
        lo = f"{uvmin_kl:.3f}klambda"
    if uvmax_kl is not None and np.isfinite(uvmax_kl) and uvmax_kl > 0:
        hi = f"{uvmax_kl:.3f}klambda"

    if lo and hi:
        return f"{lo}~{hi}"
    if lo:
        return f">{lo}"
    if hi:
        return f"<{hi}"
    return ""


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        try:
            path.unlink()
        except OSError:
            pass


def remove_casa_products(imagename: str) -> None:
    stem = Path(imagename)
    parent = stem.parent
    prefix = stem.name
    if not parent.exists():
        return
    for p in parent.glob(prefix + ".*"):
        remove_path(p)


def copy_ms(src_ms: Path, out_ms: Path) -> Path:
    if out_ms.exists():
        shutil.rmtree(out_ms)
    print(f"[COPY] {src_ms} -> {out_ms}")
    shutil.copytree(src_ms, out_ms)
    return out_ms


# -------------------------------------------------------------------
# BASIC CASA HELPERS
# -------------------------------------------------------------------
def flag_zero_visibilities(ms_path: Path) -> None:
    print(f"[FLAG] flagging exact zeros in {ms_path}")
    flagdata(
        vis=str(ms_path),
        mode="clip",
        clipzeros=True,
        flagbackup=False,
    )


def _beam_value_to_arcsec(x) -> float:
    if isinstance(x, dict):
        value = float(x["value"])
        unit = str(x.get("unit", "")).strip().lower()
        if unit in ("arcsec", "arcseconds", "asec"):
            return value
        if unit in ("arcmin", "arcminutes"):
            return value * 60.0
        if unit in ("deg", "degree", "degrees"):
            return value * 3600.0
        raise ValueError(f"Unsupported beam unit: {unit!r}")
    return float(x)


def read_restoring_beam_arcsec(image_path: str | Path) -> tuple[float, float, float]:
    info = imhead(imagename=str(image_path), mode="summary")
    if not isinstance(info, dict):
        raise RuntimeError(f"imhead summary failed for {image_path}")

    rb = info.get("restoringbeam")
    if rb is None:
        raise RuntimeError(f"No restoringbeam found in {image_path}")

    bmaj = _beam_value_to_arcsec(rb["major"])
    bmin = _beam_value_to_arcsec(rb["minor"])

    pa = rb.get("positionangle", {})
    if isinstance(pa, dict):
        pa_deg = float(pa.get("value", np.nan))
        pa_unit = str(pa.get("unit", "deg")).strip().lower()
        if pa_unit.startswith("rad"):
            pa_deg = np.rad2deg(pa_deg)
    else:
        pa_deg = float(pa)

    return bmaj, bmin, pa_deg


def choose_beam_based_cell_arcsec(
    bmaj_arcsec: float,
    bmin_arcsec: float,
    pixels_per_beam: float = PIXELS_PER_BEAM,
) -> float:
    beam_ref = min(bmaj_arcsec, bmin_arcsec)
    cell = beam_ref / pixels_per_beam
    cell = max(MIN_CELL_ARCSEC, min(MAX_CELL_ARCSEC, cell))
    return cell


def choose_imsize_for_beam_normalized_fov(
    bmin_arcsec: float,
    cell_arcsec: float,
    fov_in_beams: float = FOV_IN_BEAMS,
) -> tuple[int, float]:
    target_fov_arcsec = fov_in_beams * bmin_arcsec
    imsize = int(np.ceil(target_fov_arcsec / cell_arcsec))
    if imsize % 2 == 1:
        imsize += 1
    imsize = max(MIN_IMSIZE, min(MAX_IMSIZE, imsize))
    actual_fov_arcsec = imsize * cell_arcsec
    return imsize, actual_fov_arcsec


def choose_multiscale_scales(cell_arcsec: float, bmaj_arcsec: float, bmin_arcsec: float) -> list[int]:
    beam_pix = max(1, int(round(min(bmaj_arcsec, bmin_arcsec) / cell_arcsec)))
    return [0, max(1, beam_pix), max(2, 3 * beam_pix)]


def image_stats(image_path: Path) -> dict:
    s = imstat(imagename=str(image_path))
    peak = float(np.nanmax(s["max"])) if "max" in s else np.nan
    rms = float(np.nanmax(s["rms"])) if "rms" in s else np.nan
    dr = peak / rms if np.isfinite(peak) and np.isfinite(rms) and rms > 0 else np.nan
    return {
        "peak_jy_per_beam": peak,
        "rms_jy_per_beam": rms,
        "dynamic_range": dr,
    }


def export_png(casa_image: Path, out_png: Path, *, symmetric: bool = False) -> None:
    if not casa_image.exists():
        raise FileNotFoundError(f"CASA image missing: {casa_image}")
    if out_png.exists():
        out_png.unlink()
    casa_image_to_png(
        str(casa_image),
        str(out_png),
        symmetric=symmetric,
        cmap="inferno",
    )


def export_uv_png(ms_path: Path, out_png: Path, *, spw: str = "", uvrange: str = "") -> None:
    if out_png.exists():
        out_png.unlink()

    print(f"[PLOTMS] uv coverage -> {out_png} | spw={spw or 'all'} | uvrange={uvrange or 'all'}")
    plotms(
        vis=str(ms_path),
        xaxis="u",
        yaxis="v",
        spw=spw,
        uvrange=uvrange,
        avgchannel="1e8",
        avgtime="1e8",
        coloraxis="spw",
        plotfile=str(out_png),
        expformat="png",
        highres=True,
        overwrite=True,
        showgui=False,
    )


# -------------------------------------------------------------------
# SPW HELPERS
# -------------------------------------------------------------------
def get_spw_candidates(ms_path: Path, top_n: int = SPW_TRY_LIMIT) -> list[dict]:
    tb = table()

    tb.open(str(ms_path / "SPECTRAL_WINDOW"))
    nspw = tb.nrows()
    spw_info = []
    for spw in range(nspw):
        freqs = np.array(tb.getcell("CHAN_FREQ", spw), dtype=float)
        if freqs.size == 0:
            continue
        f_min = float(np.min(freqs))
        f_max = float(np.max(freqs))
        f_c = float(np.median(freqs))
        bw = float(f_max - f_min)
        spw_info.append(
            {
                "spw": int(spw),
                "f_center_ghz": f_c / 1e9,
                "bandwidth_mhz": bw / 1e6,
            }
        )
    tb.close()

    tb.open(str(ms_path))
    data_desc_ids = np.array(tb.getcol("DATA_DESC_ID"))
    flags = np.array(tb.getcol("FLAG"))
    tb.close()

    tb.open(str(ms_path / "DATA_DESCRIPTION"))
    dd_to_spw = np.array(tb.getcol("SPECTRAL_WINDOW_ID"), dtype=int)
    tb.close()

    for info in spw_info:
        spw = info["spw"]
        ddids = np.where(dd_to_spw == spw)[0]
        row_mask = np.isin(data_desc_ids, ddids)
        n_rows = int(np.sum(row_mask))
        info["n_rows"] = n_rows
        if n_rows == 0:
            info["unflagged_frac"] = np.nan
            continue
        spw_flags = flags[:, :, row_mask]
        total = spw_flags.size
        flagged = int(np.count_nonzero(spw_flags))
        info["unflagged_frac"] = 1.0 - (flagged / total if total > 0 else np.nan)

    def rank_key(x: dict):
        uf = x["unflagged_frac"]
        uf = -1.0 if not np.isfinite(uf) else uf
        return (x["bandwidth_mhz"], uf)

    ranked = sorted(spw_info, key=rank_key, reverse=True)
    out = ranked[:top_n]

    print("[SPW] trying candidates:")
    for x in out:
        uf = x["unflagged_frac"]
        uf_s = "nan" if not np.isfinite(uf) else f"{uf:.3f}"
        print(
            f"    spw={x['spw']} center={x['f_center_ghz']:.3f} GHz "
            f"bw={x['bandwidth_mhz']:.2f} MHz unflagged={uf_s}"
        )
    return out


# -------------------------------------------------------------------
# TCLEAN WRAPPER
# -------------------------------------------------------------------
def run_tclean(
    ms_path: Path,
    imagename: str,
    *,
    cell_arcsec: float,
    imsize: int,
    niter: int,
    spw: str = "",
    datacolumn: str = "corrected",
    weighting: str = "briggs",
    robust: float = 0.5,
    deconvolver: str = "hogbom",
    scales: Optional[list[int]] = None,
    usemask: str = "",
    mask: str = "",
    uvrange: str = "",
) -> None:
    remove_casa_products(imagename)

    cfg = dict(TCLEAN_BASE)
    cfg.update(
        vis=str(ms_path),
        imagename=imagename,
        cell=f"{cell_arcsec:.6f}arcsec",
        imsize=int(imsize),
        niter=int(niter),
        spw=spw,
        datacolumn=datacolumn,
        weighting=weighting,
        deconvolver=deconvolver,
        uvrange=uvrange,
    )

    if weighting == "briggs":
        cfg["robust"] = robust

    if deconvolver == "multiscale":
        cfg["scales"] = [0, 5, 15] if scales is None else scales

    if usemask:
        cfg["usemask"] = usemask
    if mask:
        cfg["mask"] = mask

    print(
        f"[TCLEAN] {Path(imagename).name} | spw={spw or 'all'} | "
        f"cell={cell_arcsec:.4f}\" | imsize={imsize} | niter={niter} | "
        f"weighting={weighting} | deconvolver={deconvolver} | "
        f"uvrange={uvrange or 'all'}"
    )
    tclean(**cfg)


# -------------------------------------------------------------------
# DIAGNOSTIC RUNNERS
# -------------------------------------------------------------------
def estimate_grid(ms_path: Path, outdir: Path) -> tuple[float, int, float, float, float]:
    raise RuntimeError("estimate_grid now requires row metadata; use estimate_grid_from_metadata")


def estimate_grid_from_metadata(
    ms_path: Path,
    outdir: Path,
    row: Optional[pd.Series],
) -> tuple[float, int, float, float, float, dict]:
    gain_array_config = None if row is None else row.get("gain_array_config")
    band_info = choose_band_and_frequency(ms_path, row)
    firstpass_cell, firstpass_imsize, beam_est = choose_firstpass_imaging_setup(
        gain_array_config,
        band_info["ms_freq_ghz"],
        band_info["selected_band"],
    )

    base = outdir / "grid_firstpass"
    run_tclean(
        ms_path,
        str(base),
        cell_arcsec=firstpass_cell,
        imsize=firstpass_imsize,
        niter=0,
        datacolumn="corrected",
        usemask="",
    )
    firstpass_image = base.with_suffix(".image")
    bmaj, bmin, bpa = read_restoring_beam_arcsec(firstpass_image)
    cell = choose_beam_based_cell_arcsec(bmaj, bmin)
    imsize, fov_arcsec = choose_imsize_for_beam_normalized_fov(bmin, cell)

    print(
        f"[GRID] beam=({bmaj:.3f}\", {bmin:.3f}\", pa={bpa:.1f} deg) | "
        f"cell={cell:.4f}\" | imsize={imsize} | FoV={fov_arcsec:.2f}\" | "
        f"config={gain_array_config} | band={band_info['selected_band']} | "
        f"beam_est={beam_est if beam_est is not None else float('nan'):.3f}\""
    )
    return cell, imsize, bmaj, bmin, bpa, {
        "gain_array_config": gain_array_config,
        "band_used_for_firstpass": band_info["selected_band"],
        "band_detected_from_freq": band_info["detected_band"],
        "band_matches_frequency": band_info["band_match"],
        "spw_used_for_band_check": band_info["used_spw"],
        "spw_center_ghz_ms": band_info["ms_freq_ghz"],
        "spw_center_ghz_csv": band_info.get("csv_freq_ghz"),
        "firstpass_beam_estimate_arcsec": beam_est,
        "firstpass_cell_arcsec": firstpass_cell,
        "firstpass_imsize": firstpass_imsize,
    }


def central_mask_string(imsize: int, radius_frac: float = 0.12) -> str:
    cx = imsize // 2
    cy = imsize // 2
    r = max(6, int(round(imsize * radius_frac)))
    return f"circle[[{cx}pix,{cy}pix],{r}pix]"


def run_product_set(
    ms_path: Path,
    outdir: Path,
    key: str,
    title: str,
    *,
    cell: float,
    imsize: int,
    spw: str = "",
    weighting: str = "briggs",
    robust: float = 0.5,
    deconvolver: str = "hogbom",
    scales: Optional[list[int]] = None,
    usemask: str = "",
    mask: str = "",
    uvrange: str = "",
) -> dict:
    dirty_base = outdir / f"{key}_dirty"
    clean_base = outdir / f"{key}_clean"

    run_tclean(
        ms_path,
        str(dirty_base),
        cell_arcsec=cell,
        imsize=imsize,
        niter=DEFAULT_NITER_DIRTY,
        spw=spw,
        datacolumn="corrected",
        weighting=weighting,
        robust=robust,
        deconvolver=deconvolver,
        scales=scales,
        usemask="",
        uvrange=uvrange,
    )

    run_tclean(
        ms_path,
        str(clean_base),
        cell_arcsec=cell,
        imsize=imsize,
        niter=DEFAULT_NITER_CLEAN,
        spw=spw,
        datacolumn="corrected",
        weighting=weighting,
        robust=robust,
        deconvolver=deconvolver,
        scales=scales,
        usemask=usemask,
        mask=mask,
        uvrange=uvrange,
    )

    psf_img = dirty_base.with_suffix(".psf")
    resid_img = clean_base.with_suffix(".residual")
    dirty_img = dirty_base.with_suffix(".image")
    clean_img = clean_base.with_suffix(".image")

    dirty_png = outdir / f"{key}_dirty.png"
    clean_png = outdir / f"{key}_clean.png"
    psf_png = outdir / f"{key}_psf.png"
    resid_png = outdir / f"{key}_residual.png"
    uv_png = outdir / f"{key}_uv.png"

    export_png(dirty_img, dirty_png, symmetric=True)
    export_png(clean_img, clean_png, symmetric=False)
    export_png(psf_img, psf_png, symmetric=True)
    export_png(resid_img, resid_png, symmetric=True)
    export_uv_png(ms_path, uv_png, spw=spw, uvrange=uvrange)

    stats = image_stats(clean_img)

    return {
        "key": key,
        "title": title,
        "spw": spw,
        "weighting": weighting,
        "robust": robust,
        "deconvolver": deconvolver,
        "usemask": usemask,
        "mask": mask,
        "uvrange": uvrange,
        "dirty_png": str(dirty_png),
        "clean_png": str(clean_png),
        "psf_png": str(psf_png),
        "residual_png": str(resid_png),
        "uv_png": str(uv_png),
        **stats,
    }


def run_flagged_variant(
    src_ms: Path,
    outdir: Path,
    folder_name: str,
    key: str,
    title: str,
    *,
    cell: float,
    imsize: int,
    antenna_to_flag: Optional[str] = None,
    spw: str = "",
) -> dict:
    work_ms = outdir / f"{folder_name}_{key}.ms"
    ms = copy_ms(src_ms, work_ms)
    flag_zero_visibilities(ms)

    if antenna_to_flag:
        print(f"[FLAG TEST] flagging antenna {antenna_to_flag}")
        flagdata(vis=str(ms), mode="manual", antenna=antenna_to_flag, flagbackup=False)

    return run_product_set(
        ms,
        outdir,
        key,
        title,
        cell=cell,
        imsize=imsize,
        spw=spw,
    )


# -------------------------------------------------------------------
# PLOTTING
# -------------------------------------------------------------------
def make_summary_plot(
    *,
    folder_name: str,
    out_png: Path,
    rows: list[dict],
    panel_kind: str,
) -> None:
    if not rows:
        raise RuntimeError("No rows to plot")

    n = len(rows)
    ncols = PLOT_COLS
    nrows = ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6.7 * ncols, 5.8 * nrows),
        squeeze=False,
    )

    for axrow in axes:
        for ax in axrow:
            ax.axis("off")

    for i, row in enumerate(rows):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]

        img = mpimg.imread(row[panel_kind])
        ax.imshow(img)
        ax.axis("off")

        peak = row.get("peak_jy_per_beam", np.nan)
        rms = row.get("rms_jy_per_beam", np.nan)
        dr = row.get("dynamic_range", np.nan)

        if panel_kind == "uv_png":
            subtitle = f"spw={row.get('spw', '') or 'all'}"
        else:
            subtitle = (
                f"spw={row.get('spw', '') or 'all'} | {row.get('weighting', '')} | "
                f"{row.get('deconvolver', '')} | uv={row.get('uvrange', '') or 'all'}\n"
                f"peak={peak:.3e} Jy/bm | rms={rms:.3e} | DR={dr:.1f}"
                if np.isfinite(peak) and np.isfinite(rms) else
                f"spw={row.get('spw', '') or 'all'}"
            )

        ax.set_title(f"{row['title']}\n{subtitle}", fontsize=10)

    fig.suptitle(
        f"{folder_name} diagnostic comparison: {panel_kind.replace('_png','')}",
        fontsize=14,
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] wrote plot: {out_png}")


# -------------------------------------------------------------------
# MAIN DIAGNOSTIC DRIVER
# -------------------------------------------------------------------
def run_one(folder_name: str) -> tuple[Path, list[Path], Path]:
    sample_top, src_ms = find_sample_ms(folder_name)
    projects_df = load_projects(PROJECT_LIST)
    calib_df = load_calibrator_uv_limits(CALIBRATOR_BANDS_CSV)
    row = row_for_folder(projects_df, folder_name)
    outdir = sample_top / OUTDIR_NAME
    outdir.mkdir(parents=True, exist_ok=True)

    # probe copy for consistent beam/grid estimate
    probe_ms = outdir / f"{folder_name}_probe.ms"
    probe = copy_ms(src_ms, probe_ms)
    flag_zero_visibilities(probe)

    cell, imsize, bmaj, bmin, bpa, grid_meta = estimate_grid_from_metadata(probe, outdir, row)
    scales = choose_multiscale_scales(cell, bmaj, bmin)
    mask_str = central_mask_string(imsize)
    uv_limit_info = lookup_calibrator_uv_limits(
        calib_df,
        calibrator_name=folder_name if row is None else row.get("name", folder_name),
        band=grid_meta["band_used_for_firstpass"],
    )
    uv_limit_range = ""
    if uv_limit_info is not None:
        uv_limit_range = make_uvrange_string(uv_limit_info["uvmin_kl"], uv_limit_info["uvmax_kl"])
        print(
            f"[UVLIMIT] {folder_name} | receiver={uv_limit_info['receiver']} | "
            f"uvmin={uv_limit_info['uvmin_kl']} klambda | "
            f"uvmax={uv_limit_info['uvmax_kl']} klambda | "
            f"uvrange={uv_limit_range or 'none'}"
        )
    else:
        print(f"[UVLIMIT] {folder_name} | no catalog uv limits found")

    rows: list[dict] = []

    # 1) Baseline
    rows.append(
        run_product_set(
            probe,
            outdir,
            key="baseline",
            title="baseline",
            cell=cell,
            imsize=imsize,
        )
    )

    if uv_limit_range:
        rows.append(
            run_product_set(
                probe,
                outdir,
                key="catalog_uvsafe",
                title="catalog uv limits",
                cell=cell,
                imsize=imsize,
                uvrange=uv_limit_range,
            )
        )

    # 2) Weighting tests
    rows.append(
        run_product_set(
            probe,
            outdir,
            key="natural",
            title="natural weighting",
            cell=cell,
            imsize=imsize,
            weighting="natural",
        )
    )
    rows.append(
        run_product_set(
            probe,
            outdir,
            key="uniform",
            title="uniform weighting",
            cell=cell,
            imsize=imsize,
            weighting="uniform",
        )
    )

    # 3) Deconvolver tests
    rows.append(
        run_product_set(
            probe,
            outdir,
            key="clark",
            title="clark",
            cell=cell,
            imsize=imsize,
            deconvolver="clark",
        )
    )
    rows.append(
        run_product_set(
            probe,
            outdir,
            key="multiscale",
            title="multiscale",
            cell=cell,
            imsize=imsize,
            deconvolver="multiscale",
            scales=scales,
        )
    )
    rows.append(
        run_product_set(
            probe,
            outdir,
            key="masked",
            title="central mask",
            cell=cell,
            imsize=imsize,
            usemask="user",
            mask=mask_str,
        )
    )

    # 4) SPW tests
    for spw_info in get_spw_candidates(probe, top_n=SPW_TRY_LIMIT):
        spw = str(spw_info["spw"])
        rows.append(
            run_product_set(
                probe,
                outdir,
                key=f"spw_{spw}",
                title=f"SPW {spw}",
                cell=cell,
                imsize=imsize,
                spw=spw,
            )
        )

    # Optional targeted antenna test: fill manually if you suspect one
    SUSPECT_ANTENNA = None  # e.g. "ea12"
    if SUSPECT_ANTENNA:
        rows.append(
            run_flagged_variant(
                src_ms,
                outdir,
                folder_name,
                key="flag_suspect_ant",
                title=f"flag antenna {SUSPECT_ANTENNA}",
                cell=cell,
                imsize=imsize,
                antenna_to_flag=SUSPECT_ANTENNA,
            )
        )

    # save CSV
    summary_csv = outdir / f"{folder_name}_diagnostics.csv"
    df = pd.DataFrame(rows)
    df.insert(0, "beam_major_arcsec", bmaj)
    df.insert(1, "beam_minor_arcsec", bmin)
    df.insert(2, "beam_pa_deg", bpa)
    df.insert(3, "cell_arcsec", cell)
    df.insert(4, "imsize", imsize)
    for key, value in reversed(list(grid_meta.items())):
        df.insert(5, key, value)
    if uv_limit_info is not None:
        df.insert(5, "catalog_uvmax_kl", uv_limit_info["uvmax_kl"])
        df.insert(5, "catalog_uvmin_kl", uv_limit_info["uvmin_kl"])
        df.insert(5, "catalog_uv_receiver", uv_limit_info["receiver"])
        df.insert(5, "catalog_uvrange", uv_limit_range)
    df.to_csv(summary_csv, index=False)
    print(f"[OK] wrote CSV: {summary_csv}")

    # plots
    plot_paths = []
    for panel_kind in ["clean_png", "residual_png", "psf_png", "dirty_png", "uv_png"]:
        p = outdir / f"{folder_name}_{panel_kind.replace('_png','')}_comparison.png"
        make_summary_plot(
            folder_name=folder_name,
            out_png=p,
            rows=rows,
            panel_kind=panel_kind,
        )
        plot_paths.append(p)

    return summary_csv, plot_paths, outdir


def main(folder_name: str) -> None:
    summary_csv, plot_paths, outdir = run_one(folder_name)
    print(f"[DONE] summary CSV: {summary_csv}")
    for p in plot_paths:
        print(f"[DONE] plot: {p}")
    print(f"[DONE] outdir: {outdir}")


# if __name__ == "__main__":
#     main(FOLDER_NAME)
