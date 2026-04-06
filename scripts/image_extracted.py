from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from casatasks import tclean, flagdata, imhead
from casatools import table
from scripts.img_utils import casa_image_to_png
from scripts.sample_groups import (
    BAD_ANT,
    BAD_BASELINE,
    BAD_DATA,
    BEAM_SIZE_ISSUE,
    EXTRA_SOURCE,
    GOOD_ONES,
    NEEDS_BIGGER_IMAGE,
    NEEDS_MULTITERM,
    RESOLVED,
)
from scripts.vla_config import (
    band_for_frequency_ghz,
    band_matches_frequency,
    estimate_synthesized_beam_arcsec,
    representative_frequency_for_band_ghz,
    split_band_codes,
)


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
PROJECT_LIST = Path("/Users/u1528314/repos/radioastro-ml/collect/small_subset/small_selection.csv")
EXTRACTED_DIR = Path("/Users/u1528314/repos/radioastro-ml/collect/extracted2")
CALIBRATOR_BANDS_CSV = Path("/Users/u1528314/repos/radioastro-ml/collect/vla_calibrators_bands_v2.csv")

# -------------------------------------------------------------------
# Imaging policy
# -------------------------------------------------------------------
MODIFY_MS_IN_PLACE = False
USE_METADATA_FIRSTPASS = True
ARBITRARY_FIRSTPASS_BEAM_ARCSEC = 2.0
USE_ARBITRARY_FINAL_GRID = False
ARBITRARY_FINAL_BEAM_ARCSEC = 2.0
USE_MULTITERM_MFS = False
MULTITERM_NTERMS = 2
PRODUCT_PREFIX = "clean_corrected"
SELECTED_FOLDERS: list[str] | None = None

# beam-normalized final image policy
PIXELS_PER_BEAM = 4.0
FOV_IN_BEAMS = 64.0

# optional guardrails
MIN_IMSIZE = 128
MAX_IMSIZE = 1024
MIN_CELL_ARCSEC = 0.02
MAX_CELL_ARCSEC = 50.0

# clean config
TCLEAN_BASE = dict(
    specmode="mfs",
    weighting="briggs",
    robust=0.5,
    stokes="I",
    deconvolver="hogbom",
    gridder="standard",
    interactive=False,
    datacolumn="corrected",
)

FIRSTPASS_NITER = 0
FINAL_CLEAN_NITER = 100


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def load_projects(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def load_calibrator_uv_limits(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def find_extracted_ms_paths(extracted_dir: Path) -> list[Path]:
    hits = []
    for sample_dir in sorted(extracted_dir.iterdir()):
        if not sample_dir.is_dir():
            continue

        folder = sample_dir.name
        expected_ms = sample_dir / folder / f"{folder}.ms"
        if expected_ms.exists():
            hits.append(expected_ms)
            continue

        ms_list = sorted(sample_dir.rglob("*.ms"))
        if ms_list:
            hits.append(ms_list[0])

    return hits


def filter_ms_paths(ms_paths: list[Path], selected_folders: list[str] | None) -> list[Path]:
    if not selected_folders:
        return ms_paths

    wanted = [str(x).strip() for x in selected_folders if str(x).strip()]
    wanted_set = set(wanted)
    filtered = [ms_path for ms_path in ms_paths if ms_path.parent.parent.name in wanted_set]

    found_folders = {ms_path.parent.parent.name for ms_path in filtered}
    missing = [folder for folder in wanted if folder not in found_folders]
    if missing:
        print(f"[WARN] requested folders not found in extracted MS paths: {missing}")

    order = {folder: i for i, folder in enumerate(wanted)}
    filtered.sort(key=lambda ms_path: order.get(ms_path.parent.parent.name, 10**9))
    return filtered


def row_for_folder(df: pd.DataFrame, folder_name: str) -> Optional[pd.Series]:
    if "folder" not in df.columns:
        return None
    m = df["folder"].astype("string").fillna("").str.strip() == folder_name
    if not m.any():
        return None
    return df.loc[m].iloc[0]


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

    def parse_float(x):
        try:
            return float(x) if pd.notna(x) else np.nan
        except (TypeError, ValueError):
            return np.nan

    uvmin = parse_float(row.get("uvmin_kl"))
    uvmax = parse_float(row.get("uvmax_kl"))
    if np.isfinite(uvmin) and np.isfinite(uvmax) and uvmin > uvmax:
        uvmin, uvmax = uvmax, uvmin

    return {
        "calibrator_name": str(row.get("name", calibrator_name)),
        "receiver": str(row.get("receiver", receiver)),
        "uvmin_kl": uvmin,
        "uvmax_kl": uvmax,
        "cfg_A": row.get("cfg_A"),
        "cfg_B": row.get("cfg_B"),
        "cfg_C": row.get("cfg_C"),
        "cfg_D": row.get("cfg_D"),
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


def compute_uvlimit_coverage_stats(
    ms_path: Path,
    reference_freq_ghz: float | None,
    uvmin_kl: float | None,
    uvmax_kl: float | None,
) -> dict:
    if reference_freq_ghz is None or not np.isfinite(reference_freq_ghz) or reference_freq_ghz <= 0:
        return {
            "uv_observed_min_kl": np.nan,
            "uv_observed_max_kl": np.nan,
            "uv_fraction_inside_limits": np.nan,
            "uv_fraction_below_uvmin": np.nan,
            "uv_fraction_above_uvmax": np.nan,
            "uv_n_rows_total": 0,
        }

    c_m_s = 299792458.0
    lam_m = c_m_s / (reference_freq_ghz * 1e9)

    tb = table()
    tb.open(str(ms_path))
    try:
        uvw = np.array(tb.getcol("UVW"), dtype=float)
        flag_row = np.array(tb.getcol("FLAG_ROW"), dtype=bool) if "FLAG_ROW" in tb.colnames() else None
    finally:
        tb.close()

    uv_m = np.sqrt(uvw[0] ** 2 + uvw[1] ** 2)
    uv_kl = uv_m / lam_m / 1e3
    if flag_row is not None:
        uv_kl = uv_kl[~flag_row]

    if uv_kl.size == 0:
        return {
            "uv_observed_min_kl": np.nan,
            "uv_observed_max_kl": np.nan,
            "uv_fraction_inside_limits": np.nan,
            "uv_fraction_below_uvmin": np.nan,
            "uv_fraction_above_uvmax": np.nan,
            "uv_n_rows_total": 0,
        }

    inside = np.ones_like(uv_kl, dtype=bool)
    below = np.zeros_like(uv_kl, dtype=bool)
    above = np.zeros_like(uv_kl, dtype=bool)

    if uvmin_kl is not None and np.isfinite(uvmin_kl):
        below = uv_kl < uvmin_kl
        inside &= ~below
    if uvmax_kl is not None and np.isfinite(uvmax_kl):
        above = uv_kl > uvmax_kl
        inside &= ~above

    return {
        "uv_observed_min_kl": float(np.min(uv_kl)),
        "uv_observed_max_kl": float(np.max(uv_kl)),
        "uv_fraction_inside_limits": float(np.mean(inside)),
        "uv_fraction_below_uvmin": float(np.mean(below)),
        "uv_fraction_above_uvmax": float(np.mean(above)),
        "uv_n_rows_total": int(uv_kl.size),
    }


def _read_spw_frequency_info(ms_path: Path) -> list[dict]:
    tb = table()
    tb.open(str(ms_path / "SPECTRAL_WINDOW"))
    try:
        spw_info = []
        for spw in range(tb.nrows()):
            freqs = np.array(tb.getcell("CHAN_FREQ", spw), dtype=float)
            if freqs.size == 0:
                continue
            f_min = float(np.min(freqs))
            f_max = float(np.max(freqs))
            spw_info.append(
                {
                    "spw": int(spw),
                    "f_center_ghz": float(np.median(freqs) / 1e9),
                    "bandwidth_hz": f_max - f_min,
                }
            )
        return spw_info
    finally:
        tb.close()


def get_ms_reference_frequency_ghz(
    ms_path: Path,
    preferred_spw: int | None = None,
) -> tuple[float | None, int | None]:
    spw_info = _read_spw_frequency_info(ms_path)
    if not spw_info:
        return None, None

    if preferred_spw is not None:
        for info in spw_info:
            if info["spw"] == preferred_spw:
                return info["f_center_ghz"], info["spw"]

    best = max(spw_info, key=lambda x: x["bandwidth_hz"])
    return best["f_center_ghz"], best["spw"]


def choose_band_and_frequency(
    ms_path: Path,
    row: Optional[pd.Series],
) -> dict:
    row_band_codes = split_band_codes(None if row is None else row.get("band_code"))
    band_guess_codes = split_band_codes(None if row is None else row.get("band_guess"))
    preferred_spw = None
    if row is not None:
        raw_spw = row.get("spw_selected")
        if pd.notna(raw_spw):
            try:
                preferred_spw = int(raw_spw)
            except (TypeError, ValueError):
                preferred_spw = None

    ms_freq_ghz, used_spw = get_ms_reference_frequency_ghz(ms_path, preferred_spw=preferred_spw)
    csv_freq_ghz = None
    if row is not None:
        raw_csv_freq = row.get("spw_center_ghz")
        if pd.notna(raw_csv_freq):
            try:
                csv_freq_ghz = float(raw_csv_freq)
            except (TypeError, ValueError):
                csv_freq_ghz = None

    detected_band = band_for_frequency_ghz(ms_freq_ghz)
    if detected_band and band_matches_frequency(ms_freq_ghz, row_band_codes):
        selected_band = detected_band
    elif detected_band and not row_band_codes:
        selected_band = detected_band
    elif row_band_codes:
        selected_band = row_band_codes[0]
    elif band_guess_codes:
        selected_band = band_guess_codes[0]
    else:
        selected_band = detected_band

    band_match = None
    if row_band_codes and ms_freq_ghz is not None:
        band_match = band_matches_frequency(ms_freq_ghz, row_band_codes)

    return {
        "selected_band": selected_band,
        "detected_band": detected_band,
        "row_band_codes": row_band_codes,
        "band_match": band_match,
        "ms_freq_ghz": ms_freq_ghz,
        "csv_freq_ghz": csv_freq_ghz,
        "used_spw": used_spw,
    }


def remove_casa_products(imagename: str) -> None:
    stem = Path(imagename)
    parent = stem.parent
    prefix = stem.name

    if not parent.exists():
        return

    for p in parent.glob(prefix + ".*"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            try:
                p.unlink()
            except OSError:
                pass


def ensure_imaging_ms(ms_path: Path, modify_in_place: bool = True) -> Path:
    if modify_in_place:
        return ms_path

    out_ms = ms_path.with_name(ms_path.stem + "_imgprep.ms")
    if out_ms.exists():
        shutil.rmtree(out_ms)
    shutil.copytree(ms_path, out_ms)
    return out_ms


def flag_zero_visibilities(ms_path: Path) -> None:
    print(f"[FLAG] flagging exact zeros in {ms_path}")
    flagdata(
        vis=str(ms_path),
        mode="clip",
        clipzeros=True,
        flagbackup=False,
    )


def run_tclean(
    ms_path: Path,
    imagename: str,
    *,
    cell_arcsec: float,
    imsize: int,
    niter: int,
    uvrange: str = "",
    use_multiterm_mfs: bool = False,
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
    )
    if use_multiterm_mfs:
        cfg["deconvolver"] = "mtmfs"
        cfg["nterms"] = int(MULTITERM_NTERMS)

    print(
        f"[TCLEAN] vis={ms_path.name} "
        f"imagename={imagename} "
        f"cell={cell_arcsec:.6f}arcsec "
        f"imsize={imsize} "
        f"niter={niter} "
        f"uvrange={uvrange or 'all'} "
        f"deconvolver={cfg.get('deconvolver')} "
        f"nterms={cfg.get('nterms', 1)}"
    )
    tclean(**cfg)


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


def read_restoring_beam_arcsec(image_path: str | Path) -> Tuple[float, float, float]:
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


def choose_firstpass_imaging_setup(
    config: str | None,
    reference_freq_ghz: float | None,
    band: str | None,
) -> tuple[float, int, float | None]:
    if not USE_METADATA_FIRSTPASS:
        cell_arcsec = choose_beam_based_cell_arcsec(
            ARBITRARY_FIRSTPASS_BEAM_ARCSEC,
            ARBITRARY_FIRSTPASS_BEAM_ARCSEC,
            pixels_per_beam=PIXELS_PER_BEAM,
        )
        imsize, _ = choose_imsize_for_beam_normalized_fov(
            ARBITRARY_FIRSTPASS_BEAM_ARCSEC,
            cell_arcsec,
            fov_in_beams=FOV_IN_BEAMS,
        )
        return cell_arcsec, imsize, ARBITRARY_FIRSTPASS_BEAM_ARCSEC

    if reference_freq_ghz is None or not np.isfinite(reference_freq_ghz) or reference_freq_ghz <= 0:
        reference_freq_ghz = representative_frequency_for_band_ghz(band)

    estimated_beam_arcsec = estimate_synthesized_beam_arcsec(config, reference_freq_ghz)

    if estimated_beam_arcsec is None or not np.isfinite(estimated_beam_arcsec) or estimated_beam_arcsec <= 0:
        cell_arcsec = 0.5
        imsize = 256
        return cell_arcsec, imsize, None

    cell_arcsec = choose_beam_based_cell_arcsec(
        estimated_beam_arcsec,
        estimated_beam_arcsec,
        pixels_per_beam=PIXELS_PER_BEAM,
    )
    imsize, _ = choose_imsize_for_beam_normalized_fov(
        estimated_beam_arcsec,
        cell_arcsec,
        fov_in_beams=FOV_IN_BEAMS,
    )
    return cell_arcsec, imsize, estimated_beam_arcsec


def choose_final_imaging_setup(
    bmaj_arcsec: float,
    bmin_arcsec: float,
) -> tuple[float, int, float]:
    if USE_ARBITRARY_FINAL_GRID:
        final_cell = choose_beam_based_cell_arcsec(
            ARBITRARY_FINAL_BEAM_ARCSEC,
            ARBITRARY_FINAL_BEAM_ARCSEC,
            pixels_per_beam=PIXELS_PER_BEAM,
        )
        final_imsize, final_fov_arcsec = choose_imsize_for_beam_normalized_fov(
            ARBITRARY_FINAL_BEAM_ARCSEC,
            final_cell,
            fov_in_beams=FOV_IN_BEAMS,
        )
        return final_cell, final_imsize, final_fov_arcsec

    final_cell = choose_beam_based_cell_arcsec(
        bmaj_arcsec, bmin_arcsec, pixels_per_beam=PIXELS_PER_BEAM
    )
    final_imsize, final_fov_arcsec = choose_imsize_for_beam_normalized_fov(
        bmin_arcsec, final_cell, fov_in_beams=FOV_IN_BEAMS
    )
    return final_cell, final_imsize, final_fov_arcsec


def image_to_png_if_exists(
    casa_image: Path,
    png_path: Path,
    *,
    title: str | None = None,
    draw_beam_ellipse: bool = False,
) -> None:
    if casa_image.exists():
        casa_image_to_png(
            str(casa_image),
            str(png_path),
            title=title,
            draw_beam_ellipse=draw_beam_ellipse,
        )


def process_one_sample(
    ms_path: Path,
    row: Optional[pd.Series] = None,
    calib_df: Optional[pd.DataFrame] = None,
) -> dict:
    sample_leaf = ms_path.parent
    sample_top = sample_leaf.parent
    folder_name = sample_top.name

    ms_for_imaging = ensure_imaging_ms(ms_path, modify_in_place=MODIFY_MS_IN_PLACE)
    gain_array_config = None if row is None else row.get("gain_array_config")
    band_info = choose_band_and_frequency(ms_for_imaging, row)
    uv_limit_info = None
    if calib_df is not None:
        uv_limit_info = lookup_calibrator_uv_limits(
            calib_df,
            calibrator_name=folder_name if row is None else row.get("name", folder_name),
            band=band_info["selected_band"],
        )
    uv_range = ""
    uv_cov = {
        "uv_observed_min_kl": np.nan,
        "uv_observed_max_kl": np.nan,
        "uv_fraction_inside_limits": np.nan,
        "uv_fraction_below_uvmin": np.nan,
        "uv_fraction_above_uvmax": np.nan,
        "uv_n_rows_total": 0,
    }
    if uv_limit_info is not None:
        uv_range = make_uvrange_string(uv_limit_info["uvmin_kl"], uv_limit_info["uvmax_kl"])
        uv_cov = compute_uvlimit_coverage_stats(
            ms_for_imaging,
            band_info["ms_freq_ghz"],
            uv_limit_info["uvmin_kl"],
            uv_limit_info["uvmax_kl"],
        )
    firstpass_cell, firstpass_imsize, estimated_beam_arcsec = choose_firstpass_imaging_setup(
        gain_array_config,
        band_info["ms_freq_ghz"],
        band_info["selected_band"],
    )

    # 1) flag zeros
    flag_zero_visibilities(ms_for_imaging)

    # 2) first-pass dirty image to estimate beam
    firstpass_base = sample_top / "beam_firstpass_dirty"
    print(
        f"[FIRSTPASS] {folder_name} | "
        f"mode={'metadata' if USE_METADATA_FIRSTPASS else 'arbitrary'} | "
        f"config={gain_array_config} | "
        f"band_csv={band_info['row_band_codes']} | "
        f"band_detected={band_info['detected_band']} | "
        f"selected_band={band_info['selected_band']} | "
        f"freq_ms={band_info['ms_freq_ghz'] if band_info['ms_freq_ghz'] is not None else float('nan'):.3f} GHz | "
        f"spw={band_info['used_spw']} | "
        f"uvrange={uv_range or 'all'} | "
        f"beam_est={estimated_beam_arcsec if estimated_beam_arcsec is not None else float('nan'):.3f}\" | "
        f"cell={firstpass_cell:.4f}\" | imsize={firstpass_imsize}"
    )
    run_tclean(
        ms_for_imaging,
        str(firstpass_base),
        cell_arcsec=firstpass_cell,
        imsize=firstpass_imsize,
        niter=FIRSTPASS_NITER,
        uvrange=uv_range,
    )

    firstpass_image = firstpass_base.with_suffix(".image")
    bmaj, bmin, bpa = read_restoring_beam_arcsec(firstpass_image)

    # 3) final imaging grid
    final_cell, final_imsize, final_fov_arcsec = choose_final_imaging_setup(
        bmaj,
        bmin,
    )

    # 4) final dirty
    dirty_base = sample_top / f"{PRODUCT_PREFIX}_dirty"
    run_tclean(
        ms_for_imaging,
        str(dirty_base),
        cell_arcsec=final_cell,
        imsize=final_imsize,
        niter=0,
        uvrange=uv_range,
    )

    # 5) final clean
    clean_base = sample_top / f"{PRODUCT_PREFIX}_clean"
    run_tclean(
        ms_for_imaging,
        str(clean_base),
        cell_arcsec=final_cell,
        imsize=final_imsize,
        niter=FINAL_CLEAN_NITER,
        uvrange=uv_range,
        use_multiterm_mfs=USE_MULTITERM_MFS,
    )

    final_clean_image = (
        Path(str(clean_base) + ".image.tt0")
        if USE_MULTITERM_MFS
        else clean_base.with_suffix(".image")
    )
    final_bmaj, final_bmin, final_bpa = read_restoring_beam_arcsec(final_clean_image)
    uv_inside_pct = 100.0 * uv_cov["uv_fraction_inside_limits"] if np.isfinite(uv_cov["uv_fraction_inside_limits"]) else np.nan

    clean_title = (
        f"{folder_name} | {band_info['selected_band'] or '?'} | {gain_array_config or '?'} | "
        f"beam={final_bmaj:.2f}\"x{final_bmin:.2f}\" | "
        f"uv={uv_range or 'all'} | in={uv_inside_pct if np.isfinite(uv_inside_pct) else float('nan'):.1f}%"
    )
    dirty_title = (
        f"{folder_name} dirty | {band_info['selected_band'] or '?'} | {gain_array_config or '?'} | "
        f"beam={final_bmaj:.2f}\"x{final_bmin:.2f}\" | "
        f"uv={uv_range or 'all'} | in={uv_inside_pct if np.isfinite(uv_inside_pct) else float('nan'):.1f}%"
    )

    # 6) export pngs
    image_to_png_if_exists(
        dirty_base.with_suffix(".image"),
        sample_top / f"{PRODUCT_PREFIX}_dirty.png",
        title=dirty_title,
        draw_beam_ellipse=True,
    )
    image_to_png_if_exists(
        final_clean_image,
        sample_top / f"{PRODUCT_PREFIX}_clean.png",
        title=clean_title,
        draw_beam_ellipse=True,
    )

    result = {
        "folder": folder_name,
        "ms": str(ms_path),
        "beam_major_arcsec": bmaj,
        "beam_minor_arcsec": bmin,
        "beam_pa_deg": bpa,
        "firstpass_beam_major_arcsec": bmaj,
        "firstpass_beam_minor_arcsec": bmin,
        "firstpass_beam_pa_deg": bpa,
        "final_beam_major_arcsec": final_bmaj,
        "final_beam_minor_arcsec": final_bmin,
        "final_beam_pa_deg": final_bpa,
        "gain_array_config": gain_array_config,
        "band_code_csv": " ".join(band_info["row_band_codes"]),
        "band_detected_from_freq": band_info["detected_band"],
        "band_used_for_firstpass": band_info["selected_band"],
        "band_matches_frequency": band_info["band_match"],
        "spw_used_for_band_check": band_info["used_spw"],
        "spw_center_ghz_ms": band_info["ms_freq_ghz"],
        "spw_center_ghz_csv": band_info["csv_freq_ghz"],
        "firstpass_beam_estimate_arcsec": estimated_beam_arcsec,
        "firstpass_cell_arcsec": firstpass_cell,
        "firstpass_imsize": firstpass_imsize,
        "firstpass_mode": "metadata" if USE_METADATA_FIRSTPASS else "arbitrary",
        "final_grid_mode": "arbitrary" if USE_ARBITRARY_FINAL_GRID else "beam_based",
        "clean_mode": "mtmfs" if USE_MULTITERM_MFS else "standard",
        "product_prefix": PRODUCT_PREFIX,
        "catalog_uv_receiver": None if uv_limit_info is None else uv_limit_info["receiver"],
        "catalog_uvmin_kl": np.nan if uv_limit_info is None else uv_limit_info["uvmin_kl"],
        "catalog_uvmax_kl": np.nan if uv_limit_info is None else uv_limit_info["uvmax_kl"],
        "catalog_uvrange": uv_range,
        "uv_observed_min_kl": uv_cov["uv_observed_min_kl"],
        "uv_observed_max_kl": uv_cov["uv_observed_max_kl"],
        "uv_fraction_inside_limits": uv_cov["uv_fraction_inside_limits"],
        "uv_fraction_below_uvmin": uv_cov["uv_fraction_below_uvmin"],
        "uv_fraction_above_uvmax": uv_cov["uv_fraction_above_uvmax"],
        "uv_n_rows_total": uv_cov["uv_n_rows_total"],
        "cell_arcsec": final_cell,
        "imsize": final_imsize,
        "fov_arcsec": final_fov_arcsec,
        "fov_in_beams_minor": final_fov_arcsec / bmin if bmin > 0 else np.nan,
        "pixels_per_beam_minor": bmin / final_cell if final_cell > 0 else np.nan,
    }

    print(
        f"[DONE] {folder_name} | "
        f"beam=({final_bmaj:.3f}\", {final_bmin:.3f}\", {final_bpa:.1f} deg) | "
        f"final_grid={'arbitrary' if USE_ARBITRARY_FINAL_GRID else 'beam_based'} | "
        f"uvrange={uv_range or 'all'} | "
        f"inside={uv_inside_pct if np.isfinite(uv_inside_pct) else float('nan'):.1f}% | "
        f"cell={final_cell:.4f}\" | imsize={final_imsize} | "
        f"FoV={final_fov_arcsec:.2f}\" | "
        f"FoV/beams={final_fov_arcsec / final_bmin:.2f}"
    )

    return result


def main():
    df = load_projects(PROJECT_LIST)
    calib_df = load_calibrator_uv_limits(CALIBRATOR_BANDS_CSV)
    ms_paths = find_extracted_ms_paths(EXTRACTED_DIR)
    ms_paths = filter_ms_paths(ms_paths, SELECTED_FOLDERS)

    print(f"[INFO] found {len(ms_paths)} extracted MS files")

    rows = []
    for ms_path in ms_paths:
        folder = ms_path.parent.parent.name
        row = row_for_folder(df, folder)
        try:
            out = process_one_sample(ms_path, row=row, calib_df=calib_df)

            if row is not None:
                out["name"] = str(row.get("name", folder))
                out["minutes"] = row.get("extracted_gain_onsource_min", np.nan)
                out["status_csv"] = str(row.get("status", ""))
            else:
                out["name"] = folder
                out["minutes"] = np.nan
                out["status_csv"] = ""

            rows.append(out)

        except Exception as e:
            print(f"[ERROR] {ms_path}: {type(e).__name__}: {e}")
            rows.append(
                {
                    "folder": ms_path.parent.parent.name,
                    "ms": str(ms_path),
                    "error": f"{type(e).__name__}: {e}",
                }
            )

    summary_csv = EXTRACTED_DIR / "beam_imaging_summary.csv"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    print(f"[OK] wrote summary to {summary_csv}")


# if __name__ == "__main__":
#     main()
