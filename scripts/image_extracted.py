from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from casatasks import tclean, flagdata, imhead
from casatools import image, table
from casaplotms import plotms
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
    UV_LIM,
    GOOD_ONES_2,
    NEED_SELFCAL_2,
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
DEFAULT_EXTRACTED_DIR = Path("/Users/u1528314/repos/radioastro-ml/collect/extracted")
CALIBRATOR_BANDS_CSV = Path("/Users/u1528314/repos/radioastro-ml/collect/vla_calibrators_bands_v2.csv")

# -------------------------------------------------------------------
# Imaging policy
# -------------------------------------------------------------------
MODIFY_MS_IN_PLACE = False
USE_METADATA_FIRSTPASS = True
ARBITRARY_FIRSTPASS_BEAM_ARCSEC = 2.0
USE_ARBITRARY_FINAL_GRID = False
ARBITRARY_FINAL_BEAM_ARCSEC = 2.0
USE_MULTITERM_MFS = True
MULTITERM_NTERMS = 2
PRODUCT_PREFIX = "clean_corrected"
SELECTED_FOLDERS: list[str] | None = ["0653+370"]
SELFCAL_DIRNAME = "selfcal"
APPLY_CATALOG_UVLIMIT_FILTERING = False

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
MULTITERM_NITER = 5000
MULTITERM_NSIGMA = 3.0
MULTITERM_CYCLENITER = 100
FINAL_CLEAN_BOX_MASK_NBEAMS: float | None = 48.0


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def load_projects(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def resolve_extracted_dir(
    selected_folders: list[str] | None,
    default_dir: Path = DEFAULT_EXTRACTED_DIR,
) -> Path:
    if not selected_folders:
        return default_dir

    wanted = [str(x).strip() for x in selected_folders if str(x).strip()]
    if not wanted:
        return default_dir

    missing = [folder for folder in wanted if not (default_dir / folder).exists()]
    if missing:
        print(f"[WARN] requested folders not found under {default_dir}: {missing}")

    return default_dir


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

        ms_list = sorted(
            ms_path
            for ms_path in sample_dir.rglob("*.ms")
            if SELFCAL_DIRNAME not in ms_path.relative_to(sample_dir).parts
        )
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


def sample_top_for_ms(ms_path: Path) -> Path:
    return ms_path.parent.parent


def image_output_dir_for_sample(sample_top: Path) -> Path:
    return sample_top


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


def ensure_imaging_ms(
    ms_path: Path,
    modify_in_place: bool = True,
    output_dir: Path | None = None,
) -> Path:
    if modify_in_place:
        return ms_path

    out_parent = ms_path.parent if output_dir is None else output_dir
    out_parent.mkdir(parents=True, exist_ok=True)
    out_ms = out_parent / f"{ms_path.stem}_imgprep.ms"
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
    apply_multiterm_clean_controls: bool = True,
    usemask: str = "",
    mask: str = "",
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
        if apply_multiterm_clean_controls:
            cfg["niter"] = int(MULTITERM_NITER)
            cfg["nsigma"] = float(MULTITERM_NSIGMA)
            cfg["cycleniter"] = int(MULTITERM_CYCLENITER)
    if usemask:
        cfg["usemask"] = usemask
    if mask:
        cfg["mask"] = mask

    print(
        f"[TCLEAN] vis={ms_path.name} "
        f"imagename={imagename} "
        f"cell={cell_arcsec:.6f}arcsec "
        f"imsize={imsize} "
        f"niter={cfg['niter']} "
        f"uvrange={uvrange or 'all'} "
        f"deconvolver={cfg.get('deconvolver')} "
        f"nterms={cfg.get('nterms', 1)} "
        f"nsigma={cfg.get('nsigma', 'default')} "
        f"cycleniter={cfg.get('cycleniter', 'default')} "
        f"mt_clean_controls={apply_multiterm_clean_controls} "
        f"usemask={cfg.get('usemask', 'none')} "
        f"mask={cfg.get('mask', 'none')}"
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


def load_image_2d(image_path: str | Path) -> np.ndarray:
    ia = image()
    ia.open(str(image_path))
    try:
        arr = np.asarray(ia.getchunk())
    finally:
        ia.close()

    arr = np.squeeze(arr)
    if arr.ndim == 4:
        arr = arr[:, :, 0, 0]
    elif arr.ndim == 3:
        arr = arr[:, :, 0]
    elif arr.ndim != 2:
        raise RuntimeError(f"Unexpected image ndim={arr.ndim} for {image_path}")

    return np.asarray(arr, dtype=float).T


def robust_sigma(values: np.ndarray) -> float:
    x = values[np.isfinite(values)]
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return 1.4826 * mad


def clean_residual_qa_metrics(clean_image: Path, residual_image: Path) -> dict:
    clean = load_image_2d(clean_image)
    residual = load_image_2d(residual_image)

    clean_vals = clean[np.isfinite(clean)]
    residual_vals = residual[np.isfinite(residual)]
    if clean_vals.size == 0:
        raise RuntimeError(f"No finite pixels in {clean_image}")
    if residual_vals.size == 0:
        raise RuntimeError(f"No finite pixels in {residual_image}")

    residual_robust_sigma = robust_sigma(residual_vals)
    residual_max_abs = float(np.max(np.abs(residual_vals)))
    residual_p99_abs = float(np.percentile(np.abs(residual_vals), 99.0))
    residual_p995_abs = float(np.percentile(np.abs(residual_vals), 99.5))
    clean_max_abs = float(np.max(np.abs(clean_vals)))
    return {
        "residual_robust_sigma_jy_per_beam": residual_robust_sigma,
        "residual_p99_abs_over_sigma": (
            residual_p99_abs / residual_robust_sigma
            if np.isfinite(residual_robust_sigma) and residual_robust_sigma > 0
            else float("nan")
        ),
        "residual_p995_abs_over_sigma": (
            residual_p995_abs / residual_robust_sigma
            if np.isfinite(residual_robust_sigma) and residual_robust_sigma > 0
            else float("nan")
        ),
        "residual_peak_to_sigma": (
            residual_max_abs / residual_robust_sigma
            if np.isfinite(residual_robust_sigma) and residual_robust_sigma > 0
            else float("nan")
        ),
        "dynamic_range": (
            clean_max_abs / residual_robust_sigma
            if np.isfinite(residual_robust_sigma) and residual_robust_sigma > 0
            else float("nan")
        ),
    }


def ratio(after: float, before: float) -> float:
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


def read_one_row_csv(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] could not read {path}: {exc}")
        return {}
    if df.empty:
        return {}
    return dict(df.iloc[0])


def parse_metric(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def first_finite_metric(row: dict, keys: list[str]) -> float:
    for key in keys:
        value = parse_metric(row.get(key))
        if np.isfinite(value):
            return value
    return float("nan")


def load_selfcal_qa_metrics(sample_top: Path, original_metrics: dict) -> dict:
    metrics_csv = sample_top / SELFCAL_DIRNAME / "selfcal_improvement_metrics.csv"
    row = read_one_row_csv(metrics_csv)
    if not row:
        return {}

    out = {
        "selfcal_residual_robust_sigma_jy_per_beam": first_finite_metric(
            row,
            [
                "selfcal_residual_robust_sigma_jy_per_beam",
            ],
        ),
        "selfcal_residual_peak_to_sigma": first_finite_metric(
            row,
            [
                "selfcal_residual_peak_to_sigma",
                "selfcal_residual_max_abs_over_robust_sigma",
            ],
        ),
        "selfcal_residual_p99_abs_over_sigma": first_finite_metric(
            row,
            [
                "selfcal_residual_p99_abs_over_sigma",
            ],
        ),
        "selfcal_residual_p995_abs_over_sigma": first_finite_metric(
            row,
            [
                "selfcal_residual_p995_abs_over_sigma",
            ],
        ),
        "selfcal_dynamic_range": first_finite_metric(
            row,
            [
                "selfcal_dynamic_range",
                "selfcal_residual_dynamic_range",
            ],
        ),
    }
    out["residual_robust_sigma_ratio_selfcal_over_original"] = ratio(
        out["selfcal_residual_robust_sigma_jy_per_beam"],
        original_metrics.get("residual_robust_sigma_jy_per_beam"),
    )
    out["residual_peak_to_sigma_ratio_selfcal_over_original"] = ratio(
        out["selfcal_residual_peak_to_sigma"],
        original_metrics.get("residual_peak_to_sigma"),
    )
    out["residual_p99_abs_over_sigma_ratio_selfcal_over_original"] = ratio(
        out["selfcal_residual_p99_abs_over_sigma"],
        original_metrics.get("residual_p99_abs_over_sigma"),
    )
    out["residual_p995_abs_over_sigma_ratio_selfcal_over_original"] = ratio(
        out["selfcal_residual_p995_abs_over_sigma"],
        original_metrics.get("residual_p995_abs_over_sigma"),
    )
    out["dynamic_range_ratio_selfcal_over_original"] = ratio(
        out["selfcal_dynamic_range"],
        original_metrics.get("dynamic_range"),
    )
    return out


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


def format_plot_metric(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "none"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def beam_box_mask_string(
    *,
    imsize: int,
    cell_arcsec: float,
    bmaj_arcsec: float,
    bmin_arcsec: float,
    nbeams: float,
) -> str:
    if cell_arcsec <= 0:
        raise ValueError(f"cell_arcsec must be > 0, got {cell_arcsec}")
    if bmaj_arcsec <= 0 or bmin_arcsec <= 0:
        raise ValueError(
            f"beam axes must be > 0, got bmaj={bmaj_arcsec}, bmin={bmin_arcsec}"
        )
    if nbeams <= 0:
        raise ValueError(f"nbeams must be > 0, got {nbeams}")

    width_pix = max(1, int(np.ceil((nbeams * bmaj_arcsec) / cell_arcsec)))
    height_pix = max(1, int(np.ceil((nbeams * bmin_arcsec) / cell_arcsec)))

    cx = imsize // 2
    cy = imsize // 2
    half_width = max(1, int(np.ceil(width_pix / 2.0)))
    half_height = max(1, int(np.ceil(height_pix / 2.0)))

    x0 = max(0, cx - half_width)
    y0 = max(0, cy - half_height)
    x1 = min(imsize - 1, cx + half_width)
    y1 = min(imsize - 1, cy + half_height)
    return f"box[[{x0}pix,{y0}pix],[{x1}pix,{y1}pix]]"


def image_to_png_if_exists(
    casa_image: Path,
    png_path: Path,
    *,
    title: str | None = None,
    draw_beam_ellipse: bool = False,
    symmetric: bool = True,
    cmap: str = "inferno",
) -> None:
    if casa_image.exists():
        casa_image_to_png(
            str(casa_image),
            str(png_path),
            title=title,
            draw_beam_ellipse=draw_beam_ellipse,
            symmetric=symmetric,
            cmap=cmap,
        )


def export_uv_png(
    ms_path: Path,
    png_path: Path,
    *,
    spw: str = "",
    uvrange: str = "",
) -> None:
    print(f"[PLOTMS] uv coverage -> {png_path}")
    plotms(
        vis=str(ms_path),
        xaxis="u",
        yaxis="v",
        spw=spw,
        uvrange=uvrange,
        avgchannel="1e8",
        avgtime="1e8",
        coloraxis="spw",
        plotfile=str(png_path),
        expformat="png",
        highres=True,
        overwrite=True,
        showgui=False,
    )


def export_amp_vs_uvdist_png(
    ms_path: Path,
    png_path: Path,
    *,
    spw: str = "",
    uvrange: str = "",
) -> None:
    print(f"[PLOTMS] amp vs uv-dist -> {png_path}")
    ymin, ymax = _amplitude_yrange(
        ms_path,
        spw=spw,
        uvrange=uvrange,
        avgchannel=2,
        min_span=2.0,
        mode="floor",
    )
    data_col = "corrected"
    tb = table()
    tb.open(str(ms_path))
    try:
        if "CORRECTED_DATA" not in tb.colnames():
            data_col = "data"
    finally:
        tb.close()
    plotms(
        vis=str(ms_path),
        xaxis="UVdist",
        yaxis="amp",
        ydatacolumn=data_col,
        spw=spw,
        uvrange=uvrange,
        correlation="RR,LL",
        averagedata=True,
        avgchannel="2",
        avgtime="0",
        avgscan=False,
        avgfield=False,
        coloraxis="antenna1",
        plotrange=[-1, -1, ymin, ymax],
        title="amp vs uv-dist | avgch=2 avgtime=0 avgscan=False avgfield=False",
        plotfile=str(png_path),
        expformat="png",
        highres=True,
        overwrite=True,
        showgui=False,
    )


def _parse_spw_selection(spw: str) -> set[int] | None:
    spw = str(spw).strip()
    if not spw:
        return None

    selected: set[int] = set()
    for part in spw.split(","):
        token = part.strip()
        if not token:
            continue
        token = token.split(":")[0].strip()
        if "~" in token:
            lo_str, hi_str = token.split("~", 1)
            try:
                lo = int(lo_str)
                hi = int(hi_str)
            except ValueError:
                continue
            for val in range(min(lo, hi), max(lo, hi) + 1):
                selected.add(val)
            continue
        try:
            selected.add(int(token))
        except ValueError:
            continue
    return selected or None


def _parse_uvrange_limits(uvrange: str) -> tuple[float | None, float | None]:
    text = str(uvrange).strip().lower().replace(" ", "")
    if not text:
        return None, None

    def parse_klambda(value: str) -> float | None:
        if value.endswith("klambda"):
            value = value[:-7]
        try:
            return float(value)
        except ValueError:
            return None

    if text.startswith(">"):
        return parse_klambda(text[1:]), None
    if text.startswith("<"):
        return None, parse_klambda(text[1:])
    if "~" in text:
        lo_str, hi_str = text.split("~", 1)
        return parse_klambda(lo_str), parse_klambda(hi_str)
    return None, None


def _load_selected_visibility_amplitudes(
    ms_path: Path,
    *,
    spw: str = "",
    uvrange: str = "",
    avgchannel: int = 1,
) -> tuple[np.ma.MaskedArray, np.ndarray, np.ndarray, str]:
    tb = table()

    tb.open(str(ms_path / "DATA_DESCRIPTION"))
    try:
        ddid_to_spw = np.array(tb.getcol("SPECTRAL_WINDOW_ID"), dtype=int)
    finally:
        tb.close()

    ref_freq_ghz, _ = get_ms_reference_frequency_ghz(ms_path)
    c_m_s = 299792458.0
    lam_m = c_m_s / (ref_freq_ghz * 1e9) if ref_freq_ghz and np.isfinite(ref_freq_ghz) and ref_freq_ghz > 0 else None

    tb.open(str(ms_path))
    try:
        data_col = "CORRECTED_DATA" if "CORRECTED_DATA" in tb.colnames() else "DATA"
        data = np.array(tb.getcol(data_col))
        flags = np.array(tb.getcol("FLAG"), dtype=bool)
        uvw = np.array(tb.getcol("UVW"), dtype=float)
        antenna1 = np.array(tb.getcol("ANTENNA1"), dtype=int)
        ddid = np.array(tb.getcol("DATA_DESC_ID"), dtype=int)
        flag_row = np.array(tb.getcol("FLAG_ROW"), dtype=bool) if "FLAG_ROW" in tb.colnames() else np.zeros(tb.nrows(), dtype=bool)
    finally:
        tb.close()

    uvdist_m = np.sqrt(uvw[0] ** 2 + uvw[1] ** 2)
    row_mask = ~flag_row

    selected_spws = _parse_spw_selection(spw)
    if selected_spws is not None:
        row_spw = ddid_to_spw[ddid]
        row_mask &= np.isin(row_spw, list(selected_spws))

    if lam_m is not None:
        uvdist_kl = uvdist_m / lam_m / 1e3
        uvmin_kl, uvmax_kl = _parse_uvrange_limits(uvrange)
        if uvmin_kl is not None:
            row_mask &= uvdist_kl >= uvmin_kl
        if uvmax_kl is not None:
            row_mask &= uvdist_kl <= uvmax_kl

    if not np.any(row_mask):
        return np.ma.array(np.empty((0, 0, 0))), np.array([]), np.array([]), data_col

    data = data[:, :, row_mask]
    flags = flags[:, :, row_mask]
    uvdist_m = uvdist_m[row_mask]
    antenna1 = antenna1[row_mask]

    ncorr = data.shape[0]
    corr_idx = [0] if ncorr == 1 else [0, ncorr - 1]

    amps = np.abs(data[corr_idx]).astype(float)
    amp_flags = flags[corr_idx]

    if avgchannel > 1:
        nchan = amps.shape[1]
        ntrim = (nchan // avgchannel) * avgchannel
        if ntrim == 0:
            ntrim = nchan
            avgchannel = 1
        amps = amps[:, :ntrim, :]
        amp_flags = amp_flags[:, :ntrim, :]
        if avgchannel > 1:
            amps = amps.reshape(amps.shape[0], ntrim // avgchannel, avgchannel, amps.shape[2])
            amp_flags = amp_flags.reshape(amp_flags.shape[0], ntrim // avgchannel, avgchannel, amp_flags.shape[2])
            amps = np.ma.array(amps, mask=amp_flags).mean(axis=2)
        else:
            amps = np.ma.array(amps, mask=amp_flags)
    else:
        amps = np.ma.array(amps, mask=amp_flags)

    return amps, uvdist_m, antenna1, data_col


def _amplitude_yrange(
    ms_path: Path,
    *,
    spw: str = "",
    uvrange: str = "",
    avgchannel: int = 1,
    min_span: float = 1.0,
    mode: str = "cap",
) -> tuple[float, float]:
    amps, _, _, _ = _load_selected_visibility_amplitudes(
        ms_path,
        spw=spw,
        uvrange=uvrange,
        avgchannel=avgchannel,
    )
    if amps.size == 0:
        return 0.0, min_span

    amp_flat = np.ma.filled(amps, np.nan).reshape(-1)
    finite = amp_flat[np.isfinite(amp_flat)]
    if finite.size == 0:
        return 0.0, min_span

    ymin = float(np.nanmin(finite))
    ymax = float(np.nanmax(finite))
    median = float(np.nanmedian(finite))
    span = ymax - ymin

    if mode == "cap" and span > min_span:
        half = 0.5 * min_span
        return median - half, median + half

    if mode == "floor" and span < min_span:
        half = 0.5 * min_span
        return median - half, median + half

    if span <= 0:
        half = 0.5 * min_span
        return median - half, median + half

    return ymin, ymax


def export_normalized_amp_vs_uvdist_png(
    ms_path: Path,
    png_path: Path,
    *,
    spw: str = "",
    uvrange: str = "",
    avgchannel: int = 2,
) -> None:
    print(f"[MATPLOTLIB] normalized amp vs uv-dist -> {png_path}")
    amps, uvdist_m, antenna1, data_col = _load_selected_visibility_amplitudes(
        ms_path,
        spw=spw,
        uvrange=uvrange,
        avgchannel=avgchannel,
    )
    if data_col != "CORRECTED_DATA":
        print(f"[WARN] {ms_path.name}: CORRECTED_DATA missing, using DATA for normalized amp vs uv-dist")

    if amps.size == 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, "No valid visibilities", ha="center", va="center")
        ax.axis("off")
        fig.savefig(png_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    amp_flat = np.ma.filled(amps, np.nan).reshape(-1)
    uv_flat = np.broadcast_to(uvdist_m[np.newaxis, np.newaxis, :], amps.shape).reshape(-1)
    ant_flat = np.broadcast_to(antenna1[np.newaxis, np.newaxis, :], amps.shape).reshape(-1)

    keep = np.isfinite(amp_flat) & np.isfinite(uv_flat)
    amp_flat = amp_flat[keep]
    uv_flat = uv_flat[keep]
    ant_flat = ant_flat[keep]

    if amp_flat.size == 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, "No finite amplitudes after filtering", ha="center", va="center")
        ax.axis("off")
        fig.savefig(png_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    median_amp = np.nanmedian(amp_flat) if amp_flat.size else np.nan

    if not np.isfinite(median_amp) or median_amp <= 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, "Median amplitude unavailable", ha="center", va="center")
        ax.axis("off")
        fig.savefig(png_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    norm_amp = amp_flat / median_amp
    ymax = max(2.0, float(np.nanmax(norm_amp))) if norm_amp.size else 2.0

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        uv_flat,
        norm_amp,
        c=ant_flat,
        s=4,
        alpha=0.55,
        cmap="tab20",
        linewidths=0,
        rasterized=True,
    )
    ax.set_xlabel("UVdist")
    ax.set_ylabel("amp / median(A)")
    ax.set_ylim(0.0, ymax)
    ax.set_title("Normalized amp vs uv-dist\navgch=2 avgtime=0 avgscan=False avgfield=False")
    ax.grid(alpha=0.2)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("ANTENNA1")

    fig.tight_layout()
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_optional_export(label: str, fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except Exception as e:
        print(f"[WARN] optional diagnostic '{label}' failed: {type(e).__name__}: {e}")


def export_spectrum_png(
    ms_path: Path,
    png_path: Path,
    *,
    spw: str = "",
    uvrange: str = "",
    avgbaseline: bool = False,
) -> None:
    print(f"[PLOTMS] spectrum -> {png_path}")
    ymin, ymax = _amplitude_yrange(
        ms_path,
        spw=spw,
        uvrange=uvrange,
        avgchannel=1,
        min_span=1.0,
        mode="floor",
    )
    data_col = "corrected"
    tb = table()
    tb.open(str(ms_path))
    try:
        if "CORRECTED_DATA" not in tb.colnames():
            data_col = "data"
    finally:
        tb.close()
    avg_label = f"avgtime=1e8 avgscan=True avgfield=True avgbaseline={avgbaseline}"
    plotms(
        vis=str(ms_path),
        xaxis="freq",
        yaxis="amp",
        ydatacolumn=data_col,
        spw=spw,
        uvrange=uvrange,
        correlation="RR,LL",
        averagedata=True,
        avgtime="1e8",
        avgscan=True,
        avgfield=True,
        avgbaseline=avgbaseline,
        coloraxis="antenna1",
        plotrange=[-1, -1, ymin, ymax],
        title=f"spectrum | {avg_label}",
        plotfile=str(png_path),
        expformat="png",
        highres=True,
        overwrite=True,
        showgui=False,
    )


def process_one_sample(
    ms_path: Path,
    row: Optional[pd.Series] = None,
    calib_df: Optional[pd.DataFrame] = None,
) -> dict:
    sample_top = sample_top_for_ms(ms_path)
    folder_name = sample_top.name
    source_label = "original"
    output_dir = image_output_dir_for_sample(sample_top)
    output_dir.mkdir(parents=True, exist_ok=True)

    ms_for_imaging = ensure_imaging_ms(
        ms_path,
        modify_in_place=MODIFY_MS_IN_PLACE,
        output_dir=output_dir,
    )
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
    applied_uvrange = uv_range if APPLY_CATALOG_UVLIMIT_FILTERING else ""
    firstpass_cell, firstpass_imsize, estimated_beam_arcsec = choose_firstpass_imaging_setup(
        gain_array_config,
        band_info["ms_freq_ghz"],
        band_info["selected_band"],
    )

    # 1) flag zeros
    flag_zero_visibilities(ms_for_imaging)

    # 2) first-pass dirty image to estimate beam
    firstpass_base = output_dir / "beam_firstpass_dirty"
    print(
        f"[FIRSTPASS] {folder_name} | "
        f"mode={'metadata' if USE_METADATA_FIRSTPASS else 'arbitrary'} | "
        f"config={gain_array_config} | "
        f"band_csv={band_info['row_band_codes']} | "
        f"band_detected={band_info['detected_band']} | "
        f"selected_band={band_info['selected_band']} | "
        f"freq_ms={band_info['ms_freq_ghz'] if band_info['ms_freq_ghz'] is not None else float('nan'):.3f} GHz | "
        f"spw={band_info['used_spw']} | "
        f"uvrange={applied_uvrange or 'all'} | "
        f"catalog_uvrange={uv_range or 'none'} | "
        f"beam_est={estimated_beam_arcsec if estimated_beam_arcsec is not None else float('nan'):.3f}\" | "
        f"cell={firstpass_cell:.4f}\" | imsize={firstpass_imsize}"
    )
    run_tclean(
        ms_for_imaging,
        str(firstpass_base),
        cell_arcsec=firstpass_cell,
        imsize=firstpass_imsize,
        niter=FIRSTPASS_NITER,
        uvrange=applied_uvrange,
        use_multiterm_mfs=USE_MULTITERM_MFS,
    )

    firstpass_image = (
        Path(str(firstpass_base) + ".image.tt0")
        if USE_MULTITERM_MFS
        else firstpass_base.with_suffix(".image")
    )
    bmaj, bmin, bpa = read_restoring_beam_arcsec(firstpass_image)

    # 3) final imaging grid
    final_cell, final_imsize, final_fov_arcsec = choose_final_imaging_setup(
        bmaj,
        bmin,
    )

    # 4) final dirty
    dirty_base = output_dir / f"{PRODUCT_PREFIX}_dirty"
    run_tclean(
        ms_for_imaging,
        str(dirty_base),
        cell_arcsec=final_cell,
        imsize=final_imsize,
        niter=0,
        uvrange=applied_uvrange,
        use_multiterm_mfs=USE_MULTITERM_MFS,
        apply_multiterm_clean_controls=False,
    )
    dirty_image_for_mask = (
        Path(str(dirty_base) + ".image.tt0")
        if USE_MULTITERM_MFS
        else dirty_base.with_suffix(".image")
    )
    dirty_bmaj, dirty_bmin, _ = read_restoring_beam_arcsec(dirty_image_for_mask)

    final_clean_usemask = ""
    final_clean_mask = ""
    final_clean_mask_nbeams = None
    if (
        FINAL_CLEAN_BOX_MASK_NBEAMS is not None
        and np.isfinite(FINAL_CLEAN_BOX_MASK_NBEAMS)
        and FINAL_CLEAN_BOX_MASK_NBEAMS > 0
    ):
        final_clean_usemask = "user"
        final_clean_mask_nbeams = float(FINAL_CLEAN_BOX_MASK_NBEAMS)
        final_clean_mask = beam_box_mask_string(
            imsize=final_imsize,
            cell_arcsec=final_cell,
            bmaj_arcsec=dirty_bmaj,
            bmin_arcsec=dirty_bmin,
            nbeams=final_clean_mask_nbeams,
        )
        print(
            f"[MASK] {folder_name} | mode=beam_box | "
            f"nbeams={format_plot_metric(final_clean_mask_nbeams)} | "
            f"beam=({format_plot_metric(dirty_bmaj)}\", {format_plot_metric(dirty_bmin)}\") | "
            f"mask={final_clean_mask}"
        )
    mask_plot_label = (
        f"mask={format_plot_metric(final_clean_mask_nbeams)} beams"
        if final_clean_mask_nbeams is not None
        else "mask=none"
    )

    # 5) final clean
    clean_base = output_dir / f"{PRODUCT_PREFIX}_clean"
    run_tclean(
        ms_for_imaging,
        str(clean_base),
        cell_arcsec=final_cell,
        imsize=final_imsize,
        niter=FINAL_CLEAN_NITER,
        uvrange=applied_uvrange,
        use_multiterm_mfs=USE_MULTITERM_MFS,
        usemask=final_clean_usemask,
        mask=final_clean_mask,
    )

    final_clean_image = (
        Path(str(clean_base) + ".image.tt0")
        if USE_MULTITERM_MFS
        else clean_base.with_suffix(".image")
    )
    final_residual_image = (
        Path(str(clean_base) + ".residual.tt0")
        if USE_MULTITERM_MFS
        else clean_base.with_suffix(".residual")
    )
    final_bmaj, final_bmin, final_bpa = read_restoring_beam_arcsec(final_clean_image)
    qa_metrics = clean_residual_qa_metrics(final_clean_image, final_residual_image)
    selfcal_qa_metrics = load_selfcal_qa_metrics(sample_top, qa_metrics)
    uv_inside_pct = 100.0 * uv_cov["uv_fraction_inside_limits"] if np.isfinite(uv_cov["uv_fraction_inside_limits"]) else np.nan

    clean_title = (
        f"{folder_name} | {source_label} | {band_info['selected_band'] or '?'} | {gain_array_config or '?'} | "
        f"beam={format_plot_metric(final_bmaj)}\"x{format_plot_metric(final_bmin)}\" | "
        f"uv={applied_uvrange or 'all'} | catalog={uv_range or 'none'} | "
        f"in={format_plot_metric(uv_inside_pct)}% | {mask_plot_label}"
    )
    dirty_title = (
        f"{folder_name} dirty | {source_label} | {band_info['selected_band'] or '?'} | {gain_array_config or '?'} | "
        f"beam={format_plot_metric(final_bmaj)}\"x{format_plot_metric(final_bmin)}\" | "
        f"uv={applied_uvrange or 'all'} | catalog={uv_range or 'none'} | "
        f"in={format_plot_metric(uv_inside_pct)}% | mask=none"
    )
    residual_title = (
        f"{folder_name} residual | {mask_plot_label}\n"
        f"sigma=1.4826*MAD(residual)={format_plot_metric(qa_metrics['residual_robust_sigma_jy_per_beam'])} Jy/bm\n"
        f"max=max(|residual|)/sigma={format_plot_metric(qa_metrics['residual_peak_to_sigma'])}\n"
        f"p99=P99(|residual|)/sigma={format_plot_metric(qa_metrics['residual_p99_abs_over_sigma'])}\n"
        f"p995=P99.5(|residual|)/sigma={format_plot_metric(qa_metrics['residual_p995_abs_over_sigma'])}\n"
        f"DR=max(|clean|)/sigma={format_plot_metric(qa_metrics['dynamic_range'])}"
    )

    # 6) export pngs
    image_to_png_if_exists(
        Path(str(dirty_base) + ".image.tt0") if USE_MULTITERM_MFS else dirty_base.with_suffix(".image"),
        output_dir / f"{PRODUCT_PREFIX}_dirty.png",
        title=" ",
        draw_beam_ellipse=True,
    )
    image_to_png_if_exists(
        final_clean_image,
        output_dir / f"{PRODUCT_PREFIX}_clean.png",
        title=" ",
        draw_beam_ellipse=True,
    )
    image_to_png_if_exists(
        final_residual_image,
        output_dir / f"{PRODUCT_PREFIX}_residual.png",
        title=" ",
        symmetric=True,
        cmap="inferno",
    )
    run_optional_export(
        "uv coverage",
        export_uv_png,
        ms_for_imaging,
        output_dir / f"{PRODUCT_PREFIX}_uv.png",
        spw="",
        uvrange=applied_uvrange,
    )
    run_optional_export(
        "amp vs uv-dist",
        export_amp_vs_uvdist_png,
        ms_for_imaging,
        output_dir / f"{PRODUCT_PREFIX}_amp_vs_uvdist.png",
        spw="",
        uvrange=applied_uvrange,
    )
    run_optional_export(
        "amp / median(A)",
        export_normalized_amp_vs_uvdist_png,
        ms_for_imaging,
        output_dir / f"{PRODUCT_PREFIX}_amp_vs_uvdist_norm.png",
        spw="",
        uvrange=applied_uvrange,
    )
    run_optional_export(
        "spectrum by antenna",
        export_spectrum_png,
        ms_for_imaging,
        output_dir / f"{PRODUCT_PREFIX}_spectrum_by_ant.png",
        spw="",
        uvrange=applied_uvrange,
        avgbaseline=False,
    )
    run_optional_export(
        "spectrum",
        export_spectrum_png,
        ms_for_imaging,
        output_dir / f"{PRODUCT_PREFIX}_spectrum.png",
        spw="",
        uvrange=applied_uvrange,
        avgbaseline=True,
    )

    result = {
        "folder": folder_name,
        "ms": str(ms_path),
        "image_output_dir": str(output_dir),
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
        "final_clean_mask_mode": "beam_box" if final_clean_mask_nbeams is not None else "none",
        "final_clean_box_mask_nbeams": final_clean_mask_nbeams if final_clean_mask_nbeams is not None else np.nan,
        "final_clean_usemask": final_clean_usemask,
        "final_clean_mask": final_clean_mask,
        "product_prefix": PRODUCT_PREFIX,
        "catalog_uv_receiver": None if uv_limit_info is None else uv_limit_info["receiver"],
        "catalog_uvmin_kl": np.nan if uv_limit_info is None else uv_limit_info["uvmin_kl"],
        "catalog_uvmax_kl": np.nan if uv_limit_info is None else uv_limit_info["uvmax_kl"],
        "catalog_uvrange": uv_range,
        "applied_uvrange": applied_uvrange,
        "uvlimit_filtering_enabled": APPLY_CATALOG_UVLIMIT_FILTERING,
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
        **qa_metrics,
        **selfcal_qa_metrics,
    }

    print(
        f"[DONE] {folder_name} | "
        f"beam=({final_bmaj:.3f}\", {final_bmin:.3f}\", {final_bpa:.1f} deg) | "
        f"final_grid={'arbitrary' if USE_ARBITRARY_FINAL_GRID else 'beam_based'} | "
        f"uvrange={applied_uvrange or 'all'} | "
        f"catalog_uvrange={uv_range or 'none'} | "
        f"inside={uv_inside_pct if np.isfinite(uv_inside_pct) else float('nan'):.1f}% | "
        f"cell={final_cell:.4f}\" | imsize={final_imsize} | "
        f"FoV={final_fov_arcsec:.2f}\" | "
        f"FoV/beams={final_fov_arcsec / final_bmin:.2f}"
    )

    return result


def sort_summary_by_residual_p995_abs_over_sigma(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "residual_p995_abs_over_sigma" not in df.columns:
        return df

    sort_key = pd.to_numeric(df["residual_p995_abs_over_sigma"], errors="coerce")
    df = df.assign(_residual_p995_abs_over_sigma_sort=sort_key)
    df = df.sort_values(
        ["_residual_p995_abs_over_sigma_sort", "folder"],
        ascending=[True, True],
        na_position="last",
    )
    return df.drop(columns=["_residual_p995_abs_over_sigma_sort"])


def main():
    extracted_dir = resolve_extracted_dir(SELECTED_FOLDERS)
    df = load_projects(PROJECT_LIST)
    calib_df = load_calibrator_uv_limits(CALIBRATOR_BANDS_CSV)
    ms_paths = find_extracted_ms_paths(extracted_dir)
    ms_paths = filter_ms_paths(ms_paths, SELECTED_FOLDERS)

    print(f"[INFO] using extracted dir: {extracted_dir}")
    print(f"[INFO] found {len(ms_paths)} extracted MS files")
    print(f"[INFO] APPLY_CATALOG_UVLIMIT_FILTERING={APPLY_CATALOG_UVLIMIT_FILTERING}")

    rows = []
    for original_ms_path in ms_paths:
        sample_top = sample_top_for_ms(original_ms_path)
        folder = sample_top.name
        ms_path = original_ms_path
        row = row_for_folder(df, folder)
        try:
            out = process_one_sample(ms_path, row=row, calib_df=calib_df)
            out["original_ms"] = str(original_ms_path)

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
                    "folder": folder,
                    "ms": str(ms_path),
                    "original_ms": str(original_ms_path),
                    "image_output_dir": str(image_output_dir_for_sample(sample_top)),
                    "error": f"{type(e).__name__}: {e}",
                }
            )

    summary_csv = extracted_dir / "beam_imaging_summary.csv"
    sort_summary_by_residual_p995_abs_over_sigma(rows).to_csv(summary_csv, index=False)
    print(f"[OK] wrote summary to {summary_csv}")


# if __name__ == "__main__":
#     main()
