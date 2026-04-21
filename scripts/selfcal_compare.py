from __future__ import annotations

import csv
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from casatasks import imhead, immath, tclean
from casatools import image, table

from scripts.sample_groups import UV_LIM
from scripts.vla_config import (
    band_for_frequency_ghz,
    band_matches_frequency,
    estimate_synthesized_beam_arcsec,
    split_band_codes,
)


PROJECT_LIST = Path("/Users/u1528314/repos/radioastro-ml/collect/small_subset/small_selection.csv")
CALIBRATOR_BANDS_CSV = Path("/Users/u1528314/repos/radioastro-ml/collect/vla_calibrators_bands_v2.csv")
DEFAULT_EXTRACTED_DIR = Path("/Users/u1528314/repos/radioastro-ml/collect/extracted")

SELECTED_FOLDERS: list[str] | None = None
SELFCAL_DIRNAME = "selfcal"
SELFCAL_MS_SUFFIX = "_selfcal.ms"
COMPARE_DIRNAME = "compare_original_selfcal"
SUMMARY_CSV_NAME = "selfcal_compare_uvlim_summary.csv"
CONTACT_SHEET_NAME = "selfcal_compare_uvlim.png"
PER_SAMPLE_METRICS_CSV_NAME = "selfcal_improvement_metrics.csv"

# Use the uv-limit only to estimate the beam/cell, matching the old useful
# uvlim_recal behavior. The final comparison images use FINAL_UVRANGE.
USE_CATALOG_UVRANGE_FOR_FIRSTPASS = True
FINAL_UVRANGE = ""

ORIGINAL_DATACOLUMN = "auto"  # auto = CORRECTED_DATA if present, else DATA
SELFCAL_DATACOLUMN = "auto"   # auto = CORRECTED_DATA if present, else DATA

PIXELS_PER_BEAM = 4.0
FOV_IN_BEAMS = 64.0
MIN_IMSIZE = 128
MAX_IMSIZE = 1024
MIN_CELL_ARCSEC = 0.02
MAX_CELL_ARCSEC = 50.0
FIRSTPASS_NITER = 0
FINAL_NITER = 100

TCLEAN_BASE = dict(
    specmode="mfs",
    weighting="briggs",
    robust=0.5,
    stokes="I",
    deconvolver="hogbom",
    gridder="standard",
    interactive=False,
    savemodel="none",
)

SHARED_SCALE_PERCENTILE = 99.5
DIFF_SCALE_PERCENTILE = 99.5
SOURCE_EXCLUDE_RADIUS_PIX = 20
EDGE_EXCLUDE_PIX = 4


def parse_float(value) -> float:
    try:
        text = str(value).strip()
        if text == "" or text.lower() == "nan":
            return float("nan")
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def resolve_extracted_dir(
    selected_folders: list[str] | None,
    default_dir: Path = DEFAULT_EXTRACTED_DIR,
) -> Path:
    candidate_dirs = [
        Path("/Users/u1528314/repos/radioastro-ml/collect/extracted"),
        Path("/Users/u1528314/repos/radioastro-ml/collect/extracted2"),
        Path("/Users/u1528314/repos/radioastro-ml/collect/extracted3"),
    ]
    if not selected_folders:
        return default_dir

    wanted = [str(x).strip() for x in selected_folders if str(x).strip()]
    scored: list[tuple[int, int, Path]] = []
    for idx, candidate in enumerate(candidate_dirs):
        hits = sum((candidate / folder).exists() for folder in wanted)
        scored.append((hits, -idx, candidate))

    best_hits, _, best_dir = max(scored)
    if best_hits == 0:
        print(f"[WARN] no selected folders found; using {default_dir}")
        return default_dir
    if best_dir != default_dir:
        print(
            f"[WARN] selected folders were not found under {default_dir}; "
            f"using {best_dir} instead"
        )
    return best_dir


def is_inside_ignored_subdir(path: Path, sample_dir: Path) -> bool:
    try:
        parts = path.relative_to(sample_dir).parts
    except ValueError:
        return False
    ignored = {SELFCAL_DIRNAME, "selfcal_test", "diagnostics"}
    return any(part in ignored for part in parts)


def find_original_ms_paths(extracted_dir: Path) -> list[Path]:
    hits: list[Path] = []
    for sample_dir in sorted(extracted_dir.iterdir()):
        if not sample_dir.is_dir():
            continue

        folder = sample_dir.name
        expected = sample_dir / folder / f"{folder}.ms"
        if expected.exists():
            hits.append(expected)
            continue

        candidates = sorted(
            p for p in sample_dir.rglob("*.ms")
            if not is_inside_ignored_subdir(p, sample_dir)
            and not p.name.endswith("_imgprep.ms")
        )
        if candidates:
            hits.append(candidates[0])
    return hits


def sample_top_for_ms(ms_path: Path) -> Path:
    return ms_path.parent.parent


def selfcal_ms_for_sample(sample_top: Path) -> Path:
    return sample_top / SELFCAL_DIRNAME / f"{sample_top.name}{SELFCAL_MS_SUFFIX}"


def filter_ms_paths(ms_paths: list[Path], selected_folders: list[str] | None) -> list[Path]:
    if not selected_folders:
        return ms_paths

    wanted = [str(x).strip() for x in selected_folders if str(x).strip()]
    wanted_set = set(wanted)
    filtered = [p for p in ms_paths if sample_top_for_ms(p).name in wanted_set]
    found = {sample_top_for_ms(p).name for p in filtered}
    missing = [folder for folder in wanted if folder not in found]
    if missing:
        print(f"[WARN] requested folders not found in extracted MS paths: {missing}")

    order = {folder: i for i, folder in enumerate(wanted)}
    filtered.sort(key=lambda p: order.get(sample_top_for_ms(p).name, 10**9))
    return filtered


def row_for_folder(rows: list[dict], folder: str) -> dict | None:
    for row in rows:
        if str(row.get("folder", "")).strip() == folder:
            return row
    return None


def ms_columns(ms_path: Path) -> list[str]:
    tb = table()
    tb.open(str(ms_path))
    try:
        return list(tb.colnames())
    finally:
        tb.close()


def choose_datacolumn(ms_path: Path, setting: str) -> str:
    cols = set(ms_columns(ms_path))
    setting = str(setting).strip().lower()

    if setting in {"data", "corrected"}:
        required = "DATA" if setting == "data" else "CORRECTED_DATA"
        if required not in cols:
            raise RuntimeError(f"{ms_path} does not have {required}")
        return setting

    if "CORRECTED_DATA" in cols:
        return "corrected"
    if "DATA" in cols:
        return "data"
    raise RuntimeError(f"{ms_path} has neither DATA nor CORRECTED_DATA")


def _read_spw_frequency_info(ms_path: Path) -> list[dict]:
    tb = table()
    tb.open(str(ms_path / "SPECTRAL_WINDOW"))
    try:
        info = []
        for spw in range(tb.nrows()):
            freqs = np.array(tb.getcell("CHAN_FREQ", spw), dtype=float)
            if freqs.size == 0:
                continue
            info.append(
                {
                    "spw": spw,
                    "f_center_ghz": float(np.median(freqs) / 1e9),
                    "bandwidth_hz": float(np.max(freqs) - np.min(freqs)),
                }
            )
        return info
    finally:
        tb.close()


def get_ms_reference_frequency_ghz(ms_path: Path, preferred_spw: int | None = None) -> tuple[float | None, int | None]:
    info = _read_spw_frequency_info(ms_path)
    if not info:
        return None, None

    if preferred_spw is not None:
        for row in info:
            if row["spw"] == preferred_spw:
                return row["f_center_ghz"], row["spw"]

    best = max(info, key=lambda row: row["bandwidth_hz"])
    return best["f_center_ghz"], best["spw"]


def choose_band_and_frequency(ms_path: Path, project_row: dict | None) -> dict:
    row_band_codes = split_band_codes(None if project_row is None else project_row.get("band_code"))
    band_guess_codes = split_band_codes(None if project_row is None else project_row.get("band_guess"))

    preferred_spw = None
    if project_row is not None:
        spw = parse_float(project_row.get("spw_selected"))
        if np.isfinite(spw):
            preferred_spw = int(spw)

    ms_freq_ghz, used_spw = get_ms_reference_frequency_ghz(ms_path, preferred_spw=preferred_spw)
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

    return {
        "selected_band": selected_band,
        "detected_band": detected_band,
        "row_band_codes": row_band_codes,
        "ms_freq_ghz": ms_freq_ghz,
        "used_spw": used_spw,
    }


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


def lookup_uv_limits(calib_rows: list[dict], *, calibrator_name: str, band: str | None) -> dict | None:
    receiver = band_to_receiver_code(band)
    if receiver is None:
        return None

    target_name = calibrator_name.strip().upper()
    target_receiver = receiver.strip().upper()
    for row in calib_rows:
        if str(row.get("name", "")).strip().upper() != target_name:
            continue
        if str(row.get("receiver", "")).strip().upper() != target_receiver:
            continue

        uvmin = parse_float(row.get("uvmin_kl"))
        uvmax = parse_float(row.get("uvmax_kl"))
        if np.isfinite(uvmin) and np.isfinite(uvmax) and uvmin > uvmax:
            uvmin, uvmax = uvmax, uvmin
        return {
            "calibrator_name": str(row.get("name", calibrator_name)),
            "receiver": str(row.get("receiver", receiver)),
            "uvmin_kl": uvmin,
            "uvmax_kl": uvmax,
        }

    return None


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


def choose_firstpass_imaging_setup(project_row: dict | None, band_info: dict) -> tuple[float, int, float | None]:
    config = None if project_row is None else project_row.get("gain_array_config")
    freq = band_info.get("ms_freq_ghz")
    estimated_beam = estimate_synthesized_beam_arcsec(config, freq)
    if estimated_beam is None or not np.isfinite(estimated_beam) or estimated_beam <= 0:
        return 0.5, 256, None

    cell = max(MIN_CELL_ARCSEC, min(MAX_CELL_ARCSEC, estimated_beam / PIXELS_PER_BEAM))
    imsize = choose_imsize(estimated_beam, cell)
    return cell, imsize, estimated_beam


def choose_imsize(bmin_arcsec: float, cell_arcsec: float) -> int:
    target_fov = FOV_IN_BEAMS * bmin_arcsec
    imsize = int(math.ceil(target_fov / cell_arcsec))
    if imsize % 2:
        imsize += 1
    return max(MIN_IMSIZE, min(MAX_IMSIZE, imsize))


def beam_value_to_arcsec(value) -> float:
    if isinstance(value, dict):
        number = float(value["value"])
        unit = str(value.get("unit", "")).strip().lower()
        if unit in {"arcsec", "arcseconds", "asec"}:
            return number
        if unit in {"arcmin", "arcminutes"}:
            return number * 60.0
        if unit in {"deg", "degree", "degrees"}:
            return number * 3600.0
        raise ValueError(f"Unsupported beam unit: {unit!r}")
    return float(value)


def read_restoring_beam_arcsec(image_path: Path) -> tuple[float, float, float]:
    info = imhead(imagename=str(image_path), mode="summary")
    beam = info.get("restoringbeam")
    if beam is None:
        raise RuntimeError(f"No restoring beam in {image_path}")

    bmaj = beam_value_to_arcsec(beam["major"])
    bmin = beam_value_to_arcsec(beam["minor"])
    pa = beam.get("positionangle", {})
    if isinstance(pa, dict):
        pa_deg = float(pa.get("value", np.nan))
        if str(pa.get("unit", "deg")).strip().lower().startswith("rad"):
            pa_deg = float(np.rad2deg(pa_deg))
    else:
        pa_deg = float(pa)
    return bmaj, bmin, pa_deg


def choose_final_grid_from_firstpass(firstpass_image: Path) -> dict:
    bmaj, bmin, bpa = read_restoring_beam_arcsec(firstpass_image)
    cell = max(MIN_CELL_ARCSEC, min(MAX_CELL_ARCSEC, min(bmaj, bmin) / PIXELS_PER_BEAM))
    imsize = choose_imsize(bmin, cell)
    return {
        "firstpass_beam_major_arcsec": bmaj,
        "firstpass_beam_minor_arcsec": bmin,
        "firstpass_beam_pa_deg": bpa,
        "cell_arcsec": cell,
        "imsize": imsize,
        "fov_arcsec": imsize * cell,
    }


def remove_products(imagename: Path) -> None:
    parent = imagename.parent
    prefix = imagename.name
    if not parent.exists():
        return

    for path in parent.glob(prefix + ".*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                path.unlink()
            except OSError:
                pass


def run_tclean(
    ms_path: Path,
    imagename: Path,
    *,
    datacolumn: str,
    cell_arcsec: float,
    imsize: int,
    niter: int,
    uvrange: str,
) -> None:
    remove_products(imagename)
    cfg = dict(TCLEAN_BASE)
    cfg.update(
        vis=str(ms_path),
        imagename=str(imagename),
        datacolumn=datacolumn,
        cell=f"{cell_arcsec:.6f}arcsec",
        imsize=int(imsize),
        niter=int(niter),
        uvrange=uvrange,
    )
    print(
        f"[TCLEAN] {imagename.name}: ms={ms_path.name} col={datacolumn} "
        f"cell={cell_arcsec:.6f}\" imsize={imsize} niter={niter} uv={uvrange or 'all'}"
    )
    tclean(**cfg)


def load_image_2d(image_path: Path) -> np.ndarray:
    ia = image()
    ia.open(str(image_path))
    try:
        arr = np.asarray(ia.getchunk())
    finally:
        ia.close()

    arr = np.squeeze(arr)
    if arr.ndim == 4:
        arr2d = arr[:, :, 0, 0]
    elif arr.ndim == 3:
        arr2d = arr[:, :, 0]
    elif arr.ndim == 2:
        arr2d = arr
    else:
        raise RuntimeError(f"Unexpected image ndim={arr.ndim} for {image_path}")

    return np.asarray(arr2d, dtype=float).T


def robust_sigma(values: np.ndarray) -> float:
    x = values[np.isfinite(values)]
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return 1.4826 * mad


def offsource_mask(img: np.ndarray, radius_pix: int = SOURCE_EXCLUDE_RADIUS_PIX) -> np.ndarray:
    finite = np.isfinite(img)
    if not np.any(finite):
        return finite

    peak_y, peak_x = np.unravel_index(np.nanargmax(np.abs(img)), img.shape)
    yy, xx = np.indices(img.shape)
    rr = np.sqrt((yy - peak_y) ** 2 + (xx - peak_x) ** 2)
    mask = finite & (rr >= radius_pix)

    if EDGE_EXCLUDE_PIX > 0:
        mask[:EDGE_EXCLUDE_PIX, :] = False
        mask[-EDGE_EXCLUDE_PIX:, :] = False
        mask[:, :EDGE_EXCLUDE_PIX] = False
        mask[:, -EDGE_EXCLUDE_PIX:] = False

    return mask


def image_metrics(image_path: Path) -> dict:
    img = load_image_2d(image_path)
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        raise RuntimeError(f"No finite pixels in {image_path}")

    off = img[offsource_mask(img)]
    peak = float(np.nanmax(img))
    peak_abs = float(np.nanmax(np.abs(img)))
    rms_all = float(np.sqrt(np.nanmean(finite ** 2)))
    rms_off = float(np.sqrt(np.nanmean(off ** 2))) if off.size else float("nan")
    sig_off = robust_sigma(off)
    p99_abs = float(np.nanpercentile(np.abs(finite), 99))
    p995_abs = float(np.nanpercentile(np.abs(finite), 99.5))

    return {
        "peak_jy_per_beam": peak,
        "peak_abs_jy_per_beam": peak_abs,
        "min_jy_per_beam": float(np.nanmin(img)),
        "rms_all_jy_per_beam": rms_all,
        "rms_offsource_jy_per_beam": rms_off,
        "robust_sigma_offsource_jy_per_beam": sig_off,
        "dynamic_range_rms_offsource": peak_abs / rms_off if np.isfinite(rms_off) and rms_off > 0 else float("nan"),
        "dynamic_range_robust_offsource": peak_abs / sig_off if np.isfinite(sig_off) and sig_off > 0 else float("nan"),
        "p99_abs_jy_per_beam": p99_abs,
        "p995_abs_jy_per_beam": p995_abs,
    }


def residual_metrics(clean_image_path: Path, residual_image_path: Path) -> dict:
    clean = load_image_2d(clean_image_path)
    residual = load_image_2d(residual_image_path)

    clean_vals = clean[np.isfinite(clean)]
    residual_vals = residual[np.isfinite(residual)]
    if clean_vals.size == 0:
        raise RuntimeError(f"No finite pixels in {clean_image_path}")
    if residual_vals.size == 0:
        raise RuntimeError(f"No finite pixels in {residual_image_path}")

    residual_rms_all = float(np.sqrt(np.mean(residual_vals ** 2)))
    residual_robust_sigma = robust_sigma(residual_vals)
    residual_max_abs = float(np.max(np.abs(residual_vals)))
    residual_p99_abs = float(np.percentile(np.abs(residual_vals), 99.0))
    residual_p995_abs = float(np.percentile(np.abs(residual_vals), 99.5))
    clean_max_abs = float(np.max(np.abs(clean_vals)))

    return {
        "residual_rms_all_jy_per_beam": residual_rms_all,
        "residual_robust_sigma_jy_per_beam": residual_robust_sigma,
        "residual_max_abs_jy_per_beam": residual_max_abs,
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
        "residual_max_abs_over_robust_sigma": (
            residual_max_abs / residual_robust_sigma
            if np.isfinite(residual_robust_sigma) and residual_robust_sigma > 0
            else float("nan")
        ),
        "dynamic_range": (
            clean_max_abs / residual_robust_sigma
            if np.isfinite(residual_robust_sigma) and residual_robust_sigma > 0
            else float("nan")
        ),
        "residual_dynamic_range": (
            clean_max_abs / residual_robust_sigma
            if np.isfinite(residual_robust_sigma) and residual_robust_sigma > 0
            else float("nan")
        ),
    }


def visibility_delta_metrics(original_ms: Path, selfcal_ms: Path, original_col: str, selfcal_col: str) -> dict:
    def read_col(ms_path: Path, col: str):
        tb = table()
        tb.open(str(ms_path))
        try:
            casa_col = "CORRECTED_DATA" if col == "corrected" else "DATA"
            if casa_col not in tb.colnames():
                return None
            return np.asarray(tb.getcol(casa_col))
        finally:
            tb.close()

    try:
        a = read_col(original_ms, original_col)
        b = read_col(selfcal_ms, selfcal_col)
        if a is None or b is None or a.shape != b.shape:
            return {
                "vis_compare_status": "shape_mismatch_or_missing",
                "vis_shape_original": "" if a is None else str(a.shape),
                "vis_shape_selfcal": "" if b is None else str(b.shape),
            }

        delta = b - a
        mean_abs_a = float(np.nanmean(np.abs(a)))
        mean_abs_b = float(np.nanmean(np.abs(b)))
        mean_abs_delta = float(np.nanmean(np.abs(delta)))
        max_abs_delta = float(np.nanmax(np.abs(delta)))
        return {
            "vis_compare_status": "ok",
            "vis_mean_abs_original": mean_abs_a,
            "vis_mean_abs_selfcal": mean_abs_b,
            "vis_mean_abs_delta": mean_abs_delta,
            "vis_max_abs_delta": max_abs_delta,
            "vis_mean_abs_delta_frac": mean_abs_delta / mean_abs_a if mean_abs_a > 0 else float("nan"),
        }
    except Exception as exc:
        return {
            "vis_compare_status": f"error: {type(exc).__name__}: {exc}",
        }


def write_comparison_png(
    original_image: Path,
    selfcal_image: Path,
    diff_image: Path,
    out_png: Path,
    *,
    folder: str,
    row: dict,
) -> None:
    selfcal = load_image_2d(selfcal_image) * 1e3

    vals = selfcal[np.isfinite(selfcal)].ravel()
    vlim = float(np.nanpercentile(np.abs(vals), SHARED_SCALE_PERCENTILE)) if vals.size else 1.0
    if not np.isfinite(vlim) or vlim <= 0:
        vlim = 1.0

    fig, ax = plt.subplots(1, 1, figsize=(4.8, 4.8))
    im = ax.imshow(selfcal, origin="lower", vmin=-vlim, vmax=vlim, cmap="inferno")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("mJy/beam")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def ratio(after: float, before: float) -> float:
    if not np.isfinite(after) or not np.isfinite(before):
        return float("nan")
    if before == 0:
        return float("inf") if after > 0 else 1.0
    return float(after / before)


def process_one(
    original_ms: Path,
    *,
    project_rows: list[dict],
    calibrator_rows: list[dict],
) -> dict:
    sample_top = sample_top_for_ms(original_ms)
    folder = sample_top.name
    selfcal_ms = selfcal_ms_for_sample(sample_top)
    compare_dir = sample_top / SELFCAL_DIRNAME / COMPARE_DIRNAME
    compare_dir.mkdir(parents=True, exist_ok=True)

    if not selfcal_ms.exists():
        raise FileNotFoundError(f"selfcal MS not found: {selfcal_ms}")

    project_row = row_for_folder(project_rows, folder)
    band_info = choose_band_and_frequency(original_ms, project_row)
    uv_limit = lookup_uv_limits(
        calibrator_rows,
        calibrator_name=folder,
        band=band_info["selected_band"],
    )
    firstpass_uvrange = ""
    if USE_CATALOG_UVRANGE_FOR_FIRSTPASS and uv_limit is not None:
        firstpass_uvrange = make_uvrange_string(uv_limit["uvmin_kl"], uv_limit["uvmax_kl"])

    original_col = choose_datacolumn(original_ms, ORIGINAL_DATACOLUMN)
    selfcal_col = choose_datacolumn(selfcal_ms, SELFCAL_DATACOLUMN)

    firstpass_cell, firstpass_imsize, estimated_beam = choose_firstpass_imaging_setup(project_row, band_info)
    firstpass_base = compare_dir / "beam_firstpass_original"
    run_tclean(
        original_ms,
        firstpass_base,
        datacolumn=original_col,
        cell_arcsec=firstpass_cell,
        imsize=firstpass_imsize,
        niter=FIRSTPASS_NITER,
        uvrange=firstpass_uvrange,
    )

    grid = choose_final_grid_from_firstpass(firstpass_base.with_suffix(".image"))
    original_base = compare_dir / "original_clean"
    selfcal_base = compare_dir / "selfcal_clean"
    diff_base = compare_dir / "selfcal_minus_original"

    run_tclean(
        original_ms,
        original_base,
        datacolumn=original_col,
        cell_arcsec=grid["cell_arcsec"],
        imsize=grid["imsize"],
        niter=FINAL_NITER,
        uvrange=FINAL_UVRANGE,
    )
    run_tclean(
        selfcal_ms,
        selfcal_base,
        datacolumn=selfcal_col,
        cell_arcsec=grid["cell_arcsec"],
        imsize=grid["imsize"],
        niter=FINAL_NITER,
        uvrange=FINAL_UVRANGE,
    )

    remove_products(diff_base)
    immath(
        imagename=[str(selfcal_base.with_suffix(".image")), str(original_base.with_suffix(".image"))],
        expr="IM0 - IM1",
        outfile=str(diff_base.with_suffix(".image")),
    )

    original_clean_image = original_base.with_suffix(".image")
    selfcal_clean_image = selfcal_base.with_suffix(".image")
    original_residual_image = original_base.with_suffix(".residual")
    selfcal_residual_image = selfcal_base.with_suffix(".residual")
    diff_image = diff_base.with_suffix(".image")

    original_metrics = image_metrics(original_clean_image)
    selfcal_metrics = image_metrics(selfcal_clean_image)
    original_residual_metrics = residual_metrics(original_clean_image, original_residual_image)
    selfcal_residual_metrics = residual_metrics(selfcal_clean_image, selfcal_residual_image)
    diff_metrics = image_metrics(diff_image)
    vis_metrics = visibility_delta_metrics(original_ms, selfcal_ms, original_col, selfcal_col)

    row = {
        "folder": folder,
        "status": "ok",
        "original_ms": str(original_ms),
        "selfcal_ms": str(selfcal_ms),
        "compare_dir": str(compare_dir),
        "original_datacolumn": original_col,
        "selfcal_datacolumn": selfcal_col,
        "selected_band": band_info["selected_band"],
        "detected_band": band_info["detected_band"],
        "ms_freq_ghz": band_info["ms_freq_ghz"],
        "used_spw": band_info["used_spw"],
        "gain_array_config": "" if project_row is None else project_row.get("gain_array_config", ""),
        "catalog_uvmin_kl": float("nan") if uv_limit is None else uv_limit["uvmin_kl"],
        "catalog_uvmax_kl": float("nan") if uv_limit is None else uv_limit["uvmax_kl"],
        "firstpass_uvrange": firstpass_uvrange,
        "final_uvrange": FINAL_UVRANGE,
        "firstpass_estimated_beam_arcsec": estimated_beam if estimated_beam is not None else float("nan"),
        **grid,
        "original_image": str(original_clean_image),
        "selfcal_image": str(selfcal_clean_image),
        "original_residual": str(original_residual_image),
        "selfcal_residual": str(selfcal_residual_image),
        "diff_image": str(diff_image),
    }

    for key, value in original_metrics.items():
        row[f"original_{key}"] = value
    for key, value in selfcal_metrics.items():
        row[f"selfcal_{key}"] = value
    for key, value in original_residual_metrics.items():
        row[f"original_{key}"] = value
    for key, value in selfcal_residual_metrics.items():
        row[f"selfcal_{key}"] = value
    for key, value in diff_metrics.items():
        row[f"diff_{key}"] = value

    row.update(vis_metrics)
    row["peak_abs_ratio_selfcal_over_original"] = ratio(
        row["selfcal_peak_abs_jy_per_beam"], row["original_peak_abs_jy_per_beam"]
    )
    row["residual_rms_all_ratio_selfcal_over_original"] = ratio(
        row["selfcal_residual_rms_all_jy_per_beam"],
        row["original_residual_rms_all_jy_per_beam"],
    )
    row["residual_robust_sigma_ratio_selfcal_over_original"] = ratio(
        row["selfcal_residual_robust_sigma_jy_per_beam"],
        row["original_residual_robust_sigma_jy_per_beam"],
    )
    row["residual_max_abs_over_robust_sigma_ratio_selfcal_over_original"] = ratio(
        row["selfcal_residual_max_abs_over_robust_sigma"],
        row["original_residual_max_abs_over_robust_sigma"],
    )
    row["residual_peak_to_sigma_ratio_selfcal_over_original"] = ratio(
        row["selfcal_residual_peak_to_sigma"],
        row["original_residual_peak_to_sigma"],
    )
    row["residual_p99_abs_over_sigma_ratio_selfcal_over_original"] = ratio(
        row["selfcal_residual_p99_abs_over_sigma"],
        row["original_residual_p99_abs_over_sigma"],
    )
    row["residual_p995_abs_over_sigma_ratio_selfcal_over_original"] = ratio(
        row["selfcal_residual_p995_abs_over_sigma"],
        row["original_residual_p995_abs_over_sigma"],
    )
    row["residual_dynamic_range_ratio_selfcal_over_original"] = ratio(
        row["selfcal_residual_dynamic_range"],
        row["original_residual_dynamic_range"],
    )
    # Back-compatible summary aliases now refer to residual-image metrics.
    row["rms_offsource_ratio_selfcal_over_original"] = row[
        "residual_rms_all_ratio_selfcal_over_original"
    ]
    row["robust_sigma_ratio_selfcal_over_original"] = row[
        "residual_robust_sigma_ratio_selfcal_over_original"
    ]
    row["dynamic_range_ratio_selfcal_over_original"] = row[
        "residual_dynamic_range_ratio_selfcal_over_original"
    ]

    comparison_png = compare_dir / f"{folder}_selfcal_final.png"
    write_comparison_png(
        original_clean_image,
        selfcal_clean_image,
        diff_image,
        comparison_png,
        folder=folder,
        row=row,
    )
    row["comparison_png"] = str(comparison_png)
    row["interpretation"] = interpret_row(row)
    row["selfcal_metrics_csv"] = str(write_sample_metrics(row))

    print(
        f"[RESULT] {folder}: residual robust ratio={row['residual_robust_sigma_ratio_selfcal_over_original']:.3g} "
        f"residual peak/sigma ratio={row['residual_peak_to_sigma_ratio_selfcal_over_original']:.3g} "
        f"DR ratio={row['dynamic_range_ratio_selfcal_over_original']:.3g} "
        f"vis delta frac={row.get('vis_mean_abs_delta_frac', float('nan'))} "
        f"({row['interpretation']})"
    )
    return row


def write_sample_metrics(row: dict) -> Path:
    metrics_csv = Path(row["selfcal_ms"]).parent / PER_SAMPLE_METRICS_CSV_NAME
    metric_keys = [
        "folder",
        "status",
        "interpretation",
        "Interpretation",
        "original_ms",
        "selfcal_ms",
        "comparison_png",
        "original_datacolumn",
        "selfcal_datacolumn",
        "selected_band",
        "ms_freq_ghz",
        "firstpass_uvrange",
        "final_uvrange",
        "cell_arcsec",
        "imsize",
        "original_peak_abs_jy_per_beam",
        "selfcal_peak_abs_jy_per_beam",
        "peak_abs_ratio_selfcal_over_original",
        "original_residual_rms_all_jy_per_beam",
        "selfcal_residual_rms_all_jy_per_beam",
        "residual_rms_all_ratio_selfcal_over_original",
        "original_residual_robust_sigma_jy_per_beam",
        "selfcal_residual_robust_sigma_jy_per_beam",
        "residual_robust_sigma_ratio_selfcal_over_original",
        "original_residual_max_abs_jy_per_beam",
        "selfcal_residual_max_abs_jy_per_beam",
        "original_residual_p995_abs_over_sigma",
        "selfcal_residual_p995_abs_over_sigma",
        "residual_p995_abs_over_sigma_ratio_selfcal_over_original",
        "original_residual_p99_abs_over_sigma",
        "selfcal_residual_p99_abs_over_sigma",
        "residual_p99_abs_over_sigma_ratio_selfcal_over_original",
        "original_residual_peak_to_sigma",
        "selfcal_residual_peak_to_sigma",
        "residual_peak_to_sigma_ratio_selfcal_over_original",
        "original_residual_max_abs_over_robust_sigma",
        "selfcal_residual_max_abs_over_robust_sigma",
        "residual_max_abs_over_robust_sigma_ratio_selfcal_over_original",
        "original_dynamic_range",
        "selfcal_dynamic_range",
        "dynamic_range_ratio_selfcal_over_original",
        "original_residual_dynamic_range",
        "selfcal_residual_dynamic_range",
        "residual_dynamic_range_ratio_selfcal_over_original",
        "original_rms_offsource_jy_per_beam",
        "selfcal_rms_offsource_jy_per_beam",
        "rms_offsource_ratio_selfcal_over_original",
        "original_robust_sigma_offsource_jy_per_beam",
        "selfcal_robust_sigma_offsource_jy_per_beam",
        "robust_sigma_ratio_selfcal_over_original",
        "original_dynamic_range_robust_offsource",
        "selfcal_dynamic_range_robust_offsource",
        "diff_rms_all_jy_per_beam",
        "vis_compare_status",
        "vis_mean_abs_original",
        "vis_mean_abs_selfcal",
        "vis_mean_abs_delta",
        "vis_max_abs_delta",
        "vis_mean_abs_delta_frac",
    ]
    row["Interpretation"] = row.get("interpretation", "")
    out = {key: row.get(key, "") for key in metric_keys}
    with metrics_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=metric_keys)
        writer.writeheader()
        writer.writerow(out)
    print(f"[OK] wrote per-sample selfcal metrics: {metrics_csv}")
    return metrics_csv


def write_summary(rows: list[dict], summary_csv: Path) -> None:
    if not rows:
        return
    def sort_key(row: dict) -> tuple[float, str]:
        p995 = parse_float(row.get("original_residual_p995_abs_over_sigma"))
        if not np.isfinite(p995):
            p995 = float("inf")
        return p995, str(row.get("folder", ""))

    rows = sorted(
        rows,
        key=sort_key,
    )
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with summary_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] wrote summary: {summary_csv}")


def write_contact_sheet(rows: list[dict], out_png: Path) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok" and row.get("comparison_png")]
    if not ok_rows:
        print("[INFO] no successful comparison PNGs for contact sheet")
        return

    n = len(ok_rows)
    fig, axes = plt.subplots(n, 1, figsize=(17, 4.9 * n), squeeze=False)
    for ax, row in zip(axes[:, 0], ok_rows):
        ax.imshow(mpimg.imread(row["comparison_png"]))
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote contact sheet: {out_png}")


def fmt_metric(value, fmt: str = ".3g", fallback: str = "?") -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback
    if not np.isfinite(value):
        return fallback
    return format(value, fmt)


def interpret_row(row: dict) -> str:
    robust_ratio = parse_float(row.get("residual_robust_sigma_ratio_selfcal_over_original"))
    peak_ratio = parse_float(row.get("residual_peak_to_sigma_ratio_selfcal_over_original"))
    if not np.isfinite(peak_ratio):
        peak_ratio = parse_float(row.get("residual_max_abs_over_robust_sigma_ratio_selfcal_over_original"))
    dr_ratio = parse_float(row.get("dynamic_range_ratio_selfcal_over_original"))
    if not np.isfinite(dr_ratio):
        dr_ratio = parse_float(row.get("residual_dynamic_range_ratio_selfcal_over_original"))

    if (
        np.isfinite(robust_ratio)
        and np.isfinite(peak_ratio)
        and np.isfinite(dr_ratio)
        and robust_ratio < 1.0
        and peak_ratio < 1.0
        and dr_ratio > 1.0
    ):
        return "improved"

    if (
        (np.isfinite(robust_ratio) and robust_ratio > 1.0)
        or (np.isfinite(peak_ratio) and peak_ratio > 1.0)
        or (np.isfinite(dr_ratio) and dr_ratio < 1.0)
    ):
        return "worse"

    return "mixed / small change"


def print_interpretation(rows: list[dict]) -> None:
    print("\n[SUMMARY] Selfcal vs original, same imaging grid")
    print("[SUMMARY] Ratios are selfcal/original.")
    print("[SUMMARY] Residual metrics use all finite clean.residual pixels, no center mask.")
    print("[SUMMARY] Good signs: residual robust < 1, residual peak/sigma < 1, DR > 1.")
    print("[SUMMARY] vis_delta_frac near 0 means the selfcal MS is almost unchanged.\n")

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    err_rows = [row for row in rows if row.get("status") != "ok"]

    if ok_rows:
        header = (
            f"{'folder':<12} {'res_rms':>8} {'res_rob':>8} {'max/sig':>8} {'DR':>8} "
            f"{'vis_delta':>10}  interpretation"
        )
        print(header)
        print("-" * len(header))
        for row in ok_rows:
            folder = str(row.get("folder", ""))[:12]
            rms = fmt_metric(row.get("residual_rms_all_ratio_selfcal_over_original"))
            robust = fmt_metric(row.get("residual_robust_sigma_ratio_selfcal_over_original"))
            maxsig = fmt_metric(row.get("residual_peak_to_sigma_ratio_selfcal_over_original"))
            dr = fmt_metric(row.get("dynamic_range_ratio_selfcal_over_original"))
            vis_delta = fmt_metric(row.get("vis_mean_abs_delta_frac"))
            print(
                f"{folder:<12} {rms:>8} {robust:>8} {maxsig:>8} {dr:>8} "
                f"{vis_delta:>10}  {interpret_row(row)}"
            )
    else:
        print("[SUMMARY] No successful comparisons.")

    if err_rows:
        print("\n[SUMMARY] Errors / skipped samples:")
        for row in err_rows:
            print(f"  {row.get('folder', '?')}: {row.get('error', 'unknown error')}")


def main(
    extracted_dir: str | Path | None = None,
    *,
    selected_folders: list[str] | None = SELECTED_FOLDERS,
) -> list[dict]:
    extracted_dir = (
        resolve_extracted_dir(selected_folders, default_dir=DEFAULT_EXTRACTED_DIR)
        if extracted_dir is None
        else Path(extracted_dir).expanduser()
    )
    project_rows = read_csv_rows(PROJECT_LIST)
    calibrator_rows = read_csv_rows(CALIBRATOR_BANDS_CSV)
    original_paths = filter_ms_paths(find_original_ms_paths(extracted_dir), selected_folders)

    print(f"[INFO] using extracted dir: {extracted_dir}")
    print(f"[INFO] comparing {len(original_paths)} original/selfcal MS pairs")
    print("[INFO] one common grid per source; final clean uses all baselines")

    rows: list[dict] = []
    for original_ms in original_paths:
        folder = sample_top_for_ms(original_ms).name
        print(f"[COMPARE] {folder}")
        try:
            rows.append(
                process_one(
                    original_ms,
                    project_rows=project_rows,
                    calibrator_rows=calibrator_rows,
                )
            )
        except Exception as exc:
            print(f"[ERROR] {folder}: {type(exc).__name__}: {exc}")
            rows.append(
                {
                    "folder": folder,
                    "status": "error",
                    "original_ms": str(original_ms),
                    "selfcal_ms": str(selfcal_ms_for_sample(sample_top_for_ms(original_ms))),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    summary_csv = extracted_dir / SUMMARY_CSV_NAME
    contact_png = extracted_dir / CONTACT_SHEET_NAME
    write_summary(rows, summary_csv)
    write_contact_sheet(rows, contact_png)
    print_interpretation(rows)
    return rows


# if __name__ == "__main__":
#     main()
