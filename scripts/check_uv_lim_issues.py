from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


REPO_ROOT = Path("/Users/u1528314/repos/radioastro-ml")
COLLECT_DIR = REPO_ROOT / "collect"
CALIBRATOR_BANDS_CSV = COLLECT_DIR / "vla_calibrators_bands_v2.csv"
OUTPUT_PATH = REPO_ROOT / "uv_lim_issue.txt"
EXTRACTED_GROUPS: list[tuple[str, Path]] = [
    ("UV_LIM", COLLECT_DIR / "extracted"),
    ("UV_LIM_2", COLLECT_DIR / "extracted2"),
    ("UV_LIM_3", COLLECT_DIR / "extracted3"),
]
_BAND_RANGES_GHZ = {
    "L": (1.0, 2.0),
    "S": (2.0, 4.0),
    "C": (4.0, 8.0),
    "X": (8.0, 12.0),
    "KU": (12.0, 18.0),
    "K": (18.0, 26.5),
    "KA": (26.5, 40.0),
    "Q": (40.0, 50.0),
}


def table():
    try:
        from casatools import table as casa_table
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "casatools is required to run this script. Run it in a CASA-enabled Python environment."
        ) from exc
    return casa_table()


def load_calibrator_uv_limits(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_extracted_ms_paths(extracted_dir: Path) -> list[Path]:
    hits: list[Path] = []
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


def parse_float(value) -> float:
    try:
        if value in (None, ""):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def lookup_calibrator_uv_limits(
    calib_rows: list[dict],
    *,
    calibrator_name: str | None,
    band: str | None,
) -> dict | None:
    name = normalize_calibrator_name(calibrator_name)
    receiver = band_to_receiver_code(band)
    if not name or receiver is None:
        return None

    for row in calib_rows:
        row_name = normalize_calibrator_name(row.get("name"))
        row_receiver = str(row.get("receiver", "")).strip().upper()
        if row_name != name or row_receiver != receiver:
            continue

        uvmin = parse_float(row.get("uvmin_kl"))
        uvmax = parse_float(row.get("uvmax_kl"))
        if np.isfinite(uvmin) and np.isfinite(uvmax) and uvmin > uvmax:
            uvmin, uvmax = uvmax, uvmin

        return {
            "calibrator_name": str(row.get("name", calibrator_name)),
            "receiver": row_receiver,
            "uvmin_kl": uvmin,
            "uvmax_kl": uvmax,
        }

    return None


def band_for_frequency_ghz(freq_ghz: float | None) -> str | None:
    if freq_ghz is None or not math.isfinite(freq_ghz):
        return None
    for band, (lo, hi) in _BAND_RANGES_GHZ.items():
        if lo <= freq_ghz <= hi:
            return band
    return None


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


def get_ms_reference_frequency_ghz(ms_path: Path) -> tuple[float | None, int | None]:
    spw_info = _read_spw_frequency_info(ms_path)
    if not spw_info:
        return None, None

    best = max(spw_info, key=lambda x: x["bandwidth_hz"])
    return best["f_center_ghz"], best["spw"]


def choose_band_and_frequency(ms_path: Path) -> dict:
    ms_freq_ghz, used_spw = get_ms_reference_frequency_ghz(ms_path)
    detected_band = band_for_frequency_ghz(ms_freq_ghz)
    return {
        "selected_band": detected_band,
        "ms_freq_ghz": ms_freq_ghz,
        "used_spw": used_spw,
    }


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


def write_line(handle, line: str = "") -> None:
    handle.write(line + "\n")
    handle.flush()


def has_uv_limit_issue(uv_cov: dict) -> bool:
    below = uv_cov.get("uv_fraction_below_uvmin", np.nan)
    above = uv_cov.get("uv_fraction_above_uvmax", np.nan)
    total_outside = 0.0
    if np.isfinite(below):
        total_outside += float(below)
    if np.isfinite(above):
        total_outside += float(above)
    return total_outside > 0.0


def format_outside_percent(uv_cov: dict) -> float:
    below = uv_cov.get("uv_fraction_below_uvmin", np.nan)
    above = uv_cov.get("uv_fraction_above_uvmax", np.nan)
    total_outside = 0.0
    if np.isfinite(below):
        total_outside += float(below)
    if np.isfinite(above):
        total_outside += float(above)
    return 100.0 * total_outside


def process_group(
    *,
    variable_name: str,
    extracted_dir: Path,
    calib_rows: list[dict],
    handle,
) -> int:
    print(f"[START] {variable_name} from {extracted_dir}")
    write_line(handle, f"{variable_name} = [")

    issue_count = 0
    ms_paths = find_extracted_ms_paths(extracted_dir)

    for idx, ms_path in enumerate(ms_paths, start=1):
        folder_name = ms_path.parent.parent.name
        band_info = choose_band_and_frequency(ms_path)
        uv_limit_info = lookup_calibrator_uv_limits(
            calib_rows,
            calibrator_name=folder_name,
            band=band_info["selected_band"],
        )

        if uv_limit_info is None:
            print(
                f"[SKIP] {variable_name} {idx}/{len(ms_paths)} {folder_name} | "
                f"band={band_info['selected_band'] or '?'} | no catalog uv limits"
            )
            continue

        uv_cov = compute_uvlimit_coverage_stats(
            ms_path,
            band_info["ms_freq_ghz"],
            uv_limit_info["uvmin_kl"],
            uv_limit_info["uvmax_kl"],
        )

        outside_pct = format_outside_percent(uv_cov)
        print(
            f"[CHECK] {variable_name} {idx}/{len(ms_paths)} {folder_name} | "
            f"band={band_info['selected_band'] or '?'} | "
            f"freq={band_info['ms_freq_ghz'] if band_info['ms_freq_ghz'] is not None else float('nan'):.3f} GHz | "
            f"outside={outside_pct:.3f}%"
        )

        if has_uv_limit_issue(uv_cov):
            write_line(handle, f'"{folder_name}",')
            issue_count += 1

    write_line(handle, "]")
    write_line(handle)
    print(f"[DONE] {variable_name} | issues={issue_count}")
    return issue_count


def main() -> None:
    calib_rows = load_calibrator_uv_limits(CALIBRATOR_BANDS_CSV)

    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        total_issues = 0
        for variable_name, extracted_dir in EXTRACTED_GROUPS:
            total_issues += process_group(
                variable_name=variable_name,
                extracted_dir=extracted_dir,
                calib_rows=calib_rows,
                handle=handle,
            )

    print(f"[FINISH] wrote {OUTPUT_PATH} | total_issues={total_issues}")

