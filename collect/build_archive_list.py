from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from fetcher import NRAOQuery, TIMESPANS, INSTRUMENTS, DATAPRODS, CONFIGS, BANDS, PROPRIETARY
from product_details import fetch_product_details, summarize_product_details


# =============================================================================
# Output schemas
# =============================================================================

HIT_FIELDS = [
    "name",
    "ra",
    "dec",
    "band_guess",        # from your input row inference (receiver/wavelength)
    "band_code",         # from NRAO details API (execblock band_code)
    "size_gb",
    "date_utc",
    "gain_calibrator_name",
    "gain_onsource_min",
    "gain_array_config",
    "access_url",
]

MISS_FIELDS = [
    "name",
    "ra",
    "dec",
    "band_guess",
    "configs",
    "reason",
]


# =============================================================================
# CSV append + resume helpers
# =============================================================================

def _append_row_csv(out_csv: str | Path, row: Dict[str, Any], *, fields: list[str]) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df_row = pd.DataFrame([row], columns=fields)
    write_header = not out_csv.exists()
    df_row.to_csv(out_csv, mode="a", header=write_header, index=False)


def _load_existing_names(csv_path: str | Path, *, name_col: str = "name") -> set[str]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path, usecols=[name_col])
    names = set(str(x).strip() for x in df[name_col].dropna().tolist())
    names.discard("")
    return names


# =============================================================================
# Calibrator row -> band/config guesses
# =============================================================================

def infer_band_from_row(row: pd.Series):
    """
    Try receiver first, then wavelength (e.g., '6cm', '20cm').
    If both fail, return None (we'll just not apply a band filter).
    """
    for key in ("receiver", "wavelength"):
        if key in row and pd.notna(row[key]):
            s = str(row[key]).strip()
            if not s:
                continue
            try:
                return BANDS.get(s)
            except Exception:
                pass
    return None


def parse_configs_from_row(row: pd.Series) -> list[str]:
    """
    If your selected table has 'usable_configs' like 'A,B,C', keep those.
    Otherwise fall back to all configs.
    """
    if "usable_configs" in row and pd.notna(row["usable_configs"]):
        s = str(row["usable_configs"])
        cfgs = [x.strip() for x in s.split(",") if x.strip() in CONFIGS.ALL]
        return cfgs if cfgs else CONFIGS.ALL
    return CONFIGS.ALL


# =============================================================================
# Core: find one good archive hit
# =============================================================================

def find_one_valid_hit_for_calibrator(
    name: str,
    pos: SkyCoord,
    cfgs: list[str],
    band,  # Band | None
    *,
    radius: u.Quantity = 5 * u.arcsec,
    max_candidates: int = 10,
) -> Optional[Dict[str, Any]]:

    q = (
        NRAOQuery(limit=max_candidates)
        .select("t_min", "access_estsize", "access_url")
        .where_timespan(TIMESPANS.FROM_2016_SEP)
        .where_in_circle(pos, radius)
        .where_instruments(INSTRUMENTS.VLA_VARIANTS())
        .where_dataproduct(DATAPRODS.VISIBILITY)
        .where_configs(cfgs)
        .where_proprietary_status(PROPRIETARY.PUBLIC)
        .order_by("access_estsize ASC")
    )
    if band is not None:
        q = q.where_band(band)

    # skip catastrophic TAP failures
    try:
        df = q.get()
    except Exception as e:
        print(f"[WARN] {name}: TAP query failed, skipping. ({type(e).__name__}: {e})")
        return None

    if df.empty:
        return None

    for i in range(len(df)):
        access_url = df.loc[i, "access_url"]
        if not isinstance(access_url, str) or not access_url:
            continue

        try:
            size_gb = float(df.loc[i, "access_estsize"]) / 1_000_000.0  # kB -> GB (SI)
        except Exception:
            size_gb = None

        try:
            tmin_mjd = float(df.loc[i, "t_min"])
            date_utc = Time(tmin_mjd, format="mjd", scale="utc").strftime("%Y-%m-%d")
        except Exception:
            date_utc = None

        try:
            details = fetch_product_details(access_url)
            summary = summarize_product_details(details, cal_center=pos)
        except Exception:
            continue

        if not summary.get("has_caltables", False):
            continue
        if not summary.get("probable_gain_calibrator"):
            continue

        gs = summary.get("gain_calibrator_stats") or {}
        gain_min = gs.get("total_on_source_min")

        cfg_time_s = gs.get("array_config_time_s") or {}
        gain_cfg = max(cfg_time_s.items(), key=lambda kv: kv[1])[0] if cfg_time_s else summary.get("array_config")

        return {
            "name": name,
            "ra": pos.ra.to_string(unit=u.hourangle, sep=":", precision=6),
            "dec": pos.dec.to_string(unit=u.deg, sep=":", alwayssign=True, precision=6),
            "band_guess": getattr(band, "key", None),
            "band_code": summary.get("band_code"),
            "size_gb": size_gb,
            "date_utc": date_utc,
            "gain_calibrator_name": summary.get("probable_gain_calibrator"),
            "gain_onsource_min": gain_min,
            "gain_array_config": gain_cfg,
            "access_url": access_url,
        }

    return None


# =============================================================================
# Runner: resume + missed tracking
# =============================================================================

def build_calibrator_hit_table(
    in_csv: str | Path,
    hit_csv: str | Path,
    missed_csv: str | Path,
    *,
    max_per_calibrator_candidates: int = 25,
):
    df = pd.read_csv(in_csv)

    done_hits = _load_existing_names(hit_csv)
    done_misses = _load_existing_names(missed_csv)

    if done_hits:
        print(f"[INFO] Found existing HIT CSV with {len(done_hits)} calibrators.")
    if done_misses:
        print(f"[INFO] Found existing MISSED CSV with {len(done_misses)} calibrators.")

    already_done = done_hits | done_misses
    if already_done:
        print(f"[INFO] Total already processed (hit or miss): {len(already_done)}")

    for name, g in df.groupby("name", sort=False):
        if name in already_done:
            print(f"[SKIP] {name} already processed.")
            continue

        row0 = g.iloc[0]

        pos = SkyCoord(
            str(row0["ra"]),
            str(row0["dec"]),
            unit=(u.hourangle, u.deg),
            frame="icrs",
        )

        band = infer_band_from_row(row0)
        cfgs = parse_configs_from_row(row0)

        print(f"\n=== {name} | band_guess={getattr(band, 'key', None)} | cfgs={cfgs} ===")

        try:
            out = find_one_valid_hit_for_calibrator(
                name=name,
                pos=pos,
                cfgs=cfgs,
                band=band,
                max_candidates=max_per_calibrator_candidates,
            )
        except Exception as e:
            miss_row = {
                "name": name,
                "ra": pos.ra.to_string(unit=u.hourangle, sep=":", precision=6),
                "dec": pos.dec.to_string(unit=u.deg, sep=":", alwayssign=True, precision=6),
                "band_guess": getattr(band, "key", None),
                "configs": ",".join(cfgs),
                "reason": f"exception: {type(e).__name__}: {e}",
            }
            _append_row_csv(missed_csv, miss_row, fields=MISS_FIELDS)
            already_done.add(name)
            print(f"[MISS] {name}: exception -> recorded in missed CSV.")
            continue

        if out is None:
            miss_row = {
                "name": name,
                "ra": pos.ra.to_string(unit=u.hourangle, sep=":", precision=6),
                "dec": pos.dec.to_string(unit=u.deg, sep=":", alwayssign=True, precision=6),
                "band_guess": getattr(band, "key", None),
                "configs": ",".join(cfgs),
                "reason": "no valid dataset found (needs has_caltables + probable_gain_calibrator)",
            }
            _append_row_csv(missed_csv, miss_row, fields=MISS_FIELDS)
            already_done.add(name)
            print(f"[MISS] {name}: recorded in missed CSV.")
            continue

        _append_row_csv(hit_csv, out, fields=HIT_FIELDS)
        already_done.add(name)

        # optional: flag mismatches between guessed band and archive band_code
        bg = out.get("band_guess")
        bc = out.get("band_code")
        if bg and bc and str(bg).upper() != str(bc).upper():
            print(f"[WARN] {name}: band_guess={bg} but details band_code={bc}")

        print(f"[OK] wrote: {name} -> {out['size_gb']:.3f} GB | {out['date_utc']} | gain={out['gain_calibrator_name']}")


if __name__ == "__main__":
    build_calibrator_hit_table(
        in_csv="vla_calibrators_selected.csv",
        hit_csv="vla_calibrators_one_hit.csv",
        missed_csv="vla_calibrators_one_hit_missed.csv",
    )