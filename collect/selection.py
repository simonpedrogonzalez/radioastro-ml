from __future__ import annotations

import pandas as pd
import pyvo
from astropy.coordinates import SkyCoord
import astropy.units as u

from astropy.time import Time

def date_to_mjd(date_str: str) -> float:
    t = Time(date_str, format="iso", scale="utc")
    return t.mjd

def mjd_to_yyyy_mm_dd(mjd: float) -> str:
    t = Time(mjd, format="mjd", scale="utc")
    return t.to_datetime().strftime("%Y-%m-%d")

def find_obscore_table(svc: pyvo.dal.TAPService) -> str:
    # List all tables
    q = "SELECT table_name FROM TAP_SCHEMA.tables"
    df = svc.run_sync(q).to_table().to_pandas()
    names = set(df["table_name"].astype(str))
    preferred = [
        "ivoa.obscore",
        "tap_schema.obscore",
        "ivoa.ObsCore",
        "obscore.obscore",
        "obscore.ObsCore",
    ]
    for name in preferred:
        if name in names:
            print(f"[INFO] Found ObsCore table: {name}")
            return name
    candidates = sorted([n for n in names if "obscore" in n.lower()])
    if candidates:
        print(f"[INFO] Found ObsCore table (fallback): {candidates[0]}")
        return candidates[0]
    raise RuntimeError(
        f"No ObsCore table found. Sample tables: {sorted(names)[:30]}"
    )

TAP_URL = "https://data-query.nrao.edu/tap"
OBSCORE_TABLE = find_obscore_table(pyvo.dal.TAPService(TAP_URL))

MAX_ROWS = 1
RADIUS = 20 * u.arcsec
MIN_MJD = date_to_mjd('2016-10-01') # date in which cal obs were included

CONFIG_KEYS = ("A", "B", "C", "D")

# crude but effective VLA band ranges
BAND_RANGES_HZ = {
    "L":  (1e9, 2e9),         # ~20 cm
    "S":  (2e9, 4e9),         # ~13 cm
    "C":  (4e9, 8e9),         # ~6 cm
    "X":  (8e9, 12e9),        # ~3.7 cm
    "KU": (12e9, 18e9),       # ~2 cm
    "K":  (18e9, 26.5e9),     # ~1.3 cm
    "KA": (26.5e9, 40e9),     # ~9 mm
    "Q":  (40e9, 50e9),       # ~7 mm
}

def parse_usable_configs(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip() in CONFIG_KEYS]

def band_range_from_row(row) -> tuple[float, float]:
    rx = str(row["receiver"]).strip().upper()
    if rx in BAND_RANGES_HZ:
        return BAND_RANGES_HZ[rx]
    # fallback: use wavelength string if receiver missing
    wl = str(row["wavelength"]).strip().lower()
    if wl.endswith("cm"):
        cm = float(wl.replace("cm", ""))
        nu = (u.speed_of_light / (cm * u.cm)).to_value(u.Hz)
        # +/- 25% window as a coarse heuristic
        return (0.75 * nu, 1.25 * nu)
    raise ValueError(f"Cannot infer band from receiver={row['receiver']} wavelength={row['wavelength']}")

def query_candidates_for_calibrator_band(row: pd.Series) -> pd.DataFrame:
    print(f"row: {row}")
    c = SkyCoord(str(row["ra"]), str(row["dec"]), unit=(u.hourangle, u.deg), frame="icrs")

    cfgs = parse_usable_configs(row["usable_configs"])
    if not cfgs:
        raise ValueError("No usable configs in row")

    f_lo, f_hi = band_range_from_row(row)

    # “overlap” condition: dataset freq range intersects desired band range
    freq_overlap = f"(freq_max >= {f_lo} AND freq_min <= {f_hi})"

    q = f"""
    SELECT TOP {MAX_ROWS}
        obs_publisher_did, project_code, target_name,
        t_min, t_max, t_exptime,
        instrument_name, configuration,
        dataproduct_type,
        freq_min, freq_max,
        access_estsize, proprietary_status
    FROM {OBSCORE_TABLE}
    WHERE
        CONTAINS(
            POINT('ICRS', s_ra, s_dec),
            CIRCLE('ICRS', {c.ra.deg}, {c.dec.deg}, {RADIUS.to_value(u.deg)})
        ) = 1
        AND t_min >= {MIN_MJD}
        AND instrument_name IN ('VLA','EVLA')
        AND dataproduct_type = 'visibility'
        AND configuration IN ({",".join([f"'{x}'" for x in cfgs])})
        AND {freq_overlap}
    ORDER BY t_min DESC
    """

    print(q)
    svc = pyvo.dal.TAPService(TAP_URL)
    return svc.run_sync(q).to_table().to_pandas()

# --- Example: use your one calibrator’s two rows ---
df = pd.read_csv("collect/vla_calibrators_selected.csv")
df_one = df[df["name"] == "0005+383"]

for _, row in df_one.iterrows():
    hits = query_candidates_for_calibrator_band(row)
    print(row["wavelength"], row["receiver"], "->", len(hits), "candidate datasets")
    print(hits[["obs_publisher_did","project_code","configuration","t_min","freq_min","freq_max"]].head(10))
