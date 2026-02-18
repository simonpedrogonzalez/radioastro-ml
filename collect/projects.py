from __future__ import annotations

import os
import time
import random
from typing import Optional, Tuple

import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
import pyvo

TAP_URL = "https://data-query.nrao.edu/tap"

INPUT_CSV = "collect/vla_calibrators_selected.csv"
OUT_CSV = "collect/nrao_archive_hits.csv"

# --- Search behavior ---
RADIUS = 5.0 * u.arcmin           # start larger to avoid small pointing/frame mismatches; tighten later
ONLY_VIS = True                   # dataproduct_type='visibility'
ONLY_VLA = True                   # instrument_name in ('VLA','EVLA') BUT NULL-tolerant in query
MAX_ROWS_PER_SOURCE = 5

# --- Keep datasets small (NULL-tolerant filters in ADQL) ---
MAX_BYTES: Optional[int] = 200 * 1024 * 1024   # 200 MB cap (None to disable)
MAX_EXPTIME_S: Optional[float] = 300.0         # 5 min cap (None to disable)
ONLY_PUBLIC = True                              # (public OR NULL)
REQUIRE_ACCESS_URL = False                      # leave False; access_url can be NULL even when downloadable via web UI
MIN_MJD = 56000  # ~2012-01-01


# --- Runtime / resume ---
WRITE_NOHIT_ROWS = True
DEBUG = True            # prints which RA interpretation was chosen + query time


def run_sync_retry(
    svc: pyvo.dal.TAPService,
    query: str,
    *,
    tries: int = 7,
    base_sleep: float = 1.0,
):
    last = None
    for i in range(tries):
        try:
            return svc.run_sync(query)
        except Exception as e:
            last = e
            msg = str(e).lower()
            transient = (
                "conflict with recovery" in msg
                or "canceling statement" in msg
                or "timeout" in msg
                or "temporarily unavailable" in msg
                or "service unavailable" in msg
            )
            if (not transient) or (i == tries - 1):
                raise
            sleep = base_sleep * (2**i) + random.random()
            print(f"[WARN] transient TAP error; retrying in {sleep:.1f}s: {e}", flush=True)
            time.sleep(sleep)
    raise last


def find_obscore_table(svc: pyvo.dal.TAPService) -> str:
    q = "SELECT table_name FROM TAP_SCHEMA.tables"
    df = run_sync_retry(svc, q).to_table().to_pandas()
    names = set(df["table_name"].astype(str))
    for c in ["tap_schema.obscore", "ivoa.obscore", "ivoa.ObsCore", "obscore.obscore", "obscore.ObsCore"]:
        if c in names:
            return c
    ob = sorted([n for n in names if "obscore" in n.lower()])
    if ob:
        return ob[0]
    raise RuntimeError(f"No ObsCore table found. Sample tables: {sorted(names)[:50]}")


def probe_table_columns(svc: pyvo.dal.TAPService, table_fqtn: str) -> list[str]:
    q = f"""
    SELECT column_name
    FROM TAP_SCHEMA.columns
    WHERE table_name = '{table_fqtn}'
    ORDER BY column_name
    """
    t = run_sync_retry(svc, q).to_table()
    if "column_name" in t.colnames and len(t) > 0:
        return [str(x) for x in t["column_name"]]
    raise RuntimeError(f"No columns found in TAP_SCHEMA.columns for table {table_fqtn}")


def pick_first_existing(cols: list[str], *candidates: str) -> str | None:
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None


def build_select_cols(cols: list[str]) -> list[str]:
    desired = [
        "obs_publisher_did",
        "project_code",
        "target_name",
        "t_min",
        "t_max",
        "t_exptime",
        "instrument_name",
        "dataproduct_type",
        "access_format",
        "access_estsize",
        "access_url",
        "proprietary_status",
        "configuration",
        "freq_min",
        "freq_max",
        "s_ra",
        "s_dec",
    ]
    return [c for c in desired if c in set(cols)]


def make_checkpoint_row(
    *,
    name: str,
    ra_deg: float,
    dec_deg: float,
    select_cols: list[str],
    status: str,
    nrows: int,
    error: str | None = None,
    ra_mode: str | None = None,
) -> pd.DataFrame:
    base = {c: pd.NA for c in select_cols if c != "*"}
    row = {
        "query_name": name,
        "query_ra_deg": ra_deg,
        "query_dec_deg": dec_deg,
        "_status": status,
        "_nrows": nrows,
        "_error": error,
        "_ra_mode": ra_mode,
    }
    cols = ["query_name", "query_ra_deg", "query_dec_deg"] + [c for c in select_cols if c != "*"] + [
        "_status",
        "_nrows",
        "_error",
        "_ra_mode",
    ]
    out = {c: row.get(c, base.get(c, pd.NA)) for c in cols}
    return pd.DataFrame([out])


def append_rows(out_csv: str, df_rows: pd.DataFrame):
    write_header = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
    df_rows.to_csv(out_csv, mode="a", header=write_header, index=False)


def load_last_processed_name(out_csv: str) -> str | None:
    if not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0:
        return None
    try:
        prev = pd.read_csv(out_csv, usecols=["query_name"])
        if prev.empty:
            return None
        return str(prev["query_name"].iloc[-1])
    except Exception:
        return None


def adql_cone(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    *,
    table: str,
    col_ra: str,
    col_dec: str,
    select_cols: list[str],
    col_dataproduct: str | None,
    col_instrument: str | None,
) -> str:
    where = [
        f"CONTAINS(POINT('ICRS', {col_ra}, {col_dec}), "
        f"CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_deg})) = 1"
    ]

    if ONLY_VIS and col_dataproduct is not None:
        where.append(f"{col_dataproduct} = 'visibility'")

    # NULL-tolerant instrument filter (important!)
    if ONLY_VLA and col_instrument is not None:
        where.append(f"({col_instrument} IN ('VLA','EVLA')") # OR {col_instrument} IS NULL)")

    # NULL-tolerant “public”
    if ONLY_PUBLIC and "proprietary_status" in select_cols:
        where.append("(proprietary_status = 'PUBLIC')")#  OR proprietary_status IS NULL)")

    # NULL-tolerant size/exptime caps
    if MAX_BYTES is not None and "access_estsize" in select_cols:
        where.append(f"(access_estsize <= {int(MAX_BYTES)})")# OR access_estsize IS NULL)")
    
    if MAX_EXPTIME_S is not None and "t_exptime" in select_cols:
        where.append(f"(t_exptime <= {float(MAX_EXPTIME_S)})")# OR t_exptime IS NULL)")

    # Do NOT require access_url at this stage (it can be NULL even for things you can download via AAT)
    if REQUIRE_ACCESS_URL and "access_url" in select_cols:
        where.append("(access_url IS NOT NULL)")
    
    if "t_min" in select_cols:
        where.append(f"(t_min >= {MIN_MJD})")# OR t_min IS NULL)")

    select_list = ", ".join(select_cols) if select_cols else "*"

    order_by = []
    if "access_estsize" in select_cols:
        order_by.append("access_estsize ASC")
    if "t_exptime" in select_cols:
        order_by.append("t_exptime ASC")
    order_clause = ("ORDER BY " + ", ".join(order_by)) if order_by else ""

    return f"""
    SELECT TOP {MAX_ROWS_PER_SOURCE} {select_list}
    FROM {table}
    WHERE {" AND ".join(where)}
    {order_clause}
    """


def adql_probe_any_hit(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    *,
    table: str,
    col_ra: str,
    col_dec: str,
) -> str:
    # super cheap probe: no filters, tiny select, TOP 1
    return f"""
    SELECT TOP 1 obs_id
    FROM {table}
    WHERE CONTAINS(POINT('ICRS', {col_ra}, {col_dec}),
                   CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_deg})) = 1
    """


def parse_coord_auto(
    svc: pyvo.dal.TAPService,
    table: str,
    col_ra: str,
    col_dec: str,
    ra_val,
    dec_val,
) -> Tuple[SkyCoord, str]:
    """
    If RA is numeric/ambiguous, try interpreting it as degrees and as hours,
    run a cheap TOP 1 probe for each, and pick the one that returns a row.
    """
    ra_s = str(ra_val).strip()
    dec_s = str(dec_val).strip()

    # If RA clearly sexagesimal -> hourangle
    if (":" in ra_s) or ("h" in ra_s.lower()):
        c = SkyCoord(ra_s, dec_s, unit=(u.hourangle, u.deg), frame="icrs")
        return c, "hms"

    # Otherwise RA numeric-ish
    ra_f = float(ra_s)
    # DEC might be sexagesimal or numeric
    if (":" in dec_s) or ("d" in dec_s.lower()):
        dec_obj = dec_s
        dec_unit = None
    else:
        dec_obj = float(dec_s)
        dec_unit = u.deg

    # Candidate A: RA as degrees
    c_deg = SkyCoord(ra_f * u.deg, dec_obj if dec_unit is None else dec_obj * dec_unit, frame="icrs")
    # Candidate B: RA as hours
    c_hr = SkyCoord(ra_f * u.hourangle, dec_obj if dec_unit is None else dec_obj * dec_unit, frame="icrs")

    # Probe each (use a slightly larger probe radius so the test is robust)
    probe_r = max(RADIUS, 10.0 * u.arcmin).to_value(u.deg)

    q_deg = adql_probe_any_hit(c_deg.ra.deg, c_deg.dec.deg, probe_r, table=table, col_ra=col_ra, col_dec=col_dec)
    q_hr = adql_probe_any_hit(c_hr.ra.deg, c_hr.dec.deg, probe_r, table=table, col_ra=col_ra, col_dec=col_dec)

    try:
        hit_deg = len(run_sync_retry(svc, q_deg).to_table()) > 0
    except Exception:
        hit_deg = False
    try:
        hit_hr = len(run_sync_retry(svc, q_hr).to_table()) > 0
    except Exception:
        hit_hr = False

    if hit_deg and not hit_hr:
        return c_deg, "deg"
    if hit_hr and not hit_deg:
        return c_hr, "hours"
    if hit_hr and hit_deg:
        # both hit: pick degrees (more common), but you can change preference
        return c_deg, "deg"

    # neither hit: fall back to degrees
    return c_deg, "deg_fallback"


def main():
    df = pd.read_csv(INPUT_CSV)

    last_name = load_last_processed_name(OUT_CSV)
    if last_name is not None:
        print(f"[INFO] Found existing {OUT_CSV}; last processed query_name = {last_name}", flush=True)
        names = df["name"].astype(str).tolist()
        try:
            start_idx = names.index(last_name) + 1
        except ValueError:
            start_idx = 0
    else:
        start_idx = 0

    svc = pyvo.dal.TAPService(TAP_URL)

    obscore_table = find_obscore_table(svc)
    print(f"[INFO] Using ObsCore table: {obscore_table}", flush=True)

    cols = probe_table_columns(svc, obscore_table)
    col_ra = pick_first_existing(cols, "s_ra", "ra", "obs_ra")
    col_dec = pick_first_existing(cols, "s_dec", "dec", "obs_dec")
    col_dataproduct = pick_first_existing(cols, "dataproduct_type", "dataproduct", "product_type")
    col_instrument = pick_first_existing(cols, "instrument_name", "instrument", "telescope_name")

    if col_ra is None or col_dec is None:
        raise RuntimeError(f"Could not find RA/Dec columns. Detected columns: {cols}")

    select_cols = build_select_cols(cols)
    if not select_cols:
        select_cols = ["*"]

    total = len(df)
    df_iter = df.iloc[start_idx:]

    print(f"[INFO] Starting at index {start_idx}/{total}", flush=True)
    print(
        f"[INFO] Filters: RADIUS={RADIUS.to_value(u.arcmin):.2f} arcmin, "
        f"ONLY_VIS={ONLY_VIS}, ONLY_VLA={ONLY_VLA}, ONLY_PUBLIC={ONLY_PUBLIC}, "
        f"MAX_ROWS_PER_SOURCE={MAX_ROWS_PER_SOURCE}, MAX_BYTES={MAX_BYTES}, MAX_EXPTIME_S={MAX_EXPTIME_S}",
        flush=True,
    )

    for j, (_, r) in enumerate(df_iter.iterrows(), start=start_idx):
        name = str(r["name"])
        print(f"[{j+1}/{total}] Searching for {name}", flush=True)

        c, ra_mode = parse_coord_auto(svc, obscore_table, col_ra, col_dec, r["ra"], r["dec"])
        if DEBUG:
            print(f"    coord ra_deg={c.ra.deg:.6f} dec_deg={c.dec.deg:.6f} (ra_mode={ra_mode})", flush=True)

        q = adql_cone(
            c.ra.deg,
            c.dec.deg,
            RADIUS.to_value(u.deg),
            table=obscore_table,
            col_ra=col_ra,
            col_dec=col_dec,
            select_cols=select_cols,
            col_dataproduct=col_dataproduct,
            col_instrument=col_instrument,
        )

        t0 = time.time()
        try:
            res = run_sync_retry(svc, q).to_table().to_pandas()
        except Exception as e:
            print(f"[WARN] query failed for {name}: {e}", flush=True)
            if WRITE_NOHIT_ROWS:
                ck = make_checkpoint_row(
                    name=name,
                    ra_deg=float(c.ra.deg),
                    dec_deg=float(c.dec.deg),
                    select_cols=select_cols,
                    status="error",
                    nrows=0,
                    error=str(e)[:300],
                    ra_mode=ra_mode,
                )
                append_rows(OUT_CSV, ck)
            continue

        dt = time.time() - t0
        print(f"    -> {len(res)} rows in {dt:.1f}s", flush=True)

        if res.empty:
            if WRITE_NOHIT_ROWS:
                ck = make_checkpoint_row(
                    name=name,
                    ra_deg=float(c.ra.deg),
                    dec_deg=float(c.dec.deg),
                    select_cols=select_cols,
                    status="nohit",
                    nrows=0,
                    error=None,
                    ra_mode=ra_mode,
                )
                append_rows(OUT_CSV, ck)
            continue

        res.insert(0, "query_name", name)
        res.insert(1, "query_ra_deg", float(c.ra.deg))
        res.insert(2, "query_dec_deg", float(c.dec.deg))
        res["_status"] = "hit"
        res["_nrows"] = len(res)
        res["_error"] = pd.NA
        res["_ra_mode"] = ra_mode

        append_rows(OUT_CSV, res)
        print(f"    [OK] appended {len(res)} rows to {OUT_CSV}", flush=True)

    print("[DONE] Finished loop.", flush=True)


if __name__ == "__main__":
    main()
