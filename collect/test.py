from __future__ import annotations

import time
import random
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
import pyvo

TAP_URL = "https://data-query.nrao.edu/tap"

INPUT_CSV = "collect/vla_calibrators_selected.csv"

RADIUS = 5.0 * u.arcmin
MAX_BYTES: Optional[int] = 200 * 1024 * 1024
MAX_EXPTIME_S: Optional[float] = 300.0
REQUIRE_ACCESS_URL = False

# how many calibrators to test (set None for all)
N_TEST = 10

# print sample values when a step kills results
SHOW_SAMPLE_VALUES = True


def run_sync_retry(svc: pyvo.dal.TAPService, query: str, *, tries: int = 6, base_sleep: float = 1.0):
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
    raise RuntimeError("No ObsCore table found")


def probe_table_columns(svc: pyvo.dal.TAPService, table: str) -> list[str]:
    q = f"""
    SELECT column_name
    FROM TAP_SCHEMA.columns
    WHERE table_name = '{table}'
    ORDER BY column_name
    """
    t = run_sync_retry(svc, q).to_table()
    return [str(x) for x in t["column_name"]]


def pick_first_existing(cols: list[str], *candidates: str) -> str | None:
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None


def parse_coord(ra_val, dec_val) -> SkyCoord:
    # keep it simple: if RA has ":" or "h" => hourangle; else degrees
    ra_s = str(ra_val).strip()
    dec_s = str(dec_val).strip()
    ra_is_hms = (":" in ra_s) or ("h" in ra_s.lower())
    dec_is_sex = (":" in dec_s) or ("d" in dec_s.lower())
    if ra_is_hms and dec_is_sex:
        return SkyCoord(ra_s, dec_s, unit=(u.hourangle, u.deg), frame="icrs")
    if ra_is_hms:
        return SkyCoord(ra_s, float(dec_s) * u.deg, frame="icrs")
    if dec_is_sex:
        return SkyCoord(float(ra_s) * u.deg, dec_s, frame="icrs")
    return SkyCoord(float(ra_s) * u.deg, float(dec_s) * u.deg, frame="icrs")


def mk_query(
    table: str,
    *,
    col_ra: str,
    col_dec: str,
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    extra_where: list[str],
    top: int = 200,
    select: str = "*",
    order_by: str | None = None,
) -> str:
    where = [
        f"CONTAINS(POINT('ICRS', {col_ra}, {col_dec}), CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_deg})) = 1"
    ] + extra_where
    order_clause = f"ORDER BY {order_by}" if order_by else ""
    return f"SELECT TOP {top} {select} FROM {table} WHERE " + " AND ".join(where) + f" {order_clause}"


@dataclass
class Step:
    name: str
    clause: Optional[str]  # None means "skip"


def count_rows(svc: pyvo.dal.TAPService, q: str) -> int:
    # use COUNT(*) to avoid pulling lots of rows
    q_count = "SELECT COUNT(*) AS n FROM (" + q.replace("SELECT TOP", "SELECT") + ") AS sub"
    # Some ADQL backends dislike subqueries. Fallback to TOP 1e6 and count client-side if needed.
    try:
        t = run_sync_retry(svc, q_count).to_table()
        return int(t["n"][0])
    except Exception:
        # fallback: grab just obs_id column up to big top and count
        q2 = q.replace("*", "obs_id")
        t = run_sync_retry(svc, q2).to_table()
        return len(t)


def peek_values(svc: pyvo.dal.TAPService, q_base: str, cols: list[str], table: str):
    if not SHOW_SAMPLE_VALUES:
        return
    wanted = [c for c in ["dataproduct_type", "instrument_name", "proprietary_status", "access_format"] if c in cols]
    if not wanted:
        return
    q = mk_query(
        table,
        col_ra="s_ra" if "s_ra" in cols else "s_ra",
        col_dec="s_dec" if "s_dec" in cols else "s_dec",
        ra_deg=0.0,
        dec_deg=0.0,
        radius_deg=0.0,
        extra_where=[],
    )
    # We can't reuse q_base safely here; instead caller prints uniques from a fetched sample below.
    return


def main():
    df = pd.read_csv(INPUT_CSV)
    if N_TEST is not None:
        df = df.head(N_TEST)

    svc = pyvo.dal.TAPService(TAP_URL)

    table = find_obscore_table(svc)
    cols = probe_table_columns(svc, table)

    col_ra = pick_first_existing(cols, "s_ra", "ra", "obs_ra")
    col_dec = pick_first_existing(cols, "s_dec", "dec", "obs_dec")

    if col_ra is None or col_dec is None:
        raise RuntimeError(f"Could not find RA/Dec columns in {table}")

    print(f"[INFO] table={table}  ra_col={col_ra}  dec_col={col_dec}")
    print(f"[INFO] RADIUS={RADIUS}")

    steps: list[Step] = [
        Step("cone only", None),
        Step(" + dataproduct_type='visibility'", "dataproduct_type = 'visibility'" if "dataproduct_type" in cols else None),
        Step(" + instrument in (VLA,EVLA)", "instrument_name IN ('VLA','EVLA')" if "instrument_name" in cols else None),
        Step(" + proprietary_status='public'", "proprietary_status = 'public'" if "proprietary_status" in cols else None),
        Step(" + size cap", f"access_estsize <= {int(MAX_BYTES)}" if (MAX_BYTES is not None and "access_estsize" in cols) else None),
        Step(" + exptime cap", f"t_exptime <= {float(MAX_EXPTIME_S)}" if (MAX_EXPTIME_S is not None and "t_exptime" in cols) else None),
        Step(" + access_url not null", "access_url IS NOT NULL" if (REQUIRE_ACCESS_URL and "access_url" in cols) else None),
    ]

    # columns to fetch for debugging when something goes to zero
    debug_select = ", ".join([c for c in [
        "dataproduct_type", "instrument_name", "proprietary_status",
        "access_estsize", "t_exptime", "access_format", "obs_publisher_did", "project_code", "target_name"
    ] if c in cols]) or "obs_id"

    for _, r in df.iterrows():
        name = str(r["name"])
        c = parse_coord(r["ra"], r["dec"])
        print(f"\n=== {name}  ra={c.ra.deg:.6f} deg  dec={c.dec.deg:.6f} deg ===")

        extra_where: list[str] = []
        prev_n = None

        for si, step in enumerate(steps):
            if step.clause is not None:
                extra_where.append(step.clause)

            q = mk_query(
                table,
                col_ra=col_ra,
                col_dec=col_dec,
                ra_deg=c.ra.deg,
                dec_deg=c.dec.deg,
                radius_deg=RADIUS.to_value(u.deg),
                extra_where=extra_where,
                top=500,               # enough for counting fallback
                select="obs_id",        # cheap
            )

            t0 = time.time()
            n = count_rows(svc, q)
            dt = time.time() - t0

            marker = ""
            if prev_n is not None and n == 0 and prev_n > 0:
                marker = "  <<< THIS FILTER KILLED IT"
            print(f"{si:02d}. {step.name:<30} n={n:<6} ({dt:.1f}s){marker}")

            if marker and SHOW_SAMPLE_VALUES:
                # fetch a small sample from the PREVIOUS stage (without current killing clause)
                extra_prev = extra_where[:-1]
                q_prev = mk_query(
                    table,
                    col_ra=col_ra,
                    col_dec=col_dec,
                    ra_deg=c.ra.deg,
                    dec_deg=c.dec.deg,
                    radius_deg=RADIUS.to_value(u.deg),
                    extra_where=extra_prev,
                    top=50,
                    select=debug_select,
                )
                sample = run_sync_retry(svc, q_prev).to_table().to_pandas()
                print("   [sample uniques before-kill]")
                for col in ["dataproduct_type", "instrument_name", "proprietary_status", "access_format"]:
                    if col in sample.columns:
                        vals = sorted(sample[col].dropna().astype(str).unique().tolist())
                        print(f"     {col}: {vals[:20]}")
                if "access_estsize" in sample.columns:
                    s = sample["access_estsize"].dropna()
                    if not s.empty:
                        print(f"     access_estsize: min={s.min()}  median={s.median()}  max={s.max()}")

            prev_n = n


if __name__ == "__main__":
    main()
