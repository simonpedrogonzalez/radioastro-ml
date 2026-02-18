from __future__ import annotations

from urllib.parse import urlparse
import re
import time
from typing import Any, Optional, Dict

import requests


NRAO_DETAILS_API = "https://data.nrao.edu/archive-service/restapi_product_details_view"
DEFAULT_TIMEOUT = 30


def extract_sdm_id_from_access_url(access_url: str) -> str:
    """
    Extract the SDM ID from an NRAO portal access_url like:
      https://data.nrao.edu/portal/#/productViewer/<SDM_ID>

    Returns the <SDM_ID> string, e.g.:
      24B-465.sb47226343.eb47329790.60643.9672167824
    """
    if not access_url:
        raise ValueError("access_url is empty")

    # Most common: ".../#/productViewer/<sdm_id>"
    m = re.search(r"/#/(?:productViewer|productviewer)/([^/?#]+)", access_url)
    if m:
        return m.group(1)

    # Fallback: last path-ish token
    u = urlparse(access_url)
    tail = u.path.rstrip("/").split("/")[-1]
    if tail and "." in tail:
        return tail

    raise ValueError(f"Could not parse sdm_id from access_url: {access_url!r}")


def build_product_details_url(sdm_id: str) -> str:
    return f"{NRAO_DETAILS_API}?sdm_id={sdm_id}"


def fetch_product_details(
    access_url: str,
    *,
    session: Optional[requests.Session] = None,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = 5,
    backoff: float = 1.0,
) -> Dict[str, Any]:
    """
    Given an NRAO access_url (portal productViewer URL), fetch product details JSON
    via the archive-service REST API.

    Returns parsed JSON (dict).
    """
    sdm_id = extract_sdm_id_from_access_url(access_url)
    api_url = build_product_details_url(sdm_id)

    sess = session or requests.Session()
    last_err: Exception | None = None

    for i in range(retries):
        try:
            r = sess.get(api_url, timeout=timeout)
            if r.status_code == 429:
                # rate-limited
                sleep = backoff * (2**i)
                time.sleep(sleep)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            sleep = backoff * (2**i)
            time.sleep(sleep)

    raise RuntimeError(f"Failed to fetch product details for sdm_id={sdm_id}") from last_err


# --- tiny convenience wrapper if you already have sdm_id ---
def fetch_product_details_by_sdm_id(
    sdm_id: str,
    *,
    session: Optional[requests.Session] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    api_url = build_product_details_url(sdm_id)
    sess = session or requests.Session()
    r = sess.get(api_url, timeout=timeout)
    r.raise_for_status()
    return r.json()

import astropy.units as u
from astropy.time import Time
from collections import defaultdict
from typing import Any, Dict, Tuple


def _intent_tokens(intent: str) -> set[str]:
    return set((intent or "").upper().split())


def _pick_gain_calibrator(scan_rows: list[dict[str, Any]]) -> tuple[str | None, dict[str, Any] | None]:
    """
    Heuristic:
      - gain cal = CALIBRATE_PHASE (often also CALIBRATE_AMPLI)
      - exclude pure flux/bandpass/delay calibrators
      - if multiple, pick the one with most gain-cal time (sum scan durations if present, else most scans)
    """
    # candidate targets that have PHASE (gain-ish)
    per_target = {}

    for s in scan_rows:
        intent = s.get("intent") or ""
        toks = _intent_tokens(intent)

        if "CALIBRATE" not in toks and "CALIBRATE_PHASE" not in toks and "CALIBRATE_AMPLI" not in toks:
            continue

        # must include phase to be a gain calibrator in practice
        if "CALIBRATE_PHASE" not in toks:
            continue

        # exclude "primary" calibrator intents (can still contain PHASE sometimes in weird cases)
        if toks & {"CALIBRATE_FLUX", "CALIBRATE_BANDPASS", "CALIBRATE_DELAY"}:
            continue

        tgt = s.get("target_name")
        if not tgt:
            continue

        # duration in days? (in your API it looked like `duration` is in days)
        # scan_rows have 'duration' sometimes; if not, we fallback to count
        dur = s.get("duration")
        if dur is None:
            dur_sec = 0.0
        else:
            # NRAO API `duration` is in days (matches your earlier values like 0.0013)
            dur_sec = float(dur) * 86400.0

        rec = per_target.setdefault(
            tgt,
            {"target_name": tgt, "n_scans": 0, "sum_gaincal_sec": 0.0, "example_intents": set()},
        )
        rec["n_scans"] += 1
        rec["sum_gaincal_sec"] += dur_sec
        rec["example_intents"].add(intent)

    if not per_target:
        return None, None

    # pick by sum duration if any nonzero durations exist, else by scan count
    any_dur = any(v["sum_gaincal_sec"] > 0 for v in per_target.values())
    if any_dur:
        best = max(per_target.values(), key=lambda r: r["sum_gaincal_sec"])
    else:
        best = max(per_target.values(), key=lambda r: r["n_scans"])

    # make examples printable
    best["example_intents"] = sorted(best["example_intents"])
    return best["target_name"], best


def _parse_dt(dt_str: str) -> Time | None:
    # NRAO API strings look like: "2024-11-29 23:23:15.150"
    if not dt_str:
        return None
    try:
        return Time(dt_str, format="iso", scale="utc")
    except Exception:
        return None


def _scan_duration_seconds(scan: dict[str, Any]) -> float:
    """
    Prefer explicit duration if present (NRAO seems to give it in days),
    else compute from start/end timestamps if present.
    """
    dur = scan.get("duration")
    if dur is not None:
        try:
            return float(dur) * 86400.0  # days -> seconds
        except Exception:
            pass

    t0 = _parse_dt(scan.get("starttime"))
    t1 = _parse_dt(scan.get("endtime"))
    if t0 is None or t1 is None:
        return 0.0
    return max(0.0, (t1 - t0).to_value("s"))


def _scan_config_letter(scan: dict[str, Any], eb: dict[str, Any]) -> str | None:
    """
    For VLA, the array configuration letter is usually the execblock-level `configuration`.
    Some APIs also have an observation_configuration_number/id (spectral setup),
    but that's NOT the A/B/C/D letter. We return the array config letter.
    """
    return eb.get("configuration")


def _gain_calibrator_stats(
    *,
    scan_rows: list[dict[str, Any]],
    eb: dict[str, Any],
    gain_target: str,
) -> dict[str, Any]:
    """
    Computes:
      - total on-source time (sum of scan durations, no gaps)
      - breakdown by array config letter (usually one)
      - list of scans with (scan_num, start, end, dur_s, intent)
    """
    scans = []
    total_s = 0.0
    by_cfg_s = defaultdict(float)

    for s in scan_rows:
        if (s.get("target_name") or "") != gain_target:
            continue

        intent = (s.get("intent") or "").upper()
        # keep only the gain-cal scans
        toks = set(intent.split())
        if "CALIBRATE_PHASE" not in toks:
            continue
        if toks & {"CALIBRATE_FLUX", "CALIBRATE_BANDPASS", "CALIBRATE_DELAY"}:
            continue

        dur_s = _scan_duration_seconds(s)
        total_s += dur_s

        cfg = _scan_config_letter(s, eb) or "UNKNOWN"
        by_cfg_s[cfg] += dur_s

        scans.append(
            {
                "scan_num": s.get("scan_num"),
                "starttime": s.get("starttime"),
                "endtime": s.get("endtime"),
                "dur_s": dur_s,
                "intent": s.get("intent"),
                "array_config": cfg,
            }
        )

    # sort by scan_num if numeric-ish
    def _scan_key(x):
        try:
            return int(str(x.get("scan_num")))
        except Exception:
            return str(x.get("scan_num"))

    scans.sort(key=_scan_key)

    return {
        "gain_target": gain_target,
        "total_on_source_s": total_s,
        "total_on_source_min": total_s / 60.0,
        "array_config_time_s": dict(by_cfg_s),
        "array_config_time_min": {k: v / 60.0 for k, v in by_cfg_s.items()},
        "scans": scans,
    }

def summarize_product_details(details: Dict[str, Any], *, cal_center=None, match_radius=5*u.arcsec):
    d = details["details"]
    eb = d["execution_blocks"][0]

    out = {
        "sdm_id": eb.get("sdm_id"),
        "project_code": eb.get("project_code"),
        "instrument": eb.get("instrument_name"),
        "array_config": eb.get("configuration"),
        "band_code": eb.get("band_code"),
        "cal_status": eb.get("cal_status"),
        "has_caltables": bool(eb.get("cals")),
        "caltable_files": [c.get("file_name") for c in (eb.get("cals") or [])],
    }

    # targets present (from target_durs)
    targets = []
    for cfg in eb.get("configurations") or []:
        for td in cfg.get("target_durs") or []:
            targets.append((td.get("target_name"), td.get("ra"), td.get("dec"), td.get("duration")))
    out["targets"] = targets

    # does it contain calibrator by coord?
    if cal_center is not None and targets:
        from astropy.coordinates import SkyCoord
        coords = SkyCoord([t[1] for t in targets]*u.deg, [t[2] for t in targets]*u.deg, frame="icrs")
        sep = coords.separation(cal_center)
        out["min_target_sep_arcsec"] = float(sep.min().to_value(u.arcsec))
        out["contains_calibrator_by_coord"] = bool((sep <= match_radius).any())
    else:
        out["min_target_sep_arcsec"] = None
        out["contains_calibrator_by_coord"] = None

    scan_rows = eb.get("scan_rows") or []

    # calibrate scans (any calibration)
    out["calibrate_scans"] = [
        (s.get("scan_num"), s.get("target_name"), s.get("intent"))
        for s in scan_rows
        if s.get("intent") and "CALIBRATE" in (s.get("intent") or "").upper()
    ]

    # probable gain calibrator (reuse your earlier helper)
    gain_name, gain_info = _pick_gain_calibrator(scan_rows)
    out["probable_gain_calibrator"] = gain_name
    out["gain_calibrator_info"] = gain_info

    # NEW: add config + true on-source time from scan rows (no gaps)
    if gain_name:
        out["gain_calibrator_stats"] = _gain_calibrator_stats(
            scan_rows=scan_rows,
            eb=eb,
            gain_target=gain_name,
        )
    else:
        out["gain_calibrator_stats"] = None

    return out


def print_summary(summary):
    print("cal_status:", summary["cal_status"], "| has_caltables:", summary["has_caltables"])
    print("contains calibrator:", summary["contains_calibrator_by_coord"], "| min sep arcsec:", summary["min_target_sep_arcsec"])
    print("targets:", [t[0] for t in summary["targets"]])

    print("probable_gain_calibrator:", summary.get("probable_gain_calibrator"))

    gs = summary.get("gain_calibrator_stats")
    if gs:
        print(f"gain on-source time: {gs['total_on_source_min']:.2f} min ({gs['total_on_source_s']:.1f} s)")
        print("gain time by array config (min):", gs["array_config_time_min"])
        # show first few scans for sanity
        for s in gs["scans"][:5]:
            print(f"  scan {s['scan_num']}: {s['dur_s']:.1f}s | {s['array_config']} | {s['intent']}")
    else:
        print("gain_calibrator_stats: None")

    print("calibrate_scans:", summary["calibrate_scans"][:5])


# ONE
# details = fetch_product_details(df.loc[0, "access_url"])
# print(details.keys())


# POLITE
# import requests

# with requests.Session() as sess:
#     infos = []
#     for url in df["access_url"].dropna().unique():
#         infos.append(fetch_product_details(url, session=sess))

