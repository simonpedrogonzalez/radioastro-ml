from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json
import random

import pandas as pd


# --- Canonical values for this catalog ---
DEFAULT_VALID_WAVELENGTHS = {
    "90cm", "20cm", "6cm", "3.7cm", "2cm", "1.3cm", "0.7cm"
}

# NOTE:
# - include "C" (it appears in the file)
# - include None (missing)
DEFAULT_VALID_CONFIG_CODES = {"P", "S", "W", "X", "C", None}

# sanity range (not a "catalog truth")
# keep strict default, but we special-case 90cm below
DEFAULT_FLUX_RANGE_JY = (0.0, 100.0)


@dataclass
class ValidationReport:
    ok: bool
    issues_df: pd.DataFrame
    bands_df: pd.DataFrame
    summary: Dict[str, Any]


def _flatten_to_bands_df(sources: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for s in sources:
        name = s.get("name")
        frame = s.get("frame")
        ra = s.get("ra")
        dec = s.get("dec")
        bands = s.get("bands") or []
        for i, b in enumerate(bands):
            cfg = b.get("config") or {}
            rows.append({
                "name": name,
                "frame": frame,
                "ra": ra,
                "dec": dec,
                "band_idx": i,
                "wavelength": b.get("wavelength"),
                "receiver": b.get("receiver"),
                "flux_jy": b.get("flux_jy"),
                "uvmin_kl": b.get("uvmin_kl"),
                "uvmax_kl": b.get("uvmax_kl"),
                "cfg_A": cfg.get("A"),
                "cfg_B": cfg.get("B"),
                "cfg_C": cfg.get("C"),
                "cfg_D": cfg.get("D"),
            })
    return pd.DataFrame(rows)


def validate_calibrator_catalog(
    sources: List[Dict[str, Any]],
    *,
    valid_wavelengths: Optional[set[str]] = None,
    valid_config_codes: Optional[set[Optional[str]]] = None,
    flux_range_jy: Tuple[float, float] = DEFAULT_FLUX_RANGE_JY,
    suspicious_uvmin_lt: float = 1.0,   # flag uvmin < 1 kλ (often a parse artifact)
    spotcheck_n: int = 10,
    seed: int = 0,
    save_issues_csv: Optional[str] = None,
    save_bands_csv: Optional[str] = None,
    # --- knobs ---
    normalize_7cm_to_6cm: bool = True,   # ignore the 6cm/7cm naming inconsistency
    ignore_zero_flux: bool = True,       # ignore 0.00 Jy "unknown/unusable" rows as errors
    flux_hi_90cm: float = 500.0,         # allow higher flux at 90cm specifically
) -> ValidationReport:
    """
    Validates the parsed VLA calibrator JSON list.

    Customizations:
      - 7cm is treated as 6cm (normalization) so it won't trigger invalid_wavelength.
      - flux_jy == 0.0 is treated as "unknown/unusable" and NOT an error.
      - config codes accept {P,S,W,X,C} plus missing (None/NaN).
      - flux sanity range is [lo, hi] except 90cm uses [lo, flux_hi_90cm].
    """
    valid_wavelengths = valid_wavelengths or DEFAULT_VALID_WAVELENGTHS
    valid_config_codes = valid_config_codes or DEFAULT_VALID_CONFIG_CODES

    issues: List[Dict[str, Any]] = []

    # Source-level checks
    for s in sources:
        if not s.get("name"):
            issues.append({"level": "source", "name": None, "issue": "missing_name", "detail": s})
        bands = s.get("bands")
        if not isinstance(bands, list) or len(bands) == 0:
            issues.append({"level": "source", "name": s.get("name"), "issue": "no_bands", "detail": None})

    bands_df = _flatten_to_bands_df(sources)

    # If empty, return early with issues
    if bands_df.empty:
        issues_df = pd.DataFrame(issues)
        summary = {"num_sources": len(sources), "num_band_rows": 0}
        return ValidationReport(ok=False, issues_df=issues_df, bands_df=bands_df, summary=summary)

    # Normalize wavelength naming issues (7cm -> 6cm)
    if normalize_7cm_to_6cm:
        bands_df["wavelength"] = bands_df["wavelength"].replace({
            "4cm": "3.7cm",
            "3.6cm": "3.7cm",
        })

    # Convenience mask for "unknown" flux rows
    is_zero_flux = bands_df["flux_jy"].notna() & (bands_df["flux_jy"] == 0.0)

    # Band-level required fields
    for col in ["name", "wavelength", "receiver"]:
        missing = bands_df[bands_df[col].isna()]
        for _, r in missing.iterrows():
            issues.append({
                "level": "band",
                "name": r.get("name"),
                "issue": f"missing_{col}",
                "detail": {
                    "wavelength": r.get("wavelength"),
                    "receiver": r.get("receiver"),
                    "flux_jy": r.get("flux_jy"),
                },
            })

    # Wavelength domain check (after normalization)
    bad_w = bands_df[~bands_df["wavelength"].isin(valid_wavelengths)]
    for _, r in bad_w.iterrows():
        issues.append({
            "level": "band",
            "name": r["name"],
            "issue": "invalid_wavelength",
            "detail": {"wavelength": r["wavelength"], "receiver": r["receiver"], "flux_jy": r["flux_jy"]},
        })

    # Flux sanity check
    # - ignore 0.0 Jy sentinel rows if requested
    # - allow a higher ceiling specifically for 90cm
    lo, hi = flux_range_jy
    hi_per_row = bands_df["wavelength"].map(lambda w: flux_hi_90cm if w == "90cm" else hi)

    bad_flux_mask = (
        bands_df["flux_jy"].notna()
        & (~is_zero_flux if ignore_zero_flux else True)
        & ((bands_df["flux_jy"] < lo) | (bands_df["flux_jy"] > hi_per_row))
    )
    bad_flux = bands_df[bad_flux_mask]
    for _, r in bad_flux.iterrows():
        expected_hi = flux_hi_90cm if r["wavelength"] == "90cm" else hi
        issues.append({
            "level": "band",
            "name": r["name"],
            "issue": "flux_out_of_range",
            "detail": {"wavelength": r["wavelength"], "flux_jy": r["flux_jy"], "expected": [lo, expected_hi]},
        })

    # UV sanity checks
    uvmin = bands_df["uvmin_kl"]
    uvmax = bands_df["uvmax_kl"]

    # uvmin > uvmax (both present)
    bad_uv_order = bands_df[uvmin.notna() & uvmax.notna() & (uvmin > uvmax)]
    for _, r in bad_uv_order.iterrows():
        issues.append({
            "level": "band",
            "name": r["name"],
            "issue": "uvmin_gt_uvmax",
            "detail": {"wavelength": r["wavelength"], "uvmin_kl": r["uvmin_kl"], "uvmax_kl": r["uvmax_kl"]},
        })

    # suspiciously small uvmin (often caused by glue like 0.7 from 0.7cm)
    bad_uv_small = bands_df[uvmin.notna() & (uvmin < suspicious_uvmin_lt)]
    for _, r in bad_uv_small.iterrows():
        issues.append({
            "level": "band",
            "name": r["name"],
            "issue": "suspicious_uvmin_small",
            "detail": {"wavelength": r["wavelength"], "uvmin_kl": r["uvmin_kl"], "threshold": suspicious_uvmin_lt},
        })

    # Config code sanity
    # Treat None/NaN as missing (OK). Validate only non-missing.
    cfg_cols = ["cfg_A", "cfg_B", "cfg_C", "cfg_D"]
    for _, r in bands_df.iterrows():
        for c in cfg_cols:
            v = r[c]
            if pd.isna(v):
                continue
            if v not in valid_config_codes:
                issues.append({
                    "level": "band",
                    "name": r["name"],
                    "issue": "invalid_config_code",
                    "detail": {"wavelength": r["wavelength"], "col": c, "value": v},
                })

    # Summary stats
    summary = {
        "num_sources": len(sources),
        "num_band_rows": int(len(bands_df)),
        "sources_with_no_bands": int(
            sum(1 for s in sources if not (isinstance(s.get("bands"), list) and len(s.get("bands")) > 0))
        ),
        "unique_sources_with_bands": int(bands_df["name"].nunique()),
        "num_zero_flux_rows": int(is_zero_flux.sum()),
        "normalize_7cm_to_6cm": bool(normalize_7cm_to_6cm),
        "ignore_zero_flux": bool(ignore_zero_flux),
        "flux_range_jy": [lo, hi],
        "flux_hi_90cm": float(flux_hi_90cm),
        "wavelength_counts": bands_df["wavelength"].value_counts(dropna=False).to_dict(),
        "receiver_counts": bands_df["receiver"].value_counts(dropna=False).to_dict(),
        "flux_by_wavelength_describe": (
            bands_df.groupby("wavelength")["flux_jy"]
            .describe()
            .reset_index()
            .to_dict(orient="records")
        ),
    }

    issues_df = (
        pd.DataFrame(issues).sort_values(["level", "issue", "name"], na_position="last")
        if issues else
        pd.DataFrame(columns=["level", "name", "issue", "detail"])
    )

    if save_issues_csv:
        issues_df.to_csv(save_issues_csv, index=False)
    if save_bands_csv:
        bands_df.to_csv(save_bands_csv, index=False)

    # Spot-check suggestions
    rng = random.Random(seed)
    spot_names = rng.sample(
        list(bands_df["name"].unique()),
        k=min(spotcheck_n, bands_df["name"].nunique())
    )
    summary["spotcheck_names"] = spot_names

    ok = len(issues_df) == 0
    return ValidationReport(ok=ok, issues_df=issues_df, bands_df=bands_df, summary=summary)


# ---- Example usage ----
if __name__ == "__main__":
    with open("collect/vla_calibrators.json") as f:
        sources = json.load(f)

    report = validate_calibrator_catalog(
        sources,
        suspicious_uvmin_lt=1.0,
        spotcheck_n=10,
        seed=0,
        save_issues_csv="collect/vla_calibrators_issues.csv",
        save_bands_csv="collect/vla_calibrators_bands.csv",
        normalize_7cm_to_6cm=True,
        ignore_zero_flux=True,
        flux_hi_90cm=500.0,
    )

    print("OK:", report.ok)
    print("Issues:", len(report.issues_df))
    print("Zero-flux rows (ignored):", report.summary["num_zero_flux_rows"])
    print("Suggested spot-check names:", report.summary["spotcheck_names"])

    if len(report.issues_df):
        print(report.issues_df.head(30).to_string(index=False))

    # Example: filter candidates (e.g., strong, compact-ish at 6cm in A config)
    df = report.bands_df
    candidates = df[
        (df["wavelength"] == "6cm") &
        (df["flux_jy"] >= 0.5) &
        (df["cfg_A"].isin(["S", "P"]))
    ].sort_values("flux_jy", ascending=False)

    print("\nTop 20 candidates (6cm, flux>=0.5, A in {S,P}):")
    print(candidates.head(20)[["name", "wavelength", "receiver", "flux_jy", "cfg_A", "cfg_B", "cfg_C", "cfg_D"]].to_string(index=False))
