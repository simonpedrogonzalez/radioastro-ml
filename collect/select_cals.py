from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

# -----------------------------
# USER CONTROLS
# -----------------------------
INPUT_JSON = "collect/vla_calibrators.json"
OUTPUT_CSV = "collect/vla_calibrators_selected.csv"

MIN_FLUX_JY = 0.5
REFERENCE_WAVELENGTH = "6cm"  # optional: used only for the flux cut below

weird = [
    "0455-205", "1220+292", "1337-129", "1733-130", "2023+318", "1748+700",
    "1118-465", "0354+801", "1914+166", "0632+159", "1300+142",
]

# "usable" means: safe to treat as point-like for *amplitude* calibration
# based on the NRAO definitions: keep only "P"
USABLE_CODES = {"P"}  # if you want P or S, change to {"P", "S"}

CFG_ORDER = ["A", "B", "C", "D"]

def is_usable(code: Any) -> bool:
    return code in USABLE_CODES

def cfgs_to_str(ok: Dict[str, bool]) -> str:
    cfgs = [c for c in CFG_ORDER if ok.get(c)]
    return ",".join(cfgs) if cfgs else ""

def get_band_by_wavelength(bands: List[Dict[str, Any]], wl: str) -> Dict[str, Any] | None:
    return next((b for b in bands if b.get("wavelength") == wl), None)

def build_rows(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for s in sources:
        name = s.get("name")
        if not name or name in weird:
            continue

        bands = s.get("bands") or []

        # Optional: require the source to have the reference wavelength and pass flux cut there
        ref = get_band_by_wavelength(bands, REFERENCE_WAVELENGTH)
        if ref is None:
            continue
        ref_flux = ref.get("flux_jy")
        if ref_flux is None or ref_flux < MIN_FLUX_JY:
            continue

        # Emit ONE ROW PER BAND for this source
        for band in bands:
            wl = band.get("wavelength")
            rx = band.get("receiver")
            flux = band.get("flux_jy")

            cfg = band.get("config") or {}
            codes = {c: cfg.get(c) for c in CFG_ORDER}
            usable = {c: is_usable(codes[c]) for c in CFG_ORDER}

            usable_configs = cfgs_to_str(usable)

            rows.append({
                "name": name,
                "ra": s.get("ra"),
                "dec": s.get("dec"),
                "wavelength": wl,
                "receiver": rx,
                "flux_jy": flux,
                "usable_configs": usable_configs,
                "usable_A": usable["A"],
                "usable_B": usable["B"],
                "usable_C": usable["C"],
                "usable_D": usable["D"],
                "cfg_A": codes["A"],
                "cfg_B": codes["B"],
                "cfg_C": codes["C"],
                "cfg_D": codes["D"],
            })

    return rows

if __name__ == "__main__":
    sources = json.loads(Path(INPUT_JSON).read_text())
    rows = build_rows(sources)

    df = pd.DataFrame(rows)
    # (optional) stable sort similar to your snippet
    df = df.sort_values(["name", "wavelength"], kind="stable")
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Wrote {len(df)} rows to {OUTPUT_CSV}")
