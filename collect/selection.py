from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd


# -----------------------------
# USER CONTROLS
# -----------------------------

INPUT_JSON = "collect/vla_calibrators.json"
OUTPUT_JSON = "collect/vla_calibrators_selected.json"
OUTPUT_CSV  = "collect/vla_calibrators_selected.csv"

REFERENCE_WAVELENGTH = "6cm"
MIN_FLUX_JY = 0.5

# Names you want to exclude no matter what

weird = ['0455-205',
'1220+292',
'1337-129',
'1733-130',
'2023+318',
'1748+700',
'1118-465',
'0354+801',
'1914+166',
'0632+159',
'1300+142']

# What we accept as "compact enough"
GOOD_CFG = {"P", "S", None}


# -----------------------------
# HELPERS
# -----------------------------

def is_probably_point_source(cfg):
    return cfg.get("A") in {"P", "S"}   # or GOOD_CFG


# -----------------------------
# MAIN
# -----------------------------

def select_calibrators(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected = []

    for s in sources:
        name = s.get("name")
        if not name or name in weird:
            continue

        bands = s.get("bands") or []

        # Find the reference band
        band = next(
            (b for b in bands if b.get("wavelength") == REFERENCE_WAVELENGTH),
            None,
        )
        if band is None:
            continue

        flux = band.get("flux_jy")
        if flux is None or flux < MIN_FLUX_JY:
            continue

        cfg = band.get("config") or {}
        if not is_probably_point_source(cfg):
            continue

        # Passed all cuts
        selected.append({
            "name": name,
            "ra": s.get("ra"),
            "dec": s.get("dec"),
            "wavelength": band.get("wavelength"),
            "receiver": band.get("receiver"),
            "flux_jy": flux,
            "cfg_A": cfg.get("A"),
            "cfg_B": cfg.get("B"),
            "cfg_C": cfg.get("C"),
            "cfg_D": cfg.get("D"),
        })

    return selected


if __name__ == "__main__":
    sources = json.loads(Path(INPUT_JSON).read_text())
    picked = select_calibrators(sources)

    print(f"Selected {len(picked)} calibrators")

    Path(OUTPUT_JSON).write_text(json.dumps(picked, indent=2))
    pd.DataFrame(picked).to_csv(OUTPUT_CSV, index=False)

    print(f"Wrote:\n  {OUTPUT_JSON}\n  {OUTPUT_CSV}")
