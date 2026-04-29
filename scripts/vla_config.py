from __future__ import annotations

import math
import re
from typing import Iterable


vla_config_properties = {
    "B_max_km": {
        "A": 36.4,
        "B": 11.1,
        "C": 3.4,
        "D": 1.03,
    },
    "B_min_km": {
        "A": 0.68,
        "B": 0.21,
        "C": 0.0355,
        "D": 0.035,
    },
    "synthesized_beamwidth_arcsec": {
        "74 MHz":   {"A": 24.0,   "B": 80.0,  "C": 260.0, "D": 850.0},
        "350 MHz":  {"A": 5.6,    "B": 18.5,  "C": 60.0,  "D": 200.0},
        "1.5 GHz":  {"A": 1.3,    "B": 4.3,   "C": 14.0,  "D": 46.0},
        "3.0 GHz":  {"A": 0.65,   "B": 2.1,   "C": 7.0,   "D": 23.0},
        "6.0 GHz":  {"A": 0.33,   "B": 1.0,   "C": 3.5,   "D": 12.0},
        "10 GHz":   {"A": 0.20,   "B": 0.60,  "C": 2.1,   "D": 7.2},
        "15 GHz":   {"A": 0.13,   "B": 0.42,  "C": 1.4,   "D": 4.6},
        "22 GHz":   {"A": 0.089,  "B": 0.28,  "C": 0.95,  "D": 3.1},
        "33 GHz":   {"A": 0.059,  "B": 0.19,  "C": 0.63,  "D": 2.1},
        "45 GHz":   {"A": 0.043,  "B": 0.14,  "C": 0.47,  "D": 1.5},
    },
    "largest_angular_scale_arcsec": {
        "74 MHz":   {"A": 800.0, "B": 2200.0, "C": 20000.0, "D": 20000.0},
        "350 MHz":  {"A": 155.0, "B": 515.0,  "C": 4150.0,  "D": 4150.0},
        "1.5 GHz":  {"A": 36.0,  "B": 120.0,  "C": 970.0,   "D": 970.0},
        "3.0 GHz":  {"A": 18.0,  "B": 58.0,   "C": 490.0,   "D": 490.0},
        "6.0 GHz":  {"A": 8.9,   "B": 29.0,   "C": 240.0,   "D": 240.0},
        "10 GHz":   {"A": 5.3,   "B": 17.0,   "C": 145.0,   "D": 145.0},
        "15 GHz":   {"A": 3.6,   "B": 12.0,   "C": 97.0,    "D": 97.0},
        "22 GHz":   {"A": 2.4,   "B": 7.9,    "C": 66.0,    "D": 66.0},
        "33 GHz":   {"A": 1.6,   "B": 5.3,    "C": 44.0,    "D": 44.0},
        "45 GHz":   {"A": 1.2,   "B": 3.9,    "C": 32.0,    "D": 32.0},
    },
}


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


def _normalize_config(config: str | None) -> str | None:
    if config is None:
        return None
    cfg = str(config).strip().upper()
    return cfg if cfg in {"A", "B", "C", "D"} else None


def split_band_codes(band_code: str | None) -> list[str]:
    if band_code is None:
        return []
    parts = re.split(r"[\s,/;+|]+", str(band_code).upper().strip())
    return [part for part in parts if part in _BAND_RANGES_GHZ]


def band_for_frequency_ghz(freq_ghz: float | None) -> str | None:
    if freq_ghz is None or not math.isfinite(freq_ghz):
        return None
    for band, (lo, hi) in _BAND_RANGES_GHZ.items():
        if lo <= freq_ghz <= hi:
            return band
    return None


def band_matches_frequency(freq_ghz: float | None, bands: Iterable[str]) -> bool:
    detected = band_for_frequency_ghz(freq_ghz)
    if detected is None:
        return False
    return detected in {str(b).upper() for b in bands}


def representative_frequency_for_band_ghz(band: str | None) -> float | None:
    codes = split_band_codes(band)
    if not codes:
        return None
    lo, hi = _BAND_RANGES_GHZ[codes[0]]
    return 0.5 * (lo + hi)


def _reference_beam_table() -> list[tuple[float, dict[str, float]]]:
    table = []
    for label, beams in vla_config_properties["synthesized_beamwidth_arcsec"].items():
        freq_ghz = float(label.split()[0])
        if "MHz" in label:
            freq_ghz /= 1000.0
        table.append((freq_ghz, beams))
    table.sort(key=lambda item: item[0])
    return table


def estimate_synthesized_beam_arcsec(
    config: str | None,
    freq_ghz: float | None,
) -> float | None:
    cfg = _normalize_config(config)
    if cfg is None or freq_ghz is None or not math.isfinite(freq_ghz) or freq_ghz <= 0:
        return None

    refs = _reference_beam_table()
    if not refs:
        return None

    if freq_ghz <= refs[0][0]:
        ref_freq, ref_beams = refs[0]
        return ref_beams[cfg] * (ref_freq / freq_ghz)

    if freq_ghz >= refs[-1][0]:
        ref_freq, ref_beams = refs[-1]
        return ref_beams[cfg] * (ref_freq / freq_ghz)

    for (f0, beams0), (f1, beams1) in zip(refs, refs[1:]):
        if f0 <= freq_ghz <= f1:
            b0 = float(beams0[cfg])
            b1 = float(beams1[cfg])
            logf = math.log(freq_ghz)
            t = (logf - math.log(f0)) / (math.log(f1) - math.log(f0))
            return math.exp(math.log(b0) + t * (math.log(b1) - math.log(b0)))

    return None
