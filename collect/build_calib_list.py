from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any


# -----------------------------
# Strict patterns (only what we expect)
# -----------------------------

HEADER_RE = re.compile(
    r"^(?P<name>\S+)\s+"
    r"(?P<frame>J2000|B1950)\s+"
    r"(?P<klass>[A-Z])\s+"
    r"(?P<ra>\S+)\s+"
    r"(?P<dec>\S+)"
    r"(?:\s+(?P<meta>.*))?$"
)

DASHES_RE = re.compile(r"^-{5,}\s*$")
BAND_HDR1_RE = re.compile(r"^BAND\s+")
BAND_HDR2_RE = re.compile(r"^=+\s*$")

# Strict start: wavelength + receiver, rest handled by token logic (no regex ambiguity)
BAND_ROW_START_RE = re.compile(
    r"^(?P<wavelength>\d+(?:\.\d+)?cm)\s+(?P<receiver>[A-Za-z])(?:\s+(?P<rest>.*))?$"
)

# Optional file preamble lines (only these)
PREAMBLE_ALLOWED = {
    "IAU NAME EQUINOX  PC RA(hh,mm,ss)    DEC(ddd,mm,ss)   POS.REF ALT.NAME",
    "===================================================================",
}


# -----------------------------
# Data
# -----------------------------

@dataclass
class Band:
    wavelength: str
    receiver: str
    config: Dict[str, Optional[str]]
    flux_jy: Optional[float]          # <-- flux can be missing or '?'
    uvmin_kl: Optional[float]
    uvmax_kl: Optional[float]
    notes: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wavelength": self.wavelength,
            "receiver": self.receiver,
            "config": self.config,
            "flux_jy": self.flux_jy,
            "uvmin_kl": self.uvmin_kl,
            "uvmax_kl": self.uvmax_kl,
            "notes": self.notes,
        }


@dataclass
class Source:
    name: str
    frame: str
    klass: str
    ra: str
    dec: str
    meta: Optional[str]
    bands: List[Band]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "frame": self.frame,
            "class": self.klass,
            "ra": self.ra,
            "dec": self.dec,
            "meta": self.meta,
            "bands": [b.to_dict() for b in self.bands],
        }


# -----------------------------
# Fail-fast parser
# -----------------------------

class ParseError(RuntimeError):
    pass


CFG_SET = set("PSWXC?")  # allowed config codes (case-insensitive in input)

_NUM_RE = re.compile(r"^\d+(?:\.\d+)?$")


def _parse_band_row_strict(line: str, i: int, err):
    """
    Deterministic, fail-fast band row parser.

    Grammar:
      wavelength receiver [0..4 config tokens] [flux] [uvmin] [uvmax] [notes...]

    Where:
      - config tokens: one-letter from {P,S,W,X,C,?} (case-insensitive)
      - flux: number OR "?" OR missing entirely
      - uvmin/uvmax: numbers, only allowed if flux is present (number or "?")
      - notes: anything remaining
    """
    m = BAND_ROW_START_RE.match(line.strip())
    if not m:
        err(i, "Band row did not start with '<wavelength> <receiver>'")

    wavelength = m.group("wavelength")
    receiver = m.group("receiver")
    rest = (m.group("rest") or "")
    toks = rest.split() if rest.strip() else []

    # 0..4 config tokens
    cfg: list[str] = []
    j = 0
    while j < len(toks) and len(cfg) < 4:
        t = toks[j]
        if len(t) == 1 and t.upper() in CFG_SET:
            cfg.append(t.upper())
            j += 1
        else:
            break

    config = {"A": None, "B": None, "C": None, "D": None}
    for k, v in zip(["A", "B", "C", "D"], cfg):
        if v == "?":
            v = None
        config[k] = v

    # flux: number or "?" (optional)
    flux: Optional[float] = None
    flux_present = False
    if j < len(toks):
        t = toks[j]
        if t == "?":
            flux = None
            flux_present = True
            j += 1
        elif _NUM_RE.match(t):
            flux = float(t)
            flux_present = True
            j += 1

    # uvmin/uvmax: numbers (optional) BUT only if flux token was present
    uvmin: Optional[float] = None
    uvmax: Optional[float] = None

    if j < len(toks) and _NUM_RE.match(toks[j]):
        if not flux_present:
            err(i, "Found uvmin/uvmax but missing flux (ambiguous band row).")
        uvmin = float(toks[j])
        j += 1

    if j < len(toks) and _NUM_RE.match(toks[j]):
        if not flux_present:
            err(i, "Found uvmin/uvmax but missing flux (ambiguous band row).")
        uvmax = float(toks[j])
        j += 1

    notes = " ".join(toks[j:]).strip() or None

    # Extra strictness: if there are non-empty tokens before flux that are not config tokens,
    # they would have been left in toks and then become "notes". That is allowed ONLY as notes,
    # not as "mystery config". This matches your "fail on weird formats" goal.
    # If you want to disallow notes entirely, set notes must be None and err otherwise.

    return wavelength, receiver, config, flux, uvmin, uvmax, notes


def parse_calibrators(path: str) -> List[Dict[str, Any]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()

    sources: List[Source] = []
    cur: Optional[Source] = None

    in_table = False
    seen_any_header = False

    def err(i: int, msg: str) -> None:
        ctx_lo = max(0, i - 3)
        ctx_hi = min(len(lines), i + 4)
        ctx = "\n".join(
            (">> " if j == i else "   ") + f"{j+1:5d}: {lines[j]}"
            for j in range(ctx_lo, ctx_hi)
        )
        raise ParseError(f"{msg}\n\nContext:\n{ctx}")

    for i, line in enumerate(lines):
        if line.strip() == "":
            continue  # blank lines allowed anywhere

        # Allow *only* known preamble before first header
        if not seen_any_header and line in PREAMBLE_ALLOWED:
            continue

        ls = line.strip()

        m_header = HEADER_RE.match(ls)
        if m_header:
            seen_any_header = True
            frame = m_header.group("frame")

            # Start a new source only on J2000 (treat B1950 as alias line we ignore)
            if frame == "J2000":
                cur = Source(
                    name=m_header.group("name"),
                    frame=frame,
                    klass=m_header.group("klass"),
                    ra=m_header.group("ra"),
                    dec=m_header.group("dec"),
                    meta=(m_header.group("meta") or "").strip() or None,
                    bands=[],
                )
                sources.append(cur)
                in_table = False
            else:
                if cur is None:
                    err(i, "B1950 header encountered before any J2000 source header.")
                in_table = False
            continue

        if not seen_any_header:
            err(i, "Non-preamble content found before first source header.")

        if cur is None:
            err(i, "Content found but no active source (missing header?).")

        if DASHES_RE.match(ls):
            in_table = False
            continue

        if BAND_HDR1_RE.match(ls) or BAND_HDR2_RE.match(ls):
            in_table = True
            continue

        if in_table:
            wavelength, receiver, config, flux, uvmin, uvmax, notes = _parse_band_row_strict(ls, i, err)

            band = Band(
                wavelength=wavelength,
                receiver=receiver,
                config=config,
                flux_jy=flux,
                uvmin_kl=uvmin,
                uvmax_kl=uvmax,
                notes=notes,
            )
            cur.bands.append(band)
            continue

        err(i, "Unexpected line outside of band table (unknown format).")

    return [s.to_dict() for s in sources]


if __name__ == "__main__":
    sources = parse_calibrators("collect/calibrators.txt")
    print("Num calibrators:", len(sources))
    Path("collect/vla_calibrators.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    print("Wrote collect/vla_calibrators.json")
