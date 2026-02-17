from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterable, Union, Any

import pandas as pd
import pyvo
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord

# Optional: only needed if you want name -> coords via SIMBAD
from astroquery.simbad import Simbad


# =============================================================================
# Core constants / helpers
# =============================================================================

TAP_URL = "https://data-query.nrao.edu/tap"


def date_to_mjd(date_str: str) -> float:
    """'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' -> MJD (UTC)."""
    return Time(date_str, format="iso", scale="utc").mjd


def mjd_to_iso(mjd: float) -> str:
    """MJD -> 'YYYY-MM-DD HH:MM:SS' (UTC)."""
    return Time(mjd, format="mjd", scale="utc").iso


def _as_mjd(x: Union[str, float, int, None]) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (float, int)):
        return float(x)
    return float(date_to_mjd(str(x)))


def find_obscore_table(svc: pyvo.dal.TAPService) -> str:
    """Detect fully-qualified ObsCore table name for this TAP service."""
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
            return name

    candidates = sorted([n for n in names if "obscore" in n.lower()])
    if candidates:
        return candidates[0]

    raise RuntimeError(f"No ObsCore table found. Sample tables: {sorted(names)[:30]}")


def get_obscore_columns(svc: pyvo.dal.TAPService, obscore_table: str) -> list[str]:
    q = f"""
    SELECT column_name
    FROM TAP_SCHEMA.columns
    WHERE table_name = '{obscore_table}'
    ORDER BY column_name
    """
    t = svc.run_sync(q).to_table()
    return [str(x) for x in t["column_name"]]


def _pretty_format_query(q: str) -> str:
    """
    Make ADQL query more human-readable:
    - SELECT, FROM, WHERE, ORDER BY on separate lines
    - Each AND condition on its own line
    """
    q = q.strip()

    # Normalize spacing first
    q = q.replace(" WHERE ", "\nWHERE ")
    q = q.replace(" FROM ", "\nFROM ")
    q = q.replace(" ORDER BY ", "\nORDER BY ")

    if "WHERE " in q:
        head, where_part = q.split("WHERE ", 1)

        if "ORDER BY" in where_part:
            where_body, order_part = where_part.split("ORDER BY", 1)
            order_part = "ORDER BY " + order_part
        else:
            where_body = where_part
            order_part = ""

        # Split AND conditions
        conditions = where_body.strip().split(" AND ")

        where_block = "WHERE\n    " + "\n    AND ".join(c.strip() for c in conditions)

        q = head.strip() + "\n" + where_block
        if order_part:
            q += "\n" + order_part.strip()

    return q


# =============================================================================
# SIMBAD name -> coordinates (robust name lookup)
# =============================================================================

@dataclass
class SimbadResult:
    coords: Optional[SkyCoord] = None
    table: Any = None  # astropy.table.Table typically


def simbad_search_object(identifier: str, *, cols: Optional[list[str]] = None, **kwargs) -> SimbadResult:
    """
    Search object name in SIMBAD and return coordinates (ICRS).
    Useful when archive target_name is messy.
    NOTE: DOESNT WORK FOR MOST RADIOASTRO NAMES >:'(
    """
    simbad = Simbad()
    if cols is None:
        cols = ["coordinates", "otype"]
    simbad.add_votable_fields(*cols)

    table = simbad.query_object(identifier, **kwargs)
    if table is None:
        return SimbadResult(coords=None, table=None)

    # SIMBAD returns RA/DEC strings
    try:
        ra_str = str(table["RA"][0])
        dec_str = str(table["DEC"][0])
        coords = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg), frame="icrs")
    except Exception:
        coords = None

    return SimbadResult(coords=coords, table=table)


# =============================================================================
# Registry objects: TIME SPANS, INSTRUMENTS, DATAPRODS, CONFIGS, BANDS, COLS
# =============================================================================

@dataclass(frozen=True)
class TimeSpan:
    start_mjd: Optional[float] = None
    end_mjd: Optional[float] = None  # None => open-ended


class TIMESPANS:
    # 2012-01-01 -> now (open-ended)
    FROM_2012 = TimeSpan(start_mjd=date_to_mjd("2012-01-01"), end_mjd=None)
    # 2016-09-01 -> now (open-ended) (often used for calibrated-MS availability assumptions)
    FROM_2016_SEP = TimeSpan(start_mjd=date_to_mjd("2016-09-01"), end_mjd=None)


class PROPRIETARY:
    """
    Values for ObsCore 'proprietary_status' (NRAO TAP).
    Commonly used: 'PUBLIC'. Others may appear depending on archive rules.
    """
    PUBLIC = "PUBLIC"
    PROPRIETARY = "PROPRIETARY"

    # Sometimes archives use variations; keep these aliases just in case
    PRIVATE = "PRIVATE"
    UNKNOWN = "UNKNOWN"

    @staticmethod
    def any_public() -> list[str]:
        """Statuses you might treat as publicly accessible."""
        return ["PUBLIC"]


class INSTRUMENTS:
    # NRAO doc mentions these appear in instrument_name
    VLA = "VLA"
    EVLA = "EVLA"
    ALMA = "ALMA"
    GBT = "GBT"
    VLBA = "VLBA"
    GMVA = "GMVA"

    @staticmethod
    def VLA_VARIANTS() -> list[str]:
        return ["VLA", "EVLA"]


class DATAPRODS:
    VISIBILITY = "visibility"
    IMAGE = "image"


class CONFIGS:
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    ALL = ["A", "B", "C", "D"]


@dataclass(frozen=True)
class Band:
    key: str          # e.g., "C", "X", "L", ...
    f_lo_hz: float
    f_hi_hz: float
    label: str        # e.g., "~6cm", "~3.7cm"


class BANDS:
    # VLA-ish band ranges (good enough for overlap filtering)
    L  = Band("L",  1e9,     2e9,     "~20cm")
    S  = Band("S",  2e9,     4e9,     "~13cm")
    C  = Band("C",  4e9,     8e9,     "~6cm")
    X  = Band("X",  8e9,     12e9,    "~3.7cm")
    KU = Band("KU", 12e9,    18e9,    "~2cm")
    K  = Band("K",  18e9,    26.5e9,  "~1.3cm")
    KA = Band("KA", 26.5e9,  40e9,    "~9mm")
    Q  = Band("Q",  40e9,    50e9,    "~7mm")

    _BY_KEY = {b.key: b for b in [L, S, C, X, KU, K, KA, Q]}

    @classmethod
    def get(cls, x: str) -> Band:
        """
        Accepts:
          - receiver key: 'C', 'X', 'L', 'Ku', 'Ka', ...
          - wavelength string: '6cm', '20cm', '2cm', '9mm', '7mm' (best-effort)
        """
        s = str(x).strip().upper()

        # receiver keys
        s_norm = s.replace(" ", "")
        if s_norm in cls._BY_KEY:
            return cls._BY_KEY[s_norm]

        # wavelength hints (best-effort mapping)
        wl = s.lower().replace(" ", "")
        if wl.endswith("cm"):
            cm = float(wl.replace("cm", ""))
            # very rough mapping to common VLA labels
            if 15 <= cm <= 30:
                return cls.L
            if 9 <= cm <= 15:
                return cls.S
            if 4.5 <= cm <= 7.5:
                return cls.C
            if 3.0 <= cm <= 4.5:
                return cls.X
            if 1.6 <= cm <= 2.6:
                return cls.KU
            if 1.0 <= cm <= 1.6:
                return cls.K
        if wl.endswith("mm"):
            mm = float(wl.replace("mm", ""))
            if 8 <= mm <= 12:
                return cls.KA
            if 6 <= mm <= 8:
                return cls.Q

        raise ValueError(f"Could not infer band from {x!r}")


class COLS:
    """
    Convenience namespace for columns. This is intentionally “stringy”.
    You can still SELECT '*' or any string.
    """
    obs_publisher_did = "obs_publisher_did"
    project_code = "project_code"
    target_name = "target_name"
    t_min = "t_min"
    t_max = "t_max"
    t_exptime = "t_exptime"
    instrument_name = "instrument_name"
    configuration = "configuration"
    dataproduct_type = "dataproduct_type"
    freq_min = "freq_min"
    freq_max = "freq_max"
    em_min = "em_min"
    em_max = "em_max"
    s_ra = "s_ra"
    s_dec = "s_dec"
    access_estsize = "access_estsize"
    access_url = "access_url"
    access_format = "access_format"
    proprietary_status = "proprietary_status"
    pol_states = "pol_states"
    calib_level = "calib_level"
    # new-ish (May 2024 per NRAO doc) – may or may not exist depending on table
    num_antennas = "num_antennas"
    max_uv_dist = "max_uv_dist"
    spw_names = "spw_names"
    center_frequencies = "center_frequencies"
    bandwidths = "bandwidths"
    nums_channels = "nums_channels"
    spectral_resolutions = "spectral_resolutions"
    aggregate_bandwidth = "aggregate_bandwidth"


# =============================================================================
# Query builder
# =============================================================================

@dataclass
class NRAOQuery:
    """
    Minimal fluent query builder for NRAO TAP ObsCore.
    - Uses TAP_URL and auto-detects ObsCore table on init.
    - Selects all columns by default (SELECT *).
    """
    limit: int = 10

    def __post_init__(self):
        self.svc = pyvo.dal.TAPService(TAP_URL)
        self.table = find_obscore_table(self.svc)
        self.available_cols = set(get_obscore_columns(self.svc, self.table))
        self._select: list[str] = ["*"]
        self._where: list[str] = []
        self._order_by: Optional[str] = None

    # --- selection / output controls ---

    def select(self, *cols: str) -> "NRAOQuery":
        self._select = list(cols) if cols else ["*"]
        return self

    def top(self, n: int) -> "NRAOQuery":
        self.limit = int(n)
        return self

    def order_by(self, expr: str) -> "NRAOQuery":
        self._order_by = expr
        return self

    # --- filters ---

    def where_timespan(self, span: TimeSpan) -> "NRAOQuery":
        if span.start_mjd is not None and "t_min" in self.available_cols:
            self._where.append(f"(t_min >= {float(span.start_mjd)})")
        if span.end_mjd is not None and "t_min" in self.available_cols:
            self._where.append(f"(t_min < {float(span.end_mjd)})")
        return self

    def where_dates(
        self,
        *,
        start: Union[str, float, int, None] = None,
        end: Union[str, float, int, None] = None,
    ) -> "NRAOQuery":
        """
        start/end can be MJD (float/int) or date strings like 'YYYY-MM-DD'.
        Applies to t_min (start time of observation).
        """
        s = _as_mjd(start)
        e = _as_mjd(end)
        if s is not None and "t_min" in self.available_cols:
            self._where.append(f"(t_min >= {s})")
        if e is not None and "t_min" in self.available_cols:
            self._where.append(f"(t_min < {e})")
        return self

    def where_in_circle(
        self,
        center: Union[SkyCoord, tuple[float, float], str],
        radius: Union[float, int, u.Quantity],
    ) -> "NRAOQuery":
        """
        center:
          - SkyCoord
          - (ra_deg, dec_deg) tuple
          - string name (resolved via SIMBAD)
        radius:
          - degrees (float/int) or astropy Quantity (e.g. 20*u.arcsec)
        """
        if isinstance(radius, u.Quantity):
            rad_deg = radius.to_value(u.deg)
        else:
            rad_deg = float(radius)

        if isinstance(center, str):
            sr = simbad_search_object(center)
            if sr.coords is None:
                raise ValueError(f"SIMBAD could not resolve {center!r}")
            c = sr.coords
        elif isinstance(center, tuple):
            c = SkyCoord(center[0] * u.deg, center[1] * u.deg, frame="icrs")
        else:
            c = center

        if "s_ra" not in self.available_cols or "s_dec" not in self.available_cols:
            raise RuntimeError("ObsCore table does not expose s_ra/s_dec; cannot do cone search")

        self._where.append(
            "CONTAINS("
            "POINT('ICRS', s_ra, s_dec), "
            f"CIRCLE('ICRS', {c.ra.deg}, {c.dec.deg}, {rad_deg})"
            ") = 1"
        )
        return self

    def where_instruments(self, instruments: Iterable[str]) -> "NRAOQuery":
        if "instrument_name" in self.available_cols:
            items = ",".join([f"'{str(x)}'" for x in instruments])
            self._where.append(f"(instrument_name IN ({items}))")
        return self

    def where_dataproduct(self, dataproduct_type: str) -> "NRAOQuery":
        if "dataproduct_type" in self.available_cols:
            self._where.append(f"(dataproduct_type = '{dataproduct_type}')")
        return self

    def where_band(self, band: Band) -> "NRAOQuery":
        """
        Filter by frequency overlap with band [f_lo, f_hi].
        Requires freq_min/freq_max.
        """
        if "freq_min" in self.available_cols and "freq_max" in self.available_cols:
            self._where.append(f"(freq_max >= {band.f_lo_hz} AND freq_min <= {band.f_hi_hz})")
        return self

    def where_configs(self, configs: Iterable[str]) -> "NRAOQuery":
        if "configuration" in self.available_cols:
            items = ",".join([f"'{str(x)}'" for x in configs])
            self._where.append(f"(configuration IN ({items}))")
        return self

    def where_proprietary_status(self, status: str) -> "NRAOQuery":
        if "proprietary_status" in self.available_cols:
            self._where.append(f"(proprietary_status = '{status}')")
        return self

    # --- build / execute ---

    def build(self) -> str:
        select_list = ", ".join(self._select) if self._select else "*"
        where_clause = ""
        if self._where:
            where_clause = "WHERE " + " AND ".join(self._where)
        order_clause = f"ORDER BY {self._order_by}" if self._order_by else ""
        return f"SELECT TOP {int(self.limit)} {select_list} FROM {self.table} {where_clause} {order_clause}"

    def get(self) -> pd.DataFrame:
        q = self.build()
        
        print("\n" + "=" * 80)
        print("Executing ADQL query:")
        print("-" * 80)
        print(_pretty_format_query(q))
        print("=" * 80 + "\n")
        
        return self.svc.run_sync(q).to_table().to_pandas()


# =============================================================================
# Example usage
# =============================================================================

# if __name__ == "__main__":
    # Example 1: query by coordinates + band + VLA variants + visibility + configs + timespan
    # q = (
    #     NRAOQuery(limit=10)
    #     .select(
    #         COLS.obs_publisher_did,
    #         COLS.project_code,
    #         COLS.target_name,
    #         COLS.t_min,
    #         COLS.configuration,
    #         COLS.instrument_name,
    #         COLS.freq_min,
    #         COLS.freq_max,
    #         COLS.proprietary_status,
    #     )
    #     .where_timespan(TIMESPANS.FROM_2016_SEP)
    #     .where_in_circle(("1.488230870833333", "38.33754126944444"), 20 * u.arcsec)  # tuple is (ra_deg, dec_deg) if numeric
    #     .where_instruments(INSTRUMENTS.VLA_VARIANTS())
    #     .where_dataproduct(DATAPRODS.VISIBILITY)
    #     .where_configs([CONFIGS.A, CONFIGS.B, CONFIGS.C, CONFIGS.D])
    #     .where_band(BANDS.C)
    #     .order_by("t_min DESC")
    # )
    # print(q.build())
    # df = q.get()
    # print(df.head())

    # Example 2: query by SIMBAD name
    # q2 = (
    #     NRAOQuery(limit=5)
    #     .where_timespan(TIMESPANS.FROM_2012)
    #     .where_in_circle("3C286", 30 * u.arcsec)
    #     .where_instruments(INSTRUMENTS.VLA_VARIANTS())
    #     .where_dataproduct(DATAPRODS.VISIBILITY)
    #     .where_band(BANDS.C)
    # )
    # print(q2.build())
