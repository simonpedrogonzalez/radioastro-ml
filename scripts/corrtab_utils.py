from __future__ import annotations
from casatools import table, simulator
import numpy as np
from casatasks import rmtables
from typing import Optional, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple, Union



FilterSpec = Tuple[str, str, Any]  # (op, col, value)



def get_unflagged_antennas(msname: str):
    """
    Returns:
        name_to_idx: dict[str, int]
        idx_to_name: dict[int, str]
    Only antennas that appear in unflagged visibilities are included.
    """

    tb = table()

    # --- 1) Read antenna names ---
    tb.open(f"{msname}/ANTENNA")
    names = np.asarray(tb.getcol("NAME"))
    tb.close()

    # --- 2) Find antennas used in unflagged rows ---
    tb.open(msname)

    ant1 = np.asarray(tb.getcol("ANTENNA1"), dtype=np.int32)
    ant2 = np.asarray(tb.getcol("ANTENNA2"), dtype=np.int32)

    # FLAG shape: (nCorr, nChan, nRow)
    flag = np.asarray(tb.getcol("FLAG"))

    tb.close()

    # A row is "usable" if not fully flagged across corr+chan
    # shape reduction over corr+chan
    fully_flagged = np.all(flag, axis=(0, 1))
    good_rows = ~fully_flagged

    # Collect antennas appearing in good rows
    used_ants = set(ant1[good_rows]) | set(ant2[good_rows])
    used_ants = sorted(int(a) for a in used_ants)

    # --- 3) Build mappings ---
    name_to_idx = {str(names[i]): int(i) for i in used_ants}
    idx_to_name = {int(i): str(names[i]) for i in used_ants}

    return name_to_idx, idx_to_name

class GCOLS:
    TIME = "TIME"
    FIELD_ID = "FIELD_ID"
    SPECTRAL_WINDOW_ID = "SPECTRAL_WINDOW_ID"
    ANTENNA1 = "ANTENNA1"
    ANTENNA2 = "ANTENNA2"
    INTERVAL = "INTERVAL"
    SCAN_NUMBER = "SCAN_NUMBER"
    OBSERVATION_ID = "OBSERVATION_ID"
    CPARAM = "CPARAM"
    PARAMERR = "PARAMERR"
    FLAG = "FLAG"
    SNR = "SNR"
    WEIGHT = "WEIGHT"

from dataclasses import dataclass
import numpy as np

@dataclass
class GTab:
    ROWID: np.ndarray
    TIME: np.ndarray
    FIELD_ID: np.ndarray
    SPECTRAL_WINDOW_ID: np.ndarray
    ANTENNA1: np.ndarray
    ANTENNA2: np.ndarray
    INTERVAL: np.ndarray
    SCAN_NUMBER: np.ndarray
    OBSERVATION_ID: np.ndarray
    # optional columns (can be None if not readable)
    CPARAM: np.ndarray | None = None
    PARAMERR: np.ndarray | None = None
    FLAG: np.ndarray | None = None
    SNR: np.ndarray | None = None
    WEIGHT: np.ndarray | None = None

    @property
    def nrow(self) -> int:
        return int(self.TIME.shape[0])

    def col(self, name: str) -> np.ndarray:
        v = getattr(self, name)
        if v is None:
            raise KeyError(f"Column '{name}' not loaded / unavailable in this table")
        return v

    @classmethod
    def from_casa_table(cls, tb, *, load_optional: bool = False) -> "GTab":
        nrow = tb.nrows()
        rid = np.arange(nrow, dtype=np.int64)

        def g(name):
            return np.asarray(tb.getcol(name))

        out = cls(
            ROWID=rid,
            TIME=g("TIME").astype(float),
            FIELD_ID=g("FIELD_ID").astype(np.int32),
            SPECTRAL_WINDOW_ID=g("SPECTRAL_WINDOW_ID").astype(np.int32),
            ANTENNA1=g("ANTENNA1").astype(np.int32),
            ANTENNA2=g("ANTENNA2").astype(np.int32),
            INTERVAL=g("INTERVAL").astype(float),
            SCAN_NUMBER=g("SCAN_NUMBER").astype(np.int32),
            OBSERVATION_ID=g("OBSERVATION_ID").astype(np.int32),
        )

        if load_optional:
            # best-effort loads; if a column is ragged/missing, keep None
            for name in ["CPARAM", "PARAMERR", "FLAG", "SNR", "WEIGHT"]:
                try:
                    setattr(out, name, np.asarray(tb.getcol(name)))
                except Exception:
                    setattr(out, name, None)

        return out

    def take_rows(self, rows: np.ndarray) -> "GTab":
        """
        Return a new GTab containing only the selected rows.

        Notes
        -----
        - Per-row scalar columns (TIME, ANTENNA1, etc.) are shape (nRow,) and are sliced with x[rows].
        - CPARAM-like columns are usually shape (..., nRow) with nRow on the last axis, so we slice with x[..., rows].
        - Optional columns that are None stay None.
        """
        rows = np.asarray(rows, dtype=np.int64)

        def take1(x: np.ndarray) -> np.ndarray:
            return np.asarray(x)[rows]

        def takelast(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x)
            if x.ndim == 0:
                return x
            # In CASA gain tables, row dimension for these is typically the last axis
            if x.shape[-1] != self.nrow:
                raise ValueError(f"Expected last dim == nrow ({self.nrow}), got shape {x.shape}")
            return x[..., rows]

        return GTab(
            ROWID=take1(self.ROWID),
            TIME=take1(self.TIME),
            FIELD_ID=take1(self.FIELD_ID),
            SPECTRAL_WINDOW_ID=take1(self.SPECTRAL_WINDOW_ID),
            ANTENNA1=take1(self.ANTENNA1),
            ANTENNA2=take1(self.ANTENNA2),
            INTERVAL=take1(self.INTERVAL),
            SCAN_NUMBER=take1(self.SCAN_NUMBER),
            OBSERVATION_ID=take1(self.OBSERVATION_ID),

            CPARAM=None if self.CPARAM is None else takelast(self.CPARAM),
            PARAMERR=None if self.PARAMERR is None else takelast(self.PARAMERR),
            FLAG=None if self.FLAG is None else takelast(self.FLAG),
            SNR=None if self.SNR is None else takelast(self.SNR),
            WEIGHT=None if self.WEIGHT is None else takelast(self.WEIGHT),
        )

class GTabQuery:
    """
    Lightweight query builder for GTab:
      - where_in(col, values)
      - where_eq(col, value)
      - where_between(col, lo, hi)
      - group_by(cols)
      - sort_by(cols, ascending=True|list[bool])          <-- added
      - get_indices(tab)
      - apply(tab): returns GTab or dict[key, GTab] if grouped
    """
    def __init__(self):
        self._filters: List[FilterSpec] = []
        self._group_cols: Tuple[str, ...] | None = None

        # sorting
        self._sort_cols: Tuple[str, ...] | None = None
        self._sort_asc: Tuple[bool, ...] | None = None

    def where_in(self, col: str, values: Sequence[Any]) -> "GTabQuery":
        vals = np.asarray(list(values))
        self._filters.append(("in", col, vals))
        return self

    def where_eq(self, col: str, value: Any) -> "GTabQuery":
        self._filters.append(("eq", col, value))
        return self

    def where_between(self, col: str, lo: float, hi: float) -> "GTabQuery":
        self._filters.append(("between", col, (float(lo), float(hi))))
        return self

    def group_by(self, cols: Sequence[str]) -> "GTabQuery":
        cols = tuple(cols)
        if len(cols) == 0:
            raise ValueError("group_by cols is empty")
        self._group_cols = cols
        return self

    def sort_by(self, cols: Sequence[str], ascending: bool | Sequence[bool] = True) -> "GTabQuery":
        cols = tuple(cols)
        if len(cols) == 0:
            raise ValueError("sort_by cols is empty")

        if isinstance(ascending, bool):
            asc = (ascending,) * len(cols)
        else:
            asc = tuple(bool(a) for a in ascending)
            if len(asc) != len(cols):
                raise ValueError(f"ascending must have same length as cols ({len(cols)}), got {len(asc)}")

        self._sort_cols = cols
        self._sort_asc = asc
        return self

    def _mask(self, tab: "GTab") -> np.ndarray:
        n = tab.nrow
        m = np.ones(n, dtype=bool)
        for op, col, val in self._filters:
            x = tab.col(col)
            if op == "in":
                m &= np.isin(x, val)
            elif op == "eq":
                m &= (x == val)
            elif op == "between":
                lo, hi = val
                m &= (x >= lo) & (x <= hi)
            else:
                raise ValueError(f"Unknown filter op: {op}")
        return m

    def _sorted_rows(self, sub: "GTab") -> np.ndarray:
        """
        Return an index array (0..sub.nrow-1) that sorts `sub` by configured columns.
        If no sort is configured, returns identity ordering.
        """
        n = sub.nrow
        if self._sort_cols is None:
            return np.arange(n, dtype=np.int64)

        assert self._sort_asc is not None
        cols = self._sort_cols
        asc = self._sort_asc

        # We use lexsort. Note: lexsort uses last key as primary.
        keys = []
        for c, a in zip(cols, asc):
            k = np.asarray(sub.col(c))
            # for descending, negate numeric; otherwise use stable reverse order trick
            if a:
                keys.append(k)
            else:
                if np.issubdtype(k.dtype, np.number):
                    keys.append(-k)
                else:
                    # fallback for non-numeric descending: stable reverse after sorting ascending on this key
                    # (works well enough for typical int columns; for true string cols you'd want a different approach)
                    keys.append(k)

        order = np.lexsort(tuple(keys[::-1]))  # reverse so first col is primary
        # If any non-numeric descending keys exist, apply a stable reverse per-key is hard;
        # for CASA gaintab cols (ints/floats), this is usually fully correct.
        return order.astype(np.int64)

    def get_indices(self, tab: "GTab") -> np.ndarray:
        return np.where(self._mask(tab))[0].astype(np.int64)

    def apply(self, tab: "GTab") -> Union["GTab", Dict[Tuple[Any, ...], "GTab"]]:
        rows = self.get_indices(tab)
        sub = tab.take_rows(rows)

        # apply sorting (before grouping, so groups keep sorted order internally)
        order = self._sorted_rows(sub)
        sub = sub.take_rows(order)

        if self._group_cols is None:
            return sub

        # group rows in `sub` by group columns
        cols = [np.asarray(sub.col(c)) for c in self._group_cols]
        keys = np.stack(cols, axis=1)  # (nRow, k)
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)

        out: Dict[Tuple[Any, ...], "GTab"] = {}
        for gi, key_row in enumerate(uniq):
            g_rows = np.where(inv == gi)[0].astype(np.int64)
            key = tuple(key_row.tolist())
            out[key] = sub.take_rows(g_rows)  # stays sorted
        return out

def make_corrtab_identity(gtab: str, *, also_clear_flags: bool = False):
    tb = table()
    tb.open(gtab, nomodify=False)
    try:
        c = tb.getcol("CPARAM")
        c[...] = 1.0 + 0.0j
        tb.putcol("CPARAM", c)

        tb.flush()
    finally:
        tb.close()
    return gtab

def verify_corrtab_is_identity(gtab: str, tol: float = 0.0):
    tb = table(); tb.open(gtab)
    try:
        c = tb.getcol("CPARAM")
        err = np.max(np.abs(c - (1.0 + 0.0j)))
    finally:
        tb.close()
    if err > tol:
        raise RuntimeError(f"{gtab} is not unity. max|g-1|={err}")
    return True

def make_template_gain_corrtab(ms: str, gtab: str, *, seed: int = 0):
    rmtables(gtab)
    sm = simulator()
    sm.openfromms(ms)
    sm.setseed(seed)
    sm.setgain(mode="random", table=gtab, amplitude=0)
    sm.done()
    make_corrtab_identity(gtab, also_clear_flags=False)
    verify_corrtab_is_identity(gtab)
    return gtab
