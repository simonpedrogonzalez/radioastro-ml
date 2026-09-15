"""Gain-table data, selection, and narrow CASA construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np


FilterSpec = Tuple[str, str, Any]


def _new_table():
    try:
        from casatools import table
    except ImportError as exc:
        raise RuntimeError("CASA casatools is required for gain-table operations") from exc
    return table()


def _new_simulator():
    try:
        from casatools import simulator
    except ImportError as exc:
        raise RuntimeError("CASA casatools is required for gain-table operations") from exc
    return simulator()


def _remove_casa_tables(path: str) -> None:
    try:
        from casatasks import rmtables
    except ImportError as exc:
        raise RuntimeError("CASA casatasks is required for gain-table operations") from exc
    rmtables(path)


def get_unflagged_antennas(msname: str):
    """Return name/ID maps for antennas occurring in unflagged visibilities."""
    tb = _new_table()
    tb.open(f"{msname}/ANTENNA")
    try:
        names = np.asarray(tb.getcol("NAME"))
    finally:
        tb.close()

    tb.open(msname)
    try:
        ant1 = np.asarray(tb.getcol("ANTENNA1"), dtype=np.int32)
        ant2 = np.asarray(tb.getcol("ANTENNA2"), dtype=np.int32)
        flag = np.asarray(tb.getcol("FLAG"))
    finally:
        tb.close()

    fully_flagged = np.all(flag, axis=(0, 1))
    good_rows = ~fully_flagged
    used_ants = sorted(int(value) for value in set(ant1[good_rows]) | set(ant2[good_rows]))
    name_to_idx = {str(names[index]): int(index) for index in used_ants}
    idx_to_name = {int(index): str(names[index]) for index in used_ants}
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
    CPARAM: np.ndarray | None = None
    PARAMERR: np.ndarray | None = None
    FLAG: np.ndarray | None = None
    SNR: np.ndarray | None = None
    WEIGHT: np.ndarray | None = None

    @property
    def nrow(self) -> int:
        return int(self.TIME.shape[0])

    def col(self, name: str) -> np.ndarray:
        value = getattr(self, name)
        if value is None:
            raise KeyError(f"Column '{name}' not loaded / unavailable in this table")
        return value

    @classmethod
    def from_casa_table(cls, tb, *, load_optional: bool = False) -> "GTab":
        nrow = tb.nrows()

        def get(name):
            return np.asarray(tb.getcol(name))

        result = cls(
            ROWID=np.arange(nrow, dtype=np.int64),
            TIME=get("TIME").astype(float),
            FIELD_ID=get("FIELD_ID").astype(np.int32),
            SPECTRAL_WINDOW_ID=get("SPECTRAL_WINDOW_ID").astype(np.int32),
            ANTENNA1=get("ANTENNA1").astype(np.int32),
            ANTENNA2=get("ANTENNA2").astype(np.int32),
            INTERVAL=get("INTERVAL").astype(float),
            SCAN_NUMBER=get("SCAN_NUMBER").astype(np.int32),
            OBSERVATION_ID=get("OBSERVATION_ID").astype(np.int32),
        )
        if load_optional:
            for name in ["CPARAM", "PARAMERR", "FLAG", "SNR", "WEIGHT"]:
                try:
                    setattr(result, name, np.asarray(tb.getcol(name)))
                except Exception:
                    setattr(result, name, None)
        return result

    def take_rows(self, rows: np.ndarray) -> "GTab":
        rows = np.asarray(rows, dtype=np.int64)

        def take_row_axis(value: np.ndarray) -> np.ndarray:
            return np.asarray(value)[rows]

        def take_last_axis(value: np.ndarray) -> np.ndarray:
            value = np.asarray(value)
            if value.ndim == 0:
                return value
            if value.shape[-1] != self.nrow:
                raise ValueError(
                    f"Expected last dim == nrow ({self.nrow}), got shape {value.shape}"
                )
            return value[..., rows]

        return GTab(
            ROWID=take_row_axis(self.ROWID),
            TIME=take_row_axis(self.TIME),
            FIELD_ID=take_row_axis(self.FIELD_ID),
            SPECTRAL_WINDOW_ID=take_row_axis(self.SPECTRAL_WINDOW_ID),
            ANTENNA1=take_row_axis(self.ANTENNA1),
            ANTENNA2=take_row_axis(self.ANTENNA2),
            INTERVAL=take_row_axis(self.INTERVAL),
            SCAN_NUMBER=take_row_axis(self.SCAN_NUMBER),
            OBSERVATION_ID=take_row_axis(self.OBSERVATION_ID),
            CPARAM=None if self.CPARAM is None else take_last_axis(self.CPARAM),
            PARAMERR=None if self.PARAMERR is None else take_last_axis(self.PARAMERR),
            FLAG=None if self.FLAG is None else take_last_axis(self.FLAG),
            SNR=None if self.SNR is None else take_last_axis(self.SNR),
            WEIGHT=None if self.WEIGHT is None else take_last_axis(self.WEIGHT),
        )


def _plain_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_plain_value(item) for item in value]
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    return value


class GTabQuery:
    """Mutable, fluent selection/grouping query for a :class:`GTab`."""

    def __init__(self):
        self._filters: List[FilterSpec] = []
        self._group_cols: Tuple[str, ...] | None = None
        self._sort_cols: Tuple[str, ...] | None = None
        self._sort_asc: Tuple[bool, ...] | None = None

    def where_in(self, col: str, values: Sequence[Any]) -> "GTabQuery":
        self._filters.append(("in", col, np.asarray(list(values))))
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

    def sort_by(
        self,
        cols: Sequence[str],
        ascending: bool | Sequence[bool] = True,
    ) -> "GTabQuery":
        cols = tuple(cols)
        if len(cols) == 0:
            raise ValueError("sort_by cols is empty")
        if isinstance(ascending, bool):
            asc = (ascending,) * len(cols)
        else:
            asc = tuple(bool(value) for value in ascending)
            if len(asc) != len(cols):
                raise ValueError(
                    f"ascending must have same length as cols ({len(cols)}), got {len(asc)}"
                )
        self._sort_cols = cols
        self._sort_asc = asc
        return self

    def _mask(self, tab: GTab) -> np.ndarray:
        mask = np.ones(tab.nrow, dtype=bool)
        for operation, column, value in self._filters:
            values = tab.col(column)
            if operation == "in":
                mask &= np.isin(values, value)
            elif operation == "eq":
                mask &= values == value
            elif operation == "between":
                lower, upper = value
                mask &= (values >= lower) & (values <= upper)
            else:
                raise ValueError(f"Unknown filter op: {operation}")
        return mask

    def _sorted_rows(self, sub: GTab) -> np.ndarray:
        if self._sort_cols is None:
            return np.arange(sub.nrow, dtype=np.int64)
        assert self._sort_asc is not None
        keys = []
        for column, ascending in zip(self._sort_cols, self._sort_asc):
            key = np.asarray(sub.col(column))
            if not ascending and np.issubdtype(key.dtype, np.number):
                key = -key
            keys.append(key)
        return np.lexsort(tuple(keys[::-1])).astype(np.int64)

    def get_indices(self, tab: GTab) -> np.ndarray:
        return np.where(self._mask(tab))[0].astype(np.int64)

    def apply(self, tab: GTab) -> Union[GTab, Dict[Tuple[Any, ...], GTab]]:
        rows = self.get_indices(tab)
        sub = tab.take_rows(rows)
        sub = sub.take_rows(self._sorted_rows(sub))
        if self._group_cols is None:
            return sub

        columns = [np.asarray(sub.col(column)) for column in self._group_cols]
        keys = np.stack(columns, axis=1)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        result: Dict[Tuple[Any, ...], GTab] = {}
        for group_index, key_row in enumerate(unique):
            group_rows = np.where(inverse == group_index)[0].astype(np.int64)
            result[tuple(key_row.tolist())] = sub.take_rows(group_rows)
        return result

    def to_report_dict(self) -> dict[str, object]:
        filters = [
            {
                "operation": operation,
                "column": column,
                "value": _plain_value(value),
            }
            for operation, column, value in self._filters
        ]
        return {
            "type": "gain_table_query",
            "filters": filters,
            "group_by": None if self._group_cols is None else list(self._group_cols),
            "sort_by": None if self._sort_cols is None else list(self._sort_cols),
            "sort_ascending": None if self._sort_asc is None else list(self._sort_asc),
        }

    def to_report_text(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        config = self.to_report_dict()
        return (
            "GTabQuery("
            f"filters={config['filters']!r}, "
            f"group_by={config['group_by']!r}, "
            f"sort_by={config['sort_by']!r}, "
            f"sort_ascending={config['sort_ascending']!r})"
        )


def make_corrtab_identity(gtab: str, *, also_clear_flags: bool = False):
    del also_clear_flags
    tb = _new_table()
    tb.open(gtab, nomodify=False)
    try:
        values = tb.getcol("CPARAM")
        values[...] = 1.0 + 0.0j
        tb.putcol("CPARAM", values)
        tb.flush()
    finally:
        tb.close()
    return gtab


def verify_corrtab_is_identity(gtab: str, tol: float = 0.0):
    tb = _new_table()
    tb.open(gtab)
    try:
        values = tb.getcol("CPARAM")
        error = np.max(np.abs(values - (1.0 + 0.0j)))
    finally:
        tb.close()
    if error > tol:
        raise RuntimeError(f"{gtab} is not unity. max|g-1|={error}")
    return True


def make_template_gain_corrtab(ms: str, gtab: str, *, seed: int = 0):
    _remove_casa_tables(gtab)
    sm = _new_simulator()
    sm.openfromms(ms)
    sm.setseed(seed)
    sm.setgain(mode="random", table=gtab, amplitude=0)
    sm.done()
    make_corrtab_identity(gtab, also_clear_flags=False)
    verify_corrtab_is_identity(gtab)
    return gtab


__all__ = [
    "FilterSpec",
    "GCOLS",
    "GTab",
    "GTabQuery",
    "get_unflagged_antennas",
    "make_corrtab_identity",
    "make_template_gain_corrtab",
    "verify_corrtab_is_identity",
]
