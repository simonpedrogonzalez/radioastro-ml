"""Construct the supported phase-centre CASA component records."""

from __future__ import annotations

import math
import operator
from pathlib import Path
from typing import Any, Iterable


def _new_table():
    try:
        from casatools import table
    except ImportError as exc:  # pragma: no cover - exercised outside CASA
        raise RuntimeError("CASA casatools is required to inspect Measurement Sets") from exc
    return table()


def _flat_values(value: Any) -> list[Any]:
    """Flatten CASA/numpy values without importing numpy at module import time."""
    if hasattr(value, "ravel"):
        value = value.ravel().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()

    result: list[Any] = []

    def visit(item: Any) -> None:
        if isinstance(item, (str, bytes)):
            result.append(item)
        elif isinstance(item, Iterable):
            for child in item:
                visit(child)
        else:
            result.append(item)

    visit(value)
    return result


def _row_id(value: int, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_float(value: float, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _require_row(tb: Any, row: int, *, table_name: str) -> None:
    if row >= int(tb.nrows()):
        raise IndexError(
            f"{table_name} row {row} does not exist; table has {int(tb.nrows())} rows"
        )


def _measurement_reference(tb: Any, column: str, row: int) -> str:
    keywords = dict(tb.getcolkeywords(column) or {})
    measure_info = dict(keywords.get("MEASINFO") or {})
    variable_column = measure_info.get("VarRefCol")
    if variable_column:
        if variable_column not in set(tb.colnames()):
            raise RuntimeError(
                f"{column} names missing variable reference column {variable_column!r}"
            )
        code = int(tb.getcell(str(variable_column), row))
        codes = [int(item) for item in _flat_values(measure_info.get("TabRefCodes", []))]
        names = [str(item) for item in _flat_values(measure_info.get("TabRefTypes", []))]
        if len(codes) != len(names):
            raise RuntimeError(f"{column} has inconsistent TabRefCodes/TabRefTypes metadata")
        for candidate, name in zip(codes, names):
            if candidate == code:
                if not name:
                    break
                return name
        raise RuntimeError(f"{column} reference code {code} has no TabRefTypes mapping")

    fixed = measure_info.get("Ref")
    if fixed:
        return str(fixed)
    raise RuntimeError(f"{column} has neither MEASINFO.VarRefCol nor MEASINFO.Ref")


def _column_units(tb: Any, column: str) -> list[str]:
    keywords = dict(tb.getcolkeywords(column) or {})
    return [str(item) for item in _flat_values(keywords.get("QuantumUnits", []))]


def _angle_to_radians(value: float, unit: str) -> float:
    normalized = unit.strip().lower()
    if normalized in {"rad", "radian", "radians"}:
        return value
    if normalized in {"deg", "degree", "degrees"}:
        return math.radians(value)
    if normalized in {"arcmin", "arcminute", "arcminutes"}:
        return math.radians(value / 60.0)
    if normalized in {"arcsec", "arcsecond", "arcseconds", "asec"}:
        return math.radians(value / 3600.0)
    raise RuntimeError(f"Unsupported direction unit {unit!r}")


def _frequency_to_hz(value: float, unit: str) -> float:
    scales = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}
    normalized = unit.strip().lower()
    if normalized not in scales:
        raise RuntimeError(f"Unsupported frequency unit {unit!r}")
    return value * scales[normalized]


def get_phase_center(ms: str | Path, *, field_id: int = 0) -> str:
    """Return the static field phase centre as a CASA direction string."""
    path = Path(ms).expanduser().resolve()
    row = _row_id(field_id, name="field_id")
    tb = _new_table()
    tb.open(str(path / "FIELD"), nomodify=True)
    try:
        _require_row(tb, row, table_name="FIELD")
        columns = set(tb.colnames())
        if "NUM_POLY" not in columns:
            raise RuntimeError("FIELD table has no NUM_POLY column")
        if int(tb.getcell("NUM_POLY", row)) != 0:
            raise ValueError("Polynomial FIELD phase centres are not supported")
        if "EPHEMERIS_ID" in columns and int(tb.getcell("EPHEMERIS_ID", row)) >= 0:
            raise ValueError("Ephemeris FIELD phase centres are not supported")

        values = [float(item) for item in _flat_values(tb.getcell("PHASE_DIR", row))]
        if len(values) != 2:
            raise RuntimeError(
                f"Static FIELD.PHASE_DIR must contain two coordinates, found {len(values)}"
            )
        units = _column_units(tb, "PHASE_DIR")
        if len(units) == 1:
            units *= 2
        if len(units) != 2:
            raise RuntimeError(f"FIELD.PHASE_DIR must declare two units, found {units!r}")
        frame = _measurement_reference(tb, "PHASE_DIR", row)
    finally:
        tb.close()

    longitude = _angle_to_radians(values[0], units[0])
    latitude = _angle_to_radians(values[1], units[1])
    if not math.isfinite(longitude) or not math.isfinite(latitude):
        raise RuntimeError("FIELD.PHASE_DIR contains non-finite coordinates")
    return f"{frame} {longitude:.16g}rad {latitude:.16g}rad"


def get_reference_frequency(ms: str | Path, *, spw_id: int = 0) -> str:
    """Return one SPW reference frequency as a CASA frequency string."""
    path = Path(ms).expanduser().resolve()
    row = _row_id(spw_id, name="spw_id")
    tb = _new_table()
    tb.open(str(path / "SPECTRAL_WINDOW"), nomodify=True)
    try:
        _require_row(tb, row, table_name="SPECTRAL_WINDOW")
        frequency = float(tb.getcell("REF_FREQUENCY", row))
        units = _column_units(tb, "REF_FREQUENCY")
        if len(units) != 1:
            raise RuntimeError(
                f"SPECTRAL_WINDOW.REF_FREQUENCY must declare one unit, found {units!r}"
            )
        frame = _measurement_reference(tb, "REF_FREQUENCY", row)
    finally:
        tb.close()

    frequency_hz = _frequency_to_hz(frequency, units[0])
    if not math.isfinite(frequency_hz) or frequency_hz <= 0:
        raise RuntimeError("SPECTRAL_WINDOW.REF_FREQUENCY must be finite and positive")
    return f"{frame} {frequency_hz:.16g}Hz"


def phase_center_point_source(
    ms: str | Path,
    flux_jy: float,
    *,
    field_id: int = 0,
    spw_id: int = 0,
) -> dict[str, object]:
    """Build CASA parameters for an unpolarized constant-spectrum point source."""
    flux = _positive_float(flux_jy, name="flux_jy")
    return {
        "flux": [flux, 0.0, 0.0, 0.0],
        "fluxunit": "Jy",
        "polarization": "Stokes",
        "dir": get_phase_center(ms, field_id=field_id),
        "shape": "point",
        "freq": get_reference_frequency(ms, spw_id=spw_id),
        "spectrumtype": "constant",
    }


def phase_center_point_source_from_snr(
    ms: str | Path,
    snr: float,
    image_rms_jy_per_beam: float,
    *,
    field_id: int = 0,
    spw_id: int = 0,
) -> dict[str, object]:
    """Build the supported point source with flux equal to ``snr * RMS``."""
    signal_to_noise = _positive_float(snr, name="snr")
    rms = _positive_float(image_rms_jy_per_beam, name="image_rms_jy_per_beam")
    return phase_center_point_source(
        ms,
        signal_to_noise * rms,
        field_id=field_id,
        spw_id=spw_id,
    )


__all__ = [
    "get_phase_center",
    "get_reference_frequency",
    "phase_center_point_source",
    "phase_center_point_source_from_snr",
]
