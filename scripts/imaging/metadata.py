"""Measurement Set path and metadata discovery."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

from scripts.vla_config import band_for_frequency_ghz, split_band_codes

from .models import CSVMeta, DataColumnValidation, MSBandMeta, ResolvedMS


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATOR_META_CSV = REPOSITORY_ROOT / "collect/small_subset/small_selection.csv"
DEFAULT_EXTRACTED_MS_ROOT = REPOSITORY_ROOT / "collect/extracted"
_VISIBILITY_ID = re.compile(r"^\d{4}[+-]\d{3}$")
_COLUMN_NAMES = {"corrected": "CORRECTED_DATA", "data": "DATA"}


def repository_path(path: str | Path) -> Path:
    """Resolve configuration paths relative to the repository, never the CWD."""
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (REPOSITORY_ROOT / value).resolve()


def _infer_visibility_id(path: Path) -> Optional[str]:
    for part in reversed(path.parts):
        stem = Path(part).stem
        if _VISIBILITY_ID.fullmatch(stem):
            return stem
    return None


def _is_measurement_set(path: Path) -> bool:
    return path.is_dir() and (path.suffix.lower() == ".ms" or (path / "table.dat").exists())


def resolve_path(ms: str | Path, extracted_root: str | Path = DEFAULT_EXTRACTED_MS_ROOT) -> ResolvedMS:
    """Resolve an existing MS path or a visibility ID to exactly one MS."""
    supplied = Path(ms).expanduser()
    if supplied.exists():
        resolved = supplied.resolve()
        if not _is_measurement_set(resolved):
            raise ValueError(f"Existing path is not a Measurement Set: {resolved}")
        return ResolvedMS(resolved, _infer_visibility_id(resolved))

    value = str(ms).strip()
    if not _VISIBILITY_ID.fullmatch(value):
        raise FileNotFoundError(
            f"Measurement Set path does not exist and {value!r} is not a visibility ID"
        )

    root = repository_path(extracted_root)
    search_root = root / value if (root / value).is_dir() else root
    named_candidates = sorted(
        path.resolve() for path in search_root.rglob(f"{value}.ms") if _is_measurement_set(path)
    )
    candidates = named_candidates
    if not candidates and search_root != root:
        candidates = sorted(
            path.resolve() for path in search_root.rglob("*.ms") if _is_measurement_set(path)
        )

    if len(candidates) != 1:
        rendered = "\n".join(f"  - {path}" for path in candidates) or "  (none)"
        raise RuntimeError(
            f"Expected exactly one Measurement Set for visibility ID {value!r} below "
            f"{search_root}, found {len(candidates)}. Candidates:\n{rendered}"
        )
    return ResolvedMS(candidates[0], value)


def _clean_optional(value: object) -> Optional[str]:
    text = "" if value is None else str(value).strip()
    return text or None


def _optional_float(value: object) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _matching_csv_row(visibility_id: str, csv_path: Path) -> Optional[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Calibrator metadata CSV does not exist: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Calibrator metadata CSV has no header: {csv_path}")
        matches = [
            row
            for row in reader
            if visibility_id
            in {
                str(row.get("folder", "")).strip(),
                str(row.get("visibility_id", "")).strip(),
                str(row.get("name", "")).strip(),
            }
        ]
    if len(matches) > 1:
        raise RuntimeError(
            f"Found {len(matches)} metadata rows for visibility ID {visibility_id!r} in {csv_path}"
        )
    return matches[0] if matches else None


def resolve_csv_meta(
    visibility_id: Optional[str], csv_path: str | Path
) -> Tuple[Optional[CSVMeta], Optional[int]]:
    """Return public catalog metadata and the private preferred-SPW hint."""
    if visibility_id is None:
        return None, None
    row = _matching_csv_row(visibility_id, repository_path(csv_path))
    if row is None:
        return None, None

    band_value = _clean_optional(row.get("band_code")) or _clean_optional(row.get("band_guess"))
    preferred_spw = _optional_float(row.get("spw_selected"))
    meta = CSVMeta(
        visibility_id=visibility_id,
        array_configuration=_clean_optional(
            row.get("gain_array_config") or row.get("array_configuration")
        ),
        catalog_band_codes=tuple(split_band_codes(band_value)),
        catalog_frequency_ghz=_optional_float(
            row.get("spw_center_ghz") or row.get("catalog_frequency_ghz")
        ),
    )
    return meta, None if preferred_spw is None else int(preferred_spw)


def _read_spw_frequencies(ms_path: Path) -> Tuple[Tuple[int, float, float], ...]:
    try:
        from casatools import table
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("CASA casatools is required to inspect Measurement Sets") from exc

    tb = table()
    tb.open(str(ms_path / "SPECTRAL_WINDOW"))
    try:
        result = []
        for spw in range(tb.nrows()):
            frequencies = np.asarray(tb.getcell("CHAN_FREQ", spw), dtype=float)
            frequencies = frequencies[np.isfinite(frequencies)]
            if frequencies.size:
                result.append(
                    (int(spw), float(np.median(frequencies) / 1e9), float(np.ptp(frequencies)))
                )
        return tuple(result)
    finally:
        tb.close()


def resolve_ms_band(
    ms_path: Path,
    csv_meta: Optional[CSVMeta],
    *,
    preferred_spw: Optional[int] = None,
) -> MSBandMeta:
    spws = _read_spw_frequencies(ms_path)
    if not spws:
        raise RuntimeError(f"No finite channel frequencies found in {ms_path / 'SPECTRAL_WINDOW'}")
    selected = next((item for item in spws if item[0] == preferred_spw), None)
    if selected is None:
        selected = max(spws, key=lambda item: item[2])
    spw, frequency_ghz, _ = selected
    band = band_for_frequency_ghz(frequency_ghz)
    if band is None:
        raise RuntimeError(
            f"Representative frequency {frequency_ghz:.9g} GHz from SPW {spw} is outside "
            "the supported VLA receiver bands"
        )
    matches = None
    if csv_meta is not None and csv_meta.catalog_band_codes:
        matches = band in set(csv_meta.catalog_band_codes)
    return MSBandMeta(band, frequency_ghz, spw, matches)


def ms_column_names(ms_path: Path) -> Tuple[str, ...]:
    try:
        from casatools import table
    except ImportError as exc:
        raise RuntimeError("CASA casatools is required to inspect Measurement Sets") from exc
    tb = table()
    tb.open(str(ms_path))
    try:
        return tuple(str(name) for name in tb.colnames())
    finally:
        tb.close()


def validate_data_column(
    ms_path: Path,
    requested: str,
    *,
    chunk_rows: int = 4096,
) -> DataColumnValidation:
    """Validate one explicit CASA data column with bounded-memory reads."""
    if requested not in _COLUMN_NAMES:
        raise ValueError(f"datacolumn must be 'corrected' or 'data', got {requested!r}")
    if chunk_rows <= 0:
        raise ValueError(f"chunk_rows must be positive, got {chunk_rows}")
    try:
        from casatools import table
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("CASA casatools and NumPy are required to validate MS data") from exc

    physical = _COLUMN_NAMES[requested]
    tb = table()
    tb.open(str(ms_path))
    try:
        columns = tuple(str(name) for name in tb.colnames())
        if physical not in columns:
            raise RuntimeError(
                f"Requested datacolumn={requested!r} maps to missing table column {physical!r}; "
                f"available columns: {', '.join(columns)}"
            )
        if "FLAG" not in columns:
            raise RuntimeError(
                f"Cannot validate datacolumn={requested!r}: FLAG is missing; available columns: "
                f"{', '.join(columns)}"
            )

        rows_examined = unflagged = finite = nonzero = 0
        total_rows = int(tb.nrows())
        for start in range(0, total_rows, chunk_rows):
            count = min(chunk_rows, total_rows - start)
            data = np.asarray(tb.getcol(physical, start, count))
            flags = np.asarray(tb.getcol("FLAG", start, count), dtype=bool)
            if flags.shape != data.shape:
                flags = np.broadcast_to(flags, data.shape)
            usable = ~flags
            if "FLAG_ROW" in columns:
                flag_rows = np.asarray(tb.getcol("FLAG_ROW", start, count), dtype=bool).reshape(-1)
                row_shape = (1,) * (data.ndim - 1) + (flag_rows.size,)
                usable &= ~np.broadcast_to(flag_rows.reshape(row_shape), data.shape)
            finite_mask = usable & np.isfinite(data)
            nonzero_mask = finite_mask & (np.abs(data) > 0)
            rows_examined += count
            unflagged += int(np.count_nonzero(usable))
            finite += int(np.count_nonzero(finite_mask))
            nonzero += int(np.count_nonzero(nonzero_mask))
    finally:
        tb.close()

    reasons = []
    if unflagged == 0:
        reasons.append("it contains no unflagged samples")
    elif finite == 0:
        reasons.append("none of its unflagged samples are finite")
    elif nonzero == 0:
        reasons.append("all finite unflagged samples are zero-valued")
    if reasons:
        raise RuntimeError(
            f"Requested datacolumn={requested!r} ({physical}) is unusable because "
            f"{'; '.join(reasons)}; available columns: {', '.join(columns)}"
        )
    return DataColumnValidation(
        requested, physical, columns, rows_examined, unflagged, finite, nonzero
    )
