"""Utilities for simulating sources on an extracted MS sampling pattern."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from casatools import componentlist, simulator, table

from scripts.image_extracted import filter_ms_paths, find_extracted_ms_paths
from scripts.io_utils import copy_ms


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACTED_DIR = REPO_ROOT / "collect" / "extracted"
VISIBILITY_COLUMNS = ("DATA", "CORRECTED_DATA", "MODEL_DATA")


def find_visibility_ms(visibility_id: str) -> Path:
    """Resolve one visibility ID to its original extracted MeasurementSet."""
    matches = filter_ms_paths(
        find_extracted_ms_paths(EXTRACTED_DIR),
        [str(visibility_id)],
    )
    if not matches:
        raise FileNotFoundError(
            f"No extracted MeasurementSet found for {visibility_id!r} under "
            f"{EXTRACTED_DIR}"
        )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one MeasurementSet for {visibility_id!r}, found {matches}"
        )
    return matches[0].resolve()


def _output_ms_path(source_ms: Path, output_ms_name: str) -> Path:
    name = str(output_ms_name).strip()
    if not name or Path(name).name != name:
        raise ValueError("output_ms_name must be a non-empty directory name, not a path")
    if not name.endswith(".ms"):
        name += ".ms"
    output_ms = source_ms.parent / name
    if output_ms.resolve() == source_ms.resolve():
        raise ValueError("The output MS must not be the original extracted MS")
    return output_ms


def _zero_visibility_columns(ms_path: Path, chunk_rows: int = 2048) -> None:
    """Zero measurement columns in bounded-memory chunks."""
    tb = table()
    tb.open(str(ms_path), nomodify=False)
    try:
        columns = set(tb.colnames())
        nrows = tb.nrows()
        for column in VISIBILITY_COLUMNS:
            if column not in columns:
                continue
            print(f"[ZERO] {ms_path.name}:{column}")
            for startrow in range(0, nrows, chunk_rows):
                nrow = min(chunk_rows, nrows - startrow)
                values = tb.getcol(column, startrow=startrow, nrow=nrow)
                tb.putcol(
                    column,
                    np.zeros_like(values),
                    startrow=startrow,
                    nrow=nrow,
                )
    finally:
        tb.close()


def _write_component_list(sources: list[dict], component_path: Path) -> None:
    if component_path.exists():
        shutil.rmtree(component_path)

    cl = componentlist()
    try:
        for source in sources:
            if not isinstance(source, dict):
                raise TypeError("Each source must be a dict of componentlist.addcomponent parameters")
            cl.addcomponent(**source)
        cl.rename(str(component_path))
    finally:
        cl.done()


def simulate_ms(
    visibility_id: str,
    sources: list[dict],
    output_ms_name: str,
) -> Path:
    """Predict ``sources`` into a copied extracted MS and return its path.

    The original extracted MS is opened only by CASA's simulator as a sampling
    source indirectly through its copy; it is never modified.
    """
    if not sources:
        raise ValueError("sources must contain at least one component")

    source_ms = find_visibility_ms(visibility_id)
    output_ms = _output_ms_path(source_ms, output_ms_name)
    copy_ms(str(source_ms), str(output_ms))
    _zero_visibility_columns(output_ms)

    component_path = output_ms.parent / f"{output_ms.stem}.components.cl"
    _write_component_list(zv, component_path)

    sm = simulator()
    try:
        if not sm.openfromms(str(output_ms)):
            raise RuntimeError(f"CASA simulator could not open {output_ms}")
        if not sm.predict(complist=str(component_path), incremental=False):
            raise RuntimeError(f"CASA simulator prediction failed for {output_ms}")
    finally:
        sm.close()

    print(f"[DONE] simulated MS: {output_ms}")
    return output_ms

