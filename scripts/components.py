"""Reusable CASA source-component descriptions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from casatools import table


FREQUENCY_REFERENCE_NAMES = {
    0: "REST",
    1: "LSRK",
    2: "LSRD",
    3: "BARY",
    4: "GEO",
    5: "TOPO",
    6: "GALACTO",
    7: "LGROUP",
    8: "CMB",
}

def pointsource_constant_1Jy_phasecenter_fromMS(ms: str | Path) -> dict:
    """Return addcomponent parameters for a constant 1 Jy phase-centre source."""
    ms = Path(ms).expanduser().resolve()

    tb = table()
    tb.open(str(ms / "FIELD"), nomodify=True)
    try:
        if tb.nrows() == 0:
            raise RuntimeError(f"FIELD table is empty: {ms}")
        phase_dir = np.asarray(tb.getcell("PHASE_DIR", 0), dtype=float).reshape(2, -1)
        direction_keywords = tb.getcolkeywords("PHASE_DIR")
    finally:
        tb.close()

    direction_frame = str(
        direction_keywords.get("MEASINFO", {}).get("Ref", "J2000")
    )
    phase_centre = (
        f"{direction_frame} {float(phase_dir[0, 0]):.16g}rad "
        f"{float(phase_dir[1, 0]):.16g}rad"
    )

    tb = table()
    tb.open(str(ms / "SPECTRAL_WINDOW"), nomodify=True)
    try:
        if tb.nrows() == 0:
            raise RuntimeError(f"SPECTRAL_WINDOW table is empty: {ms}")
        reference_frequency_hz = float(tb.getcell("REF_FREQUENCY", 0))
        reference_code = int(tb.getcell("MEAS_FREQ_REF", 0))
    finally:
        tb.close()

    frequency_frame = FREQUENCY_REFERENCE_NAMES.get(reference_code)
    if frequency_frame is None:
        raise RuntimeError(f"Unsupported CASA frequency reference code: {reference_code}")

    return {
        "flux": [1.0, 0.0, 0.0, 0.0],
        "fluxunit": "Jy",
        "polarization": "Stokes",
        "dir": phase_centre,
        "shape": "point",
        "freq": f"{frequency_frame} {reference_frequency_hz:.16g}Hz",
        "spectrumtype": "constant",
    }


def pointsource_0012_399_phasecenter_fromMS(ms: str | Path) -> dict:
    """Return the pipeline-fitted J0012-3954 model at the MS phase centre."""
    source = pointsource_constant_1Jy_phasecenter_fromMS(ms)
    frequency_frame = source["freq"].split(maxsplit=1)[0]
    source.update(
        flux=[0.7009622255785544, 0.0, 0.0, 0.0],
        freq=f"{frequency_frame} 4.68GHz",
        # spectrumtype="spectral index",
        spectrumtype="constant"
        # index=-0.06782510991621188,
    )
    return source
