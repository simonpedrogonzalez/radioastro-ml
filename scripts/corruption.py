from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Literal
import numpy as np
from casatools import table, msmetadata, simulator
from casatasks import rmtables
from corrfn import CorrFn

@dataclass(frozen=True)
class TimeGrid:
    solint: str | int = "int"  # can become int (seconds)
    interp: Literal["linear"] = "linear" # idk if there could be another one that makes sense

    def __post_init__(self):
        if self.solint == "int":
            return

        if isinstance(self.solint, int):
            return

        value = self.solint.strip().lower()

        if value.endswith("s"):
            seconds = int(value[:-1])
        elif value.endswith("m"):
            seconds = int(value[:-1]) * 60
        else:
            raise ValueError(
                f"Invalid solint '{self.solint}'. Use 'int', '#s', or '#m'."
            )

        object.__setattr__(self, "solint", seconds)

class Corruption:
    def build_corrtable(self, ms: str, corrtab: str, *, seed: int = 0):
        raise NotImplementedError

class GainCorruption(Corruption):
    """
    Gain corruption on time.
    """
    pass

class AntennaGainCorruption(GainCorruption):
    """
    Generates a per-antenna gain corruption table (G).
    Uses CASA to create a G table, then overwrites the values in Python.
    """

    def __init__(
        self,
        timegrid: TimeGrid,
        amp_fn: CorrFn = None,
        phase_fn: CorrFn = None
    ):
        self.tg = timegrid
        self.amp_fn = amp_fn
        self.phase_fn = phase_fn

    def _make_template_corrtab(self, ms: str, corrtab: str, *, seed: int):
        """
        Create a CASA-made gain table with correct schema.
        We'll overwrite CPARAM, so the exact mode doesn't matter much.
        """
        rmtables(corrtab)
        sm = simulator()
        sm.openfromms(ms)
        sm.setseed(int(seed))
        sm.setgain(mode="random", table=corrtab, amplitude=0) # TODO: check this means identity
        sm.done()
        return corrtab

    def build_caltable(self, ms: str, corrtab: str, *, seed: int = 0):

        self._make_template_corrtab(ms, corrtab, seed=seed)

        tb = table()
        tb.open(corrtab, nomodify=False)
        try:
            TIME = np.asarray(tb.getcol("TIME"), dtype=float)
            ANT  = np.asarray(tb.getcol("ANTENNA1"), dtype=int)
            CP   = tb.getcol("CPARAM")
            FL   = tb.getcol("FLAG")

            # CP shape typically (nCorr, nChan, nRow) or (nCorr, 1, nRow)
            nCorr, nChan, nRow = CP.shape[0], CP.shape[1], CP.shape[2]

            # Build per-antenna time knots
            # TODO: build knots at user defined times (interval)
            # First we will use the times defined by the template table.
            ants = np.unique(ANT)
            corr_ids = range(nCorr)
            chan_ids = range(nChan)

            CP_new = CP.copy()

            # Group indices per antenna (time-ordered)
            for ant in ants:
                # Select antenna rows
                idx = np.where(ANT == ant & FL == False)[0]
                if idx.size == 0:
                    continue
                # Get it sorted by time
                order = np.argsort(TIME[idx])
                idx = idx[order]
                t = TIME[idx]

                # Build a drift series on these times
                # TODO: here the interpolation / interval should take place
                for ic in corr_ids:
                    for ich in chan_ids:
                        rng = np.random.default_rng(seed + 100000*ant + 100*ic + ich)

                        # base drift ~ RMS 1
                        d = self.drift_fn(t, rng)

                        # amplitude component (fractional around 1)
                        amp = 1.0 + self.str.amp_rms * d
                        amp = np.clip(amp, 1.0 - self.str.amp_clip, 1.0 + self.str.amp_clip)

                        # phase component (radians)
                        ph = np.deg2rad(self.str.phase_rms_deg) * d
                        ph = np.clip(ph, -np.deg2rad(self.str.phase_clip_deg), np.deg2rad(self.str.phase_clip_deg))

                        g = amp * np.exp(1j * ph)

                        # Respect existing flags: only overwrite unflagged entries
                        good = ~FL[ic, ich, idx]
                        CP_new[ic, ich, idx[good]] = g[good]

            tb.putcol("CPARAM", CP_new)
            tb.flush()
        finally:
            tb.close()

        return corrtab



def force_gtab_unity(gtab: str, *, also_clear_flags: bool = False):
    tb = table()
    tb.open(gtab, nomodify=False)
    try:
        c = tb.getcol("CPARAM")
        c[...] = 1.0 + 0.0j
        tb.putcol("CPARAM", c)

        if also_clear_flags and "FLAG" in tb.colnames():
            f = tb.getcol("FLAG")
            f[...] = False
            tb.putcol("FLAG", f)

        tb.flush()
    finally:
        tb.close()
    return gtab

def verify_gtab_is_unity(gtab: str, tol: float = 0.0):
    tb = table(); tb.open(gtab)
    try:
        c = tb.getcol("CPARAM")
        err = np.max(np.abs(c - (1.0 + 0.0j)))
    finally:
        tb.close()
    if err > tol:
        raise RuntimeError(f"{gtab} is not unity. max|g-1|={err}")
    return True

def make_template_gtab(ms: str, gtab: str, *, seed: int = 0):
    rmtables(gtab)
    sm = simulator()
    sm.openfromms(ms)
    sm.setseed(seed)
    sm.setgain(mode="random", table=gtab, amplitude=0.0001)  # whatever
    sm.done()
    force_gtab_unity(gtab, also_clear_flags=False)
    verify_gtab_is_unity(gtab)
    return gtab

@dataclass(frozen=True)
class Granularity:
    per_antenna: bool = True
    per_corr: bool = False
    per_chan: bool = False
