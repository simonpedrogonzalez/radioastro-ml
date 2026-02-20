from __future__ import annotations
import numpy as np
from casatools import table, simulator
from .corrfn import CorrFn
from .corrtab_utils import make_template_gain_corrtab

class Corruption:
    def build_corrtable(self, ms: str, corrtab: str, *, seed: int = 0):
        raise NotImplementedError

class GainCorruption(Corruption):
    pass

class AntennaGainCorruption(GainCorruption):
    def __init__(self, timegrid, amp_fn: CorrFn | None = None, phase_fn: CorrFn | None = None):
        self.tg = timegrid
        self.amp_fn = amp_fn
        self.phase_fn = phase_fn

    def build_corrtable(self, ms: str, corrtab: str, *, seed: int = 0):
        # 1) Create a CASA-valid template table (then we overwrite CPARAM)
        make_template_gain_corrtab(ms, corrtab, seed=seed)

        tb = table()
        tb.open(corrtab, nomodify=False)
        try:
            TIME = np.asarray(tb.getcol("TIME"), dtype=float)      # (nRow,)
            CP   = tb.getcol("CPARAM")                             # (nCorr, nChan, nRow)
            ANT1 = np.asarray(tb.getcol("ANTENNA1"), dtype=np.int32)

            assert CP.ndim == 3, CP.shape
            nCorr, nChan, nRow = CP.shape

            # 2) Decide which rows to update (here: all rows, time-ordered)
            # idx = np.arange(nRow, dtype=np.int64)
            idx = np.where(ANT1 == 1)[0]

            
            idx = idx[np.argsort(TIME[idx])]
            t = TIME[idx]

            rng = np.random.default_rng(seed)

            # 3) Generate amp/phase series
            if self.amp_fn is None:
                amp = np.ones_like(t, dtype=float)
            else:
                amp = self.amp_fn.eval(t, rng=rng)   # should return absolute amp ~ 1

            if self.phase_fn is None:
                phase = np.zeros_like(t, dtype=float)
            else:
                phase = self.phase_fn.eval(t, rng=rng)  # radians

            if amp.shape != t.shape or phase.shape != t.shape:
                raise ValueError(f"amp/phase must be shape {t.shape}, got {amp.shape}, {phase.shape}")

            gain = amp * np.exp(1j * phase)  # (nRow,)

            # 4) Broadcast into all corr/chan for those rows
            CP_new = CP.copy()
            CP_new[:, :, idx] = gain[None, None, :]

            tb.putcol("CPARAM", CP_new)
            tb.flush()
        finally:
            tb.close()

        return self
    
    def apply_corrtable(self, ms: str, corrtab: str, seed: int = 0):
        sm = simulator()
        sm.openfromms(ms)
        sm.setseed(seed)

        sm.setapply(
            table=corrtab,
            type="G",
            # field=GAINCAL_FIELD,
            interp="linear",
            calwt=False,
        )

        sm.corrupt()
        sm.done()

        return self
