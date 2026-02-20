from casatools import table, simulator
import numpy as np
from casatasks import rmtables

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
