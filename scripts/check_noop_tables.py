# compares column values between original and output MS (exact)

import os
import shutil
import numpy as np

from casatools import simulator, table, image
from casatasks import tclean, rmtables

# Settings
MS_IN  = "3c391_ctm_mosaic_10s_spw0.gaincal_corr.ms"
MS_OUT = "3c391_ctm_mosaic_10s_spw0.gaincal_corr.noopcorrupt.ms"

FIELD  = "J1822-0938"
SPW    = ""
IMBASE_IN  = "img_gaincal_orig"
IMBASE_OUT = "img_gaincal_noop"

TCLEAN_KW = dict(
    field=FIELD,
    spw=SPW,
    specmode="mfs",
    gridder="standard",
    deconvolver="hogbom",
    stokes="I",
    weighting="briggs",
    robust=0.5,
    cell="2.5arcsec",
    imsize=[320, 320],
    niter=0,
    interactive=False,
)

CHUNK_NROW = 256

def die(msg: str):
    raise RuntimeError(msg)

def copy_ms(ms_in: str, ms_out: str):
    if not os.path.exists(ms_in):
        die(f"Input MS not found: {ms_in}")
    if os.path.exists(ms_out):
        print(f"[INFO] Removing existing output MS: {ms_out}")
        shutil.rmtree(ms_out)
    print(f"[INFO] Copying MS -> {ms_out}")
    shutil.copytree(ms_in, ms_out)

def simulator_noop_corrupt(msname: str):
    print("[INFO] Running simulator reset()+corrupt() (should be a no-op)")
    sm = simulator()
    ok = sm.openfromms(msname)
    if not ok:
        die("simulator.openfromms failed")
    sm.reset()
    ok2 = sm.corrupt()
    if not ok2:
        die("simulator.corrupt failed")
    sm.close()

def list_ms_tables(msname: str):
    """
    Return a list of table paths to compare: main + subtables that exist.
    """
    paths = [msname]

    candidates = [
        "ANTENNA", "DATA_DESCRIPTION", "DOPPLER", "FEED", "FIELD",
        "FLAG_CMD", "FREQ_OFFSET", "HISTORY", "OBSERVATION", "POINTING",
        "POLARIZATION", "PROCESSOR", "SOURCE", "SPECTRAL_WINDOW", "STATE",
        "SYSCAL", "WEATHER"
    ]
    for t in candidates:
        p = os.path.join(msname, t)
        if os.path.exists(p):
            paths.append(p)
    return paths

def _compare_arrays_strict(a, b, ctx):
    """Bitwise exact comparison of arrays
    """
    if a.dtype != b.dtype:
        die(f"[DIFF] dtype mismatch at {ctx}: {a.dtype} vs {b.dtype}")
    if a.shape != b.shape:
        die(f"[DIFF] shape mismatch at {ctx}: {a.shape} vs {b.shape}")
    if a.dtype.kind in ("f", "c"):
        same = np.array_equal(a, b, equal_nan=True)
    else:
        same = np.array_equal(a, b)
    if not same:
        if a.dtype.kind in ("f", "c"):
            diff = np.nanmax(np.abs(a - b))
            die(f"[DIFF] value mismatch at {ctx} (max|diff|={diff})")
        die(f"[DIFF] value mismatch at {ctx}")

ALLOWED_EXTRA_COLS = {"MODEL_DATA", "CORRECTED_DATA"} # These are created by the corrupt() method
CHUNK_NROW = 256

def _iter_row_chunks(nrow: int, chunk_nrow: int):
    start = 0
    while start < nrow:
        n = min(chunk_nrow, nrow - start)
        yield start, n
        start += n

def _getcol_chunk(tb, tpath: str, col: str, start: int, n: int):
    tb.open(tpath)
    arr = tb.getcol(col, startrow=start, nrow=n)
    tb.close()
    return np.asarray(arr)

def _assert_col_equal(tpath: str, colA: str, colB: str, chunk_nrow: int = 256):
    tb = table()
    tb.open(tpath)
    nrow = tb.nrows()
    tb.close()

    for start, n in _iter_row_chunks(nrow, chunk_nrow):
        a = _getcol_chunk(tb, tpath, colA, start, n)
        b = _getcol_chunk(tb, tpath, colB, start, n)
        _compare_arrays_strict(a, b, ctx=f"{tpath}:{colA} vs {colB}[{start}:{start+n}]")

def _assert_col_all_zero(tpath: str, col: str, chunk_nrow: int = 256):
    tb = table()
    tb.open(tpath)
    nrow = tb.nrows()
    tb.close()

    for start, n in _iter_row_chunks(nrow, chunk_nrow):
        a = _getcol_chunk(tb, tpath, col, start, n)
        if a.dtype.kind in ("c", "f"):
            if not np.array_equal(a, np.zeros_like(a), equal_nan=False):
                maxabs = np.nanmax(np.abs(a))
                die(f"[DIFF] {tpath}:{col} not all zeros (max|val|={maxabs}) at rows {start}:{start+n}")
        else:
            if not np.array_equal(a, np.zeros_like(a)):
                die(f"[DIFF] {tpath}:{col} not all zeros at rows {start}:{start+n}")


def _compare_column_robust(t1: str, t2: str, col: str, nrow: int, chunk_nrow: int = 256):
    """
    Compare column to column, taking into account casa 'undefined cells'
    If defined, compare values to be equal. If undefined cells, make sure both columns have
    the same undefined cells.
    """
    tb = table()

    try:
        for start, n in _iter_row_chunks(nrow, chunk_nrow):
            tb.open(t1)
            a = np.asarray(tb.getcol(col, startrow=start, nrow=n))
            tb.close()

            tb.open(t2)
            b = np.asarray(tb.getcol(col, startrow=start, nrow=n))
            tb.close()

            _compare_arrays_strict(a, b, ctx=f"{t1}:{col}[{start}:{start+n}]")
        return
    except Exception as e:
        print(f"[WARN] getcol failed for {col} ({e}). Falling back to per-row compare.")

    for start, n in _iter_row_chunks(nrow, chunk_nrow):
        tb.open(t1)
        def1 = [tb.iscelldefined(col, r) for r in range(start, start + n)]
        tb.close()

        tb.open(t2)
        def2 = [tb.iscelldefined(col, r) for r in range(start, start + n)]
        tb.close()

        if def1 != def2:
            die(f"[DIFF] Defined/undefined pattern differs for {col} rows {start}:{start+n}")

        for i, r in enumerate(range(start, start + n)):
            if not def1[i]:
                continue
            tb.open(t1)
            a = tb.getcell(col, r)
            tb.close()

            tb.open(t2)
            b = tb.getcell(col, r)
            tb.close()

            _compare_arrays_strict(np.asarray(a), np.asarray(b), ctx=f"{t1}:{col}[row {r}]")


def compare_table(t1: str, t2: str, chunk_nrow: int = 256):
    """Checks a t1 and a noop corruption output t2 are equal

    Checks equal values for all columns, checks t1 DATA == t2 CORRECTED_DATA

    Parameters
    ----------
    t1 : str
        original table
    t2 : str
        corruption output table
    chunk_nrow : int, optional
        chunking, by default 256
    """
    tb = table()

    if not os.path.exists(t1):
        die(f"Missing table: {t1}")
    if not os.path.exists(t2):
        die(f"Missing table: {t2}")

    tb.open(t1)
    cols1 = tb.colnames()
    nrow1 = tb.nrows()
    kw1 = tb.keywordnames()
    tb.close()

    tb.open(t2)
    cols2 = tb.colnames()
    nrow2 = tb.nrows()
    kw2 = tb.keywordnames()
    tb.close()

    if nrow1 != nrow2:
        die(f"[DIFF] Row count mismatch at {t1} vs {t2}: {nrow1} vs {nrow2}")

    if kw1 != kw2:
        die(f"[DIFF] Keyword name list mismatch at {t1} vs {t2}: {kw1} vs {kw2}")

    set1, set2 = set(cols1), set(cols2)
    common = sorted(set1 & set2)
    only1 = sorted(set1 - set2)
    only2 = sorted(set2 - set1)

    bad_only1 = [c for c in only1 if c not in ALLOWED_EXTRA_COLS]
    bad_only2 = [c for c in only2 if c not in ALLOWED_EXTRA_COLS]
    if bad_only1 or bad_only2:
        die(
            "[DIFF] Unexpected extra columns.\n"
            f"  Only in {t1}: {bad_only1}\n"
            f"  Only in {t2}: {bad_only2}\n"
            f"  Allowed extras: {sorted(ALLOWED_EXTRA_COLS)}"
        )

    if only1 or only2:
        print(
            f"[WARN] Column mismatch tolerated for {t1} vs {t2}.\n"
            f"  Only in {t1}: {only1}\n"
            f"  Only in {t2}: {only2}\n"
            f"  Comparing common columns only: {len(common)}"
        )

    for col in common:
        try:
            tb.open(t1)
            desc1 = tb.getcoldesc(col)
            tb.close()
            is_fixed = not desc1.get("isvar", False)
        except Exception:
            is_fixed = True

        if is_fixed:
            _compare_column_robust(t1, t2, col, nrow1, chunk_nrow=chunk_nrow)
        else:
            print(f'[INFO] Col {col} not fixed')

            tb.open(t1)
            va = tb.getvarcol(col)
            tb.close()

            tb.open(t2)
            vb = tb.getvarcol(col)
            tb.close()

            if va.keys() != vb.keys():
                die(f"[DIFF] varcol keys mismatch at {t1}:{col}")
            for k in va.keys():
                _compare_arrays_strict(np.asarray(va[k]), np.asarray(vb[k]), ctx=f"{t1}:{col}:{k}")

    # No-op test
    if t1.endswith(".ms") and t2.endswith(".ms"):
        # If CORRECTED_DATA exists it must equal DATA
        for ms in (t1, t2):
            tb.open(ms)
            cols = set(tb.colnames())
            tb.close()

            if "CORRECTED_DATA" in cols:
                print(f"[CHECK] {ms}: CORRECTED_DATA == DATA")
                _assert_col_equal(ms, "CORRECTED_DATA", "DATA", chunk_nrow=chunk_nrow)

            if "MODEL_DATA" in cols:
                print(f"[INFO] {ms}: MODEL_DATA present")


def compare_ms_deep(ms1: str, ms2: str):
    print("[INFO] Deep comparing MS tables/columns (strict)")
    tlist1 = list_ms_tables(ms1)
    tlist2 = list_ms_tables(ms2)

    set1 = set(os.path.relpath(p, ms1) for p in tlist1)
    set2 = set(os.path.relpath(p, ms2) for p in tlist2)
    if set1 != set2:
        die(f"[DIFF] Subtable set mismatch:\n  {ms1}: {sorted(set1)}\n  {ms2}: {sorted(set2)}")

    for rel in sorted(set1):
        t1 = ms1 if rel == "." else os.path.join(ms1, rel)
        t2 = ms2 if rel == "." else os.path.join(ms2, rel)
        print(f"  [CHECK] {rel}")
        compare_table(t1, t2, chunk_nrow=CHUNK_NROW)

    print("[PASS] MS deep-compare OK (strict)")


print("=== NO-OP corruption sanity check ===")
print(f"MS_IN : {MS_IN}")
print(f"MS_OUT: {MS_OUT}")

copy_ms(MS_IN, MS_OUT)
simulator_noop_corrupt(MS_OUT)

compare_ms_deep(MS_IN, MS_OUT)

print("=== ALL SANITY CHECKS PASSED ===")
