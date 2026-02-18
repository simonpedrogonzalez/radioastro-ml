#1. copies an input MS -> output MS
#2. runs simulator.reset simulator.corrupt on the output MS, a noop
#3. images both with same tclean params
#4. compares images pixel-per-pixel

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
    sm.reset() # reset list of corruption operations
    ok2 = sm.corrupt()
    if not ok2:
        die("simulator.corrupt failed")
    sm.close()

def make_dirty_image(msname: str, imagename: str):
    print(f"[INFO] Imaging (dirty) {msname} -> {imagename}.image")
    for suf in [".image", ".model", ".psf", ".residual", ".sumwt", ".pb", ".weight", ".mask"]:
        rmtables(imagename + suf)

    tclean(vis=msname, imagename=imagename, **TCLEAN_KW)

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

def compare_images_strict(img1: str, img2: str):
    ia = image()
    if not os.path.exists(img1):
        die(f"Missing image: {img1}")
    if not os.path.exists(img2):
        die(f"Missing image: {img2}")

    ia.open(img1)
    a = ia.getchunk()
    ia.close()

    ia.open(img2)
    b = ia.getchunk()
    ia.close()

    _compare_arrays_strict(np.asarray(a), np.asarray(b), ctx=f"IMAGE:{img1} vs {img2}")
    print("[PASS] Image pixel-by-pixel compare OK (strict)")


print("=== NO-OP corruption sanity check ===")
print(f"MS_IN : {MS_IN}")
print(f"MS_OUT: {MS_OUT}")

copy_ms(MS_IN, MS_OUT)
simulator_noop_corrupt(MS_OUT)

make_dirty_image(MS_IN, IMBASE_IN)
make_dirty_image(MS_OUT, IMBASE_OUT)

compare_images_strict(IMBASE_IN + ".image", IMBASE_OUT + ".image")

print("=== ALL SANITY CHECKS PASSED ===")
