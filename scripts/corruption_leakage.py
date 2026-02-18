# 1) copies MS_IN -> MS_OUT
# 2) leakage corruption casatools.simulator.setleakage
# 3) imaging (IQUV)
# 4) writes diff image (after - before) for each Stokes plane

import os
import shutil
import numpy as np

from casatasks import tclean, rmtables, immath, flagdata, visstat
from casatools import simulator, image

# Settings
MS_IN  = "3c391_ctm_mosaic_10s_spw0.gaincal_corr.ms"
MS_OUT = "3c391_ctm_mosaic_10s_spw0.gaincal_corr.leakage_obvious.ms"

GAINCAL_FIELD = "J1822-0938"
SPW = "0"
SEED = 12345

# --- Make leakage OBVIOUS ---
# Typical D-terms are ~0.005–0.03. For a test, use 0.10–0.20 (10–20%) so it screams.
# amplitude = [real, imag] magnitude-ish. We'll do purely real leakage to keep it simple.
LEAK_REAL = 0.15
LEAK_IMAG = 0.00
OFFSET_REAL = 0.00
OFFSET_IMAG = 0.00

TCLEAN_KW = dict(
    field=GAINCAL_FIELD,
    spw=SPW,
    specmode="mfs",
    gridder="standard",
    deconvolver="hogbom",
    niter=0,
    weighting="briggs",
    robust=0.5,
    imsize=[256, 256],
    cell=["2.5arcsec", "2.5arcsec"],
    stokes="IQUV",           # <-- important: leakage shows up mainly in Q/U/V
    datacolumn="data",
    interactive=False,
    savemodel="none",
)

IMG_BEFORE = "img_gaincal_before_leak"
IMG_AFTER  = "img_gaincal_after_leak"
IMG_DIFF   = "img_gaincal_after_minus_before_leak"

def die(msg: str):
    raise RuntimeError(msg)

def copy_ms(src: str, dst: str):
    if not os.path.exists(src):
        die(f"Missing input MS: {src}")
    if os.path.exists(dst):
        print(f"[WARN] Removing existing {dst}")
        shutil.rmtree(dst)
    print(f"[INFO] Copying MS: {src} -> {dst}")
    shutil.copytree(src, dst)

def assert_has_data(msname: str, field: str, spw: str):
    s = flagdata(vis=msname, mode="summary", field=field, spw=spw, display="none")
    print(f"[INFO] Flag summary keys: {list(s.keys())[:10]}")
    st = visstat(vis=msname, datacolumn="data", field=field, spw=spw, axis="amp")
    print(f"[INFO] visstat npts={st.get('npts')} min={st.get('min')} max={st.get('max')} mean={st.get('mean')}")
    if st.get("npts", [0])[0] == 0:
        raise RuntimeError(f"No DATA points selected for field={field}, spw={spw} in {msname}")

def rm_im_products(imbase: str):
    for suf in [".image", ".model", ".psf", ".residual", ".sumwt", ".pb", ".weight", ".mask"]:
        rmtables(imbase + suf)

def make_dirty(msname: str, outbase: str):
    rm_im_products(outbase)
    print(f"[INFO] Dirty imaging {msname} field={GAINCAL_FIELD} stokes={TCLEAN_KW['stokes']} -> {outbase}.image")
    tclean(vis=msname, imagename=outbase, **TCLEAN_KW)

def image_to_numpy(imname: str) -> np.ndarray:
    ia = image()
    ia.open(imname)
    arr = ia.getchunk()   # shape often (nx, ny, stokes, chan) for mfs
    ia.close()
    return np.asarray(arr)

def simulator_obvious_leakage_corrupt(msname: str):
    print(f"[INFO] Applying simulator leakage corruption to {msname}")
    sm = simulator()
    sm.openfromms(msname)
    sm.reset()
    sm.setseed(SEED)

    sm.setleakage(
        mode="constant",
        amplitude=[LEAK_REAL, LEAK_IMAG],
        offset=[OFFSET_REAL, OFFSET_IMAG],
    )

    sm.corrupt()
    sm.done()

# --- run ---
copy_ms(MS_IN, MS_OUT)

make_dirty(MS_IN, IMG_BEFORE)

simulator_obvious_leakage_corrupt(MS_OUT)

make_dirty(MS_OUT, IMG_AFTER)

rm_im_products(IMG_DIFF)
print(f"[INFO] Writing diff image {IMG_DIFF}.image = AFTER - BEFORE")
immath(
    imagename=[IMG_AFTER + ".image", IMG_BEFORE + ".image"],
    expr="IM0 - IM1",
    outfile=IMG_DIFF + ".image"
)

a = image_to_numpy(IMG_BEFORE + ".image")
b = image_to_numpy(IMG_AFTER + ".image")

equal = np.array_equal(a, b)
print(f"[RESULT] Image equality (expected False): {equal}")

d = b - a
finite = np.isfinite(d)
if np.any(finite):
    print(f"[INFO] diff stats: max|d|={np.nanmax(np.abs(d))}  rms(d)={np.sqrt(np.nanmean(d[finite]*d[finite]))}")
else:
    print("[WARN] diff image contains no finite pixels (check imaging selection / stokes availability).")

print("[DONE] Outputs:")
print(f"  - {MS_OUT}")
print(f"  - {IMG_BEFORE}.image   (IQUV)")
print(f"  - {IMG_AFTER}.image    (IQUV)")
print(f"  - {IMG_DIFF}.image     (IQUV)")
