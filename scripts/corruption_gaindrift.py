#1. copies MS_IN -> MS_OUT
#2. gain corruption casatools.simulator.setgain
#3. imaging
#4. writes diff image

import os
import shutil
import numpy as np

from casatasks import tclean, rmtables, immath
from casatools import simulator, image

#Settings
MS_IN  = "3c391_ctm_mosaic_10s_spw0.gaincal_corr.ms"
MS_OUT = "3c391_ctm_mosaic_10s_spw0.gaincal_corr.obviouscorrupt.ms"

GAINCAL_FIELD = "J1822-0938"
SPW = "0"
SEED = 12345

GAIN_RMS_REAL = 0.2
GAIN_RMS_IMAG = 0.2

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
    stokes="I",
    datacolumn="data",
    interactive=False,
    savemodel="none",
)

IMG_BEFORE = "img_gaincal_before"
IMG_AFTER  = "img_gaincal_after"
IMG_DIFF   = "img_gaincal_after_minus_before"

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

from casatasks import flagdata, visstat

def assert_has_data(msname: str, field: str, spw: str):
    # flag summary
    s = flagdata(vis=msname, mode="summary", field=field, spw=spw, display="none")
    # Quick: if CASA returns dict, look at flagged fraction
    print(f"[INFO] Flag summary keys: {list(s.keys())[:10]}")
    # visstat to catch NaNs/infs and zero npts
    st = visstat(vis=msname, datacolumn="data", field=field, spw=spw, axis="amp")
    print(f"[INFO] visstat npts={st.get('npts')} min={st.get('min')} max={st.get('max')} mean={st.get('mean')}")
    if st.get("npts", [0])[0] == 0:
        raise RuntimeError(f"No DATA points selected for field={field}, spw={spw} in {msname}")


def rm_im_products(imbase: str):
    for suf in [".image", ".model", ".psf", ".residual", ".sumwt", ".pb", ".weight", ".mask"]:
        rmtables(imbase + suf)

def make_dirty(msname: str, outbase: str):
    rm_im_products(outbase)
    print(f"[INFO] Dirty imaging {msname} field={GAINCAL_FIELD} -> {outbase}.image")
    tclean(vis=msname, imagename=outbase, **TCLEAN_KW)

def image_to_numpy(imname: str) -> np.ndarray:
    ia = image()
    ia.open(imname)
    arr = ia.getchunk()
    ia.close()
    return np.asarray(arr)

def simulator_obvious_gain_corrupt(msname: str):
    print(f"[INFO] Applying simulator gain corruption to {msname}")
    sm = simulator()
    sm.openfromms(msname)
    sm.reset()
    sm.setseed(SEED)

    sm.setgain(mode="fbm", amplitude=[GAIN_RMS_REAL, GAIN_RMS_IMAG])

    sm.corrupt()
    sm.done()


copy_ms(MS_IN, MS_OUT)
make_dirty(MS_IN, IMG_BEFORE)
simulator_obvious_gain_corrupt(MS_OUT)
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
if equal:
    print("[WARN] Images are equal.")
else:
    d = b - a
    print(f"[INFO] diff stats: max|d|={np.max(np.abs(d))}  rms(d)={np.sqrt(np.mean(d*d))}")

print("[DONE] Outputs:")
print(f"  - {MS_OUT}")
print(f"  - {IMG_BEFORE}.image")
print(f"  - {IMG_AFTER}.image")
print(f"  - {IMG_DIFF}.image")
