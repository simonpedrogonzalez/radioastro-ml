# 1) copies MS_IN -> MS_OUT
# 2) "pointing-like" corruption via antenna-based amplitude drift (PB attenuation proxy)
# 3) dirty imaging before/after
# 4) writes diff image

import os
import shutil
import numpy as np

from casatasks import tclean, rmtables, immath, flagdata, visstat
from casatools import simulator, image

# Settings
MS_IN  = "3c391_ctm_mosaic_10s_spw0.gaincal_corr.ms"
MS_OUT = "3c391_ctm_mosaic_10s_spw0.gaincal_corr.pointinglike.ms"

GAINCAL_FIELD = "J1822-0938"
SPW = "0"
SEED = 12345

PB_FWHM_ARCSEC = 588.0
MISPOINT_ARCSEC = 120.0 

AMP_DRIFT_RMS = 0.05


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
    print(f"[INFO] Dirty imaging {msname} field={GAINCAL_FIELD} -> {outbase}.image")
    tclean(vis=msname, imagename=outbase, **TCLEAN_KW)

def image_to_numpy(imname: str) -> np.ndarray:
    ia = image()
    ia.open(imname)
    arr = ia.getchunk()
    ia.close()
    return np.asarray(arr)

def gaussian_pb_attenuation(theta_arcsec: float, fwhm_arcsec: float) -> float:
    sigma = fwhm_arcsec / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return float(np.exp(-0.5 * (theta_arcsec / sigma)**2))

def simulator_pointing_like_corrupt(msname: str):
    """
    Pointing-equivalent corruption:
      - Apply a mean amplitude loss equal to a (large) mispointing in the PB
      - Add slow fbm amplitude wander around that mean (standing in for pointing jitter/wander)
    """
    mean_att = gaussian_pb_attenuation(MISPOINT_ARCSEC, PB_FWHM_ARCSEC)
    print(f"[INFO] PB_FWHM={PB_FWHM_ARCSEC:.1f}\"  mispoint={MISPOINT_ARCSEC:.1f}\" -> mean attenuation ~ {mean_att:.4f}")

    sm = simulator()
    sm.openfromms(msname)
    sm.reset()
    sm.setseed(SEED)


    try:
        sm.setgain(mode="fbm",
                   amplitude=[AMP_DRIFT_RMS, 0.0],
                   offset=[mean_att, 0.0])
    except TypeError:
        print("[WARN] Your CASA 'setgain' may not accept offset=. Falling back to drift-only.")
        sm.setgain(mode="fbm", amplitude=[AMP_DRIFT_RMS, 0.0])

    sm.corrupt()
    sm.done()

# --- run ---
assert_has_data(MS_IN, GAINCAL_FIELD, SPW)

copy_ms(MS_IN, MS_OUT)

make_dirty(MS_IN, IMG_BEFORE)

simulator_pointing_like_corrupt(MS_OUT)

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
    print("[WARN] diff image contains no finite pixels (check upstream imaging / selection).")

print("[DONE] Outputs:")
print(f"  - {MS_OUT}")
print(f"  - {IMG_BEFORE}.image")
print(f"  - {IMG_AFTER}.image")
print(f"  - {IMG_DIFF}.image")
