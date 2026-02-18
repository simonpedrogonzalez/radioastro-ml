# 1) copies MS_IN -> MS_OUT
# 2) trop corruption via casatools.simulator.settrop (screen model)
# 3) dirty imaging (niter=0)
# 4) writes diff image (AFTER - BEFORE)

import os
import shutil
import numpy as np

from casatasks import tclean, rmtables, immath, flagdata, visstat
from casatools import simulator, image, msmetadata


MS_IN  = "data/J1822_spw0.calibrated.ms"
MS_OUT = "data/J1822_spw0.tropcorrupt.ms"

GAINCAL_FIELD = "J1822-0938"
SPW = "0"
SEED = 12345

TROP_MODE = "screen"   # 'screen' or 'individual antennas' per docs
PWV_MM = 50.0        # total PWV [mm] (bigger -> stronger effect)
DELTAPWV_FRAC = 0.7   # RMS PWV fluctuations as fraction of PWV (big for "obvious")
BETA = 1.7             # exponent of fractional brownian motion
WINDSPEED_MPS = 44.0    # screen advection speed [m/s]
SIMINT_S = -1        # simulation timestep [s] (-1 lets tool choose)

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

IMG_BEFORE = "img_gaincal_trop_before"
IMG_AFTER  = "img_gaincal_trop_after"
IMG_DIFF   = "img_gaincal_trop_after_minus_before"


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

def get_field_id(msname: str, fieldname: str) -> int:
    md = msmetadata()
    md.open(msname)
    try:
        ids = md.fieldsforname(fieldname)
        if ids is None or len(ids) == 0:
            die(f"Could not find field '{fieldname}' in {msname}")
        if len(ids) > 1:
            print(f"[WARN] Multiple field IDs for name '{fieldname}': {ids}. Using first.")
        return int(ids[0])
    finally:
        md.close()

def simulator_obvious_trop_corrupt(msname: str, fieldname: str, spw_str: str):
    print(f"[INFO] Applying simulator trop corruption to {msname}")
    field_id = get_field_id(msname, fieldname)
    spw_id = int(spw_str)

    sm = simulator()
    sm.openfromms(msname)
    sm.reset()
    sm.setseed(SEED)

    sm.setdata(spwid=[spw_id], fieldid=[field_id])

    sm.settrop(
        mode=TROP_MODE,
        pwv=PWV_MM,
        deltapwv=DELTAPWV_FRAC,
        beta=BETA,
        windspeed=WINDSPEED_MPS,
        simint=SIMINT_S,
    )

    sm.corrupt()
    sm.done()



copy_ms(MS_IN, MS_OUT)

make_dirty(MS_IN, IMG_BEFORE)
simulator_obvious_trop_corrupt(MS_OUT, GAINCAL_FIELD, SPW)
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

if not equal:
    d = b - a
    finite = np.isfinite(d)
    if np.any(finite):
        print(f"[INFO] diff stats: max|d|={np.nanmax(np.abs(d))}  rms(d)={np.sqrt(np.nanmean(d*d))}")
    else:
        print("[WARN] Diff image appears to be all non-finite (NaN/Inf). Check selection and tclean outputs.")

print("[DONE] Outputs:")
print(f"  - {MS_OUT}")
print(f"  - {IMG_BEFORE}.image")
print(f"  - {IMG_AFTER}.image")
print(f"  - {IMG_DIFF}.image")
