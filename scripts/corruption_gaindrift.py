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
MS_IN  = "data/3c391_ctm_mosaic_10s_spw0.gaincal_corr.ms"
MS_OUT = "3c391_ctm_mosaic_10s_spw0.gaincal_corr.ms"

IMG_BEFORE_C = "img_gaincal_before_clean"
IMG_AFTER_C  = "img_gaincal_after_clean"

GAINCAL_FIELD = "J1822-0938"
SPW = "0"
SEED = 12345

GAIN_RMS_REAL = 0.2
GAIN_RMS_IMAG = 0.2

TCLEAN_KW = dict(
    field=GAINCAL_FIELD,
    spw=SPW,
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

def make_clean(msname, outbase):
    rm_im_products(outbase)
    print(f"[INFO] CLEAN imaging {msname} field={GAINCAL_FIELD} -> {outbase}.image/.residual")
    TCLEAN_KW.update(dict(
        niter=1000,
        # threshold="0.0Jy",
    ))
    tclean(vis=msname, imagename=outbase, **TCLEAN_KW)

def make_dirty(msname: str, outbase: str):
    rm_im_products(outbase)
    print(f"[INFO] Dirty imaging {msname} field={GAINCAL_FIELD} -> {outbase}.image")
    TCLEAN_CLEAN_KW.update(dict(
        niter=0,
        # threshold="0.0Jy",
    ))
    tclean(vis=msname, imagename=outbase, **TCLEAN_KW)

def copy_ms(src: str, dst: str):
    if not os.path.exists(src):
        die(f"Missing input MS: {src}")
    if os.path.exists(dst):
        print(f"[WARN] Removing existing {dst}")
        shutil.rmtree(dst)
    print(f"[INFO] Copying MS: {src} -> {dst}")
    shutil.copytree(src, dst)

from casatasks import flagdata, visstat

def rm_im_products(imbase: str):
    for suf in [".image", ".model", ".psf", ".residual", ".sumwt", ".pb", ".weight", ".mask"]:
        rmtables(imbase + suf)

def image_to_numpy(imname: str) -> np.ndarray:
    ia = image()
    ia.open(imname)
    arr = ia.getchunk()
    ia.close()
    return np.asarray(arr)

def img_peak(imname: str) -> float:
    arr = image_to_numpy(imname)
    return float(np.max(np.abs(arr)))

def img_rms(imname: str) -> float:
    arr = image_to_numpy(imname)
    return float(np.sqrt(np.mean(arr.astype(np.float64)**2)))

def simulator_obvious_gain_corrupt(msname: str):
    print(f"[INFO] Applying simulator gain corruption to {msname}")
    sm = simulator()
    sm.openfromms(msname)
    sm.reset()
    sm.setseed(SEED)

    sm.setgain(mode="fbm", amplitude=[GAIN_RMS_REAL, GAIN_RMS_IMAG])

    sm.corrupt()
    sm.done()

def make_diff(img_before, img_after, img_out):
    rm_im_products(img_out)
    a = image_to_numpy(img_before + ".image")
    b = image_to_numpy(img_after + ".image")
    equal = np.array_equal(a, b)
    print(f"[RESULT] Image equality (expected False): {equal}")
    if equal:
        print("[WARN] Images are equal.")
    else:
        d = b - a
        print(f"[INFO] diff stats: max|d|={np.max(np.abs(d))}  rms(d)={np.sqrt(np.mean(d*d))}")
    print(f"[INFO] Writing diff image {img_out}.image = AFTER - BEFORE")
    immath(
        imagename=[img_after + ".image", img_before + ".image"],
        expr="IM0 - IM1",
        outfile=img_out + ".image"
    )

def make_frac_residuals(residual_im: str,
    reference_im: str,
    out_im: str,
    ):
    # Compute reference scale
    peak = img_peak(reference_im)
    if peak == 0.0:
        raise RuntimeError(f"Reference image {reference_im} has zero peak")

    print(f"[INFO] Fractional residual: dividing {residual_im} by peak={peak:.6g}")

    rmtables(out_im)

    immath(
        imagename=[residual_im],
        expr=f"IM0/{peak}",
        outfile=out_im,
    )

from casatasks import plotms

def plot_before_after_vis_time(ms_before: str, ms_after: str, field: str, spw: str):
    """
      - before_amp_vs_time.png
      - before_phase_vs_time.png
      - after_amp_vs_time.png
      - after_phase_vs_time.png
    """
    def p(vis: str, yaxis: str, plotfile: str):
        plotms(
            vis=vis,
            field=field,
            spw=spw,
            xaxis="time",
            yaxis=yaxis,          # "amp" or "phase"
            avgchannel="9999",    # average over channels
            avgscan=True,
            coloraxis="antenna1", # shows antenna-dependent behavior
            showgui=False,
            plotfile=plotfile,
            overwrite=True,
        )

    p(ms_before, "amp",   "before_amp_vs_time.png")
    p(ms_before, "phase", "before_phase_vs_time.png")
    p(ms_after,  "amp",   "after_amp_vs_time.png")
    p(ms_after,  "phase", "after_phase_vs_time.png")

copy_ms(MS_IN, MS_OUT)
make_dirty(MS_IN, IMG_BEFORE)
make_clean(MS_IN,  IMG_BEFORE_C)

# Add corruption
simulator_obvious_gain_corrupt(MS_OUT)

make_dirty(MS_OUT, IMG_AFTER)
make_clean(MS_OUT, IMG_AFTER_C)

make_diff(IMG_BEFORE_C, IMG_AFTER_C)
make_diff(IMG_BEFORE, IMG_AFTER)

IMG_FRAC_RES = "img_gaincal_after_fracres.image"

# Fractional residuals: what fraction of the true source brightness
# is unexplained
make_frac_residuals(
    residual_im=f"{IMG_AFTER_C}.residual",
    reference_im="{IMG_BEFORE_C}.image",
    out_im=IMG_FRAC_RES,
)
rms_frac = img_rms(IMG_FRAC_RES)
print(f"[CHECK] Fractional RMS residual = {rms_frac:.6g}")

plot_before_after_vis_time(MS_IN, MS_OUT, GAINCAL_FIELD, SPW)

print("[DONE] Outputs:")
print(f"  - {MS_OUT}")
print(f"  - {IMG_BEFORE}.image")
print(f"  - {IMG_AFTER}.image")
print(f"  - {IMG_DIFF}.image")
print(f"  - {IMG_FRAC_RES}.image")

