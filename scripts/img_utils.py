import os
import numpy as np
from casatasks import immath, rmtables, tclean, exportfits
from casatools import simulator, image
import matplotlib.pyplot as plt
from casatools import image as ia_tool
from astropy.io import fits
from astropy.wcs import WCS

def image_to_numpy(imname: str) -> np.ndarray:
    ia = image()
    ia.open(imname)
    arr = ia.getchunk()
    ia.close()
    return np.asarray(arr)

def rm_im_products(imbase: str):
    for suf in [".image", ".model", ".psf", ".residual", ".sumwt", ".pb", ".weight", ".mask"]:
        print("[INFO] Removing " + imbase + suf)
        rmtables(imbase + suf)
        if os.path.exists(imbase + suf):
            raise RuntimeError(
                f"Failed to remove {path}. Likely open in CARTA (or permissions). "
                f"Close it in CARTA or write to a new output name."
            )

def make_clean(msname: str, outbase: str, config: dict):
    rm_im_products(outbase)
    print(f"[INFO] CLEAN imaging {msname} field={config['field']} -> {outbase}.residual")
    tclean(vis=msname, imagename=outbase, niter=1000, **config)

def make_dirty(msname: str, outbase: str, config: dict):
    rm_im_products(outbase)
    print(f"[INFO] Dirty imaging {msname} field={config['field']} -> {outbase}.image")
    tclean(vis=msname, imagename=outbase, niter=0, **config)

def img_rms(imname: str) -> float:
    arr = image_to_numpy(imname)
    return float(np.sqrt(np.mean(arr.astype(np.float64)**2)))

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


def img_peak(imname: str) -> float:
    arr = image_to_numpy(imname)
    return float(np.max(np.abs(arr)))

def make_frac_residuals(residual_im: str,
    reference_im: str,
    out_im: str,
    ):
    # Compute reference scale
    peak = img_peak(f"{reference_im}.image")
    if peak == 0.0:
        raise RuntimeError(f"Reference image {reference_im}.image has zero peak")

    print(f"[INFO] Fractional residual: dividing {residual_im}.residual by peak={peak:.6g}")

    rm_im_products(out_im)

    immath(
        imagename=[f"{residual_im}.residual"],
        expr=f"IM0/{peak}",
        outfile=f"{out_im}.image",
    )

def casa_image_to_png(
    imname: str,
    out_png: str,
    chan: int = 0,
    stokes: int = 0,
    robust_pct: float = 99.5,
    symmetric: bool = True,
    log: bool = False,
    cmap: str = "inferno",
    dpi: int = 180,
):
    """
    Save a CASA image as a PNG with CARTA-like autoscaling and RA/Dec axes.
    """

    fitsname = out_png.replace(".png", ".fits")

    # 1) Export CASA image → FITS (keeps WCS)
    exportfits(
        imagename=imname,
        fitsimage=fitsname,
        overwrite=True,
        dropstokes=False,
        dropdeg=True,
    )

    # 2) Load FITS + WCS
    with fits.open(fitsname) as hdul:
        data = hdul[0].data
        wcs = WCS(hdul[0].header)

    # CASA/FITS order is usually (stokes, chan, y, x)
    data = np.squeeze(data)

    if data.ndim == 4:
        img2d = data[stokes, chan, :, :]
        wcs2d = wcs.celestial
    elif data.ndim == 3:
        img2d = data[chan, :, :]
        wcs2d = wcs.celestial
    else:
        img2d = data
        wcs2d = wcs.celestial

    finite = np.isfinite(img2d)
    if not np.any(finite):
        raise RuntimeError(f"No finite pixels in {imname}")

    vals = img2d[finite]

    # 3) CARTA-like autoscaling
    if log:
        vmin_tmp, vmax_tmp = np.percentile(vals, [100 - robust_pct, robust_pct])
        shift = 0.0 if vmin_tmp > 0 else -vmin_tmp + 1e-12
        img_plot = np.log10(np.maximum(img2d + shift, 1e-12))
        vals2 = img_plot[np.isfinite(img_plot)]
        vmin, vmax = np.percentile(vals2, [100 - robust_pct, robust_pct])
    else:
        if symmetric:
            vmax = np.percentile(np.abs(vals), robust_pct)
            vmin = -vmax
        else:
            vmin, vmax = np.percentile(vals, [100 - robust_pct, robust_pct])
        img_plot = img2d

    # 4) Plot with RA/Dec axes
    fig = plt.figure()
    ax = plt.subplot(projection=wcs2d)
    im = ax.imshow(
        img_plot,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"{imname}  (chan={chan}, stokes={stokes})")

    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()
