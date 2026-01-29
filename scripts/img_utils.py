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
        # print("[INFO] Removing " + imbase + suf)
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

def crop_center(img2d: np.ndarray, half_size: int = 64) -> np.ndarray:
    ny, nx = img2d.shape
    cy, cx = ny // 2, nx // 2
    y0, y1 = max(0, cy - half_size), min(ny, cy + half_size)
    x0, x1 = max(0, cx - half_size), min(nx, cx + half_size)
    return img2d[y0:y1, x0:x1]

def fracres_metrics(frac_image_path: str, crop_half: int | None = 64) -> dict:
    arr = image_to_numpy(frac_image_path)
    arr = np.squeeze(arr)

    if arr.ndim == 4:
        arr = arr[:, :, 0, 0]
    elif arr.ndim == 3:
        arr = arr[:, :, 0]
    img = arr.T

    if crop_half is not None:
        img = crop_center(img, half_size=crop_half)

    x = img[np.isfinite(img)]
    ax = np.abs(x)

    out = {}
    out["rms"] = float(np.sqrt(np.mean(x**2)))
    out["p99_abs"] = float(np.percentile(ax, 99))
    for thr in [0.005, 0.01, 0.02]:
        out[f"frac_abs_gt_{thr}"] = float(np.mean(ax > thr))
    return out

def compare_fracres(before_img: str, after_img: str, crop_half: int | None = 64) -> dict:
    b = fracres_metrics(before_img, crop_half=crop_half)
    a = fracres_metrics(after_img, crop_half=crop_half)

    ratios = {}
    for k in b.keys():
        denom = b[k]
        ratios[k] = float(a[k] / denom) if denom not in (0.0, np.nan) and np.isfinite(denom) else float("inf")

    return {"before": b, "after": a, "ratio_after_over_before": ratios}


from matplotlib.patches import Rectangle
from astropy.wcs import WCS
from astropy.wcs import FITSFixedWarning
import warnings


def _robust_sigma(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad)


def _load_casa_image_as_2d_with_wcs(imname: str, chan: int = 0, stokes: int = 0, fitsname: str | None = None):
    """
    Export CASA image to FITS, load (img2d, wcs2d, fitsname).
    img2d is a 2D numpy array in FITS pixel order (y, x).
    """
    if fitsname is None:
        fitsname = imname.rstrip("/").replace(".image", "") + ".tmp_export.fits"

    exportfits(
        imagename=imname,
        fitsimage=fitsname,
        overwrite=True,
        dropstokes=False,
        dropdeg=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        with fits.open(fitsname) as hdul:
            data = np.squeeze(hdul[0].data)
            wcs = WCS(hdul[0].header)

    # Typical FITS order from CASA export: (stokes, chan, y, x)
    if data.ndim == 4:
        img2d = data[stokes, chan, :, :]
        wcs2d = wcs.celestial
    elif data.ndim == 3:
        img2d = data[chan, :, :]
        wcs2d = wcs.celestial
    elif data.ndim == 2:
        img2d = data
        wcs2d = wcs.celestial
    else:
        raise RuntimeError(f"Unexpected FITS data ndim={data.ndim} for {imname}")

    return img2d, wcs2d, fitsname


def _center_crop_slices(ny: int, nx: int, half_size: int):
    cy, cx = ny // 2, nx // 2
    y0, y1 = max(0, cy - half_size), min(ny, cy + half_size)
    x0, x1 = max(0, cx - half_size), min(nx, cx + half_size)
    return slice(y0, y1), slice(x0, x1)


def fracres_before_after_png(
    before_im: str,
    after_im: str,
    out_png: str,
    *,
    chan: int = 0,
    stokes: int = 0,
    crop_half: int = 64,
    robust_pct: float = 99.5,
    symmetric: bool = True,
    cmap: str = "inferno",
    dpi: int = 180,
    keep_fits: bool = False,
    thresholds: tuple[float, ...] = (0.005, 0.01, 0.02),
):
   # load
    before2d, wcs2d_b, fits_b = _load_casa_image_as_2d_with_wcs(before_im, chan=chan, stokes=stokes)
    after2d,  wcs2d_a, fits_a = _load_casa_image_as_2d_with_wcs(after_im,  chan=chan, stokes=stokes)

    if before2d.shape != after2d.shape:
        raise RuntimeError(f"Shape mismatch: before {before2d.shape} vs after {after2d.shape}")

    ny, nx = before2d.shape
    ys, xs = _center_crop_slices(ny, nx, half_size=crop_half)
    b_crop = before2d[ys, xs]
    a_crop = after2d[ys, xs]

    print(f"[ROI] y: {ys.start}:{ys.stop}  x: {xs.start}:{xs.stop}")
    print(f"[ROI] shape: {(ys.stop-ys.start, xs.stop-xs.start)}  (expected {(2*crop_half, 2*crop_half)})")
    print(f"[IMG] full shape (ny,nx): {before2d.shape}")

    def _metrics(x: np.ndarray) -> dict:
        x = x[np.isfinite(x)]
        ax = np.abs(x)
        out = {}
        out["rms"] = float(np.sqrt(np.mean(x**2))) if x.size else float("nan")
        out["p99_abs"] = float(np.percentile(ax, 99)) if x.size else float("nan")
        for thr in thresholds:
            out[f"area>|{thr}|"] = float(np.mean(ax > thr)) if x.size else float("nan")
        return out

    b_m = _metrics(b_crop)
    a_m = _metrics(a_crop)

    def _ratio(a: float, b: float) -> float:
        if not np.isfinite(a) or not np.isfinite(b):
            return float("nan")
        if b == 0.0:
            return float("inf") if a > 0 else 1.0
        return float(a / b)

    r_m = {k: _ratio(a_m[k], b_m[k]) for k in b_m.keys()}

    # shared scale across both images
    finite_all = np.isfinite(before2d) & np.isfinite(after2d)
    if not np.any(finite_all):
        raise RuntimeError("No finite pixels in either image.")

    vals_all = np.concatenate([before2d[finite_all], after2d[finite_all]])
    if symmetric:
        vmax = float(np.percentile(np.abs(vals_all), robust_pct))
        vmin = -vmax
    else:
        vmin, vmax = np.percentile(vals_all, [100 - robust_pct, robust_pct])
        vmin, vmax = float(vmin), float(vmax)

    # ROI rectangle (pixel coords)
    y0, x0 = ys.start, xs.start
    h, w = (ys.stop - ys.start), (xs.stop - xs.start)

    # ----- Stable layout: 2 image axes + 1 colorbar axis + 1 stats axis -----
    fig = plt.figure(figsize=(12.5, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=2, ncols=3,
        height_ratios=[1.0, 0.28],
        width_ratios=[1.0, 1.0, 0.05],
    )

    ax1 = fig.add_subplot(gs[0, 0], projection=wcs2d_b)
    ax2 = fig.add_subplot(gs[0, 1], projection=wcs2d_a)
    cax = fig.add_subplot(gs[0, 2])      # dedicated colorbar axis
    tax = fig.add_subplot(gs[1, :])      # dedicated text axis
    tax.axis("off")

    im1 = ax1.imshow(before2d, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
    im2 = ax2.imshow(after2d,  origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)

    for ax, title in [(ax1, "Before (frac-res)"), (ax2, "After (frac-res)")]:
        ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor="white", linewidth=1.5))
        ax.set_title(title)
        ax.set_xlabel("RA")
        ax.set_ylabel("Dec")

    cb = fig.colorbar(im2, cax=cax)
    cb.set_label("fraction of peak")

    # Stats text (simple, one block)
    lines = []
    lines.append(f"ROI: center crop ±{crop_half}px ({w}×{h} px)")
    lines.append("Metric            before     after    ratio")
    lines.append(f"RMS             {b_m['rms']:.3g}   {a_m['rms']:.3g}   {r_m['rms']:.2f}×")
    lines.append(f"p99 |x|         {b_m['p99_abs']:.3g}   {a_m['p99_abs']:.3g}   {r_m['p99_abs']:.2f}×")
    for thr in thresholds:
        k = f"area>|{thr}|"
        lines.append(f"|x|>{thr:<6}     {100*b_m[k]:5.2f}%   {100*a_m[k]:5.2f}%   {r_m[k]:.2f}×")

    tax.text(0.01, 0.95, "\n".join(lines), ha="left", va="top",
             family="monospace", fontsize=10)

    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)

    if not keep_fits:
        for f in [fits_b, fits_a]:
            try:
                os.remove(f)
            except OSError:
                pass
