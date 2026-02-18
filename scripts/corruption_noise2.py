#1. copies MS_IN -> MS_OUT
#2. noise corruption casatools.simulator.setnoise
#3. imaging
#4. writes diff image

import os
import shutil
import numpy as np

from casatasks import tclean, rmtables, immath
from casatools import simulator, image

#Settings
MS_IN  = "data/J1822_spw0.calibrated.ms"
MS_OUT = "J1822_spw0.noise.corrupted.ms"

IMG_BEFORE_C = "img_noise_before_clean"
IMG_AFTER_C  = "img_noise_after_clean"

GAINCAL_FIELD = "J1822-0938"
SPW = "0"
# SPW = "0:48~63"
SEED = 12345


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
    noise="0.0Jy"
)

IMG_BEFORE = "img_noise_before"
IMG_AFTER  = "img_noise_after"
IMG_DIFF   = "img_noise_diff"
IMG_DIFF_C = "img_noise_diff_clean"

NOISE_SIGMA = "10.0Jy"
NOISE_MODE  = "simplenoise"

import numpy as np
import hashlib
from casatools import table

import hashlib
import numpy as np
from casatools import table

def hash_casa_table_cols(
    tabpath: str,
    cols: list[str],
    *,
    row_stride: int = 1,
    max_rows: int | None = None,
) -> str:
    """
    Hash selected columns of a CASA table.

    Parameters
    ----------
    tabpath : str
        Path to CASA table (MS main table or caltable directory)
    cols : list of str
        Column names to hash (must exist in the table)
    row_stride : int
        Hash every k-th row (1 = all rows)
    max_rows : int or None
        Cap number of rows hashed (after striding)

    Returns
    -------
    sha256 hex digest
    """
    tb = table()
    tb.open(tabpath)

    h = hashlib.sha256()
    h.update(tabpath.encode())

    nrows = tb.nrows()
    rows = np.arange(0, nrows, row_stride)
    if max_rows is not None:
        rows = rows[:max_rows]

    h.update(np.int64(nrows).tobytes())
    h.update(np.int64(len(rows)).tobytes())

    for col in cols:
        if col not in tb.colnames():
            raise KeyError(f"Column '{col}' not in table {tabpath}")

        h.update(col.encode())

        data = tb.getcol(col)

        # CASA convention: last axis is row
        if isinstance(data, np.ndarray):
            data = np.take(data, rows, axis=data.ndim - 1)

            # Normalize dtype to avoid float/int ambiguity
            arr = np.ascontiguousarray(data)
            h.update(str(arr.dtype).encode())
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
        else:
            # strings / lists
            for i in rows:
                h.update(str(data[i]).encode())

    tb.close()
    return h.hexdigest()



def die(msg: str):
    raise RuntimeError(msg)

def make_clean(msname, outbase):
    rm_im_products(outbase)
    print(f"[INFO] CLEAN imaging {msname} field={GAINCAL_FIELD} -> {outbase}.image/.residual")
    tclean(vis=msname, imagename=outbase, niter=1000, **TCLEAN_KW)

def make_dirty(msname: str, outbase: str):
    rm_im_products(outbase)
    print(f"[INFO] Dirty imaging {msname} field={GAINCAL_FIELD} -> {outbase}.image")
    tclean(vis=msname, imagename=outbase, niter=0, **TCLEAN_KW)

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

from casatools import table


import numpy as np
from casatools import simulator, table
from casatasks import applycal



from casatools import table as tbtool
from casatools import msmetadata as msmdtool

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
    peak = img_peak(f"{reference_im}.image")
    if peak == 0.0:
        raise RuntimeError(f"Reference image {reference_im}.image has zero peak")

    print(f"[INFO] Fractional residual: dividing {residual_im}.residual by peak={peak:.6g}")

    rm_im_products(out_im)
    rmtables(f"{out_im}.image")

    immath(
        imagename=[f"{residual_im}.residual"],
        expr=f"IM0/{peak}",
        outfile=f"{out_im}.image",
    )

def check_was_applied():
    from casatools import table
    import numpy as np

    tb = table()
    tb.open(MS_OUT)
    d0 = tb.getcol("DATA")
    tb.close()

    tb = table()
    tb.open(MS_IN)
    d1 = tb.getcol("DATA")
    tb.close()

    print("||DATA(before) - DATA(after)|| =", np.linalg.norm(d0 - d1))


def sim_noise(msname: str):
    print(f"[INFO] Applying setnoise to {msname}")
    sm = simulator()
    sm.openfromms(msname)
    sm.reset()
    sm.setseed(SEED)

    ttab = msname + ".Ncorrupt"
    rmtables(ttab)

    print("[INFO] calling setnoise...")
    sm.setnoise(mode=NOISE_MODE, simplenoise=NOISE_SIGMA)

    print("[INFO] setnoise returned; table exists?", os.path.exists(ttab))

    print("[INFO] calling corrupt...")
    sm.corrupt()
    print("[INFO] corrupt returned")

    sm.done()
    print("[INFO] done; table exists?", os.path.exists(ttab))
    return ttab

# print(f"MS_IN hash: {hash_casa_table_cols(MS_IN, cols=["DATA"])}")
copy_ms(MS_IN, MS_OUT)
# print(f"MS_OUT hash:  {hash_casa_table_cols(MS_OUT, cols=["DATA"])}")

make_dirty(MS_IN, IMG_BEFORE)
make_clean(MS_IN,  IMG_BEFORE_C)

# Add corruption

ttab = sim_noise(MS_OUT)

# from casatools import table
# tb = table()
# tb.open(ttab)
# print("noise table columns:", tb.colnames())
# print("nrows:", tb.nrows())
# tb.close()


make_dirty(MS_OUT, IMG_AFTER)
make_clean(MS_OUT, IMG_AFTER_C)

make_diff(IMG_BEFORE_C, IMG_AFTER_C, IMG_DIFF_C)
make_diff(IMG_BEFORE, IMG_AFTER, IMG_DIFF)

IMG_FRAC_RES_BEFORE = "img_noise_before_fracres"
IMG_FRAC_RES_AFTER = "img_noise_after_fracres"
IMG_FRAC_RES_DIFF = "img_noise_fracres_diff"

# Fractional residuals: what fraction of the true source brightness
# is unexplained
make_frac_residuals(
    residual_im=IMG_AFTER_C,
    reference_im=IMG_BEFORE_C,
    out_im=IMG_FRAC_RES_AFTER,
)

make_frac_residuals(
    residual_im=IMG_BEFORE_C,
    reference_im=IMG_BEFORE_C,
    out_im=IMG_FRAC_RES_BEFORE,
)

make_diff(IMG_FRAC_RES_BEFORE, IMG_FRAC_RES_AFTER, IMG_FRAC_RES_DIFF)

check_was_applied()
# print(f"MS_OUT hash:  {hash_casa_table_cols(MS_OUT, cols=["DATA"])}")

print("[DONE] Outputs:")
print(f"  - {MS_OUT}")
print(f"  - {IMG_BEFORE}.image")
print(f"  - {IMG_AFTER}.image")
print(f"  - {IMG_DIFF}.image")
print(f"  - {IMG_FRAC_RES_BEFORE}.image")
print(f"  - {IMG_FRAC_RES_AFTER}.image")

