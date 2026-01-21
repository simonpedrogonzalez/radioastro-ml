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
MS_IN  = "data/J1822_spw0.calibrated.ms"
MS_OUT = "J1822_spw0.gaindrift.corrupted.ms"

IMG_BEFORE_C = "img_gaincal_before_clean"
IMG_AFTER_C  = "img_gaincal_after_clean"

GAINCAL_FIELD = "J1822-0938"
SPW = "0"
SEED = 12345

GAIN_RMS_AMP=0.7

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
IMG_DIFF   = "img_gaincal_diff"
IMG_DIFF_C = "img_gaincal_diff_clean"

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

def plot_gain_per_antenna(gtab, spw="0"):
    tb = table()
    tb.open(gtab)
    ants = sorted(set(tb.getcol("ANTENNA1")))
    tb.close()

    for ant in ants:
        ant_sel = f"{ant}"

        plotms(
            vis=gtab,
            spw=spw,
            antenna=ant_sel,
            xaxis="time",
            yaxis="gainamp",
            coloraxis="corr",
            plotfile=f"images/gainamp_ant{ant}.png",
            overwrite=True,
            showgui=False,
        )

        plotms(
            vis=gtab,
            spw=spw,
            antenna=ant_sel,
            xaxis="time",
            yaxis="gainphase",
            coloraxis="corr",
            plotfile=f"images/gainphase_ant{ant}.png",
            overwrite=True,
            showgui=False,
        )

# Example:
# 

import numpy as np
from casatools import simulator, table
from casatasks import applycal

def _mad(x):
    m = np.median(x)
    return np.median(np.abs(x - m))

def fix_gain_table(
    gtab_in: str,
    gtab_out: str = None,
    zmax: float = 10.0,          # robust-z threshold for |g| outliers
    fix_last_spike: bool = True, # also replace last point if it's an outlier
):
    """
    Create a sanitized copy of a CASA gain caltable by clipping/replacing
    extreme |g| values per antenna+correlation.

    Strategy:
      - compute |g| on unflagged rows
      - robust center = median(|g|)
      - robust scale  = 1.4826*MAD(|g|)
      - mark outliers where (|g|-median)/scale > zmax
      - replace outliers with previous good sample (time-ordered) if available,
        otherwise with median amplitude preserving phase.
    """
    if gtab_out is None:
        gtab_out = gtab_in + ".fixed"

    rmtables(gtab_out)

    tb = table()

    # copy table directory (CASA tables are directories)
    # safest is to use CASA table copy
    tb.open(gtab_in)
    tb.copy(gtab_out, deep=True)
    tb.close()

    tb.open(gtab_out, nomodify=False)

    TIME = tb.getcol("TIME")
    ANT  = tb.getcol("ANTENNA1")
    CP   = tb.getcol("CPARAM")  # (ncorr,1,nrow)
    FL   = tb.getcol("FLAG")    # (ncorr,1,nrow)

    ncorr = CP.shape[0]
    nrow  = CP.shape[2]

    # work per antenna + corr
    ants = np.unique(ANT)

    n_fixed = 0

    for ant in ants:
        m_ant = (ANT == ant)

        # time order within this antenna
        idx = np.where(m_ant)[0]
        idx = idx[np.argsort(TIME[idx])]

        for ic in range(ncorr):
            g = CP[ic, 0, idx].copy()
            f = FL[ic, 0, idx].copy()

            good = ~f
            if good.sum() < 10:
                continue

            amp = np.abs(g[good])

            med = np.median(amp)
            sig = 1.4826 * _mad(amp)
            if not np.isfinite(sig) or sig <= 0:
                sig = np.std(amp) + 1e-12

            # robust z
            amp_all = np.abs(g)
            z = (amp_all - med) / (sig + 1e-12)

            # outliers among unflagged only
            out = (~f) & (z > zmax)

            # also specifically target last-point spike if requested
            if fix_last_spike:
                last_good_pos = np.where(~f)[0]
                if len(last_good_pos) > 0:
                    j_last = last_good_pos[-1]
                    if z[j_last] > zmax:
                        out[j_last] = True
            
            # phase outliers
            # wrapped phase difference between consecutive samples
            # dphi[j] corresponds to jump from j-1 -> j (so dphi[0] is dummy)
            dphi = np.zeros_like(g, dtype=float)
            dphi[1:] = np.angle(g[1:] * np.conj(g[:-1]))   # in [-pi, pi]

            # consider increments where both endpoints are unflagged
            good_inc = (~f[1:]) & (~f[:-1])
            if good_inc.sum() >= 10:
                d = dphi[1:][good_inc]  # only valid increments

                # robust center/scale of phase increments
                med_d = np.median(d)
                sig_d = 1.4826 * _mad(d)
                if not np.isfinite(sig_d) or sig_d <= 0:
                    sig_d = np.std(d) + 1e-12

                z_d = np.abs((dphi - med_d) / (sig_d + 1e-12))

                # mark sample j as phase-outlier if jump into it is huge
                out_phase = np.zeros_like(out, dtype=bool)
                out_phase[1:] = (z_d[1:] > zmax) & (~f[1:]) & (~f[:-1])

                # optionally: explicitly target last sample if its jump is huge
                if fix_last_spike:
                    last_good_pos = np.where(~f)[0]
                    if len(last_good_pos) > 1:
                        j_last = last_good_pos[-1]
                        j_prev = last_good_pos[-2]
                        # "jump into last" is dphi[j_last] if consecutive indices,
                        # but if there are gaps, use direct jump from prev good
                        jump_last = np.angle(g[j_last] * np.conj(g[j_prev]))
                        # compare jump_last against typical increment scale
                        z_last = np.abs((jump_last - med_d) / (sig_d + 1e-12))
                        if z_last > zmax:
                            out_phase[j_last] = True

                out |= out_phase



            if not out.any():
                continue

            # replace outliers
            for j in np.where(out)[0]:
                # find previous good (non-flagged, non-outlier) sample
                prev = None
                for k in range(j-1, -1, -1):
                    if (not f[k]) and (not out[k]):
                        prev = k
                        break

                if prev is not None:
                    g[j] = g[prev]  # copy complex gain
                else:
                    # fallback: set amplitude to median but keep phase
                    phase = np.angle(g[j]) if amp_all[j] > 0 else 0.0
                    g[j] = med * np.exp(1j * phase)

                n_fixed += 1

            # write back
            CP[ic, 0, idx] = g

    tb.putcol("CPARAM", CP)
    tb.close()

    print(f"[INFO] Wrote fixed gain table: {gtab_out}")
    print(f"[INFO] Replaced {n_fixed} outlier gain entries (z>{zmax})")
    return gtab_out

def sim_gain_corrupt_clip_extreme(msname: str):
    print(f"[INFO] Generating gain corruption table for {msname}")

    sm = simulator()
    sm.openfromms(msname)
    sm.setseed(SEED)

    gtab = msname + ".Gcorrupt"
    rmtables(gtab) 
    # NOTE: interval arg is ignored for fbm in CASA (as you've found)
    sm.setgain(mode="fbm", table=gtab, amplitude=GAIN_RMS_AMP, interval="10m")

    sm.done()

    # sm.corrupt()
    # sm.done()

    print(f"[INFO] Raw gain table: {gtab}")

    gtab_fixed = fix_gain_table(gtab, zmax=10.0, fix_last_spike=True)

    print(f"[INFO] Applying fixed gain table to MS: {msname}")
    
    sm = simulator()
    sm.openfromms(msname)
    sm.setseed(SEED)
    sm.setapply(table=gtab_fixed, type='G', field=GAINCAL_FIELD, interp='linear', calwt=False)
    sm.corrupt()
    sm.done()

    return gtab_fixed

from casatools import table as tbtool
from casatools import msmetadata as msmdtool

def set_multiplicative_gains_to_1_for_all_except_one_antenna(
    gtab_out: str,
    msname: str, # just to get antenna names
    keep_ant: int
):

    if gtab_out is None:
        gtab_out = f"{gtab_in}.keepAnt{keep_ant}"

    tb = tbtool()
    tb.open(gtab_out, nomodify=False)
    try:
        colnames = tb.colnames()

        # Identify antenna column name
        if "ANTENNA1" in colnames:
            ant1 = tb.getcol("ANTENNA1")
        elif "ANTENNA" in colnames:
            ant1 = tb.getcol("ANTENNA")
        else:
            raise RuntimeError(f"Could not find ANTENNA1/ANTENNA in {gtab_out}. Columns: {colnames}")

        # Neutralize all rows except for the ones with this antenna involved
        rows = np.where(ant1 != keep_ant)[0]
        if rows.size == 0:
            print(f"[INFO] keep_only_one_antenna_in_gtab: nothing to change (only antenna {keep_ant} present?)")
            return gtab_out

        # Overwrite CPARAM (complex gains) if present
        if "CPARAM" in colnames:
            cparam = tb.getcol("CPARAM")  # (npol, nchan, nrow)
            cparam[:, :, rows] = 1.0 + 0.0j
            tb.putcol("CPARAM", cparam)
        else:
            # Some tables may use FPARAM (rare for gain tables); handle just in case.
            if "FPARAM" in colnames:
                print("[WARNING] FPARAM present when setting antenna gains to 1+0i")
                fparam = tb.getcol("FPARAM")
                # Best-effort: set to 1 in the same row positions
                # (shape can vary depending on table type)
                if fparam.ndim == 3:
                    fparam[:, :, rows] = 1.0
                elif fparam.ndim == 2:
                    fparam[:, rows] = 1.0
                else:
                    raise RuntimeError(f"Unexpected FPARAM shape {fparam.shape} in {gtab_out}")
                tb.putcol("FPARAM", fparam)
            else:
                raise RuntimeError(f"No CPARAM/FPARAM in {gtab_out}. Columns: {colnames}")

        tb.flush()
    finally:
        tb.close()

    print(f"[INFO] Wrote single-antenna corruption table: {gtab_out} (kept ant id {keep_ant})")
    return gtab_out


def sim_gain_corrupt_clip_extreme_single_antenna(msname: str, keep_ant: int):

    print(f"[INFO] Generating gain corruption table for {msname}")

    sm = simulator()
    sm.openfromms(msname)
    sm.setseed(SEED)

    gtab = msname + ".Gcorrupt"
    rmtables(gtab) 
    # NOTE: interval arg is ignored for fbm in CASA (as you've found)
    sm.setgain(mode="fbm", table=gtab, amplitude=GAIN_RMS_AMP, interval="10m")

    sm.done()

    # sm.corrupt()
    # sm.done()

    print(f"[INFO] Raw gain table: {gtab}")

    gtab_fixed = fix_gain_table(gtab, zmax=10.0, fix_last_spike=True)
    gtab_fixed = set_multiplicative_gains_to_1_for_all_except_one_antenna(gtab_fixed, msname, keep_ant)

    print(f"[INFO] Applying fixed gain table to MS: {msname}")
    
    sm = simulator()
    sm.openfromms(msname)
    sm.setseed(SEED)
    sm.setapply(table=gtab_fixed, type='G', field=GAINCAL_FIELD, interp='linear', calwt=False)
    sm.corrupt()
    sm.done()

    return gtab_fixed




def sim_gain_corrupt(msname: str):
    print(f"[INFO] Applying simulator gain corruption to {msname}")
    sm = simulator()
    sm.openfromms(msname)
    sm.setseed(SEED)
    gtab = msname + ".Gcorrupt"
    rmtables(gtab) 
    sm.setgain(mode="fbm", table=gtab, amplitude=GAIN_RMS_AMP)

    sm.corrupt()
    sm.done()

    return gtab


def shift_gain_table_to_unity(gtab_out: str):
    """
    Convert a gtab with gains given by N(0, sigma) to 1 + N(0, sigma) so they can be 
    added multiplicatively by the corruptor
    """

    tb = tbtool()
    tb.open(gtab_out, nomodify=False)
    try:
        if "CPARAM" not in tb.colnames():
            raise RuntimeError(f"{gtab_out} has no CPARAM column")

        c = tb.getcol("CPARAM")  # shape (npol, nchan, nrow)

        # Shift to unity (complex)
        c = c + (1.0 + 0.0j)

        tb.putcol("CPARAM", c)
        tb.flush()
    finally:
        tb.close()

    return gtab_out

def sim_gain_corrupt_random(msname: str):
    print(f"[INFO] Applying random simulator gain corruption to {msname}")
    sm = simulator()
    sm.openfromms(msname)
    sm.setseed(SEED)
    gtab = msname + ".Gcorrupt"
    rmtables(gtab) 
    sm.setgain(mode="random", table=gtab, amplitude=GAIN_RMS_AMP)
    sm.done()

    gtab_fixed = shift_gain_table_to_unity(gtab)

    print(f"[INFO] Applying fixed gain table to MS: {msname}")
    
    sm = simulator()
    sm.openfromms(msname)
    sm.setseed(SEED)
    sm.setapply(table=gtab_fixed, type='G', field=GAINCAL_FIELD, interp='linear', calwt=False)
    sm.corrupt()
    sm.done()

    return gtab_fixed


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

    rmtables(f"{out_im}.image")

    immath(
        imagename=[f"{residual_im}.residual"],
        expr=f"IM0/{peak}",
        outfile=f"{out_im}.image",
    )


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
            avgscan=False,
            coloraxis="antenna1", # shows antenna-dependent behavior
            showgui=False,
            plotfile=f"images/{plotfile}",
            overwrite=True,
            showlegend=True,
            correlation="RR"
        )

    p(ms_before, "amp",   "before_amp_vs_time.png")
    p(ms_before, "phase", "before_phase_vs_time.png")
    p(ms_after,  "amp",   "after_amp_vs_time.png")
    p(ms_after,  "phase", "after_phase_vs_time.png")

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


print(f"MS_IN hash: {hash_casa_table_cols(MS_IN, cols=["DATA"])}")
copy_ms(MS_IN, MS_OUT)
print(f"MS_OUT hash:  {hash_casa_table_cols(MS_OUT, cols=["DATA"])}")

make_dirty(MS_IN, IMG_BEFORE)
make_clean(MS_IN,  IMG_BEFORE_C)

# Add corruption
gtab = sim_gain_corrupt(MS_OUT)
# gtab = sim_gain_corrupt_clip_extreme(MS_OUT)
# gtab = sim_gain_corrupt_clip_extreme_single_antenna(MS_OUT, keep_ant=3)
# gtab = sim_gain_corrupt_random(MS_OUT)

print(f"GTAB hash:  {hash_casa_table_cols(gtab, cols=["TIME", "ANTENNA1", "CPARAM"])}")

plot_gain_per_antenna(gtab, spw="0")

make_dirty(MS_OUT, IMG_AFTER)
make_clean(MS_OUT, IMG_AFTER_C)

make_diff(IMG_BEFORE_C, IMG_AFTER_C, IMG_DIFF_C)
make_diff(IMG_BEFORE, IMG_AFTER, IMG_DIFF)

IMG_FRAC_RES = "img_gaincal_after_fracres"

# Fractional residuals: what fraction of the true source brightness
# is unexplained
make_frac_residuals(
    residual_im=IMG_AFTER_C,
    reference_im=IMG_BEFORE_C,
    out_im=IMG_FRAC_RES,
)
# rms_frac = img_rms(f"{IMG_FRAC_RES}.image")
# print(f"[CHECK] Fractional RMS residual = {rms_frac:.6g}")

plot_before_after_vis_time(MS_IN, MS_OUT, GAINCAL_FIELD, SPW)
check_was_applied()
print(f"MS_OUT hash:  {hash_casa_table_cols(MS_OUT, cols=["DATA"])}")

print("[DONE] Outputs:")
print(f"  - {MS_OUT}")
print(f"  - {IMG_BEFORE}.image")
print(f"  - {IMG_AFTER}.image")
print(f"  - {IMG_DIFF}.image")
print(f"  - {IMG_FRAC_RES}.image")

