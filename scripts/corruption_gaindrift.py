#1. copies MS_IN -> MS_OUT
#2. applies some gain corruption
#3. makes a bunch of plots

# SETUP
import os; os.environ.setdefault("DISPLAY", ":0")
# MY HELPERS
import importlib
from scripts import io_utils, img_utils
for lib in [io_utils, img_utils]:
    importlib.reload(lib)
from .img_utils import make_frac_residuals, casa_image_to_png, make_clean, make_dirty, make_diff, img_rms, fracres_before_after_png
from .io_utils import copy_ms, hash_casa_table_cols, col_diff

import numpy as np
from casatasks import tclean, rmtables, immath, gaincal, ft
from casaplotms import plotms
import hashlib
from casatasks import flagdata, visstat, applycal
from casatools import simulator, table, image
from casatools import msmetadata as msmdtool

#Settings
MS_IN  = "data/J1822_spw0.calibrated.ms"
MS_OUT = "J1822_spw0.gaindrift.corrupted.ms"

GAINCAL_FIELD = "J1822-0938"
SPW = "0"
SEED = 12345

GAIN_RMS_AMP=1.0

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

def rm_im_products(imbase: str):
    for suf in [".image", ".model", ".psf", ".residual", ".sumwt", ".pb", ".weight", ".mask"]:
        rmtables(imbase + suf)


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
    sm.setgain(mode="fbm", table=gtab, amplitude=GAIN_RMS_AMP)

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


def set_multiplicative_gains_to_1_for_all_except_one_antenna(
    gtab_out: str,
    msname: str, # just to get antenna names
    keep_ant: int
):

    if gtab_out is None:
        gtab_out = f"{gtab_in}.keepAnt{keep_ant}"

    tb = table()
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

    tb = table()
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


def set_multiplicative_gains_to_1_for_all_except_some_antennas(
    gtab_in: str,
    msname: str,                 # kept for symmetry with your other funcs (not strictly needed)
    keep_ants: list[int],
    gtab_out: str | None = None,
):
    """
    Copy gain table gtab_in -> gtab_out, then set CPARAM=1+0j for all rows whose antenna
    is NOT in keep_ants. Leaves rows for antennas in keep_ants untouched.

    This assumes the gain table is a per-antenna table (ANTENNA1 present), like setgain() outputs.
    """
    keep_ants = sorted(set(int(a) for a in keep_ants))
    if len(keep_ants) == 0:
        raise ValueError("keep_ants is empty")

    if gtab_out is None:
        keep_str = "_".join(map(str, keep_ants[:12]))
        gtab_out = f"{gtab_in}.keep{len(keep_ants)}ants_{keep_str}"

    # Deep copy the table directory first (so we don't mutate gtab_in)
    tb = table()
    tb.open(gtab_in)
    tb.copy(gtab_out, deep=True)
    tb.close()

    tb.open(gtab_out, nomodify=False)
    try:
        colnames = tb.colnames()

        # Antenna column
        if "ANTENNA1" in colnames:
            antcol = "ANTENNA1"
        elif "ANTENNA" in colnames:
            antcol = "ANTENNA"
        else:
            raise RuntimeError(f"Could not find ANTENNA1/ANTENNA in {gtab_out}. Columns: {colnames}")

        ant = tb.getcol(antcol)

        # Rows to neutralize: antennas NOT in keep_ants
        keep_set = set(keep_ants)
        rows = np.array([i for i, a in enumerate(ant) if int(a) not in keep_set], dtype=np.int64)

        if rows.size == 0:
            print(f"[INFO] keep_ants covers all antennas present; nothing to neutralize.")
            return gtab_out

        # Prefer CPARAM; fallback FPARAM if needed
        if "CPARAM" in colnames:
            cparam = tb.getcol("CPARAM")  # (ncorr, nchan(usually 1), nrow)
            cparam[:, :, rows] = 1.0 + 0.0j
            tb.putcol("CPARAM", cparam)
        elif "FPARAM" in colnames:
            fparam = tb.getcol("FPARAM")
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

    print(f"[INFO] Wrote multi-antenna corruption table: {gtab_out}")
    print(f"[INFO] Kept corruption for antennas: {keep_ants}")
    print(f"[INFO] Neutralized gains for {rows.size} rows (antennas not in keep_ants)")
    return gtab_out


def sim_gain_corrupt_clip_extreme_n_antennas(msname: str, keep_ants: list[int], amp=None):
    """
    Generate FBM gain table, fix outliers, then keep corruption only for keep_ants
    (provided via keep_ants); neutralize all others to unity; apply to MS.
    """
    keep_ants = list(keep_ants)

    print(f"[INFO] Generating gain corruption table for {msname}")

    sm = simulator()
    sm.openfromms(msname)
    sm.setseed(SEED)

    gtab = msname + ".Gcorrupt"
    rmtables(gtab)

    if amp is None:
        amp = GAIN_RMS_AMP

    sm.setgain(mode="fbm", table=gtab, amplitude=amp)
    sm.done()

    print(f"[INFO] Raw gain table: {gtab}")

    gtab_fixed = fix_gain_table(gtab, zmax=10.0, fix_last_spike=True)
    gtab_fixed = set_multiplicative_gains_to_1_for_all_except_some_antennas(
        gtab_in=gtab_fixed,
        msname=msname,
        keep_ants=keep_ants,
        gtab_out=None,
    )

    print(f"[INFO] Applying fixed gain table to MS: {msname}")

    sm = simulator()
    sm.openfromms(msname)
    sm.setseed(SEED)
    sm.setapply(table=gtab_fixed, type="G", field=GAINCAL_FIELD, interp="linear", calwt=False)
    sm.corrupt()
    sm.done()

    return gtab_fixed

def plot_gtab_corrupt_vs_recovered_all_ants_2x2(
    corr_gtab: str,
    cal_gtab: str,
    spw: str = "0",
    out_png: str = "images/gtab_corrupt_vs_recovered_2x2.png",
    overwrite: bool = True,
    cleanup_tmp: bool = True,
):
    """
    2x2 grid:
      [ corrupt: gainamp  | corrupt: gainphase ]
      [ recov : gainamp   | recov : gainphase  ]

    All antennas shown, colored by antenna (ANTENNA1).
    """
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    os.makedirs("images", exist_ok=True)

    def p(vis: str, yaxis: str, plotfile: str):
        plotms(
            vis=vis,
            spw=spw,
            xaxis="time",
            yaxis=yaxis,            # "gainamp" or "gainphase"
            coloraxis="antenna1",   # color by antenna id (gtables are per-antenna)
            showgui=False,
            plotfile=plotfile,
            overwrite=overwrite,
            showlegend=True,
        )

    # 1) Generate the four plots
    tmp = {
        "c_amp":   "images/_tmp_gtab_corrupt_gainamp.png",
        "c_phase": "images/_tmp_gtab_corrupt_gainphase.png",
        "r_amp":   "images/_tmp_gtab_recovered_gainamp.png",
        "r_phase": "images/_tmp_gtab_recovered_gainphase.png",
    }

    p(corr_gtab, "gainamp",   tmp["c_amp"])
    p(corr_gtab, "gainphase", tmp["c_phase"])
    p(cal_gtab,  "gainamp",   tmp["r_amp"])
    p(cal_gtab,  "gainphase", tmp["r_phase"])

    # 2) Stitch into a 2x2 grid
    imgs = [
        Image.open(tmp["c_amp"]).convert("RGB"),
        Image.open(tmp["c_phase"]).convert("RGB"),
        Image.open(tmp["r_amp"]).convert("RGB"),
        Image.open(tmp["r_phase"]).convert("RGB"),
    ]

    # Use smallest tile size to avoid upscaling
    w = min(im.size[0] for im in imgs)
    h = min(im.size[1] for im in imgs)
    imgs = [im.resize((w, h), resample=Image.Resampling.LANCZOS) for im in imgs]

    canvas = Image.new("RGB", (2 * w, 2 * h), (255, 255, 255))
    canvas.paste(imgs[0], (0,   0))
    canvas.paste(imgs[1], (w,   0))
    canvas.paste(imgs[2], (0,   h))
    canvas.paste(imgs[3], (w,   h))
    canvas.save(out_png)

    if cleanup_tmp:
        for path in tmp.values():
            try:
                os.remove(path)
            except OSError:
                pass

    print(f"[INFO] Wrote 2x2 gain-table comparison: {out_png}")
    return out_png

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


def sim_all(msname: str):
    """
    Apply, in one go:
      1) reasonable tropospheric phase screen (settrop, screen mode)
      2) reasonable thermal noise using ATM model (setnoise, tsys-atm) with the SAME PWV
      3) multiplicative gain drift (fbm) applied ONLY to antennas 0-6 (first 7 ants)

    Assumes you already have:
      - SEED, GAINCAL_FIELD, GAIN_RMS_AMP (or we set a local one)
      - fix_gain_table(...)
      - set_multiplicative_gains_to_1_for_all_except_some_antennas(...)
    """
    print(f"[INFO] sim_all: applying trop + tsys-atm noise + partial gain drift to {msname}")

    # ----------------------------
    # 1) Choose "reasonable" values
    # ----------------------------
    # Moderate-ish PWV (mm). 0.5–2 is good/typical-ish, 3–5 is poorer weather.
    PWV_MM = 100000

    # Troposphere "screen" parameters (phase screen made from fluctuating PWV)
    # deltapwv is the RMS fluctuation (mm). 0.05–0.3 is a reasonable range.
    DELTAPWV_MM = 1.0
    BETA = 1.9
    WINDSPEED_MPS = 10000.0

    # Gain drift strength (dimensionless). Keep modest so it looks "real" not insane.
    GAIN_DRIFT_AMP = 1.0

    # Only these antennas get gain drift
    KEEP_ANTS = [0, 1, 2, 3, 4, 5, 6]

    # ----------------------------
    # 2) Build a gain corruption table (fbm), then sanitize, then keep only ants 0-6
    # ----------------------------
    gtab_raw = msname + ".Gcorrupt"
    rmtables(gtab_raw)

    sm = simulator()
    sm.openfromms(msname)
    sm.setseed(SEED)

    # Generate the table (do NOT corrupt yet — we will apply via setapply after editing)
    sm.setgain(mode="fbm", table=gtab_raw, amplitude=GAIN_DRIFT_AMP, interval="10m")
    sm.done()

    print(f"[INFO] sim_all: raw gain table: {gtab_raw}")

    # Fix spikes/outliers (your existing robust clipper)
    gtab_fixed = fix_gain_table(gtab_raw, zmax=10.0, fix_last_spike=True)

    # Neutralize all antennas except KEEP_ANTS (your helper)
    gtab_partial = set_multiplicative_gains_to_1_for_all_except_some_antennas(
        gtab_in=gtab_fixed,
        msname=msname,
        keep_ants=KEEP_ANTS,
        gtab_out=None,
    )

    # ----------------------------
    # 3) Apply: (trop + noise + partial gain drift) then corrupt MS once
    # ----------------------------
    sm = simulator()
    sm.openfromms(msname)
    sm.reset()
    sm.setseed(SEED)

    # --- Troposphere (phase screen) ---
    # Keyword names can vary slightly by CASA version; this is the standard form.
    # If your CASA complains about any keyword, drop it and retry (pwv/deltapwv are the big ones).
    sm.settrop(
        mode="individual",
        pwv=PWV_MM,
        deltapwv=DELTAPWV_MM,
        beta=BETA,
        windspeed=WINDSPEED_MPS,
    )

    # --- Thermal noise (ATM) with SAME PWV ---
    sm.setnoise(mode="simplenoise", simplenoise="10.0Jy")

    # --- Apply the partial gain table multiplicatively ---
    sm.setapply(
        table=gtab_partial,
        type="G",
        field=GAINCAL_FIELD,
        interp="linear",
        calwt=False,
    )

    print("[INFO] sim_all: calling corrupt() ...")
    sm.corrupt()
    sm.done()
    print("[INFO] sim_all: done.")

    return {
        "gtab_raw": gtab_raw,
        "gtab_fixed": gtab_fixed,
        "gtab_partial": gtab_partial,
        "pwv_mm": PWV_MM,
        "deltapwv_mm": DELTAPWV_MM,
        "gain_drift_amp": GAIN_DRIFT_AMP,
        "keep_ants": KEEP_ANTS,
    }

from PIL import Image
import os

def plot_before_after_vis_time_2x2(
    ms_before: str,
    ms_after: str,
    field: str,
    spw: str,
    out_png: str = "images/before_after_amp_phase_2x2.png",
    correlation: str = "RR",
):
    """
    Make a single 2x2 PNG:
      [ BEFORE amp   | BEFORE phase ]
      [ AFTER  amp   | AFTER  phase ]
    """
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    def _plot(vis: str, yaxis: str, plotfile: str):
        plotms(
            vis=vis,
            field=field,
            spw=spw,
            xaxis="time",
            yaxis=yaxis,          # "amp" or "phase"
            avgchannel="9999",    # average over channels
            avgscan=False,
            coloraxis="baseline",
            showgui=False,
            plotfile=plotfile,
            overwrite=True,
            showlegend=True,
            correlation=correlation,
        )

    # 1) Make the four individual plots (temporary)
    tmp_before_amp   = "images/_tmp_before_amp_vs_time.png"
    tmp_before_phase = "images/_tmp_before_phase_vs_time.png"
    tmp_after_amp    = "images/_tmp_after_amp_vs_time.png"
    tmp_after_phase  = "images/_tmp_after_phase_vs_time.png"

    _plot(ms_before, "amp",   tmp_before_amp)
    _plot(ms_before, "phase", tmp_before_phase)
    _plot(ms_after,  "amp",   tmp_after_amp)
    _plot(ms_after,  "phase", tmp_after_phase)

    # 2) Stitch into a 2x2 grid
    imgs = [
        Image.open(tmp_before_amp).convert("RGB"),
        Image.open(tmp_before_phase).convert("RGB"),
        Image.open(tmp_after_amp).convert("RGB"),
        Image.open(tmp_after_phase).convert("RGB"),
    ]

    # Make all tiles the same size (use the smallest to avoid upscaling)
    w = min(im.size[0] for im in imgs)
    h = min(im.size[1] for im in imgs)
    imgs = [im.resize((w, h), resample=Image.Resampling.LANCZOS) for im in imgs]

    canvas = Image.new("RGB", (2 * w, 2 * h), (255, 255, 255))
    canvas.paste(imgs[0], (0,   0))
    canvas.paste(imgs[1], (w,   0))
    canvas.paste(imgs[2], (0,   h))
    canvas.paste(imgs[3], (w,   h))

    canvas.save(out_png)

    # Optional cleanup of temporaries
    for p in [tmp_before_amp, tmp_before_phase, tmp_after_amp, tmp_after_phase]:
        try:
            os.remove(p)
        except OSError:
            pass

    print(f"[INFO] Wrote 2x2 plotms grid: {out_png}")
    return out_png




def main():
    # SETUP
    print(f"MS_IN hash: {hash_casa_table_cols(MS_IN, cols=["DATA"])}")
    copy_ms(MS_IN, MS_OUT)
    print(f"MS_OUT hash:  {hash_casa_table_cols(MS_OUT, cols=["DATA"])}")
    col_diff(MS_IN, MS_OUT)

    # CORRUPTION
    # Add corruption
    # gtab = sim_gain_corrupt(MS_OUT)
    # gtab = sim_gain_corrupt_clip_extreme(MS_OUT)
    # gtab = sim_gain_corrupt_clip_extreme_single_antenna(MS_OUT, keep_ant=3)
    # gtab = sim_gain_corrupt_random(MS_OUT)
    # sim_info = sim_all(MS_OUT) 
    # gtab = sim_info['gtab_partial']


    gtab = sim_gain_corrupt_clip_extreme_n_antennas(
        MS_OUT,
        keep_ants=[0,1,2,3,4,5],
    )

    # PLOTS

    IMG_BEFORE_C = "img_gaincal_before_clean"
    IMG_AFTER_C  = "img_gaincal_after_clean"

    IMG_BEFORE = "img_gaincal_before"
    IMG_AFTER  = "img_gaincal_after"

    IMG_DIFF   = "img_gaincal_diff"
    IMG_DIFF_C = "img_gaincal_diff_clean"

    IMG_FRAC_RES = "img_gaincal_after_fracres"
    IMG_FRAC_RES_BEFORE = "img_gaincal_before_fracres"

    # plot_gain_per_antenna(gtab, spw="0")
    make_dirty(MS_IN, IMG_BEFORE, TCLEAN_KW)
    make_clean(MS_IN,  IMG_BEFORE_C, TCLEAN_KW)

    make_dirty(MS_OUT, IMG_AFTER, TCLEAN_KW)
    make_clean(MS_OUT, IMG_AFTER_C, TCLEAN_KW)

    make_diff(IMG_BEFORE_C, IMG_AFTER_C, IMG_DIFF_C)
    make_diff(IMG_BEFORE, IMG_AFTER, IMG_DIFF)

    make_frac_residuals(
        residual_im=IMG_AFTER_C,
        reference_im=IMG_BEFORE_C,
        out_im=IMG_FRAC_RES,
    )

    make_frac_residuals(
        residual_im=IMG_BEFORE_C,
        reference_im=IMG_BEFORE_C,
        out_im=IMG_FRAC_RES_BEFORE,
    )

    fracres_before_after_png(
        before_im=f"{IMG_FRAC_RES_BEFORE}.image",
        after_im=f"{IMG_FRAC_RES}.image",
        out_png="images/fracres_before_after.png",
        crop_half=64,
    )

    # casa_image_to_png(f"{IMG_FRAC_RES}.image", f"images/{IMG_FRAC_RES}.png")

    # rms_frac = img_rms(f"{IMG_FRAC_RES}.image")
    # print(f"[CHECK] Fractional RMS residual = {rms_frac:.6g}")

    plot_before_after_vis_time(MS_IN, MS_OUT, GAINCAL_FIELD, SPW)
    col_diff(MS_IN, MS_OUT)
    print(f"MS_OUT hash:  {hash_casa_table_cols(MS_OUT, cols=["DATA"])}")

    print("[DONE] Outputs:")
    print(f"  - {MS_OUT}")
    print(f"  - {IMG_BEFORE}.image")
    print(f"  - {IMG_AFTER}.image")
    print(f"  - {IMG_DIFF}.image")
    print(f"  - {IMG_FRAC_RES}.image")


def main_recoverable_corruption():
    
    # SETUP
    print(f"MS_IN hash: {hash_casa_table_cols(MS_IN, cols=['DATA'])}")
    copy_ms(MS_IN, MS_OUT)
    print(f"MS_OUT hash (pre-corrupt): {hash_casa_table_cols(MS_OUT, cols=['DATA'])}")
    col_diff(MS_IN, MS_OUT)

    # CORRUPT
    gtab_injected = sim_gain_corrupt_clip_extreme_n_antennas(
        MS_OUT,
        keep_ants=[0, 1, 2, 3, 4, 5],
        amp=0.05
    )
    print(f"[INFO] Injected corruption table: {gtab_injected}")

    # Before / After visibilities
    plot_before_after_vis_time_2x2(MS_IN, MS_OUT, GAINCAL_FIELD, SPW)

    # RECOVER

    gtab_solved = MS_OUT + f".recover"
    rmtables(gtab_solved)

    # NOTW:
    # For "ap" solnorm=False
    # if want to see the solver match the injected curve shape, I think

    print(f"[INFO] Solving gains back into: {gtab_solved}")
    # set some point source model

    # from casatools import componentlist

    # cl = componentlist()
    # cl.addcomponent(
    #     dir="J2000 00h00m00s 00d00m00s",  # phase center (will be overridden by ft field)
    #     flux=1.0,                        # Jy
    #     freq="1GHz",                     # dummy, CASA rescales per spw
    #     shape="point"
    # )
    # cl.rename("J1822_point.cl")
    # cl.close()

    # ft(
    #     vis=MS_OUT,
    #     field="J1822-0938",
    #     spw="0",
    #     complist="J1822_point.cl",
    #     usescratch=True
    # )

    # from casatasks import visstat
    # st = visstat(
    #     vis=MS_OUT,
    #     field="J1822-0938",
    #     spw="0",
    #     datacolumn="model"
    # )
    # print(st)


    gaincal(
        vis=MS_OUT,
        caltable=gtab_solved,
        field=GAINCAL_FIELD,
        spw=SPW,
        refant="ea21",
        gaintype="G",
        calmode="ap",
        solint="int", # scan
        minsnr=5,
        solnorm=True
    )

    plot_gtab_corrupt_vs_recovered_all_ants_2x2(
        corr_gtab=gtab_injected,
        cal_gtab=gtab_solved,
        spw="0",
        out_png="images/J1822_gtab_corrupt_vs_recovered_2x2.png",
    )

    MS_REC = MS_OUT + ".recovered_for_imaging.ms"
    rmtables(MS_REC)
    copy_ms(MS_OUT, MS_REC)
    print("[INFO] Applying solved gains to undo corruption (-> CORRECTED_DATA)")
    applycal(
        vis=MS_REC,
        field=GAINCAL_FIELD,
        spw=SPW,
        gaintable=[gtab_solved],
        interp=["linear"],
        calwt=False,
    )
    tb = table()
    tb.open(MS_REC, nomodify=False)
    data = tb.getcol("CORRECTED_DATA")
    tb.putcol("DATA", data)
    tb.close()
    
    IMG_BASE_C     = "img_base_clean"
    IMG_CORR_C     = "img_corrupted_clean"
    IMG_RECOV_C    = "img_recovered_clean"
    
    make_clean(MS_IN,  IMG_BASE_C,  TCLEAN_KW)
    make_clean(MS_OUT, IMG_CORR_C,  TCLEAN_KW)
    make_clean(MS_REC, IMG_RECOV_C, TCLEAN_KW)


    IMG_FRAC_BASE = "img_base_fracres"
    make_frac_residuals(
        residual_im=IMG_BASE_C,
        reference_im=IMG_BASE_C,
        out_im=IMG_FRAC_BASE,
    )

    IMG_FRAC_CORR = "img_corrupted_fracres"
    make_frac_residuals(
        residual_im=IMG_CORR_C,
        reference_im=IMG_BASE_C,
        out_im=IMG_FRAC_CORR,
    )

    IMG_FRAC_REC = "img_recovered_fracres"
    make_frac_residuals(
        residual_im=IMG_RECOV_C,
        reference_im=IMG_BASE_C,
        out_im=IMG_FRAC_REC,
    )

    fracres_before_after_png(
        before_im=f"{IMG_FRAC_BASE}.image",      # or baseline vs corrupted measure
        after_im=f"{IMG_FRAC_REC}.image",
        out_png="images/fracres_base_recovered.png",
        crop_half=64,
    )

    fracres_before_after_png(
        before_im=f"{IMG_FRAC_BASE}.image",      # or baseline vs corrupted measure
        after_im=f"{IMG_FRAC_CORR}.image",
        out_png="images/fracres_base_corrupted.png",
        crop_half=64,
    )

    fracres_before_after_png(
        before_im=f"{IMG_FRAC_CORR}.image",      # or baseline vs corrupted measure
        after_im=f"{IMG_FRAC_REC}.image",
        out_png="images/fracres_corrupted_recovered.png",
        crop_half=64,
    )

    print("[DONE OptionA] Outputs:")
    print(f"  - Corrupted MS: {MS_OUT}")
    print(f"  - Injected gain table: {gtab_injected}")
    print(f"  - Solved (recovered) gain table: {gtab_solved}")
    print(f"  - Clean images: {IMG_BASE_C}.image, {IMG_CORR_C}.image, {IMG_RECOV_C}.image")
