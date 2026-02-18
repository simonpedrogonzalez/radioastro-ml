import numpy as np
from casatools import table

def _wrap_pi(x):
    return (x + np.pi) % (2*np.pi) - np.pi

def _timerange_mask_tod(time_sec: np.ndarray, timerange: str) -> np.ndarray:
    """Time-of-day mask using TIME mod 86400. Works if obs < 1 day."""
    def hms_to_sec(hms: str) -> int:
        hh, mm, ss = hms.split(":")
        return int(hh)*3600 + int(mm)*60 + int(ss)

    t0, t1 = timerange.split("~")
    s0, s1 = hms_to_sec(t0), hms_to_sec(t1)
    tod = np.mod(time_sec, 86400.0)

    if s0 <= s1:
        return (tod >= s0) & (tod <= s1)
    else:
        return (tod >= s0) | (tod <= s1)

def _fetch_baseline_map(
    ms: str,
    a: int,
    b: int,
    *,
    ddid: int,
    corr_idx: int,
    chan_idx: int,
    col: str = "DATA",
    timerange: str | None = None,
):
    """
    Returns dict time->V_ab where V_ab is the visibility for the *ordered* baseline a->b.
    If a row is stored as (b,a), we conjugate it so it becomes a->b.
    """
    tb = table(); tb.open(ms)
    try:
        q = (
            f"DATA_DESC_ID=={ddid} && "
            f"((ANTENNA1=={a} && ANTENNA2=={b}) || (ANTENNA1=={b} && ANTENNA2=={a}))"
        )
        subt = tb.query(q)

        t = subt.getcol("TIME").astype(float)
        data = subt.getcol(col)  # (nCorr, nChan, nRow)
        ant1 = subt.getcol("ANTENNA1").astype(int)
        ant2 = subt.getcol("ANTENNA2").astype(int)

        if timerange is not None:
            m = _timerange_mask_tod(t, timerange)
            t = t[m]
            data = data[:, :, m]
            ant1 = ant1[m]
            ant2 = ant2[m]

        if data.shape[-1] == 0:
            raise ValueError(f"No rows for baseline {a}->{b} (ddid={ddid}) in {ms}")

        vis = data[corr_idx, chan_idx, :].astype(np.complex128)

        # rows stored reversed (b,a) should be conjugated to represent (a,b)
        flip = (ant1 == b) & (ant2 == a)
        if np.any(flip):
            vis = vis.copy()
            vis[flip] = np.conj(vis[flip])

        return {float(tt): vv for tt, vv in zip(t, vis)}

    finally:
        try: subt.close()
        except Exception: pass
        tb.close()

def closure_phase_series_int(
    ms: str,
    *,
    a: int,
    b: int,
    c: int,
    ddid: int,
    chan_idx: int,
    corr_idx: int = 0,
    col: str = "DATA",
    timerange: str | None = None,
):
    """
    Closure phase for triangle (a,b,c):
      phi = arg( V_ab * V_bc * V_ca )
    using canonical baseline orientations and exact time matching.
    Returns: times (float), phi_rad (float array), phi_deg (float array)
    """

    # Get the complex visibilities for each baseline from the data
    m_ab = _fetch_baseline_map(ms, a, b, ddid=ddid, corr_idx=corr_idx, chan_idx=chan_idx, col=col, timerange=timerange)
    m_bc = _fetch_baseline_map(ms, b, c, ddid=ddid, corr_idx=corr_idx, chan_idx=chan_idx, col=col, timerange=timerange)
    m_ca = _fetch_baseline_map(ms, c, a, ddid=ddid, corr_idx=corr_idx, chan_idx=chan_idx, col=col, timerange=timerange)

    # check that we have matching measurements in time
    common = np.array(sorted(set(m_ab) & set(m_bc) & set(m_ca)), dtype=float)
    if common.size == 0:
        raise ValueError("No common TIME samples across AB, BC, CA. (ddid/chan/corr/timerange mismatch?)")

    # Extract matching in time complex visibilities for each baseline
    # We have a time series of visibilities for each baseline
    vab = np.array([m_ab[t] for t in common], dtype=np.complex128)
    vbc = np.array([m_bc[t] for t in common], dtype=np.complex128)
    vca = np.array([m_ca[t] for t in common], dtype=np.complex128)

    # Closure product, the antenna phases should cancel out
    # so 'Babc' phase should be independent of antenna-based phase corruption
    Babc = vab * vbc * vca
    # Extract closure phase
    phi = np.angle(Babc)
    # return times, closure phase rads, closure phase in degs
    return common, phi, phi * 180.0/np.pi

def format_closure_report(
    *,
    a, b, c,
    ddid,
    chan_idx,
    corr_idx,
    col,
    timerange,
    d01,
    d02,
    d12,
):
    import numpy as np

    def stats(x):
        xdeg = x * 180.0 / np.pi
        return (
            f"N={xdeg.size:3d}  "
            f"mean={np.mean(xdeg):7.2f} deg  "
            f"rms={np.sqrt(np.mean(xdeg**2)):7.2f} deg  "
            f"p95={np.percentile(np.abs(xdeg), 95):7.2f} deg"
        )

    lines = [
        f"[closure] tri=({a},{b},{c})  ddid={ddid}  chan={chan_idx}  corr={corr_idx}",
        f"          col={col}  timerange={timerange}",
        f"  base → corr : {stats(d01)}",
        f"  base → rec  : {stats(d02)}",
        f"  corr → rec  : {stats(d12)}",
    ]

    return "\n".join(lines)


def compare_closure_three_int(
    ms_base: str,
    ms_corr: str,
    ms_rec: str,
    *,
    a: int, b: int, c: int,
    ddid: int,
    chan_idx: int,
    corr_idx: int = 0,
    timerange: str | None = None,
    col: str = "DATA",
):  
    print("Comparing closure phases....")

    # Compute the closure phases for each dataset
    t0, p0, _ = closure_phase_series_int(ms_base, a=a, b=b, c=c, ddid=ddid, chan_idx=chan_idx, corr_idx=corr_idx, col=col, timerange=timerange)
    t1, p1, _ = closure_phase_series_int(ms_corr, a=a, b=b, c=c, ddid=ddid, chan_idx=chan_idx, corr_idx=corr_idx, col=col, timerange=timerange)
    t2, p2, _ = closure_phase_series_int(ms_rec,  a=a, b=b, c=c, ddid=ddid, chan_idx=chan_idx, corr_idx=corr_idx, col=col, timerange=timerange)

    # Find common points in the time series to compare (should be all of them if things went well)
    common = np.array(sorted(set(t0) & set(t1) & set(t2)), dtype=float)
    if common.size == 0:
        raise ValueError("No common TIME across the three MSes for this selection.")

    # Aligns the time series using common
    def remap(t, p):
        m = {float(tt): pp for tt, pp in zip(t, p)}
        return np.array([m[tt] for tt in common])

    P0 = remap(t0, p0)
    P1 = remap(t1, p1)
    P2 = remap(t2, p2)

    # Computes angular differences
    d01 = _wrap_pi(P1 - P0) # corruption vs base
    d02 = _wrap_pi(P2 - P0) # recovered vs base
    d12 = _wrap_pi(P2 - P1) # recovered vs corruption

    def stats(x):
        xdeg = x * 180.0/np.pi
        return dict(
            N=int(xdeg.size), # number of samples compared
            mean_deg=float(np.mean(xdeg)), # Mean difference in degrees, should be ~0 if closure is respected
            rms_deg=float(np.sqrt(np.mean(xdeg**2))), # Noise, scatter in time of the differences. Increases if there's some baseline dependent effect
            p95_abs_deg=float(np.percentile(np.abs(xdeg), 95)), # Worst case deviation, what are the worst closure violations
        )

    print("d01 deg:", np.round(d01 * 180/np.pi, 3))

    report = format_closure_report(
        a=a, b=b, c=c,
        ddid=ddid,
        chan_idx=chan_idx,
        corr_idx=corr_idx,
        col=col,
        timerange=timerange,
        d01=d01,
        d02=d02,
        d12=d12,
    )

    print(report)


    # times, aligned closure phases, aligned phases differences
    return common, P0, P1, P2, d01, d02, d12, report

import numpy as np
import matplotlib.pyplot as plt

def plot_phase_closure(
    times,
    P0,
    P1,
    P2,
    *,
    unwrap=False,
    degrees=True,
    title="Closure phase vs time",
    out_png=None,
):
    """
    Plot closure phase time series for:
      - baseline (P0)
      - corrupted (P1)
      - recovered (P2)

    Parameters
    ----------
    times : array-like
        TIME values (seconds)
    P0, P1, P2 : array-like
        Closure phases (radians)
    unwrap : bool
        If True, unwrap phase before plotting (useful for trends).
        Default False (wrapped in (-pi, pi]).
    degrees : bool
        Plot in degrees (default) or radians.
    out_png : str or None
        If provided, save figure to this path.
    """

    def prep(x):
        y = np.unwrap(x) if unwrap else x
        return y * 180.0/np.pi if degrees else y

    y0 = prep(P0)
    y1 = prep(P1)
    y2 = prep(P2)

    # Convert TIME to relative minutes for readability
    t = np.asarray(times, dtype=float)
    t = (t - t.min()) / 60.0  # minutes since start

    plt.figure(figsize=(10, 4.5))

    plt.plot(t, y0, "o-", label="baseline",  lw=1.5, ms=4)
    plt.plot(t, y1, "o-", label="corrupted", lw=1.5, ms=4)
    plt.plot(t, y2, "o-", label="recovered", lw=1.5, ms=4)

    plt.axhline(0, color="k", lw=0.8, alpha=0.3)

    plt.xlabel("Time (minutes since start)")
    plt.ylabel("Closure phase (deg)" if degrees else "Closure phase (rad)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    if out_png is not None:
        plt.savefig(out_png, dpi=150)
        print(f"[INFO] Wrote closure phase plot: {out_png}")
    else:
        plt.show()


import numpy as np
from casatools import table

def apply_const_ant_phases_to_ms(
    ms_in: str,
    ms_out: str,
    *,
    phases_rad: np.ndarray,   # length = n_ant; phases_rad[a] = φ_a
    col: str = "DATA",
):
    """
    Copy ms_in -> ms_out before calling this!
    Then this edits ms_out in-place: DATA := exp(iφ_a) * exp(-iφ_b) * DATA for each row (a=ANTENNA1, b=ANTENNA2).
    Applies per antenna constant phase changes -> closure phase should be preserved.
    """
    tb = table(); tb.open(ms_out, nomodify=False)
    try:
        ant1 = tb.getcol("ANTENNA1").astype(int)
        ant2 = tb.getcol("ANTENNA2").astype(int)
        data = tb.getcol(col)  # (nCorr, nChan, nRow) complex

        rows = np.arange(data.shape[2], dtype=np.int64)

        # build per-row complex factor g_a * conj(g_b) = exp(iφ_a) * exp(-iφ_b)
        ph = phases_rad
        fac = np.exp(1j * (ph[ant1[rows]] - ph[ant2[rows]])).astype(np.complex128)  # (nrow_sel,)

        # apply across corr & chan
        data[:, :, rows] *= fac[np.newaxis, np.newaxis, :]

        tb.putcol(col, data)
        tb.flush()
    finally:
        tb.close()


def create_random_antenna_phases(
    ms: str,
    *,
    seed: int = 42,
    phase_range=(-np.pi, np.pi),
    col: str = "DATA",
):
    """
    Identify antennas that participate in at least one unflagged visibility
    and generate random per-antenna phases for them.

    Returns
    -------
    ants : np.ndarray (int)
        Sorted unique antenna IDs that appear unflagged
    phases : np.ndarray (float)
        Array of phases indexed by antenna ID (length = max_ant_id + 1).
        Antennas not in `ants` get phase = 0.
    """

    tb = table()
    tb.open(ms)
    try:
        ant1 = tb.getcol("ANTENNA1").astype(int)
        ant2 = tb.getcol("ANTENNA2").astype(int)
        flag = tb.getcol("FLAG")          # (nCorr, nChan, nRow)
    finally:
        tb.close()

    # Row is unflagged if ANY corr+chan is unflagged
    unflagged_row = ~np.all(flag, axis=(0, 1))

    if not np.any(unflagged_row):
        raise RuntimeError("No unflagged data found in MS")

    ants = np.unique(
        np.concatenate([
            ant1[unflagged_row],
            ant2[unflagged_row],
        ])
    )

    ants = np.sort(ants)

    rng = np.random.default_rng(seed)

    n_ant_total = int(max(ants)) + 1
    phases = np.zeros(n_ant_total, dtype=float)

    phases[ants] = rng.uniform(
        phase_range[0],
        phase_range[1],
        size=len(ants),
    )

    print(f"[INFO] Found {len(ants)} unflagged antennas: {ants.tolist()}")
    print(f"[INFO] Generated random phases in [{phase_range[0]:.2f}, {phase_range[1]:.2f}] rad")

    return ants, phases


def constant_per_antena_phase_corruption(
    ms_in, ms_out, *,
    seed: int = 42,
    phase_range=(-np.pi, np.pi),
    col: str = "DATA"
):  
    ants, phases = create_random_antenna_phases(
        ms_in,
        seed=seed,
        phase_range=phase_range,
        col=col
    )

    apply_const_ant_phases_to_ms(
        ms_in,
        ms_out,
        phases_rad=phases,
        col=col,
    )

    return phases


def baseline_phase_corruption(
    ms_out: str,
    *,
    a: int,
    b: int,
    ddid: int = 0,
    chan_idx: int = 32,
    corr_idx: int = 0,
    phase_rad: float | None = None, # constant phase
    seed: int = 42, # for random per-row phase
    random_std_rad: float | None = None, # std for N(0, std) in rads
    col: str = "DATA",
    timerange: str | None = None,
):
    """
    Multiply ONLY baseline (a,b) by exp(+i theta_row) (ordered a->b).
    Rows stored as (b,a) are conjugated for applying in the ordered convention,
    then written back in the original stored orientation.
    This is baseline-dependent => it BREAKS closure for triangles touching (a,b).
    """
    tb = table(); tb.open(ms_out, nomodify=False)
    try:
        q = (
            f"DATA_DESC_ID=={ddid} && "
            f"((ANTENNA1=={a} && ANTENNA2=={b}) || (ANTENNA1=={b} && ANTENNA2=={a}))"
        )
        subt = tb.query(q)

        t = subt.getcol("TIME").astype(float)
        ant1 = subt.getcol("ANTENNA1").astype(int)
        ant2 = subt.getcol("ANTENNA2").astype(int)
        data = subt.getcol(col)  # (nCorr, nChan, nRow)

        if timerange is not None:
            # reuse your existing helper
            m = _timerange_mask_tod(t, timerange)
            t = t[m]; ant1 = ant1[m]; ant2 = ant2[m]
            data = data[:, :, m]

        nrow = data.shape[2]
        if nrow == 0:
            raise RuntimeError("No rows in selection (check ddid/ants/timerange).")

        # build per-row theta
        if random_std_rad is not None:
            rng = np.random.default_rng(seed)
            theta = rng.normal(0.0, random_std_rad, size=nrow)
        else:
            if phase_rad is None:
                raise ValueError("Provide phase_rad or random_std_rad.")
            theta = np.full(nrow, float(phase_rad))

        # ordered a->b convention:
        # if stored row is (b,a), then ordered visibility is conj(stored).
        flip = (ant1 == b) & (ant2 == a)

        vis = data[corr_idx, chan_idx, :].astype(np.complex128)

        # convert to ordered (a->b)
        vis_ord = vis.copy()
        vis_ord[flip] = np.conj(vis_ord[flip])

        # apply corruption in ordered space
        vis_ord *= np.exp(1j * theta)

        # convert back to stored orientation
        vis_new = vis_ord.copy()
        vis_new[flip] = np.conj(vis_new[flip])

        data = data.copy()
        data[corr_idx, chan_idx, :] = vis_new
        subt.putcol(col, data)

        tb.flush()
        print(f"[INFO] baseline phase corruption applied on ({a},{b}) with nrow={nrow}")
    finally:
        try: subt.close()
        except Exception: pass
        tb.close()
