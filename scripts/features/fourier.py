"""Block `fft`: ranked power concentration of C/s and R/s (and the PSF-whitened residual q = P_R / P_PSF).

Per channel x: y = x − mean_Ω(x) (zero outside Ω), z = phase_centered_fft(y, header), P = |z|². Only the
independent cells are used (one of each Hermitian pair, self-conjugate cells once, DC excluded: N²/2 + 1 for an
even N×N grid). p = P / ΣP sorted decreasingly; S_k = sum of the k largest p. Stored per case: log10 S_k, the
ranked curve log10 S at CURVE_FRACTIONS, the (u, v) radius of the top cell in cycles per beam and the angular
concentration |Σ P e^{2iθ}| / Σ P. The training-control excess z-score is computed at training time (excess).
"""
from __future__ import annotations

import math

import numpy as np

from scripts.report_fourier_toy import phase_centered_fft

K_LABELS = ("k1", "k3", "k10", "k30", "k0.1pct", "k1pct")
CURVE_FRACTIONS = np.logspace(-4.5, 0, 24)   # rank fraction k / N_cells of the stored ranked curves
CHANNELS = ("C", "R", "Rq")
SPECTRUM_KEYS = (*(f"log10_S.{k}" for k in K_LABELS), "peak_cycles_per_beam", "angular_concentration")
NAMES = (*(f"fft.{ch}.{key}" for ch in CHANNELS for key in SPECTRUM_KEYS), "fft.Rq.n_cells")
PSF_THRESHOLD = 1e-3   # q keeps the independent cells with P_PSF >= PSF_THRESHOLD * max P_PSF (fixed, not tuned)


def ks(n_cells):
    return [1, 3, 10, 30, math.ceil(.001 * n_cells), math.ceil(.01 * n_cells)]


def independent_cells(shape):
    """Boolean mask in fftshift (display) order: one cell of each Hermitian pair, self-conjugate cells once, no DC."""
    ny, nx = shape
    iy, ix = np.indices(shape)
    cell = iy * nx + ix
    partner = (-iy % ny) * nx + (-ix % nx)
    return np.fft.fftshift((cell <= partner) & (cell != 0))


def power(x, header, omega):
    """|FFT|² of the Ω-mean-subtracted plane (zero outside Ω), with its (u, v) grids in wavelengths."""
    y = np.where(omega, x - x[omega].mean(), 0.)
    z, u, v = phase_centered_fft(y, header)
    return np.abs(z) ** 2, u, v


def psf_power(psf, header):
    """|FFT(PSF)|², without mean subtraction: the case's gridded, weighted uv coverage."""
    z, _, _ = phase_centered_fft(np.asarray(psf, float), header)
    return np.abs(z) ** 2


def whitened(P_R, P_psf, keep):
    """q = P_R / P_PSF on the kept cells (independent and P_PSF >= PSF_THRESHOLD · max over independent cells)."""
    ok = keep & (P_psf >= PSF_THRESHOLD * P_psf[keep].max())
    q = np.zeros_like(P_R)
    q[ok] = P_R[ok] / P_psf[ok]
    return q, ok


def spectrum(P, u, v, header, keep):
    """Features (SPECTRUM_KEYS) and ranked curve of the power P over the cells `keep`; NaN for zero power."""
    p, uu, vv = P[keep], u[keep], v[keep]
    n = len(p)
    total = float(p.sum())
    if not (n and np.isfinite(total) and total > 0):
        return dict.fromkeys(SPECTRUM_KEYS, np.nan), np.full(len(CURVE_FRACTIONS), np.nan)
    order = np.argsort(-p, kind="stable")
    cum = np.cumsum(p[order]) / total
    log_s = [math.log10(cum[min(k, n) - 1]) for k in ks(n)]
    curve = np.log10(cum[np.clip(np.ceil(CURVE_FRACTIONS * n).astype(int), 1, n) - 1])
    beam = math.radians(math.sqrt(header["BMAJ"] * header["BMIN"]))
    top = order[0]
    theta = np.arctan2(vv, uu)
    values = [*log_s, float(math.hypot(uu[top], vv[top]) * beam),
              float(abs(np.sum(p * np.exp(2j * theta))) / total)]
    return dict(zip(SPECTRUM_KEYS, values)), curve


def fourier(planes):
    """All raw fft columns (NAMES) and the curves {C, R, Rq}; planes must include the PSF (load_case(psf=True))."""
    header, omega = planes["header"], planes["omega"]
    keep = independent_cells(omega.shape)
    out, curves = {}, {}
    P = {}
    for ch in ("C", "R"):
        P[ch], u, v = power(planes[ch], header, omega)
        values, curves[ch] = spectrum(P[ch], u, v, header, keep)
        out.update({f"fft.{ch}.{k}": x for k, x in values.items()})
    q, ok = whitened(P["R"], psf_power(planes["psf"], planes["psf_header"]), keep)
    values, curves["Rq"] = spectrum(q, u, v, header, ok)
    out.update({f"fft.Rq.{k}": x for k, x in values.items()})
    out["fft.Rq.n_cells"] = float(ok.sum())
    return {name: out[name] for name in NAMES}, curves


def excess(values, controls):
    """Robust z-score against the control rows: (x − median_ctrl) / (1.4826 · MAD_ctrl), per column."""
    controls = np.asarray(controls, float)
    med = np.nanmedian(controls, axis=0)
    mad = 1.4826 * np.nanmedian(np.abs(controls - med), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(mad > 0, (np.asarray(values, float) - med) / mad, np.nan)
