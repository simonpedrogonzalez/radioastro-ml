#!/usr/bin/env python3
"""Section-1 Fourier coordinate checks and a standalone explanatory HTML report.

Run: MPLCONFIGDIR=/private/tmp/radioastro-v2-mpl ml/.venv/bin/python \
    scripts/report_fourier_toy.py
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/radioastro-v2-mpl")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "collect/experiments/v2_visible_features"


def phase_centered_fft(values, header, reference_scale=1.0):
    """Original pixels → unitary FFT with its origin at FITS CRPIX (no input shift).

    J maps pixel (x,y) offsets to tangent-plane radians; q = inv(J).T @ f.
    Returns display-ordered complex response and its 2-D (u,v) coordinate grids.
    """
    a = np.asarray(values, dtype=float)
    if a.ndim != 2 or not np.all(np.isfinite(a)):
        raise ValueError("FFT needs a finite, unmasked two-dimensional plane")
    if not np.isfinite(reference_scale) or reference_scale <= 0:
        raise ValueError("Reference scale must be finite and positive")
    w = WCS(header, fix=False).celestial
    if w.world_axis_units != ["deg", "deg"]:
        raise ValueError("Expected a celestial WCS in degrees")
    j = np.deg2rad(w.pixel_scale_matrix)
    p0 = np.asarray(w.wcs.crpix) - 1.0
    fy, fx = np.meshgrid(np.fft.fftfreq(a.shape[0]),
                         np.fft.fftfreq(a.shape[1]), indexing="ij")
    phase = np.exp(2j * np.pi * (fx * p0[0] + fy * p0[1]))
    z = phase * np.fft.fft2(a / reference_scale, norm="ortho")
    uv = np.einsum("ij,jyx->iyx", np.linalg.inv(j).T, np.array([fx, fy]))
    return np.fft.fftshift(z), np.fft.fftshift(uv[0]), np.fft.fftshift(uv[1])


def figure_uri(fig, path=None):
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=125, bbox_inches="tight", facecolor="white")
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(stream.getvalue())
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def write_html(path, title, body, script=""):
    """Small report shell; figures and browser code are embedded for offline use."""
    Path(path).write_text("<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>"
        "body{font:16px/1.6 system-ui,sans-serif;color:#233044;background:#f3f5f8;margin:0}"
        "main{max-width:1280px;margin:auto;padding:32px}h1,h2,h3{line-height:1.25;color:#142c46}"
        "h1{font-size:34px}h2{margin-top:38px}p{max-width:1000px}"
        "section,.card{background:white;border:1px solid #dae1e8;border-radius:12px;padding:22px;margin:20px 0}"
        "img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}"
        "th,td{text-align:left;border-bottom:1px solid #dce2e9;padding:8px}th{background:#edf2f7}"
        ".scroll{overflow-x:auto}.note{border-left:4px solid #25827c;background:#edf8f5;padding:14px}"
        ".caution{border-left:4px solid #be8122;background:#fff5df;padding:14px}"
        "code{font-size:13px;overflow-wrap:anywhere}pre{white-space:pre-wrap;font-size:13px}"
        "select,button{font:inherit;padding:8px;border:1px solid #9aaabd;border-radius:6px;margin:5px}"
        "label{display:inline-block;margin-right:12px}canvas{max-width:100%;background:white}"
        ".muted{color:#5b6b7e;font-size:14px}a{color:#176b86}"
        "</style><main>" + f"<h1>{html.escape(title)}</h1>" + body
        + "</main>" + (f"<script>{script}</script>" if script else "") + "</html>", encoding="utf-8")


def toy_header(p0=(128.0, 128.0), angle=0.0):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.crval = [25.0, -20.0]
    w.wcs.crpix = np.asarray(p0) + 1
    w.wcs.cdelt = [-0.6 / 3600, 0.8 / 3600]
    t = np.deg2rad(angle)
    w.wcs.pc = [[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]
    return w.to_header()


def direct_response(values, header, q):
    w = WCS(header, fix=False).celestial
    y, x = np.indices(values.shape)
    offsets = np.array([x - (w.wcs.crpix[0] - 1), y - (w.wcs.crpix[1] - 1)])
    angles = np.einsum("ij,jyx->iyx", np.deg2rad(w.pixel_scale_matrix), offsets)
    theta = 2 * np.pi * np.einsum("i,iyx->yx", q, angles)
    c = np.sum(values * np.cos(theta)) / np.sqrt(values.size)
    s = np.sum(values * np.sin(theta)) / np.sqrt(values.size)
    return c - 1j * s


def run_checks():
    checks = []
    for p0, rotation in [((128., 128.), 0.), ((127.3, 130.7), 23.)]:
        header = toy_header(p0, rotation)
        y, x = np.indices((256, 256))
        theta = 2 * np.pi * (9 * (x - p0[0]) + 5 * (y - p0[1])) / 256
        for name, a, expected in [("cosine", np.cos(theta), 128 + 0j),
                                   ("sine", np.sin(theta), -128j),
                                   ("sum", np.cos(theta) + np.sin(theta), 128 - 128j)]:
            z, u, v = phase_centered_fft(a, header)
            pos = (128 + 5, 128 + 9)
            direct = direct_response(a, header, [u[pos], v[pos]])
            np.testing.assert_allclose(z[pos], direct, atol=2e-11)
            np.testing.assert_allclose(z[pos], expected, atol=2e-11)
            z2, _, _ = phase_centered_fft(2 * a, header)
            np.testing.assert_allclose(z2, 2 * z, atol=2e-11)
            np.testing.assert_allclose(abs(z2[pos])**2, 4 * abs(z[pos])**2, rtol=1e-12)
            np.testing.assert_allclose(np.sum(abs(z)**2), np.sum(a**2), rtol=1e-12)
            np.testing.assert_allclose(z[128 - 5, 128 - 9], z[pos].conjugate(), atol=2e-11)
            checks.append(dict(pattern=name, phase_center=list(p0), rotation_deg=rotation,
                real=float(z[pos].real), imaginary=float(z[pos].imag),
                direct_error=float(abs(z[pos] - direct)), amplitude_ratio=2.0, power_ratio=4.0))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checks = run_checks()
    header = toy_header()
    y, x = np.indices((256, 256))
    theta = 2 * np.pi * (9 * (x - 128) + 5 * (y - 128)) / 256
    half = 2 * np.pi * (9.5 * (x - 128) + 5 * (y - 128)) / 256
    cases = [("Cosine", np.cos(theta), 9.,
              "φ = 0: the response is real (+128 at both arrows); the imaginary panel is zero."),
             ("Sine", np.sin(theta), 9.,
              "φ = −π/2: the response is imaginary (−128 at +q, +128 at −q); the real panel is zero."),
             ("Cosine + sine", np.cos(theta) + np.sin(theta), 9.,
              "Adding the two waves puts both real and imaginary responses at the same ±q locations."),
             ("Twice the cosine", 2 * np.cos(theta), 9.,
              "A doubles: the response doubles to +256 at each arrow, and |Z|² becomes four times larger."),
             ("Half-bin cosine", np.cos(half), 9.5,
              "kₓ = 9.5: the arrows fall between FFT bins, so the response spreads across nearby bins.")]
    parts = ["<section><h2>The wave and its two responses</h2>",
        "<pre>I(ℓ,m) = A cos[2π(u₀ℓ + v₀m) + φ]\n"
        "Z(u,v) = (1/256) Σₚ I(ℓₚ,mₚ) exp[−2πi(uℓₚ + vmₚ)] = C(u,v) − i S(u,v)\n"
        "C = (1/256) Σₚ Iₚ cos[2π(uℓₚ + vmₚ)]\n"
        "S = (1/256) Σₚ Iₚ sin[2π(uℓₚ + vmₚ)] = −Im Z</pre>",
        "<p><b>I</b> is image brightness at east/north sky offsets <b>ℓ,m</b> (radians); "
        "<b>u,v</b> are spatial frequencies in wavelengths (cycles per radian). "
        "The sum runs over image pixels p, and i² = −1. "
        "<b>A</b> changes stripe contrast; <b>φ</b> shifts the stripes. "
        "The angle of <b>q = (u₀,v₀)</b> sets the direction across the fringes; "
        "its length sets their spacing, 1/|q| radians. The green arrows show ±q, "
        "which are perpendicular to the fringes in the east–north image. "
        "Image arrows show direction only; uv arrows show location.</p>",
        "<p><b>Z</b> is the complex Fourier response at a chosen uv location. The FFT uses "
        "exp(−iθ) = cos θ − i sin θ: its real part <b>C</b> measures a cosine match, "
        "and its imaginary part is <b>−S</b>, the negative sine match. "
        "Thus a cosine and a quarter-cycle-shifted sine can share the same uv locations "
        "while appearing in different panels. A real image has conjugate responses "
        "at +q and −q. The sum is divided by 256 because this "
        "256 × 256 FFT uses unitary normalization.</p>",
        "<p>Here the wave is made first with pixel frequencies (kₓ,kᵧ) = (9,5); "
        "the image's signed angular pixel scales determine its uv arrows. "
        "Whole-number cycles land on FFT bins. The last wave uses (9.5,5) to put "
        "its response between bins. Green marks show the expected locations even when "
        "one plotted component is zero.</p></section>"]
    j = np.deg2rad(WCS(header, fix=False).pixel_scale_matrix)
    if not (j[0, 0] < 0 < j[1, 1] and np.allclose(j[[0, 1], [1, 0]], 0)):
        raise ValueError("This east–north display expects the toy's unrotated signed WCS")
    arcsec = 180 / np.pi * 3600
    edge = np.arange(257) - .5 - 128
    east = j[0, 0] * edge * arcsec
    north = j[1, 1] * edge * arcsec
    sky_extent = (east.min(), east.max(), north.min(), north.max())
    green = "#008e67"
    fractions = {}
    for name, a, kx, explanation in cases:
        z, u, v = phase_centered_fft(a, header)
        q = np.linalg.inv(j).T @ (np.array([kx, 5.]) / 256)
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
        m = axes[0].imshow(a[:, ::-1], extent=sky_extent, origin="lower", aspect="equal",
                           interpolation="nearest", cmap="RdBu_r", vmin=-2, vmax=2)
        direction = q / np.linalg.norm(q) * 38  # Unit direction in a display-sized arcsecond arrow.
        for sign in (1, -1):
            tip = sign * direction
            axes[0].annotate("", xy=tip, xytext=(0, 0),
                             arrowprops=dict(arrowstyle="-|>", color=green, lw=2.5))
        axes[0].set(title=name, xlabel="East offset ℓ (arcsec) →", ylabel="North offset m (arcsec)")
        fig.colorbar(m, ax=axes[0], label="Brightness")
        for ax, vals, title in zip(axes[1:], [z.real, z.imag], ["Real Z = cosine match", "Imag Z = −sine match"]):
            limit = 100 if kx == 9.5 else 256
            m = ax.pcolormesh(u / 1000, v / 1000, vals, cmap="RdBu_r", vmin=-limit, vmax=limit, shading="nearest")
            for sign in (1, -1):
                tip = sign * q / 1000
                ax.annotate("", xy=.82 * tip, xytext=(0, 0),
                            arrowprops=dict(arrowstyle="-|>", color=green, lw=2))
            ax.scatter([q[0]/1000, -q[0]/1000], [q[1]/1000, -q[1]/1000],
                       facecolors="none", edgecolors=green, s=340, linewidths=1.8)
            ax.set(title=title, xlabel="u (kλ)", ylabel="v (kλ)", xlim=(-18,18), ylim=(-13,13))
            ax.set_aspect("equal")
            fig.colorbar(m, ax=ax, label="FFT response")
        uri = figure_uri(fig, args.output / "toy_assets" / (name.lower().replace(" ", "_") + ".png"))
        top_two = np.sort((abs(z)**2).ravel())[-2:].sum() / np.sum(abs(z)**2)
        fractions[name] = float(top_two)
        parts.append(f"<section><h2>{html.escape(name)}</h2><img alt='{html.escape(name)} image and complex FFT' src='{uri}'>"
            f"<p>{explanation}{' Its FFT colour scale is tightened to ±100 to show the spread.' if kx == 9.5 else ''}</p></section>")
    assert fractions["Cosine"] > 1 - 1e-12
    assert fractions["Half-bin cosine"] < .8
    write_html(args.output / "toy_fourier.html", "One wave → one Fourier response", "".join(parts))
    (args.output / "toy_checks.json").write_text(json.dumps(dict(checks=checks, strongest_pair_power_fraction=fractions), indent=2) + "\n")
    print(args.output / "toy_fourier.html")


if __name__ == "__main__":
    main()
