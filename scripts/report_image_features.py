"""Per-case report of the image features (docs/image_features_plan.md §2.1 and §4).

  python -m scripts.report_image_features --run runs/dataset-v2-matched-val --table runs/image-features-v2/val.npz \
    --train-table runs/image-features-v2/train.npz --output runs/image-features-v2/case_report --jobs 16

Section 1 is the gate, on TRAINING rows only:
  * ranked fft curves (log10 S vs rank fraction) of the training controls (median, 5–95% band) against the
    corrupted rows grouped by number of corruptions, for C, raw R and the PSF-whitened q = P_R / P_PSF;
    the ratio of the control MAD of log10 S_k to the all-row MAD and the control-vs-corrupted separation;
    the decision whether the experiment's fft block uses Rq instead of R (written to gate.json);
  * the radial check (option B): per region, the median share of squared energy of ΔC (relative to each source's
    control) and of R, against the region's share of pixels, by corruption type, plus R's [2, 4) contrast.
Section 2: per block, two validation pairs (one source's control vs its corrupted case farthest from it in that
block's training-standardized features: the source with the largest such distance and a different source at the
75th percentile), overridable with --pair CASE_A:CASE_B. Planes are recomputed from the images with
scripts/features/planes.py; sky images use the standard scripts.imaging.plot_utils.casa_image_to_png.
"""
from __future__ import annotations

import argparse
import html
import json
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

from ml.image_feature_experiment import block_columns
from scripts.features import energy, fourier, radial
from scripts.features.planes import load_case, read_masks, read_plane
from scripts.clean_recovery_noise_sweep import convolve_model
from scripts.imaging.plot_utils import casa_image_to_png, shared_fits_display_limits

warnings.filterwarnings("ignore", category=FITSFixedWarning)
ROOT = Path(__file__).resolve().parents[1]
MASK_NAMES = {0: "control", 1: "amp", 2: "phase", 3: "amp+phase", 4: "noise", 5: "amp+noise", 6: "phase+noise",
              7: "amp+phase+noise"}
TYPES = (("control", (0,)), ("noise-only", (4,)), ("gain-only", (1, 2, 3)), ("mixed", (5, 6, 7)))
GROUP_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#b3261e"}
ALL22_BY_BLOCK = {
    "energy": ("metrics.residual.scaled_mad_jy_per_beam", "metrics.residual.peak_over_scaled_mad",
               "model.flux_positive_jy", "model.flux_negative_magnitude_jy", "model.flux_over_restored_peak_per_beam"),
    "radial": ("metrics.dynamic_range_scaled_mad", "model.flux_inside_beam_jy", "model.flux_outside_beam_jy",
               "model.nonzero_pixels", "clean.iterations"),
    "fft": ("log1p_even_energy_over_mad2", "log1p_odd_energy_over_mad2", "metrics.residual.p99_5_over_scaled_mad",
            "dirty.log1p_even_energy_over_mad2", "dirty.log1p_odd_energy_over_mad2")}
STYLE = ("<style>body{font:15px system-ui;max-width:1300px;margin:auto;padding:1.5rem;color:#24323a;background:#fcfcfb}"
         "img{max-width:100%}table{border-collapse:collapse;margin:.5rem 0 1.2rem}td,th{padding:.25rem .5rem;"
         "border:1px solid #ccd;text-align:right;font-size:13px}td:first-child,th:first-child{text-align:left}"
         "th{background:#eef3f6}.note{color:#4a5a63}.pass{color:#0a7a3e;font-weight:600}.fail{color:#b3261e;font-weight:600}"
         ".imgs{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}.pair{border-top:3px solid #ccd;margin-top:2rem}"
         "code{font-size:90%}</style>")


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_table(path):
    with np.load(_resolve(path), allow_pickle=False) as z:
        t = {k: z[k] for k in z.files}
    t["names"] = [str(n) for n in t["names"]]
    t["case_ids"] = t["case_ids"].astype(str)
    return t


def split_data(run, split):
    with np.load(run / f"{split}.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def aligned(table, data):
    """Table rows in data's row order."""
    index = {c: i for i, c in enumerate(table["case_ids"])}
    rows = np.asarray([index[str(c)] for c in data["case_ids"]])
    return {k: (v[rows] if getattr(v, "shape", ())[:1] == (len(table["case_ids"]),) else v) for k, v in table.items()}


def savefig(fig, path):
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return f"assets/{path.name}"


def tbl(header, rows):
    return ("<table><tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in header) + "</tr>"
            + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>")


def f3(x):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.3g}"


# ---------------------------------------------------------------- gate: fft

def mad(x):
    x = x[np.isfinite(x)]
    return float(np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def fft_gate(table, data, assets):
    """Ranked-curve panels, MAD tightness and control-vs-corrupted separation of log10 S_k per channel."""
    from sklearn.metrics import roc_auc_score
    mask = data["mask"].astype(int)
    ncorr = np.asarray([bin(m).count("1") for m in mask])
    ctrl = mask == 0
    fr = table["curve_fractions"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=False)
    for ax, ch in zip(axes, fourier.CHANNELS):
        c = table[f"curve_{ch}"]
        lo, med, hi = np.nanpercentile(c[ctrl], [5, 50, 95], axis=0)
        ax.fill_between(fr, lo, hi, color="#999", alpha=.35, label=f"controls 5–95% ({ctrl.sum()})")
        ax.plot(fr, med, color="k", lw=2, label="controls median")
        for g in (1, 2, 3):
            sel = ncorr == g
            q = np.nanpercentile(c[sel], [25, 50, 75], axis=0)
            ax.plot(fr, q[1], color=GROUP_COLORS[g], lw=1.6, label=f"{g} corruption(s) median, IQR ({sel.sum()})")
            ax.fill_between(fr, q[0], q[2], color=GROUP_COLORS[g], alpha=.12)
        ax.set_xscale("log"); ax.set_xlabel("rank fraction k / N_cells"); ax.set_ylabel("log10 S (top-k power share)")
        ax.set_title({"C": "C (convolved model)", "R": "R (raw residual power)", "Rq": "q = P_R / P_PSF"}[ch])
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    img = savefig(fig, assets / "gate_ranked_curves.png")
    rows, scores = [], {}
    for ch in fourier.CHANNELS:
        idx = [table["names"].index(f"fft.{ch}.log10_S.{k}") for k in fourier.K_LABELS]
        X = table["X"][:, idx]
        z = fourier.excess(X, X[ctrl])
        ratios, aucs, frac3 = [], [], []
        for j, k in enumerate(fourier.K_LABELS):
            ok = np.isfinite(z[:, j])
            ratio = mad(X[ctrl, j]) / mad(X[:, j])
            auc = roc_auc_score(~ctrl[ok], np.abs(z[ok, j]))
            f = float(np.mean(np.abs(z[~ctrl & ok, j]) > 3))
            ratios.append(ratio); aucs.append(auc); frac3.append(f)
            rows.append([ch, k, f3(np.nanmedian(X[ctrl, j])), f3(mad(X[ctrl, j])), f3(mad(X[:, j])), f3(ratio),
                         f3(auc), f3(f)])
        scores[ch] = dict(mad_ratio=float(np.mean(ratios)), auroc_abs_z=float(np.mean(aucs)), frac_abs_z_gt3=float(np.mean(frac3)))
    use_q = scores["Rq"]["mad_ratio"] < scores["R"]["mad_ratio"] and scores["Rq"]["auroc_abs_z"] > scores["R"]["auroc_abs_z"]
    decision = dict(fft_residual="Rq" if use_q else "R", scores=scores,
                    rule="Rq replaces R only if its mean control/all MAD ratio of log10 S_k is smaller AND its mean "
                         "control-vs-corrupted AUROC of |excess_z| is larger (training rows).")
    table_html = tbl(["channel", "k", "control median log10 S_k", "control MAD", "all-row MAD", "MAD ratio (ctrl/all)",
                      "AUROC |z| ctrl vs corrupted", "corrupted |z| > 3"], rows)
    return img, table_html, decision


# ---------------------------------------------------------------- gate: radial check

def _shares(values, regions):
    total = float(values.sum())
    return np.asarray([float(values[r].sum()) / total if total > 0 else np.nan for r in regions])


def _radial_source(job):
    """Per case of one training source: ΔC (vs the source's control) and R energy shares, pixel shares."""
    run, prepared, cases, control = job
    run = Path(run)

    def planes(case_id):
        rec = json.loads((run / "cases" / case_id / "case.json").read_text())
        model, _ = read_plane(run / rec["products"]["model"])
        clean, header = read_plane(run / rec["products"]["clean"])
        R, _ = read_plane(run / rec["products"]["residual"])
        return convolve_model(model, header), R
    m = read_masks(str(run / prepared["reference_masks"]))
    omega, rho = m["omega"].astype(bool), m["rho"]
    c_regions = radial.region_masks(rho, omega, radial.C_REGIONS)
    r_regions = radial.region_masks(rho, omega, radial.R_REGIONS)
    C0, _ = planes(control)
    out = []
    for case_id in cases:
        C, R = planes(case_id)
        dC2, R2 = np.where(omega, (C - C0) ** 2, 0.), np.where(omega, R ** 2, 0.)
        out.append((case_id, _shares(dC2, c_regions), _shares(R2, r_regions)))
    pix = (np.asarray([r.sum() for r in c_regions]) / omega.sum(), np.asarray([r.sum() for r in r_regions]) / omega.sum())
    return out, pix


def radial_check(train_run, data, table, jobs):
    plan = json.loads((train_run / "plan.json").read_text())
    prepared = {p["id"]: p for p in plan["prepared_sources"]}
    work = []
    for sid in sorted(set(map(str, data["source_ids"]))):
        sel = np.asarray(data["source_ids"]).astype(str) == sid
        ctrl = data["case_ids"][sel & (data["mask"] == 0)]
        if len(ctrl) != 1:
            raise ValueError(f"{sid}: expected one control, found {len(ctrl)}")
        work.append((str(train_run), prepared[sid], list(map(str, data["case_ids"][sel])), str(ctrl[0])))
    with ProcessPoolExecutor(jobs) as pool:
        results = list(pool.map(_radial_source, work))
    dc, rs, cpix, rpix = {}, {}, [], []
    for out, pix in results:
        for case_id, a, b in out:
            dc[case_id], rs[case_id] = a, b
        cpix.append(pix[0]); rpix.append(pix[1])
    dC = np.stack([dc[str(c)] for c in data["case_ids"]]); R = np.stack([rs[str(c)] for c in data["case_ids"]])
    cpix, rpix = np.median(cpix, 0), np.median(rpix, 0)
    contrast = {}
    for k in range(len(radial.R_REGIONS)):
        contrast[k] = (table["X"][:, table["names"].index(f"radial.R.r{k}.pos_contrast")]
                       + table["X"][:, table["names"].index(f"radial.R.r{k}.neg_contrast")])
    mask = data["mask"].astype(int)
    rows_c, rows_r, med = [], [], {}
    warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)   # control ΔC ≡ 0
    for name, masks in TYPES:
        sel = np.isin(mask, masks)
        med[name] = dict(dC=np.nanmedian(dC[sel], 0), R=np.nanmedian(R[sel], 0),
                         contrast=[float(np.nanmedian(contrast[k][sel])) for k in contrast])
        if name != "control":
            rows_c.append([name, int(sel.sum())] + [f"{f3(a)} <small>({f3(p)})</small>" for a, p in zip(med[name]["dC"], cpix)])
        rows_r.append([name, int(sel.sum())] + [f"{f3(a)} <small>({f3(p)})</small><br>contrast {f3(c)}"
                                                for a, p, c in zip(med[name]["R"], rpix, med[name]["contrast"])])
    lab = lambda regs: [f"[{lo:g}, {hi:g})" if np.isfinite(hi) else f"[{lo:g}, edge)" for lo, hi in regs]
    tail = max(float(med[n]["dC"][-1]) for n, _ in TYPES if n != "control")
    # Geometry baseline: control and noise-only rows. Gain rows are expected to deviate (that is the signal the
    # density contrast measures, contrast = share / pixel share), so they are shown, not tested.
    area = max(float(np.max(np.abs(med[n]["R"] - rpix))) for n in ("control", "noise-only"))
    g, nz = med["gain-only"]["contrast"][1], med["noise-only"]["contrast"][1]
    checks = [("C changes stay inside ρ < 4: max median ΔC share in [4, edge)", tail, tail < .01),
              ("R energy share follows area for control and noise-only rows: max |median R share − pixel share|", area, area < .05),
              (f"R contrast in [2, 4) separates gain-only ({g:.3g}) from noise-only ({nz:.3g})", g / nz, g > nz)]
    html_ = ("<h3>ΔC energy share per C region (median; pixel share in brackets)</h3>"
             + tbl(["type", "rows", *lab(radial.C_REGIONS)], rows_c)
             + "<h3>R energy share per R region (median; pixel share in brackets) and pos+neg contrast</h3>"
             + tbl(["type", "rows", *lab(radial.R_REGIONS)], rows_r)
             + tbl(["finding", "value", "result"], [[html.escape(t), f3(v), f"<span class='{'pass' if ok else 'fail'}'>"
                                                     f"{'PASS' if ok else 'FAIL'}</span>"] for t, v, ok in checks]))
    return html_, [dict(finding=t, value=float(v), passed=bool(ok)) for t, v, ok in checks]


# ---------------------------------------------------------------- pairs

def block_matrix(block, ttab, vtab, train_ctrl, fft_residual):
    """Training-standardized block features (fft as in the experiment: excess z + supporting)."""
    names, tr, va = block_columns(block, ttab, vtab, train_ctrl, fft_residual)
    mu, sd = np.nanmean(tr, 0), np.nanstd(tr, 0)
    sd = np.where(sd > 0, sd, 1.)
    return names, tr, va, mu, sd


def select_pairs(val, Z):
    """Per source: control vs its farthest corrupted case. Returns (extreme, 75th-percentile) pairs."""
    src = np.asarray(val["source_ids"]).astype(str)
    best = []
    for s in sorted(set(src)):
        idx = np.flatnonzero(src == s)
        c = idx[val["mask"][idx] == 0][0]
        other = idx[val["mask"][idx] != 0]
        d = np.sqrt(np.nansum((Z[other] - Z[c]) ** 2, axis=1))
        best.append((float(d.max()), s, int(c), int(other[np.argmax(d)])))
    best.sort()
    top = best[-1]
    target = np.percentile([b[0] for b in best], 75)
    rest = [b for b in best if b[1] != top[1]]
    typical = min(rest, key=lambda b: abs(b[0] - target))
    return [dict(kind="largest distance", distance=top[0], a=top[2], b=top[3]),
            dict(kind="75th percentile", distance=typical[0], a=typical[2], b=typical[3])]


def recipe(val, i, run):
    rec = json.loads((run / "cases" / str(val["case_ids"][i]) / "case.json").read_text())
    A, phi, a = val["params"][i]
    m = int(val["mask"][i])
    r = val["noise_ratio"][i]
    return (f"<b>{MASK_NAMES[m]}</b> (mask {m}); A {A:.4g}, φ {phi:.4g}°, noise multiplier {a:.4g} (r {r:.3g}); antenna "
            f"{rec['case'].get('antenna')}; eC {val['eC'][i]:.4g}, eD {val['eD'][i]:.4g}; case <code>{val['case_ids'][i]}</code>")


def sky_panels(pa, pb, labels, assets, stem):
    """C/s and R/s of both cases with the standard plotter (one shared scale per channel) + ρ region map."""
    files = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for ch in ("C", "R"):
            paths = []
            for p, lab in ((pa, labels[0]), (pb, labels[1])):
                h = p["header"].copy(); h["BUNIT"] = "1"
                path = Path(tmp) / f"{ch}_{lab}.fits"
                fits.PrimaryHDU(p[ch].astype(np.float32), h).writeto(path)
                paths.append(path)
            lim = shared_fits_display_limits(paths, value_multiplier=1.)
            for path, lab in zip(paths, labels):
                out = assets / f"{stem}_{ch}_{lab}.png"
                casa_image_to_png(path, out, title=f"{ch}/s, case {lab}", draw_beam=True, display_limits_mjy_per_beam=lim)
                files.append(f"assets/{out.name}")
    w = WCS(pa["header"]).celestial
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = plt.subplot(projection=w)
    edges = [.5, 1, 2, 4, 8, 16]
    ax.imshow(np.digitize(pa["rho"], edges), origin="lower", cmap="Greys", alpha=.5)
    cs = ax.contour(pa["rho"], levels=edges, colors="k", linewidths=.8, origin="lower")
    ax.clabel(cs, fmt=lambda v: f"ρ={v:g}", fontsize=7)
    ax.set_title("ρ region boundaries (beam units, source-centred)"); ax.set_xlabel("RA"); ax.set_ylabel("Dec")
    files.append(savefig(fig, assets / f"{stem}_rho.png"))
    return files


def profile_plot(pa, pb, labels, path):
    edges = np.logspace(np.log10(.0625), np.log10(64), 25)
    mid = np.sqrt(edges[1:] * edges[:-1])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, ch in zip(axes, ("C", "R")):
        for p, lab, col in ((pa, labels[0], "#2a78d6"), (pb, labels[1], "#d7191c")):
            which = np.digitize(p["rho"], edges) - 1
            x2 = p[ch] ** 2
            prof = [x2[p["omega"] & (which == i)].mean() if np.any(p["omega"] & (which == i)) else np.nan
                    for i in range(len(mid))]
            ax.plot(mid, prof, marker=".", color=col, label=f"case {lab}")
        for e in (radial.C_REGIONS if ch == "C" else radial.R_REGIONS):
            ax.axvline(e[0], color="#bbb", lw=.8) if e[0] > 0 else None
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("ρ (beams)"); ax.set_ylabel(f"mean ({ch}/s)² per pixel")
        ax.set_title(f"{ch}: energy per pixel vs ρ (grey: block region edges)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    return savefig(fig, path)


def bar_plot(names, a, b, labels, path, title, log=True):
    fig, ax = plt.subplots(figsize=(max(7, .45 * len(names)), 3.8))
    x = np.arange(len(names))
    ax.bar(x - .2, np.abs(a), .4, label=f"case {labels[0]}", color="#2a78d6")
    ax.bar(x + .2, np.abs(b), .4, label=f"case {labels[1]}", color="#d7191c")
    if log:
        ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7.5)
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    return savefig(fig, path)


def radial_heatmaps(va, vb, labels, path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for row, (ch, regs, stats) in enumerate((("C", radial.C_REGIONS, radial.C_STATS), ("R", radial.R_REGIONS, radial.R_STATS))):
        keys = [[f"radial.{ch}.r{k}.{s}" for s in stats] for k in range(len(regs))]
        A = np.asarray([[va[n] for n in r] for r in keys]); B = np.asarray([[vb[n] for n in r] for r in keys])
        with np.errstate(divide="ignore", invalid="ignore"):
            panels = (np.log10(A), np.log10(B), np.log10(B / A))
        titles = (f"log10 {ch} case {labels[0]}", f"log10 {ch} case {labels[1]}", f"log10({labels[1]}/{labels[0]})")
        for ax, M, t in zip(axes[row], panels, titles):
            M = np.where(np.isfinite(M), M, np.nan)
            lim = np.nanmax(np.abs(M)) if t.startswith("log10(") and np.isfinite(M).any() else None
            im = ax.imshow(M, cmap="RdBu_r" if lim else "viridis", vmin=-lim if lim else None, vmax=lim, aspect="auto")
            for (i, j), v in np.ndenumerate(M):
                ax.text(j, i, f3(v), ha="center", va="center", fontsize=7)
            ax.set_xticks(range(len(stats))); ax.set_xticklabels(stats, fontsize=8)
            ax.set_yticks(range(len(regs))); ax.set_yticklabels([f"[{lo:g},{hi:g})" for lo, hi in regs], fontsize=8)
            ax.set_title(t, fontsize=9); fig.colorbar(im, ax=ax, fraction=.046)
    fig.tight_layout()
    return savefig(fig, path)


def fft_maps(pa, pb, labels, path):
    keep = fourier.independent_cells(pa["omega"].shape)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.4))
    for row, (p, lab) in enumerate(((pa, labels[0]), (pb, labels[1]))):
        beam = np.radians(np.sqrt(p["header"]["BMAJ"] * p["header"]["BMIN"]))
        PC, u, v = fourier.power(p["C"], p["header"], p["omega"])
        PR, _, _ = fourier.power(p["R"], p["header"], p["omega"])
        Ppsf = fourier.psf_power(p["psf"], p["psf_header"])
        q, ok = fourier.whitened(PR, Ppsf, keep)
        for ax, P, cells, t in zip(axes[row], (PC, PR, Ppsf, q), (keep, keep, keep, ok),
                                   ("P_C", "P_R", "P_PSF", "q = P_R / P_PSF (kept cells)")):
            M = np.where(cells & (P > 0), P, np.nan)
            with np.errstate(divide="ignore"):
                im = ax.pcolormesh(u * beam, v * beam, np.log10(M), shading="nearest", cmap="magma")
            if np.isfinite(M).any():
                top = np.argsort(np.where(np.isfinite(M), M, -np.inf).ravel())[::-1][:10]
                ax.plot((u * beam).ravel()[top], (v * beam).ravel()[top], "o", mfc="none", mec="cyan", ms=7, label="top 10 cells")
            ax.set_aspect("equal"); ax.set_title(f"case {lab}: log10 {t}", fontsize=9)
            ax.set_xlabel("u (cycles / beam)"); ax.set_ylabel("v (cycles / beam)")
            fig.colorbar(im, ax=ax, fraction=.046)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    return savefig(fig, path)


def curve_plot(ttab, tctrl, vtab, ia, ib, labels, path):
    fr = ttab["curve_fractions"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, ch in zip(axes, fourier.CHANNELS):
        lo, med, hi = np.nanpercentile(ttab[f"curve_{ch}"][tctrl], [5, 50, 95], axis=0)
        ax.fill_between(fr, lo, hi, color="#999", alpha=.35, label="train controls 5–95%")
        ax.plot(fr, med, "k", lw=1.5, label="train controls median")
        ax.plot(fr, vtab[f"curve_{ch}"][ia], color="#2a78d6", lw=1.8, label=f"case {labels[0]}")
        ax.plot(fr, vtab[f"curve_{ch}"][ib], color="#d7191c", lw=1.8, label=f"case {labels[1]}")
        ax.set_xscale("log"); ax.set_title(f"ranked curve {ch}"); ax.set_xlabel("rank fraction"); ax.grid(alpha=.3)
    axes[0].set_ylabel("log10 S"); axes[0].legend(fontsize=7)
    return savefig(fig, path)


def excess_plot(ttab, tctrl, vtab, ia, ib, labels, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.4), sharey=True)
    x = np.arange(len(fourier.K_LABELS))
    for ax, ch in zip(axes, fourier.CHANNELS):
        idx = [ttab["names"].index(f"fft.{ch}.log10_S.{k}") for k in fourier.K_LABELS]
        za = fourier.excess(vtab["X"][ia, idx], ttab["X"][tctrl][:, idx])
        zb = fourier.excess(vtab["X"][ib, idx], ttab["X"][tctrl][:, idx])
        ax.bar(x - .2, za, .4, color="#2a78d6", label=f"case {labels[0]}")
        ax.bar(x + .2, zb, .4, color="#d7191c", label=f"case {labels[1]}")
        ax.set_xticks(x); ax.set_xticklabels(fourier.K_LABELS); ax.axhline(0, color="k", lw=.6)
        ax.set_yscale("symlog", linthresh=3); ax.set_title(f"{ch}: excess_z per k"); ax.grid(alpha=.3, axis="y")
    axes[0].legend(fontsize=8)
    return savefig(fig, path)


def pair_section(block, pair, val, vtab, ttab, tctrl, stdz, run, prepared, assets, all22, notes, n):
    ia, ib = pair["a"], pair["b"]
    labels = ("A", "B")
    stem = f"pair{n}_{block}"
    pa, pb = (load_case(run, json.loads((run / "cases" / str(val["case_ids"][i]) / "case.json").read_text())["products"],
                        prepared[str(val["source_ids"][i])], psf=True) for i in (ia, ib))
    names, _, va_cols, mu, sd = stdz[block]
    za, zb = (va_cols[ia] - mu) / sd, (va_cols[ib] - mu) / sd
    order = np.argsort(-np.nan_to_num(np.abs(zb - za)))[:5]
    changes = "".join(
        f"<li><code>{html.escape(names[j])}</code>: {f3(va_cols[ia, j])} → {f3(va_cols[ib, j])}"
        f"{f' (×{va_cols[ib, j] / va_cols[ia, j]:.3g})' if np.isfinite(va_cols[ia, j]) and abs(va_cols[ia, j]) > 1e-12 else ''}"
        f", Δz {zb[j] - za[j]:+.2f}</li>" for j in order)
    a22 = tbl(["all22 feature", "case A", "case B"],
              [[f"<code>{html.escape(f)}</code>", f3(all22["X"][ia, all22["names"].index(f)]), f3(all22["X"][ib, all22["names"].index(f)])]
               for f in ALL22_BY_BLOCK[block]])
    imgs = sky_panels(pa, pb, labels, assets, stem)
    figs = [profile_plot(pa, pb, labels, assets / f"{stem}_profile.png")] if block == "radial" else []
    vA = dict(zip(vtab["names"], vtab["X"][ia])); vB = dict(zip(vtab["names"], vtab["X"][ib]))
    if block == "energy":
        a = np.asarray([vA[k] for k in energy.NAMES]); b = np.asarray([vB[k] for k in energy.NAMES])
        a[0], b[0] = 10 ** a[0], 10 ** b[0]   # plot s itself (log axis)
        lab = ["s (Jy/beam)", *energy.NAMES[1:]]
        figs.append(bar_plot(lab, a, b, labels, assets / f"{stem}_bars.png", "energy block (|value|, log y)"))
    elif block == "radial":
        figs.insert(0, radial_heatmaps(vA, vB, labels, assets / f"{stem}_heat.png"))
    else:
        figs += [fft_maps(pa, pb, labels, assets / f"{stem}_maps.png"),
                 curve_plot(ttab, tctrl, vtab, ia, ib, labels, assets / f"{stem}_curves.png"),
                 excess_plot(ttab, tctrl, vtab, ia, ib, labels, assets / f"{stem}_excess.png")]
        sup = tbl(["supporting feature", "case A", "case B"],
                  [[f"<code>{n_}</code>", f3(vA[n_]), f3(vB[n_])] for ch in fourier.CHANNELS
                   for n_ in (f"fft.{ch}.peak_cycles_per_beam", f"fft.{ch}.angular_concentration")] +
                  [["<code>fft.Rq.n_cells</code>", f3(vA["fft.Rq.n_cells"]), f3(vB["fft.Rq.n_cells"])]])
    note = notes.get(f"{block}:{val['case_ids'][ia]}:{val['case_ids'][ib]}") or notes.get(f"{block}:{n}")
    return (f"<div class='pair'><h3>{block} pair {n}: {html.escape(pair['kind'])} "
            f"(source {html.escape(str(val['source_ids'][ib]))}, distance {pair['distance']:.3g})</h3>"
            f"<p>A: {recipe(val, ia, run)}<br>B: {recipe(val, ib, run)}<br>s: A {pa['s']:.4g}, B {pb['s']:.4g} Jy/beam</p>"
            f"<p><b>Largest standardized changes ({block} block, z on training rows):</b></p><ul>{changes}</ul>"
            + (f"<p><b>Interpretation:</b> {note}</p>" if note else "<p class='note'>Interpretation: pending.</p>")
            + a22 + "<div class='imgs'>" + "".join(f"<img src='{f}'>" for f in imgs) + "</div>"
            + "".join(f"<img src='{f}'>" for f in figs) + (sup if block == "fft" else "") + "</div>")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/dataset-v2-matched-val")
    ap.add_argument("--table", default="runs/image-features-v2/val.npz")
    ap.add_argument("--train-table", default="runs/image-features-v2/train.npz")
    ap.add_argument("--output", required=True)
    ap.add_argument("--pair", action="append", default=[], metavar="BLOCK:CASE_A:CASE_B",
                    help="override the automatic pairs (repeatable); BLOCK in energy/radial/fft")
    ap.add_argument("--fft-residual", choices=("auto", "R", "Rq"), default="auto",
                    help="residual channel for the fft pair distance; auto = gate decision")
    ap.add_argument("--notes", help="JSON {\"<block>:<n>\" or \"<block>:<caseA>:<caseB>\": html interpretation}")
    ap.add_argument("--skip-radial-check", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args(argv)
    run, output = _resolve(a.run), _resolve(a.output)
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    ttab, vtab = load_table(a.train_table), load_table(a.table)
    train_run = _resolve(str(ttab["run"]))
    tdata, val = split_data(train_run, str(ttab["split"])), split_data(run, str(vtab["split"]))
    ttab, vtab = aligned(ttab, tdata), aligned(vtab, val)
    plan = json.loads((run / "plan.json").read_text())
    prepared = {p["id"]: p for p in plan["prepared_sources"]}
    # Noise ratio r = noise multiplier / nominal source SNR, as ml.multitask.load_split (reporting only).
    val["noise_ratio"] = val["params"][:, 2] / np.asarray([prepared[str(s)]["nominal_source_snr"] for s in val["source_ids"]])
    all22 = dict(names=list(map(str, val["feature_names"])), X=val["X"])
    tctrl = tdata["mask"] == 0
    # 1. gate
    gate_img, gate_table, decision = fft_gate(ttab, tdata, assets)
    radial_html, radial_checks = ("<p class='note'>skipped (--skip-radial-check)</p>", None) if a.skip_radial_check \
        else radial_check(train_run, tdata, ttab, a.jobs)
    decision["radial_checks"] = radial_checks
    (output / "gate.json").write_text(json.dumps(decision, indent=1))
    fft_res = decision["fft_residual"] if a.fft_residual == "auto" else a.fft_residual
    # 2. pairs
    stdz = {b: block_matrix(b, ttab, vtab, tctrl, fft_res) for b in ("energy", "radial", "fft")}
    case_index = {str(c): i for i, c in enumerate(val["case_ids"])}
    overrides = {}
    for item in a.pair:
        b, ca, cb = item.split(":")
        overrides.setdefault(b, []).append(dict(kind="manual", distance=float("nan"), a=case_index[ca], b=case_index[cb]))
    notes = json.loads(Path(a.notes).read_text()) if a.notes else {}
    sections, chosen = [], {}
    for b in ("energy", "radial", "fft"):
        names, _, va_cols, mu, sd = stdz[b]
        pairs = overrides.get(b) or select_pairs(val, (va_cols - mu) / sd)
        chosen[b] = [dict(p, case_a=str(val["case_ids"][p["a"]]), case_b=str(val["case_ids"][p["b"]])) for p in pairs]
        for n, p in enumerate(pairs, 1):
            print("pair", b, n, chosen[b][n - 1]["case_a"], chosen[b][n - 1]["case_b"], flush=True)
            sections.append(pair_section(b, p, val, vtab, ttab, tctrl, stdz, run, prepared, assets, all22, notes, n))
    (output / "pairs.json").write_text(json.dumps(chosen, indent=1, default=str))
    sc = decision["scores"]
    page = (f"<!doctype html><meta charset='utf-8'><title>Image features: per-case report</title>{STYLE}"
            "<h1>Image features (energy, radial, fft): gate and per-case pairs</h1>"
            f"<p class='note'>Validation run <code>{html.escape(str(a.run))}</code>; training table "
            f"<code>{html.escape(str(a.train_table))}</code> ({len(ttab['X'])} rows, {int(tctrl.sum())} controls) used for "
            "the control reference and standardization only. Plan: <code>docs/image_features_plan.md</code>.</p>"
            "<h2>1. Gate (training rows)</h2><h3>1a. fft: ranked curves</h3>"
            f"<img src='{gate_img}'>{gate_table}"
            f"<p><b>Decision:</b> fft residual channel = <b>{decision['fft_residual']}</b>. {html.escape(decision['rule'])} "
            f"R: MAD ratio {sc['R']['mad_ratio']:.3g}, AUROC {sc['R']['auroc_abs_z']:.3g}; "
            f"Rq: MAD ratio {sc['Rq']['mad_ratio']:.3g}, AUROC {sc['Rq']['auroc_abs_z']:.3g}. "
            f"Pairs below use fft residual {fft_res}.</p>"
            "<h3>1b. Radial check (option B)</h3>" + radial_html +
            "<h2>2. Per-case pairs (validation)</h2>"
            "<p class='note'>Sky images: standard plotter, dimensionless C/s and R/s, one shared colour range per channel "
            "within a pair. The ρ panel shows the region boundaries on the same pixel grid.</p>" + "".join(sections))
    (output / "report.html").write_text(page)
    print("Image-feature report:", output / "report.html", flush=True)


if __name__ == "__main__":
    main()
