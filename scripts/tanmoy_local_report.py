"""Report for the Tanmoy local experiments: report.html, four experiment pages and measurements.csv
(or, for the uvgrid / order-validation studies, one report.html of that study).

  ML:   python -m scripts.tanmoy_local_report --input runs/tanmoy-local-20261123

Reads only retained outputs (plan.json, cases/*/record.json|failure.json, FITS products, selfcal_audit.json);
never runs CASA. Every planned case appears in the CSV and in counts; missing or failed cases stay visible.
Text conclusions are computed from the measured values.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warnings
from astropy.wcs import FITSFixedWarning

warnings.simplefilter("ignore", FITSFixedWarning)

from scripts.tanmoy_local_experiments import central_crop, load_planes

# Reference categorical slots (dataviz palette, light mode); identity also carried by marker shape + labels.
C1, C2, C3, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#d9d8d4"
ORDER_STYLE = {"noise": ("noise only", "#52514e", "o"), "current": ("current: G·V + n", C1, "s"),
               "alternative": ("alternative: G·(V + n)", C2, "D")}
TOL = 0.05
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2})

CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--line:#d9d8d4;--card:#f3f2ef;--link:#1f62b3}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--line:#3a3936;--card:#242422;--link:#7fb0f0}}
:root[data-theme=dark]{--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--line:#3a3936;--card:#242422;--link:#7fb0f0}
body{background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif;max-width:1050px;margin:0 auto;padding:16px}
a{color:var(--link)} h1{font-size:1.5em;margin:.4em 0} h2{font-size:1.15em;margin-top:1.6em;border-top:1px solid var(--line);padding-top:.8em}
.lead{font-size:1.08em;background:var(--card);padding:.7em 1em;border-radius:6px}
table{border-collapse:collapse;margin:.6em 0;font-size:.88em;display:block;overflow-x:auto}
td,th{border-bottom:1px solid var(--line);padding:3px 8px;text-align:left;vertical-align:top} td.n{text-align:right;font-variant-numeric:tabular-nums}
img{max-width:100%;display:block;margin:.6em 0;background:#fff;border-radius:4px} .cap{color:var(--muted);font-size:.88em}
details{margin:.6em 0} summary{cursor:pointer;color:var(--muted)} pre{white-space:pre-wrap;font-size:.8em;background:var(--card);padding:.6em}
.bad{color:#c0392b;font-weight:600} .tag{display:inline-block;padding:0 6px;border-radius:4px;background:var(--card);font-size:.85em}
"""


def esc(x):
    return html.escape(str(x))


def fmt(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(x)
    if isinstance(x, float):
        if not math.isfinite(x):
            return str(x)
        return f"{x:.{nd}g}" if abs(x) < 1e-3 or abs(x) >= 1e4 else f"{x:.{nd}f}".rstrip("0").rstrip(".") if nd else f"{x:.0f}"
    return esc(x)


def table(header, rows):
    out = ["<table><tr>" + "".join(f"<th>{esc(h)}</th>" for h in header) + "</tr>"]
    for r in rows:
        out.append("<tr>" + "".join(f"<td class=n>{fmt(v)}</td>" if isinstance(v, (int, float)) and not isinstance(v, bool)
                                    else f"<td>{v if isinstance(v, str) and v.startswith('<') else fmt(v)}</td>" for v in r) + "</tr>")
    return "".join(out) + "</table>"


def page(title, body, back=True):
    nav = "<p><a href='report.html'>← all experiments</a></p>" if back else ""
    return (f"<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{esc(title)}</title><style>{CSS}</style><body>{nav}<h1>{esc(title)}</h1>{body}</body></html>")


class Run:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.plan = json.loads((self.root / "plan.json").read_text())
        self.assets = self.root / "report_assets"
        self.assets.mkdir(exist_ok=True)
        self.records = {}
        for c in self.plan["cases"]:
            d = self.root / "cases" / c["id"]
            if (d / "record.json").is_file():
                rec = json.loads((d / "record.json").read_text())
                if rec.get("key") != c["key"]:
                    rec = dict(id=c["id"], status="stale (cache key differs from plan)")
            elif (d / "failure.json").is_file():
                rec = json.loads((d / "failure.json").read_text())
                rec["status"] = "failed"
            else:
                rec = dict(id=c["id"], status="missing")
            rec["plan"] = c
            self.records[c["id"]] = rec
        self.planes = {}

    def ok(self, cid):
        return self.records[cid].get("status") in ("complete", "reused")

    def m(self, cid, key, default=None):
        r = self.records[cid]
        return r.get("metrics", {}).get(key, default) if self.ok(cid) else default

    def load(self, cid):
        if cid not in self.planes:
            from scripts.clean_recovery_noise_sweep import convolve_model, geometry
            rec = self.records[cid]
            p = load_planes({k: self.root / v for k, v in rec["products"].items()})
            src = self.plan["sources"][rec["plan"]["source"]]
            rho, T = geometry(p["header"], p["model"].shape, src["flux_jy"], src["ra_rad"], src["dec_rad"])
            p.update(C=convolve_model(p["model"], p["header"]), T=T, rho=rho, flux=src["flux_jy"],
                     cell=abs(p["header"]["CDELT2"]) * 3600)
            m = rec["plan"].get("crop")
            if m:   # experiment 3b: full native convolution first, then the common central crop
                from astropy.io import fits
                with fits.open(self.root / rec["products"]["psf"], memmap=False) as h:
                    p["psf"] = np.asarray(np.squeeze(h[0].data), np.float64)
                p["native_shape"] = p["model"].shape
                sl = central_crop(p["model"].shape[0], m), central_crop(p["model"].shape[1], m)
                for k in ("dirty", "clean", "residual", "model", "valid", "C", "T", "rho", "psf"):
                    p[k] = p[k][sl]
                T = p["T"]
            eC = float(np.sqrt(np.square((p["C"] - T)[p["valid"]]).sum() / np.square(T[p["valid"]]).sum()))
            if abs(eC - rec["metrics"]["eC"]) > 1e-9:
                raise ValueError(f"Plot data disagree with the record for {cid}: {eC} vs {rec['metrics']['eC']}")
            self.planes[cid] = p
        return self.planes[cid]

    def save(self, fig, name):
        fig.savefig(self.assets / name, dpi=130, facecolor="white")
        plt.close(fig)
        return f"<img src='report_assets/{name}' alt='{esc(name)}'>"


def crop(a, p, half_arcsec):
    """Centre crop of fixed angular half-width (same field on any native grid)."""
    ny, nx = a.shape
    h = int(round(half_arcsec / p["cell"]))
    return a[ny // 2 - h:ny // 2 + h + 1, nx // 2 - h:nx // 2 + h + 1], [-h * p["cell"], h * p["cell"]] * 2


def beam_marker(ax, header, extent):
    from matplotlib.patches import Ellipse
    x0, y0 = extent[0] * .75, extent[2] * .75
    ax.add_patch(Ellipse((x0, y0), header["BMIN"] * 3600, header["BMAJ"] * 3600, angle=-header["BPA"],
                         fc="none", ec="#222", lw=1))


def image_grid(run, rows, cols_ids, col_titles, half, name, suptitle=""):
    """rows: list of (label, fn(planes)->array, diverging, normalize_by_flux)."""
    fig, axes = plt.subplots(len(rows), len(cols_ids), figsize=(2.3 * len(cols_ids) + .9, 2.25 * len(rows) + .5),
                             squeeze=False, layout="constrained")
    for i, (label, fn, div) in enumerate(rows):
        data = [crop(fn(run.load(c)) / run.load(c)["flux"], run.load(c), half) if c else None for c in cols_ids]
        vals = [d[0] for d in data if d is not None]
        hi = max(float(np.abs(v).max()) for v in vals) or 1.
        lo = -hi if div else min(0., min(float(v.min()) for v in vals))
        for j, d in enumerate(data):
            ax = axes[i, j]
            ax.grid(False)
            if d is None:
                ax.text(.5, .5, "missing", ha="center", transform=ax.transAxes); ax.set_xticks([]); ax.set_yticks([])
                continue
            im = ax.imshow(d[0], origin="lower", cmap="RdBu_r" if div else "inferno", vmin=lo, vmax=hi, extent=d[1])
            beam_marker(ax, run.load(cols_ids[j])["header"], d[1])
            if i == 0:
                ax.set_title(col_titles[j], fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f"{label}\nΔDec (arcsec)", fontsize=8.5)
            if i == len(rows) - 1:
                ax.set_xlabel("ΔRA (arcsec)", fontsize=8)
        fig.colorbar(im, ax=axes[i, :], shrink=.9, label="÷ true flux (Jy/beam)")
    if suptitle:
        fig.suptitle(suptitle, fontsize=9)
    return run.save(fig, name)


# ---------------------------------------------------------------- CSV

def csv_rows(run):
    rows = []
    for cid, r in run.records.items():
        c = r["plan"]
        m, k, data = r.get("metrics", {}), r.get("clean", {}), r.get("data", {})
        flux = run.plan["sources"][c["source"]]["flux_jy"]
        off = m.get("residual_offsource", {})
        rows.append(dict(
            id=cid, experiment=c["experiment"], source=c["source"], status=r.get("status"), order=c.get("order"),
            policy=c.get("policy"), coverage=c.get("coverage"), seed=c.get("seed"), seed_label=c.get("seed_label"),
            A=c.get("A"), a=c.get("a"), r=c.get("r"), imsize=c["grid"]["image_grid"]["imsize"][0],
            cell_arcsec=c["grid"]["image_grid"]["cell_arcsec"][0], seconds=r.get("seconds"),
            requested_niter=(r.get("clean_resolved") or c["clean"]).get("niter"),
            requested_nsigma=(r.get("clean_resolved") or c["clean"]).get("nsigma"),
            requested_threshold=(r.get("clean_resolved") or c["clean"]).get("threshold"),
            requested_dirty_peak_fraction=(r.get("clean_resolved") or c["clean"]).get("dirty_peak_fraction"),
            derived_threshold_jy=(r.get("derived_threshold") or {}).get("value_jy"),
            iterdone=k.get("iterdone"), nmajordone=k.get("nmajordone"), stopcode=k.get("stopcode"),
            stop_description=k.get("stop_description"), iteration_limit_reached=k.get("iteration_limit_reached"),
            tclean_threshold_jy=k.get("tclean_threshold_jy"), final_minor_cycle_threshold_jy=k.get("final_minor_cycle_threshold_jy"),
            final_peak_residual_jy=k.get("final_peak_residual_jy"), eC=m.get("eC"),
            flux_ratio_total=m.get("flux_ratio_total"), flux_ratio_aperture=m.get("flux_ratio_aperture"),
            signed_flux_error_aperture=m.get("signed_flux_error_aperture"),
            outside_aperture_abs_flux_ratio=m.get("outside_aperture_abs_flux_ratio"),
            residual_offsource_std_over_flux=off.get("std", np.nan) / flux if off else None,
            residual_offsource_mad_over_flux=off.get("scaled_mad", np.nan) / flux if off else None,
            residual_offsource_peak_over_flux=off.get("abs_peak", np.nan) / flux if off else None,
            residual_aperture_peak_over_flux=m.get("residual_aperture", {}).get("abs_peak", np.nan) / flux if m else None,
            residual_full_peak_over_flux=m.get("residual_full", {}).get("abs_peak", np.nan) / flux if m else None,
            pixels_per_beam_minor=m.get("grid", {}).get("pixels_per_beam_minor"),
            nonzero_model_pixels=m.get("nonzero_model_pixels"),
            noise_rms_affected_jy=(data.get("noise", {}).get("affected") or {}).get("effective_noise_rms_component_jy"),
            noise_rms_unaffected_jy=(data.get("noise", {}).get("unaffected") or {}).get("effective_noise_rms_component_jy"),
            usable_fraction_removed=(r.get("cut") or {}).get("usable_fraction_removed"),
            eC_matched_support=m.get("eC_matched"),
            reused_from=r.get("reused_from")))
        if c["experiment"] == "uvgrid":
            g = m.get("grid", {})
            rows[-1].update(weighting=c["imaging"]["weighting"], fourier_cell_wavelengths=c["fourier_cell_wavelengths"],
                            field_of_view_arcsec=c["grid"]["image_grid"]["field_of_view_arcsec"][0],
                            native_shape=(g.get("native") or {}).get("shape"), evaluation_crop=c["crop"],
                            model_flux_ratio_full_native=(g.get("native") or {}).get("model_flux_ratio"),
                            data_sha256=data.get("data_sha256"))
    with (run.root / "measurements.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    return rows


def status_line(run, exp):
    rs = [r for r in run.records.values() if r["plan"]["experiment"] == exp]
    counts = {}
    for r in rs:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    secs = sum(r.get("seconds") or 0 for r in rs)
    return counts, secs


def tech(run, exp):
    rs = [r for r in run.records.values() if r["plan"]["experiment"] == exp]
    logs = sorted((run.root / "runs").glob(f"{exp}*.json"))
    detail = {"cases": {r["id"]: {k: v for k, v in r.items() if k not in ("plan", "metrics")} for r in rs},
              "worker_runs": [json.loads(p.read_text()) for p in logs]}
    return f"<details><summary>Technical details: settings, CLEAN records, worker logs</summary><pre>{esc(json.dumps(detail, indent=1, default=str)[:200000])}</pre></details>"


# ---------------------------------------------------------------- experiment 1

def order_page(run):
    seeds = [("archived", "archived seed"), ("new1", "new seed 1"), ("new2", "new seed 2")]
    src = run.plan["sources"]["0449+113"]
    get = lambda o, s: run.m(f"order-{o}-{s}", "eC")
    d = {o: [None if get(o, s) is None or get("noise", s) is None else get(o, s) - get("noise", s) for s, _ in seeds]
         for o in ("current", "alternative")}
    have = [i for i in range(3) if d["current"][i] is not None and d["alternative"][i] is not None]
    # lead sentence from the measurements
    if have:
        neg_c = sum(d["current"][i] < 0 for i in have)
        neg_a = sum(d["alternative"][i] < 0 for i in have)
        weaker = sum(d["alternative"][i] > d["current"][i] for i in have)
        mc, ma = np.mean([d["current"][i] for i in have]), np.mean([d["alternative"][i] for i in have])
        lead = (f"Adding the gain lowered eC in {neg_c}/{len(have)} noise realizations with the current order "
                f"(mean change {mc:+.3f}) and in {neg_a}/{len(have)} with the alternative order (mean {ma:+.3f}); "
                f"the alternative order gave a weaker rescue in {weaker}/{len(have)}. Per realization (current / alternative): "
                + "; ".join(f"{seeds[i][1]} {d['current'][i]:+.3f} / {d['alternative'][i]:+.3f}" for i in have) + ".")
        if weaker == len(have) and neg_a < len(have):
            verdict = "The rescue weakens consistently when the gain also multiplies the noise: the coupling assumption contributes to it."
        elif neg_a == len(have) and weaker < len(have):
            verdict = "The rescue remains under the alternative order: the order of application alone does not explain it."
        elif neg_a == len(have):
            verdict = ("The rescue remains under the alternative order but is weaker in every realization: the order "
                       "changes its size, not its existence.")
        else:
            verdict = "The realizations disagree; no general mechanism should be read from a single image."
    else:
        lead, verdict = "Not enough completed cases for a comparison.", ""
    body = [f"<p class=lead>{esc(lead)} <b>Three noise realizations, one source.</b></p>"]
    body.append("<h2>1. Question</h2><p>The current simulation boosts the source on one antenna without boosting the "
                "additive noise. Does the amplification “rescue” of a noisy source disappear or weaken when the gain also "
                "multiplies the noise?</p>")
    body.append("<h2>2. What changed and what stayed fixed</h2>" + table(["", "Setting"], [
        ["Changed", "noise only V+n · current G·V+n · alternative G·(V+n), per seed"],
        ["Source", f"0449+113, original CLEAN-model flux {src['flux_jy']:.4f} Jy, SNR₀ {src['snr0']:.0f}"],
        ["Gain", f"antenna {src['antenna']}, A = {run.records['order-current-archived']['plan']['A']:.4f}, phase 0 (archived dataset row)"],
        ["Noise", f"a = {run.records['order-noise-archived']['plan']['a']:.1f} × nominal, r = a/SNR₀ = "
                  f"{run.records['order-noise-archived']['plan']['r']:.4f}; same n for all orders of a seed"],
        ["Fixed", "visibility weights (nominal σ₀, not rescaled), flags, uv samples, sky, CLEAN mask, 256×256 grid, "
                  "restoring/evaluation beam, adaptive CLEAN (nsigma 3, default niter)"],
        ["Seeds", ", ".join(f"{lab} = {run.records[f'order-noise-{s}']['plan']['seed']}" for s, lab in seeds)]]))
    body.append("<h2>3. What happened</h2>")
    # paired dot plot + d_amp panel (two panels, one quantity each)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained", gridspec_kw=dict(width_ratios=[1.3, 1]))
    xs = ["noise", "current", "alternative"]
    for k, (s, lab) in enumerate(seeds):
        ys = [get(o, s) for o in xs]
        pts = [(i, y) for i, y in enumerate(ys) if y is not None]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=GRID, lw=1.5, zorder=1)
        for i, y in pts:
            ax.plot(i, y, ORDER_STYLE[xs[i]][2], color=ORDER_STYLE[xs[i]][1], ms=8, mec="white", mew=1.5, zorder=2)
        if pts:
            ax.annotate(lab, (pts[-1][0], pts[-1][1]), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=8, color=MUTED)
    ax.set_xticks(range(3), [ORDER_STYLE[o][0] for o in xs]); ax.set_xlim(-.3, 2.8)
    ax.set_ylabel("eC (0 exact, 1 = empty model)"); ax.set_title("Error relative to the known source", fontsize=9)
    off = {"current": -.08, "alternative": .08}
    for o in ("current", "alternative"):
        for i, (s, lab) in enumerate(seeds):
            if d[o][i] is not None:
                bx.plot(i + off[o], d[o][i], ORDER_STYLE[o][2], color=ORDER_STYLE[o][1], ms=8, mec="white", mew=1.5,
                        label=ORDER_STYLE[o][0] if i == 0 else None)
    bx.axhline(0, color=INK, lw=1)
    bx.set_xticks(range(3), [lab for _, lab in seeds]); bx.set_ylabel("d_amp = eC(gain + noise) − eC(noise only)")
    bx.set_title("Negative = adding the gain improved recovery", fontsize=9); bx.legend(fontsize=8, frameon=False)
    body.append(run.save(fig, "order_dots.png"))
    body.append("<p class=cap>Here the gain either changes only the source signal or changes both signal and noise. We kept "
                "the weights fixed. Lines connect the same noise realization.</p>")
    # flux ratio
    fig, ax = plt.subplots(figsize=(6.4, 3.2), layout="constrained")
    for i, o in enumerate(xs):
        for k, (s, lab) in enumerate(seeds):
            y = run.m(f"order-{o}-{s}", "flux_ratio_total")
            if y is not None:
                ax.plot(i + (k - 1) * .12, y, ORDER_STYLE[o][2], color=ORDER_STYLE[o][1], ms=8, mec="white", mew=1.5)
    ax.axhline(1, color=INK, lw=1); ax.text(2.45, 1, " true flux", va="center", fontsize=8, color=MUTED)
    ax.set_xticks(range(3), [ORDER_STYLE[o][0] for o in xs]); ax.set_xlim(-.4, 2.8)
    ax.set_ylabel("total CLEAN-model flux ÷ true flux"); ax.set_title("Source recovered by CLEAN (three seeds per condition, left to right)", fontsize=9)
    body.append(run.save(fig, "order_flux.png"))
    half = 4.
    cols = ["order-ctrl-V", "order-noise-archived", "order-current-archived", "order-alternative-archived"]
    if all(run.ok(c) for c in cols):
        titles = ["noiseless V", "noise only", "current G·V+n", "alternative G·(V+n)"]
        body.append(image_grid(run, [("model ⊛ beam", lambda p: p["C"], False), ("model − truth", lambda p: p["C"] - p["T"], True)],
                               cols, [f"{t}\neC={run.m(c, 'eC'):.3f}" for t, c in zip(titles, cols)], half, "order_images.png"))
        body.append(f"<p class=cap>Archived seed. Top: raw CLEAN model convolved with the evaluation beam (the noiseless "
                    f"column is effectively the truth, eC≈{run.m('order-ctrl-V', 'eC'):.1e}); bottom: model minus truth. "
                    f"Shared scale within each row, ±{half:g}″ field, ellipse = restoring beam, divided by true flux "
                    f"{src['flux_jy']:.4f} Jy.</p>")
    body.append("<h2>4. What it means</h2>")
    body.append(f"<p>{esc(verdict)} This is one source, one antenna, one gain and noise level, and three noise realizations; "
                "it describes these cases, not a population.</p>")
    ctrl = [[c, run.records[c]["status"], run.m(c, "eC"), run.m(c, "flux_ratio_total"),
             (run.records[c].get("clean") or {}).get("iterdone"), run.records[c].get("reused_from") or ""]
            for c in ("order-ctrl-V", "order-ctrl-GV", "order-ctrl-GV0")]
    body.append("<h3>Zero-noise controls</h3>" + table(["case", "status", "eC", "flux ratio", "iterations", "reused from"], ctrl))
    gv, gv0 = run.records["order-ctrl-GV"], run.records["order-ctrl-GV0"]
    same = gv.get("data", {}).get("data_sha256") == gv0.get("data", {}).get("data_sha256") is not None
    body.append(f"<p class=cap>G·V and G·(V+0) stored DATA bit-identical: {fmt(same)} (so the image was reused).</p>")
    rows = []
    for s, lab in seeds:
        chk = run.records[f"order-current-{s}"].get("order_check") or {}
        cur, alt = run.records[f"order-current-{s}"].get("data", {}), run.records[f"order-alternative-{s}"].get("data", {})
        na = lambda x, k: ((x.get("noise") or {}).get(k) or {}).get("effective_noise_rms_component_jy")
        rows.append([lab, fmt(chk.get("unaffected_rows_identical")), chk.get("max_difference_error_over_storage_tolerance"),
                     fmt(cur.get("unaffected_sha256") == alt.get("unaffected_sha256") and cur.get("unaffected_sha256") is not None),
                     na(cur, "unaffected"), na(cur, "affected"), na(alt, "affected"),
                     max(cur.get("max_error_over_storage_tolerance", np.nan), alt.get("max_error_over_storage_tolerance", np.nan))])
    body.append("<h3>Visibility checks before imaging</h3>" + table(
        ["seed", "unaffected rows identical", "(alt−cur) − (G−1)n, ÷ storage tol.", "stored unaffected DATA equal",
         "noise rms unaffected (Jy)", "noise rms affected, current", "noise rms affected, alternative",
         "max stored-DATA error ÷ tol."], rows))
    body.append("<p class=cap>A = 1 makes both orders identical algebraically (G = 1); this is covered by a unit test, not an "
                "extra image. Storage tolerance = 8 float32 ulps of the largest magnitude.</p>")
    arch = run.plan["archived"]
    prep = next((json.loads(p.read_text()).get("source_preparation") for p in sorted((run.root / "runs").glob("order*.json"))
                 if json.loads(p.read_text()).get("source_preparation")), {}) or {}
    seed0 = str(run.records["order-noise-archived"]["plan"]["seed"])
    rep = [["noiseless V DATA hash", fmt(prep.get("v_sha256") == arch["v_sha256"])],
           ["nominal noise hash (archived seed)", fmt((prep.get("regular", {}).get(seed0) or {}).get("noise_sha256") == arch["nominal_noise_sha256"])],
           ["current-order eC: runner vs archived", f"{fmt(run.m('order-current-archived', 'eC'), 10)} vs {fmt(arch['eC']['current'], 10)}"],
           ["noise-only eC: runner vs archived", f"{fmt(run.m('order-noise-archived', 'eC'), 10)} vs {fmt(arch['eC']['noise'], 10)}"]]
    body.append("<h3>Reproduction of the archived example</h3>" + table(["check", "result"], rep) +
                f"<p class=cap>Archived run: {esc(arch['run'])} (dataset row r08:m5, Week 11 note).</p>")
    body.append("<h2>5. Should we change anything?</h2>")
    if have and verdict.startswith("The rescue remains"):
        rec = ("No change to the production noise/gain order is needed to explain this rescue: it survives the alternative "
               "order here. Keep the dataset convention; treat the choice of order as a separate physical-modelling decision.")
    elif have and verdict.startswith("The rescue weakens"):
        rec = ("The rescue depends on the coupling assumption. Before changing the generator, decide which order describes the "
               "intended instrumental error; a bounded follow-up is one more amplitude or noise level.")
    else:
        rec = "No change yet; a bounded follow-up (one more amplitude or noise level) would be needed."
    body.append(f"<p>{esc(rec)}</p>")
    app = []
    for s, lab in seeds[1:]:
        ids = [f"order-noise-{s}", f"order-current-{s}", f"order-alternative-{s}"]
        if all(run.ok(c) for c in ids):
            app.append(image_grid(run, [("model − truth", lambda p: p["C"] - p["T"], True)], ids,
                                  [f"{ORDER_STYLE[o][0]}\neC={run.m(c, 'eC'):.3f}" for o, c in zip(xs, ids)], half,
                                  f"order_appendix_{s}.png", suptitle=lab))
    body.append("<h2>Appendix: the other seeds</h2>" + "".join(app) + "<p class=cap>Same crop and beam; each row has its own "
                "shared diverging scale.</p>")
    body.append(tech(run, "order"))
    (run.root / "order.html").write_text(page("Experiment 1 · Gain/noise application order", "".join(body)))
    return lead + " " + verdict


# ---------------------------------------------------------------- experiment 2

def stopping_page(run):
    pid = lambda cov, pol, r: f"stop-{cov}-{pol}-r{r:g}"
    R = (0.02, 0.08, 0.16)
    covs = (("full", "full uv-coverage"), ("cut50", "50% angular cut"))
    pols = (("adaptive", "normal noise-based stopping", C1, "o"), ("fixed", "fixed stopping level", C2, "s"))
    src = run.plan["sources"]["2241+098"]
    body = []
    # lead from measurements
    def series(cov, pol, key):
        return [run.m(pid(cov, pol, r), key) for r in R]
    limit = [cid for cid, r in run.records.items() if r["plan"]["experiment"] == "stopping" and (r.get("clean") or {}).get("iteration_limit_reached")]
    facts = []
    for cov, lab in covs:
        fa, ff = series(cov, "adaptive", "flux_ratio_total"), series(cov, "fixed", "flux_ratio_total")
        ea, ef = series(cov, "adaptive", "eC"), series(cov, "fixed", "eC")
        if None in fa + ff + ea + ef:
            facts.append(f"{lab}: incomplete")
            continue
        facts.append(f"{lab}: from r=0.02 to 0.16 the model flux ratio goes {fa[0]:.2f}→{fa[-1]:.2f} with adaptive stopping "
                     f"and {ff[0]:.2f}→{ff[-1]:.2f} with the fixed level; eC {ea[0]:.3f}→{ea[-1]:.3f} vs {ef[0]:.3f}→{ef[-1]:.3f}")
    lead = "; ".join(facts) + "."
    body.append(f"<p class=lead>{esc(lead)} <b>One source, one noise realization.</b>"
                + (f" <span class=bad>{len(limit)} case(s) reached the 10,000-iteration limit (not converged to the threshold).</span>" if limit else "")
                + "</p>")
    body.append("<h2>1. Question</h2><p>Does more noise worsen CLEAN recovery partly because CLEAN's noise-based stopping "
                "rule (nsigma = 3) stops earlier when the residual is noisier?</p>")
    thr = {cov: (run.records[pid(cov, "fixed", R[0])].get("derived_threshold") or {}) for cov, _ in covs}
    cuts = run.records[pid("cut50", "adaptive", R[0])].get("cut") or {}
    body.append("<h2>2. What changed and what stayed fixed</h2>" + table(["", "Setting"], [
        ["Changed", "noise strength r = a/SNR₀ ∈ {0.02, 0.08, 0.16}; stopping policy; uv-coverage (full / 50% angular cut)"],
        ["Normal stopping", "nsigma 3 + absolute threshold 1e-7 × dirty peak (the production default)"],
        ["Fixed stopping", "nsigma 0, no dirty-peak fraction, absolute threshold = 3 × off-source residual scaled MAD of the "
                           "r = 0.02 normal run of the same coverage: " +
                           ", ".join(f"{c}: {fmt(t.get('value_jy'))} Jy" for c, t in thr.items())],
        ["Fixed", f"2241+098 (flux {src['flux_jy']:.4f} Jy, SNR₀ {src['snr0']:.0f}), no gain errors, one noise seed, noise "
                  "generated before the cut, unchanged weights, full-coverage 256×256 grid, CLEAN mask and restoring beam, "
                  "minor-iteration ceiling 10,000 for both policies (production uses 1,000,000)"],
        ["Cut", f"atan2(v,u) mod 180° in [0°, 90°) and conjugate flagged: {fmt(cuts.get('usable_fraction_removed'))} of usable "
                "visibility samples removed (angular fraction 0.5)"]]))
    body.append("<h2>3. What happened</h2>")
    keys = [("flux_ratio_total", "model flux ÷ true flux", 1.0), ("eC", "eC", None), ("iterdone", "CLEAN iterations", None)]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), layout="constrained", sharex=True)
    for i, (cov, clab) in enumerate(covs):
        for j, (key, ylab, ref) in enumerate(keys):
            ax = axes[i, j]
            for pol, plab, col, mk in pols:
                ys = [((run.records[pid(cov, pol, r)].get("clean") or {}).get("iterdone") if key == "iterdone" else run.m(pid(cov, pol, r), key))
                      if run.ok(pid(cov, pol, r)) else None for r in R]
                pts = [(r, y) for r, y in zip(R, ys) if y is not None]
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "-" + mk, color=col, ms=7, mec="white", mew=1.2,
                        label=plab if (i, j) == (0, 0) else None)
                for r in R:
                    if (run.records[pid(cov, pol, r)].get("clean") or {}).get("iteration_limit_reached"):
                        y = run.records[pid(cov, pol, r)]["clean"]["iterdone"] if key == "iterdone" else run.m(pid(cov, pol, r), key)
                        ax.plot(r, y, "o", ms=13, mfc="none", mec=INK, mew=1)
            if ref is not None:
                ax.axhline(ref, color=INK, lw=1)
            if key == "iterdone":
                ax.set_yscale("log"); ax.axhline(10_000, color=MUTED, lw=1, ls="--")
            ax.set_ylabel(ylab)
            if i == 0:
                ax.set_title(ylab, fontsize=9)
            if j == 0:
                ax.text(-.32, .5, clab, transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=9, weight="bold")
            if i == 1:
                ax.set_xlabel("noise strength r = a/SNR₀")
            ax.set_xticks(R)
    axes[0, 0].legend(fontsize=8, frameon=False)
    body.append(run.save(fig, "stopping_lines.png"))
    body.append("<p class=cap>Rows: uv-coverage. Black line: true flux (left) or the 10,000-iteration ceiling (right, dashed). "
                "A black ring marks a case that hit the iteration ceiling.</p>")
    rows = []
    for cov, _ in covs:
        for pol, plab, *_ in pols:
            for r in R:
                rec = run.records[pid(cov, pol, r)]
                k = rec.get("clean") or {}
                rows.append([cov, plab, r, rec["status"], k.get("iterdone"), k.get("nmajordone"), k.get("stop_description"),
                             k.get("tclean_threshold_jy"), k.get("tclean_nsigma"), k.get("final_minor_cycle_threshold_jy"),
                             k.get("final_peak_residual_jy"), fmt(k.get("iteration_limit_reached"))])
    body.append(table(["coverage", "policy", "r", "status", "iterations", "major cycles", "stop reason", "absolute threshold (Jy)",
                       "nsigma", "last minor-cycle threshold (Jy)", "final peak residual (Jy)", "iteration limit"], rows))
    body.append("<p class=cap>The last minor-cycle threshold is CASA's cycleThresh, not necessarily the global stopping "
                "threshold; CASA does not return the threshold it derives from nsigma, so it stays missing.</p>")
    ids = [pid("full", pol, r) for pol in ("adaptive", "fixed") for r in R]
    if all(run.ok(c) for c in ids):
        titles = [f"{'normal' if c.split('-')[2] == 'adaptive' else 'fixed'}, r={c.rsplit('r', 1)[1]}\neC={run.m(c, 'eC'):.3f}" for c in ids]
        body.append(image_grid(run, [("model ⊛ beam", lambda p: p["C"], False),
                                     ("residual (material left unfitted)", lambda p: p["residual"], True)],
                               ids, titles, 12., "stopping_images.png"))
        body.append(f"<p class=cap>Full coverage. Each row has one shared scale across all six cases; ±12″ field; ÷ true flux "
                    f"{src['flux_jy']:.4f} Jy; ellipse = restoring beam.</p>")
    body.append("<h2>4. What it means</h2>")
    msgs = []
    for cov, lab in covs:
        fa, ff = series(cov, "adaptive", "flux_ratio_total"), series(cov, "fixed", "flux_ratio_total")
        ea, ef = series(cov, "adaptive", "eC"), series(cov, "fixed", "eC")
        oa, of = series(cov, "adaptive", "outside_aperture_abs_flux_ratio"), series(cov, "fixed", "outside_aperture_abs_flux_ratio")
        if None in fa + ff + ea + ef:
            continue
        drop_a, drop_f = fa[0] - fa[-1], ff[0] - ff[-1]
        flux_txt = (f"the fixed level reduces the missing-flux trend (flux drop {drop_f:+.2f} vs {drop_a:+.2f})" if drop_f < drop_a - .02
                    else f"the missing-flux trend persists under the fixed level (flux drop {drop_f:+.2f} vs {drop_a:+.2f})")
        ec_txt = ("lowers eC at the highest noise" if ef[-1] < ea[-1] - .005 else
                  "raises eC at the highest noise" if ef[-1] > ea[-1] + .005 else "leaves eC at the highest noise about the same")
        msgs.append(f"{lab}: {flux_txt}, and {ec_txt} ({ea[-1]:.3f} → {ef[-1]:.3f}); model flux outside the source aperture "
                    f"at r=0.16 is {oa[-1]:.2f} vs {of[-1]:.2f} of the true flux.")
    body.append("<ul>" + "".join(f"<li>{esc(x)}</li>" for x in msgs) + "</ul>")
    body.append("<p>More components or lower residuals alone are not better recovery; eC and the outside-aperture flux show "
                "whether the extra flux is the source or false structure. Differences between coverages cannot be assigned to "
                "CLEAN stopping alone (the cut changes sampling and sensitivity). One noise realization: illustrative, not a "
                "significance test." + (" Cases at the iteration ceiling did not reach the fixed threshold." if limit else "") + "</p>")
    body.append("<h2>5. Should we change anything?</h2><p>" + esc(stopping_recommendation(run, series, covs)) + "</p>")
    body.append(tech(run, "stopping"))
    (run.root / "stopping.html").write_text(page("Experiment 2 · CLEAN stopping and noise", "".join(body)))
    return lead


def stopping_recommendation(run, series, covs):
    fa, ff = series("full", "adaptive", "flux_ratio_total"), series("full", "fixed", "flux_ratio_total")
    ea, ef = series("full", "adaptive", "eC"), series("full", "fixed", "eC")
    if None in fa + ff + ea + ef:
        return "Incomplete; no recommendation."
    if ef[-1] < ea[-1] - .005 and (ff[0] - ff[-1]) < (fa[0] - fa[-1]) - .02:
        return ("Adaptive stopping accounts for part of the noise effect. Keep production stopping unchanged for the dataset "
                "(it is the realistic pipeline behaviour) but interpret noise damage as partly a stopping-rule effect; a bounded "
                "follow-up is repeating the full-coverage pair with two more noise seeds.")
    return ("Noise affects recovery beyond the adaptive threshold here; no change to the stopping rule is indicated. Bounded "
            "follow-up only if needed: the full-coverage pair with two more noise seeds.")


# ---------------------------------------------------------------- experiment 3

def sampling_page(run):
    ids = ["sample-256", "sample-512", "sample-1024"]
    src = run.plan["sources"]["2241+098"]
    flux = src["flux_jy"]
    ok = [c for c in ids if run.ok(c)]
    rows, pts = [], []
    for c in ids:
        rec = run.records[c]
        m, k = rec.get("metrics") or {}, rec.get("clean") or {}
        g = m.get("grid") or {}
        rows.append([c, rec["status"], rec["plan"]["grid"]["image_grid"]["imsize"][0], g.get("cell_arcsec"), g.get("pixels_per_beam_minor"),
                     m.get("eC"), (m.get("residual_offsource") or {}).get("scaled_mad", np.nan) / flux if m else None,
                     (m.get("residual_full") or {}).get("abs_peak", np.nan) / flux if m else None,
                     m.get("flux_ratio_total"), k.get("iterdone"), fmt(k.get("iteration_limit_reached")), k.get("stop_description"), rec.get("seconds")])
        if m:
            pts.append((g["pixels_per_beam_minor"], m["eC"], m["residual_offsource"]["scaled_mad"] / flux,
                        m["residual_full"]["abs_peak"] / flux, rec.get("seconds"), k.get("iteration_limit_reached")))
    if len(pts) == 3:
        e = [p[1] for p in pts]
        trend = ("plateaus" if abs(e[2] - e[0]) < .1 * e[0] else "decreases consistently" if e[0] > e[1] > e[2]
                 else "does not change monotonically")
        lim = any(p[5] for p in pts)
        stops = sorted({str((run.records[c].get("clean") or {}).get("stop_description")) for c in ids})
        lead = (f"Noiseless eC {trend} with finer pixels: {e[0]:.2e} (256 px) → {e[1]:.2e} (512) → {e[2]:.2e} (1024), "
                f"all far below the 0.05 working tolerance; elapsed {pts[0][4]:.0f} s → {pts[2][4]:.0f} s."
                + (" At least one case reached the iteration ceiling." if lim else " No case reached the iteration ceiling; "
                   "CASA stopped with: " + "; ".join(stops) + " (before the requested absolute threshold)."))
    else:
        lead, trend, lim = "Incomplete.", "", False
    body = [f"<p class=lead>{esc(lead)} <b>One centred point source, noiseless.</b></p>"]
    body.append("<h2>1. Question</h2><p>Does a smaller image pixel lower the numerical reconstruction floor enough to matter "
                "for eC? (Image sampling only; not the visibility gridding kernel.)</p>")
    c0 = run.plan["sources"]["2241+098"]["reference"]["grid"]["image_grid"]["cell_arcsec"][0]
    body.append("<h2>2. What changed and what stayed fixed</h2>" + table(["", "Setting"], [
        ["Changed", f"grid (256, c), (512, c/2), (1024, c/4), c = {c0:.5f}″ (historical reference grid, "
                    "runs/clean-recovery-original-flux-20261104)"],
        ["Fixed", f"noiseless 2241+098 (flux {flux:.4f} Jy), same angular field ({256 * c0:.2f}″) and centre, CLEAN mask (6 "
                  "beams), Briggs 0.5 weighting, restoring/evaluation beam"],
        ["CLEAN", f"nsigma 0, absolute threshold 1e-7 × true flux = {1e-7 * flux:.3g} Jy, no dirty-peak fraction, ceiling "
                  "10,000 minor iterations (a numerical probe, not a production setting)"],
        ["Metric", "eC on each native grid over the whole field; raw model (Jy/pixel) convolved directly (zero-padded) "
                   "with the peak-one beam → Jy/beam, truth evaluated analytically on the same grid; no interpolation"]]))
    body.append("<h2>3. What happened</h2>")
    if pts:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), layout="constrained")
        for ax, (j, lab) in zip(axes, [(2, "residual scaled MAD ÷ flux (off-source)"), (3, "residual |peak| ÷ flux"), (1, "eC")]):
            xs = [p[0] for p in pts]; ys = [p[j] for p in pts]
            ax.plot(xs, ys, "-o", color=C1, ms=7, mec="white", mew=1.2)
            for x, y, p in zip(xs, ys, pts):
                ax.annotate(f"{y:.2e}", (x, y), xytext=(4, 5), textcoords="offset points", fontsize=7.5, color=MUTED)
                if p[5]:
                    ax.plot(x, y, "o", ms=13, mfc="none", mec=INK)
            if all(y > 0 for y in ys):
                ax.set_yscale("log")
            ax.set_xscale("log", base=2); ax.set_xticks(xs, [f"{x:.1f}" for x in xs])
            ax.set_xlabel("pixels across beam minor FWHM"); ax.set_title(lab, fontsize=9)
        body.append(run.save(fig, "sampling_lines.png"))
    body.append(table(["case", "status", "imsize", "cell (″)", "pixels / beam minor", "eC", "residual MAD ÷ flux",
                       "residual |peak| ÷ flux", "model flux ÷ true", "iterations", "iteration limit", "stop reason", "elapsed (s)"], rows))
    if len(ok) == 3:
        body.append(image_grid(run, [("residual", lambda p: p["residual"], True), ("model − truth", lambda p: p["C"] - p["T"], True)],
                               ids, [f"{c.split('-')[1]} px\neC={run.m(c, 'eC'):.2e}" for c in ids], 6., "sampling_images.png"))
        body.append("<p class=cap>Common angular axes (±6″), shared signed scale per row, ÷ true flux; ellipse = restoring beam.</p>")
        fig, ax = plt.subplots(figsize=(6.4, 3), layout="constrained")
        for c, col, mk in zip(ids, (C1, C2, C3), ("o", "s", "D")):
            p = run.load(c)
            ny, nx = p["residual"].shape
            x = (np.arange(nx) - nx // 2) * p["cell"]
            sel = np.abs(x) <= 6
            ax.plot(x[sel], p["residual"][ny // 2, sel] / flux, "-", color=col, lw=1.5, label=f"{c.split('-')[1]} px")
        ax.axhline(0, color=INK, lw=.8); ax.set_xlabel("ΔRA through the centre (arcsec)"); ax.set_ylabel("residual ÷ flux")
        ax.legend(fontsize=8, frameon=False); ax.set_title("Central residual slice", fontsize=9)
        body.append(run.save(fig, "sampling_slice.png"))
    body.append("<h2>4. What it means</h2>")
    if len(pts) == 3:
        body.append(f"<p>The noiseless floor {trend}. Its largest value, {max(p[1] for p in pts):.2e}, is "
                    f"{max(p[1] for p in pts) / TOL:.1e} of the 0.05 working tolerance (a practical tolerance, not a significance "
                    "threshold), so image sampling is not what limits the eC values studied in the other experiments. A centred point "
                    "source is a favourable case; this does not prove arbitrary structures are well sampled.</p>")
        body.append("<h2>5. Should we change anything?</h2><p>No: keep the cheaper production sampling (4 pixels across the beam "
                    "minor axis, 256 px). No gridder/kernel sweep is indicated.</p>")
    body.append(tech(run, "sampling"))
    (run.root / "sampling.html").write_text(page("Experiment 3 · Noiseless image sampling", "".join(body)))
    return lead


# ---------------------------------------------------------------- experiment 4

def selfcal_page(run):
    path = run.root / "selfcal_audit.json"
    if not path.is_file():
        (run.root / "selfcal.html").write_text(page("Experiment 4 · Self-calibration decisions", "<p class=bad>Audit not run.</p>"))
        return "Audit not run."
    a = json.loads(path.read_text())
    if a.get("status") != "ok":
        (run.root / "selfcal.html").write_text(page("Experiment 4 · Self-calibration decisions",
                                                    f"<p class=bad>Missing input: {esc(a)}</p>"))
        return "Self-calibration input missing."
    cases = {c["case"]: c for c in a["cases"]}
    cnt = a["counts"]["category"]
    labels = {"low_initial_snr_block": "blocked: initial S/N too low", "rejected_correction": "blocked: solution lowered S/N",
              "retained_improved": "kept, recovery improved/solved", "retained_similar_or_worse": "kept, recovery similar/degraded",
              "mask_or_model_failure": "mask/model failure", "other": "other (incl. errors)", "unreadable": "unreadable record"}
    n = sum(cnt.values())
    kept = cnt.get("retained_improved", 0) + cnt.get("retained_similar_or_worse", 0)
    lead = (f"Of {n} recorded cases, hif_selfcal kept a correction in {kept} ({kept / n:.0%}): {cnt.get('retained_improved', 0)} "
            f"with improved or solved recovery and {cnt.get('retained_similar_or_worse', 0)} with similar or degraded recovery; it "
            f"refused {cnt.get('low_initial_snr_block', 0)} because the estimated S/N was too low before trying, and rejected "
            f"{cnt.get('rejected_correction', 0)} trial solutions because the image S/N fell.")
    body = [f"<p class=lead>{esc(lead)} <b>Existing run, read only.</b></p>"]
    body.append("<h2>1. Question</h2><p>What do the self-calibration acceptance and rejection checks in our pipeline run actually "
                "decide, and how do those decisions relate to source recovery?</p>")
    body.append("<h2>2. What was read</h2>" + table(["", ""], [
        ["Run", f"{a['run']} (B recipe, phase+amp self-cal, 256 px, central mask, flagging on)"],
        ["Source of records", f"{a['source']} — synced mirror: description JSON per case + before/after PNGs; no native weblog"],
        ["Pipeline", a.get("pipeline")], ["Recorded non-default arguments", esc(json.dumps({k: v for k, v in (a.get('recorded_arguments') or {}).items() if k != 'usermask'}))],
        ["Not available", "<ul>" + "".join(f"<li>{esc(x)}</li>" for x in a["missing_native_fields"]) + "</ul>"]]))
    body.append("<h2>3. What happened</h2>")
    order = [k for k in labels if k in cnt]
    fig, ax = plt.subplots(figsize=(7.5, .45 * len(order) + .8), layout="constrained")
    ax.barh(range(len(order)), [cnt[k] for k in order], color=C1, height=.6)
    for i, k in enumerate(order):
        ax.text(cnt[k], i, f" {cnt[k]}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(order)), [labels[k] for k in order]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("cases"); ax.set_title(f"Recorded outcome per case ({n} cases, none unavailable)" if "unreadable" not in cnt else "Recorded outcome", fontsize=9)
    body.append(run.save(fig, "selfcal_counts.png"))
    sel = [(k, cases[a["selected"][k]]) for k in labels if k in a["selected"]]
    if sel:
        fig, ax = plt.subplots(figsize=(8.5, 3.6), layout="constrained")
        for i, (k, c) in enumerate(sel):
            b, af, ref = c.get("eC_before"), c.get("eC_after"), c.get("eC_reference")
            if b is not None and af is not None:
                ax.plot([i - .12, i + .12], [b, af], "-", color=GRID, lw=1.5)
                ax.plot(i - .12, b, "o", color="#52514e", ms=8, mec="white", label="before" if i == 0 else None)
                ax.plot(i + .12, af, "s", color=C1, ms=8, mec="white", label="after" if i == 0 else None)
            if ref is not None:
                ax.plot(i, ref, "_", color=C2, ms=18, mew=2.5, label="matched reference" if i == 0 else None)
        ax.set_yscale("log"); ax.set_xticks(range(len(sel)), [labels[k].replace(", ", ",\n").replace(": ", ":\n").replace("d/", "d/\n") for k, _ in sel], fontsize=7)
        ax.set_ylabel("eC (log)"); ax.legend(fontsize=8, frameon=False); ax.set_title("Selected examples (first case ID per category)", fontsize=9)
        body.append(run.save(fig, "selfcal_paired.png"))
        body.append("<p class=cap>The matched reference is the eC of the same case without gain errors — a benchmark, not a "
                    "guaranteed achievable minimum (noise-only cases can end below it). Recovery labels are the recorded QA v5.1 labels, "
                    "which use an absolute 0.05 eC tolerance: a small absolute drop (e.g. 0.019 → 0.0002) is still “similar”.</p>")
    rows, thumbs = [], []
    src_dir = Path(a["source"])
    for k, c in sel:
        snr = c.get("estimated_snr_per_solint") or {}
        tried = ", ".join(f"{t['solint']} {'pass' if t['passed'] else 'fail'}" for t in c.get("attempted") or []) or "nothing attempted"
        why = c.get("stop_reason_lib") or "; ".join(sorted(set(x.strip() for x in c.get("stop_reasons") or []))) or "—"
        kept_txt = f"yes ({c['retained_interval']})" if c.get("application") == "applied" else "no"
        rec_txt = f"{c.get('recovery')} (eC {fmt(c.get('eC_before'))} → {fmt(c.get('eC_after'))}, reference {fmt(c.get('eC_reference'))})"
        snr_txt = ", ".join(f"{s} {v:g}" for s, v in snr.items())
        detail = (f"<details><summary>{esc(c['label'])}</summary>estimated S/N per solint: {esc(snr_txt) or '—'}<br>"
                  f"configured solints: {esc(c.get('solints_configured'))}<br>attempts: {esc(json.dumps(c.get('attempted')))}<br>"
                  f"image S/N {esc(c['image_snr'])}, RMS {esc(c['image_rms'])}, flag fraction after {fmt(c.get('flag_fraction_after'))}<br>"
                  f"group {esc(c.get('group'))}/{esc(c.get('subgroup'))}, D = {fmt(c.get('D'))}<br>{esc(c.get('description'))}</details>")
        rows.append([labels[k], f"<span>{esc(tried)}</span>", f"<span>{esc(why)}</span>", kept_txt, f"<span>{esc(rec_txt)}</span>", detail])
        imgs = sorted(c.get("image") or [])
        if len(imgs) == 2:
            for p in imgs:
                shutil.copy2(p, run.assets / Path(p).name)
            after, before = sorted(Path(p).name for p in imgs)
            thumbs.append(f"<figure style='display:inline-block;margin:4px'><img src='report_assets/{esc(before)}' width=200 style='display:inline'>"
                          f"<img src='report_assets/{esc(after)}' width=200 style='display:inline'><figcaption class=cap>{esc(labels[k])}: "
                          f"before | after (same per-case colour scale, from the QA report)</figcaption></figure>")
    body.append(table(["Category", "What was attempted", "Why it stopped", "Was a correction kept?", "Did source recovery improve?", "Details"], rows))
    if thumbs:
        body.append("<p>Restored-image thumbnails reused from the QA report; each before/after pair shares one colour scale.</p>" + "".join(thumbs))
    low = [c for c in cases.values() if c["category"] == "low_initial_snr_block" and c.get("estimated_snr_per_solint")]
    rej = [c for c in cases.values() if c["category"] == "rejected_correction" and c.get("estimated_snr_per_solint")]
    longest = lambda c: max(c["estimated_snr_per_solint"].values())
    stats = []
    for name, group in (("blocked before trying", low), ("trial rejected", rej),
                        ("correction kept", [c for c in cases.values() if c["category"].startswith("retained") and c.get("estimated_snr_per_solint")])):
        if group:
            v = np.array([longest(c) for c in group])
            stats.append([name, len(group), float(np.min(v)), float(np.median(v)), float(np.max(v))])
    body.append("<h3>Estimated S/N of the longest solution interval</h3>" + table(["outcome", "cases", "min", "median", "max"], stats) +
                "<p class=cap>From the pipeline log lines “Estimated SNR per solint” kept in the synced records. The proceed threshold "
                "itself is not recorded in this run.</p>")
    body.append("<h2>4. What it means</h2><p>The pipeline has two different gates: an eligibility gate on the estimated per-interval "
                "S/N before any solution is tried, and an acceptance gate after a trial (image S/N must not fall, RMS must not rise "
                "beyond 5%). A blocked case is one the algorithm declined to attempt; that is not proof it is unsolvable. A kept "
                "correction passed the image-S/N checks; that is not proof it is physically correct — the recovery column shows "
                "whether eC actually moved. Flagging and self-calibration are separate actions; one solvability test does not cover "
                "both.</p>")
    body.append("<h2>5. Should we change anything?</h2><p>No pipeline change from this audit. If the eligibility gate matters for the "
                "ML labels, record the configured proceed threshold and per-interval gain-solution flags natively in the next QA run "
                "(no replacement runs now).</p>")
    (run.root / "selfcal.html").write_text(page("Experiment 4 · Existing self-calibration decisions", "".join(body)))
    return lead


# ---------------------------------------------------------------- index

def uvgrid_checks(run, ids):
    """Pre-sweep / post-sweep consistency from retained products (no CASA): pixel size, crop alignment, mask support,
    identical visibilities and invariants, Fourier-cell ratios."""
    from astropy.io import fits
    from astropy.wcs import WCS
    ok = [c for c in ids if run.ok(c)]
    if not ok or ids[0] not in ok:
        return []
    base = run.records[ids[0]]
    out = []
    for c in ok[1:]:
        r = run.records[c]
        h0, h = run.load(ids[0])["header"], run.load(c)["header"]
        n, m = run.load(c)["native_shape"][0], r["plan"]["crop"]
        sl = central_crop(n, m)
        w0, w = WCS(h0).celestial, WCS(h).celestial
        yy, xx = np.mgrid[0:m:17, 0:m:17]
        sky0 = w0.pixel_to_world(xx, yy)
        sky = w.pixel_to_world(xx + sl.start, yy + sl.start)
        sep = float(sky0.separation(sky).arcsec.max())
        with np.load(run.root / base["products"]["support"]) as s0, np.load(run.root / r["products"]["support"]) as s1:
            mask_same = bool(np.array_equal(s0["clean_search"], s1["clean_search"][sl, sl]))
            outside = int(s1["clean_search"].sum() - s1["clean_search"][sl, sl].sum())
        model = load_planes({k: run.root / v for k, v in r["products"].items()})["model"]
        out.append(dict(
            case=c, cell_equal=abs(abs(h["CDELT2"]) - abs(h0["CDELT2"])) <= 1e-12 * abs(h0["CDELT2"]),
            crop_max_sky_offset_arcsec=sep, mask_support_identical=mask_same, mask_pixels_outside_crop=outside,
            model_components_outside_crop=int(np.count_nonzero(model) - np.count_nonzero(model[sl, sl])),
            data_identical=r["data"]["data_sha256"] == base["data"]["data_sha256"],
            invariants_identical=r.get("invariants") == base.get("invariants"),
            fourier_cell_ratio=base["plan"]["fourier_cell_wavelengths"] / r["plan"]["fourier_cell_wavelengths"],
            native_shape_requested=r["plan"]["grid"]["image_grid"]["imsize"], native_shape_actual=list(run.load(c)["native_shape"])))
    return out


def uvgrid_report(run):
    ids = [c["id"] for c in run.plan["cases"]]
    src = run.plan["sources"]["2241+098"]
    flux = src["flux_jy"]
    ok = [c for c in ids if run.ok(c)]
    rows, pts = [], []
    for c in ids:
        rec = run.records[c]
        m, k, pc = rec.get("metrics") or {}, rec.get("clean") or {}, rec["plan"]
        rows.append([c, rec["status"], pc["grid"]["image_grid"]["imsize"][0], pc["grid"]["image_grid"]["cell_arcsec"][0],
                     pc["grid"]["image_grid"]["field_of_view_arcsec"][0], pc["fourier_cell_wavelengths"], m.get("eC"),
                     (m.get("residual_offsource") or {}).get("scaled_mad", np.nan) / flux if m else None,
                     (m.get("residual_full") or {}).get("abs_peak", np.nan) / flux if m else None,
                     m.get("flux_ratio_aperture"), k.get("iterdone"), k.get("nmajordone"), fmt(k.get("iteration_limit_reached")),
                     k.get("stop_description"), rec.get("seconds")])
        if m:
            pts.append((pc["fourier_cell_wavelengths"], m["eC"], m["residual_offsource"]["scaled_mad"] / flux,
                        m["residual_full"]["abs_peak"] / flux, rec.get("seconds"), k.get("iteration_limit_reached"), c))
    checks = uvgrid_checks(run, ids)
    (run.root / "checks").mkdir(exist_ok=True)
    (run.root / "checks" / "uvgrid.json").write_text(json.dumps(checks, indent=1, default=str) + "\n")
    stops = sorted({str((run.records[c].get("clean") or {}).get("stop_description")) for c in ok})
    if len(pts) == 3:
        e = [p[1] for p in pts]
        rel = (e[2] - e[0]) / e[0]
        trend = ("unchanged floor (less than 10% change)" if abs(rel) < .1 else "consistent reduction" if e[0] > e[1] > e[2]
                 else "consistent increase" if e[0] < e[1] < e[2] else "non-monotonic change")
        lead = (f"We kept image pixels fixed and made the Fourier grid 2× and 4× finer. Noiseless eC: {e[0]:.3e} (256 px) → "
                f"{e[1]:.3e} (512) → {e[2]:.3e} (1024): {trend}; all far below the 0.05 working tolerance. Elapsed "
                f"{pts[0][4]:.0f} s → {pts[2][4]:.0f} s. CASA stop reason(s): {'; '.join(stops)}.")
    else:
        lead, trend = f"Incomplete: {len(ok)} of {len(ids)} cases available. We kept image pixels fixed and made the Fourier grid finer.", ""
    body = [f"<p class=lead>{esc(lead)} <b>One centred point source, noiseless, natural weighting.</b></p>"]
    body.append("<p>eC = ‖C − T‖₂ / ‖T‖₂ (C: raw CLEAN model convolved with the peak-one restoring beam; T: the known source "
                "convolved identically): 0 is exact recovery, 1 the error of an empty model. Parent study: "
                "experiment 3 in runs/tanmoy-local-20261123 (Briggs 0.5, pixels shrunk; not rerun here).</p>")
    body.append("<h2>1. Question</h2><p>With the image pixel size fixed, does a finer Fourier-space grid (Δu = 1/(N c)) "
                "lower the noiseless residual or improve the source recovered by CLEAN? Enlarging N enlarges the computational "
                "field; it adds no baselines and does not improve angular resolution.</p>")
    c0 = src["reference"]["grid"]["image_grid"]["cell_arcsec"][0]
    m0 = run.plan["cases"][0]["crop"]
    body.append("<h2>2. What changed and what stayed fixed</h2>" + table(["", "Setting"], [
        ["Changed", f"image size N = 256, 512, 1024 → computational field {256 * c0:.2f}″, {512 * c0:.2f}″, {1024 * c0:.2f}″ "
                    f"and nominal Fourier cell {run.plan['cases'][0]['fourier_cell_wavelengths']:.1f}, "
                    f"{run.plan['cases'][1]['fourier_cell_wavelengths']:.1f}, {run.plan['cases'][2]['fourier_cell_wavelengths']:.1f} wavelengths"],
        ["Weighting", "<b>natural</b> for all three (Briggs density weighting would also change with the field); new "
                      "natural-weighted 256 baseline, not the old Briggs result"],
        ["Fixed", f"image cell c = {c0:.8f}″ (reference grid); identical noiseless 2241+098 DATA (flux {flux:.4f} Jy), flags, "
                  "nominal weights, phase centre; gridder standard, mtmfs nterms 1, loop gain 0.1, no taper; user CLEAN mask: "
                  "central circle of 6 reference-beam majors (no automask); stored reference restoring/evaluation beam"],
        ["CLEAN", f"nsigma 0, absolute threshold 1e-7 × true flux = {1e-7 * flux:.3g} Jy, no dirty-peak fraction, "
                  "10,000 minor iterations (numerical probe)"],
        ["Metric", f"full native model convolved (zero-padded) first, then C, T and residual cropped to the same central "
                   f"{m0}×{m0} pixels ({m0 * c0:.2f}″) — identical pixel centres, no interpolation; aperture rho ≤ 1, "
                   "off-source annulus as in experiment 3"],
        ["Not exposed", "CASA pads the image internally before the FFT (tclean padding, left at its default); the internal "
                        "FFT size is not reported, so spacings above are nominal 1/(N c). The 2× / 4× ratios hold either way."]]))
    if checks:
        body.append("<h2>Checks before interpreting</h2>" + table(
            ["case vs 256", "same cell", "crop sky offset (″)", "mask support identical", "mask px outside crop",
             "model components outside crop", "DATA identical", "weights/flags/UVW identical", "Fourier-cell ratio", "N requested / actual"],
            [[x["case"], fmt(x["cell_equal"]), x["crop_max_sky_offset_arcsec"], fmt(x["mask_support_identical"]),
              x["mask_pixels_outside_crop"], x["model_components_outside_crop"], fmt(x["data_identical"]),
              fmt(x["invariants_identical"]), x["fourier_cell_ratio"],
              f"{x['native_shape_requested'][0]} / {x['native_shape_actual'][0]}"] for x in checks]))
    body.append("<h2>3. What happened</h2>")
    if pts:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), layout="constrained")
        for ax, (j, lab) in zip(axes, [(2, "residual scaled MAD ÷ flux (off-source)"), (3, "residual |peak| ÷ flux"), (1, "eC")]):
            xs = [q[0] for q in pts]; ys = [q[j] for q in pts]
            ax.plot(xs, ys, "-o", color=C1, ms=7, mec="white", mew=1.2)
            for x, y, q in zip(xs, ys, pts):
                ax.annotate(f"{y:.4e}", (x, y), xytext=(0, 7), textcoords="offset points", fontsize=7.5, color=MUTED,
                            ha="center")
                if q[5]:
                    ax.plot(x, y, "o", ms=13, mfc="none", mec=INK)
            # linear axis from zero: a sub-percent change must not look like a trend
            ax.set_ylim(min(0., 1.3 * min(ys)), 1.3 * max(ys)); ax.axhline(0, color=INK, lw=.8)
            ax.set_xscale("log", base=2); ax.set_xticks(xs, [f"{x:.0f}" for x in xs]); ax.invert_xaxis()
            ax.set_xlabel("Fourier cell (wavelengths) → finer"); ax.set_title(lab, fontsize=9)
        body.append(run.save(fig, "uvgrid_lines.png"))
        body.append("<p class=cap>x axis runs from coarse (256 px) to fine (1024 px). Common central "
                    f"{m0}×{m0}-pixel region; values ÷ true flux; linear axes from zero; exact values in the table. A ring marks a case that reached the iteration ceiling.</p>")
    body.append(table(["case", "status", "N", "cell (″)", "field (″)", "Fourier cell (λ)", "eC", "residual MAD ÷ flux",
                       "residual |peak| ÷ flux", "aperture model flux ÷ true", "iterations", "major cycles", "iteration limit",
                       "stop reason", "elapsed (s)"], rows))
    if len(ok) == len(ids):
        body.append(image_grid(run, [("residual", lambda p: p["residual"], True), ("model − truth", lambda p: p["C"] - p["T"], True)],
                               ids, [f"N={c.split('-')[1]}\neC={run.m(c, 'eC'):.3e}" for c in ids], 6., "uvgrid_images.png"))
        body.append("<p class=cap>Identical angular crop (±6″ about the phase centre), shared signed scale per row, ÷ true "
                    "flux (Jy/beam); ellipse = restoring beam.</p>")
        fig, ax = plt.subplots(figsize=(6.4, 3), layout="constrained")
        for c, col in zip(ids, (C1, C2, C3)):
            p = run.load(c)
            ny, nx = p["residual"].shape
            x = (np.arange(nx) - nx // 2) * p["cell"]
            sel = np.abs(x) <= 6
            ax.plot(x[sel], p["residual"][ny // 2, sel] / flux, "-", color=col, lw=1.5, label=f"N={c.split('-')[1]}")
        ax.axhline(0, color=INK, lw=.8); ax.set_xlabel("ΔRA through the centre (arcsec)"); ax.set_ylabel("residual ÷ flux")
        ax.legend(fontsize=8, frameon=False); ax.set_title("Central residual slice", fontsize=9)
        body.append(run.save(fig, "uvgrid_slice.png"))
        b = run.load(ids[0])
        diff = []
        for c in ids[1:]:
            p = run.load(c)
            diff.append([c, float(np.abs(p["dirty"] - b["dirty"]).max() / flux), float(np.abs(b["dirty"]).max() / flux),
                         float(np.abs(p["psf"] - b["psf"]).max()), float(np.abs(p["residual"] - b["residual"]).max() / flux)])
        body.append("<h3>Dirty image and PSF relative to N = 256 (common central crop)</h3>" + table(
            ["case", "max |dirty − dirty₂₅₆| ÷ flux", "max |dirty₂₅₆| ÷ flux", "max |PSF − PSF₂₅₆| (peak 1)",
             "max |residual − residual₂₅₆| ÷ flux"], diff))
        dmax, rmax = max(d[1] for d in diff), max(d[4] for d in diff)
        body.append(f"<p>The finer grids change the dirty image and PSF by up to {dmax:.1e} of the peak, but the final "
                    f"residual by at most {rmax:.1e} of the true flux: gridding changes what CLEAN sees, and CLEAN removes "
                    "the difference down to the same floor.</p><p class=cap>Dirty image and PSF differences show changed "
                    "gridding; residual differences with agreeing inputs would point to deconvolution.</p>")
    body.append("<h2>4. What it means</h2>")
    if len(pts) == 3:
        e = [q[1] for q in pts]
        if trend.startswith("unchanged"):
            meaning = (f"Finer Fourier gridding left the noiseless floor unchanged under fixed natural weights (eC "
                       f"{e[0]:.6e} at every grid; largest relative difference {(max(e) - min(e)) / min(e):.1e}). With the stop reason(s) above this shows no material benefit under these "
                       "settings; it is not a converged-threshold test and does not rule out all gridding effects.")
        elif trend == "consistent reduction":
            meaning = (f"eC fell from {e[0]:.3e} to {e[2]:.3e}: a finer uv grid / larger computational field helps this "
                       "reconstruction under natural weights. It does not identify a kernel defect or add uv coverage.")
        else:
            meaning = f"eC changes without a consistent reduction ({trend}); no gridding benefit is established."
        body.append(f"<p>{esc(meaning)} The largest value, {max(e):.2e}, is {max(e) / TOL:.1e} of the 0.05 working tolerance "
                    "(a practical tolerance, not a significance threshold). One centred point source: a null result does not "
                    "establish behaviour for off-centre or extended sources.</p>")
        material = max(e) >= .1 * TOL and trend == "consistent reduction"
        body.append("<h2>5. Should we change anything?</h2><p>" + (
            "Possibly: the reduction is material relative to the tolerance; a bounded follow-up (the same three grids with "
            "Briggs 0.5, labelled as a combined gridding-and-weighting response) would be the next check."
            if material else "No: keep the present production grid (256 px). The effect, if any, is far below the 0.05 "
            "tolerance, so the optional Briggs 0.5 repeat is not indicated.") + "</p>")
    body.append(tech(run, "uvgrid"))
    worker = [json.loads(q.read_text()) for q in sorted((run.root / "runs").glob("*.json"))]
    body.append(f"<p class=cap>Data: <a href='measurements.csv'>measurements.csv</a> (one row per planned case). Worker wall "
                f"time {sum(w.get('seconds') or 0 for w in worker):.0f} s over {len(worker)} CASA worker run(s) including "
                f"source preparation; peak scratch {max([w.get('peak_scratch_bytes') or 0 for w in worker] + [0]) / 2**30:.2f} GiB.</p>")
    (run.root / "report.html").write_text(page("Experiment 3b · Finer Fourier-space gridding", "".join(body), back=False))
    check_links(run.root)
    print("REPORT", run.root / "report.html")
    print("[uvgrid]", lead)


# ---------------------------------------------------------------- experiment 1b

def orderval_verdict(dc, da):
    """Per-case label from paired seeds: d = d_amp per order; Δorder = d_alt − d_cur = eC(alt) − eC(cur)."""
    have = [i for i in range(len(dc)) if dc[i] is not None and da[i] is not None]
    if not have:
        return "not enough completed cases"
    delta = [da[i] - dc[i] for i in have]
    if all(x > 0 for x in delta) and all(da[i] >= 0 for i in have):
        return "rescue disappears under the alternative order"
    if all(x > 0 for x in delta):
        return "rescue persists but weakens under the alternative order"
    if all(da[i] < 0 for i in have):
        return "rescue persists"
    return "seed-dependent"


def orderval_report(run):
    seeds = [("archived", "archived seed"), ("new1", "new seed 1"), ("new2", "new seed 2")]
    xs = ["noise", "current", "alternative"]
    sel = json.loads((run.root / "selection.json").read_text())
    body, summary = [], []
    sids = list(run.plan["sources"])
    per = {}
    for sid in sids:
        get = lambda o, s, k="eC": run.m(f"{sid}-{o}-{s}", k)
        d = {o: [None if get(o, s) is None or get("noise", s) is None else get(o, s) - get("noise", s) for s, _ in seeds]
             for o in ("current", "alternative")}
        per[sid] = d
        summary.append((sid, orderval_verdict(d["current"], d["alternative"]), d))
    lead = " ".join(f"{sid}: {v}." for sid, v, _ in summary)
    body.append(f"<p class=lead>{esc(lead)} <b>Three noise realizations per case; {len(sids)} new sources, each selected "
                "for a strong archived rescue.</b></p>")
    body.append("<h2>1. Question</h2><p>Experiment 1 found the amplification “rescue” of noisy 0449+113 under the current "
                "simulation order. Does the same comparison repeat on other archived rescue cases: does the rescue persist, "
                "weaken or vanish when the gain also multiplies the noise, G·(V+n) instead of G·V+n?</p>")
    # selection
    rows = [[s["source"], s["split"], s["case"], s["noise_case"], s["antenna"], s["A"], s["a"], s["r"], s["noise_seed"],
             s["eC_noise"], s["eC_amp_noise"], s["d_amp"], s["aperture_ratio_noise"], s["aperture_ratio_amp_noise"]]
            for s in sel["selected"]]
    body.append("<h2>2. Selected cases (frozen before imaging)</h2>" + table(
        ["source", "records", "amp+noise case", "noise-only case", "antenna", "A", "a", "r = a/SNR₀", "noise seed",
         "eC noise only", "eC amp+noise", "d_amp", "aperture flux ratio, noise only", "aperture flux ratio, amp+noise"], rows))
    cnt = ", ".join(f"{k}: {v}" for k, v in sorted(sel["counts"].items(), key=lambda kv: -kv[1]))
    body.append(f"<p class=cap>Rules: {esc(sel['rules']['mixture'])}; counterpart = {esc(sel['rules']['counterpart'])}; "
                f"d_amp ≤ {sel['rules']['rescue_cutoff']} (illustrative cutoff, not significance); A ≤ {sel['rules']['amp_max']}; "
                f"{esc(sel['rules']['aperture'])}; excluded source(s) {', '.join(sel['rules']['excluded_sources'])}; "
                f"{esc(sel['rules']['order'])}. Candidate counts: {esc(cnt)}. Eligible sources: {sel['n_sources_eligible']}. "
                f"Full audit: <a href='candidates.csv'>candidates.csv</a>."
                + (f" <b>Limitation: {esc(sel['limitation'])}</b>" if sel.get("limitation") else "") + "</p>")
    body.append("<h2>3. What changed and what stayed fixed</h2>" + table(["", "Setting"], [
        ["Changed", "noise only V+n · current G·V+n · alternative G·(V+n), per seed"],
        ["Fixed", "each case's archived flux, flags policy, antenna, exact A and a, reference grid, CLEAN mask, restoring/"
                  "evaluation beam, adaptive CLEAN (= dataset default); visibility weights from nominal σ₀, not rescaled"],
        ["Seeds", "archived noise realization + two deterministic new seeds fixed in the plan (not chosen for rescue)"],
        ["eC", "runner eC over the valid image (as experiment 1); eC on the dataset's fixed support Ω reproduces the archived value"]]))
    body.append("<h2>4. What happened</h2>")
    fig, axes = plt.subplots(len(sids), 2, figsize=(9.5, 3.5 * len(sids)), layout="constrained", squeeze=False,
                             gridspec_kw=dict(width_ratios=[1.3, 1]))
    for r_i, sid in enumerate(sids):
        ax, bx = axes[r_i]
        get = lambda o, s: run.m(f"{sid}-{o}-{s}", "eC")
        for k, (s, lab) in enumerate(seeds):
            ys = [get(o, s) for o in xs]
            pts = [(i, y) for i, y in enumerate(ys) if y is not None]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=INK if s == "archived" else GRID,
                    lw=2 if s == "archived" else 1.5, zorder=1)
            for i, y in pts:
                ax.plot(i, y, ORDER_STYLE[xs[i]][2], color=ORDER_STYLE[xs[i]][1], ms=8, mec="white", mew=1.5, zorder=2)
            if pts:
                ax.annotate(lab, pts[-1], xytext=(6, 0), textcoords="offset points", va="center", fontsize=8,
                            color=INK if s == "archived" else MUTED)
        ax.set_xticks(range(3), [ORDER_STYLE[o][0] for o in xs]); ax.set_xlim(-.3, 2.9)
        ax.set_ylabel("eC (0 exact, 1 = empty model)"); ax.set_title(f"{sid}: error relative to the known source", fontsize=9)
        d = per[sid]
        for o, off in (("current", -.1), ("alternative", .1)):
            for i in range(3):
                if d[o][i] is not None:
                    bx.plot(i + off, d[o][i], ORDER_STYLE[o][2], color=ORDER_STYLE[o][1], ms=8, mec="white", mew=1.5,
                            label=ORDER_STYLE[o][0] if i == 0 else None)
        for i in range(3):
            if d["current"][i] is not None and d["alternative"][i] is not None:
                bx.plot(i + .28, d["alternative"][i] - d["current"][i], "^", color=C3, ms=7, mec="white", mew=1.2,
                        label="Δorder = eC(alt) − eC(cur)" if i == 0 else None)
        bx.axhline(0, color=INK, lw=1)
        bx.set_xticks(range(3), [lab for _, lab in seeds]); bx.set_ylabel("damage (eC difference)")
        bx.set_title("d_amp per order (negative = gain helped); Δorder > 0 = alternative weaker", fontsize=8.5)
        bx.legend(fontsize=7.5, frameon=False)
    body.append(run.save(fig, "orderval_dots.png"))
    body.append("<p class=cap>Here the gain either changes only the source signal or changes both signal and noise. We kept "
                "the weights fixed. Dark line = archived seed (the one selected for its rescue); grey lines = new seeds.</p>")
    # flux
    fig, axes = plt.subplots(1, len(sids), figsize=(5.2 * len(sids), 3.3), layout="constrained", squeeze=False)
    for ax, sid in zip(axes[0], sids):
        for i, o in enumerate(xs):
            for k, (s, lab) in enumerate(seeds):
                ap = run.m(f"{sid}-{o}-{s}", "flux_ratio_aperture")
                out = run.m(f"{sid}-{o}-{s}", "outside_aperture_abs_flux_ratio")
                if ap is not None:
                    ax.plot(i + (k - 1) * .12, ap, ORDER_STYLE[o][2], color=ORDER_STYLE[o][1], ms=8, mec="white", mew=1.5)
                if out is not None:
                    ax.plot(i + (k - 1) * .12, out, ORDER_STYLE[o][2], color=ORDER_STYLE[o][1], ms=6, mfc="none", mew=1.2)
        ax.axhline(1, color=INK, lw=1); ax.text(2.4, 1, " true flux", va="center", fontsize=8, color=MUTED)
        ax.set_xticks(range(3), [ORDER_STYLE[o][0] for o in xs]); ax.set_xlim(-.4, 2.9)
        ax.set_ylabel("÷ true flux"); ax.set_title(f"{sid}: filled = model flux in source aperture; open = |model| outside", fontsize=8.5)
    body.append(run.save(fig, "orderval_flux.png"))
    body.append("<p class=cap>Three seeds per condition, left to right (archived first). Aperture = restoring-beam FWHM "
                "ellipse. Total signed flux alone cannot separate source recovery from cancelling false components.</p>")
    half = 4.
    for sid in sids:
        cols = [f"{sid}-ctrl-V", f"{sid}-noise-archived", f"{sid}-current-archived", f"{sid}-alternative-archived"]
        if all(run.ok(c) for c in cols):
            titles = ["noiseless V", "noise only", "current G·V+n", "alternative G·(V+n)"]
            body.append(image_grid(run, [("model − truth", lambda p: p["C"] - p["T"], True)], cols,
                                   [f"{t}\neC={run.m(c, 'eC'):.3f}" for t, c in zip(titles, cols)], half,
                                   f"orderval_images_{sid}.png", suptitle=f"{sid}, archived seed"))
    body.append(f"<p class=cap>Raw CLEAN model convolved with the evaluation beam minus truth, ÷ true flux; shared diverging "
                f"scale within each case, ±{half:g}″ field, ellipse = restoring beam.</p>")
    # per-case conclusions
    rows = []
    for sid, verdict, d in summary:
        for i, (s, lab) in enumerate(seeds):
            rows.append([sid, lab, d["current"][i], d["alternative"][i],
                         None if d["current"][i] is None or d["alternative"][i] is None else d["alternative"][i] - d["current"][i],
                         verdict if i == 0 else ""])
    body.append("<h2>5. Per-case result</h2>" + table(["source", "seed", "d_amp current", "d_amp alternative", "Δorder", "case verdict"], rows))
    new = [(d["current"][i], d["alternative"][i]) for _, _, d in summary for i in (1, 2)
           if d["current"][i] is not None and d["alternative"][i] is not None]
    if new:
        body.append(f"<p>New seeds only ({len(new)} realizations): current-order rescue (d_amp &lt; 0) in "
                    f"{sum(c < 0 for c, _ in new)}/{len(new)}, alternative-order rescue in {sum(a < 0 for _, a in new)}/{len(new)}, "
                    f"alternative weaker in {sum(a > c for c, a in new)}/{len(new)}. Archived seeds were selected for a "
                    "negative current-order damage, so they are summarized separately above.</p>")
    body.append("<p>These cases were selected for a strong rescue; they do not estimate how common rescue is, and the "
                "seeds of one source are not independent sources.</p>")
    # reproduction + checks
    rep = []
    for sid in sids:
        arch = run.plan["archived"][sid]
        prep = {}
        for p in sorted((run.root / "runs").glob("order*.json")):
            prep = (json.loads(p.read_text()).get("source_preparations") or {}).get(sid) or prep
        seed0 = str(arch["noise_seed"])
        for o, k in (("noise", "noise"), ("current", "current")):
            mine = run.m(f"{sid}-{o}-archived", "eC_matched")
            rep.append([sid, f"{o} eC on Ω: runner vs archived", f"{fmt(mine, 12)} vs {fmt(arch['eC'][k], 12)}",
                        fmt(None if mine is None else abs(mine - arch["eC"][k]))])
        rep.append([sid, "noiseless V DATA hash equal", fmt(prep.get("v_sha256") == arch["v_sha256"] if prep else None), ""])
        rep.append([sid, "archived noise realization hash equal",
                    fmt((prep.get("regular", {}).get(seed0) or {}).get("noise_sha256") == arch["nominal_noise_sha256"] if prep else None), ""])
    body.append("<h3>Reproduction of the archived examples</h3>" + table(["source", "check", "result", "|difference|"], rep))
    chk = []
    for sid in sids:
        for s, lab in seeds:
            oc = run.records[f"{sid}-current-{s}"].get("order_check") or {}
            cur, alt = run.records[f"{sid}-current-{s}"].get("data", {}), run.records[f"{sid}-alternative-{s}"].get("data", {})
            chk.append([sid, lab, fmt(oc.get("unaffected_rows_identical")), oc.get("max_difference_error_over_storage_tolerance"),
                        fmt(cur.get("unaffected_sha256") == alt.get("unaffected_sha256") and cur.get("unaffected_sha256") is not None)])
        gv, gv0 = run.records[f"{sid}-ctrl-GV"], run.records[f"{sid}-ctrl-GV0"]
        chk.append([sid, "zero noise", "G·(V+0) DATA = G·V DATA", fmt(gv.get("data", {}).get("data_sha256") == gv0.get("data", {}).get("data_sha256")
                                                                       and gv.get("data", {}).get("data_sha256") is not None), gv0.get("status")])
    body.append("<h3>Visibility checks before imaging</h3>" + table(
        ["source", "seed", "unaffected rows identical", "(alt−cur) − (G−1)n ÷ storage tol.", "stored unaffected DATA equal"], chk))
    body.append("<p class=cap>The original 0449+113 result (experiment 1) was run on the local machine "
                "(runs/tanmoy-local-20261123) and is not copied here; compare with its report.</p>")
    body.append(tech(run, "order"))
    (run.root / "report.html").write_text(page("Experiment 1b · Does the gain/noise order result repeat?", "".join(body), back=False))
    check_links(run.root)
    print("REPORT", run.root / "report.html")
    print("[order-validation]", lead)
    return lead


def build(root):
    run = Run(root)
    if run.plan.get("study") == "uvgrid":
        csv_rows(run)
        return uvgrid_report(run)
    if run.plan.get("study") == "order-validation":
        csv_rows(run)
        return orderval_report(run)
    rows = csv_rows(run)
    summaries = dict(order=order_page(run), stopping=stopping_page(run), sampling=sampling_page(run), selfcal=selfcal_page(run))
    names = dict(order="1 · Gain/noise application order", stopping="2 · CLEAN stopping and noise",
                 sampling="3 · Noiseless image sampling", selfcal="4 · Existing self-calibration decisions")
    status = []
    for exp in ("order", "stopping", "sampling"):
        counts, secs = status_line(run, exp)
        status.append([exp, sum(counts.values()), ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())), secs])
    worker = [json.loads(p.read_text()) for p in sorted((run.root / "runs").glob("*.json"))]
    total = sum(w.get("seconds") or 0 for w in worker)
    peak = max([w.get("peak_scratch_bytes") or 0 for w in worker] + [0])
    body = ["<p class=lead>Four small, controlled local checks of questions raised by Tanmoy. eC = ‖C − T‖₂ / ‖T‖₂, where C is "
            "the raw CLEAN model convolved with the restoring beam and T the known source convolved identically: 0 is exact "
            "model recovery, 1 is the error of an empty model, and values can exceed 1.</p>"]
    body.append("<ul>" + "".join(f"<li><a href='{k}.html'><b>{esc(names[k])}</b></a> — {esc(v)}</li>" for k, v in summaries.items()) + "</ul>")
    body.append("<h2>Runs</h2>" + table(["experiment", "planned images", "status", "case seconds (sum)"], status) +
                f"<p class=cap>{len(run.plan['cases'])} planned imaging cases (budget 27); worker wall time {total:.0f} s across "
                f"{len(worker)} CASA worker runs (including source preparation); peak scratch {peak / 2**30:.2f} GiB; one CASA "
                f"worker, serial. Data: <a href='measurements.csv'>measurements.csv</a> ({len(rows)} rows, one per planned case).</p>")
    body.append(f"<details><summary>Plan provenance</summary><pre>{esc(json.dumps({k: v for k, v in run.plan.items() if k != 'cases'}, indent=1, default=str)[:60000])}</pre></details>")
    (run.root / "report.html").write_text(page("Tanmoy questions · small local experiments", "".join(body), back=False))
    check_links(run.root)
    print("REPORT", run.root / "report.html")
    for k, v in summaries.items():
        print(f"[{k}] {v}")


def check_links(root):
    import re
    missing = []
    for p in root.glob("*.html"):
        for ref in re.findall(r"(?:src|href)='([^'#:]+)'", p.read_text()):
            if not (root / ref).exists():
                missing.append((p.name, ref))
    if missing:
        raise ValueError(f"Unresolved local assets: {missing[:10]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    build(ap.parse_args().input)


if __name__ == "__main__":
    main()
