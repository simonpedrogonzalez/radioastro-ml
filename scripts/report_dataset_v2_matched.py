"""Matched Dataset v2 reports: sampling preview (no CASA) and dataset/smoke report.

  --preview --output DIR [--config JSON] [--smoke]   recipes for every source of a split
  --dataset RUN [--compare OTHER_RUN]                 labels, features, resources, determinism
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from scripts import create_dataset_v1 as g
from scripts.create_dataset_v2 import indexed_sources, original_flux_spec
from scripts.create_dataset_v2_matched import configuration
from scripts.dataset_v2 import TYPES
from scripts.dataset_v2_matched import BRANCHES, sample_recipes
from scripts.preprocessing.schema import atomic_write_json

# Validated categorical order (dataviz reference palette, light mode); masks 1-7 in fixed order.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
CONTROL = "#8a8a85"
MASK_COLORS = (CONTROL, *SERIES)
MASK_LABELS = ("control", "amp", "phase", "amp+phase", "noise", "amp+noise", "phase+noise", "all three")
BAND_COLORS = {"strong": SERIES[0], "weak": SERIES[1]}
STYLE = ("<style>body{font:15px system-ui;max-width:1200px;margin:auto;padding:1.5rem;color:#24323a;background:#fcfcfb}"
         "img{max-width:100%}table{border-collapse:collapse;margin:.5rem 0}td,th{padding:.35rem .7rem;border:1px solid #ccd;"
         "text-align:right}td:first-child,th:first-child{text-align:left}th{background:#eef3f6}.note{color:#4a5a63}</style>")


def _tidy(ax, log_axes=""):
    for name in log_axes:
        axis = ax.xaxis if name == "x" else ax.yaxis
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(color="#d9dee1", lw=.5, alpha=.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _table(header, rows):
    return ("<table><tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in header) + "</tr>" +
            "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows) + "</table>")


# ---------------------------------------------------------------- sampling preview

def preview(config, output, smoke=False):
    output = output.resolve()
    assets = output / "sampling_preview_assets"
    assets.mkdir(parents=True, exist_ok=True)
    ids, _, audit = indexed_sources(config, smoke)
    manifest = json.loads((g.THERMAL_DIR / "report.json").read_text())
    snr = {sid: original_flux_spec(g._thermal_source(g.THERMAL_DIR, manifest, sid))["nominal_source_snr"] for sid in ids}
    # Antenna eligibility needs CASA preparation; parameters do not depend on it.
    recipes = [dict(r, antenna=None) for sid in ids for r in sample_recipes(sid, snr[sid], [0], config)]
    R = int(config["recipes_per_source"])
    per_source = 7 * R * int(config["noise_slots"]) + int(config["noise_slots"])
    atomic_write_json(output / "sampling_preview.json", dict(configuration=config, source_index=audit,
                                                             nominal_source_snr=snr, recipes=recipes))
    att = [r for r in recipes if r["amp_branch"] == "attenuation"]
    amp = [r for r in recipes if r["amp_branch"] == "amplification"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    specs = [(axes[0, 0], "Phase (deg)", recipes, "phi_deg", "phase_band", False,
              [*config["ranges"]["phase_negative"], *config["ranges"]["phase_positive"]]),
             (axes[0, 1], "Attenuation A (linear)", att, "A", "amp_band", False, config["ranges"]["amp_attenuation"]),
             (axes[1, 0], "Amplification A (log)", amp, "A", "amp_band", True, config["ranges"]["amp_amplification"]),
             (axes[1, 1], "Thermal ratio r (log)", recipes, "r", None, True, config["ranges"]["noise_r"])]
    for ax, label, items, key, band, logarithmic, bounds in specs:
        values = np.array([r[key] for r in items])
        bins = np.geomspace(values.min(), values.max(), 31) if logarithmic else 30
        groups = ("strong", "weak") if band else ("strong",)
        data = [values[[r[band] == b for r in items]] if band else values for b in groups]
        ax.hist(data, bins=bins, stacked=True, color=[BAND_COLORS[b] for b in groups], label=list(groups),
                edgecolor="#fcfcfb", linewidth=.6)
        for bound in bounds:
            ax.axvline(bound, color="#4a5a63", ls=":", lw=.8)
        if logarithmic:
            ax.set_xscale("log")
        ax.set(xlabel=label, ylabel="Recipes")
        _tidy(ax, "x" if logarithmic else "")
    axes[0, 0].legend(title="Severity band", frameon=False)
    fig.tight_layout(); fig.savefig(assets / "marginals.png", dpi=150); plt.close(fig)
    combos = [("strong", "strong"), ("strong", "weak"), ("weak", "strong"), ("weak", "weak")]
    combo_colors = dict(zip(combos, SERIES[:4]))
    for branch, items in (("attenuation", att), ("amplification", amp)):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
        pairs = [("phi_deg", "A", "Phase (deg)", "A"), ("A", "r", "A", "r"), ("phi_deg", "r", "Phase (deg)", "r")]
        for ax, (xk, yk, xl, yl) in zip(axes, pairs):
            for combo in combos:
                pts = [r for r in items if (r["amp_band"], r["phase_band"]) == combo]
                ax.scatter([r[xk] for r in pts], [r[yk] for r in pts], s=14, alpha=.75, color=combo_colors[combo],
                           edgecolors="#fcfcfb", linewidths=.4, label=f"amp {combo[0]} / phase {combo[1]}")
            logs = ""
            for key, setter, axis in ((xk, ax.set_xscale, "x"), (yk, ax.set_yscale, "y")):
                if key == "r" or (key == "A" and branch == "amplification"):
                    setter("log"); logs += axis
            ax.set(xlabel=xl, ylabel=yl)
            _tidy(ax, logs)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, .9)); fig.savefig(assets / f"pairs_{branch}.png", dpi=150); plt.close(fig)
    example = [r for r in recipes if r["source"] == ids[0] and r["group"] == "attenuation:negative"]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter([r["unit_cube"]["amp"] for r in example], [r["unit_cube"]["phase"] for r in example], s=40,
               color=SERIES[0])
    for edge in np.linspace(0, 1, len(example) + 1):
        ax.axvline(edge, color="#d9dee1", lw=.6); ax.axhline(edge, color="#d9dee1", lw=.6)
    ax.axvline(config["weak_fraction"], color=SERIES[1], ls="--", lw=1)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="amp coordinate", ylabel="phase coordinate",
           title=f"{ids[0]} attenuation/negative LHS")
    fig.tight_layout(); fig.savefig(assets / "unit_cube.png", dpi=150); plt.close(fig)
    groups = [[f"{a} / {p}", sum(r["group"] == f"{a}:{p}" for r in recipes),
               sum(r["group"] == f"{a}:{p}" and r["amp_band"] == "weak" for r in recipes),
               sum(r["group"] == f"{a}:{p}" and r["phase_band"] == "weak" for r in recipes)] for a, p in BRANCHES]
    masks = [[MASK_LABELS[m], R * len(ids) * config["noise_slots"] if m else len(ids) * config["noise_slots"]] for m in range(8)]
    parts = ["<!doctype html><meta charset='utf-8'><title>Matched v2 sampling preview</title>", STYLE,
             "<h1>Dataset v2 matched sampling preview</h1>",
             f"<p>{len(ids)} {config['split']} sources, R = {R} recipes per source, {config['noise_slots']} noise slot(s): "
             f"<strong>{per_source} cases = {per_source} rows per source</strong> (7R + 1 control), "
             f"{per_source * len(ids):,} in total. Every row has all three damage labels observed. "
             f"Each recipe (A, phase, r, antenna) is reused by all seven corrupted presence masks.</p>",
             "<p><a href='sampling_preview.json'>Recipes and provenance (JSON)</a></p>",
             "<h2>Rows per presence mask</h2>", _table(["Mask", "Rows"], masks),
             "<h2>Strata</h2><p>Recipes are stratified equally over amplitude branch × phase sign, with a Latin hypercube in "
             f"(amp, phase, noise) inside each stratum. A fraction w = {config['weak_fraction']} of each amp/phase coordinate maps "
             "to the weak band next to no-corruption; the rest maps to the strong range.</p>",
             _table(["Branch", "Recipes", "Weak amp", "Weak phase"], groups),
             _table(["Parameter", "Strong range", "Weak band"],
                    [[k, config["ranges"][k], config["weak_ranges"].get(k, "—")] for k in config["ranges"]]),
             "<h2>Marginals</h2><p class='note'>Stacked by severity band. Dotted lines mark the strong-range bounds. "
             "r is log-uniform from max(r_min, 1/SNR0) for each source.</p><img src='sampling_preview_assets/marginals.png'>",
             "<h2>Pairwise distributions</h2><p class='note'>One point per recipe; colour is the amp/phase severity-band "
             "combination, identical in every panel.</p>",
             "<img src='sampling_preview_assets/pairs_attenuation.png'><img src='sampling_preview_assets/pairs_amplification.png'>",
             "<h2>One LHS stratum in unit coordinates</h2><p class='note'>Grid lines are the LHS strata; the dashed line is the weak-band "
             "boundary in the amp coordinate.</p><img src='sampling_preview_assets/unit_cube.png'>"]
    (output / "sampling_preview.html").write_text("".join(parts))
    print("Sampling preview:", output / "sampling_preview.html", flush=True)


# ---------------------------------------------------------------- dataset report

def _load(run):
    plan = json.loads((run / "plan.json").read_text())
    with np.load(run / f"{plan['split']}.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    return plan, data


def compare_runs(run, other):
    """Scientific outputs of identical case IDs in two independently sharded runs."""
    a = {p.parent.name: json.loads(p.read_text()) for p in (run / "cases").glob("*/case.json")}
    b = {p.parent.name: json.loads(p.read_text()) for p in (other / "cases").glob("*/case.json")}
    common = sorted(set(a) & set(b))
    diff = dict(eC=0., eD=0., X=0.)
    identical = 0
    for cid in common:
        da = abs(a[cid]["eC"] - b[cid]["eC"]), abs(a[cid]["eD"] - b[cid]["eD"])
        dx = float(np.nanmax(np.abs(np.subtract(a[cid]["X"], b[cid]["X"]))))
        diff = dict(eC=max(diff["eC"], da[0]), eD=max(diff["eD"], da[1]), X=max(diff["X"], dx))
        identical += a[cid]["eC"] == b[cid]["eC"] and a[cid]["eD"] == b[cid]["eD"] and a[cid]["X"] == b[cid]["X"]
    hashes = {(json.loads(p.read_text())["v_sha256"], json.loads(p.read_text())["nominal_noise_sha256"])
              for d in (run, other) for p in (d / "units").glob("*.json") if not p.name.endswith(".seal.json")}
    return dict(other=str(other), common_cases=len(common), only_here=len(set(a) - set(b)),
                only_other=len(set(b) - set(a)), bit_identical=identical, max_abs_diff=diff,
                distinct_v_noise_hash_pairs=len(hashes))


def report(run, compare=None):
    run = run.resolve()
    plan, data = _load(run)
    status = json.loads((run / "status.json").read_text())
    assets = run / "report_assets"
    assets.mkdir(exist_ok=True)
    names = [str(x) for x in data["feature_names"]]
    b, d, p, mask, params = data["b"], data["d"], data["p"], data["mask"], data["params"]
    n = len(mask)
    # Absent types must be exact zeros; every active label must be finite.
    absent_ok = bool(np.all(d[b == 0] == 0) and np.all(p[b == 0] == 0))
    active_ok = bool(np.isfinite(d[b == 1]).all() and np.isfinite(p[b == 1]).all())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, label in ((axes[0], "eC", "eC (CLEAN recovery error)"), (axes[1], "eD", "eD (dirty error)")):
        ax.boxplot([data[key][mask == m] for m in range(8)], tick_labels=MASK_LABELS, widths=.55,
                   medianprops=dict(color="#24323a"), flierprops=dict(markersize=3))
        for m in range(8):
            ax.scatter(np.full((mask == m).sum(), m + 1), data[key][mask == m], s=10, color=MASK_COLORS[m], alpha=.8, zorder=3)
        ax.set(ylabel=label); ax.tick_params(axis="x", rotation=30)
        _tidy(ax)
    fig.tight_layout(); fig.savefig(assets / "errors_by_mask.png", dpi=150); plt.close(fig)
    xinfo = [("A (log)", 0, True), ("Phase (deg)", 1, False), ("Noise multiplier a (log)", 2, True)]
    for target, values, label in (("d", d, "Signed CLEAN damage d_k"), ("p", p, "Dirty change p_k")):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
        for k, (xl, col, logarithmic) in enumerate(xinfo):
            ax = axes[k]
            for m in range(1, 8):
                sel = (mask == m) & (b[:, k] == 1)
                if sel.any():
                    ax.scatter(params[sel, col], values[sel, k], s=16, color=MASK_COLORS[m], alpha=.85,
                               edgecolors="#fcfcfb", linewidths=.4, label=MASK_LABELS[m])
            ax.axhline(0, color="#4a5a63", lw=.6)
            if logarithmic:
                ax.set_xscale("log")
            ax.set(title=TYPES[k], xlabel=xl)
            _tidy(ax, "x" if logarithmic else "")
        axes[0].set_ylabel(label + " (active rows)")
        h = {lab: han for ax in axes for han, lab in zip(*ax.get_legend_handles_labels())}
        fig.legend([h[x] for x in MASK_LABELS if x in h], [x for x in MASK_LABELS if x in h], loc="upper center", ncol=7, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, .9)); fig.savefig(assets / f"{target}_by_parameter.png", dpi=150); plt.close(fig)
    feature_rows = []
    for j, name in enumerate(names):
        col = data["X"][:, j]
        finite = np.isfinite(col)
        feature_rows.append([name, f"{np.min(col[finite]):.4g}", f"{np.median(col[finite]):.4g}",
                             f"{np.max(col[finite]):.4g}", int((~finite).sum())])
    res = status.get("resources", {})
    retained = np.asarray(res.get("retained_bytes_per_case", [0]), float) / 2**20
    seconds = np.asarray(res.get("unit_seconds", [0]), float)
    counts = [[MASK_LABELS[m], int((mask == m).sum())] for m in range(8)]
    stats = [[TYPES[k], int(b[:, k].sum()), f"{d[b[:, k] == 1, k].min():.4f}", f"{np.median(d[b[:, k] == 1, k]):.4f}",
              f"{d[b[:, k] == 1, k].max():.4f}", int((d[b[:, k] == 1, k] < 0).sum()), f"{np.median(p[b[:, k] == 1, k]):.4f}"]
             for k in range(3)]
    parts = ["<!doctype html><meta charset='utf-8'><title>Matched v2 dataset report</title>", STYLE,
             f"<h1>Dataset v2 matched: {html.escape(plan['split'])}{' smoke' if plan['smoke'] else ''}</h1>",
             f"<p>{len(set(data['source_ids']))} source(s), {plan['counts']['units']} sampling-template units, "
             f"{n} rows = {plan['counts']['cases']} cases, R = {plan['configuration']['recipes_per_source']}. "
             f"Labels: absent types exactly zero <strong>{'PASS' if absent_ok else 'FAIL'}</strong>; active labels finite "
             f"<strong>{'PASS' if active_ok else 'FAIL'}</strong>; noiseless V and nominal noise identical across all units "
             "<strong>PASS</strong> (enforced by verification).</p>",
             "<h2>Rows per presence mask</h2>", _table(["Mask", "Rows"], counts),
             "<h2>Errors by mask</h2><img src='report_assets/errors_by_mask.png'>",
             "<h2>Damage labels (active rows)</h2>",
             _table(["Type", "Active rows", "min d", "median d", "max d", "negative d", "median p"], stats),
             "<img src='report_assets/d_by_parameter.png'><img src='report_assets/p_by_parameter.png'>",
             "<p class='note'>d_k = eC(case) − eC(case without k). p_k = ‖D(case) − D(case without k)‖ / ‖D0‖. "
             "Colour is the presence mask, identical in every panel. Dirty-change points of different masks largely overlap: "
             "the dirty image is linear in the visibilities, so removing one corruption changes it almost independently of the "
             "others (exactly so for noise, which is added after the gains with the same realization).</p>",
             "<h2>Features (20)</h2>", _table(["Feature", "min", "median", "max", "nonfinite"], feature_rows),
             "<h2>Resources per unit</h2>",
             _table(["Measure", "Value"], [
                 ["Unit wall time (s): median / max", f"{np.median(seconds):.0f} / {seconds.max():.0f}"],
                 ["Peak scratch per unit (MB)", f"{res.get('peak_scratch_bytes', 0) / 2**20:.0f}"],
                 ["Peak RSS of a unit process (MB)", f"{res.get('peak_rss_bytes', 0) / 2**20:.0f}"],
                 ["Retained per case (MB): median / max", f"{np.median(retained):.2f} / {retained.max():.2f}"],
                 ["Retained in total (MB)", f"{retained.sum():.0f}"]])]
    if compare is not None:
        c = compare_runs(run, compare.resolve())
        atomic_write_json(run / "determinism.json", c)
        parts += ["<h2>Determinism across sharding</h2>",
                  f"<p>Compared with <code>{html.escape(c['other'])}</code> (different --num-shards/--parallel).</p>",
                  _table(["Measure", "Value"], [["Common cases", c["common_cases"]], ["Only here / only other", f"{c['only_here']} / {c['only_other']}"],
                                                 ["Bit-identical eC, eD and X", c["bit_identical"]],
                                                 ["Max |Δ| eC / eD / X", f"{c['max_abs_diff']['eC']:.3g} / {c['max_abs_diff']['eD']:.3g} / {c['max_abs_diff']['X']:.3g}"],
                                                 ["Distinct (V, noise) hash pairs over both runs", c["distinct_v_noise_hash_pairs"]]])]
    (run / "report.html").write_text("".join(parts))
    print("Report:", run / "report.html", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preview", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dataset", type=Path)
    p.add_argument("--compare", type=Path)
    a = p.parse_args()
    if a.preview:
        preview(configuration(a.config), a.output, a.smoke)
    else:
        report(a.dataset, a.compare)


if __name__ == "__main__":
    main()
