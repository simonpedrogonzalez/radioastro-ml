"""Compare one model family trained on different datasets, on the same validation rows, as a function of true eC.

Read-only: uses each multitask run's saved predictions_val_<model>.npz (decoded column 3 = eC) and the
validation <split>.npz (truth). Every run must have been scored on exactly the same validation rows.

Per run: true vs predicted eC (linear and log), and, in one figure aligned under the eC distributions
(validation and each run's training set) on a shared log eC axis, per eC bin (equal-count bins of the validation eC,
shared by all runs) the within-bin R^2, RMSE and relative RMSE (RMSE / mean true eC), each with a 95%
source-bootstrap interval (validation sources resampled with replacement). The dashed line on each panel
is the median of that run's per-bin values. A table gives the paired difference to the reference run per
bin, with its own paired bootstrap interval, so "better at large/small eC" can be told apart from noise.

  bin/submit ml-cpu python -m ml.compare_val_by_ec --output ml/runs/compare-xgb_mlp-v2-v21-by-eC
  # defaults: --model xgb_mlp --val runs/dataset-v2-matched-val
  #   --run v2=ml/runs/matched-v2-full-all22-dirty --run v2.1=ml/runs/matched-v2.1-all22-dirty
  #   --run v2.1-nocap=ml/runs/matched-v2.1-noampcap-all22-dirty
  #   --run v2.2=ml/runs/matched-v2.2-all22-dirty   (first run is the reference)
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

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = (("v2", "ml/runs/matched-v2-full-all22-dirty"),
                ("v2.1", "ml/runs/matched-v2.1-all22-dirty"),
                ("v2.1-nocap", "ml/runs/matched-v2.1-noampcap-all22-dirty"),
                ("v2.2", "ml/runs/matched-v2.2-all22-dirty"))
DEFAULT_TRAINS = (("v2", "runs/dataset-v2-matched-train"),
                  ("v2.1", "runs/dataset-v2.1-train/manifest.json"),
                  ("v2.1-nocap", "runs/dataset-v2.1-noampcap-train/manifest.json"),
                  ("v2.2", "runs/dataset-v2.2-train/manifest.json"))
EC_COLUMN = 3   # decoded multitask output order: d_amp, d_phase, d_noise, eC, ...
# Validated categorical order (dataviz reference palette, light mode), as in the dataset reports.
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
STYLE = ("<style>body{font:15px system-ui;max-width:1250px;margin:auto;padding:1.5rem;color:#24323a;background:#fcfcfb}"
         "img{max-width:100%}table{border-collapse:collapse;margin:.5rem 0 1.2rem}td,th{padding:.3rem .6rem;"
         "border:1px solid #ccd;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#eef3f6}"
         ".note{color:#4a5a63}code{font-size:90%}.better{color:#0a7a3e;font-weight:600}.worse{color:#b3261e;font-weight:600}</style>")


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load(val_root, runs, model):
    """Truth and each run's eC predictions, aligned on the validation row ids."""
    val_root = _resolve(val_root)
    npz = val_root / "val.npz" if val_root.is_dir() else val_root
    with np.load(npz, allow_pickle=False) as z:
        truth = dict(eC=np.asarray(z["eC"], float), row_ids=np.asarray(z["row_ids"]).astype(str),
                     source_ids=np.asarray(z["source_ids"]).astype(str), mask=np.asarray(z["mask"], int))
    pred = {}
    for name, run in runs:
        run = _resolve(run)
        with np.load(run / f"predictions_val_{model}.npz", allow_pickle=False) as z:
            ids, decoded = np.asarray(z["row_ids"]).astype(str), np.asarray(z["decoded"], float)
        if not np.array_equal(ids, truth["row_ids"]):
            raise ValueError(f"{run}: validation rows differ from {npz} (was it scored on another val set?)")
        pred[name] = decoded[:, EC_COLUMN]
        summary = json.loads((run / "results.json").read_text())
        pred[name + "__train_rows"] = summary.get("train_rows")
    return truth, pred


def load_train(trains, pred):
    """Training eC of each run (run folder or v2.1 manifest, filters applied); missing paths are skipped."""
    from scripts.dataset_v21 import load_manifest, read_part
    out = {}
    for name, path in trains:
        path = _resolve(path)
        if not path.exists():
            print(f"Training set for {name} not found ({path}); its distribution is omitted", flush=True)
            continue
        eC = np.asarray(load_manifest(path)["eC"] if path.suffix == ".json" else read_part(path, "train")[0]["eC"], float)
        expected = pred.get(name + "__train_rows")
        if expected is not None and expected != len(eC):
            raise ValueError(f"{name}: {path} has {len(eC)} rows but the run was trained on {expected}")
        out[name] = eC
    return out


def equal_count_edges(y, bins):
    edges = np.quantile(y, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    return np.unique(edges)


def weighted_metrics(y, p, w):
    """Within-bin R^2, RMSE and RMSE / mean(y) for row weights w (shape boots x rows)."""
    w = np.atleast_2d(w)
    sw = w.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sse = (w * (y - p) ** 2).sum(1)
        mean = (w * y).sum(1) / sw
        sst = (w * (y - mean[:, None]) ** 2).sum(1)
        rmse = np.sqrt(sse / sw)
        return dict(r2=1 - sse / sst, rmse=rmse, rel_rmse=rmse / mean)


def bootstrap_weights(sources, boots, seed):
    """Row weights = how many times each row's source was drawn (sources resampled with replacement)."""
    names, index = np.unique(sources, return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(names), np.full(len(names), 1 / len(names)), size=boots)
    return counts[:, index].astype(float)


def analyze(truth, pred, names, edges, boots, seed):
    y = truth["eC"]
    which = np.digitize(y, edges[1:-1])
    W = bootstrap_weights(truth["source_ids"], boots, seed)
    ref = names[0]
    bins, overall = [], {}
    for name in names:
        point = weighted_metrics(y, pred[name], np.ones(len(y)))
        boot = weighted_metrics(y, pred[name], W)
        overall[name] = {k: dict(value=float(point[k][0]), lo=float(np.nanpercentile(boot[k], 2.5)),
                                 hi=float(np.nanpercentile(boot[k], 97.5))) for k in point}
    for b in range(len(edges) - 1):
        sel = which == b
        entry = dict(lo=float(y[sel].min()), hi=float(y[sel].max()), median_eC=float(np.median(y[sel])),
                     sd_eC=float(np.std(y[sel])),
                     n=int(sel.sum()), sources=int(len(set(truth["source_ids"][sel]))), runs={})
        ref_boot = weighted_metrics(y[sel], pred[ref][sel], W[:, sel])
        for name in names:
            point = weighted_metrics(y[sel], pred[name][sel], np.ones(int(sel.sum())))
            boot = weighted_metrics(y[sel], pred[name][sel], W[:, sel])
            entry["runs"][name] = {}
            for k in point:
                diff = boot[k] - ref_boot[k]
                entry["runs"][name][k] = dict(
                    value=float(point[k][0]), lo=float(np.nanpercentile(boot[k], 2.5)),
                    hi=float(np.nanpercentile(boot[k], 97.5)),
                    diff=float(point[k][0] - weighted_metrics(y[sel], pred[ref][sel], np.ones(int(sel.sum())))[k][0]),
                    diff_lo=float(np.nanpercentile(diff, 2.5)), diff_hi=float(np.nanpercentile(diff, 97.5)))
            entry["runs"][name]["median_signed_error"] = float(np.median(pred[name][sel] - y[sel]))
        bins.append(entry)
    return bins, overall


# ---------------------------------------------------------------- plots

def mark(ax):
    """Dotted reference lines at the eC thresholds 0.05 and 1."""
    for v in (.05, 1.):
        ax.axvline(v, color="#24323a", ls=":", lw=1, zorder=1)
        ax.text(v, 1.0, f" {v:g}", transform=ax.get_xaxis_transform(), fontsize=7, color="#24323a", va="top", ha="left")


def _tidy(ax, log_x=False):
    if log_x:
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(color="#d9dee1", lw=.5, alpha=.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_diagonal(truth, pred, names, overall, assets):
    y = truth["eC"]
    fig, axes = plt.subplots(2, len(names), figsize=(4.6 * len(names), 9), squeeze=False)
    all_p = np.concatenate([pred[n] for n in names])
    lo, hi = min(y.min(), all_p.min()), max(y.max(), all_p.max())
    floor = max(1e-3, np.quantile(y[y > 0], .001))
    for c, name in enumerate(names):
        p = pred[name]
        for row, log in ((0, False), (1, True)):
            ax = axes[row, c]
            px = np.clip(p, floor, None) if log else p
            ax.scatter(y, px, s=6, alpha=.35, color=COLORS[c], linewidths=0)
            # running median of the prediction in equal-count bins of the truth
            e = np.quantile(y, np.linspace(0, 1, 21))
            k = np.clip(np.digitize(y, e[1:-1]), 0, 19)
            mx = [np.median(y[k == i]) for i in range(20) if (k == i).any()]
            my = [np.median(px[k == i]) for i in range(20) if (k == i).any()]
            ax.plot(mx, my, color="#24323a", lw=1.6, label="median prediction per 5% bin")
            if log:
                ax.plot([floor, hi], [floor, hi], color="#8a8a85", lw=.9, ls="--", label="y = x")
                ax.set(xscale="log", yscale="log", xlim=(floor, hi * 1.1), ylim=(floor, hi * 1.1))
            else:
                ax.plot([lo, hi], [lo, hi], color="#8a8a85", lw=.9, ls="--", label="y = x")
                ax.set(xlim=(lo, hi), ylim=(lo, hi))
                mark(ax)
            ax.set(xlabel="true eC", ylabel="predicted eC" + (" (log; ≤ floor clipped)" if log else ""))
            if row == 0:
                o = overall[name]
                ax.set_title(f"{name}\nR² {o['r2']['value']:.3f} · RMSE {o['rmse']['value']:.3f}", fontsize=11)
            _tidy(ax, log); ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(assets / "diagonal.png", dpi=110); plt.close(fig)
    return "assets/diagonal.png"


def plot_aligned(truth, train, bins, names, assets):
    """eC distributions on top, then within-bin R^2, RMSE and relative RMSE, all on one shared log eC axis.

    Shaded bands and dotted lines are the metric bins, drawn through every panel, so a point can be read
    against how many val rows its bin holds, how wide the bin is and how much training data sits there.
    """
    y = truth["eC"]
    pool = np.concatenate([y] + list(train.values()))
    lo_x, hi_x = min(0., float(pool.min())), float(pool.max()) * 1.03
    hist_bins = np.linspace(lo_x, hi_x, 151)
    fig, axes = plt.subplots(4, 1, figsize=(15, 15), sharex=True, gridspec_kw=dict(height_ratios=[1.4, 1, 1, 1]))
    edges = [b["lo"] for b in bins] + [bins[-1]["hi"]]
    inner = [(a["hi"] + b["lo"]) / 2 for a, b in zip(bins[:-1], bins[1:])]   # boundary between adjacent bins
    spans = list(zip([lo_x] + inner, inner + [hi_x]))
    for ax in axes:
        for i, (a, b) in enumerate(spans):
            if i % 2:
                ax.axvspan(a, b, color="#eef3f6", zorder=0, lw=0)
        for x in inner:
            ax.axvline(x, color="#b9c4ca", ls="-", lw=.7, zorder=1)   # bin edges: solid, lighter than the 0.05/1 marks
    ax = axes[0]
    w = np.full(len(y), 1 / len(y))
    ax.hist(y, bins=hist_bins, weights=w, color="#8a8a85", alpha=.45,
            label=f"validation (n={len(y)})")
    for c, name in enumerate(names):
        if name in train:
            t = train[name]
            ax.hist(t, bins=hist_bins, weights=np.full(len(t), 1 / len(t)), histtype="step",
                    lw=1.7, color=COLORS[c], label=f"train {name} (n={len(t)})")
    ax.set_ylim(0, ax.get_ylim()[1] * 2)
    for i, (a, e) in enumerate(spans):   # bin numbers (details in the table below the figure), staggered
        ax.text((a + e) / 2, .985 - .06 * (i % 3), f"{i + 1}", ha="center", va="top", fontsize=8,
                color="#24323a", transform=ax.get_xaxis_transform())
    ax.set(ylabel="fraction of rows per bin",
           title="eC distributions (train sets as trained, validation shaded); numbered bands = metric bins")
    _tidy(ax); ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    x = np.array([b["median_eC"] for b in bins])
    dodge = .004 * (hi_x - lo_x)
    offsets = (np.arange(len(names)) - (len(names) - 1) / 2) * dodge   # small horizontal dodge
    for ax, (key, label) in zip(axes[1:], (("r2", "within-bin R²"), ("rmse", "RMSE of eC"),
                                           ("rel_rmse", "relative RMSE\n(RMSE / mean true eC)"))):
        for c, name in enumerate(names):
            v = np.array([b["runs"][name][key]["value"] for b in bins])
            lo = np.array([b["runs"][name][key]["lo"] for b in bins])
            hi = np.array([b["runs"][name][key]["hi"] for b in bins])
            med = float(np.nanmedian(v))
            ax.errorbar(x + offsets[c], v, yerr=[np.maximum(v - lo, 0), np.maximum(hi - v, 0)], fmt="o-",
                        ms=5.5, lw=1.4, capsize=2.5, color=COLORS[c], label=f"{name} (median over bins {med:.3g})",
                        zorder=3)
            ax.axhline(med, color=COLORS[c], ls="--", lw=1.1, alpha=.9, zorder=2)
        if key == "r2":
            ax.set_yscale("symlog", linthresh=1)   # linear in [-1, 1]; very negative bins compressed
            ax.axhline(0, color="#24323a", lw=.7, zorder=2)
            ax.set_title("within-bin R² (symlog below −1; small sd in a bin makes its R² unstable)", fontsize=10)
        ax.set_ylabel(label)
        _tidy(ax); ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    for ax in axes:
        mark(ax)
    axes[-1].set(xlim=(lo_x, hi_x), xlabel="true eC (points at the bin median)")
    fig.tight_layout()
    fig.savefig(assets / "by_eC_aligned.png", dpi=110); plt.close(fig)
    return "assets/by_eC_aligned.png"


# ---------------------------------------------------------------- html

def fmt(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.3g}"


def table(header, rows):
    return ("<table><tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in header) + "</tr>" +
            "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows) + "</table>")


def diff_cell(m, higher_is_better):
    """Paired difference to the reference, marked when its 95% interval excludes 0."""
    text = f"{m['diff']:+.3g} [{m['diff_lo']:+.3g}, {m['diff_hi']:+.3g}]"
    if m["diff_lo"] > 0 or m["diff_hi"] < 0:
        good = (m["diff_lo"] > 0) == higher_is_better
        return f"<span class='{'better' if good else 'worse'}'>{text}</span>"
    return text


def report(output, args, truth, pred, names, bins, overall, images):
    ref = names[0]
    parts = [f"<!doctype html><meta charset='utf-8'><title>eC by severity</title>{STYLE}",
             f"<h1>{html.escape(args.model)}: eC accuracy by true eC, per training set</h1>",
             f"<p class='note'>Same validation rows for every run: <code>{html.escape(str(args.val))}</code> "
             f"({len(truth['eC'])} rows, {len(set(truth['source_ids']))} sources). Bins are equal-count bins of the "
             f"validation eC ({len(bins)} bins), shared by all runs. Intervals: 95% source bootstrap "
             f"({args.boots} resamples of validation sources). Reference for paired differences: <b>{html.escape(ref)}</b>.</p>",
             "<h2>Runs</h2>",
             table(["Run", "Path", "Train rows", "R² (all val)", "RMSE (all val)"],
                   [[html.escape(n), f"<code>{html.escape(p)}</code>", str(pred.get(n + '__train_rows') or '—'),
                     f"{fmt(overall[n]['r2']['value'])} [{fmt(overall[n]['r2']['lo'])}, {fmt(overall[n]['r2']['hi'])}]",
                     f"{fmt(overall[n]['rmse']['value'])} [{fmt(overall[n]['rmse']['lo'])}, {fmt(overall[n]['rmse']['hi'])}]"]
                    for n, p in args.runs]),
             "<h2>True vs predicted eC</h2>",
             "<p class='note'>Top: linear axes. Bottom: log axes, predictions below the floor drawn at the floor. "
             "The dark line is the median prediction in 5% equal-count bins of the truth: below the dashed y = x "
             "line means the model under-predicts at that severity.</p>",
             f"<img src='{images['diagonal']}' alt='True vs predicted eC per run'>",
             "<h2>Accuracy per eC bin</h2>",
             "<p class='note'><b>Within-bin R²</b> compares the errors with the spread of true eC <i>inside</i> the bin. "
             "Narrow bins have little spread, so R² there is low or negative even for small errors; compare runs with "
             "each other, not with the overall R². <b>RMSE</b> grows with eC by nature; <b>relative RMSE</b> (RMSE / mean "
             "true eC of the bin) removes that scale.</p>",
             "<p class='note'>All panels share one linear eC axis, with dotted lines at eC = 0.05 and 1; dashed lines are each run's median over bins (value in the legend). Top: eC distributions of the validation rows (shaded) "
             "and of each model's training set as it was trained (filters applied), as fractions per bin. The "
             "alternating numbered bands are the metric bins (rows, eC range and within-bin spread in the table below); "
             "each point sits at its bin's median eC.</p>",
             f"<img src='{images['aligned']}' alt='eC distributions aligned with per-bin R2, RMSE and relative RMSE'>",
             table(["bin", "true eC range", "median eC", "rows", "sources", "sd of true eC (within bin)"],
                   [[i + 1, f"{b['lo']:.3g}–{b['hi']:.3g}", fmt(b["median_eC"]), b["n"], b["sources"], fmt(b["sd_eC"])]
                    for i, b in enumerate(bins)]),
             "<h2>Numbers per bin</h2>",
             f"<p class='note'>Value [95% interval]. Δ = run − {html.escape(ref)}, paired bootstrap interval; "
             "<span class='better'>green</span>/<span class='worse'>red</span> when the interval excludes 0. "
             "Bias = median of (predicted − true).</p>"]
    for key, label, higher in (("rmse", "RMSE", False), ("rel_rmse", "relative RMSE", False), ("r2", "within-bin R²", True)):
        rows = []
        for b in bins:
            row = [f"{b['lo']:.3g}–{b['hi']:.3g}", b["n"], b["sources"]]
            for n in names:
                m = b["runs"][n][key]
                row.append(f"{fmt(m['value'])} [{fmt(m['lo'])}, {fmt(m['hi'])}]")
                if n != ref:
                    row.append(diff_cell(m, higher))
            rows.append(row)
        header = ["eC bin", "rows", "sources"]
        for n in names:
            header += [n] + ([f"Δ {n}"] if n != ref else [])
        parts += [f"<h3>{label}</h3>", table(header, rows)]
    rows = [[f"{b['lo']:.3g}–{b['hi']:.3g}"] + [fmt(b["runs"][n]["median_signed_error"]) for n in names] for b in bins]
    parts += ["<h3>Bias (median predicted − true)</h3>", table(["eC bin"] + names, rows),
              "<p class='note'>All numbers: <code>summary.json</code>.</p>"]
    (output / "report.html").write_text("\n".join(parts))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--val", default="runs/dataset-v2-matched-val", help="Validation run folder or .npz (truth)")
    p.add_argument("--run", action="append", metavar="NAME=RUN", help="Multitask run; repeatable; first is the reference")
    p.add_argument("--train", action="append", metavar="NAME=PATH",
                   help="Training set of run NAME (run folder or manifest), for the distribution panel. "
                        "Default: the v2, v2.1, v2.1 no-cap and v2.2 training sets.")
    p.add_argument("--model", default="xgb_mlp")
    p.add_argument("--bins", type=int, default=10, help="Equal-count eC bins (default 10)")
    p.add_argument("--boots", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    a.runs = [tuple(x.split("=", 1)) for x in a.run] if a.run else list(DEFAULT_RUNS)
    if any(len(r) != 2 for r in a.runs) or len({r[0] for r in a.runs}) != len(a.runs):
        p.error("--run must be NAME=RUN with unique names")
    names = [r[0] for r in a.runs]
    a.trains = [tuple(x.split("=", 1)) for x in a.train] if a.train else [t for t in DEFAULT_TRAINS if t[0] in names]
    output = _resolve(a.output)
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    truth, pred = load(a.val, a.runs, a.model)
    edges = equal_count_edges(truth["eC"], a.bins)
    bins, overall = analyze(truth, pred, names, edges, a.boots, a.seed)
    train = load_train(a.trains, pred)
    images = dict(diagonal=plot_diagonal(truth, pred, names, overall, assets),
                  aligned=plot_aligned(truth, train, bins, names, assets))
    (output / "summary.json").write_text(json.dumps(dict(model=a.model, val=str(a.val), runs=a.runs, reference=names[0],
                                                         bins=bins, overall=overall, boots=a.boots, seed=a.seed),
                                                    indent=1) + "\n")
    report(output, a, truth, pred, names, bins, overall, images)
    print(f"Report: {output / 'report.html'}")


if __name__ == "__main__":
    main()
