"""eC accuracy by true eC on the TRAINING and the VALIDATION rows, one line per training dataset.

Question: is a run bad at large eC on validation because its training set held nothing like those validation
rows, or are large corruptions hard to fit even on the rows it was trained on?

Read-only: uses each multitask run's saved predictions_{train,val}_<model>.npz (decoded column 3 = eC), the
validation <split>.npz and each run's training set (run folder or v2.1 manifest, filters applied) for the truth.
Both panels use the same eC bin edges: the equal-count bins of the validation eC, plus extra bins above the
validation maximum, where only training rows exist. Per bin: RMSE and relative RMSE (RMSE / mean true eC) with a
95% source bootstrap (sources resampled with replacement within each split). Linear y axes, shared by the
train and validation panels.

Training predictions: for xgb_mlp / xgb_xgb the expert features of the training rows are source-held-out
(out-of-fold), but the second stage was fitted on these very rows, so the training curve is a fit error
(how well the model can represent the rows it saw), not a generalisation error.

  bin/run ml python -m ml.compare_train_val_by_ec --output ml/runs/compare-train-val-xgb_mlp-v2-v21-v22-by-eC
  # defaults: --model xgb_mlp --val runs/dataset-v2-matched-val, runs v2, v2.1, v2.1-nocap, v2.2, v2.3 (unweighted;
  #   runs whose folder does not exist yet are skipped)
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ml.compare_val_by_ec import (COLORS, DEFAULT_RUNS, DEFAULT_TRAINS, EC_COLUMN, STYLE, _log_ec_ticks, _resolve,
                                  _tidy, bootstrap_weights, equal_count_edges, fmt, mark, table, weighted_metrics)

# One line per training dataset version; the v2.2 tail-weighted runs share v2.2's rows and are left out by default.
DATASETS = ("v2", "v2.1", "v2.1-nocap", "v2.2", "v2.3")
METRICS = (("rmse", "RMSE of eC"), ("rel_rmse", "relative RMSE (RMSE / mean true eC)"))


def read_truth(path, split):
    from scripts.dataset_v21 import load_manifest, read_part
    path = _resolve(path)
    if path.suffix == ".json":
        data = load_manifest(path, split)
    elif path.suffix == ".npz":
        data = dict(np.load(path, allow_pickle=False))
    else:
        data = read_part(path, split)[0]
    return dict(eC=np.asarray(data["eC"], float), row_ids=np.asarray(data["row_ids"]).astype(str),
                source_ids=np.asarray(data["source_ids"]).astype(str))


def predictions(run, split, model, truth):
    with np.load(_resolve(run) / f"predictions_{split}_{model}.npz", allow_pickle=False) as z:
        ids, decoded = np.asarray(z["row_ids"]).astype(str), np.asarray(z["decoded"], float)
    if not np.array_equal(ids, truth["row_ids"]):
        raise ValueError(f"{run}: {split} prediction rows differ from the {split} truth rows")
    return decoded[:, EC_COLUMN]


def per_bin(truth, pred, edges, boots, seed):
    y = truth["eC"]
    which = np.digitize(y, edges[1:-1])
    W = bootstrap_weights(truth["source_ids"], boots, seed)
    out = []
    for b in range(len(edges) - 1):
        sel = which == b
        if sel.sum() < 2:
            out.append(dict(n=int(sel.sum())))
            continue
        point = weighted_metrics(y[sel], pred[sel], np.ones(int(sel.sum())))
        boot = weighted_metrics(y[sel], pred[sel], W[:, sel])
        entry = dict(n=int(sel.sum()), sources=len(set(truth["source_ids"][sel])), median_eC=float(np.median(y[sel])),
                     bias=float(np.median(pred[sel] - y[sel])))
        for k in ("rmse", "rel_rmse", "r2"):
            entry[k] = dict(value=float(point[k][0]), lo=float(np.nanpercentile(boot[k], 2.5)),
                            hi=float(np.nanpercentile(boot[k], 97.5)))
        out.append(entry)
    return out


def zoom_limit(results, names, key, val_max):
    """Upper y limit that shows every bin with median eC in [0.05, val max] (both splits, 95% intervals included)."""
    top = [b[key]["hi"] for n in names for s in ("train", "val") for b in results[n][s]
           if key in b and .05 <= b["median_eC"] <= val_max and np.isfinite(b[key]["hi"])]
    return 1.08 * max(top)


def plot(results, names, edges, val_max, key, label, lo_x, hi_x, path, ymax=None):
    """Train (top) and validation (bottom), same log eC axis and bins, linear y shared by both panels.
    ymax: zoomed view; points above it are drawn as triangles on the top edge, labelled with their value."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, sharey=True)
    inner = edges[1:-1]
    spans = list(zip([lo_x] + list(inner), list(inner) + [hi_x]))
    offsets = 10 ** ((np.arange(len(names)) - (len(names) - 1) / 2) * .004 * np.log10(hi_x / lo_x))
    for ax, split in zip(axes, ("train", "val")):
        for i, (a, b) in enumerate(spans):
            if i % 2:
                ax.axvspan(a, b, color="#eef3f6", zorder=0, lw=0)
        for x in inner:
            ax.axvline(x, color="#b9c4ca", lw=.7, zorder=1)
        ax.axvspan(val_max, hi_x, color="#f3ede4", zorder=0, lw=0)   # beyond the largest validation eC
        for c, name in enumerate(names):
            bins = results[name][split]
            ok = [i for i, b in enumerate(bins) if "rmse" in b]
            x = np.array([bins[i]["median_eC"] for i in ok])
            v = np.array([bins[i][key]["value"] for i in ok])
            lo = np.array([bins[i][key]["lo"] for i in ok])
            hi = np.array([bins[i][key]["hi"] for i in ok])
            ax.errorbar(x * offsets[c], v, yerr=[np.maximum(v - lo, 0), np.maximum(hi - v, 0)], fmt="o-", ms=5.5,
                        lw=1.4, capsize=2.5, color=COLORS[c], zorder=3,
                        label=f"{name} (n={results[name][split + '_rows']})")
            if ymax is not None:
                for xi, vi in zip(x * offsets[c], v):
                    if vi > ymax:
                        ax.plot(xi, ymax, marker="^", ms=8, color=COLORS[c], clip_on=False, zorder=4)
                        ax.annotate(f"{vi:.2f}", (xi, ymax), xytext=(0, -12 - 9 * c), textcoords="offset points",
                                    ha="center", fontsize=7, color=COLORS[c])
        ax.set(ylabel=label)
        ax.set_ylim(0, ymax)
        ax.set_title(f"{'training rows (fit error: second stage saw these rows)' if split == 'train' else 'validation rows (same rows for every run)'}",
                     fontsize=11, loc="left")
        ax.text(np.sqrt(val_max * hi_x), .97, "above val max", transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=8, color="#6b5a3e")
        mark(ax); _tidy(ax)
        ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1), title="trained on")
    axes[-1].set(xscale="log", xlim=(lo_x, hi_x), xlabel="true eC (log; points at the bin median of that split)")
    _log_ec_ticks(axes, lo_x, hi_x)
    fig.tight_layout()
    fig.savefig(path, dpi=110); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--val", default="runs/dataset-v2-matched-val")
    p.add_argument("--run", action="append", metavar="NAME=RUN", help="Multitask run; repeatable (default v2..v2.3)")
    p.add_argument("--train", action="append", metavar="NAME=PATH", help="Training set of run NAME (default: known sets)")
    p.add_argument("--model", default="xgb_mlp")
    p.add_argument("--bins", type=int, default=10, help="Equal-count validation eC bins (default 10)")
    p.add_argument("--boots", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    runs = [tuple(x.split("=", 1)) for x in a.run] if a.run else [r for r in DEFAULT_RUNS if r[0] in DATASETS and _resolve(r[1]).exists()]
    trains = dict(tuple(x.split("=", 1)) for x in a.train) if a.train else dict(DEFAULT_TRAINS)
    names = [n for n, _ in runs]
    output = _resolve(a.output)
    (output / "assets").mkdir(parents=True, exist_ok=True)

    val = read_truth(a.val, "val")
    edges = equal_count_edges(val["eC"], a.bins)
    val_max = float(val["eC"].max())
    train_truth = {n: read_truth(trains[n], "train") for n in names}
    train_max = max(float(t["eC"].max()) for t in train_truth.values())
    extra = [e for e in (val_max * 1.0001, 4., 8.) if e > val_max and e < train_max]   # bins only training rows reach
    edges = np.r_[edges[:-1], extra, np.inf]

    results = {}
    for (name, run) in runs:
        tr = train_truth[name]
        summary = json.loads((_resolve(run) / "results.json").read_text())
        if summary.get("part_weights"):
            print(f"Warning: {name} was trained with part weights {summary['part_weights']}; train metrics here are unweighted")
        results[name] = dict(run=run, train_set=trains[name], train_rows=len(tr["eC"]), val_rows=len(val["eC"]),
                             train=per_bin(tr, predictions(run, "train", a.model, tr), edges, a.boots, a.seed),
                             val=per_bin(val, predictions(run, "val", a.model, val), edges, a.boots, a.seed))

    pool = np.concatenate([val["eC"]] + [t["eC"] for t in train_truth.values()])
    lo_x = 10.0 ** np.floor(np.log10(np.quantile(pool[pool > 0], .02)))
    hi_x = train_max * 1.1
    images = {}
    for key, label in METRICS:
        images[key] = f"assets/train_val_{key}.png"
        plot(results, names, edges, val_max, key, label, lo_x, hi_x, output / images[key])
        images[key + "_zoom"] = f"assets/train_val_{key}_zoom.png"
        plot(results, names, edges, val_max, key, label, lo_x, hi_x, output / images[key + "_zoom"],
             ymax=zoom_limit(results, names, key, val_max))

    def bin_label(i):
        lo = "−∞" if not np.isfinite(edges[i]) else f"{edges[i]:.3g}"
        hi = "∞" if not np.isfinite(edges[i + 1]) else f"{edges[i + 1]:.3g}"
        return f"{lo}–{hi}"

    parts = [f"<!doctype html><meta charset='utf-8'><title>Train vs val eC</title>{STYLE}",
             f"<h1>{html.escape(a.model)}: eC error by true eC on training and validation rows</h1>",
             "<p class='note'>Does a run fail at large eC on validation because its training set did not cover those "
             "rows, or because large eC is hard to fit at all? If the training curve stays low where validation rises, "
             "the model fits what it saw and the problem is coverage / generalisation; if training also rises, large "
             "corruptions are hard to represent with these features.</p>",
             f"<p class='note'>Bins: {a.bins} equal-count bins of the validation eC, plus extra bins above the validation "
             f"maximum ({val_max:.3g}, tan band) where only training rows exist. Same edges in both panels; points at each "
             f"split's bin median. 95% source bootstrap ({a.boots} resamples). <b>Training predictions are in-sample for "
             "the second stage</b> (expert features are source-held-out, out-of-fold), so the top panel is a fit error. "
             "Training metrics are unweighted.</p>",
             table(["Run", "Path", "Training set", "train rows", "val rows"],
                   [[html.escape(n), f"<code>{html.escape(r['run'])}</code>", f"<code>{html.escape(r['train_set'])}</code>",
                     r["train_rows"], r["val_rows"]] for n, r in results.items()])]
    for key, label in METRICS:
        parts += [f"<h2>{label}</h2>",
                  "<p class='note'>Zoomed (linear y): limited to the bins with median eC between 0.05 and the validation "
                  "maximum. Points above the limit are triangles on the top edge, labelled with their value.</p>",
                  f"<img src='{images[key + '_zoom']}' alt='{label} per eC bin, train above val, zoomed'>",
                  "<p class='note'>Full range (linear y).</p>",
                  f"<img src='{images[key]}' alt='{label} per eC bin, train above val'>"]
    parts.append("<h2>Numbers per bin</h2><p class='note'>Value [95% interval] (rows). Bias = median(pred − true).</p>")
    for key, label in METRICS + (("bias", "bias"),):
        header, rows = ["eC bin"], []
        for n in names:
            header += [f"{n} train", f"{n} val"]
        for i in range(len(edges) - 1):
            row = [bin_label(i)]
            for n in names:
                for split in ("train", "val"):
                    b = results[n][split][i]
                    if key not in b:
                        row.append(f"— ({b['n']})")
                    elif key == "bias":
                        row.append(f"{b['bias']:+.3g}")
                    else:
                        m = b[key]
                        row.append(f"{fmt(m['value'])} [{fmt(m['lo'])}, {fmt(m['hi'])}] ({b['n']})")
            rows.append(row)
        parts += [f"<h3>{label}</h3>", table(header, rows)]
    parts.append("<p class='note'>All numbers: <code>summary.json</code>.</p>")
    (output / "report.html").write_text("\n".join(parts))
    (output / "summary.json").write_text(json.dumps(dict(model=a.model, val=a.val, edges=[float(e) for e in edges],
                                                         boots=a.boots, seed=a.seed, runs=results), indent=1) + "\n")
    print(f"Report: {output / 'report.html'}")


if __name__ == "__main__":
    main()
