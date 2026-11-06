"""Learning curve for the amplification-cap setup (cap 2), by number of training sources.

python -m ml.amp_learning_curve --train RUN_TRAIN --val RUN_VAL --output OUT [--sizes 25 50 75] [--repeats 5] [--jobs 8]

Same setup as ml/amp_cap_experiment.py with cap 2: xgb_shared (ml/multitask.py), all20
features, training rows with A <= 2, evaluation on the fixed validation rows with A <= 2.
Settings are fixed for every point: the tree settings and the target scales are chosen
once on the full cap-2 training set (validation J, the multitask rule) and reused. Each
point trains on a random subset of whole training sources (repeated with different
draws); the largest point is all sources. Rows are only filtered, never modified.
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np

from ml import multitask as mt
from ml.amp_cap_experiment import EVAL_CAP, VIEW, keep_amp, rmse_r2
from scripts.preprocessing.schema import atomic_write_json

CAP = 2.
SEED = 20261105
SERIES = ("#2a78d6", "#eb6834")


def scores(pred, data):
    """RMSE per quantity on `data` (physical units). Active = that type injected."""
    A, b, d = data["params"][:, 0], data["b"].astype(bool), data["d"]
    out = {}
    out["d_amp active"] = rmse_r2(pred[b[:, 0], 0], d[b[:, 0], 0])[0]
    out["d_amp amplification (1 < A ≤ 2)"] = rmse_r2(pred[b[:, 0] & (A > 1), 0], d[b[:, 0] & (A > 1), 0])[0]
    out["d_amp attenuation (A < 1)"] = rmse_r2(pred[b[:, 0] & (A < 1), 0], d[b[:, 0] & (A < 1), 0])[0]
    out["d_amp absent"] = rmse_r2(pred[~b[:, 0], 0], d[~b[:, 0], 0])[0]
    out["d_phase active"] = rmse_r2(pred[b[:, 1], 1], d[b[:, 1], 1])[0]
    out["d_noise active"] = rmse_r2(pred[b[:, 2], 2], d[b[:, 2], 2])[0]
    out["eC"] = rmse_r2(pred[:, 3], data["y_all"])[0]
    return out


def zero_scores(data):
    pred = np.zeros((len(data["X"]), 7))
    return scores(pred, data)


def write_report(output, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = list(summary["zero_val"])
    points = summary["points"]
    sizes = sorted({p["sources"] for p in points})
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for ax, metric in zip(axes.flat, metrics):
        for i, split in enumerate(("train", "val")):
            per = [[p[split][metric] for p in points if p["sources"] == n and p[split][metric] is not None] for n in sizes]
            mean = [np.mean(v) if v else np.nan for v in per]
            ax.plot(sizes, mean, "-o", color=SERIES[i], lw=2, ms=5, label=f"{split} (mean over draws)")
            for n, v in zip(sizes, per):
                ax.scatter([n] * len(v), v, s=12, color=SERIES[i], alpha=.35, linewidths=0)
        zero = summary["zero_val"][metric]
        if zero is not None:
            ax.axhline(zero, color="#8a8a85", lw=1, ls="--", label="val predict 0")
        ax.set(title=metric, xlabel="training sources", ylabel="RMSE", ylim=(0, None))
        ax.grid(alpha=.2)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False)
    fig.tight_layout()
    fig.savefig(output / "learning_curve.png", dpi=130); plt.close(fig)

    css = ("body{font:15px system-ui;max-width:1300px;margin:auto;padding:1.5rem;color:#1b2932}"
           "table{border-collapse:collapse;margin:.4rem 0 1.2rem}th,td{border:1px solid #ccd;padding:.3rem .5rem;text-align:right}"
           "th:first-child,td:first-child{text-align:left}.scroll{overflow-x:auto}img{max-width:100%}")
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>Amplification learning curve</title><style>{css}</style></head><body>",
             f"<h1>Learning curve: cap 2, xgb_shared, {summary['feature_view']}</h1>",
             f"<p>Training rows with A ≤ {CAP:g}, from random subsets of whole training sources ({summary['repeats']} draws per size, "
             f"seed {SEED}; the largest size uses all {summary['all_sources']} sources once). Every point is evaluated on the same "
             f"{summary['val_rows']} validation rows with A ≤ {EVAL_CAP:g} from {summary['val_sources']} held-out sources. "
             f"Tree settings {html.escape(str(summary['params']))} and the target scales were chosen once on the full cap-2 training set "
             "and kept fixed. RMSE is in physical units on active rows (that type injected), except the 'absent' and eC panels. "
             "Train RMSE is on the rows that point was trained on.</p>",
             "<p>Reading: validation still falling at the largest size supports more sources. Validation flat while train stays much lower "
             "points to generalization or regularization. Both high and flat points to the features or the model.</p>",
             "<img src='learning_curve.png' alt='Train and validation RMSE against the number of training sources'>",
             "<p><a href='results.json'>All points</a></p>"]
    for split in ("val", "train"):
        parts.append(f"<h2>{split} RMSE (mean, min to max over draws)</h2><div class='scroll'><table><tr><th>sources</th><th>rows (mean)</th>"
                     + "".join(f"<th>{html.escape(m)}</th>" for m in metrics) + "</tr>")
        for n in sizes:
            group = [p for p in points if p["sources"] == n]
            cells = []
            for m in metrics:
                v = [p[split][m] for p in group if p[split][m] is not None]
                cells.append("<td>—</td>" if not v else f"<td>{np.mean(v):.4f}" + (f" ({min(v):.4f}–{max(v):.4f})" if len(v) > 1 else "") + "</td>")
            parts.append(f"<tr><td>{n}</td><td>{np.mean([p['train_rows'] for p in group]):.0f}</td>{''.join(cells)}</tr>")
        if split == "val":
            parts.append("<tr><td>predict 0</td><td></td>" + "".join(
                "<td>—</td>" if summary["zero_val"][m] is None else f"<td>{summary['zero_val'][m]:.4f}</td>" for m in metrics) + "</tr>")
        parts.append("</table></div>")
    parts.append("</body></html>")
    (output / "report.html").write_text("".join(parts))


def run(train_root, val_root, output, sizes, repeats, jobs, view=VIEW):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise ValueError("Output already exists; choose a fresh directory")
    train_all, val_all = mt.load_split(train_root, "train", view), mt.load_split(val_root, "val", view)
    if set(train_all["source_ids"]) & set(val_all["source_ids"]):
        raise ValueError("Train/validation sources overlap")
    train_cap = mt.subset(train_all, np.flatnonzero(keep_amp(train_all, CAP)))
    val = mt.subset(val_all, np.flatnonzero(keep_amp(val_all, EVAL_CAP)))
    sources = np.asarray(sorted(set(map(str, train_cap["source_ids"]))))
    sizes = sorted({min(n, len(sources)) for n in sizes} | {len(sources)})
    if sizes[0] < 3:
        raise ValueError("Each subset needs at least three sources")
    full = mt._fit_tree_family(train_cap, val, "xgb_shared", jobs)
    params, scale = full["params"], full["scale"]
    print("Fixed settings", params, flush=True)
    rng = np.random.default_rng(SEED)
    points = []
    for n in sizes:
        draws = [sources] if n == len(sources) else [rng.choice(sources, n, replace=False) for _ in range(repeats)]
        for r, chosen in enumerate(draws):
            rows = np.flatnonzero(np.isin(np.asarray(list(map(str, train_cap["source_ids"]))), chosen))
            train = mt.subset(train_cap, rows)
            if not np.all(train["b"].sum(0) > 0):
                raise ValueError("A subset lacks active rows for some corruption type")
            model = mt.fit_xgb(train["X"], train, params, scale, shared=True, jobs=jobs)
            point = dict(sources=int(n), draw=r, train_rows=len(rows), chosen=sorted(map(str, chosen)),
                         train=scores(mt.decode(mt.predict_xgb(model, train["X"]), scale), train),
                         val=scores(mt.decode(mt.predict_xgb(model, val["X"]), scale), val))
            points.append(point)
            print(f"{n} sources, draw {r}: val d_amp active RMSE {point['val']['d_amp active']:.4f}", flush=True)
    summary = dict(model="xgb_shared", feature_view=view, cap=CAP, eval_cap=EVAL_CAP, seed=SEED, repeats=repeats,
                   params=params, scale=scale.tolist(), all_sources=len(sources),
                   val_rows=len(val["X"]), val_sources=len(set(val["source_ids"])), val_row_ids=val["row_ids"].tolist(),
                   zero_val=zero_scores(val), points=points)
    atomic_write_json(output / "results.json", summary)
    write_report(output, summary)
    print("Report:", output / "report.html", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sizes", type=int, nargs="+", default=[25, 50, 75])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--feature-view", choices=mt.FEATURE_VIEWS, default=VIEW)
    a = p.parse_args()
    run(a.train, a.val, a.output, a.sizes, a.repeats, a.jobs, a.feature_view)


if __name__ == "__main__":
    main()
