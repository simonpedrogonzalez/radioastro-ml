"""Recipe-density curve: does training on more recipes per source help, with all sources kept?

python -m ml.recipe_density_curve --train RUN_TRAIN --val RUN_VAL --output OUT
       [--per-branch 1 2 3] [--repeats 5] [--feature-view all22] [--cap A] [--jobs 8]

Answers whether denser simulation of the same sources (more sampling templates per
source) would improve held-out results, before generating any. Every training source
is kept; only its recipes are subsampled, stratified by branch (attenuation or
amplification x phase sign): k recipes per branch gives 4k recipes per source
(4, 8, 12 with the default R = 12). A recipe is kept or dropped with all 7 of its rows;
the shared control row of each source is always kept. Rows are only filtered, never
modified. Model: xgb_shared from ml/multitask.py; its tree settings and target scales
are chosen once on the full training set (validation J) and kept fixed. Every point is
evaluated on the same validation rows. Optional --cap keeps rows with gain A <= cap in
both splits (as in ml/amp_cap_experiment.py).
"""
from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ml import multitask as mt
from ml.amp_cap_experiment import keep_amp
from scripts.preprocessing.schema import atomic_write_json

SEED = 20261106
KINDS = ("amp", "phase", "noise")
SERIES = ("#2a78d6", "#eb6834")


def recipe_groups(train_root):
    """recipe id -> (source, branch group) from the immutable training plan(s).

    For a dataset v2.1 manifest the ids carry the part prefix, and a branch group is per
    part, so each part keeps its own stratification when recipes are subsampled.
    """
    if mt.is_manifest(train_root):
        from scripts.dataset_v21 import _resolve
        manifest = json.loads(_resolve(train_root).read_text())
        out = {}
        for e in manifest["parts"]:
            plan = json.loads((_resolve(e["run"]) / "plan.json").read_text())
            out.update({f"{e['tag']}/{r['id']}": (r["source"], f"{e['tag']}:{r['group']}")
                        for sp in plan["source_plans"] for r in sp["recipes"]})
        return out
    plan = json.loads((train_root / "plan.json").read_text())
    return {r["id"]: (r["source"], r["group"]) for sp in plan["source_plans"] for r in sp["recipes"]}


def draw_recipes(groups, available, per_branch, rng):
    """Choose `per_branch` recipes from every (source, branch) cell that has rows."""
    cells = defaultdict(list)
    for rid in sorted(available):
        cells[groups[rid]].append(rid)
    if any(len(v) < per_branch for v in cells.values()):
        raise ValueError(f"Some source/branch cell has fewer than {per_branch} recipes")
    return {rid for v in cells.values() for rid in rng.choice(v, per_branch, replace=False)}


def scores(pred, data):
    """RMSE (and active R²) per output on `data`, physical units."""
    b, d = data["b"].astype(bool), data["d"]
    out = {}
    for k, kind in enumerate(KINDS):
        on = b[:, k]
        err = pred[on, k] - d[on, k]
        var = float(np.sum((d[on, k] - d[on, k].mean())**2)) if on.any() else 0.
        out[f"d_{kind} active"] = float(np.sqrt(np.mean(err**2))) if on.any() else None
        out[f"d_{kind} active R²"] = float(1 - np.sum(err**2) / var) if var > 1e-15 else None
        out[f"d_{kind} absent"] = float(np.sqrt(np.mean(pred[~on, k]**2))) if (~on).any() else None   # true damage is 0
    err = pred[:, 3] - data["y_all"]
    out["eC"] = float(np.sqrt(np.mean(err**2)))
    out["eC R²"] = float(1 - np.sum(err**2) / np.sum((data["y_all"] - data["y_all"].mean())**2))
    return out


def write_report(output, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = summary["points"]
    sizes = sorted({p["recipes_per_source"] for p in points})
    rmse = [m for m in summary["zero_val"] if "R²" not in m]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for ax, metric in zip(axes.flat, rmse):
        for i, split in enumerate(("train", "val")):
            per = [[p[split][metric] for p in points if p["recipes_per_source"] == n and p[split][metric] is not None] for n in sizes]
            ax.plot(sizes, [np.mean(v) if v else np.nan for v in per], "-o", color=SERIES[i], lw=2, ms=5, label=f"{split} (mean over draws)")
            for n, v in zip(sizes, per):
                ax.scatter([n] * len(v), v, s=12, color=SERIES[i], alpha=.35, linewidths=0)
        zero = summary["zero_val"][metric]
        if zero is not None and "absent" not in metric:
            ax.axhline(zero, color="#8a8a85", lw=1, ls="--", label="val predict 0")
        ax.set(title=metric, xlabel="recipes per source", ylabel="RMSE", ylim=(0, None), xticks=sizes)
        ax.grid(alpha=.2)
    for ax in axes.flat[len(rmse):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False)
    fig.tight_layout()
    fig.savefig(output / "density_curve.png", dpi=130); plt.close(fig)

    metrics = list(summary["zero_val"])
    css = ("body{font:15px system-ui;max-width:1300px;margin:auto;padding:1.5rem;color:#1b2932}"
           "table{border-collapse:collapse;margin:.4rem 0 1.2rem}th,td{border:1px solid #ccd;padding:.3rem .5rem;text-align:right}"
           "th:first-child,td:first-child{text-align:left}.scroll{overflow-x:auto}img{max-width:100%}")
    cap = "all gains" if summary["cap"] is None else f"rows with A ≤ {summary['cap']:g} in both splits"
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>Recipe density curve</title><style>{css}</style></head><body>",
             f"<h1>Recipe density: xgb_shared, {html.escape(summary['feature_view'])}</h1>",
             f"<p>All {summary['train_sources']} training sources are kept at every point. Each source keeps k recipes per branch "
             f"(4 branches: attenuation or amplification × phase sign), so 4k recipes per source, each with all 7 of its rows; the "
             f"control row of each source is always kept. {summary['repeats']} random draws per density (seed {SEED}); the full "
             f"density is fitted once. Evaluation: the same {summary['val_rows']} validation rows from {summary['val_sources']} "
             f"held-out sources ({cap}). Tree settings {html.escape(str(summary['params']))} and target scales were chosen once on "
             "the full training set and kept fixed. RMSE is in physical units; 'active' rows have that corruption injected, 'absent' "
             "rows have true damage 0.</p>",
             "<p>Reading: validation still falling at the full density means more recipes per source (denser simulation of the same "
             "sources) should help. Flat validation means it would not, whatever the train error does.</p>",
             "<img src='density_curve.png' alt='Train and validation RMSE against recipes per source'>",
             "<p><a href='results.json'>All points and the drawn recipes</a></p>"]
    for split in ("val", "train"):
        parts.append(f"<h2>{split} (mean, min to max over draws)</h2><div class='scroll'><table><tr><th>recipes per source</th><th>rows</th>"
                     + "".join(f"<th>{html.escape(m)}</th>" for m in metrics) + "</tr>")
        for n in sizes:
            group = [p for p in points if p["recipes_per_source"] == n]
            cells = []
            for m in metrics:
                v = [p[split][m] for p in group if p[split][m] is not None]
                digits = 3 if "R²" in m else 4
                cells.append("<td>—</td>" if not v else f"<td>{np.mean(v):.{digits}f}"
                             + (f" ({min(v):.{digits}f}–{max(v):.{digits}f})" if len(v) > 1 else "") + "</td>")
            parts.append(f"<tr><td>{n}</td><td>{np.mean([p['train_rows'] for p in group]):.0f}</td>{''.join(cells)}</tr>")
        if split == "val":
            parts.append("<tr><td>predict 0</td><td></td>" + "".join(
                "<td>—</td>" if summary["zero_val"][m] is None else f"<td>{summary['zero_val'][m]:.4f}</td>" for m in metrics) + "</tr>")
        parts.append("</table></div>")
    parts.append("</body></html>")
    (output / "report.html").write_text("".join(parts))


def run(train_root, val_root, output, per_branch, repeats, jobs, view, cap):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise ValueError("Output already exists; choose a fresh directory")
    train, val = mt.load_split(train_root, "train", view), mt.load_split(val_root, "val", view)
    if set(train["source_ids"]) & set(val["source_ids"]):
        raise ValueError("Train/validation sources overlap")
    if cap is not None:
        train = mt.subset(train, np.flatnonzero(keep_amp(train, cap)))
        val = mt.subset(val, np.flatnonzero(keep_amp(val, cap)))
    groups = recipe_groups(train_root)
    recipe = np.asarray(list(map(str, train["recipe_ids"])))
    available = set(recipe) - {""}
    full_k = min(sum(1 for r in available if groups[r] == cell) for cell in {groups[r] for r in available})
    levels = sorted({k for k in per_branch if k < full_k} | {full_k})
    fitted = mt._fit_tree_family(train, val, "xgb_shared", jobs)
    params, scale = fitted["params"], fitted["scale"]
    print("Fixed settings", params, flush=True)
    rng = np.random.default_rng(SEED)
    points = []
    for k in levels:
        draws = [available] if k == full_k else [draw_recipes(groups, available, k, rng) for _ in range(repeats)]
        for r, chosen in enumerate(draws):
            rows = np.flatnonzero((recipe == "") | np.isin(recipe, sorted(chosen)))
            part = mt.subset(train, rows)
            if not np.all(part["b"].sum(0) > 0):
                raise ValueError("A subset lacks active rows for some corruption type")
            model = mt.fit_xgb(part["X"], part, params, scale, shared=True, jobs=jobs)
            point = dict(recipes_per_source=4 * k, per_branch=k, draw=r, train_rows=len(rows),
                         recipes=sorted(chosen) if k != full_k else "all",
                         train=scores(mt.decode(mt.predict_xgb(model, part["X"]), scale), part),
                         val=scores(mt.decode(mt.predict_xgb(model, val["X"]), scale), val))
            points.append(point)
            print(f"{4 * k} recipes/source, draw {r}: {len(rows)} rows, val d_amp active {point['val']['d_amp active']:.4f}", flush=True)
    summary = dict(model="xgb_shared", feature_view=view, cap=cap, seed=SEED, repeats=repeats, params=params,
                   scale=scale.tolist(), train_sources=len(set(map(str, train["source_ids"]))),
                   val_rows=len(val["X"]), val_sources=len(set(map(str, val["source_ids"]))),
                   val_row_ids=val["row_ids"].tolist(), zero_val=scores(np.zeros((len(val["X"]), 7)), val), points=points)
    atomic_write_json(output / "results.json", summary)
    write_report(output, summary)
    print("Report:", output / "report.html", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--per-branch", type=int, nargs="+", default=[1, 2], help="Recipes per branch below the full density")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--feature-view", choices=mt.FEATURE_VIEWS, default="all22")
    p.add_argument("--cap", type=float, default=None, help="Keep only rows with gain A <= cap (both splits)")
    p.add_argument("--jobs", type=int, default=8)
    a = p.parse_args()
    run(a.train, a.val, a.output, a.per_branch, a.repeats, a.jobs, a.feature_view, a.cap)


if __name__ == "__main__":
    main()
