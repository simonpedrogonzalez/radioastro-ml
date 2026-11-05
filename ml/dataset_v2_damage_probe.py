"""Read-only Dataset v2 usefulness probe: shared XGBoost on signed damage only.

Repeats the Dataset 1 recovery-damage report for one model (shared multi-output
XGBoost, combined8 features) on the sources of an in-progress Dataset v2 run whose
cases are all committed. It only reads the run directory and writes to its own
output folder, so it never touches the generation (no status.json, no npz there).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import ParameterGrid
from xgboost import XGBRegressor

from ml.logreg import COMBINED8_FEATURES
from ml.regression import GRID
from ml.regression_training import masked_squared_error
from scripts.create_dataset_v2 import file_sha
from scripts.dataset_v2 import TYPES, join_rows

RUN = Path("runs/dataset-v2-20261105")
COLORS_MASK = ("#555555", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#7B61A8", "#C49A00", "#222222")
MASK_LABELS = ("control", "amp", "phase", "amp+phase", "noise", "amp+noise", "phase+noise", "all three")
PARAM = {"amp": ("A", "Injected gain A (log)"), "phase": ("phi_deg", "Injected phase (deg)"),
         "noise": ("r", "Thermal ratio r (log)")}


def complete_split(split_dir):
    """Rows of every source whose planned cases are all committed and sealed."""
    plan = json.loads((split_dir / "plan.json").read_text())
    plan_hash = file_sha(split_dir / "plan.json")
    records = {}
    for c in plan["cases"]:
        path = split_dir / "cases" / f"{c['id']}.json"
        seal = path.with_suffix(".seal.json")
        if not seal.is_file():  # seal is written last, so its presence means committed
            continue
        rec = json.loads(path.read_text())
        if json.loads(seal.read_text())["sha256"] != file_sha(path) or rec["case"] != c or rec["plan_sha256"] != plan_hash:
            raise ValueError(f"Invalid committed case {path}")
        records[c["id"]] = rec
    done = [p["source"] for p in plan["source_plans"] if all(c["id"] in records for c in p["cases"])]
    rows = join_rows(dict(rows=[r for r in plan["rows"] if r["source"] in set(done)]), records)
    vectors = {v["id"]: v for v in plan["vectors"]}
    n = len(rows)
    data = dict(X=np.asarray([r["X"] for r in rows], float),
                b=np.asarray([r["b"] for r in rows], int), o=np.asarray([r["o"] for r in rows], int),
                pi=np.asarray([r["pi"] for r in rows], float), mask=np.asarray([r["mask"] for r in rows], int),
                d=np.asarray([[np.nan if x is None else x for x in r["d"]] for r in rows], float),
                y_all=np.asarray([r["y_all"] for r in rows], float),
                source_ids=np.asarray([r["source"] for r in rows]), row_ids=np.asarray([r["id"] for r in rows]),
                params=np.asarray([[vectors[r["vector"]][k] if r["vector"] else v for k, v in
                                    (("A", 1.), ("phi_deg", 0.), ("r", np.nan))] for r in rows], float))
    if data["X"].shape != (n, 8) or not np.isfinite(data["X"]).all():
        raise ValueError("Invalid features")
    return data, dict(planned_sources=len(plan["source_plans"]), complete_sources=len(done),
                      committed_cases=len(records), planned_cases=len(plan["cases"]))


def damage_loss(p, data):
    """Reference L_d: inverse-probability weighted squared error over all N rows."""
    w = data["o"] / data["pi"]
    err = np.where(data["o"] == 1, p - np.nan_to_num(data["d"]), 0.)
    per = (w * err**2).sum(0) / len(p)
    act, absn = data["b"] == 1, data["b"] == 0
    active = [(w[:, k] * err[:, k]**2)[act[:, k]].sum() / max(act[:, k].sum(), 1) for k in range(3)]
    absent = [(err[:, k]**2)[absn[:, k]].mean() if absn[:, k].any() else np.nan for k in range(3)]
    obs = (data["o"] == 1) & act
    r2 = []
    for k in range(3):
        t, q = data["d"][obs[:, k], k], p[obs[:, k], k]
        r2.append(float(1 - np.sum((q - t)**2) / np.sum((t - t.mean())**2)) if len(t) > 1 else np.nan)
    return dict(L_d=float(per.mean()), L_dk=per.tolist(), L_dk_active=[float(x) for x in active],
                L_dk_absent=[float(x) for x in absent], R2_active_observed=r2,
                rows=int(len(p)), observed=data["o"].sum(0).tolist(),
                active=data["b"].sum(0).tolist(), observed_active=obs.sum(0).tolist())


def fit(train, params, jobs):
    w = train["o"] / train["pi"]
    y = np.nan_to_num(train["d"])
    base = ((w * y).sum(0) / w.sum(0)).tolist()
    return XGBRegressor(objective=masked_squared_error(w), multi_strategy="multi_output_tree",
                        base_score=base, tree_method="hist", random_state=42, n_jobs=jobs, **params).fit(train["X"], y)


def predict(model, x):
    return np.asarray(model.predict(x)).reshape(len(x), 3)


def write_report(out, data, pred, metrics, config):
    names = (r"\(\sqrt{L_d}\)", r"\(\sqrt{L_{d,\mathrm{amp}}}\)", r"\(\sqrt{L_{d,\mathrm{phase}}}\)", r"\(\sqrt{L_{d,\mathrm{noise}}}\)")
    parts = ["<!doctype html><meta charset='utf-8'><title>Dataset 2 recovery damage probe</title>",
             "<script defer src='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js'></script>",
             "<style>body{font:16px system-ui;max-width:1500px;margin:auto;padding:2rem;color:#19242a}table{border-collapse:collapse;width:100%;font-size:.9rem;margin:.6rem 0}td,th{border:1px solid #ccd;padding:.45rem;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#eef3f6}strong{font-weight:750}img{max-width:100%}a{color:#175ea5}.note{color:#41515b}</style>",
             "<h1>Dataset 2 recovery damage: combined8, shared XGBoost (interim)</h1>",
             "<p><a href='config.json'>Configuration</a> · <a href='metrics.json'>Metrics JSON</a> · <a href='predictions.npz'>Predictions</a></p>",
             "<h2>Summary of all</h2>",
             f"<p>Interim Dataset 2 snapshot taken while generation continues. Only sources with every planned case committed are used: "
             f"train {config['train']['complete_sources']}/{config['train']['planned_sources']} sources ({len(data['train']['X']):,} rows), "
             f"validation {config['validation']['complete_sources']}/{config['validation']['planned_sources']} sources ({len(data['validation']['X']):,} rows). "
             "Split sources are disjoint. Damage parameters are continuous Latin-hypercube draws (phase, attenuation/amplification, thermal ratio r) "
             "with mixed injections; each mixed row observes only one randomly selected active damage. "
             "The reference row <em>Predict 0</em> is the trivial model that predicts no damage; bold is the lowest value per column.</p>"]
    rows = [("Shared XGBoost", "xgb_shared"), ("Predict 0 (reference)", "zero")]
    def table(key, label, fmt, best):
        cols = [(s, k) for s in ("validation", "train") for k in range(3)]
        v = lambda s, n, k: metrics[s][n][key][k]
        html = [f"<table><tr><th rowspan='2'>{label}</th><th colspan='3'>Validation</th><th colspan='3'>Train</th></tr><tr>" +
                "".join(f"<th>{t}</th>" for _ in range(2) for t in TYPES) + "</tr>"]
        for name, n in rows:
            cells = "".join((f"<td><strong>{fmt(v(s,n,k))}</strong></td>" if v(s, n, k) == best(v(s, m, k) for _, m in rows) else f"<td>{fmt(v(s,n,k))}</td>") for s, k in cols)
            html.append(f"<tr><td>{name}</td>{cells}</tr>")
        return "".join(html) + "</table>"
    count = lambda s: "/".join(str(int(x)) for x in metrics[s]["xgb_shared"]["observed_active"])
    parts.append("<h3>Active rows: damage error where the injection is present</h3>"
                 f"<p>Observed active labels amp/phase/noise: validation {count('validation')}, train {count('train')}. "
                 "This is the headline metric: absent rows carry no damage and are trivially easy, so they are reported separately below.</p>")
    parts.append(table("L_dk_active", "RMSE, active rows", lambda x: f"{np.sqrt(x):.4f}", min))
    parts.append(table("R2_active_observed", "R², active rows", lambda x: f"{x:.3f}", max))
    parts.append(r"<p class='note'>Active RMSE for type \(k\): \(\sqrt{\frac{1}{N_k^{\mathrm{act}}}\sum_{i:\,b_{ik}=1}\frac{o_{ik}}{\pi_{ik}}(\hat d_{ik}-d_{ik})^2}\), in damage units, "
                 "with the inverse-probability mean over all active rows (mixed rows observe one selected damage). "
                 "<em>Predict 0</em> active RMSE is the RMS true damage, i.e. the label scale, so the model/Predict 0 ratio is the typical error relative to the typical damage. "
                 "R² uses the observed active rows relative to their own mean, so Predict 0 is negative.</p>")
    parts.append("<h3>Absent rows: false damage where the injection is not present</h3>"
                 f"<p>Rows with the injection absent (true damage exactly 0), validation {'/'.join(str(int(x)) for x in (data['validation']['b'] == 0).sum(0))}, "
                 f"train {'/'.join(str(int(x)) for x in (data['train']['b'] == 0).sum(0))} rows. Predict 0 is perfect here by construction.</p>")
    parts.append(table("L_dk_absent", "RMSE, absent rows", lambda x: f"{np.sqrt(x):.4f}", min))
    parts.append("<h3>All rows (Dataset 1 format)</h3><p class='note'>Pooled RMSE over active and absent rows, as in the Dataset 1 report. "
                 "It mixes the easy absent zeros with the active damages, so it is shown for comparison only. "
                 r"Hyperparameters were chosen on this validation \(L_d\) over the Dataset 1 grid.</p>")
    cols = [(s, k) for s in ("validation", "train") for k in range(4)]
    val = lambda s, n, k: np.sqrt(metrics[s][n]["L_d"] if k == 0 else metrics[s][n]["L_dk"][k - 1])
    parts.append("<table><tr><th rowspan='2'>Model</th><th colspan='4'>Validation</th><th colspan='4'>Train</th></tr><tr>" +
                 "".join(f"<th>{x}</th>" for _ in range(2) for x in names) + "</tr>")
    for label, n in rows:
        cells = "".join((f"<td><strong>{val(s,n,k):.4f}</strong></td>" if val(s, n, k) <= min(val(s, m, k) for _, m in rows) else f"<td>{val(s,n,k):.4f}</td>") for s, k in cols)
        parts.append(f"<tr><td>{label}</td>{cells}</tr>")
    parts.append(r"</table><p class='note'>Square roots of \(L_d=\frac13\sum_k L_{dk}\) and \(L_{dk}=\frac1N\sum_i \frac{o_{ik}}{\pi_{ik}}(\hat d_{ik}-d_{ik})^2\) over all \(N\) rows, so values are in damage units. metrics.json keeps the squared losses.</p>")
    for s in ("validation", "train"):
        d, p = data[s], pred[s]
        obs = (d["o"] == 1)
        parts.append(f"<h2>{s.title()}</h2><p>{len(d['X']):,} rows from {len(set(d['source_ids']))} sources; active amp/phase/noise: "
                     + "/".join(str(int(x)) for x in d["b"].sum(0)) + "; observed labels per output: "
                     + "/".join(str(int(x)) for x in obs.sum(0)) + ".</p>")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharex=True, sharey=True)
        act = obs & (d["b"] == 1)
        vals = np.concatenate([d["d"][act], p[act]])
        lo, hi = float(vals.min()), float(vals.max()); pad = max((hi - lo) * .03, 1e-8)
        lim = (lo - pad, hi + pad)
        for k, t in enumerate(TYPES):
            ax = axes[k]
            for m in range(8):
                sel = obs[:, k] & (d["b"][:, k] == 1) & (d["mask"] == m)
                if sel.any():
                    ax.scatter(d["d"][sel, k], p[sel, k], s=13, alpha=.6, color=COLORS_MASK[m], label=MASK_LABELS[m])
            ax.plot(lim, lim, "k--", lw=.8); ax.axhline(0, color="#aaa", lw=.5); ax.axvline(0, color="#aaa", lw=.5)
            ax.set(xlim=lim, ylim=lim, title=t.capitalize(), xlabel="True damage"); ax.set_aspect("equal", adjustable="box")
            ax.grid(color="#c8cfd3", alpha=.55, lw=.5)
        axes[0].set_ylabel("Predicted damage (shared XGBoost)")
        hl = {lab: han for ax in axes for han, lab in zip(*ax.get_legend_handles_labels())}
        fig.legend([hl[x] for x in MASK_LABELS if x in hl], [x for x in MASK_LABELS if x in hl], loc="upper center", ncol=8, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(out / f"{s}_diagonal.png", dpi=150, bbox_inches="tight"); plt.close(fig)
        parts.append("<h3>True versus predicted damage, active rows</h3><p class='note'>Observed active labels only; mixed rows show only their selected damage. "
                     "Color is the injection combination; dashed line is exact prediction. Panels share one numeric range.</p>"
                     f"<p><a href='{s}_diagonal.png'><img src='{s}_diagonal.png'></a></p>")
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), sharey=True)
        for k, t in enumerate(TYPES):
            ax = axes[k]
            for m in range(8):
                sel = (d["b"][:, k] == 0) & (d["mask"] == m)
                if sel.any():
                    ax.hist(p[sel, k], bins=np.linspace(lim[0], lim[1], 60), histtype="step", lw=1.3, color=COLORS_MASK[m], label=MASK_LABELS[m])
            ax.axvline(0, color="k", ls="--", lw=.8); ax.set(title=f"{t.capitalize()} absent", xlabel="Predicted damage (true = 0)", yscale="log")
            ax.legend(fontsize=7, frameon=False)
        axes[0].set_ylabel("Rows")
        fig.tight_layout(); fig.savefig(out / f"{s}_absent.png", dpi=150, bbox_inches="tight"); plt.close(fig)
        parts.append("<h3>Predicted damage on absent rows</h3><p class='note'>Rows where that injection is not present, so the true damage is 0; "
                     "anything away from 0 is damage falsely attributed to that type. Lines split rows by which other injections are present (same x-range as above).</p>"
                     f"<p><a href='{s}_absent.png'><img src='{s}_absent.png'></a></p>")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout="constrained")
        table = []
        for k, t in enumerate(TYPES):
            ax = axes[k]
            sel = obs[:, k] & (d["b"][:, k] == 1)
            x = d["params"][sel, k]
            logx = t != "phase"
            edges = np.geomspace(x.min(), x.max(), 9) if logx else np.linspace(x.min(), x.max(), 9)
            idx = np.clip(np.digitize(x, edges[1:-1]), 0, 7)
            centers = np.sqrt(edges[:-1] * edges[1:]) if logx else (edges[:-1] + edges[1:]) / 2
            for series, color, label in ((d["d"][sel, k], "black", "True"), (p[sel, k], "#D55E00", "Shared XGBoost")):
                g = [series[idx == i] for i in range(8)]
                keep = [i for i in range(8) if len(g[i]) > 1]
                mean = np.array([g[i].mean() for i in keep]); sd = np.array([g[i].std(ddof=1) for i in keep])
                ax.plot(centers[keep], mean, color=color, marker="o", ms=3, lw=1.5, label=label)
                ax.fill_between(centers[keep], mean - sd, mean + sd, color=color, alpha=.12)
                if label == "True":
                    table.append((t, [(edges[i], edges[i + 1], len(g[i]), g[i].mean(), g[i].std(ddof=1)) for i in keep]))
            ax.axhline(0, color="#999", lw=.5)
            ax.set(xscale="log" if logx else "linear", title=t.capitalize(), xlabel=PARAM[t][1], ylabel="Mean signed damage"); ax.grid(alpha=.15)
            if logx: ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.5, 1.07))
        fig.savefig(out / f"{s}_mean_by_level.png", dpi=150, bbox_inches="tight"); plt.close(fig)
        parts.append("<h3>Mean by injected parameter</h3><p class='note'>Observed active rows, binned into 8 equal-width (phase) or log-width (A, r) bins of the injected "
                     "parameter of that damage type. Lines are bin means; shaded areas ±1 sample standard deviation. Amplitude pools attenuation (A&lt;1) and amplification (A&gt;1).</p>"
                     f"<p><a href='{s}_mean_by_level.png'><img src='{s}_mean_by_level.png'></a></p>")
        parts.append("<h3>Label scale by injected parameter</h3><table><tr><th>Type</th><th>Bin</th><th>Rows</th><th>Mean true d</th><th>Sample σ</th></tr>")
        for t, items in table:
            for a, b_, n, mean, sd in items:
                parts.append(f"<tr><td>{t}</td><td>{a:.5f} – {b_:.5f}</td><td>{n}</td><td>{mean:.5f}</td><td>{sd:.5f}</td></tr>")
        parts.append("</table>")
    parts.append(r"<p class='note'>Labels: \(d_k=y_{\mathrm{all}}-y_{\mathrm{without}\ k}\), with \(y=e_{TM,2}\) from the raw CLEAN model convolved with the peak-one restoring beam; "
                 "the matched removal case shares the source, antenna, noise seed and other active parameters. Features are combined8 of the full case.</p>")
    (out / "report.html").write_text("\n".join(parts))


def report_only(out):
    """Recompute metrics from saved predictions and rewrite the report; no fitting."""
    z = np.load(out / "predictions.npz", allow_pickle=False)
    keys = ("X", "b", "o", "pi", "mask", "d", "y_all", "source_ids", "row_ids", "params")
    data = {s: {k: z[f"{s}_{k}"] for k in keys} for s in ("train", "validation")}
    pred = {s: z[f"{s}_pred"] for s in data}
    metrics = {s: dict(xgb_shared=damage_loss(pred[s], d), zero=damage_loss(np.zeros_like(pred[s]), d)) for s, d in data.items()}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_report(out, data, pred, metrics, json.loads((out / "config.json").read_text()))
    print("Report:", out / "report.html", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=RUN)
    ap.add_argument("--output", type=Path, default=Path("ml/runs"))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--report-only", type=Path, help="Rewrite report.html from a saved probe directory")
    a = ap.parse_args()
    if a.report_only:
        report_only(a.report_only.resolve()); return
    run = a.run.resolve()
    out = a.output.resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_dataset_v2_damage_probe_xgb_shared")
    out.mkdir(parents=True)
    tr, ctr = complete_split(run / "train")
    va, cva = complete_split(run / "val")
    if set(tr["source_ids"]) & set(va["source_ids"]):
        raise ValueError("Train and validation sources overlap")
    scores = []
    for params in ParameterGrid(GRID):
        scores.append(dict(params=params, validation_L_d=damage_loss(predict(fit(tr, params, a.jobs), va["X"]), va)["L_d"]))
    best = min(scores, key=lambda s: s["validation_L_d"])
    model = fit(tr, best["params"], a.jobs)
    model.save_model(out / "xgb_shared_all.json")
    data = dict(train=tr, validation=va)
    pred = {s: predict(model, d["X"]) for s, d in data.items()}
    metrics = {s: dict(xgb_shared=damage_loss(pred[s], d), zero=damage_loss(np.zeros_like(pred[s]), d)) for s, d in data.items()}
    config = dict(run=str(run), model="xgb_shared (multi_output_tree, IPW masked squared error)", seed=42, grid=GRID,
                  selection=scores, best=best, feature_names=list(COMBINED8_FEATURES),
                  train=ctr, validation=cva, created=datetime.now(timezone.utc).isoformat(),
                  train_sources=sorted(set(tr["source_ids"])), validation_sources=sorted(set(va["source_ids"])))
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(out / "predictions.npz", **{f"{s}_pred": pred[s] for s in pred},
                        **{f"{s}_{k}": v for s, d in data.items() for k, v in d.items()})
    write_report(out, data, pred, metrics, config)
    print("Report:", out / "report.html", flush=True)


if __name__ == "__main__":
    main()
