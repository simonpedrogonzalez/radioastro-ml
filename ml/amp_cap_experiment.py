"""Amplification-cap experiment on matched Dataset v2 (filtering only, no re-simulation).

python -m ml.amp_cap_experiment --train RUN_TRAIN --val RUN_VAL --output OUT [--jobs 8]

Trains xgb_shared (ml/multitask.py, unchanged) on the all20 features three times,
keeping only training rows with amplitude gain A <= cap (caps 2, 3, 8.5468), and
evaluates all three on the same validation subset A <= 2. Rows are dropped, never
modified: every label (d, eC, b) was computed per row by the generator, so removing
other rows does not change it. Rows without amplitude corruption have A = 1 and are
always kept. Everything else (features, grid, selection rule, seed) is identical.
"""
from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

import numpy as np

from ml import multitask as mt
from scripts.preprocessing.schema import atomic_write_json

CAPS = (2., 3., 8.5468)
EVAL_CAP = 2.
VIEW = "all20"
KINDS = ("amp", "phase", "noise")


def keep_amp(data, cap):
    """Boolean rows with gain A <= cap (params[:, 0] is A; A = 1 when amp is absent)."""
    A = np.asarray(data["params"][:, 0], float)
    if not np.all(A[data["b"][:, 0] == 0] == 1.):
        raise ValueError("Rows without amplitude corruption must have A = 1")
    return A <= cap


def rmse_r2(pred, truth):
    if len(truth) == 0:
        return None, None
    err = pred - truth
    rmse = float(np.sqrt(np.mean(err**2)))
    var = float(np.sum((truth - truth.mean())**2))
    return rmse, (None if var <= 1e-15 else float(1 - np.sum(err**2) / var))


def amp_slices(pred, data):
    """d_amp scores on the evaluation subset, split by amplitude branch."""
    A, b, d = data["params"][:, 0], data["b"][:, 0].astype(bool), data["d"][:, 0]
    out = {}
    for name, keep in (("amplification (1 < A <= 2)", b & (A > 1)), ("attenuation (A < 1)", b & (A < 1)),
                       ("all active", b), ("absent (true d = 0)", ~b)):
        rmse, r2 = rmse_r2(pred[keep, 0], d[keep])
        out[name] = dict(rows=int(keep.sum()), rmse=rmse, r2=None if name.startswith("absent") else r2,
                         zero_rmse=rmse_r2(np.zeros(keep.sum()), d[keep])[0])
    return out


def fmt(x, digits=4):
    return "—" if x is None else f"{x:.{digits}f}"


def write_report(output, summary, val, predictions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assets = output / "report_assets"
    assets.mkdir(exist_ok=True)
    A, b, d = val["params"][:, 0], val["b"][:, 0].astype(bool), val["d"][:, 0]
    groups = (("Amplification (1 < A ≤ 2)", b & (A > 1), "#eb6834"), ("Attenuation (A < 1)", b & (A < 1), "#2a78d6"),
              ("Amp absent", ~b, "#8a8a85"))
    preds = [predictions[str(c)][:, 0] for c in CAPS]
    lo = float(min(d.min(), *(p.min() for p in preds))); hi = float(max(d.max(), *(p.max() for p in preds)))
    pad = .04 * (hi - lo or 1.); lo, hi = lo - pad, hi + pad
    fig, axes = plt.subplots(1, len(CAPS), figsize=(4.2 * len(CAPS), 4.2), sharex=True, sharey=True)
    for ax, cap, p in zip(axes, CAPS, preds):
        for label, keep, color in groups:
            ax.scatter(d[keep], p[keep], s=8, alpha=.5, c=color, linewidths=0, label=label)
        ax.plot([lo, hi], [lo, hi], color="#444", lw=.8)
        ax.set(xlim=(lo, hi), ylim=(lo, hi), title=f"Trained with A ≤ {cap:g}", xlabel="True d_amp")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Predicted d_amp")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, -.04))
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(assets / "d_amp_diagonal.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    caps = [str(c) for c in CAPS]
    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>Amplification cap experiment</title>",
             "<style>body{font:15px system-ui;max-width:1200px;margin:auto;padding:1.5rem;color:#1b2932}"
             "table{border-collapse:collapse;margin:.5rem 0 1.5rem}th,td{border:1px solid #ccd;padding:.35rem .6rem;text-align:right}"
             "th:first-child,td:first-child{text-align:left}img{max-width:100%}</style></head><body>",
             "<h1>Does strong amplification in training help or hurt moderate amplification?</h1>",
             f"<p>xgb_shared on all20 features, trained three times on training rows with gain A ≤ cap, and evaluated on the same "
             f"validation rows with A ≤ {EVAL_CAP:g} ({summary['val_rows']} rows, {summary['val_sources']} sources). Rows were only filtered, "
             "no label was changed. Rows without amplitude corruption (A = 1) are kept in every version. Hyperparameters are selected "
             "on this same validation subset for each version (the multitask rule), so the scores are comparable but slightly optimistic. "
             "RMSE is in damage units. Active rows have that corruption injected, absent rows have true damage exactly 0.</p>",
             "<p><a href='results.json'>All metrics and selected settings</a></p>",
             "<h2>Training sets</h2><table><tr><th>Version</th><th>Training rows</th><th>Amp-active rows</th><th>Max A</th><th>Selected settings</th></tr>"]
    for c in caps:
        t = summary["versions"][c]
        parts.append(f"<tr><td>A ≤ {float(c):g}</td><td>{t['train_rows']}</td><td>{t['train_amp_active']}</td>"
                     f"<td>{fmt(t['train_max_A'], 3)}</td><td>{html.escape(str(t['params']))}</td></tr>")
    parts.append("</table><h2>Amplitude damage on the evaluation subset</h2>"
                 "<table><tr><th>Rows</th><th>n</th>" + "".join(f"<th>A ≤ {float(c):g}<br>RMSE</th><th>A ≤ {float(c):g}<br>R²</th>" for c in caps)
                 + "<th>Predict 0<br>RMSE</th></tr>")
    for name in summary["versions"][caps[0]]["amp_slices"]:
        cells = []
        for c in caps:
            s = summary["versions"][c]["amp_slices"][name]
            cells.append(f"<td>{fmt(s['rmse'])}</td><td>{fmt(s['r2'], 3)}</td>")
        s0 = summary["versions"][caps[0]]["amp_slices"][name]
        parts.append(f"<tr><td>{html.escape(name)}</td><td>{s0['rows']}</td>{''.join(cells)}<td>{fmt(s0['zero_rmse'])}</td></tr>")
    parts.append("</table><h2>All outputs on the evaluation subset</h2><table><tr><th>Output</th><th>Metric</th>"
                 + "".join(f"<th>A ≤ {float(c):g}</th>" for c in caps) + "</tr>")
    rows = []
    for k in KINDS:
        rows += [(f"d_{k}", "active RMSE", lambda m, k=k: m[k]["damage_active"]["rmse"], 4),
                 (f"d_{k}", "active R²", lambda m, k=k: m[k]["damage_active"].get("r2"), 3),
                 (f"d_{k}", "absent RMSE", lambda m, k=k: m[k]["damage_absent"]["rmse"], 4)]
    rows += [("eC (total error)", "RMSE", lambda m: m["y_all"]["rmse"], 4), ("eC (total error)", "R²", lambda m: m["y_all"]["r2"], 3)]
    rows += [(f"presence_{k}", "average precision", lambda m, k=k: m[k]["average_precision"], 3) for k in KINDS]
    rows += [("d_amp", "false amp when only phase present, RMSE", lambda m: m["amp"]["leakage"]["phase"]["rmse"], 4)]
    for output_name, metric, get, digits in rows:
        cells = "".join(f"<td>{fmt(get(summary['versions'][c]['metrics']), digits)}</td>" for c in caps)
        parts.append(f"<tr><td>{output_name}</td><td>{metric}</td>{cells}</tr>")
    parts.append("</table><h2>Amplitude damage diagonals</h2><p>Evaluation subset only. Same axes in all panels.</p>"
                 "<img src='report_assets/d_amp_diagonal.png' alt='True versus predicted d_amp for the three training caps'></body></html>")
    (output / "report.html").write_text("".join(parts))


def run(train_root, val_root, output, jobs=8):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise ValueError("Output already exists; choose a fresh directory")
    train_all, val_all = mt.load_split(train_root, "train", VIEW), mt.load_split(val_root, "val", VIEW)
    if set(train_all["source_ids"]) & set(val_all["source_ids"]):
        raise ValueError("Train/validation sources overlap")
    val = mt.subset(val_all, np.flatnonzero(keep_amp(val_all, EVAL_CAP)))
    summary = dict(model="xgb_shared", feature_view=VIEW, caps=list(CAPS), eval_cap=EVAL_CAP,
                   val_rows=len(val["X"]), val_sources=len(set(val["source_ids"])),
                   val_row_ids=val["row_ids"].tolist(),
                   selection="validation J on the A <= 2 evaluation subset (same rule as ml/multitask.py)", versions={})
    predictions = {}
    for cap in CAPS:
        train = mt.subset(train_all, np.flatnonzero(keep_amp(train_all, cap)))
        print(f"Cap {cap:g}: {len(train['X'])} training rows", flush=True)
        fitted = mt._fit_tree_family(train, val, "xgb_shared", jobs)
        raw = mt.predict_xgb(fitted["model"], val["X"])
        pred = mt.decode(raw, fitted["scale"])
        predictions[str(cap)] = pred
        fitted["model"].save_model(output / f"xgb_shared_cap{cap:g}.json")
        summary["versions"][str(cap)] = dict(
            train_rows=len(train["X"]), train_sources=len(set(train["source_ids"])),
            train_amp_active=int(train["b"][:, 0].sum()), train_max_A=float(train["params"][:, 0].max()),
            params=fitted["params"], scale=fitted["scale"].tolist(), candidates=fitted["candidates"],
            losses=mt.raw_losses(raw, val, fitted["scale"]), metrics=mt.task_metrics(pred, val),
            amp_slices=amp_slices(pred, val))
    np.savez_compressed(output / "predictions_val.npz", row_ids=val["row_ids"],
                        **{f"cap_{c}": p for c, p in predictions.items()})
    atomic_write_json(output / "results.json", summary)
    write_report(output, summary, val, predictions)
    print("Report:", output / "report.html", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=8)
    a = p.parse_args()
    run(a.train, a.val, a.output, a.jobs)


if __name__ == "__main__":
    main()
