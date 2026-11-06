"""Where do the matched-v2 multitask errors come from? Read-only analysis of saved predictions.

python -m ml.multitask_error_analysis --run ML_RUN --val RUN_VAL [--models xgb_shared xgb_xgb]

ML_RUN is an ml/multitask.py output folder (results.json, predictions_val_<model>.npz).
Writes ML_RUN/error_analysis/report.html (+ analysis.json). Nothing is refitted.
For each corruption type it splits presence (b) and damage (d) errors by band,
severity, injection mask and source, lists the worst rows, and checks how much a
single input feature alone already separates present from absent (shortcut check).
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ml import multitask as mt
from scripts.dataset_v2_matched import WEAK_RANGES
from scripts.preprocessing.schema import atomic_write_json

KINDS = ("amp", "phase", "noise")
MASK_NAMES = {m: "+".join(k for j, k in enumerate(KINDS) if m >> j & 1) or "control" for m in range(8)}
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def load(run, val_root, models):
    val = mt.load_split(val_root, "val", "all20")
    plan = json.loads((val_root / "plan.json").read_text())
    snr = {p["id"]: p["nominal_source_snr"] for p in plan["prepared_sources"]}
    val["snr0"] = np.asarray([snr[str(s)] for s in val["source_ids"]], float)
    val["r"] = val["params"][:, 2] / val["snr0"]
    preds = {}
    for name in models:
        with np.load(run / f"predictions_val_{name}.npz", allow_pickle=False) as z:
            if not np.array_equal(z["row_ids"], val["row_ids"]):
                raise ValueError(f"{name}: prediction rows do not match the validation dataset")
            preds[name] = z["decoded"]
    return val, preds


def descriptors(val):
    """Band, severity and a readable parameter for each type (active rows only meaningful)."""
    A, phi, r = val["params"][:, 0], val["params"][:, 1], val["r"]
    amp_weak = ((A >= WEAK_RANGES["amp_attenuation"][0]) & (A < 1)) | ((A > 1) & (A <= WEAK_RANGES["amp_amplification"][1]))
    amp_band = np.where(A < 1, np.where(amp_weak, "attenuation weak", "attenuation strong"),
                        np.where(amp_weak, "amplification weak", "amplification strong"))
    phase_band = np.where(np.abs(phi) <= WEAK_RANGES["phase_positive"][1], "weak", "strong")
    return dict(amp=dict(band=amp_band, severity=np.abs(np.log(A)), label="|ln A|"),
                phase=dict(band=phase_band, severity=np.abs(phi), label="|φ| (deg)"),
                noise=dict(band=np.full(len(A), "single range"), severity=r, label="r = a/SNR₀"))


def stats(pred, val, k, keep):
    """Presence and damage scores for type k on rows `keep`."""
    b = val["b"][:, k].astype(bool)
    prob, dp, d = pred[:, 4 + k], pred[:, k], val["d"][:, k]
    act, absn = keep & b, keep & ~b
    se = (dp - d)**2
    out = dict(rows=int(keep.sum()), active=int(act.sum()), absent=int(absn.sum()))
    out["recall"] = float(np.mean(prob[act] >= .5)) if act.any() else None
    out["false_alarm"] = float(np.mean(prob[absn] >= .5)) if absn.any() else None
    out["mean_prob_active"] = float(prob[act].mean()) if act.any() else None
    out["d_rmse_active"] = float(np.sqrt(se[act].mean())) if act.any() else None
    out["d_zero_rmse_active"] = float(np.sqrt(np.mean(d[act]**2))) if act.any() else None
    var = float(np.sum((d[act] - d[act].mean())**2)) if act.any() else 0.
    out["d_r2_active"] = float(1 - se[act].sum() / var) if var > 1e-15 else None
    out["d_rmse_absent"] = float(np.sqrt(se[absn].mean())) if absn.any() else None
    out["se_share"] = float(se[keep].sum() / se.sum()) if se.sum() > 0 else None
    out["missed_share"] = float(np.sum(act & (prob < .5)) / max(1, np.sum(b & (prob < .5))))
    return out


def single_feature_shortcuts(val, k, top=3):
    """Best single features for presence (AP/AUC, either sign) and for active damage (|Spearman|)."""
    from scipy.stats import spearmanr
    b = val["b"][:, k].astype(bool)
    X, names = val["X"], list(map(str, val["feature_names"]))
    rows = []
    for j, name in enumerate(names):
        x = X[:, j]
        ok = np.isfinite(x)
        if ok.sum() < 10 or b[ok].all() or (~b[ok]).all():
            continue
        auc = roc_auc_score(b[ok], x[ok])
        sign = 1 if auc >= .5 else -1
        act = ok & b
        rho = spearmanr(x[act], val["d"][act, k]).statistic if act.sum() > 2 and np.ptp(x[act]) > 0 else np.nan
        rows.append(dict(feature=name, auc=float(max(auc, 1 - auc)), ap=float(average_precision_score(b[ok], sign * x[ok])),
                         spearman_d=None if not np.isfinite(rho) else float(rho)))
    by_ap = sorted(rows, key=lambda r: -r["ap"])[:top]
    by_rho = sorted([r for r in rows if r["spearman_d"] is not None], key=lambda r: -abs(r["spearman_d"]))[:top]
    return dict(presence=by_ap, damage=by_rho, prevalence=float(b.mean()))


def analyse(val, preds):
    desc = descriptors(val)
    mask = val["b"].dot(1 << np.arange(3))
    sources = np.asarray(list(map(str, val["source_ids"])))
    out = {}
    for name, pred in preds.items():
        out[name] = {}
        for k, kind in enumerate(KINDS):
            b = val["b"][:, k].astype(bool)
            D = desc[kind]
            res = dict(overall=stats(pred, val, k, np.ones(len(b), bool)))
            res["by_band"] = {band: stats(pred, val, k, b & (D["band"] == band)) for band in sorted(set(D["band"][b]))}
            edges = np.quantile(D["severity"][b], [0, .25, .5, .75, 1])
            res["severity_label"] = D["label"]
            res["by_severity"] = {}
            for q in range(4):
                lo, hi = edges[q], edges[q + 1]
                keep = b & (D["severity"] >= lo) & ((D["severity"] <= hi) if q == 3 else (D["severity"] < hi))
                res["by_severity"][f"Q{q+1} [{lo:.4f}, {hi:.4f}]"] = stats(pred, val, k, keep)
            res["by_mask"] = {MASK_NAMES[m]: stats(pred, val, k, mask == m) for m in range(8)}
            per_source = {s: stats(pred, val, k, sources == s) for s in sorted(set(sources))}
            for s, v in per_source.items():
                v["snr0"] = float(val["snr0"][sources == s][0])
            res["by_source_worst"] = dict(sorted(per_source.items(), key=lambda kv: -(kv[1]["se_share"] or 0))[:8])
            res["sources_for_half_error"] = int(np.searchsorted(np.cumsum(sorted(
                (v["se_share"] or 0 for v in per_source.values()), reverse=True)), .5) + 1)
            res["source_count"] = len(per_source)
            err = np.abs(pred[:, k] - val["d"][:, k])
            worst = np.argsort(-err)[:15]
            res["worst_rows"] = [dict(row=str(val["row_ids"][i]), source=sources[i], mask=MASK_NAMES[int(mask[i])],
                                      A=float(val["params"][i, 0]), phi=float(val["params"][i, 1]), r=float(val["r"][i]),
                                      d_true=float(val["d"][i, k]), d_pred=float(pred[i, k]), prob=float(pred[i, 4 + k]))
                                 for i in worst]
            missed = b & (pred[:, 4 + k] < .5)
            res["missed_by_band"] = {band: int(np.sum(missed & (D["band"] == band))) for band in sorted(set(D["band"][b]))}
            res["negative_d_active"] = stats(pred, val, k, b & (val["d"][:, k] < 0))
            out[name][kind] = res
    shortcuts = {kind: single_feature_shortcuts(val, k) for k, kind in enumerate(KINDS)}
    j = list(map(str, val["feature_names"])).index("sigma_sNA_units")
    noise_check = {}
    for label, keep in (("noise absent", val["b"][:, 2] == 0), ("noise present", val["b"][:, 2] == 1)):
        x = val["X"][keep, j]
        noise_check[label] = dict(rows=int(keep.sum()), **{f"q{int(q*100)}": float(np.quantile(x, q)) for q in (0, .05, .5, .95, 1)})
    present = val["b"][:, 2] == 1
    noise_check["injected_a_min_by_source"] = {f"SNR₀ {snr:.0f}": float(val["params"][present & (val["snr0"] == snr), 2].min())
                                               for snr in sorted(set(val["snr0"][present]))[:3] + sorted(set(val["snr0"][present]))[-3:]}
    noise_check["injected_a_overall_min"] = float(val["params"][present, 2].min())
    return out, shortcuts, noise_check, desc


def fmt(x, digits=3):
    return "—" if x is None else f"{x:.{digits}f}"


def table(rows, columns):
    head = "".join(f"<th>{html.escape(c)}</th>" for c, _ in columns)
    body = "".join("<tr>" + "".join(f"<td>{f(r)}</td>" for _, f in columns) + "</tr>" for r in rows)
    return f"<div class='scroll'><table><tr>{head}</tr>{body}</table></div>"


STAT_COLUMNS = [("rows", lambda r: r[1]["rows"]), ("active", lambda r: r[1]["active"]),
                ("presence recall @0.5", lambda r: fmt(r[1]["recall"])), ("false alarm @0.5", lambda r: fmt(r[1]["false_alarm"])),
                ("d RMSE active", lambda r: fmt(r[1]["d_rmse_active"], 4)), ("predict-0 RMSE", lambda r: fmt(r[1]["d_zero_rmse_active"], 4)),
                ("d R² active", lambda r: fmt(r[1]["d_r2_active"])), ("d RMSE absent", lambda r: fmt(r[1]["d_rmse_absent"], 4)),
                ("share of d squared error", lambda r: fmt(r[1]["se_share"]))]


def plots(output, val, preds, desc, model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    pred = preds[model]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for k, kind in enumerate(KINDS):
        b = val["b"][:, k].astype(bool)
        sev = desc[kind]["severity"]
        for row, (y, ylabel) in enumerate(((pred[:, 4 + k], "predicted presence probability"),
                                           (pred[:, k] - val["d"][:, k], "d error (predicted − true)"))):
            ax = axes[row, k]
            bands = sorted(set(desc[kind]["band"][b]))
            for i, band in enumerate(bands):
                keep = b & (desc[kind]["band"] == band)
                ax.scatter(sev[keep], y[keep], s=6, alpha=.4, c=SERIES[i % 4], linewidths=0, label=band)
            if kind in ("amp", "noise"):
                ax.set_xscale("log"); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            if row == 0: ax.set_ylim(-.03, 1.03)
            ax.axhline(.5 if row == 0 else 0, color="#444", lw=.8)
            ax.set(xlabel=desc[kind]["label"], ylabel=ylabel, title=f"{kind}: active rows")
            ax.grid(alpha=.2)
            if row == 0: ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(output / "severity.png", dpi=130); plt.close(fig)

    j = list(map(str, val["feature_names"])).index("sigma_sNA_units")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = val["X"][:, j]
    bins = np.geomspace(max(np.nanmin(x), 1e-3), np.nanmax(x), 60)
    for i, (label, keep) in enumerate((("noise absent", val["b"][:, 2] == 0), ("noise present", val["b"][:, 2] == 1))):
        ax.hist(x[keep], bins=bins, alpha=.6, color=SERIES[i], label=label)
    ax.set(xscale="log", xlabel="sigma_sNA_units (residual MAD / nominal thermal RMS)", ylabel="validation rows")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(output / "noise_shortcut.png", dpi=130); plt.close(fig)


def write_report(output, analysis, shortcuts, noise_check, models):
    css = ("body{font:15px system-ui;max-width:1300px;margin:auto;padding:1.5rem;color:#1b2932}"
           "table{border-collapse:collapse;margin:.4rem 0 1.2rem}th,td{border:1px solid #ccd;padding:.3rem .5rem;text-align:right}"
           "th:first-child,td:first-child{text-align:left}.scroll{overflow-x:auto}img{max-width:100%}")
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>Multitask error analysis</title><style>{css}</style></head><body>",
             "<h1>Where the validation errors come from</h1>",
             "<p>Saved validation predictions only, no refitting. Presence uses a 0.5 threshold. <em>Share of d squared error</em> is the "
             "fraction of that model's total squared damage error (all rows) that falls in the slice, so it shows what drives the RMSE. "
             "Bands follow the generator: weak amp is A in [0.664, 0.98] or [1.02, 1.431], weak phase is |φ| ≤ 23.8°. "
             "Thermal noise has no weak band.</p>",
             "<h2>Single-feature shortcut check</h2><p>How well one input feature alone separates present from absent (best sign), "
             "and its rank correlation with the active damage. A single-feature AP close to the model's AP means the model mostly uses that cue.</p>"]
    for kind, s in shortcuts.items():
        parts.append(f"<h3>{kind} (prevalence {s['prevalence']:.3f}, the AP of a constant guess)</h3>")
        parts.append(table(s["presence"], [("best features for presence", lambda r: r["feature"]), ("AP", lambda r: fmt(r["ap"])),
                                           ("AUC", lambda r: fmt(r["auc"])), ("Spearman with active d", lambda r: fmt(r["spearman_d"]))]))
        parts.append(table(s["damage"], [("best features for active d", lambda r: r["feature"]), ("Spearman", lambda r: fmt(r["spearman_d"])),
                                         ("presence AP", lambda r: fmt(r["ap"]))]))
    parts.append("<h2>Is the noise presence score genuine?</h2><p>sigma_sNA_units is the residual MAD divided by the nominal thermal RMS. "
                 "Absent noise means the nominal thermal noise (a = 1), not zero noise. Present noise is injected at r = a/SNR₀ ≥ 0.0097, "
                 "so a ≥ 0.0097 × SNR₀.</p>")
    parts.append(table([(k, v) for k, v in noise_check.items() if isinstance(v, dict) and "rows" in v],
                       [("rows", lambda r: f"{r[0]} ({r[1]['rows']})")] +
                       [(f"quantile {q}", lambda r, q=q: fmt(r[1][f"q{q}"], 2)) for q in (0, 5, 50, 95, 100)]))
    parts.append(f"<p>Smallest injected a among present-noise rows: {noise_check['injected_a_overall_min']:.2f}. By source (lowest and highest SNR₀): "
                 + ", ".join(f"{k}: a ≥ {v:.1f}" for k, v in noise_check["injected_a_min_by_source"].items()) + ".</p>"
                 "<img src='noise_shortcut.png' alt='sigma_sNA_units histogram for noise absent and present'>")
    parts.append(f"<h2>Severity view ({html.escape(models[0])})</h2><p>Active rows only. Top: predicted presence probability "
                 "(below the line = missed). Bottom: damage error.</p><img src='severity.png' alt='presence probability and damage error against severity'>")
    for name in models:
        parts.append(f"<h2>Model {html.escape(name)}</h2>")
        for kind in KINDS:
            r = analysis[name][kind]
            o = r["overall"]
            parts.append(f"<h3>{kind}</h3><p>Overall: recall {fmt(o['recall'])}, false alarm {fmt(o['false_alarm'])}, active d RMSE "
                         f"{fmt(o['d_rmse_active'], 4)} (R² {fmt(o['d_r2_active'])}), absent d RMSE {fmt(o['d_rmse_absent'], 4)}. "
                         f"Missed detections by band: {', '.join(f'{k} {v}' for k, v in r['missed_by_band'].items())}. "
                         f"{r['sources_for_half_error']} of {r['source_count']} sources hold half the squared damage error.</p>")
            for title, block in (("By band (active rows)", r["by_band"]), (f"By severity quartile of {r['severity_label']} (active rows)", r["by_severity"]),
                                 ("By injection mask (all rows; absent rows show false alarms)", r["by_mask"]),
                                 ("Negative true damage (active rows with d < 0)", {"d < 0": r["negative_d_active"]})):
                parts.append(f"<h4>{html.escape(title)}</h4>" + table(list(block.items()), [("slice", lambda x: html.escape(x[0]))] + STAT_COLUMNS))
            parts.append("<h4>Sources with the largest share of squared damage error</h4>" + table(
                list(r["by_source_worst"].items()), [("source", lambda x: html.escape(x[0])), ("SNR₀", lambda x: f"{x[1]['snr0']:.0f}")] + STAT_COLUMNS))
            parts.append("<h4>15 largest absolute damage errors</h4>" + table(r["worst_rows"], [
                ("row", lambda x: html.escape(x["row"])), ("mask", lambda x: x["mask"]), ("A", lambda x: fmt(x["A"], 4)),
                ("φ (deg)", lambda x: fmt(x["phi"], 2)), ("r", lambda x: fmt(x["r"], 4)), ("true d", lambda x: fmt(x["d_true"], 4)),
                ("predicted d", lambda x: fmt(x["d_pred"], 4)), ("presence prob", lambda x: fmt(x["prob"]))]))
    parts.append("</body></html>")
    (output / "report.html").write_text("".join(parts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--models", nargs="+", default=["xgb_shared", "xgb_xgb"])
    a = p.parse_args()
    output = a.run / "error_analysis"
    output.mkdir(exist_ok=True)
    val, preds = load(a.run, a.val, a.models)
    analysis, shortcuts, noise_check, desc = analyse(val, preds)
    atomic_write_json(output / "analysis.json", dict(models=a.models, analysis=analysis, shortcuts=shortcuts, noise_check=noise_check))
    plots(output, val, preds, desc, a.models[0])
    write_report(output, analysis, shortcuts, noise_check, a.models)
    print("Error analysis:", output / "report.html", flush=True)


if __name__ == "__main__":
    main()
