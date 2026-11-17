"""Frozen Dataset v2 multitask models on VLA pipeline QA cases: final (after-pipeline) eC regression, v5 report.

[USER] design doc "Requested v5 evaluation setup" (2026-11-16), "Model result report": the frozen models see the
image features of the evaluation AFTER the recipe; the target is the measured total eC after the recipe and the
prediction is the model's own total-eC output (never a sum of per-kind damages, the before eC, the injected class or
the remaining gain damage). Cases are labelled with the same v5 classifiers as the pipeline report
(scripts/vla_pipeline_qa.py via scripts/vla_pipeline_qa_compare.load).

Per run configuration and model:
  Section 1, by Pipeline Result: Table 1 Success / Partial success / Failure (all cases), Table 2 within each parent
    group, Table 3 within each No improvement expected subgroup;
  Section 2, by Recovery: the same three tables with Solved / Improved / Similar / Degraded.
Columns: total cases, valid prediction/target pairs (coverage), after-eC R² and RMSE computed from the pooled pairs of
each displayed subset. Accounting states (pending, error, unavailable) stay visible as their own rows.

After features: the stored after-recipe evaluation features; for a reused evaluation (the QA run established that
nothing was applied and no flags were added, so the input was unchanged) the archived features of the case, and only
when its stored after eC equals its before eC. Inference only: no CASA, no refit, no threshold selection; run folders
are only read.

  bin/run ml ml/pipeline_qa_models.py --model ml/runs/<multitask run> --output ml/runs/<name> QA_RUN [QA_RUN ...]
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from ml import multitask as mt
from scripts.preprocessing.schema import atomic_write_json
from scripts.vla_pipeline_qa import EXCEPTION, GROUPS, RECOVERY, RESULTS, SUBGROUPS, V5_VERSION

DEFAULT_MODELS = ("xgb_xgb", "xgb_mlp")
SECTIONS = (("result", "Pipeline Result", RESULTS), ("recovery", "Recovery", RECOVERY))
ACCOUNTING = ("unavailable", "error", "unknown", "pending")


def feature_columns(summary):
    """Column names of the model's feature view, in model order."""
    from ml.logreg import COMBINED8_FEATURES
    from scripts.dataset_v2_matched_extract import CLEAN2, DIRTY6, MODEL6
    view = summary.get("feature_view", "combined8")
    names = (tuple(COMBINED8_FEATURES) + (DIRTY6 if view in ("combined8_dirty6", "all20", "all22") else ())
             + (MODEL6 if view in ("combined8_model6", "all20", "all22") else ()) + (CLEAN2 if view == "all22" else ()))
    if list(names) != list(summary["feature_names"]):
        raise ValueError(f"Model feature names do not match feature view {view}")
    return names


def after_features(source_run, case_id, ev):
    """(feature names, after-recipe feature row, source, reason-if-missing) of one measured evaluation."""
    from scripts.vla_pipeline_qa import archived_features
    if ev.get("status") != "measured":
        return None, None, None, f"no measured evaluation ({ev.get('status') or 'missing'})"
    names = ev.get("feature_names")
    if not names:
        return None, None, None, "no stored feature names"
    if ev.get("evaluation_reused"):
        before, after = (ev.get("before") or {}).get("eC"), (ev.get("after") or {}).get("eC")
        if before is None or before != after:
            return None, None, None, "reused evaluation without identical before/after eC (unchanged input not verified)"
        # QA results written before 2026-11-12 stored 20 values under 22 names: read the full archived row instead
        x, source = archived_features(source_run, case_id), f"archived case (input unchanged: {ev.get('reason')})"
    else:
        x, source = (ev.get("after") or {}).get("X"), "after-recipe evaluation"
        if x is None:
            return None, None, None, "no stored after features"
    if len(x) != len(names):
        return None, None, None, f"{len(x)} after features stored for {len(names)} names"
    return names, x, source, None


def collect(runs):
    """One row per (run, recipe, case) with the v5 labels, the measured after eC and the after features."""
    from scripts.vla_pipeline_qa import safe
    from scripts.vla_pipeline_qa_compare import load
    rows = []
    for run in runs:
        run = Path(run).resolve()
        source = Path(json.loads((run / "plan.json").read_text())["source_run"])
        for config in load(run):
            for c in config["cases"]:
                row = dict(run=config["run"], run_id=config["run_id"], config=config["config"], recipe=config["recipe"], case=c["case"],
                           label=c["label"], mask=c["mask"], group=c["group"], subgroup=c["subgroup"],
                           result=c["result"], recovery=c["recovery"], application=c["application"],
                           after=c["eC_after"], names=None, X=None, after_source=None, missing=None)
                path = run / "results" / f"{c['case']}__{safe(config['recipe'])}.json"
                if not path.exists():
                    row["missing"] = f"no result record ({c['application']})"
                else:
                    ev = json.loads(path.read_text()).get("evaluation", {})
                    names, x, src, why = after_features(source, c["case"], ev)
                    row.update(names=names, X=x, after_source=src, missing=why)
                rows.append(row)
    return rows


def matrix(rows, columns):
    """After-feature matrix in model column order; rows without (all model) features stay NaN with their reason."""
    out = np.full((len(rows), len(columns)), np.nan)
    for i, r in enumerate(rows):
        if r["X"] is None:
            continue
        lookup = dict(zip(r["names"], r["X"]))
        missing = set(columns) - set(lookup)
        if missing:
            r.update(X=None, missing=f"stored features lack {', '.join(sorted(missing))}")
            continue
        out[i] = [np.nan if lookup[n] is None else lookup[n] for n in columns]
    return out


def predict(model_dir, names, x):
    """Decoded total-eC predictions of each requested frozen model (stacked ones get the expert matrix). Rows with
    missing features give NaN."""
    eC = mt.OUTPUTS.index("eC")
    ok = np.isfinite(x).all(axis=1)
    needed = set(names) | ({"xgb_independent", "xgb_shared"} if set(names) & set(mt.STACKED) else set())
    fitted = {n: mt.load_frozen(model_dir, n) for n in needed}
    out = {n: np.full(len(x), np.nan) for n in names}
    if not ok.any():
        return out
    xs = x[ok]
    stage = (np.column_stack((xs, mt._expert_matrix(fitted["xgb_independent"], fitted["xgb_shared"], xs)))
             if set(names) & set(mt.STACKED) else None)
    for n in names:
        xi = stage if n in mt.STACKED else xs
        raw = mt.predict_mlp(fitted[n], xi) if n in mt.MLP_NAMES else mt.predict_xgb(fitted[n]["model"], xi)
        out[n][ok] = mt.decode(raw, fitted[n]["scale"])[:, eC]
    return out


def regression(prediction, target):
    """After-eC R² and RMSE from the finite pairs of one subset. RMSE needs one pair; R² needs two and a nonconstant
    target (negative R² kept)."""
    prediction, target = np.asarray(prediction, float), np.asarray(target, float)
    keep = np.isfinite(prediction) & np.isfinite(target)
    p, t = prediction[keep], target[keep]
    n = int(keep.sum())
    rmse = float(np.sqrt(np.mean((p - t) ** 2))) if n else None
    r2 = (float(1 - np.sum((p - t) ** 2) / np.sum((t - t.mean()) ** 2))
          if n >= 2 and np.ptp(t) > 0 else None)
    return dict(pairs=n, rmse=rmse, r2=r2)


def subset(rows, prediction, idx):
    """Scores and coverage of the rows idx (prediction aligned with rows)."""
    after = np.asarray([np.nan if rows[i]["after"] is None else rows[i]["after"] for i in idx], float)
    out = regression(prediction[idx], after) if idx else dict(pairs=0, rmse=None, r2=None)
    valid = [i for i, a in zip(idx, after) if np.isfinite(prediction[i]) and np.isfinite(a)]
    missing = {}
    for i in set(idx) - set(valid):
        why = rows[i]["missing"] or "no measured after eC or prediction"
        missing[why] = missing.get(why, 0) + 1
    return dict(out, total=len(idx), missing=missing,
                reused_pairs=sum((rows[i]["after_source"] or "").startswith("archived") for i in valid))


def section(rows, prediction, key, cats):
    """Tables 1-3 of one outcome dimension, each a list of (group, [(outcome, scores), ...]); 'all' = whole group."""
    values = list(cats) + [a for a in ACCOUNTING if any(r[key] == a for r in rows)]

    def split(members, no_partial=False):
        out = [("all", subset(rows, prediction, members))]
        for v in values:
            idx = [i for i in members if rows[i][key] == v]
            if not (no_partial and v == "partial_success") and (idx or v in cats):
                out.append((v, subset(rows, prediction, idx)))
        return out

    everything = list(range(len(rows)))
    groups = list(GROUPS) + [g for g in (EXCEPTION, "unavailable") if any(r["group"] == g for r in rows)]
    return dict(
        table1=[("", split(everything))],
        table2=[(g, split([i for i in everything if rows[i]["group"] == g], g != "improvement_expected"))
                for g in groups],
        table3=[(s, split([i for i in everything if rows[i]["subgroup"] == s], True)) for s in SUBGROUPS])


def plot(output, name, rows, prediction):
    """Predicted against measured after eC, colored by Recovery."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scripts.vla_pipeline_qa_compare import COLORS, NAMES
    after = np.asarray([np.nan if r["after"] is None else r["after"] for r in rows], float)
    rec = np.asarray([r["recovery"] for r in rows])
    keep = np.isfinite(after) & np.isfinite(prediction)
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    for k in RECOVERY:
        sel = keep & (rec == k)
        if sel.any():
            ax.scatter(after[sel], prediction[sel], s=8, alpha=.5, c=COLORS[k], linewidths=0,
                       label=f"{NAMES[k]} ({sel.sum()})")
    if keep.any():
        lo = float(min(after[keep].min(), prediction[keep].min()))
        hi = float(max(after[keep].max(), prediction[keep].max()))
        ax.plot([lo, hi], [lo, hi], color="#444", lw=.8)
    ax.set_xlabel("measured eC after the recipe"); ax.set_ylabel("predicted eC (after features)")
    ax.set_title(name, fontsize=10); ax.grid(alpha=.2); ax.legend(fontsize=8, title="Recovery", title_fontsize=8)
    fig.tight_layout()
    path = output / "report_assets" / f"after_eC_{name}.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=140); plt.close(fig)
    return f"report_assets/{path.name}"


def fmt(v, d=3):
    return "—" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))


def write_html(output, summary, plots):
    from scripts.vla_pipeline_qa_compare import COLORS, NAMES
    e = html.escape
    name = lambda k: NAMES.get(k, k)
    parts = ["<!doctype html><meta charset=utf-8><title>Models on pipeline QA</title><style>body{font:15px system-ui;"
             "max-width:1300px;margin:auto;padding:1.5rem;color:#1b2932}table{border-collapse:collapse;margin:.4rem 0 1.2rem}"
             "th,td{border:1px solid #ccd;padding:.3rem .5rem;text-align:right}th{background:#f1f4f6;text-align:center}"
             "td.l{text-align:left}.scroll{overflow-x:auto}img{max-width:100%}.note{color:#4a5a63}.badge{padding:1px 6px;"
             "border-radius:3px;font-size:12px;color:#fff;white-space:nowrap}</style>",
             "<h1>Frozen models on pipeline-processed cases: final eC regression</h1>",
             f"<p>Model run: <code>{e(summary['model'])}</code> ({e(str(summary['feature_view']))}); models "
             f"{', '.join(summary['models'])}. QA runs: {', '.join(e(Path(r).name) for r in summary['runs'])}. "
             f"Labels: {e(summary['v5_version'])}.</p>",
             "<ul class=note><li><b>Target</b>: measured total eC after the recipe. <b>Prediction</b>: the model's "
             "total-eC output on the after-recipe image features. No per-kind sums, before eC, injected class or "
             "remaining gain damage are scored; the initial error type is context only.</li>"
             "<li><b>R²</b> = 1 − Σ(pred − after)² / Σ(after − mean)², <b>RMSE</b> = √mean((pred − after)²), both from "
             "the pooled pairs of each row (never averaged over rows); R² is — for fewer than two pairs or a constant "
             "target; negative values are kept.</li>"
             "<li><b>Pairs</b>: finite prediction/target pairs (coverage, not a score); <i>reused</i> = pairs whose after "
             "features are the archived features because the QA run verified an unchanged input. Missing pairs are "
             "listed with their reason.</li>"
             f"<li>{summary['evaluated']} of {summary['cases']} case rows have after features. The QA cases come from the "
             "split they were planned on; if that is the validation split, it also selected the models' settings, so "
             "these are not held-out numbers.</li></ul>"]

    def badge(k):
        color = COLORS.get(k)
        return f"<span class=badge style='background:{color}'>{e(name(k))}</span>" if color else e(name(k))

    def table(first, body):
        head = ([first] if first else []) + ["Outcome", "Cases", "Pairs (reused)", "after eC R²", "after eC RMSE",
                                             "Missing pairs"]
        out = [f"<div class=scroll><table><tr>{''.join(f'<th>{h}</th>' for h in head)}</tr>"]
        for group, outcomes in body:
            for n, (k, s) in enumerate(outcomes):
                miss = "; ".join(f"{v} {e(why)}" for why, v in s["missing"].items())
                lead = (f"<td class=l rowspan={len(outcomes)}><b>{e(name(group))}</b></td>" if first and n == 0 else "")
                out.append(f"<tr>{lead}<td class=l>{'<b>all</b>' if k == 'all' else badge(k)}</td><td>{s['total']}</td>"
                           f"<td>{s['pairs']} ({s['reused_pairs']})</td><td>{fmt(s['r2'])}</td>"
                           f"<td>{fmt(s['rmse'], 4)}</td><td class=l><small>{miss}</small></td></tr>")
        return "".join(out) + "</table></div>"

    for config, per_model in summary["tables"].items():
        parts.append(f"<h2>{e(config)}</h2>")
        for model, sections in per_model.items():
            parts.append(f"<h3>{e(model)}</h3>")
            for i, (key, title, _) in enumerate(SECTIONS, 1):
                t = sections[key]
                parts.append(f"<h4>Section {i}, by {title}</h4><p>Table 1. All cases</p>" + table("", t["table1"])
                             + "<p>Table 2. By initial group (Partial success applies only to Improvement expected)</p>"
                             + table("Group", t["table2"]) + "<p>Table 3. No improvement expected, by subgroup</p>"
                             + table("Subgroup", t["table3"]))
    parts.append("<h2>Predicted against measured after eC</h2>")
    for n, path in plots.items():
        parts.append(f"<h3>{e(n)}</h3><img src='{path}' alt='{e(n)} predicted against measured after eC'>")
    (output / "report.html").write_text("".join(parts))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="vla_pipeline_qa output folders")
    ap.add_argument("--model", type=Path, required=True, help="ml.multitask output folder (results.json + models)")
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    summary_model = json.loads((a.model / "results.json").read_text())
    columns = feature_columns(summary_model)
    a.output.mkdir(parents=True, exist_ok=True)
    rows = collect(a.runs)
    if not any(r["X"] is not None for r in rows):
        raise SystemExit("No QA case has stored after features")
    preds = predict(a.model, a.models, matrix(rows, columns))
    tables = {}
    for config in dict.fromkeys((r["run_id"], r["config"]) for r in rows):     # absolute run path: never pooled
        idx = [i for i, r in enumerate(rows) if (r["run_id"], r["config"]) == config]
        sub = [rows[i] for i in idx]
        tables[f"{config[1]} ({config[0]})"] = {
            n: {key: section(sub, preds[n][idx], key, cats) for key, _, cats in SECTIONS} for n in a.models}
    plots = {n: plot(a.output, n, rows, preds[n]) for n in a.models}
    np.savez_compressed(a.output / "predictions.npz", run=[r["run_id"] for r in rows], recipe=[r["recipe"] for r in rows],
                        case=[r["case"] for r in rows], group=[r["group"] for r in rows],
                        subgroup=[str(r["subgroup"]) for r in rows], result=[r["result"] for r in rows],
                        recovery=[r["recovery"] for r in rows], after_source=[str(r["after_source"]) for r in rows],
                        after=np.asarray([np.nan if r["after"] is None else r["after"] for r in rows], float),
                        **{f"pred_eC_{n}": preds[n] for n in a.models})
    summary = dict(model=str(a.model.resolve()), feature_view=summary_model.get("feature_view"), models=a.models,
                   runs=[str(r.resolve()) for r in a.runs], cases=len(rows),
                   evaluated=sum(r["X"] is not None for r in rows), v5_version=V5_VERSION, target="measured eC after",
                   prediction="model eC output on after features", tables=tables)
    atomic_write_json(a.output / "results.json", summary)
    write_html(a.output, summary, plots)
    print("REPORT", (a.output / "report.html").resolve())


if __name__ == "__main__":
    main()
