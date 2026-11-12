"""Frozen Dataset v2 multitask models on VLA pipeline QA cases: do they see whether the pipeline worked?

Reads the image features of every case: before = the archived corrupted case (all 22 features from the source
dataset, CLEAN2 from its sidecar), after = evaluation.after.X stored by the QA evaluation of the tested recipe, applies frozen multitask
models (default xgb_xgb and xgb_mlp, with their two XGB experts) and writes one report. No CASA, no refit, run
folders are only read.

  bin/run ml ml/pipeline_qa_models.py --model ml/runs/<multitask run> --output ml/runs/<name> QA_RUN [QA_RUN ...]

Report:
  1. Before (the models' usual task on these cases): the multitask regression and presence metrics
     (ml.multitask.task_metrics) against the archived labels, and predicted-against-true diagonals.
  2. After the recipe. Per-type truth is unknown after the pipeline, so only these are scored:
     - eC after (regression);
     - remaining gain damage = eC after - reference eC (My Criteria remaining excess) against predicted
       d_amp + d_phase (the per-type damages need not add up, so this is an approximation);
     - noise presence: b_noise does not change (self-calibration does not remove thermal noise);
     - "gain damage remains" = eC after > boundary (reference + tolerance), scored by the presence outputs
       (max of amp/phase probability, flag at 0.5) and by predicted gain damage (flag above the tolerance),
       next to two baselines: the pipeline's own status (flag when it did not apply a correction) and residual
       peak/MAD of the image.
  3. Per configuration, v4 (design doc 2026-11-12): expectation group x each independent outcome (Pipeline,
     Recovery, Image) against the corruption types the presence classifier assigns after the recipe ("clean" = none
     assigned), with the same table for the before image. The legacy v3 My Criteria table is kept in a collapsed
     section.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from ml import multitask as mt
from scripts.vla_pipeline_qa import EPSILON_RMS, EPSILON_SNR, V4_VERSION
from scripts.preprocessing.schema import atomic_write_json

DEFAULT_MODELS = ("xgb_xgb", "xgb_mlp")
SETS = tuple("+".join(k for j, k in enumerate(mt.KINDS) if m >> j & 1) or "clean" for m in range(8))
SET_ORDER = (0, 1, 2, 3, 4, 5, 6, 7)          # clean, amp, phase, amp+phase, noise, amp+noise, phase+noise, all
BASELINE_FEATURE = "metrics.residual.peak_over_scaled_mad"


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


def collect(runs, cache, eps_rms, eps_snr):
    """One row per (run, recipe, case) with the v4 outcomes, legacy My Criteria and the before/after features.
    cache: folder for the v4 Image measurements (restored-image RMS/SNR, read from retained FITS only)."""
    from scripts.vla_pipeline_qa import TOLERANCE, archived_features, safe
    from scripts.vla_pipeline_qa_compare import add_image_metrics, load
    rows = []
    sources = {run: Path(json.loads((Path(run) / "plan.json").read_text())["source_run"]) for run in runs}
    for run in runs:
        configs = load(run)
        add_image_metrics(configs, cache, eps_rms, eps_snr)
        for config in configs:
            for c in config["cases"]:
                row = dict(run=config["run"], config=config["config"], recipe=config["recipe"], case=c["case"],
                           label=c["label"], mask=c.get("mask", 0), category=c["category"], action=c.get("action"),
                           expectation=c["expectation"], pipeline=c.get("pipeline"), recovery=c.get("recovery"),
                           image=c.get("image"),
                           before=c.get("before"), after=c.get("after"), reference=c.get("reference"),
                           boundary=c.get("boundary"), d=c.get("d"), X_before=None, X_after=None)
                path = Path(run).resolve() / "results" / f"{c['case']}__{safe(config['recipe'])}.json"
                if path.exists():
                    ev = json.loads(path.read_text()).get("evaluation", {})
                    if ev.get("status") == "measured":
                        names = ev["feature_names"]
                        reused = bool(ev.get("evaluation_reused"))
                        # Before = the archived source case. QA results written before 2026-11-12 stored its 20
                        # committed values under all 22 names, so the full row is read from the source dataset.
                        before_x = archived_features(sources[run], c["case"])
                        # A reused evaluation means the input was unchanged (nothing applied, no flags added): its
                        # after features are the before features (the stored copy has the same 20-value problem).
                        after_x = before_x if reused else ev["after"]["X"]
                        if len(after_x) != len(names):
                            raise ValueError(f"{path}: {len(after_x)} after features for {len(names)} names")
                        row.update(names=names, X_before=before_x, X_after=after_x, reused=reused)
                rows.append(row)
    return rows, TOLERANCE


def matrix(rows, key, columns):
    out = np.full((len(rows), len(columns)), np.nan)
    for i, r in enumerate(rows):
        lookup = dict(zip(r["names"], r[key]))
        missing = set(columns) - set(lookup)
        if missing:
            raise ValueError(f"{r['case']}: stored features lack {sorted(missing)}")
        out[i] = [np.nan if lookup[n] is None else lookup[n] for n in columns]
    return out


def predict(model_dir, names, x):
    """Decoded predictions of each requested frozen model (stacked ones get the expert matrix)."""
    needed = set(names) | ({"xgb_independent", "xgb_shared"} if set(names) & set(mt.STACKED) else set())
    fitted = {n: mt.load_frozen(model_dir, n) for n in needed}
    stage = (np.column_stack((x, mt._expert_matrix(fitted["xgb_independent"], fitted["xgb_shared"], x)))
             if set(names) & set(mt.STACKED) else None)
    out = {}
    for n in names:
        xi = stage if n in mt.STACKED else x
        raw = mt.predict_mlp(fitted[n], xi) if n in mt.MLP_NAMES else mt.predict_xgb(fitted[n]["model"], xi)
        out[n] = mt.decode(raw, fitted[n]["scale"])
    return out


def dataset_labels(runs):
    """Row id -> (d, o, pi, b) from the source datasets' split npz files (sparse damage labels, as in training)."""
    labels = {}
    for run in runs:
        source = Path(json.loads((Path(run) / "plan.json").read_text())["source_run"])
        for path in sorted(source.glob("*.npz")):
            if path.stem not in ("train", "val", "test"):
                continue
            with np.load(path, allow_pickle=False) as z:
                if not {"row_ids", "d", "o", "pi", "b"} <= set(z.files):
                    continue
                for i, rid in enumerate(z["row_ids"]):
                    labels[str(rid)] = (z["d"][i], z["o"][i], z["pi"][i], z["b"][i])
    return labels


def before_truth(rows, labels=None):
    """multitask label arrays for the archived (before) cases: per-type damage d (with the dataset's observation
    mask o and probability pi when its npz is available, else dense from rows.json), presence b, total eC."""
    n = len(rows)
    labels = labels or {}
    b = np.asarray([[r["mask"] >> k & 1 for k in range(3)] for r in rows], float)
    d, o, pi = np.full((n, 3), np.nan), np.zeros((n, 3)), np.ones((n, 3))
    for i, r in enumerate(rows):
        if r["label"] in labels:
            di, oi, pii, bi = labels[r["label"]]
            if not np.array_equal(bi, b[i]):
                raise ValueError(f"{r['label']}: dataset presence {bi} differs from the plan mask {r['mask']}")
            d[i], o[i], pi[i] = di, oi, pii
        elif r["d"] is not None:
            d[i] = [np.nan if v is None else v for v in r["d"]]
            o[i] = np.isfinite(d[i])
    return dict(X=np.zeros((n, 1)), d=d, o=o, pi=pi, b=b, y_all=np.asarray([r["before"] for r in rows], float))


def classification(truth, score, threshold):
    """Macro precision/recall/F1 at the threshold (as the multitask report), AUROC/AUPRC of the score, counts."""
    from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
    truth = np.asarray(truth, bool)
    flag = np.asarray(score) >= threshold
    p, r, f, _ = precision_recall_fscore_support(truth, flag, average="macro", labels=[False, True], zero_division=0)
    both = truth.any() and (~truth).any()
    return dict(rows=int(len(truth)), positives=int(truth.sum()), flagged=int(flag.sum()),
                tp=int((flag & truth).sum()), fp=int((flag & ~truth).sum()), fn=int((~flag & truth).sum()),
                recall_yes=float((flag & truth).sum() / truth.sum()) if truth.any() else None,
                false_alarm=float((flag & ~truth).sum() / (~truth).sum()) if (~truth).any() else None,
                precision=float(p), recall=float(r), f1=float(f),
                auroc=float(roc_auc_score(truth, score)) if both else None,
                auprc=float(average_precision_score(truth, score)) if both else None)


def after_scores(rows, pred, xa, columns, tolerance):
    """Scores of one model's after-recipe predictions (and its before predictions, for the same quantities)."""
    after = np.asarray([r["after"] for r in rows], float)
    before = np.asarray([r["before"] for r in rows], float)
    ref = np.asarray([r["reference"] for r in rows], float)
    boundary = np.asarray([r["boundary"] for r in rows], float)
    b_noise = np.asarray([r["mask"] >> 2 & 1 for r in rows], bool)
    gain = np.asarray([r["mask"] & 3 != 0 for r in rows], bool)
    not_applied = np.asarray([r["action"] != "Applied" for r in rows], float)
    residual = xa[:, columns.index(BASELINE_FEATURE)] if BASELINE_FEATURE in columns else None
    out = {}
    for when, p, true_eC in (("after", pred["after"], after), ("before", pred["before"], before)):
        remains = true_eC > boundary
        gain_pred = p[:, 0] + p[:, 1]
        presence = np.maximum(p[:, 4], p[:, 5])
        out[when] = dict(
            eC=mt.total_scores(p[:, 3], true_eC),
            gain_damage=mt.total_scores(gain_pred, true_eC - ref),
            gain_damage_injected=mt.total_scores(gain_pred[gain], (true_eC - ref)[gain]) if gain.any() else None,
            noise_presence=classification(b_noise, p[:, 6], .5),
            remains={"presence amp/phase (max p ≥ 0.5)": classification(remains, presence, .5),
                     f"predicted gain damage > {tolerance}": classification(remains, gain_pred, tolerance)})
    remains = after > boundary
    out["after"]["remains"]["baseline: pipeline did not apply a correction"] = classification(remains, not_applied, .5)
    if residual is not None:
        thr = float(np.nanmedian(residual))
        out["after"]["remains"][f"baseline: residual peak/MAD ≥ median ({thr:.3g})"] = classification(
            remains, np.nan_to_num(residual, nan=thr), thr)
    return out


def _counts(rows, sets, keep):
    return dict(total=len(keep), above_after=int(sum(rows[i]["after"] > rows[i]["boundary"] for i in keep)),
                gain_flagged=int(np.sum(sets[keep] & 3 != 0)),
                sets={SETS[m]: int(np.sum(sets[keep] == m)) for m in SET_ORDER})


def crosstab(rows, probabilities):
    """Legacy v3: My Criteria category -> {total, above boundary after, gain flagged, predicted set counts}."""
    from scripts.vla_pipeline_qa import ACCOUNTING, CATEGORIES
    sets = (probabilities >= .5).astype(int).dot(1 << np.arange(3)) if len(rows) else np.zeros(0, int)
    table = {}
    for k in CATEGORIES + ACCOUNTING:
        keep = [i for i, r in enumerate(rows) if r["category"] == k]
        if keep:
            table[k] = _counts(rows, sets, keep)
    return table


def crosstab_v4(rows, probabilities):
    """v4: expectation group -> dimension -> outcome -> counts (all = whole group). Dimensions are independent, so
    each one partitions the group on its own; accounting states (unavailable, pending, ...) are kept as rows."""
    from scripts.vla_pipeline_qa import EXPECTATIONS, IMAGE, PIPELINE, RECOVERY
    sets = (probabilities >= .5).astype(int).dot(1 << np.arange(3)) if len(rows) else np.zeros(0, int)
    table = {}
    for group in EXPECTATIONS:
        members = [i for i, r in enumerate(rows) if r["expectation"] == group]
        if not members:
            continue
        table[group] = {"all": {"all": _counts(rows, sets, members)}}
        for dim, cats in (("pipeline", PIPELINE), ("recovery", RECOVERY), ("image", IMAGE)):
            values = list(cats) + sorted({rows[i][dim] for i in members} - set(cats), key=str)
            table[group][dim] = {str(v): _counts(rows, sets, [i for i in members if rows[i][dim] == v])
                                 for v in values if any(rows[i][dim] == v for i in members)}
    return table


def plot(output, name, rows, pred, truth):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    masks = np.asarray([r["mask"] for r in rows])
    observed = truth["o"].astype(bool)
    after = np.asarray([r["after"] for r in rows], float)
    ref = np.asarray([r["reference"] for r in rows], float)
    pb, pa = pred["before"], pred["after"]
    d = np.where(observed, truth["d"], np.nan)           # observed labels only, as in the multitask diagonals
    panels = [("before: $d_{\\rm amp}$", d[:, 0], pb[:, 0]), ("before: $d_{\\rm phase}$", d[:, 1], pb[:, 1]),
              ("before: $d_{\\rm noise}$", d[:, 2], pb[:, 2]), ("before: $e_C$", truth["y_all"], pb[:, 3]),
              ("after: gain damage\n$e_C$ − reference vs $d_{\\rm amp}+d_{\\rm phase}$", after - ref, pa[:, 0] + pa[:, 1]),
              ("after: $e_C$", after, pa[:, 3])]
    fig, axes = plt.subplots(2, 4, figsize=(11.6, 6.2))
    for ax, (title, x, y) in zip(axes.flat, panels):
        keep = np.isfinite(x) & np.isfinite(y)
        for m in range(8):
            sel = keep & (masks == m)
            if sel.any():
                ax.scatter(x[sel], y[sel], s=7, alpha=.45, c=mt.MASK_COLORS[m], marker=mt.MASK_MARKERS[m], linewidths=0)
        if keep.any():
            lo, hi = float(min(x[keep].min(), y[keep].min())), float(max(x[keep].max(), y[keep].max()))
            pad = .04 * (hi - lo or 1.)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#444", lw=.8)
            ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(title, fontsize=10); ax.grid(alpha=.2); ax.tick_params(labelsize=7)
        ax.set_xlabel("true", fontsize=8); ax.set_ylabel("predicted", fontsize=8)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    handles = [Line2D([], [], ls="", marker=mt.MASK_MARKERS[m], color=mt.MASK_COLORS[m], markersize=7,
                      label=mt.MASK_LABELS[m]) for m in range(8)]
    axes.flat[-1].legend(handles=handles, loc="center", frameon=False, title="Injected corruption types", fontsize=8)
    fig.tight_layout()
    path = output / "report_assets" / f"diagonal_{name}.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=140); plt.close(fig)
    return f"report_assets/{path.name}"


def fmt(v, d=3):
    return "—" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))


def write_html(output, summary, plots):
    from scripts.vla_pipeline_qa_compare import COLORS, name as cat_name
    e = html.escape
    tol = summary["tolerance"]
    parts = ["<!doctype html><meta charset=utf-8><title>Multitask models on pipeline QA</title><style>body{font:15px "
             "system-ui;max-width:1500px;margin:auto;padding:1.5rem;color:#1b2932}table{border-collapse:collapse;margin:.4rem "
             "0 1.4rem}th,td{border:1px solid #ccd;padding:.3rem .5rem;text-align:right}th{background:#f1f4f6;text-align:"
             "center}td:first-child{text-align:left}.scroll{overflow-x:auto}img{max-width:100%}.note{color:#4a5a63}"
             ".badge{padding:1px 6px;border-radius:3px;font-size:12px;color:#fff;white-space:nowrap}</style>",
             "<h1>Multitask models on pipeline-processed cases</h1>",
             f"<p>Model run: <code>{e(summary['model'])}</code> ({e(summary['feature_view'])}); models "
             f"{', '.join(summary['models'])}. QA runs: {', '.join(e(Path(r).name) for r in summary['runs'])}. "
             f"{summary['evaluated']} of {summary['cases']} case rows have stored features (the rest are pending, failed "
             "or unavailable). No CASA was run; features were computed by the QA evaluation.</p>",
             "<ul class=note><li><b>Before</b>: the archived corrupted case, the models' usual task.</li>"
             "<li><b>After</b>: the evaluation image after the recipe. Per-type truth is unknown there; scored are eC "
             "after, remaining gain damage (eC after − reference eC) against predicted d<sub>amp</sub>+d<sub>phase</sub> "
             "(per-type damages need not add up), noise presence (unchanged by self-calibration) and whether gain damage "
             f"remains (eC after &gt; reference + {tol}).</li>"
             "<li>The QA cases come from the split they were planned on; if that is the validation split, it also "
             "selected the models' settings, so these are not held-out numbers.</li></ul>"]

    def table(head, body):
        return (f"<div class=scroll><table><tr>{''.join(f'<th>{h}</th>' for h in head)}</tr>"
                + "".join(f"<tr>{''.join(f'<td>{c}</td>' for c in row)}</tr>" for row in body) + "</table></div>")

    # 1. Before: same metrics as the multitask report
    parts.append("<h2>1. Before the recipe: multitask metrics</h2>")
    head = ["Model", "e<sub>C</sub> RMSE", "e<sub>C</sub> R²"] + [
        f"d<sub>{k}</sub> {m}" for k in mt.KINDS for m in ("RMSE", "R²", "RMSE active", "R² active", "RMSE absent")]
    body = []
    for n in summary["models"]:
        m = summary["metrics"][n]["before_multitask"]
        row = [e(n), fmt(m["y_all"]["rmse"], 4), fmt(m["y_all"]["r2"])]
        for k in mt.KINDS:
            mk = m[k]
            row += [fmt(mk["damage_rmse"], 4), fmt(mk["damage_r2"]), fmt(mk["damage_active"]["rmse"], 4),
                    fmt(mk["damage_active"].get("r2")), fmt(mk["damage_absent"]["rmse"], 4)]
        body.append(row)
    parts.append(table(head, body))
    head = ["Model"] + [f"b<sub>{k}</sub> {m}" for k in mt.KINDS for m in ("precision", "recall", "F1", "AUROC", "AUPRC")]
    body = []
    for n in summary["models"]:
        m = summary["metrics"][n]["before_multitask"]
        row = [e(n)]
        for k in mt.KINDS:
            c = mt.classification_scores(m[k], summary["evaluated"])
            row += [fmt(c[x]) for x in ("precision", "recall", "f1", "auroc", "auprc")]
        body.append(row)
    parts.append("<p>Presence classification (macro scores at 0.5, as in the multitask report):</p>" + table(head, body))

    # 2. After
    parts.append("<h2>2. After the recipe</h2><p>Regression (before shown for the same quantities):</p>")
    head = ["Model", "when", "e<sub>C</sub> RMSE", "e<sub>C</sub> R²", "gain damage RMSE", "gain damage R²",
            "gain damage R² (amp/phase injected)", "noise presence F1", "noise presence AUROC"]
    body = []
    for n in summary["models"]:
        for when in ("before", "after"):
            m = summary["metrics"][n][when]
            gi = m["gain_damage_injected"] or {}
            body.append([e(n), when, fmt(m["eC"]["rmse"], 4), fmt(m["eC"]["r2"]), fmt(m["gain_damage"]["rmse"], 4),
                         fmt(m["gain_damage"]["r2"]), fmt(gi.get("r2")), fmt(m["noise_presence"]["f1"]),
                         fmt(m["noise_presence"]["auroc"])])
    parts.append(table(head, body))
    parts.append(f"<p><b>Does gain damage remain?</b> Truth: eC &gt; reference + {tol}. Recall = share of those cases "
                 "flagged; false alarm = share of fine cases flagged; macro scores as above. Baselines use no model.</p>")
    head = ["Model", "when", "detector", "positives / rows", "flagged", "recall", "false alarm", "macro F1", "AUROC", "AUPRC"]
    body = []
    for n in summary["models"]:
        for when in ("before", "after"):
            for det, c in summary["metrics"][n][when]["remains"].items():
                if det.startswith("baseline") and n != summary["models"][0]:
                    continue
                body.append([e(n if not det.startswith("baseline") else "—"), when, e(det),
                             f"{c['positives']} / {c['rows']}", c["flagged"], fmt(c["recall_yes"]),
                             fmt(c["false_alarm"]), fmt(c["f1"]), fmt(c["auroc"]), fmt(c["auprc"])])
    parts.append(table(head, body))

    # Plots
    parts.append("<h2>Predicted against true</h2><p>Top row before (archived labels), bottom row after the recipe. "
                 "Colors/markers: injected corruption types.</p>")
    for n in summary["models"]:
        parts.append(f"<h3>{e(n)}</h3><img src='{plots[n]}' alt='{e(n)} predicted against true'>")

    # 3. Category x predicted set
    from scripts.vla_pipeline_qa_compare import DIMENSIONS, EXPECTATION_NAMES
    sets_head = list(SETS[m] for m in SET_ORDER)
    head = ["Expectation group", "Dimension", "Outcome", "Total", "Above boundary (after)", "Gain flagged"] + sets_head
    parts.append("<h2>3. Pipeline outcomes against the presence classifier (v4)</h2><p>Per configuration and model. "
                 "Rows: expectation group (fixed before processing), then each independent outcome dimension (Pipeline, "
                 "Recovery, Image) split into its categories; each dimension partitions the group on its own. Columns "
                 "after <i>Total</i> count the corruption types the classifier assigns (p ≥ 0.5) to the image after the "
                 "recipe; <i>clean</i> = none. <i>Gain flagged</i> = amp or phase assigned; <i>Above boundary</i> = eC "
                 "after still above reference + tolerance. If the models see whether the pipeline worked, gain flags "
                 "should concentrate where Recovery is not improved and eC stays above the boundary. The before table "
                 "is the same classifier on the archived corrupted image.</p>")
    for config, per_model in summary["crosstab_v4"].items():
        parts.append(f"<h3>{e(config)}</h3>")
        for n, tables in per_model.items():
            for when in ("after", "before"):
                body = []
                for group, dims in tables[when].items():
                    for dim, outcomes in dims.items():
                        for k, t in outcomes.items():
                            color = DIMENSIONS.get(dim, ((), {}))[1].get(k)
                            badge = (f"<span class=badge style='background:{color}'>{e(k)}</span>" if color else e(k))
                            body.append([e(EXPECTATION_NAMES[group]) if dim == "all" else "", "" if dim == "all" else dim,
                                         "<b>all</b>" if dim == "all" else badge, t["total"], t["above_after"],
                                         t["gain_flagged"]] + [t["sets"][x] or "" for x in sets_head])
                parts.append(f"<p><b>{e(n)}</b>, classifier on the {when} image</p>" + table(head, body))
    parts.append("<details><summary><b>Legacy v3 My Criteria</b> against the presence classifier</summary>")
    for config, per_model in summary["crosstab"].items():
        parts.append(f"<h3>{e(config)}</h3>")
        for n, tables in per_model.items():
            for when in ("after", "before"):
                head3 = ["Category", "Total", "Above boundary (after)", "Gain flagged"] + sets_head
                body = []
                for k, t in tables[when].items():
                    badge = (f"<span class=badge style='background:{COLORS[k]}'>{e(cat_name(k))}</span>" if k in COLORS
                             else e(cat_name(k)))
                    body.append([badge, t["total"], t["above_after"], t["gain_flagged"]]
                                + [t["sets"][x] or "" for x in sets_head])
                parts.append(f"<p><b>{e(n)}</b>, classifier on the {when} image</p>" + table(head3, body))
    parts.append("</details>")
    (output / "report.html").write_text("".join(parts))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="vla_pipeline_qa output folders")
    ap.add_argument("--model", type=Path, required=True, help="ml.multitask output folder (results.json + models)")
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epsilon-rms", type=float, default=EPSILON_RMS, help="v4 Image RMS tolerance (PROVISIONAL)")
    ap.add_argument("--epsilon-snr", type=float, default=EPSILON_SNR, help="v4 Image SNR tolerance (PROVISIONAL)")
    a = ap.parse_args()
    summary_model = json.loads((a.model / "results.json").read_text())
    columns = feature_columns(summary_model)
    a.output.mkdir(parents=True, exist_ok=True)
    all_rows, tolerance = collect(a.runs, a.output / "image_metrics", a.epsilon_rms, a.epsilon_snr)
    rows = [r for r in all_rows if r["X_after"] is not None and r["after"] is not None and r["boundary"] is not None]
    if not rows:
        raise SystemExit("No QA case has stored before/after features")
    xb, xa = matrix(rows, "X_before", columns), matrix(rows, "X_after", columns)
    preds = {"before": predict(a.model, a.models, xb), "after": predict(a.model, a.models, xa)}
    a.output.mkdir(parents=True, exist_ok=True)
    labels = dataset_labels(a.runs)
    truth = before_truth(rows, labels)
    metrics, plots = {}, {}
    for n in a.models:
        p = dict(before=preds["before"][n], after=preds["after"][n])
        metrics[n] = dict(before_multitask=mt.task_metrics(p["before"], truth),
                          **after_scores(rows, p, xa, list(columns), tolerance))
        plots[n] = plot(a.output, n, rows, p, truth)
    configs = sorted({r["config"] for r in rows})
    crosstabs, crosstabs_v4 = {}, {}
    for config in configs:
        idx = [i for i, r in enumerate(rows) if r["config"] == config]
        sub = [rows[i] for i in idx]
        crosstabs[config] = {n: {when: crosstab(sub, preds[when][n][idx, 4:7]) for when in ("after", "before")}
                             for n in a.models}
        crosstabs_v4[config] = {n: {when: crosstab_v4(sub, preds[when][n][idx, 4:7]) for when in ("after", "before")}
                                for n in a.models}
    np.savez_compressed(a.output / "predictions.npz", case=[r["case"] for r in rows], recipe=[r["recipe"] for r in rows],
                        run=[r["run"] for r in rows], category=[r["category"] for r in rows],
                        expectation=[r["expectation"] for r in rows], pipeline=[str(r["pipeline"]) for r in rows],
                        recovery=[str(r["recovery"]) for r in rows], image=[str(r["image"]) for r in rows],
                        **{f"{when}_{n}": preds[when][n] for when in preds for n in a.models})
    summary = dict(model=str(a.model.resolve()), feature_view=summary_model.get("feature_view"), models=a.models,
                   runs=[str(r.resolve()) for r in a.runs], cases=len(all_rows), evaluated=len(rows),
                   tolerance=tolerance, labels_from_npz=sum(r["label"] in labels for r in rows), outputs=list(mt.outputs(mt.has_dirty(np.asarray(
                       summary_model["selected"][a.models[0]]["scale"])))), metrics=metrics, crosstab=crosstabs,
                   crosstab_v4=crosstabs_v4, v4_version=V4_VERSION,
                   epsilon_rms=a.epsilon_rms, epsilon_snr=a.epsilon_snr)
    atomic_write_json(a.output / "results.json", summary)
    write_html(a.output, summary, plots)
    print("REPORT", (a.output / "report.html").resolve())


if __name__ == "__main__":
    main()
