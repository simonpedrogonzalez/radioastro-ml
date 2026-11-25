"""Does each image-feature block, by itself, improve all22 + xgb_mlp? (docs/image_features_plan.md §3)

  python -m ml.image_feature_experiment --train runs/dataset-v2-matched-train --val runs/dataset-v2-matched-val \
    --features-train runs/image-features-v2/train.npz --features-val runs/image-features-v2/val.npz \
    --output ml/runs/image-features-xgb_mlp --sets base energy radial fft --seeds 42 43 44 --jobs 8

Data: load_split(..., "all22", dirty=True) for both splits (as ml/runs/matched-v2-full-all22-dirty), then the
feature-table columns of each block are appended, joined by case_ids (the table's split-npz and plan hashes must
match). A set is `base` or a `+`-joined list of blocks (energy, radial, fft). `fft` adds, per channel (C and the
residual channel chosen by --fft-residual), the excess z-scores of log10 S_k against the TRAINING controls
(mask 0) plus peak_cycles_per_beam and angular_concentration; never the raw log10 S_k.

Per set: xgb_independent and xgb_shared are fitted once (deterministic), then multitask.expert_stage builds the
OOF second-stage inputs; per seed only the second-stage MLP (fit_mlp) is refitted. Same grid, early stopping and
target scales as multitask; selection on validation J as in every earlier run. Test is untouched.

Outputs: <output>/<set>/seed<s>/{predictions_val_xgb_mlp.npz, results.json} (multitask layout, so
ml.compare_val_by_ec can read them), <output>/<set>/experts.npz, <output>/summary.json and report.html with the
per-set metrics, the paired Δ against `base` with a 95% validation-source bootstrap, and the reproduction check of
base/seed 42 against the baseline run.
"""
from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path

import numpy as np

from ml import multitask as mt
from ml.compare_val_by_ec import STYLE, bootstrap_weights
from scripts.features.extract import file_sha256
from scripts.features.fourier import K_LABELS, excess
from scripts.preprocessing.schema import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
BLOCKS = ("energy", "radial", "fft")
FEATURE_VIEW = "all22"
BASELINE = "ml/runs/matched-v2-full-all22-dirty"
# (key, label, higher is better)
METRICS = (("eC_rmse", "eC RMSE", False), ("eC_rmse_gt1", "eC RMSE, eC > 1", False),
           ("eC_rmse_lt0.1", "eC RMSE, eC < 0.1", False), ("eC_r2", "eC R²", True),
           *((f"d_{k}_r2_active", f"d_{k} R² (active)", True) for k in mt.KINDS),
           *((f"auroc_{k}", f"presence AUROC {k}", True) for k in mt.KINDS))


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_table(path, root, data, split):
    """Feature table rows reordered to data's rows; the table must come from exactly this split npz and plan."""
    with np.load(_resolve(path), allow_pickle=False) as z:
        t = {k: z[k] for k in z.files}
    root = _resolve(root)
    for key, file in (("split_npz_sha256", root / f"{split}.npz"), ("plan_sha256", root / "plan.json")):
        if str(t[key]) != file_sha256(file):
            raise ValueError(f"{path}: {key} does not match {file}")
    index = {c: i for i, c in enumerate(map(str, t["case_ids"]))}
    missing = [c for c in map(str, data["case_ids"]) if c not in index]
    if missing:
        raise ValueError(f"{path}: {len(missing)} {split} cases missing (e.g. {missing[:3]})")
    rows = np.asarray([index[c] for c in map(str, data["case_ids"])])
    if not np.array_equal(t["row_ids"][rows].astype(str), np.asarray(data["row_ids"]).astype(str)):
        raise ValueError(f"{path}: row ids differ after joining by case id")
    return dict(names=[str(n) for n in t["names"]], X=t["X"][rows])


def block_columns(block, train_table, val_table, control, fft_residual):
    """(names, train columns, val columns) of one block; fft excess uses the training controls only."""
    names = train_table["names"]
    if train_table["names"] != val_table["names"]:
        raise ValueError("Train/validation feature tables have different columns")
    col = {n: j for j, n in enumerate(names)}
    if block in ("energy", "radial"):
        picked = [n for n in names if n.startswith(block + ".")]
        idx = [col[n] for n in picked]
        return picked, train_table["X"][:, idx], val_table["X"][:, idx]
    if block != "fft":
        raise ValueError(f"Unknown block {block!r}; blocks are {BLOCKS}")
    out_names, tr, va = [], [], []
    for ch in ("C", fft_residual):
        idx = [col[f"fft.{ch}.log10_S.{k}"] for k in K_LABELS]
        ref = train_table["X"][control][:, idx]
        tr.append(excess(train_table["X"][:, idx], ref)); va.append(excess(val_table["X"][:, idx], ref))
        out_names += [f"fft.{ch}.excess_z.{k}" for k in K_LABELS]
        sup = [f"fft.{ch}.peak_cycles_per_beam", f"fft.{ch}.angular_concentration"]
        tr.append(train_table["X"][:, [col[n] for n in sup]]); va.append(val_table["X"][:, [col[n] for n in sup]])
        out_names += sup
    return out_names, np.column_stack(tr), np.column_stack(va)


def with_blocks(train, val, tables, set_name, fft_residual):
    blocks = [] if set_name == "base" else set_name.split("+")
    if len(set(blocks)) != len(blocks) or any(b not in BLOCKS for b in blocks):
        raise ValueError(f"Bad set {set_name!r}: use 'base' or a '+'-joined list of {BLOCKS}")
    names, xtr, xva = list(map(str, train["feature_names"])), [train["X"]], [val["X"]]
    control = np.asarray(train["mask"]) == 0
    for b in blocks:
        n, a, v = block_columns(b, tables["train"], tables["val"], control, fft_residual)
        names += n; xtr.append(a); xva.append(v)
    names = np.asarray(names)
    return dict(train, X=np.column_stack(xtr), feature_names=names), dict(val, X=np.column_stack(xva), feature_names=names)


def metric_values(pred, val, w):
    """All METRICS for row weights w (n,) or (boots, n); pred = decoded multitask outputs."""
    from sklearn.metrics import roc_auc_score
    w = np.atleast_2d(w)
    y, p = val["eC"], pred[:, 3]

    def rmse(sel):
        ws = w[:, sel]
        return np.sqrt((ws * (p[sel] - y[sel]) ** 2).sum(1) / ws.sum(1))

    def r2(t, q, ws):
        mean = (ws * t).sum(1) / ws.sum(1)
        return 1 - (ws * (q - t) ** 2).sum(1) / (ws * (t - mean[:, None]) ** 2).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {"eC_rmse": rmse(np.ones(len(y), bool)), "eC_rmse_gt1": rmse(y > 1), "eC_rmse_lt0.1": rmse(y < .1),
               "eC_r2": r2(y, p, w)}
        for k, kind in enumerate(mt.KINDS):
            act = val["b"][:, k] == 1   # v2: o = 1 and pi = 1, so the active damage is fully observed
            out[f"d_{kind}_r2_active"] = r2(val["d"][act, k], pred[act, k], w[:, act])
            out[f"auroc_{kind}"] = np.asarray([roc_auc_score(val["b"][:, k], pred[:, 4 + k], sample_weight=wi)
                                               for wi in w])
    return out


def fit_set(train, val, out, seeds, jobs):
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    fitted = {}
    for name in ("xgb_independent", "xgb_shared"):
        fitted[name] = mt._fit_tree_family(train, val, name, jobs)
        print(out.name, "selected", name, fitted[name]["params"], flush=True)
    stage_train, stage_val, oof = mt.expert_stage(train, val, fitted, jobs)
    width = train["X"].shape[1]
    np.savez_compressed(out / "experts.npz", train_oof=oof, val_experts=stage_val["X"][:, width:],
                        train_row_ids=train["row_ids"], val_row_ids=val["row_ids"])
    for seed in seeds:
        fm = mt.fit_mlp(stage_train["X"], stage_train, stage_val["X"], stage_val, seed=seed)
        raw = mt.predict_mlp(fm, stage_val["X"])
        decoded = mt.decode(raw, fm["scale"])
        folder = out / f"seed{seed}"
        folder.mkdir(exist_ok=True)
        np.savez_compressed(folder / "predictions_val_xgb_mlp.npz", row_ids=val["row_ids"], raw=raw, decoded=decoded)
        atomic_write_json(folder / "results.json", dict(
            model="xgb_mlp", seed=seed, feature_view=FEATURE_VIEW, with_dirty=True, train_rows=len(train["X"]),
            val_rows=len(val["X"]), feature_names=train["feature_names"].tolist(), best_epoch=fm["best_epoch"],
            params={n: fitted[n]["params"] for n in ("xgb_independent", "xgb_shared")}, scale=fm["scale"].tolist(),
            losses=mt.raw_losses(raw, val, fm["scale"]), metrics=mt.task_metrics(decoded, val)))
        print(out.name, f"seed {seed}: best epoch {fm['best_epoch']}, val eC RMSE "
              f"{np.sqrt(np.mean((decoded[:, 3] - val['eC']) ** 2)):.4g}", flush=True)
    return round(time.time() - start, 1)


def load_predictions(output, sets, seeds, val):
    pred = {}
    for s in sets:
        for seed in seeds:
            with np.load(output / s / f"seed{seed}" / "predictions_val_xgb_mlp.npz", allow_pickle=False) as z:
                if not np.array_equal(z["row_ids"].astype(str), np.asarray(val["row_ids"]).astype(str)):
                    raise ValueError(f"{s}/seed{seed}: validation rows differ")
                pred[s, seed] = z["decoded"]
    return pred


def analyze(output, sets, seeds, val, boots, boot_seed):
    pred = load_predictions(output, sets, seeds, val)
    W = bootstrap_weights(np.asarray(val["source_ids"]).astype(str), boots, boot_seed)
    point = {k: metric_values(p, val, np.ones(len(val["eC"]))) for k, p in pred.items()}
    boot = {k: metric_values(p, val, W) for k, p in pred.items()}
    per_set, delta = {}, {}
    for s in sets:
        per_set[s] = {}
        for key, _, _ in METRICS:
            v = np.asarray([point[s, seed][key][0] for seed in seeds])
            per_set[s][key] = dict(mean=float(v.mean()), min=float(v.min()), max=float(v.max()),
                                   per_seed=dict(zip(map(str, seeds), map(float, v))))
        if s == "base":
            continue
        delta[s] = {}
        for key, _, _ in METRICS:
            d_point = np.mean([point[s, seed][key][0] - point["base", seed][key][0] for seed in seeds])
            d_boot = np.mean([boot[s, seed][key] - boot["base", seed][key] for seed in seeds], axis=0)
            delta[s][key] = dict(diff=float(d_point), lo=float(np.nanpercentile(d_boot, 2.5)),
                                 hi=float(np.nanpercentile(d_boot, 97.5)))
    return per_set, delta


def reproduction(output, val):
    """base/seed42 against the baseline multitask run's saved validation predictions."""
    base = _resolve(BASELINE) / "predictions_val_xgb_mlp.npz"
    mine = output / "base" / "seed42" / "predictions_val_xgb_mlp.npz"
    if not (base.exists() and mine.exists()):
        return None
    with np.load(base) as a, np.load(mine) as b:
        if not np.array_equal(a["row_ids"], b["row_ids"]):
            return dict(baseline=BASELINE, error="row ids differ")
        return dict(baseline=BASELINE, max_abs_raw=float(np.max(np.abs(a["raw"] - b["raw"]))),
                    max_abs_decoded=float(np.max(np.abs(a["decoded"] - b["decoded"]))),
                    max_abs_eC=float(np.max(np.abs(a["decoded"][:, 3] - b["decoded"][:, 3]))))


def _fmt(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.4g}"


def write_report(output, summary):
    sets, seeds = summary["sets"], summary["seeds"]
    rows = "".join(
        f"<tr><td>{html.escape(s)}</td><td>{summary['n_features'][s]}</td>" + "".join(
            f"<td>{_fmt(m['mean'])}<br><small>[{_fmt(m['min'])}, {_fmt(m['max'])}]</small></td>"
            for m in (summary["metrics"][s][k] for k, _, _ in METRICS)) + "</tr>" for s in sets)
    head = "<tr><th>set</th><th>features</th>" + "".join(f"<th>{html.escape(lab)}</th>" for _, lab, _ in METRICS) + "</tr>"

    def cell(d, better_high):
        text = f"{d['diff']:+.3g}<br><small>[{d['lo']:+.3g}, {d['hi']:+.3g}]</small>"
        if d["lo"] > 0 or d["hi"] < 0:
            good = (d["lo"] > 0) == better_high
            return f"<td class='{'better' if good else 'worse'}'>{text}</td>"
        return f"<td>{text}</td>"
    drows = "".join(f"<tr><td>{html.escape(s)}</td>" + "".join(cell(summary["delta"][s][k], hb) for k, _, hb in METRICS)
                    + "</tr>" for s in sets if s != "base")
    dhead = "<tr><th>set − base</th>" + "".join(f"<th>{html.escape(lab)}</th>" for _, lab, _ in METRICS) + "</tr>"
    rep = summary.get("reproduction")
    rep_text = ("not available (baseline run or base/seed42 missing)" if rep is None else
                html.escape(json.dumps(rep)))
    if summary.get("note"):
        rep_text += f"</p><p>{summary['note']}"
    added = "".join(f"<li><b>{html.escape(s)}</b>: {html.escape(', '.join(summary['added_features'][s]))}</li>"
                    for s in sets if s != "base")
    page = (f"<!doctype html><meta charset='utf-8'><title>Image features: all22 + block, xgb_mlp</title>{STYLE}"
            "<h1>Image features: does each block improve all22 + xgb_mlp?</h1>"
            f"<p class='note'>Train <code>{html.escape(summary['train'])}</code> ({summary['train_rows']} rows), validation "
            f"<code>{html.escape(summary['val'])}</code> ({summary['val_rows']} rows, {summary['val_sources']} sources). "
            f"Feature view all22 with the dirty task; seeds {', '.join(map(str, seeds))} (seeds vary only the second-stage "
            f"MLP; the XGB experts are deterministic). fft residual channel: <b>{html.escape(summary['fft_residual'])}</b>. "
            "Selection on validation J as in every earlier run (equally optimistic for every set); test untouched.</p>"
            "<h2>1. Validation metrics per set</h2><p class='note'>Mean over seeds [min, max]. eC slices use the true eC. "
            "d R² on rows where that corruption is active.</p>"
            f"<table>{head}{rows}</table>"
            "<h2>2. Paired Δ against base</h2>"
            f"<p class='note'>Per-seed difference averaged over seeds; 95% interval from {summary['bootstrap']} resamples of "
            "the validation sources (the same resample applied to both sets). Green/red: interval excludes 0 and the "
            "change is better/worse. A block helps only if its interval excludes 0 on eC RMSE overall, or on eC &gt; 1 "
            "without hurting eC &lt; 0.1.</p>"
            f"<table>{dhead}{drows}</table>"
            f"<h2>3. Reproduction check</h2><p class='note'>base/seed 42 vs <code>{BASELINE}</code>: {rep_text}. "
            "A mismatch is a finding to explain, not a failure.</p>"
            f"<h2>Added features</h2><ul class='note'>{added}</ul>"
            f"<p class='note'>Fit time per set (s): {html.escape(json.dumps(summary.get('seconds', {})))}</p>")
    (output / "report.html").write_text(page)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="runs/dataset-v2-matched-train")
    ap.add_argument("--val", default="runs/dataset-v2-matched-val")
    ap.add_argument("--features-train", default="runs/image-features-v2/train.npz")
    ap.add_argument("--features-val", default="runs/image-features-v2/val.npz")
    ap.add_argument("--output", required=True)
    ap.add_argument("--sets", nargs="+", default=["base", *BLOCKS])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--fft-residual", choices=("R", "Rq"), default="R",
                    help="residual channel of the fft block: raw P_R or PSF-whitened q (decided by the gate)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--report-only", action="store_true", help="recompute summary/report from saved predictions")
    ap.add_argument("--note", help="HTML note shown under the reproduction check (e.g. its explanation)")
    a = ap.parse_args(argv)
    if "base" not in a.sets:
        raise SystemExit("--sets must include base (the reference of the paired differences)")
    output = _resolve(a.output)
    train = mt.load_split(_resolve(a.train), "train", FEATURE_VIEW, dirty=True)
    val = mt.load_split(_resolve(a.val), "val", FEATURE_VIEW, dirty=True)
    tables = dict(train=load_table(a.features_train, a.train, train, "train"),
                  val=load_table(a.features_val, a.val, val, "val"))
    seconds, n_features, added = {}, {}, {}
    previous = json.loads((output / "summary.json").read_text()) if (output / "summary.json").exists() else {}
    for s in a.sets:
        tr, va = with_blocks(train, val, tables, s, a.fft_residual)
        n_features[s] = int(tr["X"].shape[1])
        added[s] = tr["feature_names"][train["X"].shape[1]:].tolist()
        if a.report_only:
            seconds[s] = previous.get("seconds", {}).get(s)
            continue
        if (output / s).exists():
            raise SystemExit(f"{output / s} exists; use a fresh --output or --report-only")
        seconds[s] = fit_set(tr, va, output / s, a.seeds, a.jobs)
    per_set, delta = analyze(output, a.sets, a.seeds, val, a.bootstrap, 0)
    summary = dict(train=str(a.train), val=str(a.val), features_train=str(a.features_train),
                   features_val=str(a.features_val), sets=a.sets, seeds=a.seeds, fft_residual=a.fft_residual,
                   train_rows=len(train["X"]), val_rows=len(val["X"]),
                   val_sources=len(set(map(str, val["source_ids"]))), bootstrap=a.bootstrap,
                   n_features=n_features, added_features=added, seconds=seconds, metrics=per_set, delta=delta,
                   reproduction=reproduction(output, val), note=a.note or previous.get("note"))
    atomic_write_json(output / "summary.json", summary)
    write_report(output, summary)
    print("Image-feature experiment report:", output / "report.html", flush=True)


if __name__ == "__main__":
    main()
