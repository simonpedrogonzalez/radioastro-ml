"""Training-source CV comparison of matched Dataset v2 feature groups.

python -m ml.matched_comparison --dataset RUN --output MODELS --folds 3 --jobs 4
Each outer fold reserves sources for evaluation. A separate source subset of
the remaining data selects settings; it is not used to fit the saved models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from ml import multitask as mt
from scripts.preprocessing.schema import atomic_write_json


def frozen_predictions(output, data):
    fitted = {name: mt.load_frozen(output, name) for name in mt.NAMES}
    stage = np.column_stack((data["X"], mt._expert_matrix(
        fitted["xgb_independent"], fitted["xgb_shared"], data["X"])))
    predictions = {}
    for name, obj in fitted.items():
        x = stage if name in mt.STACKED else data["X"]
        raw = mt.predict_mlp(obj, x) if name in mt.MLP_NAMES else mt.predict_xgb(obj["model"], x)
        predictions[name] = mt.decode(raw, obj["scale"])
    return predictions


def baselines(train, n):
    w = train["o"] / train["pi"]
    mean = (w * np.nan_to_num(train["d"])).sum(0) / w.sum(0)
    prevalence = train["b"].mean(0)
    return {name: np.tile(np.r_[damage, total, prevalence], (n, 1))
            for name, damage, total in (("zero_baseline", np.zeros(3), 0.),
                                        ("training_constant", mean, train["y_all"].mean()))}


def bootstrap(pred, data, repeats=1000):
    """Source bootstrap conditional on fitted OOF predictions; no model refitting."""
    w = data["o"] / data["pi"]
    se = w * (pred[:, :3] - np.nan_to_num(data["d"]))**2
    b = data["b"].astype(bool)
    sources = np.unique(data["source_ids"])
    sums, counts = [], []
    for source in sources:
        rows = data["source_ids"] == source
        sums.append(np.stack((se[rows].sum(0), (se*b)[rows].sum(0), (se*~b)[rows].sum(0))))
        counts.append(np.stack((np.full(3, rows.sum()), b[rows].sum(0), (~b)[rows].sum(0))))
    sums, counts = np.asarray(sums), np.asarray(counts)
    ix = np.random.default_rng(42).integers(len(sources), size=(repeats, len(sources)))
    den = counts[ix].sum(1)
    values = np.sqrt(np.divide(sums[ix].sum(1), den, out=np.full(den.shape, np.nan), where=den > 0))
    result = {}
    for j, label in enumerate(("pooled", "active", "absent")):
        result[label] = {}
        for k, kind in enumerate(("amp", "phase", "noise")):
            finite = values[:, j, k][np.isfinite(values[:, j, k])]
            result[label][kind] = np.quantile(finite, [.025, .975]).tolist() if len(finite) else None
    return result


def slices(pred, data):
    masks = data["b"].dot(1 << np.arange(3))
    selections = {f"mask_{m}": masks == m for m in range(8)}
    if "params" in data:
        A, phase = data["params"][:, 0], np.abs(data["params"][:, 1])
        for name, keep in (("attenuation", A < 1), ("amplification", A > 1),
                           ("attenuation_A_below_0.5", A < .5),
                           ("attenuation_A_0.5_1", (A >= .5) & (A < 1)),
                           ("amplification_A_1_3", (A > 1) & (A <= 3)),
                           ("amplification_A_above_3", A > 3),
                           ("phase_0_30", (phase > 0) & (phase <= 30)),
                           ("phase_30_90", (phase > 30) & (phase <= 90)),
                           ("phase_90_180", phase > 90)):
            selections[name] = keep
    if "noise_ratio" in data:
        r = data["noise_ratio"]
        active = data["b"][:, 2].astype(bool)
        for name, keep in (("noise_r_below_0.03", r < .03),
                           ("noise_r_0.03_0.1", (r >= .03) & (r <= .1)),
                           ("noise_r_above_0.1", r > .1)):
            selections[name] = active & keep
    return {name: mt.task_metrics(pred[keep], mt.subset(data, keep))
            for name, keep in selections.items() if keep.any()}


def compare(dataset, output, folds=3, jobs=4, repeats=1000):
    if repeats < 1: raise ValueError("Bootstrap repeats must be positive")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise ValueError("Comparison output already exists; choose a fresh directory")
    full = mt.load_split(dataset, "train", "all20")
    groups = full["source_ids"]
    if folds < 2 or len(np.unique(groups)) < folds:
        raise ValueError("Need at least two outer folds and enough distinct sources")
    splits = list(GroupKFold(folds).split(full["X"], groups=groups))
    for outer, _ in splits:
        if len(np.unique(groups[outer])) < 4:
            raise ValueError("Each outer training partition needs at least four sources: three for fitting and one for tuning")
    summary = dict(folds=[], views={}, selection="Inner source holdout selects validation J; outer sources evaluate frozen models",
                   bootstrap="95% paired-source resampling of fixed OOF predictions, seed 42; excludes refitting and selection uncertainty",
                   bootstrap_repeats=repeats, primary_damage_target="signed CLEAN damage dC; eC aliases y_all; dirty targets are audit only")
    with np.load(dataset / "train.npz", allow_pickle=False) as z:
        present = set(map(str, z["feature_names"]))
    views = [v for v in mt.FEATURE_VIEWS if v != "all22" or {"clean.iterations", "model.nonzero_pixels"} <= present]
    for view in views:
        data = mt.load_split(dataset, "train", view)
        predictions = {name: np.full((len(groups), 7), np.nan) for name in (*mt.NAMES, "zero_baseline", "training_constant")}
        for fold, (outer, heldout) in enumerate(splits):
            sources = np.unique(groups[outer])
            tune_sources = np.random.default_rng(42 + fold).permutation(sources)[:max(1, len(sources)//5)]
            tune = outer[np.isin(groups[outer], tune_sources)]
            fitting = outer[~np.isin(groups[outer], tune_sources)]
            train, val, test = (mt.subset(data, ix) for ix in (fitting, tune, heldout))
            folder = output / view / f"fold_{fold}"
            print("Comparing", view, "outer fold", fold, flush=True)
            mt.fit_data(train, val, folder, jobs, view)
            frozen = frozen_predictions(folder, test)
            frozen.update(baselines(train, len(heldout)))
            for name in predictions: predictions[name][heldout] = frozen[name]
            if view == views[0]:
                summary["folds"].append(dict(fit_sources=sorted(set(map(str, groups[fitting]))),
                                             tuning_sources=sorted(set(map(str, groups[tune]))),
                                             evaluation_sources=sorted(set(map(str, groups[heldout])))))
        scores = {}
        for name, pred in predictions.items():
            if not np.isfinite(pred).all(): raise ValueError("Incomplete outer OOF predictions")
            scores[name] = dict(metrics=mt.task_metrics(pred, data), slices=slices(pred, data),
                                rmse_source_bootstrap=bootstrap(pred, data, repeats))
        summary["views"][view] = scores
        np.savez_compressed(output / f"predictions_{view}.npz", row_ids=data["row_ids"], source_ids=groups, **predictions)
    atomic_write_json(output / "results.json", summary)
    parts = ["<!doctype html><meta charset='utf-8'><title>Matched v2 feature comparison</title>",
             "<style>body{font:15px system-ui;margin:2rem}td,th{padding:.4rem;border:1px solid #ccc}table{border-collapse:collapse}</style>",
             "<h1>Matched Dataset v2: source-held-out feature comparison</h1>",
             "<p>All data come from the training split. Inner source holdouts select settings; outer sources evaluate frozen models. "
             "Damage is signed CLEAN recovery damage. Total eC and presence heads share the original training-only scaled J objective. "
             "Intervals resample sources with fixed predictions; they exclude refitting uncertainty.</p><p><a href='results.json'>All scores, mixture/severity slices, leakage metrics and fold membership</a></p>",
             "<table><tr><th>Features</th><th>Model</th><th>Type</th><th>Active RMSE [95% interval]</th><th>Absent RMSE</th><th>Pooled RMSE</th></tr>"]
    for view, models in summary["views"].items():
        for name, scores in models.items():
            for kind in ("amp", "phase", "noise"):
                m = scores["metrics"][kind]; interval = scores["rmse_source_bootstrap"]["active"][kind]
                ci = "unavailable" if interval is None else f"{interval[0]:.4g}, {interval[1]:.4g}"
                active = m['damage_active']['rmse']; absent = m['damage_absent']['rmse']
                fmt = lambda x: "—" if x is None else f"{x:.4g}"
                parts.append(f"<tr><td>{view}</td><td>{name}</td><td>{kind}</td><td>{fmt(active)} [{ci}]</td><td>{fmt(absent)}</td><td>{m['damage_rmse']:.4g}</td></tr>")
    parts.append("</table>")
    parts.append("<h2>False amplitude/phase attribution</h2><table><tr><th>Features</th><th>Model</th><th>Absent output</th><th>Present injection</th><th>Rows</th><th>RMSE</th></tr>")
    for view, models in summary["views"].items():
        for name, scores in models.items():
            for kind, other in (("amp", "phase"), ("phase", "amp")):
                m = scores["metrics"][kind]["leakage"][other]
                score = "—" if m["rmse"] is None else f"{m['rmse']:.4g}"
                parts.append(f"<tr><td>{view}</td><td>{name}</td><td>{kind}</td><td>{other}</td><td>{m['rows']}</td><td>{score}</td></tr>")
    parts.append("</table>")
    (output / "report.html").write_text("\n".join(parts))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--bootstrap-repeats", type=int, default=1000)
    args = p.parse_args()
    compare(args.dataset, args.output, args.folds, args.jobs, args.bootstrap_repeats)


if __name__ == "__main__":
    main()
