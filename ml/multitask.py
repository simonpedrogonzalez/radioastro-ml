"""Dataset v2 signed damage, total error and multilabel presence models."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import expit, xlogy
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.preprocessing import StandardScaler

from ml.logreg import COMBINED8_FEATURES
from ml.regression import GRID
from scripts.preprocessing.schema import atomic_write_json

NAMES = ("mlp", "xgb_shared", "xgb_independent", "xgb_xgb", "xgb_mlp")
# Presence outputs are sigmoid probabilities that each corruption type was injected (labels b).
# They are not the dirty-image perturbation p stored in the matched dataset.
OUTPUTS = ("d_amp", "d_phase", "d_noise", "y_all", "presence_amp", "presence_phase", "presence_noise")
FEATURE_VIEWS = ("combined8", "combined8_dirty6", "combined8_model6", "all20", "all22")


def load_split(root, split, feature_view="combined8"):
    with np.load(root / f"{split}.npz", allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    n = len(data["X"])
    if "eC" in data:
        data["y_all"] = data["eC"]
        plan_path = root / "plan.json"
        if plan_path.exists() and "params" in data:
            plan = json.loads(plan_path.read_text())
            snr = {p["id"]: p["nominal_source_snr"] for p in plan["prepared_sources"]}
            # Severity metadata for reporting only; never appended to X.
            data["noise_ratio"] = data["params"][:, 2] / np.asarray([snr[str(s)] for s in data["source_ids"]])
    names = tuple(map(str, data.get("feature_names", COMBINED8_FEATURES)))
    if feature_view not in FEATURE_VIEWS:
        raise ValueError(f"Unknown feature view: {feature_view}")
    from scripts.dataset_v2_matched_extract import CLEAN2, DIRTY6, MODEL6
    selected = (tuple(COMBINED8_FEATURES) + (DIRTY6 if feature_view in ("combined8_dirty6", "all20", "all22") else ())
                + (MODEL6 if feature_view in ("combined8_model6", "all20", "all22") else ()) + (CLEAN2 if feature_view == "all22" else ()))
    if data["X"].shape != (n, len(names)) or not set(selected) <= set(names) or len(set(names)) != len(names):
        raise ValueError("Feature schema does not contain the requested view")
    data["X"] = data["X"][:, [names.index(name) for name in selected]]
    data["feature_names"] = np.asarray(selected)
    if data["d"].shape != (n, 3) or data["o"].shape != (n, 3) or data["pi"].shape != (n, 3) or data["b"].shape != (n, 3):
        raise ValueError("Invalid v2 tabular shapes")
    if np.isinf(data["X"]).any() or not np.isfinite(data["y_all"]).all():
        raise ValueError("Nonfinite full-case features or total error")
    if not np.isfinite(data["d"][data["o"] == 1]).all() or not np.isnan(data["d"][data["o"] == 0]).all():
        raise ValueError("Sparse damage encoding changed")
    if not np.all((data["pi"] > 0) & (data["pi"] <= 1)):
        raise ValueError("Invalid observation probabilities")
    return data


def scales(train):
    n = len(train["X"])
    w = train["o"] / train["pi"]
    d0 = np.where(train["o"], train["d"], 0.)
    if not np.all(((train["b"] == 1) & (train["o"] == 1)).sum(0) > 0):
        raise ValueError("Every fitting partition needs observed active damage for each class")
    sd = np.maximum(1e-3, np.sqrt((w * d0**2).sum(0) / n))
    sy = max(1e-3, float(np.sqrt(np.mean(train["y_all"]**2))))
    return np.r_[sd, sy]


def targets(data, scale):
    d = np.where(data["o"], data["d"], 0.) / scale[:3]
    return np.column_stack((d, data["y_all"] / scale[3], data["b"]))


def raw_losses(raw, data, scale):
    raw = np.asarray(raw, float)
    if raw.shape != (len(data["X"]), 7):
        raise ValueError("Expected seven raw outputs")
    target = targets(data, scale)
    w = data["o"] / data["pi"]
    ld_scaled = float(np.sum(w * np.square(raw[:, :3] - target[:, :3])) / (len(raw) * 3))
    ly_scaled = float(np.mean(np.square(raw[:, 3] - target[:, 3])))
    lb = float(np.mean(np.logaddexp(0, raw[:, 4:]) - data["b"] * raw[:, 4:]))
    decoded = decode(raw, scale)
    d_err = decoded[:, :3] - np.where(data["o"], data["d"], 0.)
    ld = float(np.sum(w * np.square(d_err)) / (len(raw) * 3))
    ly = float(np.mean(np.square(decoded[:, 3] - data["y_all"])))
    active, absent = damage_subset_mse(d_err, data)
    return dict(J=ld_scaled + ly_scaled + lb, L_d_scaled=ld_scaled, L_y_scaled=ly_scaled,
                L_d=ld, L_y=ly, L_b=lb,
                L_dk=[float(x) for x in (w * np.square(d_err)).sum(0) / len(raw)],
                L_dk_active=active, L_dk_absent=absent)


def damage_subset_mse(d_err, data):
    """Physical damage MSE per type, separately over active and absent rows.

    Active rows (b=1) are observed sparsely, so their inverse-probability sum is
    divided by the number of active rows (an unbiased dense active-row mean).
    Absent rows (b=0) have known zero damage and are always observed.
    """
    w = data["o"] / data["pi"]
    b = data["b"].astype(bool)
    active, absent = [], []
    for k in range(3):
        on, off = b[:, k], ~b[:, k]
        active.append(float(np.sum(w[on, k] * d_err[on, k]**2) / on.sum()) if on.any() else None)
        absent.append(float(np.mean(d_err[off, k]**2)) if off.any() else None)
    return active, absent


def decode(raw, scale):
    raw = np.asarray(raw, float)
    return np.column_stack((raw[:, :3] * scale[:3], raw[:, 3] * scale[3], expit(raw[:, 4:])))


def mixed_derivatives(raw, target, observed, probability):
    """Derivatives of summed N*J in scaled regression units."""
    raw, target = np.asarray(raw, float), np.asarray(target, float)
    observed, probability = np.asarray(observed, float), np.asarray(probability, float)
    safe = np.where(observed > 0, target[:, :3], 0.)
    w = observed / probability
    grad = np.empty_like(raw)
    hess = np.empty_like(raw)
    grad[:, :3] = (2 / 3) * w * (raw[:, :3] - safe)
    hess[:, :3] = (2 / 3) * w
    grad[:, 3] = 2 * (raw[:, 3] - target[:, 3]); hess[:, 3] = 2
    p = expit(raw[:, 4:])
    grad[:, 4:] = (p - target[:, 4:]) / 3
    hess[:, 4:] = np.maximum(p * (1 - p), 1e-6) / 3
    if not np.isfinite(grad).all() or not np.isfinite(hess).all():
        raise ValueError("Nonfinite mixed objective derivatives")
    return grad, hess


def _xgb_parameters(params, jobs, shared):
    return dict(tree_method="hist", multi_strategy="multi_output_tree" if shared else "one_output_per_tree",
                num_target=7 if shared else 1, base_score=0, disable_default_eval_metric=1,
                nthread=jobs, seed=42, max_depth=params["max_depth"], eta=params["learning_rate"],
                reg_lambda=1)


def fit_xgb(x, train, params, scale, *, shared, jobs=4):
    import xgboost as xgb

    target = targets(train, scale)
    w = train["o"] / train["pi"]
    if shared:
        matrix = xgb.DMatrix(x, label=target)
        def objective(raw, matrix):
            return mixed_derivatives(raw, target, train["o"], train["pi"])
        return xgb.train(_xgb_parameters(params, jobs, True), matrix,
                         num_boost_round=params["n_estimators"], obj=objective)
    models = []
    for j in range(7):
        label = target[:, j]
        if j < 4:
            matrix = xgb.DMatrix(x, label=label)
            weight = w[:, j] if j < 3 else np.ones(len(x))
            factor = 2 / 3 if j < 3 else 2
            def objective(raw, matrix, *, expected=label, weight=weight, factor=factor):
                return factor * weight * (raw - expected), factor * weight
            model = xgb.train({k: v for k, v in _xgb_parameters(params, jobs, False).items() if k != "multi_strategy"},
                              matrix, num_boost_round=params["n_estimators"], obj=objective)
        else:
            matrix = xgb.DMatrix(x, label=label)
            options = {k: v for k, v in _xgb_parameters(params, jobs, False).items() if k != "multi_strategy"}
            options.update(objective="binary:logistic", base_score=.5)
            model = xgb.train(options, matrix, num_boost_round=params["n_estimators"])
        models.append(model)
    return models


def predict_xgb(model, x):
    import xgboost as xgb
    matrix = xgb.DMatrix(x)
    if isinstance(model, list):
        return np.column_stack([m.predict(matrix, output_margin=True) for m in model])
    return np.asarray(model.predict(matrix, output_margin=True)).reshape(len(x), 7)


def fit_mlp(x, train, val_x, val, *, seed=42, max_epochs=300, patience=20, fixed_epochs=None):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    # Missing model/peak ratios are imputed using fitting rows only, including OOF folds.
    count = np.isfinite(x).sum(0)
    fill = np.divide(np.nansum(x, axis=0), count, out=np.zeros(x.shape[1]), where=count > 0)
    x = np.where(np.isnan(x), fill, x)
    val_x = np.where(np.isnan(val_x), fill, val_x)
    scaler = StandardScaler().fit(x)
    xx = torch.tensor(scaler.transform(x), dtype=torch.float32)
    vx = torch.tensor(scaler.transform(val_x), dtype=torch.float32)
    scale = scales(train)
    target = torch.tensor(targets(train, scale), dtype=torch.float32)
    w = torch.tensor(train["o"] / train["pi"], dtype=torch.float32)
    model = nn.Sequential(nn.Linear(x.shape[1], 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 7))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, best_epoch, best_state, waiting = float("inf"), 0, None, 0
    rng = np.random.default_rng(seed)
    epochs = fixed_epochs or max_epochs
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in np.array_split(rng.permutation(len(x)), max(1, math.ceil(len(x) / 64))):
            optimizer.zero_grad()
            pred = model(xx[batch])
            ld = (w[batch] * (pred[:, :3] - target[batch, :3]).square()).sum() / (len(batch) * 3)
            ly = (pred[:, 3] - target[batch, 3]).square().mean()
            lb = nn.functional.binary_cross_entropy_with_logits(pred[:, 4:], target[batch, 4:])
            (ld + ly + lb).backward(); optimizer.step()
        if fixed_epochs is not None:
            continue
        model.eval()
        with torch.no_grad():
            score = raw_losses(model(vx).numpy(), val, scale)["J"]
        if score < best - 1e-6:
            best, best_epoch, waiting = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waiting += 1
            if waiting >= patience:
                break
    if fixed_epochs is None:
        model.load_state_dict(best_state)
    else:
        best_epoch = fixed_epochs
    model.eval()
    return dict(model=model, scaler=scaler, fill=fill, scale=scale, best_epoch=best_epoch)


def predict_mlp(fitted, x):
    import torch
    x = np.where(np.isnan(x), fitted.get("fill", np.zeros(x.shape[1])), x)
    with torch.no_grad():
        return fitted["model"](torch.tensor(fitted["scaler"].transform(x), dtype=torch.float32)).numpy().astype(float)


def load_frozen(output, name):
    """Load one selected model without fitting or touching evaluation labels."""
    summary = json.loads((output / "results.json").read_text())
    scale = np.asarray(summary["selected"][name]["scale"], dtype=float)
    if name in ("mlp", "xgb_mlp"):
        import torch
        from torch import nn
        saved = torch.load(output / f"{name}.pt", map_location="cpu", weights_only=False)
        features = saved["state"]["0.weight"].shape[1]
        model = nn.Sequential(nn.Linear(features, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 7))
        model.load_state_dict(saved["state"]); model.eval()
        scaler = StandardScaler()
        scaler.mean_ = saved["mean"]; scaler.scale_ = saved["std"]
        scaler.var_ = scaler.scale_**2; scaler.n_features_in_ = features
        np.testing.assert_array_equal(scale, saved["scale"])
        return dict(model=model, scaler=scaler, fill=saved.get("fill", np.zeros(features)), scale=scale)
    import xgboost as xgb
    count = 7 if name == "xgb_independent" else 1
    models = []
    for j in range(count):
        model = xgb.Booster()
        model.load_model(output / f"{name}_{j}.json")
        models.append(model)
    return dict(model=models if count == 7 else models[0], scale=scale)


def task_metrics(prediction, data):
    """Scientific scores in physical units; active unknown damage stays masked."""
    from sklearn.metrics import average_precision_score

    n = len(data["X"])
    metrics = {}
    w = data["o"] / data["pi"]
    for k, kind in enumerate(("amp", "phase", "noise")):
        observed = data["o"][:, k].astype(bool)
        target = np.where(observed, data["d"][:, k], 0.)
        error = prediction[:, k] - target
        mse = float(np.sum(w[:, k] * error**2) / n)
        mae = float(np.sum(w[:, k] * np.abs(error)) / n)
        weight_sum = w[:, k].sum()
        mean = float(np.sum(w[:, k] * target) / weight_sum) if weight_sum else np.nan
        variance = float(np.sum(w[:, k] * (target - mean)**2)) if weight_sum else 0.
        r2 = None if variance <= 1e-15 else float(1 - np.sum(w[:, k] * error**2) / variance)
        b = data["b"][:, k].astype(bool)
        positive = prediction[:, 4 + k] >= .5
        tp = int(np.sum(positive & b)); fp = int(np.sum(positive & ~b)); fn = int(np.sum(~positive & b))
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
        p = np.clip(prediction[:, 4 + k], 1e-12, 1 - 1e-12)
        subsets = {}
        for label, keep in (("active", b), ("absent", ~b)):
            if not keep.any():
                subsets[label] = dict(rows=0, rmse=None, mae=None, mean_prediction=None)
                continue
            wk = w[keep, k]
            ek = error[keep]
            sub_mean = float(np.sum(wk * target[keep]) / wk.sum()) if wk.sum() else np.nan
            sub_var = float(np.sum(wk * (target[keep] - sub_mean)**2)) if wk.sum() else 0.
            subsets[label] = dict(rows=int(keep.sum()), observed=int((observed & keep).sum()),
                                  rmse=math.sqrt(float(np.sum(wk * ek**2) / keep.sum())),
                                  mae=float(np.sum(wk * np.abs(ek)) / keep.sum()),
                                  r2=None if label == "absent" or sub_var <= 1e-15 else float(1 - np.sum(wk * ek**2) / sub_var),
                                  mean_prediction=float(np.mean(prediction[keep, k])))
        leakage = {}
        for j, other in enumerate(("amp", "phase", "noise")):
            if j == k: continue
            keep = ~b & data["b"][:, j].astype(bool)
            leakage[other] = dict(rows=int(keep.sum()), rmse=float(np.sqrt(np.mean(prediction[keep, k]**2))) if keep.any() else None)
        metrics[kind] = dict(damage_mae=mae, damage_rmse=math.sqrt(mse), damage_r2=r2, leakage=leakage,
                             damage_active=subsets["active"], damage_absent=subsets["absent"],
                             observed_damage=int(observed.sum()), observed_active=int((observed & b).sum()),
                             known_active=int(b.sum()), precision=precision, recall=recall, f1=f1,
                             average_precision=float(average_precision_score(b, p)) if b.any() and (~b).any() else None,
                             bce=float(np.mean(-(b * np.log(p) + (~b) * np.log(1-p)))),
                             brier=float(np.mean(np.square(p-b))))
    yerr = prediction[:, 3] - data["y_all"]
    variance = float(np.square(data["y_all"] - data["y_all"].mean()).sum())
    metrics["y_all"] = dict(mae=float(np.mean(np.abs(yerr))), rmse=float(np.sqrt(np.mean(yerr**2))),
                            r2=None if variance <= 1e-15 else float(1 - np.square(yerr).sum() / variance),
                            negative_predictions=int((prediction[:, 3] < 0).sum()))
    return metrics


def write_report(output, train, val, predictions, summary):
    import html
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assets = output / "report_assets"
    assets.mkdir(exist_ok=True)
    data = {"train": train, "val": val}
    columns = [(split, key) for split in ("val", "train") for key in ("J", "L_d", "L_y", "L_b")]
    minima = {(split, key): min(summary["losses"][split][name][key] for name in NAMES)
              for split, key in columns}
    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>Dataset v2 three-task models</title>",
             "<style>body{font:15px system-ui;max-width:1400px;margin:auto;padding:1.5rem;color:#1b2932}table{border-collapse:collapse}th,td{border:1px solid #ccd;padding:.4rem;text-align:right}th:first-child,td:first-child{text-align:left}img{max-width:100%}</style></head><body>",
             "<h1>Dataset v2: three-task models</h1><p>Train and validation only. Fully observed and sparse damage labels are supported; "
             "J uses training-only RMS scales. Validation selected hyperparameters. Binder train scores use source-held-out expert inputs.</p>",
             "<p><a href='results.json'>Results and selected settings</a> · <a href='stage_features.npz'>Saved OOF expert inputs</a></p>",
             "<h2>Summary of all tasks</h2><table><tr><th>Model</th>" + "".join(f"<th>{s} {k}</th>" for s,k in columns) + "</tr>"]
    for name in NAMES:
        cells = []
        for split, key in columns:
            value = summary["losses"][split][name][key]
            text = f"{value:.5g}"
            if value == minima[(split, key)]: text = f"<strong>{text}</strong>"
            cells.append(f"<td>{text}</td>")
        parts.append(f"<tr><td>{html.escape(name)}</td>{''.join(cells)}</tr>")
    parts.append("</table>")
    parts.append("<h2>False damage attributed to an absent type</h2><p>RMSE in damage units, conditional on another type being present.</p><table><tr><th>Model</th><th>Split</th><th>Absent type</th><th>Present type</th><th>Rows</th><th>RMSE</th></tr>")
    for split in ("val", "train"):
        for name in (*NAMES, "zero_baseline", "training_constant"):
            for kind in ("amp", "phase", "noise"):
                for other, metric in summary["metrics"][split][name][kind].get("leakage", {}).items():
                    score = "—" if metric["rmse"] is None else f"{metric['rmse']:.4g}"
                    parts.append(f"<tr><td>{name}</td><td>{split}</td><td>{kind}</td><td>{other}</td><td>{metric['rows']}</td><td>{score}</td></tr>")
    parts.append("</table>")
    parts.append("<h2>Damage error on active and absent rows</h2><p>Physical-unit RMSE per damage output. "
                 "<em>Active</em>: rows where that error type was injected (b=1); sparse labels use inverse-observation weights "
                 "divided by the number of active rows. <em>Absent</em>: rows without that error type, whose true damage is exactly 0, "
                 "so their RMSE measures how far predictions stray from zero. The summary L_d pools both. Bold marks the column minimum.</p>")
    subset_columns = [(split, kind, label) for split in ("val", "train") for kind in ("amp", "phase", "noise")
                      for label in ("active", "absent")]
    def subset_rmse(split, name, kind, label):
        return summary["metrics"][split][name][kind][f"damage_{label}"]["rmse"]
    subset_names = [name for name in (*NAMES, "zero_baseline", "training_constant") if all("damage_active" in summary["metrics"][split].get(name, {}).get("amp", {})
                                                                      for split in ("val", "train"))]
    subset_minima = {column: min((v for v in (subset_rmse(*column[:1], name, *column[1:]) for name in subset_names)
                                  if v is not None), default=None) for column in subset_columns}
    parts.append("<table><tr><th>Model</th>" + "".join(f"<th>{s} {k}<br>{l}</th>" for s, k, l in subset_columns) + "</tr>")
    for name in subset_names:
        cells = []
        for column in subset_columns:
            value = subset_rmse(column[0], name, column[1], column[2])
            text = "—" if value is None else f"{value:.4g}"
            if value is not None and value == subset_minima[column]: text = f"<strong>{text}</strong>"
            cells.append(f"<td>{text}</td>")
        parts.append(f"<tr><td>{html.escape(name)}</td>{''.join(cells)}</tr>")
    parts.append("</table>")
    parts.append("<h2>Validation and train diagonals</h2><p>Only observed damage labels are plotted. Colors and shapes denote the eight injection masks. Each column has its own scale, shared by both rows and all models, and labelled on both axes. "
                 "presence_* columns are predicted probabilities (sigmoid, 0 to 1) that the type was injected, against the true 0/1 label. They are not the dirty-image perturbation p of the dataset report.</p>")
    mask_colors = ("#555555", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#7B61A8", "#C49A00", "#333333")
    markers = ("s", "o", "^", "D", "v", "P", "X", "*")
    def observed_pair(split, name, k):
        truth = data[split]; prediction = predictions[split][name]
        if k < 3:
            keep = truth["o"][:, k].astype(bool)
            return truth["d"][keep, k], prediction[keep, k], truth["b"][keep].dot(1 << np.arange(3))
        if k == 3:
            return truth["y_all"], prediction[:, 3], truth["b"].dot(1 << np.arange(3))
        return truth["b"][:, k-4], prediction[:, k], truth["b"].dot(1 << np.arange(3))
    bounds = []
    for k in range(7):
        values = []
        for name in NAMES:
            for split in ("train", "val"):
                a, b, _ = observed_pair(split, name, k)
                values.extend((a, b))
        flat = np.concatenate([np.asarray(v, float) for v in values if len(v)])
        lo, hi = (0., 1.) if k >= 4 else (float(flat.min()), float(flat.max()))
        pad = .04 * (hi - lo or 1.)
        bounds.append((lo-pad, hi+pad))
    for name in NAMES:
        fig, axes = plt.subplots(2, 7, figsize=(20, 5.5), sharex="col", sharey="col")
        for row, split in enumerate(("val", "train")):
            for k in range(7):
                ax = axes[row, k]
                actual, predicted, masks = observed_pair(split, name, k)
                for mask in range(8):
                    keep = masks == mask
                    if keep.any():
                        ax.scatter(actual[keep], predicted[keep], s=7, alpha=.45,
                                   c=mask_colors[mask], marker=markers[mask], linewidths=0)
                lo, hi = bounds[k]
                ax.plot([lo, hi], [lo, hi], color="#444", lw=.8)
                ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.grid(alpha=.2)
                if k >= 4: ax.set_xticks((0, .5, 1)); ax.set_yticks((0, .5, 1))
                ax.tick_params(labelsize=7)
                if row == 0: ax.set_title(OUTPUTS[k])
                if row == 1: ax.set_xlabel("True (0/1)" if k >= 4 else "True")
                else: ax.tick_params(labelbottom=False)
                if k == 0: ax.set_ylabel(f"{split} predicted")
        fig.subplots_adjust(wspace=.35, hspace=.08)
        path = assets / f"diagonal_{name}.png"; fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
        parts.append(f"<h3>{html.escape(name)}</h3><img src='report_assets/{path.name}' alt='{name} train and validation seven-output diagonals'>")
    parts.append("</body></html>")
    (output / "report.html").write_text("".join(parts))


def subset(data, indices):
    n = len(data["X"])
    return {key: value[indices] if key != "feature_names" and getattr(value, "shape", ())[:1] == (n,) else value for key, value in data.items()}


def _fit_tree_family(train, val, name, jobs):
    scale = scales(train)
    candidates = []
    for params in ParameterGrid(GRID):
        model = fit_xgb(train["X"], train, params, scale,
                        shared=name == "xgb_shared", jobs=jobs)
        score = raw_losses(predict_xgb(model, val["X"]), val, scale)["J"]
        candidates.append(dict(params=params, J=score))
    best = min(candidates, key=lambda c: c["J"])
    model = fit_xgb(train["X"], train, best["params"], scale,
                    shared=name == "xgb_shared", jobs=jobs)
    return dict(model=model, scale=scale, params=best["params"], candidates=candidates)


def _expert_matrix(independent, shared, x):
    return np.column_stack((decode(predict_xgb(independent["model"], x), independent["scale"]),
                            decode(predict_xgb(shared["model"], x), shared["scale"])))


def fit(train_root, val_root, output, jobs=4, feature_view="combined8"):
    train, val = load_split(train_root, "train", feature_view), load_split(val_root, "val", feature_view)
    return fit_data(train, val, output, jobs, feature_view)


def fit_data(train, val, output, jobs=4, feature_view="combined8"):
    if set(train["source_ids"]) & set(val["source_ids"]):
        raise ValueError("Train/validation sources overlap")
    if len(set(train["source_ids"])) < 3:
        raise ValueError("Five-family fitting requires at least three training sources for OOF binders")
    if not np.array_equal(train["feature_names"], val["feature_names"]):
        raise ValueError("Train/validation feature schemas differ")
    width = train["X"].shape[1]
    output.mkdir(parents=True, exist_ok=True)
    fitted = {}
    for name in ("xgb_independent", "xgb_shared"):
        fitted[name] = _fit_tree_family(train, val, name, jobs)
        print("Selected", name, fitted[name]["params"], flush=True)
    fitted["mlp"] = fit_mlp(train["X"], train, val["X"], val)
    groups = np.asarray(train["source_ids"])
    folds = list(GroupKFold(n_splits=3).split(train["X"], groups=groups))
    oof = np.full((len(train["X"]), 14), np.nan)
    for a, b in folds:
        if set(groups[a]) & set(groups[b]):
            raise ValueError("OOF source leakage")
        part = subset(train, a)
        ind_scale, shared_scale = scales(part), scales(part)
        ind = dict(model=fit_xgb(part["X"], part, fitted["xgb_independent"]["params"], ind_scale,
                                 shared=False, jobs=jobs), scale=ind_scale)
        shared = dict(model=fit_xgb(part["X"], part, fitted["xgb_shared"]["params"], shared_scale,
                                    shared=True, jobs=jobs), scale=shared_scale)
        oof[b] = _expert_matrix(ind, shared, train["X"][b])
    if not np.isfinite(oof).all():
        raise ValueError("Incomplete OOF expert matrix")
    stage_train = dict(train, X=np.column_stack((train["X"], oof)))
    stage_val = dict(val, X=np.column_stack((val["X"], _expert_matrix(
        fitted["xgb_independent"], fitted["xgb_shared"], val["X"]))))
    if stage_train["X"].shape[1] != width + 14 or stage_val["X"].shape[1] != width + 14:
        raise ValueError("Binder must receive input features plus 14 expert predictions")
    fitted["xgb_xgb"] = _fit_tree_family(stage_train, stage_val, "xgb_shared", jobs)
    fitted["xgb_mlp"] = fit_mlp(stage_train["X"], stage_train, stage_val["X"], stage_val)
    np.savez_compressed(output / "stage_features.npz", train_oof=oof,
                        val_experts=stage_val["X"][:, width:],
                        train_row_ids=train["row_ids"], val_row_ids=val["row_ids"])
    predictions = {}
    results = {}
    task_scores = {}
    for split, data, stage in (("train", train, stage_train), ("val", val, stage_val)):
        predictions[split] = {}
        results[split] = {}
        task_scores[split] = {}
        for name in NAMES:
            obj = fitted[name]
            x = stage["X"] if name.startswith("xgb_") and name not in ("xgb_shared", "xgb_independent") else data["X"]
            if name == "xgb_mlp": x = stage["X"]
            raw = predict_mlp(obj, x) if name in ("mlp", "xgb_mlp") else predict_xgb(obj["model"], x)
            predictions[split][name] = decode(raw, obj["scale"])
            results[split][name] = raw_losses(raw, data, obj["scale"])
            task_scores[split][name] = task_metrics(predictions[split][name], data)
            np.savez_compressed(output / f"predictions_{split}_{name}.npz", row_ids=data["row_ids"],
                                raw=raw, decoded=predictions[split][name])
    training_weight = train["o"] / train["pi"]
    training_damage = np.where(train["o"], train["d"], 0.)
    constant_damage = (training_weight * training_damage).sum(0) / training_weight.sum(0)
    prevalence = np.clip(train["b"].mean(0), 1e-6, 1-1e-6)
    for split, data in (("train", train), ("val", val)):
        for name, damage, total in (("zero_baseline", np.zeros(3), 0.),
                                    ("training_constant", constant_damage, float(train["y_all"].mean()))):
            physical = np.column_stack((np.broadcast_to(damage, (len(data["X"]), 3)),
                                        np.full(len(data["X"]), total),
                                        np.broadcast_to(prevalence, (len(data["X"]), 3))))
            scale = scales(train)
            raw = np.column_stack((physical[:, :3] / scale[:3], physical[:, 3] / scale[3],
                                   np.broadcast_to(np.log(prevalence / (1-prevalence)), (len(data["X"]), 3))))
            results[split][name] = raw_losses(raw, data, scale)
            task_scores[split][name] = task_metrics(physical, data)
    # Save tree models and MLP/scalers with selected training-only scales.
    import torch
    for name, obj in fitted.items():
        if name in ("mlp", "xgb_mlp"):
            torch.save(dict(state=obj["model"].state_dict(), mean=obj["scaler"].mean_,
                            std=obj["scaler"].scale_, fill=obj["fill"], scale=obj["scale"], epoch=obj["best_epoch"]),
                       output / f"{name}.pt")
        else:
            models = obj["model"] if isinstance(obj["model"], list) else [obj["model"]]
            for j, model in enumerate(models):
                model.save_model(output / f"{name}_{j}.json")
    summary = dict(models=list(NAMES), outputs=list(OUTPUTS), feature_names=train["feature_names"].tolist(), feature_view=feature_view,
                   train_rows=len(train["X"]), val_rows=len(val["X"]),
                   train_sources=len(set(train["source_ids"])), val_sources=len(set(val["source_ids"])),
                   train_source_ids=sorted(set(map(str, train["source_ids"]))),
                   val_source_ids=sorted(set(map(str, val["source_ids"]))),
                   selected={name: dict(params=obj.get("params"), best_epoch=obj.get("best_epoch"),
                                        scale=obj["scale"].tolist()) for name, obj in fitted.items()},
                   losses=results, metrics=task_scores,
                   selection="validation J; train binder uses source-held-out OOF experts")
    atomic_write_json(output / "results.json", summary)
    for name in NAMES:
        restored = load_frozen(output, name)
        xx = stage_val["X"] if name in ("xgb_xgb", "xgb_mlp") else val["X"]
        repeated = (predict_mlp(restored, xx) if name in ("mlp", "xgb_mlp") else
                    predict_xgb(restored["model"], xx))
        with np.load(output / f"predictions_val_{name}.npz", allow_pickle=False) as saved:
            np.testing.assert_allclose(repeated, saved["raw"], rtol=0, atol=1e-6)
    write_report(output, train, val, predictions, summary)
    print("Multitask report:", output / "report.html", flush=True)
    return summary


def report_only(train_root, val_root, output):
    summary = json.loads((output / "results.json").read_text())
    view = summary.get("feature_view", "combined8")
    train, val = load_split(train_root, "train", view), load_split(val_root, "val", view)
    predictions = {}
    for split, data in (("train", train), ("val", val)):
        predictions[split] = {}
        for name in NAMES:
            with np.load(output / f"predictions_{split}_{name}.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["row_ids"], data["row_ids"])
                scale = np.asarray(summary["selected"][name]["scale"])
                np.testing.assert_allclose(saved["decoded"], decode(saved["raw"], scale), rtol=0, atol=1e-12)
                recomputed = raw_losses(saved["raw"], data, scale)
                for key in ("J", "L_d", "L_y", "L_b"):
                    np.testing.assert_allclose(recomputed[key], summary["losses"][split][name][key], rtol=1e-10)
                predictions[split][name] = saved["decoded"]
                summary["metrics"][split][name] = task_metrics(saved["decoded"], data)
    write_report(output, train, val, predictions, summary)
    print("Report:", output / "report.html", flush=True)


def evaluate_test(test_root, output):
    """One explicit frozen-model test evaluation, without selection or fitting."""
    summary = json.loads((output / "results.json").read_text())
    test = load_split(test_root, "test", summary.get("feature_view", "combined8"))
    if set(test["source_ids"]) & (set(summary["train_source_ids"]) | set(summary["val_source_ids"])):
        raise ValueError("Test shares a parent source with development data")
    target = output / "test_results.json"
    if target.exists():
        raise ValueError("Frozen test evaluation already exists; leave it unchanged")
    fitted = {name: load_frozen(output, name) for name in NAMES}
    experts = _expert_matrix(fitted["xgb_independent"], fitted["xgb_shared"], test["X"])
    stage = np.column_stack((test["X"], experts))
    results = {}
    for name in NAMES:
        obj = fitted[name]
        x = stage if name in ("xgb_xgb", "xgb_mlp") else test["X"]
        raw = predict_mlp(obj, x) if name in ("mlp", "xgb_mlp") else predict_xgb(obj["model"], x)
        decoded = decode(raw, obj["scale"])
        results[name] = dict(losses=raw_losses(raw, test, obj["scale"]), metrics=task_metrics(decoded, test))
        np.savez_compressed(output / f"predictions_test_{name}.npz", row_ids=test["row_ids"],
                            raw=raw, decoded=decoded)
    atomic_write_json(target, dict(status="frozen one-time test evaluation",
                                   test_rows=len(test["X"]), test_source_ids=sorted(set(map(str, test["source_ids"]))),
                                   prior_test_exposure="Dataset 1 test sources were examined in a superseded earlier experiment",
                                   results=results))
    print("Frozen test results:", target, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path)
    p.add_argument("--val", type=Path)
    p.add_argument("--test", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--feature-view", choices=FEATURE_VIEWS, default="combined8")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--evaluate-test", action="store_true")
    a = p.parse_args()
    if a.evaluate_test:
        if not a.test or a.report_only:
            p.error("--evaluate-test requires --test and excludes --report-only")
        evaluate_test(a.test, a.output)
    elif a.report_only:
        if not a.train or not a.val: p.error("--report-only requires --train and --val")
        report_only(a.train, a.val, a.output)
    else:
        if not a.train or not a.val: p.error("Fit requires --train and --val")
        fit(a.train, a.val, a.output, a.jobs, a.feature_view)


if __name__ == "__main__":
    main()
