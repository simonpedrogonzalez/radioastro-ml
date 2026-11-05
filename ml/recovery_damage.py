"""Archived Dataset 1 recovery-damage labels and five combined8 regressors."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.wcs import FITSFixedWarning
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from ml.logreg import COMBINED8_FEATURES
from ml.regression import GRID, TARGETS
from scripts.clean_recovery_noise_sweep import geometry
from scripts.preprocessing.fits import load_fits_plane

DATASET = Path("collect/experiments/dataset_v1_20260921_all_psf")
CACHE = Path("ml/runs/20261028T065537442783Z_combined8_strategies")
NAMES = ("mlp", "xgb_shared", "xgb_independent", "xgb_xgb", "xgb_mlp")


def damage_loss(prediction, target, observed, probability):
    """Return L_d and per-output L_dk with N, not observed count, as denominator."""
    p, y, o, pi = map(np.asarray, (prediction, target, observed, probability))
    if p.shape != y.shape or y.shape != o.shape or o.shape != pi.shape or y.ndim != 2 or not len(y):
        raise ValueError("Damage arrays must have identical nonempty N x K shapes")
    if np.any((o != 0) & (o != 1)) or np.any(pi[o == 1] <= 0):
        raise ValueError("Invalid observation mask or probability")
    error = np.where(o == 1, p - np.where(o == 1, y, 0), 0)
    per = np.sum(np.square(error) * np.where(o == 1, 1 / np.where(o == 1, pi, 1), 0), axis=0) / len(y)
    return float(per.mean()), per.tolist()


def torch_damage_loss(prediction, target, observed, probability):
    error = torch.where(observed.bool(), prediction - torch.where(observed.bool(), target, 0), 0)
    weight = torch.where(observed.bool(), 1 / torch.where(observed.bool(), probability, 1), 0)
    return (weight * error.square()).sum() / prediction.numel()


def image_error(sample_dir):
    simulation_path = sample_dir / "simulation.json"
    manifest_path = sample_dir / "sample.json"
    sim = json.loads(simulation_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    component = sim["components"][0]
    if len(sim["components"]) != 1 or component["shape"] != "point" or component["fluxunit"] != "Jy":
        raise ValueError(f"Unsupported sky in {sample_dir}")
    direction = component["dir"].split()
    if direction[0] != "J2000" or len(direction) != 3:
        raise ValueError(f"Unsupported sky direction in {sample_dir}")
    ra, dec = (float(value.removesuffix("rad")) for value in direction[1:])
    flux = float(component["flux"][0])
    clean = load_fits_plane(sample_dir / manifest["products"]["clean"])
    residual = load_fits_plane(sample_dir / manifest["products"]["residual"])
    if clean.unit != "Jy/beam" or residual.unit != "Jy/beam" or clean.shape != residual.shape or clean.celestial_header != residual.celestial_header:
        raise ValueError(f"Incompatible restored/residual products in {sample_dir}")
    if not np.allclose(clean.beam, residual.beam, rtol=1e-6, atol=1e-10):
        raise ValueError(f"Beam mismatch within {sample_dir}")
    header = dict(clean.celestial_header, BMAJ=clean.beam[0], BMIN=clean.beam[1], BPA=clean.beam[2])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        _, truth = geometry(header, clean.shape, flux, ra, dec)
    model = clean.values.astype(np.float64) - residual.values.astype(np.float64)
    if not np.isfinite(model).all() or not np.isfinite(truth).all() or np.square(truth).sum() <= 0:
        raise ValueError(f"Invalid image domain in {sample_dir}")
    error = float(np.sqrt(np.square(model - truth).sum() / np.square(truth).sum()))
    hashes = {key: manifest["integrity"]["files"][manifest["products"][key]]["sha256"] for key in ("clean", "residual")}
    return error, sim, {"beam": clean.beam, "shape": clean.shape, "wcs": clean.celestial_header,
                        "flux": flux, "direction": direction, "sha256": hashes,
                        "simulation_sha256": hashlib.sha256(simulation_path.read_bytes()).hexdigest()}


def build_labels(splits, dataset, output):
    previous_path = output / "labels.json"
    previous = {row["sample_id"]: row for row in json.loads(previous_path.read_text())} if previous_path.exists() else {}
    records = {}
    ids = set().union(*(set(d["sample_ids"].tolist()) for d in splits.values()))
    if previous:
        ids = {str(sid) for sid in ids if not previous.get(str(sid), {}).get("valid", False)}
    for sid in ids.copy():
        ids.add(str(sid).split("_")[0] + "_not_corrupted")
    for n, sid in enumerate(sorted(ids), 1):
        folder = dataset / "samples" / sid
        try:
            y, sim, provenance = image_error(folder)
            records[sid] = {"sample_id": sid, "y": y, "simulation": sim, "provenance": provenance, "valid": True}
        except Exception as exc:
            records[sid] = {"sample_id": sid, "valid": False, "reason": str(exc)}
        if n % 200 == 0:
            print(f"Labeled {n}/{len(ids)} images", flush=True)
    rows = []
    for split, data in splits.items():
        for sid, source, kind in zip(data["sample_ids"], data["source_ids"], data["kinds"], strict=True):
            sid, source, kind = str(sid), str(source), str(kind)
            if previous.get(sid, {}).get("valid", False):
                rows.append(previous[sid])
                continue
            control = source + "_not_corrupted"
            current, base = records[sid], records[control]
            row = {"split": split, "sample_id": sid, "source_id": source, "kind": kind,
                   "control_id": control, "valid": bool(current["valid"] and base["valid"]),
                   "reason": current.get("reason") or base.get("reason", ""), "o": [1, 1, 1], "pi": [1., 1., 1.]}
            if row["valid"]:
                a, b = current["simulation"], base["simulation"]
                pa, pb = current["provenance"], base["provenance"]
                try:
                    if a["components"] != b["components"] or a["noise"]["seed"] != b["noise"]["seed"] or a["noise"]["simplenoise_jy"] != b["noise"]["simplenoise_jy"] or a["input_ms"] != b["input_ms"]:
                        raise ValueError("Sky, noise realization, nominal sigma or input visibility differ")
                    if pa["shape"] != pb["shape"]:
                        raise ValueError("Matched images have different pixel shapes")
                    for key in ("CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CDELT1", "CDELT2"):
                        if not np.isclose(float(pa["wcs"][key]), float(pb["wcs"][key]), rtol=5e-4, atol=1e-10):
                            raise ValueError(f"Matched images have incompatible {key}")
                    if kind == "noise" and ("noise_control" not in a or a["noise_control"]["baseline_noise_seed"] != b["noise"]["seed"]):
                        raise ValueError("Extra-noise baseline is not paired")
                    if kind == "clean" and sid != control:
                        raise ValueError("Ordinary control mismatch")
                    if kind in ("amp", "phase") and (len(a["stages"]) < 2 or a["stages"][1]["name"] != "corrupt_V"):
                        raise ValueError("Gain injection metadata missing")
                except Exception as exc:
                    row.update(valid=False, reason=str(exc))
            if row["valid"]:
                d = [0., 0., 0.]
                if kind != "clean": d[TARGETS.index(kind)] = current["y"] - base["y"]
                row.update(d=d, y=current["y"], control_y=base["y"],
                           presence=[int(kind == target) for target in TARGETS],
                           image_sha256=current["provenance"]["sha256"],
                           control_image_sha256=base["provenance"]["sha256"],
                           simulation_sha256=current["provenance"]["simulation_sha256"],
                           control_simulation_sha256=base["provenance"]["simulation_sha256"],
                           beam=current["provenance"]["beam"], control_beam=base["provenance"]["beam"])
            rows.append(row)
    (output / "labels.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
    invalid = [r for r in rows if not r["valid"]]
    if invalid:
        raise ValueError(f"{len(invalid)} unlabelable samples; see labels.json (first: {invalid[0]['sample_id']}: {invalid[0]['reason']})")
    return {r["sample_id"]: r for r in rows}


def load_features(dataset, cache):
    data = {}
    for split in ("train", "validation"):
        with np.load(cache / f"{split}_features.npz") as z:
            data[split] = {key: z[key] for key in z.files}
        if tuple(data[split]["feature_names"]) != COMBINED8_FEATURES:
            raise ValueError("Combined8 feature order differs")
    if [len(data[s]["X"]) for s in ("train", "validation")] != [2893, 608]:
        raise ValueError("Unexpected train/validation split sizes")
    sources = [set(data[s]["source_ids"]) for s in ("train", "validation")]
    if sources[0] & sources[1]:
        raise ValueError("Sources overlap across splits")
    for split, d in data.items():
        if set(d["strengths"][d["kinds"] == "noise"]) != {5., 10., 15., 20., 30., 40., 50., 100., 1000., 2000., 3000., 4000.}:
            raise ValueError(f"{split} is missing extended noise strengths")
    return data


def xgb(x, y, params, *, shared, jobs=4, mask=None):
    options = dict(tree_method="hist", random_state=42, n_jobs=jobs, **params)
    if shared:
        if mask is None:
            return XGBRegressor(objective="reg:squarederror", multi_strategy="multi_output_tree", **options).fit(x, y)
        from ml.regression_training import masked_squared_error
        return XGBRegressor(objective=masked_squared_error(mask), multi_strategy="multi_output_tree", base_score=y.mean(0).tolist(), **options).fit(x, y)
    if mask is None:
        return [XGBRegressor(objective="reg:squarederror", **options).fit(x, y[:, k]) for k in range(3)]
    return [XGBRegressor(objective="reg:squarederror", **options).fit(x[mask[:, k]], y[mask[:, k], k]) for k in range(3)]


def predict(model, x):
    return np.column_stack([m.predict(x) for m in model]) if isinstance(model, list) else np.asarray(model.predict(x)).reshape(len(x), 3)


def mlp_fit(x, y, val_x, val_y, *, train_o=None, train_pi=None, seed=42):
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(x)
    xx = torch.tensor(scaler.transform(x), dtype=torch.float32)
    vx = torch.tensor(scaler.transform(val_x), dtype=torch.float32)
    yy = torch.tensor(y, dtype=torch.float32)
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 3))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    o = torch.tensor(np.ones_like(y) if train_o is None else train_o, dtype=torch.float32)
    pi = torch.tensor(np.ones_like(y) if train_pi is None else train_pi, dtype=torch.float32)
    best, wait, epoch_best, state = float("inf"), 0, 0, None
    rng = np.random.default_rng(seed)
    for epoch in range(1, 301):
        model.train()
        for batch in np.array_split(rng.permutation(len(x)), max(1, int(np.ceil(len(x)/64)))):
            optimizer.zero_grad()
            loss = torch_damage_loss(model(xx[batch]), yy[batch], o[batch], pi[batch])
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            score = damage_loss(model(vx).numpy(), val_y, np.ones_like(val_y), np.ones_like(val_y))[0]
        if score < best - 1e-12:
            best, wait, epoch_best = score, 0, epoch
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 20: break
    model.load_state_dict(state); model.eval()
    return model, scaler, {"best_epoch": epoch_best, "validation_L_d": best}


def mlp_predict(pair, x):
    model, scaler = pair
    with torch.no_grad(): return model(torch.tensor(scaler.transform(x), dtype=torch.float32)).numpy().astype(float)


def write_report(run, data, labels, predictions, metrics, config):
    kind_colors = {"amp": "#0072B2", "phase": "#D55E00", "noise": "#CC79A7", "clean": "#009E73"}
    markers = {"amp": "o", "phase": "^", "noise": "x", "clean": "s"}
    model_colors = dict(zip(NAMES, ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#7B61A8")))
    model_names = {"mlp": "MLP", "xgb_shared": "Shared XGBoost", "xgb_independent": "Independent XGBoost",
                   "xgb_xgb": "XGBoost + XGBoost", "xgb_mlp": "XGBoost + MLP"}
    loss_names = (r"\(L_d\)", r"\(L_{d,\mathrm{amp}}\)", r"\(L_{d,\mathrm{phase}}\)", r"\(L_{d,\mathrm{noise}}\)")
    parts = ["<!doctype html><meta charset='utf-8'><title>Dataset 1 recovery damage</title>",
             "<script defer src='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js'></script>",
             "<style>body{font:16px system-ui;max-width:1500px;margin:auto;padding:2rem;color:#19242a}table{border-collapse:collapse;width:100%;font-size:.9rem}td,th{border:1px solid #ccd;padding:.45rem;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#eef3f6}strong{font-weight:750}img{max-width:100%}a{color:#175ea5}.note{color:#41515b}</style>",
             "<h1>Dataset 1 recovery damage: combined8</h1>",
             "<p><a href='config.json'>Configuration</a> · <a href='labels.json'>Label provenance</a> · <a href='predictions.csv'>Predictions CSV</a> · <a href='metrics.json'>Metrics JSON</a> · <a href='stage_features.npz'>Binder inputs</a></p>",
             "<h2>Summary of all</h2>",
             "<p>Existing Dataset 1, with 2,893 training samples from 100 sources and 608 validation samples from 21 separate sources. Both splits contain extra-noise injections at "
             r"\(\mathrm{SNR}_{\mathrm{corr}}=5,10,15,20,30,40,50,100,1000,2000,3000,4000\). "
             "The table compares signed damage predictions on all rows; bold is the lowest value in each column.</p>"]
    columns = [(split, k) for split in ("validation", "train") for k in range(4)]
    best = {(split, k): min((metrics[split][name]["L_d"] if k == 0 else metrics[split][name]["L_dk"][k-1]) for name in NAMES)
            for split, k in columns}
    parts.append("<table><tr><th rowspan='2'>Model</th><th colspan='4'>Validation</th><th colspan='4'>Train</th></tr><tr>" +
                 "".join(f"<th>{label}</th>" for _ in range(2) for label in loss_names) + "</tr>")
    for name in NAMES:
        cells = []
        for split, k in columns:
            value = metrics[split][name]["L_d"] if k == 0 else metrics[split][name]["L_dk"][k-1]
            shown = f"{value:.3e}"
            cells.append(f"<td><strong>{shown}</strong></td>" if np.isclose(value, best[split, k], rtol=1e-12, atol=0) else f"<td>{shown}</td>")
        parts.append(f"<tr><td>{model_names[name]}</td>{''.join(cells)}</tr>")
    parts.append(r"</table><p class='note'>\(L_d=\frac{1}{3}\sum_k L_{dk}\); each \(L_{dk}\) divides by all rows in its split. "
                 "All three labels are observed for every row, including known zeros for absent injections. "
                 "Both binders receive the same 14 inputs: eight combined8 features, three predictions from the independent per-class XGBoost regressors, "
                 "and three from the shared multiregression XGBoost model. Binder training uses source-held-out predictions from both first-stage models; "
                 "validation uses predictions from those models fitted on all training sources.</p>")
    for split in ("validation", "train"):
        d = data[split]
        y = np.asarray(d["d"])
        kinds = np.asarray(d["kinds"])
        parts.append(f"<h2>{split.title()}</h2><p>{len(y):,} samples; active amp/phase/noise: " +
                     "/".join(str(int(np.sum(kinds == k))) for k in TARGETS) +
                     f". Observed labels per output: {len(y):,}.</p>")
        fig, axes = plt.subplots(5, 3, figsize=(10.85, 16), sharex=True, sharey=True)
        values = np.concatenate([y.ravel(), *(predictions[split][name].ravel() for name in NAMES)])
        lo, hi = float(values.min()), float(values.max())
        pad = max((hi - lo) * .03, 1e-8)
        limits = (lo-pad, hi+pad)
        for i, name in enumerate(NAMES):
            for j, target in enumerate(TARGETS):
                ax = axes[i, j]
                for kind in ("amp", "phase", "noise", "clean"):
                    selected = kinds == kind
                    if selected.any():
                        ax.scatter(y[selected, j], predictions[split][name][selected, j],
                                   marker=markers[kind], color=kind_colors[kind], alpha=.6, s=13, label=kind,
                                   linewidths=.8 if kind == "noise" else None)
                ax.plot(limits, limits, "k--", lw=.8)
                ax.axhline(0, color="#aaa", lw=.5); ax.axvline(0, color="#aaa", lw=.5)
                ax.set(xlim=limits, ylim=limits)
                ax.set_aspect("equal", adjustable="box")
                ax.grid(color="#c8cfd3", alpha=.55, linewidth=.5)
                ax.tick_params(axis="x", labelbottom=i == len(NAMES)-1, bottom=i == len(NAMES)-1,
                               top=False, labelsize=8, pad=2)
                ax.tick_params(axis="y", labelleft=j == 0, left=j == 0,
                               right=False, labelsize=8, pad=2)
                if i == 0: ax.set_title(target.capitalize(), fontsize=12, fontweight="bold")
                if j == 0:
                    row_label = model_names[name].replace("Independent XGBoost", "Independent\nXGBoost").replace("XGBoost + ", "XGBoost +\n")
                    ax.set_ylabel(row_label, rotation=0, ha="right", va="center", labelpad=32, fontsize=9)
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        fig.subplots_adjust(left=.22, right=.99, bottom=.07, top=.94, wspace=0, hspace=0)
        fig.legend(handles, legend_labels, loc="upper center", ncol=4, frameon=False,
                   bbox_to_anchor=(.6, .985))
        fig.supxlabel("True damage", fontsize=11, y=.018)
        fig.supylabel("Predicted damage", fontsize=11, x=.012)
        fig.savefig(run / f"{split}_diagonal.png", dpi=150, bbox_inches="tight"); plt.close(fig)
        parts.append(f"<h3>True versus predicted damage</h3><p class='note'>All panels share the same numeric range on both axes. "
                     "Color and shape indicate injected kind; dashed line is exact prediction. Tick labels appear on the bottom and left edges of the contiguous panel grid.</p>"
                     f"<p><a href='{split}_diagonal.png'><img src='{split}_diagonal.png' alt='{split} diagonal plots'></a></p>")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout="constrained")
        for j, target in enumerate(TARGETS):
            ax = axes[j]
            selected = kinds == target
            levels = np.array(sorted(set(d["strengths"][selected])), dtype=float)
            def curve(values):
                groups = [np.asarray(values[selected & (d["strengths"] == level), j], float) for level in levels]
                return np.array([a.mean() for a in groups]), np.array([a.std(ddof=1) for a in groups])
            true_mean, true_sd = curve(y)
            ax.plot(levels, true_mean, color="black", marker="o", ms=3, lw=1.7, label="True")
            ax.fill_between(levels, true_mean-true_sd, true_mean+true_sd, color="black", alpha=.13)
            for name in NAMES:
                mean, sd = curve(predictions[split][name])
                color = model_colors[name]
                ax.plot(levels, mean, color=color, marker="o", ms=2.5, lw=1, label=model_names[name])
                ax.fill_between(levels, mean-sd, mean+sd, color=color, alpha=.075)
            ax.axhline(0, color="#999", lw=.5)
            ax.set(xscale="log", title=target.capitalize(), xlabel=r"Requested $\mathrm{SNR}_{\mathrm{corr}}$", ylabel="Mean signed damage")
            ax.grid(alpha=.15)
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(.5, 1.07), fontsize=8)
        fig.savefig(run / f"{split}_mean_by_level.png", dpi=150, bbox_inches="tight"); plt.close(fig)
        parts.append(f"<h3>Mean by requested injection level</h3><p class='note'>Lines are source means; shaded areas are ±1 sample standard deviation across sources at each level. "
                     "Each source contributes one sample per level and injected kind. The injected level is grouping metadata, while damage is the target.</p>"
                     f"<p><a href='{split}_mean_by_level.png'><img src='{split}_mean_by_level.png' alt='{split} mean and standard deviation by level'></a></p>")
        parts.append(r"<h3>Extra-noise label scale</h3><table><tr><th>Requested \(\mathrm{SNR}_{\mathrm{corr}}\)</th><th>Rows</th><th>Mean true \(d_{\mathrm{noise}}\)</th><th>Sample \(\sigma\)</th></tr>")
        for level in sorted(set(d["strengths"][kinds == "noise"])):
            values = y[(kinds == "noise") & (d["strengths"] == level), 2]
            parts.append(f"<tr><td>{level:g}</td><td>{len(values)}</td><td>{values.mean():.6g}</td><td>{values.std(ddof=1):.6g}</td></tr>")
        parts.append("</table>")
    parts.append(r"<p class='note'>Labels use \(d_k=y_{\mathrm{injected}}-y_{\mathrm{matched\ control}}\), where \(y=e_{TM,2}\). "
                 r"The archived fitted model is approximated by \(C=\mathrm{float64}(\mathrm{clean})-\mathrm{float64}(\mathrm{residual})\); "
                 "float32 FITS rounding remains. Each image uses its recorded WCS and restoring beam. "
                 r"A requested extra-noise \(\mathrm{SNR}_{\mathrm{corr}}=4000\) is a normalized visibility perturbation norm, not 4000 times the per-visibility thermal noise. "
                 "Its median equivalent extra-noise standard deviation is about 1.70 times the baseline in validation. "
                 "The resulting mean validation damage is about 0.00204; the observed increase is visible in the level table and mean ±σ plot.</p>")
    (run / "report.html").write_text("\n".join(parts))


def report_from_run(run):
    """Regenerate figures and HTML from saved labels, features and predictions."""
    run = Path(run)
    labels = {row["sample_id"]: row for row in json.loads((run / "labels.json").read_text())}
    data = {}
    for split in ("train", "validation"):
        with np.load(run / f"{split}_features_labels.npz") as saved:
            data[split] = {key: saved[key] for key in saved.files}
    predictions = {split: {} for split in data}
    with (run / "predictions.csv").open() as file:
        rows = list(csv.DictReader(file))
    lookup = {(row["split"], row["model"], row["sample_id"]): row for row in rows}
    for split, values in data.items():
        for name in NAMES:
            predictions[split][name] = np.array([[float(lookup[split, name, str(sid)][f"pred_{target}"])
                                                   for target in TARGETS] for sid in values["sample_ids"]])
    metrics = json.loads((run / "metrics.json").read_text())
    config = json.loads((run / "config.json").read_text())
    write_report(run, data, labels, predictions, metrics, config)


def experiment(dataset, cache, output, jobs=4, run=None):
    dataset, cache = dataset.resolve(), cache.resolve()
    run = Path(run).resolve() if run is not None else output.resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_recovery_damage_combined8")
    run.mkdir(parents=True, exist_ok=True)
    print("Run:", run, flush=True)
    data = load_features(dataset, cache)
    labels = build_labels(data, dataset, run)
    for split, d in data.items():
        d["d"] = np.array([labels[str(sid)]["d"] for sid in d["sample_ids"]], float)
        d["o"] = np.ones_like(d["d"]); d["pi"] = np.ones_like(d["d"])
        np.savez_compressed(run / f"{split}_features_labels.npz", **d)
    tr, va = (data[s] for s in ("train", "validation"))
    x, y, vx, vy = tr["X"], tr["d"], va["X"], va["d"]
    config = {"dataset": str(dataset), "combined8_cache": str(cache), "models": list(NAMES), "seed": 42,
              "grid": GRID, "mlp": {"widths": [64, 64], "activation": "ReLU", "optimizer": "Adam", "lr": 1e-3, "batch_size": 64, "max_epochs": 300, "patience": 20},
              "feature_names": list(COMBINED8_FEATURES), "label": "archived clean minus residual approximation", "selection": "validation L_d"}
    models = {}
    model, scaler, info = mlp_fit(x, y, vx, vy)
    models["mlp"] = (model, scaler); config["mlp_fit"] = info
    for name, shared in (("xgb_shared", True), ("xgb_independent", False)):
        scores = []
        for params in ParameterGrid(GRID):
            candidate = xgb(x, y, params, shared=shared, jobs=jobs)
            score = damage_loss(predict(candidate, vx), vy, va["o"], va["pi"])[0]
            scores.append({"params": params, "validation_L_d": score})
        choice = min(scores, key=lambda z: z["validation_L_d"])
        config[name] = {"selection": scores, "best": choice}
        models[name] = xgb(x, y, choice["params"], shared=shared, jobs=jobs)
        print(name, choice, flush=True)
    # The binders consume the predictions of both first-stage models. Training
    # predictions are held out by source, as in the earlier stacking protocol.
    from ml.combined8_strategies import source_folds
    folds = source_folds(tr["source_ids"])
    oof = np.full((len(x), 6), np.nan)
    for train, holdout in folds:
        independent = xgb(x[train], y[train], config["xgb_independent"]["best"]["params"], shared=False, jobs=jobs)
        shared = xgb(x[train], y[train], config["xgb_shared"]["best"]["params"], shared=True, jobs=jobs)
        oof[holdout] = np.column_stack([predict(independent, x[holdout]), predict(shared, x[holdout])])
    if not np.isfinite(oof).all(): raise ValueError("Incomplete source-held-out expert matrix")
    stage_train = np.column_stack([x, oof])
    stage_val = np.column_stack([vx, predict(models["xgb_independent"], vx), predict(models["xgb_shared"], vx)])
    stage_names = [*COMBINED8_FEATURES, *(f"independent_{target}" for target in TARGETS),
                   *(f"shared_{target}" for target in TARGETS)]
    np.savez_compressed(run / "stage_features.npz", train=stage_train, validation=stage_val,
                        train_sample_ids=tr["sample_ids"], validation_sample_ids=va["sample_ids"],
                        feature_names=np.asarray(stage_names))
    config["stacking"] = {"binder_inputs": stage_names, "training_predictions": "three source-held-out folds",
                          "validation_predictions": "first-stage models fitted on all training sources",
                          "folds": [{"train_sources": sorted(set(tr["source_ids"][a])),
                                     "held_out_sources": sorted(set(tr["source_ids"][b]))} for a, b in folds]}
    scores = []
    for params in ParameterGrid(GRID):
        candidate = xgb(stage_train, y, params, shared=True, jobs=jobs)
        scores.append({"params": params, "validation_L_d": damage_loss(predict(candidate, stage_val), vy, va["o"], va["pi"])[0]})
    choice = min(scores, key=lambda z: z["validation_L_d"])
    config["xgb_xgb"] = {"selection": scores, "best": choice}
    models["xgb_xgb"] = xgb(stage_train, y, choice["params"], shared=True, jobs=jobs)
    stage_mlp, stage_scaler, info = mlp_fit(stage_train, y, stage_val, vy)
    models["xgb_mlp"] = (stage_mlp, stage_scaler); config["xgb_mlp"] = info
    for name, model in models.items():
        if isinstance(model, tuple):
            torch.save(model[0].state_dict(), run / f"{name}.pt")
            np.savez(run / f"{name}_scaler.npz", mean=model[1].mean_, scale=model[1].scale_)
        else:
            for k, estimator in enumerate(model if isinstance(model, list) else [model]):
                estimator.save_model(run / f"{name}_{TARGETS[k] if isinstance(model,list) else 'all'}.json")
    predictions, metrics, rows = {}, {}, []
    for split, d, stage in (("train", tr, stage_train), ("validation", va, stage_val)):
        predictions[split], metrics[split] = {}, {}
        for name in NAMES:
            model = models[name]
            xx = stage if name.startswith("xgb_xgb") or name == "xgb_mlp" else d["X"]
            p = mlp_predict(model, xx) if isinstance(model, tuple) else predict(model, xx)
            predictions[split][name] = p
            loss, per = damage_loss(p, d["d"], d["o"], d["pi"])
            metrics[split][name] = {"rows": len(p), "L_d": loss, "L_dk": per,
                                    "observed": d["o"].sum(0).astype(int).tolist(),
                                    "active": [int(np.sum(d["kinds"] == k)) for k in TARGETS]}
            for i, sid in enumerate(d["sample_ids"]):
                row = {"split": split, "model": name, "sample_id": str(sid), "source_id": str(d["source_ids"][i]), "kind": str(d["kinds"][i]), "strength": float(d["strengths"][i])}
                for k, target in enumerate(TARGETS): row.update({f"true_{target}": float(d["d"][i,k]), f"pred_{target}": float(p[i,k]), f"o_{target}": 1, f"pi_{target}": 1.})
                rows.append(row)
    with (run / "predictions.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run / "config.json").write_text(json.dumps(config, indent=2))
    write_report(run, data, labels, predictions, metrics, config)
    print("Report:", run / "report.html", flush=True)
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=Path("ml/runs"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--run", type=Path, help="Resume an existing label run")
    parser.add_argument("--report-only", type=Path, help="Regenerate HTML and plots from saved artifacts")
    args = parser.parse_args()
    if args.report_only:
        report_from_run(args.report_only)
    else:
        experiment(args.dataset, args.cache, args.output, args.jobs, args.run)
