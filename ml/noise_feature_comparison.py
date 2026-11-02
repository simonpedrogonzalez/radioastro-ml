"""Compare scalar noise regressors using combined6/combined8 on all V1 rows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import ParameterGrid
from xgboost import XGBRegressor

from ml.logreg import ALL_FEATURES, COMBINED8_FEATURES, SIMPLE_FEATURES
from ml.regression import GRID, write_json
from ml.simple_metrics import sigma_na_from_source, sigma_sna_units

VIEWS = {"combined6": ALL_FEATURES, "combined8": COMBINED8_FEATURES}
LABELS = ("Dynamic range / residual MAD", "Residual scaled MAD (Jy/beam)",
          "Residual peak / MAD", "Residual P99.5 / MAD", "log1p(even energy / MAD²)",
          "log1p(odd energy / MAD²)", "sigma_NA (Jy/beam)", "sigma_sNA_units (residual MAD / sigma_NA)")
COLORS = {"clean": "#009E73", "amp": "#0072B2", "phase": "#D55E00", "noise": "#CC79A7"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(truth, prediction, strengths=None):
    error = prediction - truth
    result = {"n": len(truth), "mae": float(np.abs(error).mean()),
              "rmse": float(np.sqrt(np.square(error).mean())),
              "mean_prediction": float(prediction.mean())}
    if np.all(truth > 0):
        if strengths is None:
            raise ValueError("Positive-noise metrics require requested injection strengths")
        result["relative_mae_percent"] = float(np.mean(np.abs(error) / strengths) * 100)
    return result


def fit_scalar(X, y, folds, jobs):
    def estimator(params):
        return XGBRegressor(objective="reg:squarederror", tree_method="hist",
                            random_state=42, n_jobs=jobs, **params)
    candidates = []
    for params in ParameterGrid(GRID):
        scores = []
        for a, b in folds:
            model = estimator(params).fit(X[a], y[a])
            prediction = np.maximum(0, model.predict(X[b]))
            scores.append(float(np.abs(prediction - y[b]).mean() / y[a].std()))
        candidates.append({"params": params, "fold_scores": scores, "mean_scaled_mae": float(np.mean(scores))})
    best = min(candidates, key=lambda row: row["mean_scaled_mae"])["params"]
    return estimator(best).fit(X, y), best, candidates


def compare(source, output, image_run, jobs=4):
    source, image_run = source.resolve(), image_run.resolve()
    old_config = json.loads((source / "config.json").read_text())
    if old_config.get("target_definition_version") != 2:
        raise ValueError("Use a cached run with measured extra-noise SNR_corr targets")
    dataset = Path(old_config["dataset"]).parent
    inputs = {str(path): digest(path) for path in
              (source / "config.json", source / "train_features.npz", source / "validation_features.npz")}
    manifest = json.loads((Path(json.loads((dataset / "report.json").read_text())["source_run"]) / "report.json").read_text())
    source_noise = {row["id"]: row for row in manifest["samples"]}
    inputs[str(Path(json.loads((dataset / "report.json").read_text())["source_run"]) / "report.json")] = digest(Path(json.loads((dataset / "report.json").read_text())["source_run"]) / "report.json")
    data, provenance, sigma_by_source = {}, [], {}
    for split in ("train", "validation"):
        with np.load(source / f"{split}_features.npz") as a:
            np.testing.assert_array_equal(a["feature_names"], ALL_FEATURES)
            d = {key: a[key] for key in ("X", "y", "sample_ids", "source_ids", "kinds", "strengths")}
        if set(d["kinds"]) != {"clean", "amp", "phase", "noise"}:
            raise ValueError("This comparison requires all four sample types in each split")
        d["y"] = d["y"][:, 2]
        sigma = []
        for sid, source_id, kind in zip(d["sample_ids"], d["source_ids"], d["kinds"], strict=True):
            source_id = str(source_id)
            if source_id not in source_noise:
                raise ValueError(f"Missing precursor image RMS for {source_id}")
            value = sigma_na_from_source(source_noise[source_id])
            ratio = sigma_sna_units(float(d["X"][len(sigma), 1]), value)
            sigma.append((value, ratio))
            provenance.append({"split": split, "sample_id": str(sid), "source_id": source_id,
                               "kind": str(kind), "sigma_NA": value, "sigma_sNA_units": ratio})
        d["X"] = np.column_stack((d["X"], sigma))
        assert np.all(d["y"][d["kinds"] != "noise"] == 0)
        assert np.all(d["y"][d["kinds"] == "noise"] > 0)
        data[split] = d
    assert not set(data["train"]["source_ids"]) & set(data["validation"]["source_ids"])
    groups = data["train"]["source_ids"]
    folds, seen = [], np.zeros(len(groups), int)
    for fold in old_config["cv_folds"]:
        a = np.flatnonzero(np.isin(groups, fold["train_sources"]))
        b = np.flatnonzero(np.isin(groups, fold["validation_sources"]))
        assert len(a) and len(b) and len(a) + len(b) == len(groups) and not set(groups[a]) & set(groups[b])
        folds.append((a, b)); seen[b] += 1
    np.testing.assert_array_equal(seen, np.ones(len(groups), int))
    run = output.resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_sigma_na_comparison")
    run.mkdir(parents=True)
    print(f"Run: {run}", flush=True)
    for split, d in data.items():
        np.savez_compressed(run / f"{split}_features.npz", **d, feature_names=COMBINED8_FEATURES)
    with (run / "sigma_na.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(provenance[0])); writer.writeheader(); writer.writerows(provenance)
    config = {"source_run": str(source), "image_source_run": str(image_run), "dataset": str(dataset),
              "features": {name: list(features) for name, features in VIEWS.items()},
              "simple_metrics": list(SIMPLE_FEATURES), "target": "Measured extra-noise SNR_corr; zero on clean/amp/phase",
              "sigma_NA": {"definition": "flag-adjusted natural-image RMS from precursor",
                           "unit": "Jy/beam", "source": "precursor predicted_image_rms_jy_per_beam"},
              "sigma_sNA_units": "residual scaled MAD / sigma_NA",
              "population": "All clean, amp, phase and noise rows; all eligible training rows contribute to the single noise loss",
              "splits": {split: {"rows": len(d["y"]), "sources": len(set(d["source_ids"])),
                                    "kinds": {str(k): int(sum(d["kinds"] == k)) for k in sorted(set(d["kinds"]))}}
                         for split, d in data.items()},
              "seed": 42, "jobs": jobs, "cv_folds": old_config["cv_folds"], "cv_grid": GRID,
              "cv_scoring": "Noise MAE / training-fold noise target std, mean over source folds; all rows",
              "best_params": {}, "clipping": "max(0, prediction)", "input_sha256": inputs,
              "versions": {name: version(name) for name in ("numpy", "scipy", "xgboost", "scikit-learn")},
              "code_sha256": {str(p): digest(p) for p in (Path(__file__).resolve(), Path("ml/simple_metrics.py").resolve(),
                                                         Path("ml/logreg.py").resolve(), Path("scripts/simulation/noise.py").resolve())}}
    results, predictions, cv_rows, rows = {}, {}, [], []
    for view, features in VIEWS.items():
        columns = [COMBINED8_FEATURES.index(name) for name in features]
        model, best, candidates = fit_scalar(data["train"]["X"][:, columns], data["train"]["y"], folds, jobs)
        model.save_model(run / f"{view}.json")
        restored = XGBRegressor(); restored.load_model(run / f"{view}.json")
        config["best_params"][view] = best
        cv_rows.extend({"model": view, **row} for row in candidates)
        results[view], predictions[view] = {}, {}
        for split, d in data.items():
            raw = model.predict(d["X"][:, columns])
            np.testing.assert_array_equal(raw, restored.predict(d["X"][:, columns]))
            pred = np.maximum(0, raw).astype(float)
            predictions[view][split] = pred
            noise = d["kinds"] == "noise"
            results[view][split] = {"overall": metrics(d["y"], pred),
                "by_kind": {str(kind): metrics(d["y"][d["kinds"] == kind], pred[d["kinds"] == kind], d["strengths"][d["kinds"] == kind]) for kind in sorted(set(d["kinds"]))},
                "noise_by_strength": [{"strength": float(level), **metrics(d["y"][mask], pred[mask], d["strengths"][mask])}
                                      for level in sorted(set(d["strengths"][noise]))
                                      for mask in [noise & (d["strengths"] == level)]],
                "noise_by_range": {name: metrics(d["y"][mask], pred[mask], d["strengths"][mask]) for name, mask in
                                   (("5–100", noise & (d["strengths"] <= 100)), ("1000–4000", noise & (d["strengths"] >= 1000)))}}
            for i, sid in enumerate(d["sample_ids"]):
                rows.append({"model": view, "split": split, "sample_id": str(sid), "source_id": str(d["source_ids"][i]),
                             "kind": str(d["kinds"][i]), "requested_snr": float(d["strengths"][i]),
                             "true_noise": float(d["y"][i]), "predicted_noise": float(pred[i]), "raw_prediction": float(raw[i])})
        print(view, best, "validation:", results[view]["validation"]["overall"], flush=True)
    results["baselines"] = {split: {name: metrics(d["y"], np.full(len(d["y"]), value)) for name, value in
                                   (("train_mean", data["train"]["y"].mean()), ("no_error", 0.))} for split, d in data.items()}
    correlations = []
    for j, name in enumerate(COMBINED8_FEATURES):
        row = {"feature": name, "label": LABELS[j]}
        for split, d in data.items():
            mask = d["kinds"] == "noise"
            x, y, sources = d["strengths"][mask], d["X"][mask, j], d["source_ids"][mask]
            within = [float(spearmanr(x[sources == s], y[sources == s]).statistic)
                      for s in set(sources) if np.ptp(y[sources == s]) > 0]
            row[split] = {"pearson": float(pearsonr(x, y).statistic), "spearman": float(spearmanr(x, y).statistic),
                          "median_within_source_spearman": float(np.median(within)) if within else None}
        correlations.append(row)
    with (run / "predictions.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for name, value in (("config.json", config), ("results.json", results), ("cv_results.json", cv_rows), ("correlations.json", correlations)):
        write_json(run / name, value)
    for path, expected in inputs.items():
        assert digest(Path(path)) == expected, path
    from ml.noise_feature_report import write_report
    write_report(run, data, predictions, results, correlations, config, image_run)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-run", type=Path, required=True)
    parser.add_argument("--image-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("ml/runs"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    run = compare(args.from_run, args.output, args.image_run, args.jobs)
    print(f"Report: {run / 'report.html'}", flush=True)


if __name__ == "__main__":
    main()
