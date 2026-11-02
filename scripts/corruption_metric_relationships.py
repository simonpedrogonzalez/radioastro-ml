"""Audit existing V1 training metrics and render pairwise corruption plots.

Run with the ML environment; reads scientific products without modifying them.
This is the retained-data part of experiments 1/2, not a new simulation.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CACHE = Path("ml/runs/20261028T065537442783Z_combined8_strategies/train_features.npz")
DATASET = Path("collect/experiments/dataset_v1_20260921_all_psf/samples")
METRICS = ("SNR_corr_actual", "sigma_na_reference", "sigma_na_total", "sigma_simple_total", "robust_rms")
LABELS = ("Measured SNR_corr_actual", "Reference σ_NA (Jy/beam)", "Total σ_NA (Jy/beam)",
          "Total σ_simple (Jy)", "Annulus robust RMS (Jy/beam)")


def collect(cache: Path, dataset: Path) -> list[dict]:
    rows = []
    with np.load(cache, allow_pickle=False) as saved:
        ids, sources, kinds = saved["sample_ids"], saved["source_ids"], saved["kinds"]
        targets, strengths = saved["y"], saved["strengths"]
        features = saved["feature_names"].tolist()
        na_col = features.index("sigma_NA")
        for i, sid in enumerate(ids):
            sample = dataset / str(sid)
            simulation = json.loads((sample / "simulation.json").read_text())
            qa = json.loads((sample / "imaging/qa.json").read_text())
            reference = float(simulation["noise"]["simplenoise_jy"])
            control = simulation.get("noise_control", {})
            total = float(control.get("final_total_sigma_jy", reference))
            na = float(saved["X"][i, na_col])
            mad = float(qa["metrics"]["residual"]["scaled_mad_jy_per_beam"])
            assert np.isfinite([reference, total, na, mad]).all()
            np.testing.assert_allclose(mad, saved["X"][i, features.index(
                "metrics.residual.scaled_mad_jy_per_beam")], rtol=1e-12)
            if kinds[i] in ("amp", "phase"):
                measured = json.loads((sample / "corruption.json").read_text())["metrics"]
                actual = float(measured["Delta_V_L2"]) / reference
                provenance = "recomputed from retained measured Delta_V_L2 / reference_sigma"
                np.testing.assert_allclose(actual, measured["SNR_corr"], rtol=1e-12)
            elif kinds[i] == "noise":
                actual = float(control["SNR_corr_measured"])
                provenance = "retained direct visibility measurement from generation"
            else:
                actual = 0.0
                provenance = "no injected corruption; baseline thermal noise present"
            np.testing.assert_allclose(actual, targets[i].sum(), rtol=1e-12, atol=1e-12)
            rows.append(dict(sample_id=str(sid), source=str(sources[i]), kind=str(kinds[i]),
                requested_snr=float(strengths[i]), SNR_corr_actual=actual,
                SNR_corr_actual_provenance=provenance,
                SNR_corr_actual_minus_requested=actual-float(strengths[i]),
                sigma_na_reference=na, sigma_na_total=na * total / reference,
                sigma_simple_reference=reference, sigma_simple_total=total, robust_rms=mad))
    assert len({r["sample_id"] for r in rows}) == len(rows)
    return rows


def matrix(rows: list[dict], destination: Path, title: str):
    fig, axes = plt.subplots(len(METRICS), len(METRICS), figsize=(15, 14), layout="constrained")
    levels = np.asarray([r["requested_snr"] for r in rows])
    for y, key_y in enumerate(METRICS):
        for x, key_x in enumerate(METRICS):
            ax = axes[y, x]
            artist = ax.scatter([r[key_x] for r in rows], [r[key_y] for r in rows],
                       c=levels, cmap="viridis", s=7, alpha=.55)
            ax.set_xscale("symlog", linthresh=1e-8 if x else 1)
            ax.set_yscale("symlog", linthresh=1e-8 if y else 1)
            if y == len(METRICS)-1: ax.set_xlabel(LABELS[x], fontsize=8)
            if x == 0: ax.set_ylabel(LABELS[y], fontsize=8)
            ax.tick_params(labelsize=6)
            ax.grid(alpha=.2)
    fig.suptitle(title + " — color: requested injection strength")
    fig.colorbar(artist, ax=axes, shrink=.5, label="Requested injection SNR_corr")
    fig.savefig(destination, dpi=130)
    plt.close(fig)


def run(output: Path, cache: Path = DEFAULT_CACHE):
    output.mkdir(parents=True, exist_ok=True)
    rows = collect(cache, DATASET)
    with (output / "training_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    sections = []
    counts = {}
    for kind in ("noise", "amp", "phase"):
        selected = [r for r in rows if r["kind"] in (kind, "clean")]
        counts[kind] = {"positive": sum(r["kind"] == kind for r in selected),
                        "baseline": sum(r["kind"] == "clean" for r in selected),
                        "sources": len({r["source"] for r in selected})}
        matrix(selected, output / f"{kind}.png", f"Training: {kind} + regular-noise baseline")
        sections.append(f'<h2>{kind}</h2><p>{html.escape(str(counts[kind]))}</p><img src="{kind}.png">')
    (output / "audit.json").write_text(json.dumps({"cache": str(cache.resolve()), "rows": len(rows),
        "counts": counts, "status": "Existing-data comparisons only; new simulations pending noise convention"}, indent=2))
    (output / "report.html").write_text('''<!doctype html><meta charset="utf-8">
<title>Experiment 1 — retained training metric comparisons</title>
<style>body{font:17px sans-serif;margin:35px;max-width:1800px}img{width:100%}</style>
<h1>Experiment 1 — retained training metric comparisons</h1>
<p>Partial report: existing training data only. New noise sweeps and Experiment 2 are pending.</p>
<p>Each section includes all training samples of that corruption type plus the 100 shared
regular-noise baselines. No validation or test samples. Axes use symmetric logarithmic scales
to retain zero. Robust RMS means the annulus scaled MAD already recorded by the imaging pipeline.</p>
<p>Reference σ_NA is the original precursor prediction used by the previous ML reports.
Total σ_NA multiplies that reference by σ_total/σ_reference, accounting for extra noise.
Both are predictions for natural weighting; the images use the regular Briggs pipeline.
σ_simple is per real/imaginary visibility component. These quantities have different units.</p>
<p>Existing noise strengths are extra-noise SNR_corr targets, not thermal-noise multipliers.
Clean baselines have extra-noise SNR_corr=0 and retain regular thermal noise.</p>
<p>SNR_corr_actual uses measured perturbations, never the requested target. Gain values
are recomputed from retained measured ΔV L2 norms divided by the original sigma.
Noise values reuse the direct visibility measurement retained at generation; this CSV
does not reread deleted Measurement Sets. Provenance and target differences are recorded.
The separate <a href="../../noise-snr-actual-audit-20261102/report.html">paired MS audit</a>
reconstructs source 2241+098 and independently rereads DATA to check the injection.</p>
<p><a href="training_metrics.csv">All measured values (CSV)</a> · <a href="audit.json">Audit</a></p>
''' + "\n".join(sections))
    return output / "report.html"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.output))
