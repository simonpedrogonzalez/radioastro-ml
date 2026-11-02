"""Plots and HTML report for the combined6/combined8 scalar-noise comparison."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from ml.regression import write_json
from scripts.reporting import QuartoReporter


def write_report(run, data, predictions, results, correlations, config, image_run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ml.noise_feature_comparison import COLORS

    model_names = list(predictions)
    maximum = max(4000., max(float(p.max()) for v in predictions.values() for p in v.values())) * 1.04
    markers = {"clean": "s", "amp": "o", "phase": "^", "noise": "x"}
    for zoom in (False, True):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout="constrained", sharey=True)
        for row, name in enumerate(model_names):
            for col, (split, d) in enumerate(data.items()):
                axis = axes[row, col]
                for kind in markers:
                    mask = (d["kinds"] == kind) & ((d["y"] <= 100) if zoom else True)
                    axis.scatter(d["y"][mask], predictions[name][split][mask], s=14, alpha=.4,
                                 marker=markers[kind], color=COLORS[kind], label=kind)
                limit = 105 if zoom else maximum
                axis.plot([0, limit], [0, limit], "k--", lw=1, label="Perfect prediction")
                noise = d["kinds"] == "noise"
                levels = sorted(set(d["strengths"][noise & (d["strengths"] <= 100)] if zoom else d["strengths"][noise]))
                means = [predictions[name][split][noise & (d["strengths"] == s)].mean() for s in levels]
                axis.plot(levels, means, "o-", color="black", markersize=4, label="Noise-only mean")
                axis.set(xlim=(0, limit), ylim=(0, maximum), xlabel="True extra-noise SNR_corr", ylabel="Predicted noise SNR_corr",
                         title=f"{name} · {'training' if split == 'train' else 'held-out validation'}")
                axis.grid(alpha=.2)
                if row == col == 0:
                    axis.legend(fontsize=8)
        fig.savefig(run / ("true_predicted_low_range.png" if zoom else "true_predicted.png"), dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(19, 9), layout="constrained")
    for j, row in enumerate(correlations):
        axis = axes.flat[j]
        for split, d in data.items():
            mask = d["kinds"] == "noise"
            strengths, values = d["strengths"][mask], d["X"][mask, j]
            color = "#0072B2" if split == "train" else "#D55E00"
            axis.scatter(strengths, values, s=12 if split == "train" else 22, alpha=.2 if split == "train" else .55,
                         marker="o" if split == "train" else "x", color=color, label=split)
            levels = sorted(set(strengths))
            axis.plot(levels, [np.median(values[strengths == level]) for level in levels], color=color, label=f"{split} median")
        axis.set_xscale("log"); axis.set_yscale("log")
        r = row["validation"]
        axis.set_title(f'{row["label"]}\nHeld-out Pearson r={r["pearson"]:.3f}\nSpearman ρ={r["spearman"]:.3f}', fontsize=10)
        axis.set(xlabel="Injected noise SNR_corr (log scale)", ylabel="Feature value (log scale)")
        axis.set_xticks([5, 20, 100, 1000, 4000], labels=["5", "20", "100", "1000", "4000"])
        axis.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.savefig(run / "features_vs_noise.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    # The unchanged source images use the previous report's fixed display scales.
    artifacts = [image_run / "report.qmd"]
    artifacts += list(image_run.glob("image_level_*.png")) + list(image_run.glob("image_sequence*.png"))
    artifacts += [image_run / name for name in ("image_display.json", "image_preservation.json", "image_reproduction_checks.json")]
    artifacts += list((image_run / "models").glob("*.fits"))
    image_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}
    for path in artifacts:
        if path.name == "report.qmd":
            continue
        destination = run / path.relative_to(image_run)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    image_section = (image_run / "report.qmd").read_text().split("## One source from zero to 4000", 1)[1].split("## Verification", 1)[0]
    config["image_artifact_sha256"] = image_hashes
    config["code_sha256"][str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(run / "config.json", config)

    lines = ["---", 'title: "Thermal-noise regression: combined6 versus combined8"', "format:", "  html:",
             "    embed-resources: true", "    toc: true", "    page-layout: full", "---", "",
             "## Experiment", "",
             "One scalar XGBoost regressor per feature set predicts **extra-noise SNR_corr**. Both models train on **all 2,893 samples from 100 sources** and are evaluated on **all 608 samples from 21 unseen validation sources**. Clean, amplitude and phase samples have noise target zero. The reserved test partition is unused.", "",
             "**combined6** contains four image QA metrics and two parity features. **combined8** adds sigma_NA, the precursor's flag-adjusted natural-image RMS in Jy/beam, and sigma_sNA_units = residual scaled MAD / sigma_NA. The simple-metric view simple6 contains the four QA metrics and these two additions.", "",
             "sigma_NA was calculated from the source Measurement Set's unflagged parallel-hand samples and the theoretical per-visibility noise. It is constant across variants of each source. sigma_sNA_units compares the observed residual MAD with that reference in matching Jy/beam units.", "",
             "The existing three source CV folds, seed 42 and four parameter candidates (depth 2/4, 100/300 rounds, learning rate 0.05) are retained. Each feature set selects its own configuration using only training-source CV. The scalar noise loss and CV score use all rows. Predictions are clipped at zero. Plot scales do not transform model inputs.", "",
             "The previous noise-only report used 1,200/252 rows. This comparison fits combined6 again on the expanded all-type population so both feature sets use identical rows, targets and folds.", "",
             "| Model | Features | Selected depth | Boosting rounds |", "| --- | ---: | ---: | ---: |"]
    for name in model_names:
        p = config["best_params"][name]
        lines.append(f'| {name} | {len(config["features"][name])} | {p["max_depth"]} | {p["n_estimators"]} |')
    lines += ["", "[Configuration](config.json) · [CV results](cv_results.json) · [Predictions](predictions.csv) · [Numerical results](results.json) · [sigma_NA inputs](sigma_na.csv)", "",
              "## True versus predicted", "",
              "Every sample appears in its model/split panel. Colors and shapes identify the actual sample type. Dashed line: perfect prediction. Solid line: mean prediction among noise samples at each level. Training panels show in-sample predictions.", "",
              "![](true_predicted.png)", "",
              "| Population | Model | MAE | RMSE |", "| --- | --- | ---: | ---: |"]
    for split in data:
        for name in model_names:
            m = results[name][split]["overall"]
            lines.append(f'| {split} | {name} | {m["mae"]:.3f} | {m["rmse"]:.3f} |')
        for name, m in results["baselines"][split].items():
            lines.append(f'| {split} | {name} | {m["mae"]:.3f} | {m["rmse"]:.3f} |')
    base = results["combined6"]["validation"]["overall"]["mae"]
    new = results["combined8"]["validation"]["overall"]["mae"]
    change = 100 * (new / base - 1)
    lines += ["", f"Adding sigma_NA and sigma_sNA_units changes pooled validation MAE from **{base:.3f} to {new:.3f}** ({abs(change):.2f}% {'increase' if change > 0 else 'decrease'}). This is a single fixed source split and parameter grid.", "",
              "### Noise-only ranges", "", "This table selects only positive-noise samples for diagnosis; both models were fitted on all sample types.", "",
              "| Model | Noise range | Train MAE | Validation MAE | Train relative MAE | Validation relative MAE |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for name in model_names:
        for key in ("5–100", "1000–4000"):
            t, v = (results[name][split]["noise_by_range"][key] for split in data)
            lines.append(f'| {name} | {key} | {t["mae"]:.3f} | {v["mae"]:.3f} | {t["relative_mae_percent"]:.2f}% | {v["relative_mae_percent"]:.2f}% |')
    lines += ["", "Relative MAE divides each absolute noise error by its requested injected SNR. Zero-target samples are excluded from these percentages.", "",
              "### Errors by sample type", "", "| Model | Type | Validation samples | Noise MAE | Mean predicted noise |", "| --- | --- | ---: | ---: | ---: |"]
    for name in model_names:
        for kind, m in results[name]["validation"]["by_kind"].items():
            lines.append(f'| {name} | {kind} | {m["n"]} | {m["mae"]:.3f} | {m["mean_prediction"]:.3f} |')
    lines += ["", "<details><summary>Inspect true noise levels 0–100</summary>", "", "![](true_predicted_low_range.png)", "", "The y-axis retains the full prediction range.", "", "</details>", "",
              "## Each feature versus injected noise", "",
              "These panels select positive-noise samples, matching the earlier diagnostic. They show each original feature against injected SNR on logarithmic display axes. Colors distinguish training and held-out sources; lines join per-level medians. Correlations use raw values (Pearson) or ranks (Spearman).", "",
              "![](features_vs_noise.png)", "",
              "| Feature | Train Pearson | Held-out Pearson | Train Spearman | Held-out Spearman | Median within-source Spearman, held-out |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in correlations:
        t, v = row["train"], row["validation"]
        within = "—" if v["median_within_source_spearman"] is None else f'{v["median_within_source_spearman"]:.3f}'
        lines.append(f'| {row["label"]} | {t["pearson"]:.3f} | {v["pearson"]:.3f} | {t["spearman"]:.3f} | {v["spearman"]:.3f} | {within} |')
    lines += ["", "sigma_NA remains constant within a source; its within-source correlation is undefined. sigma_sNA_units can vary as residual MAD changes. Correlations are descriptive and use only positive-noise samples.", "",
              "## One source from zero to 4000", "", "The unchanged image sequence is reused from the previous report.", image_section, "",
              "## Verification", "",
              "The comparison preserves the original six feature columns, all targets and source folds. The new sigma_NA feature is read from the precursor image-noise calculation, and sigma_sNA_units is computed from each row’s residual MAD. Saved model reloads reproduce predictions exactly. Independent numerical checks are in [validation.json](validation.json).", "",
              "The image sequence retains the original CLEAN/residual products and the previously reproduced CASA component models. Their fixed display scales and file contents are unchanged. [Image reproduction checks](image_reproduction_checks.json) · [Image preservation](image_preservation.json)", ""]
    (run / "report.qmd").write_text("\n".join(lines))
    QuartoReporter(run / "report.qmd", every=1).finish()
    assert (run / "report.html").is_file()
    for filename, expected in image_hashes.items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected
