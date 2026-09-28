"""CNN artifacts and a self-contained HTML report using the existing reporter."""

import csv
import json

from ml.evaluate import _summary
from ml.report import _metric_lines, _plot_severity
from scripts.reporting import QuartoReporter


def write_run(output, config, history, results, rows, audits, baselines):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, value in (("history", history), ("exclusions", audits),
                        ("results", results), ("config", config), ("baselines", baselines)):
        (output / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
    with (output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, metric in zip(axes, ("loss", "balanced_accuracy"), strict=True):
        for split in ("train", "val"):
            axis.plot([h["epoch"] for h in history], [h[f"{split}_{metric}"] for h in history], label=split)
        axis.axvline(config["training"]["best_epoch"], color="grey", linestyle=":", label="selected epoch")
        axis.set(xlabel="Epoch", ylabel=metric.replace("_", " "))
        axis.legend()
        axis.grid(alpha=.2)
    figure.savefig(output / "learning.png", dpi=160)
    plt.close(figure)
    _plot_severity(results["val"], output / "severity.png")
    lines = [
        "---", 'title: "Square-only ResNet-18 experiment"', "format:", "  html:",
        "    embed-resources: true", "---", "",
        f"Input: **{config['input_mode']}**; seed **{config['seed']}**; device **{config['device']}**.", "",
        f"Training: **{'head only, matched two-stage schedule' if config['training'].get('head_only') else 'head warmup then last-block fine-tuning'}**.", "",
        "Labels: none=0, amp=1, phase=2. Circular sources are **not evaluated**; results apply only to square-support sources.", "",
        f"Test evaluated: **{'yes' if 'test' in results else 'no'}**. No source was moved between partitions.", "",
        "## Support audit", "",
        "Class counts below are ordered none/amp/phase. Exclusions are based only on support, not labels.", "",
        "| Split | Retained samples / sources | Retained classes | Excluded samples / sources | Excluded classes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, audit in audits.items():
        lines.append(f"| {name} | {audit['retained_samples']} / {audit['retained_sources']} | "
                     f"{audit['retained_classes']} | {audit['excluded_samples']} / {audit['excluded_sources']} | {audit['excluded_classes']} |")
    lines += ["", "## Results", "",
              "At positive SNR, detection counts predictions of amp or phase among all corrupted samples. Type correctness is conditional on detection. Clean false alarms are shown separately.", ""]
    for name, metrics in results.items():
        lines += _metric_lines(name, metrics)
    lines += ["## Diagnostic plots", "", "![Learning curves](learning.png)", "",
              "Training curves use augmented minibatches during optimization; validation uses deterministic inputs. Final training scores above evaluate the selected checkpoint without augmentation.", "",
              "![Validation detection and conditional type correctness by SNR](severity.png)", "",
              "## Matched square-only logreg baselines", "",
              "Same retained train/validation samples; unchanged StandardScaler + balanced logistic regression (C=1, lbfgs, max_iter=1000). Historical whole-population scores are not comparable.", ""]
    for name, baseline in baselines.items():
        lines += _metric_lines(name, baseline["validation"])
        lines += ["Features: " + ", ".join(f"`{f}`" for f in baseline["features"]), ""]
    lines += ["## Per-source results", "", "| Split | Source | Samples | Accuracy | Balanced accuracy | Macro F1 |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for split, source in sorted({(r["split"], r["source_id"]) for r in rows}):
        selected = [r for r in rows if (r["split"], r["source_id"]) == (split, source)]
        m = _summary([r["true_class"] for r in selected], [r["predicted_class"] for r in selected])
        lines.append(f"| {split} | {source} | {len(selected)} | {m['accuracy']:.4f} | {m['balanced_accuracy']:.4f} | {m['macro_f1']:.4f} |")
    lines += ["", "## Excluded samples", "", "| Split | Source | Sample | Reason |", "| --- | --- | --- | --- |"]
    for split, audit in audits.items():
        for row in audit["excluded"]:
            lines.append(f"| {split} | {row['source_id']} | {row['sample_id']} | {row['reason']} |")
    lines += ["", "## Reproducibility", "", "```json", json.dumps(config, indent=2), "```", "",
              "Clipping fractions are over all three full-field channels before resizing; fixed clip limit was not retuned from validation/test.", "",
              "```json", json.dumps({s: {k: v for k, v in a.items() if k.startswith('clipping_')} for s, a in audits.items()}, indent=2), "```", "",
              "Trainable parameter counts and epoch timings are recorded in history.json. All BatchNorm parameters and buffers remain frozen. Metadata is used for labels/reporting only.", ""]
    qmd = output / "report.qmd"
    qmd.write_text("\n".join(lines))
    QuartoReporter(qmd, every=1).finish()
    if not qmd.with_suffix(".html").is_file():
        raise RuntimeError(f"HTML rendering failed; artifacts remain at {output}")
