"""Compare saved QA, parity-logistic, and optional parity-CNN validation predictions."""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from ml.evaluate import evaluate
from scripts.reporting import QuartoReporter, export_detached_report


def validation_rows(run):
    with (run / "predictions.csv").open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "val"]
    if not rows:
        raise ValueError(f"No validation predictions in {run}")
    return {row["sample_id"]: row for row in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--cnn", type=Path)
    parser.add_argument("--output", type=Path, default=Path("ml/runs"))
    parser.add_argument("--export", type=Path, required=True)
    args = parser.parse_args()

    runs = {"Four QA metrics": validation_rows(args.baseline),
            "Two parity metrics": validation_rows(args.parity)}
    if args.cnn:
        config = json.loads((args.cnn / "config.json").read_text())
        if config["input_mode"] != "parity":
            raise ValueError("The CNN must use parity input")
        runs[f'Parity CNN (seed {config["seed"]})'] = validation_rows(args.cnn)
    baseline = runs["Four QA metrics"]
    identity = lambda rows: {key: (row["true_class"], row["source_id"],
                                    row["corruption_snr_target"]) for key, row in rows.items()}
    if any(identity(baseline) != identity(rows) for rows in runs.values()):
        raise ValueError("The runs do not have identical validation samples, labels, and levels")

    ids = sorted(baseline)
    batch = {"sample_id": ids,
             "label": [int(baseline[key]["true_class"]) for key in ids],
             "label_metadata": [{"corruption_snr_target": float(baseline[key]["corruption_snr_target"])}
                                for key in ids]}
    metrics = {name: evaluate([batch], {key: int(rows[key]["predicted_class"]) for key in ids})
               for name, rows in runs.items()}
    levels = ("10", "30", "50", "100")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = args.output.resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                                      + "_model_comparison")
    output.mkdir(parents=True, exist_ok=False)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for name, result in metrics.items():
        grouped = result["by_corruption_snr_target"]
        axes[0].plot([int(level) for level in levels],
                     [grouped[level]["detection_recall"] for level in levels], marker="o", label=name)
        axes[1].plot([int(level) for level in levels],
                     [grouped[level]["type_accuracy_among_detected"] for level in levels],
                     marker="o", label=name)
    for axis, title in zip(axes, ("Detected as corrupted", "Correct type among detections")):
        axis.set(title=title, xlabel="Target SNR", ylabel="Fraction", ylim=(-0.02, 1.02))
        axis.set_xticks([int(level) for level in levels])
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    figure.savefig(output / "by_snr.png", dpi=160)
    plt.close(figure)

    def cell(level, result, metric):
        values = result["by_corruption_snr_target"][level]
        total = values["sample_count"]
        detected = round(values["detection_recall"] * total)
        if metric == "detection_recall":
            return f'{values[metric]:.3f} ({detected}/{total})'
        correct = round(values["corruption_type_accuracy"] * total)
        score = values["type_accuracy_among_detected"]
        return f"{score:.3f} ({correct}/{detected})" if score is not None else "—"

    title = ("Four-metric vs parity logistic regression vs parity CNN" if args.cnn
             else "Four-metric vs parity logistic regression")
    lines = ["---", f'title: "{title}"',
             "format:", "  html:", "    page-layout: full", "    embed-resources: true", "---", "",
             "## Overall validation", "",
             "| Model | Samples | Accuracy | Macro precision | Macro recall | Macro F1 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, result in metrics.items():
        lines.append(f'| {name} | {result["sample_count"]} | {result["accuracy"]:.3f} | '
                     f'{result["macro_precision"]:.3f} | {result["macro_recall"]:.3f} | '
                     f'{result["macro_f1"]:.3f} |')
    short_names = ("QA", "Parity logreg", "Parity CNN")[:len(runs)]
    headings = [value for name in short_names for value in
                (f"{name} detected", f"{name} correct type / detected")]
    lines += ["", "## By target SNR", "",
              "| Target SNR | " + " | ".join(headings) + " |",
              "| ---: " + "| ---: " * len(headings) + "|"]
    for level in levels:
        values = [value for result in metrics.values() for value in
                  (cell(level, result, "detection_recall"),
                   cell(level, result, "type_accuracy_among_detected"))]
        lines.append(f'| {level} | ' + ' | '.join(values) + ' |')
    lines += ["", "![Detection and correct type by target SNR](by_snr.png)", ""]
    qmd = output / "report.qmd"
    qmd.write_text("\n".join(lines), encoding="utf-8")
    QuartoReporter(qmd).finish()
    if not qmd.with_suffix(".html").is_file():
        raise RuntimeError("Report rendering failed")
    exported = export_detached_report(qmd, args.export)
    print(f"Report: {qmd.with_suffix('.html')}\nExport: {exported.html_path}")


if __name__ == "__main__":
    main()
