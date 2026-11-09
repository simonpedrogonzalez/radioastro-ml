"""Presence classification on validation, ignoring present corruptions with small recovery damage.

python -m ml.presence_min_damage --run ml/runs/matched-v2-full-all22-dirty --val runs/dataset-v2-matched-val [--min-damage 0.05]

Read-only: uses the run's saved predictions_val_<model>.npz. For type k, validation rows where k is
present but |d_k| < min-damage are removed; all rows where k is absent stay. Macro F1 (present/absent
classes, threshold 0.5), AUROC and AUPRC are scikit-learn, as in ml/multitask.py. Prints a markdown table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from ml import multitask as mt

KINDS = ("amp", "phase", "noise")
LABELS = {"mlp": "MLP", "xgb_shared": "Shared XGB", "xgb_independent": "Independent XGB",
          "xgb_xgb": "XGB → XGB", "xgb_mlp": "XGB → MLP", "xgb_xgb_naive": "XGB → XGB naive",
          "xgb_mlp_naive": "XGB → MLP naive"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--min-damage", type=float, default=0.05)
    a = p.parse_args()
    val = mt.read_split(a.val, "val")
    b, d = val["b"].astype(bool), val["d"]
    models = json.loads((a.run / "results.json").read_text())["models"]
    keep = [~b[:, k] | (np.abs(d[:, k]) >= a.min_damage) for k in range(3)]
    print(f"Validation, present rows with |d_k| < {a.min_damage:g} removed per type "
          + ", ".join(f"{kind}: {int((b[:, k] & ~keep[k]).sum())} of {int(b[:, k].sum())} present removed"
                      for k, kind in enumerate(KINDS)) + "\n")
    print("| Model | " + " | ".join(f"{k.capitalize()} macro F1 / AUROC / AUPRC" for k in KINDS) + " |")
    print("| :-- |" + " --: |" * 3)
    for name in models:
        with np.load(a.run / f"predictions_val_{name}.npz", allow_pickle=False) as z:
            if not np.array_equal(z["row_ids"].astype(str), val["row_ids"].astype(str)):
                raise ValueError(f"{name}: prediction rows do not match the validation split")
            prob = z["decoded"][:, 4:7]
        cells = []
        for k in range(3):
            y, s = b[keep[k], k], prob[keep[k], k]
            f1 = precision_recall_fscore_support(y, s >= .5, average="macro", labels=[False, True], zero_division=0)[2]
            cells.append(f"{f1:.3f} / {roc_auc_score(y, s):.3f} / {average_precision_score(y, s):.3f}")
        print(f"| {LABELS.get(name, name)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
