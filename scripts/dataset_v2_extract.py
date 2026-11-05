"""ML-environment worker: extract compact v2 features and recovery error."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from ml.logreg import COMBINED8_FEATURES, combined8_from_products
from scripts.clean_recovery_noise_sweep import convolve_model
from scripts.preprocessing.schema import atomic_write_json


def extract(root, record, masks_path, nominal_rms):
    def plane(key):
        with fits.open(root / record["products"][key], memmap=False) as hdul:
            return np.asarray(np.squeeze(hdul[0].data), dtype=np.float64), hdul[0].header.copy()

    model, _ = plane("model")
    clean, header = plane("clean")
    residual, _ = plane("residual")
    dirty, _ = plane("dirty")
    with np.load(masks_path, allow_pickle=False) as masks:
        truth = np.asarray(masks["truth"], dtype=np.float64)
        omega = np.asarray(masks["omega"], dtype=bool)
        search = np.asarray(masks["clean_search"], dtype=bool)
    with np.load(root / record["products"]["support"], allow_pickle=False) as support:
        if not support["valid"][omega].all() or not np.array_equal(support["clean_search"], search):
            raise ValueError("Fixed support/search mask changed")
    if any(x.shape != truth.shape or not np.isfinite(x[omega]).all() for x in
           (model, clean, residual, dirty)) or not omega.any():
        raise ValueError("Invalid v2 image arrays")
    C = convolve_model(model, header)
    denominator = float(np.square(truth[omega]).sum())
    if denominator <= 0 or not np.isfinite(C[omega]).all():
        raise ValueError("Invalid fixed truth/convolved model")
    y = float(np.sqrt(np.square((C - truth)[omega]).sum() / denominator))
    if not np.isfinite(y) or y < 0:
        raise ValueError("Invalid recovery error")
    qa_path = root / record["products"]["clean"]
    qa = json.loads((qa_path.parent / "qa.json").read_text())
    features = combined8_from_products(qa, residual.astype(np.float32), nominal_rms)
    return dict(y=y, X=features.tolist(), feature_names=list(COMBINED8_FEATURES),
                empty_raw_model=not bool(np.any(model[omega])),
                iterations=record["convergence"], warnings=record["warnings"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--nominal-rms", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    atomic_write_json(args.output, extract(args.root, record, args.masks, args.nominal_rms))


if __name__ == "__main__":
    main()
