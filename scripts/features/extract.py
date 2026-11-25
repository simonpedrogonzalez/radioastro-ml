"""Extract the image-feature table (energy, radial, raw fft) for one verified matched v2 run.

  python -m scripts.features.extract --run runs/dataset-v2-matched-train --output runs/image-features-v2/train.npz --jobs 8

Rows follow <split>.npz (case_ids, row_ids). Checks per case: eC recomputed from C reproduces the stored eC
(rtol 1e-9) and energy.model.*_over_s · s reproduces the npz model.* columns. The table stores the sha256 of
<split>.npz and plan.json; ml.image_feature_experiment refuses a table whose hashes do not match.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from astropy.wcs import FITSFixedWarning

from scripts.features import energy, fourier, radial
from scripts.features.planes import load_case

warnings.filterwarnings("ignore", category=FITSFixedWarning)   # datfix/obsfix notes on every CASA header

NAMES = (*energy.NAMES, *radial.NAMES, *fourier.NAMES)
CURVES = fourier.CHANNELS


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def case_features(planes):
    values = {**energy.energy(planes), **radial.radial(planes)}
    fft_values, curves = fourier.fourier(planes)
    values.update(fft_values)
    return np.asarray([values[n] for n in NAMES], float), curves


def _one(job):
    run, case_id, prepared = job
    record = json.loads((Path(run) / "cases" / case_id / "case.json").read_text())
    planes = load_case(run, record["products"], prepared, psf=True)
    x, curves = case_features(planes)
    return x, np.stack([curves[c] for c in CURVES]), planes["s"], planes["eC"]


def extract(run, split=None, jobs=1):
    run = Path(run)
    plan = json.loads((run / "plan.json").read_text())
    split = split or plan["split"]
    npz = run / f"{split}.npz"
    with np.load(npz, allow_pickle=False) as z:
        data = {k: z[k] for k in ("case_ids", "row_ids", "source_ids", "eC", "X", "feature_names")}
    prepared = {p["id"]: p for p in plan["prepared_sources"]}
    work = [(str(run), str(c), prepared[str(s)]) for c, s in zip(data["case_ids"], data["source_ids"])]
    start = time.time()
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as pool:
            results = list(pool.map(_one, work, chunksize=8))
    else:
        results = [_one(w) for w in work]
    X = np.stack([r[0] for r in results])
    curves = np.stack([r[1] for r in results])
    s = np.asarray([r[2] for r in results])
    eC = np.asarray([r[3] for r in results])
    # Checks (docs/image_features_plan.md §5).
    np.testing.assert_allclose(eC, data["eC"], rtol=1e-9, atol=0, err_msg="eC recomputed from C differs from the stored eC")
    names = list(map(str, data["feature_names"]))
    model_cols = [names.index(f"model.{m}_jy") for m in energy.MODEL5]
    ours = X[:, [NAMES.index(f"energy.model.{m}_over_s") for m in energy.MODEL5]] * s[:, None]
    np.testing.assert_allclose(ours, data["X"][:, model_cols], rtol=1e-9, atol=1e-15,
                               err_msg="energy.model.* · s differs from the npz model.* columns")
    if np.isinf(X).any() or np.isinf(curves).any():
        raise ValueError("Infinite image feature values")
    nan = {n: int(c) for n, c in zip(NAMES, np.isnan(X).sum(0)) if c}
    return dict(case_ids=data["case_ids"], row_ids=data["row_ids"], names=np.asarray(NAMES), X=X,
                **{f"curve_{c}": curves[:, i] for i, c in enumerate(CURVES)},
                curve_fractions=fourier.CURVE_FRACTIONS, s=s, eC_recomputed=eC, split=np.asarray(split),
                split_npz_sha256=np.asarray(file_sha256(npz)), plan_sha256=np.asarray(file_sha256(run / "plan.json")),
                run=np.asarray(str(run))), dict(rows=len(X), seconds=round(time.time() - start, 1), nan_counts=nan)


def save(table, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **table)
    os.replace(tmp, output)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="verified matched v2 dataset folder (plan.json, <split>.npz, cases/)")
    ap.add_argument("--split", help="default: the plan's split")
    ap.add_argument("--output", required=True)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args(argv)
    if Path(a.output).exists():
        raise SystemExit(f"{a.output} exists; use a fresh output path")
    table, info = extract(a.run, a.split, a.jobs)
    save(table, a.output)
    print("IMAGE FEATURES", a.output, json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
