"""ML-environment worker: matched Dataset v2 targets and the 20-feature view.

Features (docs/dataset_v2_updated_plan.md §4): combined8 (unchanged), dirty6
(columns 1-6 of combined8 evaluated on the dirty image), model6 (raw CLEAN
model flux sums) and clean2 (CLEAN iterations, nonzero model pixels).
Targets: eC (CLEAN recovery error) and eD (dirty error).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from astropy.io import fits

from ml.logreg import COMBINED8_FEATURES, _parity_residual, combined8_from_products
from scripts.clean_recovery_noise_sweep import convolve_model, geometry
from scripts.imaging.metrics import summarize_residual_pixels
from scripts.imaging.models import Beam
from scripts.preprocessing.schema import atomic_write_json

DIRTY6 = ("dirty.dynamic_range_scaled_mad", "dirty.scaled_mad_jy_per_beam", "dirty.peak_over_scaled_mad",
          "dirty.p99_5_over_scaled_mad", "dirty.log1p_even_energy_over_mad2", "dirty.log1p_odd_energy_over_mad2")
MODEL6 = ("model.flux_positive_jy", "model.flux_negative_magnitude_jy", "model.flux_absolute_jy",
          "model.flux_inside_beam_jy", "model.flux_outside_beam_jy", "model.flux_over_restored_peak_per_beam")
CLEAN2 = ("clean.iterations", "model.nonzero_pixels")
BASE20 = tuple(COMBINED8_FEATURES) + DIRTY6 + MODEL6   # schema of cases committed before CLEAN2
FEATURES = BASE20 + CLEAN2
OPTIONAL = ("model.flux_over_restored_peak_per_beam",)   # the only feature that may be missing


def image_qa4(image, valid, peak_image, peak_valid, header):
    """The four combined8 QA columns, computed exactly as scripts/imaging/metrics.py does."""
    beam = Beam(header["BMAJ"] * 3600, header["BMIN"] * 3600, header["BPA"])
    cell = abs(header["CDELT1"]) * 3600 * abs(header["CDELT2"]) * 3600
    selected = valid & np.isfinite(image)
    m = summarize_residual_pixels(image[selected], pixel_area_arcsec2=cell, beam=beam)
    peak = float(np.max(peak_image[peak_valid & np.isfinite(peak_image)]))
    scale = m.scaled_mad_jy_per_beam
    dynamic = peak / scale if math.isfinite(scale) and scale > 0 else float("nan")
    return [dynamic, scale, m.peak_over_scaled_mad, m.p99_5_over_scaled_mad]


def parity2(image):
    """Existing residual parity function, unchanged, applied to any native 256x256 plane."""
    import torch
    return list(_parity_residual(torch.from_numpy(np.asarray(image, dtype=np.float32))))


def model6(model, omega, header, restored_peak):
    """Raw-model flux sums (Jy) over the fixed support and the FWHM aperture at the phase centre."""
    M = np.where(omega, np.asarray(model, np.float64), 0.)
    # Image reference direction is the observation phase centre (default tclean phasecenter).
    rho, _ = geometry(header, M.shape, 1., math.radians(header["CRVAL1"]), math.radians(header["CRVAL2"]))
    inside = omega & (rho <= .5)   # rho = 1 at one FWHM, so semi-axes BMAJ/2, BMIN/2
    outside = omega & ~inside
    F = float(M.sum())
    ratio = F / restored_peak if math.isfinite(restored_peak) and restored_peak > 0 else float("nan")
    return [float(np.maximum(M, 0).sum()), float(np.maximum(-M, 0).sum()), float(np.abs(M).sum()),
            float(M[inside].sum()), float(M[outside].sum()), ratio], dict(F=F, inside_pixels=int(inside.sum()))


def clean2(model, convergence):
    """Total CLEAN minor-cycle iterations and the count of nonzero raw-model pixels (either sign, whole image)."""
    iterations = convergence.get("iterdone")
    if isinstance(iterations, bool) or not isinstance(iterations, (int, float)) or not math.isfinite(iterations) or iterations < 0:
        raise ValueError(f"Missing or invalid CLEAN iteration count: {iterations!r}")
    model = np.asarray(model)
    return [float(iterations), float(np.count_nonzero(np.isfinite(model) & (model != 0)))]


def tclean_model_flux(convergence):
    """CASA modelFlux summary, or None when CASA reports none (e.g. an empty model)."""
    value = convergence.get("final_model_flux_jy")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def json_features(X):
    """Missing feature values (nonfinite) are stored as JSON null."""
    return [float(x) if x is not None and math.isfinite(float(x)) else None for x in X]


def extract(root, record, masks_path, nominal_rms, reference_record):
    def plane(path):
        with fits.open(root / path, memmap=False) as hdul:
            return np.asarray(np.squeeze(hdul[0].data), dtype=np.float64), hdul[0].header.copy()

    products = record["products"]
    model, _ = plane(products["model"])
    clean, header = plane(products["clean"])
    residual, _ = plane(products["residual"])
    dirty, _ = plane(products["dirty"])
    dirty0, _ = plane(reference_record["products"]["dirty"])
    with np.load(masks_path, allow_pickle=False) as masks:
        truth = np.asarray(masks["truth"], dtype=np.float64)
        omega = np.asarray(masks["omega"], dtype=bool)
        search = np.asarray(masks["clean_search"], dtype=bool)
    with np.load(root / products["support"], allow_pickle=False) as support:
        if not support["valid"][omega].all() or not np.array_equal(support["clean_search"], search):
            raise ValueError("Fixed support/search mask changed")
        masks_by = {k: np.asarray(support[f"{k}_mask"], bool) for k in ("dirty", "clean", "residual")}
    planes = (model, clean, residual, dirty, dirty0)
    if any(x.shape != truth.shape or not np.isfinite(x[omega]).all() for x in planes) or not omega.any():
        raise ValueError("Invalid matched v2 image arrays")
    C = convolve_model(model, header)
    eC = float(np.sqrt(np.square((C - truth)[omega]).sum() / np.square(truth[omega]).sum()))
    d0 = float(np.square(dirty0[omega]).sum())
    if d0 <= 0:
        raise ValueError("Empty noiseless reference dirty image")
    eD = float(np.sqrt(np.square((dirty - dirty0)[omega]).sum() / d0))
    qa = json.loads((root / products["clean"]).parent.joinpath("qa.json").read_text())
    c8 = combined8_from_products(qa, residual.astype(np.float32), nominal_rms)
    # Our recomputation of the residual QA must reproduce CASA-side qa.json, so the
    # dirty columns below are the same functions on the same region.
    check = image_qa4(residual, masks_by["residual"], clean, masks_by["clean"], header)
    if not np.allclose(check, c8[:4], rtol=1e-5, atol=0):
        raise ValueError(f"Residual QA recomputation differs from qa.json: {check} vs {c8[:4].tolist()}")
    dirty_features = image_qa4(dirty, masks_by["dirty"], dirty, masks_by["dirty"], header) + parity2(dirty)
    restored_peak = float(np.max(clean[masks_by["clean"] & np.isfinite(clean)]))
    m6, info = model6(model, omega, header, restored_peak)
    # The direct sums are the features; the CASA summary is only a cross-check when present.
    summary = tclean_model_flux(record["convergence"])
    whole = float(model[np.isfinite(model)].sum())
    if summary is not None and abs(whole - summary) > 1e-3 * max(abs(summary), 1e-6):
        raise ValueError(f"Model flux sum {whole} disagrees with tclean modelFlux {summary}")
    X = [*c8.tolist(), *dirty_features, *m6, *clean2(model, record["convergence"])]
    # Only the flux/peak ratio may be missing (restored peak <= 0 or nonfinite).
    if len(X) != len(FEATURES) or not all(math.isfinite(v) for v, name in zip(X, FEATURES) if name not in OPTIONAL):
        raise ValueError("Nonfinite matched v2 features")
    X = json_features(X)
    if not (math.isfinite(eC) and math.isfinite(eD)):
        raise ValueError("Invalid targets")
    return dict(eC=eC, eD=eD, X=X, feature_names=list(FEATURES),
                empty_raw_model=not bool(np.any(model[omega])),
                model_flux=dict(sum_support=info["F"], sum_all_pixels=whole, tclean_model_flux=summary,
                                aperture_pixels=info["inside_pixels"]),
                iterations=record["convergence"], warnings=record["warnings"])


def dirty_change(root, products_a, products_b, products_0, omega):
    """p = ||D_a - D_b|| / ||D0|| over the fixed support."""
    def load(path):
        with fits.open(root / path, memmap=False) as hdul:
            return np.asarray(np.squeeze(hdul[0].data), dtype=np.float64)
    a, b, d0 = load(products_a["dirty"]), load(products_b["dirty"]), load(products_0["dirty"])
    return float(np.sqrt(np.square((a - b)[omega]).sum() / np.square(d0[omega]).sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--reference-record", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--nominal-rms", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    reference = json.loads(args.reference_record.read_text())
    atomic_write_json(args.output, extract(args.root, record, args.masks, args.nominal_rms, reference))


if __name__ == "__main__":
    main()
