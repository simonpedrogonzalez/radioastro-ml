"""Per-case planes for the image features (docs/image_features_plan.md §1).

R = residual, I = restored (clean), C = convolve_model(model, header) (the same C as eC). Scale
s = max(max |I| over Ω, nominal RMS); C and R are returned divided by s. ρ is the reference_masks.npz radius
(elliptical beam units centred on the known source, ρ = 1 at one FWHM). The truth image is used only to
recompute eC as a check, never for features.
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
from astropy.io import fits

from scripts.clean_recovery_noise_sweep import convolve_model


def read_plane(path):
    with fits.open(path, memmap=False) as hdul:
        return np.asarray(np.squeeze(hdul[0].data), np.float64), hdul[0].header.copy()


@functools.lru_cache(maxsize=8)
def read_masks(path):
    with np.load(path, allow_pickle=False) as m:
        return {k: np.asarray(m[k]) for k in ("omega", "rho", "truth")}


def load_case(run, products, prepared, psf=False):
    """run: dataset folder; products: case.json "products"; prepared: the plan's prepared_sources entry."""
    run = Path(run)
    masks = read_masks(str(run / prepared["reference_masks"]))
    omega, rho = masks["omega"].astype(bool), masks["rho"].astype(np.float64)
    model, _ = read_plane(run / products["model"])
    I, header = read_plane(run / products["clean"])
    R, _ = read_plane(run / products["residual"])
    if any(x.shape != omega.shape for x in (model, I, R)) or not omega.any():
        raise ValueError("Plane shapes differ from the reference masks")
    C = convolve_model(model, header)   # clean header, exactly as scripts/dataset_v2_matched_extract.py
    truth = masks["truth"].astype(np.float64)
    eC = float(np.sqrt(np.square((C - truth)[omega]).sum() / np.square(truth[omega]).sum()))
    s = max(float(np.max(np.abs(I[omega]))), float(prepared["nominal_rms"]))
    if not (np.isfinite(s) and s > 0):
        raise ValueError(f"Invalid feature scale s={s}")
    out = dict(C=C / s, R=R / s, I=I, omega=omega, rho=rho, header=header, s=s, model=model, eC=eC)
    if psf:
        out["psf"], out["psf_header"] = read_plane(run / products["psf"])
    return out
