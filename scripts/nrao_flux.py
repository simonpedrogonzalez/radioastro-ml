"""NRAO pipeline flux of an extracted calibrator MS, from the fetcher's schema-2 sidecar.

The fetcher (nrao-archive-fetcher, workflows/radioastro_calibrators/flux_weblog.py) writes
flux_estimate.json beside each extracted MS. Only schema-2 sidecars with status "available"
are used: they record the setjy model that covers the extracted SPW (all spectral-index
coefficients) after rejecting ambiguous or invalid models. Schema-1 sidecars must first be
rewritten with the fetcher's ``flux_weblog recheck``. The model is evaluated with CASA's
setjy convention S(f) = S0 * (f/f0) ** sum_k(c_k * log10(f/f0)**k).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

SIDECAR = "flux_estimate.json"
METADATA = "extraction_metadata.json"


class FluxUnavailable(ValueError):
    """No usable flux from this route; the message says why."""


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(stokes_i_jy, reference_frequency_hz, coefficients, frequency_hz):
    x = math.log10(frequency_hz / reference_frequency_hz)
    return stokes_i_jy * 10 ** (x * sum(c * x ** k for k, c in enumerate(coefficients)))


def nrao_flux_decision(ms_path, frequency_hz):
    """Flux decision dict for the sidecar beside ms_path, or raise FluxUnavailable."""
    folder = Path(ms_path).parent
    sidecar, metadata_path = folder / SIDECAR, folder / METADATA
    if not sidecar.is_file():
        raise FluxUnavailable(f"no {SIDECAR} beside the MS")
    if not metadata_path.is_file():
        raise FluxUnavailable(f"no {METADATA} beside the MS")
    try:
        estimate = json.loads(sidecar.read_text())
        metadata = json.loads(metadata_path.read_text())
    except (OSError, ValueError) as exc:
        raise FluxUnavailable(f"unreadable NRAO flux metadata: {exc}") from exc
    if not isinstance(estimate, dict) or not isinstance(metadata, dict):
        raise FluxUnavailable("NRAO flux metadata must contain JSON objects")
    if estimate.get("schema") != 2:
        raise FluxUnavailable("schema-1 sidecar; rewrite it with the fetcher's flux_weblog recheck")
    if estimate.get("status") != "available":
        raise FluxUnavailable(f"sidecar status {estimate.get('status')!r}: {estimate.get('reason')}")
    if estimate.get("gain_calibrator_name") != metadata.get("gain_calibrator_name"):
        raise FluxUnavailable("sidecar field differs from the extracted field")
    if str(estimate.get("selected_spw")) != str(metadata.get("selected_spw")):
        raise FluxUnavailable("sidecar SPW differs from the extracted SPW")
    s0 = estimate.get("stokes_i_jy")
    f0 = estimate.get("reference_frequency_ghz")
    coefficients = estimate.get("spectral_index_coefficients")
    if not isinstance(coefficients, list) or not coefficients:
        raise FluxUnavailable("sidecar has no spectral-index coefficients")
    values = [s0, f0, frequency_hz, *coefficients]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values) or s0 <= 0 or f0 <= 0 or frequency_hz <= 0:
        raise FluxUnavailable("non-finite or non-positive flux model value")
    try:
        flux = evaluate(s0, f0 * 1e9, coefficients, frequency_hz)
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FluxUnavailable(f"flux model cannot be evaluated: {exc}") from exc
    if not (math.isfinite(flux) and flux > 0):
        raise FluxUnavailable(f"model evaluates to {flux!r} Jy at {frequency_hz} Hz")
    return dict(
        method="nrao_pipeline", flux_jy=flux, frequency_hz=float(frequency_hz),
        model=dict(stokes_i_jy=s0, reference_frequency_hz=f0 * 1e9, spectral_index_coefficients=coefficients,
                   convention="S0*(f/f0)**sum_k(c_k*log10(f/f0)**k)"),
        provenance=dict(sidecar=str(sidecar.resolve()), sidecar_sha256=_sha(sidecar),
                        extraction_metadata=str(metadata_path.resolve()), extraction_metadata_sha256=_sha(metadata_path),
                        gain_calibrator_name=estimate["gain_calibrator_name"], selected_spw=str(estimate["selected_spw"]),
                        source_file=estimate.get("source_file"), source_text=estimate.get("source_text"),
                        recovered_from=estimate.get("recovered_from"),
                        ambiguity_checked=estimate.get("ambiguity_checked"), fit_snr=estimate.get("fit_snr")),
    )
