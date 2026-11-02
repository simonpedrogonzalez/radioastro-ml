"""Verify SNR_corr directly from paired MS DATA, independently of targets (CASA).

This diagnostic compares the existing extra-SNR injection with a noise-STD
multiplier. It does not choose a convention for the larger imaging experiment.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np


def actual_snr(delta: np.ndarray, sigma_reference: float) -> float:
    """L2 norm in per-component reference-noise units, including the zero case."""
    if not math.isfinite(sigma_reference) or sigma_reference <= 0:
        raise ValueError("Reference sigma must be finite and positive")
    values = np.asarray(delta, dtype=np.complex128)
    if not np.isfinite(values).all():
        raise ValueError("Delta contains nonfinite values")
    return float(np.sqrt(np.sum(values.real**2 + values.imag**2, dtype=np.float64)) / sigma_reference)


def measure_pair(reference: Path, observed: Path, sigma: float) -> dict:
    """Reread stored DATA and sum finite, unflagged parallel-hand cross samples."""
    from casatools import table
    from scripts.corruption.metrics import _chunk_metadata, _read_correlation_indices, _valid_mask

    correlations = _read_correlation_indices(reference)
    a, b = table(), table()
    power = 0.0
    n = 0
    a.open(str(reference)); b.open(str(observed))
    try:
        assert a.nrows() == b.nrows()
        for start in range(0, a.nrows(), 4096):
            count = min(4096, a.nrows() - start)
            va = np.asarray(a.getcol("DATA", startrow=start, nrow=count), dtype=np.complex128)
            vb = np.asarray(b.getcol("DATA", startrow=start, nrow=count), dtype=np.complex128)
            ma = _chunk_metadata(a, set(a.colnames()), start, count)
            mb = _chunk_metadata(b, set(b.colnames()), start, count)
            assert va.shape == vb.shape
            for i in range(3): np.testing.assert_array_equal(ma[i], mb[i])
            flags = a.getcol("FLAG", startrow=start, nrow=count) | b.getcol("FLAG", startrow=start, nrow=count)
            valid = _valid_mask(va, flags, *ma[:2], ma[3] | mb[3], ma[2], correlations)
            valid &= np.isfinite(vb.real) & np.isfinite(vb.imag)
            delta = (vb - va)[valid]
            power += float(np.sum(delta.real**2 + delta.imag**2, dtype=np.float64))
            n += int(valid.sum())
    finally:
        b.close(); a.close()
    if not n: raise ValueError("No eligible visibility samples")
    return {"SNR_corr_actual": math.sqrt(power) / sigma,
            "Delta_V_L2_jy": math.sqrt(power), "valid_complex_samples": n,
            "delta_component_rms_jy": math.sqrt(power / (2*n)),
            "reference_sigma_jy": sigma}


def render(output: Path):
    record = json.loads((output / "audit.json").read_text())
    rows = []
    for row in record["rows"]:
        total = row["total_thermal"]["SNR_corr_actual"]
        paired = row["difference_from_regular"]["SNR_corr_actual"]
        target = row["requested"]
        error = abs(paired / target - 1) if row["control"] == "extra_SNR_corr_target" else None
        rows.append(f'<tr><td>{row["control"]}</td><td>{target:g}</td>'
                    f'<td>{paired:.10g}</td><td>{total:.10g}</td>'
                    f'<td>{error:.3g}</td></tr>' if error is not None else
                    f'<tr><td>{row["control"]}</td><td>{target:g}</td>'
                    f'<td>{paired:.10g}</td><td>{total:.10g}</td><td>—</td></tr>')
    (output / "report.html").write_text('''<!doctype html><meta charset="utf-8">
<title>Direct visibility SNR_corr audit</title>
<style>body{font:17px sans-serif;margin:40px;max-width:1250px}td,th{padding:9px;border-bottom:1px solid #ccc}table{border-collapse:collapse}</style>
<h1>Direct visibility SNR_corr audit — 2241+098</h1>
<p>All values below are calculated by rereading stored complex DATA after injection.
SNR_corr_actual = sqrt(sum(|Y − Y_reference|²))/σ0. The original per-component
reference sigma is fixed. The selection contains finite unflagged parallel-hand
cross-correlations, matching the existing metric.</p>
<p>The first block scales one fixed thermal realization by the requested standard-deviation
multiplier. Its paired difference from regular noise is |a−1| n0. Thus noiseless a=0 has
zero total thermal norm but a nonzero difference from regular noise. These are separate
diagnostics, not an adoption of a new noise convention for the imaging experiment.</p>
<p>The second block repeats the existing independent extra-noise injection, adds the shared
baseline noise, then measures Y−Y_regular. Every target passes relative tolerance 1e−5.
These paired measurements also agree with the library metric to relative tolerance 1e−12.</p>
<p>Unit/analytical checks cover complex magnitude, zero, scaling, and invalid reference sigma.
This audit does not perform imaging. Temporary MS copies were removed after verification;
original datasets were read only. <a href="audit.json">Full numerical audit</a>.</p>
<table><tr><th>Control</th><th>Requested value</th><th>SNR_corr_actual<br>difference from regular</th>
<th>Total thermal norm<br>relative to noiseless sky</th><th>Relative target error</th></tr>
''' + "\n".join(rows) + "</table>")


def run(output: Path, source_id: str):
    from scripts import create_dataset_v1 as g
    from scripts.simulation import add_thermal_noise_inplace
    from scripts.corruption import measure_corruption_metrics

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((g.THERMAL_DIR / "report.json").read_text())
    source = g._thermal_source(g.THERMAL_DIR, manifest, source_id)
    model, params, sigma = g._noise_request(source)
    v = g._prepare_V_ms(source, output)
    baseline = output / "regular.ms"
    shutil.copytree(v, baseline)
    seed = g._stable_seed(source_id, "thermal_noise")
    add_thermal_noise_inplace(baseline, noise_model=model, noise_parameters=params, seed=seed)
    regular = measure_pair(v, baseline, sigma)
    rows = []
    # Scale one fixed Gaussian realization: all levels are exactly paired.
    # These multiplier diagnostics are separate from the existing independent
    # extra-noise injections below.
    for multiplier in (0., 1., 2.5, 5., 100., 1000., 2000., 4000.):
        ms = output / f"multiplier_{multiplier:g}.ms"
        shutil.copytree(v if multiplier == 0 else baseline, ms)
        if multiplier not in (0., 1.): g._rescale_visibility_delta(v, ms, multiplier)
        total = measure_pair(v, ms, sigma)
        paired = measure_pair(baseline, ms, sigma)
        np.testing.assert_allclose(total["SNR_corr_actual"], multiplier * regular["SNR_corr_actual"], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(paired["SNR_corr_actual"], abs(multiplier-1) * regular["SNR_corr_actual"], rtol=1e-6, atol=1e-7)
        rows.append({"control": "total_std_multiplier", "requested": multiplier,
                     "total_thermal": total, "difference_from_regular": paired})
        shutil.rmtree(ms)  # Only this diagnostic's explicit newly created copy.
    for target in (1., 2.5, *g.NOISE_SNR_CORR_TARGETS):
        ms = output / f"extra_snr_{target:g}.ms"
        shutil.copytree(v, ms)
        control = g.NoiseControl(target, f"noise_snr_{target:g}")
        injection = g._apply_noise_control(source, control, v, ms, sigma, model, params, seed)
        before_baseline = measure_pair(v, ms, sigma)
        add_thermal_noise_inplace(ms, noise_model=model, noise_parameters=params, seed=seed)
        paired = measure_pair(baseline, ms, sigma)
        library_check = measure_corruption_metrics(baseline, ms, sigma)
        np.testing.assert_allclose(paired["SNR_corr_actual"], target, rtol=1e-5)
        np.testing.assert_allclose(paired["SNR_corr_actual"], library_check.SNR_corr, rtol=1e-12)
        rows.append({"control": "extra_SNR_corr_target", "requested": target,
                     "before_baseline": before_baseline, "difference_from_regular": paired,
                     "total_thermal": measure_pair(v, ms, sigma),
                     "generator_measured": injection["SNR_corr_measured"]})
        shutil.rmtree(ms)
        print("Verified extra SNR_corr", target, flush=True)
    record = {"source": source_id, "baseline_seed": seed, "regular_thermal": regular,
              "definition": "SNR_corr_actual=sqrt(sum(abs(Y-Y_reference)**2))/sigma0",
              "selection": "finite unflagged parallel-hand cross-correlations; paired flag intersection",
              "rows": rows, "passed": True}
    (output / "audit.json").write_text(json.dumps(record, indent=2) + "\n")
    render(output)
    shutil.rmtree(baseline)
    g._cleanup_V_work(source, output)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", default="2241+098")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.render_only:
        render(args.output)
    else:
        run(args.output, args.source)
