# Injection bounds pilot: historical v1 implementation

The original-flux, two-antenna, unified-report protocol is in [Simulation parametrization v2](simulation_parametrization_v2.md). This page documents the preserved 2026-11-02 pilot.

Scientific protocol: [RadioAstroML – Injection Bounds Pilot](</Users/u1528314/Documents/Obsidian Vault/Notes/RadioAstroML - Injection Bounds Pilot.md>). Reviewed against the current generator and recovery pipeline on 2026-11-02. Implement isolated sweeps and their report, without changing datasets or starting mixture generation.

## Reuse the current code

| Existing code | Reuse |
| --- | --- |
| `scripts/noise_uv_experiments.py` | `training_sources()`, `read_grid()`, and the noiseless-reference, fixed-grid/beam, FITS/support export and checkpoint logic inside `recovery_source_run()` |
| `scripts/create_dataset_v1.py` | Source discovery, `_simulation_payload()`, `_noise_request()`, `_prepare_V_ms()`, stable seeds and source metadata including predicted image RMS |
| `scripts/corruption/metrics.py` and `core.py` | `measure_constant_gain_norms()`, `solve_constant_gain()`, `measure_corruption_metrics()` and `AntennaGainCorruption.from_constant_gain_solution()` |
| `scripts/clean_recovery_noise_sweep.py` | `geometry()`, `load_products()`, metric calculations and plotting patterns |
| `scripts/simulation`, `scripts/imaging` | Noise generation, nominal weights, `image_ms()`, explicit restoring beam and compressed FITS export |

Do not build on `single_amp_vs_phase_corruption_simulation.py`: that driver applies gains to already-noisy data. Do not use `run_clean_recovery.py` unchanged: it hard-codes the old pilot/training selections. The recovery implementation currently derives the smoothed model as `I-R`, and its settings hash includes the complete level list. Both need explicit handling for this pilot.

## Small implementation changes

1. **Extract only reusable image/checkpoint code.** Move the nested imaging/export/reference checks from `recovery_source_run()` into a small `scripts/recovery_products.py` helper, keeping CASA imports lazy. Let both the existing recovery runner and the new pilot call it. Preserve existing defaults and records. Generalize case metadata rather than duplicating the imaging pipeline. Keep raw-model FITS, support masks, convergence records and invariant/product hashes.

2. **Add `scripts/injection_bounds_pilot.py` as the CASA runner.** Select three training sources by stored flux/nominal image RMS and two ants by min/max positive `V_Ak_over_sigma_L2`. Measure/cache ant norms once per source using the existing validity rules. Use seeds 0–2 and the note's exact initial grids, signs, caps and refinements. Convert physical gains to solver targets using `S_A=Q*abs(A-1)` and `S_P=2*Q*abs(sin(phi/2))` with radians. Bypass the positive-strength solver for controls and deduplicate ±180°.

3. **Preserve exact pairing.** Prepare noiseless `V` and one ordinary noisy MS per source/seed. For gain cases, corrupt a fresh noiseless copy, verify its standalone perturbation, then add the cached noise difference `ordinary-V` in chunks. For noise cases reuse `_rescale_visibility_delta(V, observed, a)`. Restore nominal weights and check flags/UVW/weight hashes. Never rescale a gain-plus-noise difference to change only noise. Share ordinary controls across ants and image thermal levels only once per source/seed.

4. **Use extensible case records.** Save immutable source/imaging/seed specifications separately from an append-only case manifest. Give each case a canonical ID and hash from its actual parameters and reference, not the entire evolving grid. Refine only after all three seeds of a context/level complete. Implement the note's two refinement rounds, extension sequence, crossing/plateau checks and unresolved statuses. Evaluate shared lower-bound candidates in every required context before accepting them. Resume completed cases without invalidating them when new levels are appended. Reuse old products only after configuration/reference checks, not by matching filenames.

5. **Add direct model convolution and a pilot report.** Add a pure helper in `clean_recovery_noise_sweep.py` for zero-padded convolution with the rotated restoring Gaussian on the FITS pixel geometry. The kernel must have peak one, not unit sum. Allow `measure()` to accept precomputed smoothed current/control models while retaining its old default for existing reports. Calculate `y=e_TM,2`, same-seed `y0`, signed `d=y-y0`, flux diagnostics, raw-model nonzero count and signed flux. Add `scripts/injection_bounds_report.py` for CASA-independent metrics, refinement recommendations and rendering. Reuse the current Matplotlib/HTML approach, but omit Kneedle.

The report writes `metrics.csv`, `bounds.json`, a next-case list and `report.html`. Include per-context seed curves/median/min–max, guides at 0.05/0.9/1, signed-effect curves, empty-model and stop-code summaries, representative images, proposed bound/status tables and run costs. Plot points link to case records/products. Missing or failed cases remain explicit and cannot satisfy a bound criterion. Refresh after each completed batch and at completion. Proposed invocation uses `bin/run casa scripts/injection_bounds_pilot.py ...` and `bin/run ml -m scripts.injection_bounds_report ...`.

## Validation before the full pilot

- Pure tests: point-model convolution matches analytic truth, an empty model gives `y=1`, negative components remain signed, and rotated beams/edges produce no wraparound or pixel shift.
- Scheduler tests: phase endpoints and shared controls deduplicate, adaptive resume is a no-op for finished cases, nonmonotonic/capped/baseline-degraded curves get the specified statuses, and missing seeds cannot establish bounds.
- One-source CASA smoke: control, amplitude, phase and scaled-noise cases. Verify injection values, identical additive noise across gain cases, fixed imaging invariants, direct-convolution metrics, report links and successful resume.

Run relevant existing recovery/corruption tests after the extraction. Use a fresh run directory. Existing modified files and recovery outputs are unrelated working state and must not be reset or overwritten.

## Review decisions and execution

The scientific definitions and proposed grids are retained. For an even-sized training partition, use the upper middle rank (`n//2`, zero-based), with source ID breaking SNR ties. Both antenna extrema break ties by lowest antenna ID. Equivalent phase endpoints and ordinary controls share physical cases. The complete initial three-template/three-seed grid has 579 unique images.

The shared imaging code is a lazy-import factory in `recovery_products.py`; it preserves the existing recovery record format and adds optional case metadata. Immutable source settings exclude adaptive levels. `cases.json` is append-only; two global refinement rounds are followed by sequential strong-end extensions. A refinement round uses the medians of the complete preceding batch. Missing seeds leave gaps and block advancement. Weak candidates are tested across baseline-good contexts, and only common tested small-error values can become lower bounds. No monotonic fit or Kneedle is used.

Gain tables are checked against the requested complex gains before application, and standalone visibility perturbations are checked before adding noise. Identical cached `ordinary−V` noise is added in chunks; float32 MS storage introduces a small, explicitly measured rounding error. Nominal weights are restored before imaging and checked together with flags/UVW. Direct convolution uses a peak-one Gaussian on the FITS tangent-plane pixel matrix with zero padding and no pixel shift.

```sh
bin/run casa scripts/injection_bounds_pilot.py --output runs/injection-bounds-smoke-20261102 --smoke
bin/run casa scripts/injection_bounds_pilot.py --output runs/injection-bounds-pilot-20261102
bin/run ml python -m scripts.injection_bounds_report --output runs/injection-bounds-pilot-20261102
```

The smoke scope is explicitly separate: one selected training template, seed 0, noiseless/ordinary controls, noise a=5, gain A=1.5 and phase +30°. Its report does not establish pilot bounds. Repeating the same command resumes unfinished cases; completed records and references are checksum-validated. Reports refresh after each source/seed batch and each scheduling round. `status.json` records process ID, elapsed time and initial/refinement/extension/reuse counts; `bounds.json` remains provisional until the search completes. All outputs stay in the new run directories.

Validation: 72 ML-environment tests ran (71 passed, one existing CASA-only test skipped), plus all 10 corruption-package tests passed under CASA. The five-image smoke and an independent audit passed: A=1.5, measured phase30.0000004°, scaled noise exactly5× within storage precision, common gain-case noise hash, identical nominal weights/flags/UVW/reference and fixed source components. Maximum paired-noise storage rounding was6.7e−8Jy. A no-op resume reused all5images and preserved every record hash. Initial smoke extraction failures were repaired and all failure checkpoints cleared; its log retains that debugging history.

The full pilot is launched in `runs/injection-bounds-pilot-20261102/` with low/middle/high templates0449+113,1415+133,1044+809 (nominal SNR81.853,2584.748,11037.421). The report also refreshes during long batches about every120seconds. Individual seed points are visible before a three-seed median exists; missing levels remain gaps. Completion and measured recommendations must be read from the live `cases.json`/`bounds.json`, not inferred from this launch record. The numerical noiseless check has a=0,q=−1 and no paired ordinary-control d; it is outside the excess-noise search axis.
