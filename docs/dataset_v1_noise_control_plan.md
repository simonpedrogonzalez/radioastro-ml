# Dataset v1: D noise controls — minimal implementation plan

Status: plan only; no samples or pipeline code changed.

Based on the vault note [Unseen-corruption Checking](</Users/u1528314/Documents/Obsidian Vault/Notes/Unseen-corruption Checking.md>): test whether extra noise is mistaken for amplitude/phase errors, using the existing trained model. **D is a sample family, not a fourth classifier class. Its correct label is `0` (none).** No retraining.

## Scope and storage

- Reuse the pinned dataset root: `/Users/u1528314/Documents/radioastro-ml/collect/experiments/dataset_v1_20260921_all_psf`.
- Build on the [incremental/PB/level-expansion plan](dataset_v1_incremental_pb_repair_plan.md), which is not implemented yet. Reuse its precursor preparation, resume checks, final imaging with `pblimit=-0.1`, and dirty/CLEAN/residual/PSF retention.
- Use the same requested levels: **5, 10, 15, 20, 30, 40, 50, 100**. Baseline is the existing level-0 sample; do not duplicate it.
- Proposed default: validation sources only; allow an explicitly selected test partition after the procedure is fixed. Preserve existing source partitions. Do not generate training controls or fit the model on D in this experiment.
- Store samples as `samples/<source>_noise_snr_<level>/`, with a companion index **`dataset_noise_controls.json`** in the same dataset root. The loader already accepts `index=...`. This keeps D out of ordinary training and leaves the main index's 17-variant completion rule and existing label IDs unchanged. Each D manifest uses `label_name="not_corrupted", label_id=0`; its index maps that name to 0.
- Resume by checking each requested source/level independently. Skip valid indexed D samples; complete missing ones without changing existing gain/baseline products. Use existing staging/integrity handling, not a second resume framework.

## One entrypoint, incremental work

**Generation stays in `scripts/create_dataset_v1.py`; there is no new generator script.** Its D mode/helper shares the existing source preparation, imaging and sample finalization. Small supporting edits in the labeling/loader and evaluation modules are still necessary; this is not a change confined entirely to one file.

Together with the incremental/PB plan, the script checks the pinned dataset before doing expensive work:

- Skip completed, valid square baseline/gain samples, including their simulation and imaging.
- Generate only missing amplitude/phase levels and samples for newly added sources; prepare a missing precursor when needed.
- Regenerate existing circular samples only for the planned PB repair, preserving their variant identity and split.
- When D generation is enabled, inspect its companion index and add only missing D source/level combinations. Completed D samples are skipped too.

Determine pending work before reconstructing a source's temporary noiseless MS. Reuse that reconstruction across its pending variants; if nothing is pending and report outputs are current, do not simulate, image or redraw reports. Missing/stale reports can be rebuilt from retained products without regenerating samples. An incomplete source does not justify repeating its completed variants.

`dataset_noise_controls.json` is only a separate **index in the same dataset directory**, not a second pipeline or a fresh dataset build. Sources selected for D have up to **25 variants: 1 baseline + 8 amplitude + 8 phase + 8 D**. Training sources remain at 17 under the proposed held-out-only D policy. These are implementation requirements, not functionality already available in the current script.

## What “same magnitude” means

Use the definition already implemented in `scripts/corruption/metrics.py`, not a new image-RMS definition:

\[
Y_0=V+n_0,\qquad Y_D=V+n_0+\eta,\qquad
S=\mathrm{SNR}_{corr}=\|\eta/\sigma_0\|_{2,I}.
\]

Here `V` is the noiseless sky, `n0` is the existing shared baseline thermal-noise realization, and `eta` is the **extra** noise. `sigma0` is the original thermal standard deviation per real/imaginary component. `I` is the existing metric's finite, unflagged, parallel-hand cross-correlation selection. Keep this reference sigma fixed, even when the final MS weights describe increased noise.

Draw an independent complex Gaussian noise vector `z`, then normalize its **realized** magnitude:

\[
\eta=z\,\frac{S}{\|z/\sigma_0\|_{2,I}}.
\]

Setting a Gaussian standard deviation alone only matches the target on average. Normalizing the draw fixes its norm; strictly, this is a fixed-norm Gaussian-derived noise control, not unconstrained independent Gaussian noise. Floating-point storage prevents literal exact equality: require **relative error ≤ `1e-5` (0.001%)** after writing and rereading the MS. Fail the sample if this is not achievable; do not silently relax the tolerance or substitute a smaller level.

This matches the gain examples' **visibility perturbation norm**, not necessarily their residual-image RMS. For `N` valid complex samples, the equivalent per-component extra RMS is `sigma_extra = sigma0 * S / sqrt(2*N)`. The total noise RMS is approximately `sqrt(sigma0² + sigma_extra²)`. For validation source `1927+612`, retained metrics give `N=1,123,528`: even `S=100` increases total RMS by only about **0.22%**. A stronger image-RMS-matched stress test would be a separate experiment; do not inflate D's noise to make it visually obvious.

## Small code changes

### 1. Generation: `scripts/create_dataset_v1.py`

Add a noise-control mode using the same source preparation and `_finalize_sample` path. Keep the existing gain branches and stable seeds unchanged; D does not need antenna selection, a gain solve, or a gain table.

One local helper is enough:

1. Copy noiseless `V` into the work MS. Use existing `add_thermal_noise_inplace` with `simplenoise=sigma0` and a dedicated stable seed derived from source and `noise_snr_<level>`; never reuse the baseline-noise seed.
2. Measure the raw draw with `measure_corruption_metrics(V, work_ms, sigma0)`. Reject zero/nonfinite norm. In a chunked DATA loop, replace it with `V + (S/measured_SNR) * (work_ms - V)`; preserve flags and MS structure.
3. Reread and measure with the same function. Check the tight tolerance before continuing. Only a bounded corrective rescale is warranted if storage rounding requires it; otherwise fail explicitly.
4. Add the existing baseline noise using the same request and source seed as the baseline/gain variants. Thus D is **extra** noise, not a replacement realization of baseline noise.
5. Set final SIGMA/WEIGHT consistently with `sigma_total = sqrt(sigma0² + sigma_extra²)`, reusing the existing weight initializer. Two calls to `add_thermal_noise_inplace` alone are insufficient: each call resets weights to its own sigma. Scope this correction to D; preserve the baseline-noise request and record the final total sigma separately.
6. Image and finalize normally, with empty gain-corruption reports. Supply the companion index and label 0 independently of the sample's `noise_snr_<level>` suffix.

Extend `_write_branch_simulation_reports` with an optional `noise_control` block in the existing `simulation.json`: `kind="increased_noise"`, target and measured `SNR_corr`, reference sigma, valid count, extra-noise seed, applied scale, equivalent extra sigma, and final total sigma. Record truthful generation/normalization stages; do not represent D as an antenna-gain corruption. Existing simulation metadata is already retained and checksum-covered, so no new metadata file or schema framework is needed.

### 2. Labels: `scripts/preprocessing/labels.py` and `dataset.py`

The current system only partly supports this: no gain report gives label 0, **but also forces both SNR fields to 0**. Add a small optional noise-control input to `assign_label` and a `sample_kind` field to its result:

- Ordinary baseline: label 0, kind `baseline`, target/measured level 0.
- Existing gain sample: label 1/2, kind `gain`, existing target/measured values unchanged.
- D: label 0, kind `increased_noise`, target/measured values from `simulation.json.noise_control`.

Return the kind alongside the existing `corruption_snr` and `corruption_snr_target` in `label_metadata`. These numeric fields describe perturbation strength, **not whether an amp/phase error exists**. Reject contradictory D metadata plus a gain report, or invalid/nonpositive D levels. Read simulation metadata once when assigning the criterion, including when `load_metadata=False`. The existing collator already carries these dictionaries; it needs no change. Raw manifest labeling also remains 0 for D.

### 3. Evaluation: `ml/evaluate.py`

Evaluate D separately through the companion index, using the unchanged model and preprocessing. The current positive-level branch assumes a gain error, so it would misleadingly call D false positives “detection recall.” Select that branch from the true class instead of `level > 0`; reject mixed gain/control groups rather than silently pooling them.

For each D level report count and **false-positive rate = number predicted amp or phase / number of D samples**. Correct rejection is its complement; amp/phase type accuracy is not applicable. Keep ordinary gain results unchanged and do not combine D with them into a new overall score. A small table is sufficient; no new reporting framework.

### 4. Update the same dataset HTML report with all variants

Extend the report assembly in `scripts/create_dataset_v1.py` and the existing `scripts/reporting/create_dataset_v1.qmd` template, including the pinned dataset's local `report.qmd`. Read both indexes for reporting only; do not merge their training populations or label maps.

- Keep the baseline/amp/phase sections covering their 17 variants. Add an **increased-noise (D)** section for each selected source, covering all eight requested levels. Thus the same dataset report covers all 25 variants where D is enabled.
- Show dirty, CLEAN, residual and PSF images using the existing plotting helpers, layout and color conventions. Include the existing baseline as a visual reference, without creating or counting another sample. Match display scales within each product's comparison grid.
- Include D's existing image metrics, target and measured `SNR_corr`, and true label `0 (none)`. Clearly identify it as increased noise, not an amp/phase gain error. Model-evaluation scores belong to the separate evaluation step; dataset creation need not run a model.
- Show available variants even if another variant failed or is missing; mark each absent requested level explicitly. Distinguish D not requested for a source from failed D generation, and count main and D samples separately.
- After generation/repair, refresh image grids and metric rows only for affected sources, then render the updated shared HTML. Recover missing/stale report outputs from retained FITS/metadata after an interruption. A second unchanged run does no plotting or rendering and never regenerates samples just to update a report.

## Essential checks

1. **Magnitude feasibility — before bulk generation.** On scratch copies of one validation MS, generate all eight levels. Reread DATA and verify `measure_corruption_metrics(V, V+eta, sigma0)` meets `1e-5` relative tolerance using the real flags/correlations/dtype. Also reconstruct a paired baseline with the same thermal seed and check the final `YD - Y0` norm against the target. Check that adding baseline noise preserved the intended perturbation within storage precision. Measure the added noise, not `YD-V`, which includes baseline noise. Retain the measured values and failures in QA before temporary MS cleanup.
2. **Noise/weights sanity.** Verify independent extra/base seeds, repeatable generation, approximately zero-mean real/imaginary extra noise, and final SIGMA/WEIGHT matching the total sigma. Sampling scatter is expected in distribution checks; the perturbation-norm tolerance is the hard acceptance check. Confirm no gain table was applied.
3. **Labels and evaluation.** A small fixture checks baseline, amp, phase and D labels/levels, including `load_metadata=False`, and rejects conflicting metadata. For D predictions `[0, 1, 2, 0]`, FPR must be 50%, with no detection-recall/type-accuracy claim. Existing gain fixtures must retain their results.
4. **Safe integration.** A one-source smoke run retains all four FITS products, intended PB support and image geometry, valid metadata/checksums, and the original partition. Rerun: no work. Remove one scratch D entry: only that level is rebuilt. Existing main-index/sample hashes stay unchanged; the default training loader never sees D.
5. **Report coverage and resume.** Inspect the smoke-run HTML: all 17 main variants plus eight D variants appear with all four image products and matching metric values, with the baseline counted once. A missing/failed variant is shown explicitly without hiding completed ones. Adding one missing D level refreshes its source and the shared HTML without rebuilding other source grids or samples; an unchanged rerun leaves report outputs untouched.

Preliminary check performed while writing this plan: an in-memory `complex64` toy using the validation source's stored `N` and sigma achieved relative norm errors below `3e-8` at all eight levels, including paired baseline subtraction. This supports numerical feasibility but **does not replace the CASA/MS check above**.
