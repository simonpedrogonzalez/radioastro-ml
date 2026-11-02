# V1 regression: common corruption SNR and training noise

Status: the original eight-level experiment is complete. All 800 training-noise samples, full model fits and final checks passed.
The [2026-10-26 noise-strength extension](noise_strength_extension.md) adds 1000,
2000, 3000 and 4000 to training and validation; it supersedes the eight-level
counts and frozen noise-index policy below for that new experiment.
Open the [HTML report](../ml/runs/20261022T203038478771Z_multi_regression_report/report.html). The current
regression code uses extra-noise SNR_corr with clean target (0,0,0). The earlier
kappa report and separate-evaluation reports are preserved as historical artifacts.
The current report pools all fault types, includes noise in Total and mixture,
and omits displayed R². See execution results below. A subsequent
[combined6 tree-strategy comparison](combined6_strategies.md) reuses these data,
folds and report format.

## Agreed targets

All three outputs measure additional perturbation using the original thermal
noise reference and existing visibility selection: `SNR_corr = ||Delta V/sigma0||_2,I`.
For noise, `Delta V = eta` is the extra injected noise, excluding the ordinary
baseline realization `n0`. Normal thermal noise remains present in every sample.

| Sample | Target `(amp, phase, noise)` |
| --- | --- |
| Ordinary baseline | `(0, 0, 0)` |
| Pure amp error | `(measured amp SNR_corr, 0, 0)` |
| Pure phase error | `(0, measured phase SNR_corr, 0)` |
| Extra-noise sample | `(0, 0, measured extra-noise SNR_corr)` |

Use gain `metrics.SNR_corr` and noise `simulation.noise_control.SNR_corr_measured`
for targets; requested SNR defines strength bins. Total/reference sigma may remain
diagnostic metadata, but is not a regression target. No subtraction of 1 is needed.
Equal SNR means equal total noise-weighted perturbation magnitude, not identical
image effects or detectability. The existing noise controls are valid despite
barely changing total noise RMS. Preserve their fixed-norm Gaussian-derived draw,
reference sigma, valid visibility selection, seeds, and measurement tolerance.

## Dataset extension

Use `collect/experiments/dataset_v1_20260921_all_psf/` and add a companion index:

- Preserve `dataset.json`, existing samples, `dataset_noise_controls.json`, and
  existing reports. Classification keeps its current labels and populations.
- Add **`dataset_noise_training.json`** containing only noise samples from sources
  already represented in main training. Retain manifest label `not_corrupted: 0`
  and metadata kind `increased_noise`; no preprocessing schema change is needed.
- Reuse `samples/<source>_noise_snr_<level>/` and levels **5, 10, 15, 20, 30, 40,
  50, 100**. Training and held-out source IDs are disjoint, so paths do not collide.
- The current 100 training sources need **800 new samples**, giving **2,493 training
  samples** including the existing 1,693. Reuse the 356 main validation samples and
  168 validation noise controls. No baseline duplication or validation regeneration.
- Preserve source splits; do not use reserved test data or add gain/noise mixtures.

## Minimal code changes

### 1. Training-noise-only mode in `scripts/create_dataset_v1.py`

Keep the existing entrypoint and scientific helpers. Add `--noise-training` and
optional repeatable `--noise-training-source` for a pilot; the latter implies
training-noise mode. Reject combinations with held-out-control flags, whose
behavior and train-source rejection remain unchanged.

Dispatch this mode before the ordinary `run_experiment()` path. That path can
retry missing gain variants, repair samples, and rewrite reports; the eight
recorded gain failures must not trigger work or block this extension.

A small orchestration function in the same file should:

1. Read the main index and existing thermal precursor. Select only train sources
   with an indexed baseline and completed precursor; reject explicitly selected
   validation/test/unavailable sources. Do not call `_ensure_thermal_entries()`,
   migrate labels, or repair existing main samples.
2. Audit the new index and require train-only membership within main training.
   Reuse `_audit_noise_controls()` and `_assert_noise_control_sample()` for sample
   metadata, integrity, measured SNR, and imaging checks.
3. Find missing source/level pairs before preparing an MS. Reuse
   `_requested_noise_controls()`, `_prepare_V_ms()`, `_noise_request()`, and
   `_finalize_sample(..., noise_control=...)`. Leave `_apply_noise_control()` intact:
   independent seeded extra draw, fixed measured norm, shared baseline noise added
   afterward, and final weights consistent with total noise.
4. Append verified samples through `_append_sample_to_dataset()` and existing
   atomic helpers. Scope interrupted-sample recovery to new training-noise paths;
   reject conflicts rather than replace valid existing samples. Prepare noiseless
   V once per pending source, check disk space, and clean temporary work afterward.
5. Write **`noise_training_report.json`** with configuration, source/level status,
   failures, measured strengths, and partition counts. Leave existing report files
   untouched; a new dataset HTML template is unnecessary. Validate completion
   independently of main gain failures. Completed reruns do no simulation/imaging.

`_iterate_and_validate_noise_controls()` currently rejects training samples.
Keep that default: add an explicit allowed-partitions argument for the new caller,
or a small training-only validation wrapper. Do not globally relax held-out guards.

### 2. Targets and training-index inclusion in `ml/regression.py`

- Rename output `kappa` to `noise`; read measured extra-noise SNR. Baseline and
  pure-gain noise targets become 0; the no-error predictor becomes `(0,0,0)`.
  Retain finite/positive noise-metadata checks and rejection of unsupported mixtures.
- Add explicit **`--noise-training-index PATH`**. Without it, retain ordinary
  training and mark constant-zero noise prediction provisional. An explicitly
  supplied missing/empty index is an error, not a silent fallback.
- Validate every extra-index manifest is a train-source noise sample before
  filtering, so mixed-partition indexes cannot silently pass. Load through existing
  `load_data()` / `FitsSimulationDataset`; concatenate all `RegressionData` fields
  into training, preserving feature/target/ID/metadata/product alignment.
- Replace the blanket prohibition on noise in training with explicit index, kind,
  source, and duplicate-ID checks. Added sources must belong to main training;
  held-out controls remain evaluation-only. Keep all variants of each source
  together in grouped CV. Never derive regression targets from class 0 alone.
- Preserve the six features, three views, independent-output XGBoost strategy,
  grid, seed, and training-only tuning. Existing normalized-MAE scoring includes
  noise once it varies within folds. Do not change classification entrypoints.
- Record target-definition version, extra-index fingerprint, source IDs, per-kind
  and per-strength counts, and training inclusion. Save raw/clipped predictions,
  models, and a fresh timestamped report; never reinterpret old checkpoints.

### 3. Evaluation and `ml/regression_report.py`

- Use `(amp, phase, noise)`, all in SNR_corr units, throughout plots, tables,
  vectors, CSV/JSON, and captions. Remove kappa-specific ranges, baseline-1 text,
  and subtraction of 1 in cross-talk and hard-sample scoring.
- Cross-talk uses direct nonnegative predictions divided by summed true strength
  for each pure-kind row; clean remains a mean. All columns now share units.
  A sum of predicted component strengths is still not the norm of their combined
  visibility perturbation. Keep that distinction explicit.
- Hard samples rank by summed absolute target errors in common SNR units, with
  each target error shown. Remove the obsolete mixed-unit explanation.
- Superseding the original reporting choice: pool all **524 held-out samples**
  (21 clean, 167 amp, 168 phase, 168 noise) in every evaluation section. Source
  splits, training, model files and predictions are unchanged.
- Detection uses **Total = amp + phase + noise**. Identification compares the
  three-component **mixture = (amp, phase, noise) / Total**. All 503 fault samples
  are eligible; positive predicted Total defines coverage. Clean has no mixture.
  Mixture MAE/MSE average samples and components, RMSE is sqrt(MSE), and aggregate
  R² averages defined component R². These replace the old gain-only share metrics.
- Strength slices pool all three fault types at the requested level. Absolute
  metrics also reuse clean samples. Relative component/Total errors divide by
  requested injected SNR, exclude clean (zero injected strength), and display
  percentages. Mixture is already normalized; its relative columns express the
  same errors as percentages. No per-level R² is computed or plotted.
- Report sections are **All types**, **Detection / ID**, and **Per corruption
  strength**. All types shows component MAE/RMSE; Detection / ID shows Total and
  mixture MAE/MSE/RMSE, with the best value in each column bold. R² is omitted
  throughout the report. R² and coverage remain in saved evaluation data.
  Strength plots start at a labeled zero on every y-axis.
- The constant baselines are **train_mean** (the component-wise means of all
  training targets) and **no_error** (0,0,0), in both fitting and report refresh.
- Label rows at the left of both line and scatter plots. Scatter color and shape
  both identify the true fault type. Remove separate noise and higher-gain hard
  sample sections, simplify data/fitting text, and move training coverage last.
- Warn about untrained noise variation only when training noise targets are all
  zero. Added samples permit training, not a claim of success. MAD-normalized
  parity features may be insensitive to diffuse extra noise; measure that outcome
  without changing features or increasing injection strengths to improve scores.

### 4. Focused validation

- Update `ml/tests/test_regression.py`: clean `(0,0,0)`, measured noise targets
  distinct from requested SNR and sigma ratios, no baseline subtraction, zero
  baseline predictor, report labels, saved-model reload, and existing metric cases.
- Test training-index admission/rejection, duplicate IDs, aligned concatenation,
  source-disjoint validation, and grouped CV containing every variant together.
- Extend `tests/test_create_dataset_v1.py`: training-only selection, new index/report
  routing, unchanged held-out mode, recovery/no-op reruns, and no retries of main
  gain failures or writes to existing reports.
- Run one training-source CASA pilot at all eight levels. Check measured extra-noise
  SNR within the existing `1e-5` relative tolerance, paired-baseline subtraction
  after final noise addition, weights, FITS products, and seeds. Snapshot old
  indexes/sample/report hashes before and after; rerun to verify no regeneration.
- Then generate the remaining sources, validate all 800 samples, run relevant ML
  and generator tests, and produce a new full regression report.

## Commands

From the repository root:

On this machine, prefix CASA commands with
`CASASITECONFIG="$PWD/runs/v1-noise-training/casa_config.py"` to reuse installed
CASA data without the startup download that stalled during this run.

```sh
# Eight new noise samples for one training source, in the existing V1 root.
bin/run casa scripts/create_dataset_v1.py --noise-training-source 0005+383

# Complete remaining training sources, reusing finished samples.
bin/run casa scripts/create_dataset_v1.py --noise-training

uv run --project ml --no-sync python -m ml.regression \
  --dataset collect/experiments/dataset_v1_20260921_all_psf/dataset.json \
  --noise-training-index collect/experiments/dataset_v1_20260921_all_psf/dataset_noise_training.json

# Recompute evaluation and render a NEW report from saved predictions; no fitting.
uv run --project ml --no-sync python -m ml.regression_report \
  --from-run ml/runs/20261022T152108411795Z_multi_regression
```

## Existing artifacts and readiness

The earlier [report](../ml/runs/20261022T010125861380Z_multi_regression/report.html)
validates the previous kappa implementation only; it remains unchanged. The
revised implementation passes 32 targeted generator/regression tests and all
34 ML tests, including an actual Quarto render.
The Obsidian **Multi Regression Setup** note still describes the superseded kappa
choice; it has not been edited as part of this implementation.

The copy audit verified all 25,288 checksum-listed files for 2,423 main V1 samples
and 168 validation controls. All 143 source prerequisites pass generator checks;
original and precursor MS headers exist. Original visibility payloads were not
fully checksummed. About 100 GiB was free at the audit; recheck before generation
and clean temporary MS work source by source.

## Execution results

- One-source pilot: all eight 0005+383 samples generated and validated.
- Independent CASA checks on scratch copies verified paired-baseline subtraction
  and SIGMA/WEIGHT/WEIGHT_SPECTRUM at all eight levels; maximum relative SNR error
  was about 3.1e-9, within the existing 1e-5 tolerance.
- Pilot rerun performed no generation and preserved all 82 pilot files and mtimes.
- All 46,654 original dataset/report files were snapshotted before generation;
  full SHA256, size and mtime verification passed after all 800 samples completed.
- Four disjoint source groups use separate temporary worker indexes/reports in
  the dataset root. The execution driver merges only validated, nonduplicated
  samples into the canonical training index, then invokes the regression CLI.
  Public generator commands remain sequential and resumable.
- Logs, snapshots, source grouping and validation scripts are in
  `runs/v1-noise-training/`; monitor `bulk.log` and `worker0.log`–`worker3.log`.
- CASA startup downloads stalled on the network; a run-local CASASITECONFIG
  disables automatic data updates and reuses installed CASA data. No global
  environment configuration was changed.
- Generation completed all 100 train sources at all eight levels, with no failures.
  The merged training index contains 800 samples; regression trains on 2,493 total.
- Fixed-SNR noise targets have tiny measured differences (relative range up to
  3.6e-8). R² is undefined for truth constant within rtol=1e-7, atol=1e-12;
  MAE/RMSE retain the measured targets. This avoids meaningless enormous negative
  per-level R² values. The tolerance is recorded in config.json; 11 regression
  tests, including this case and an actual Quarto render, pass after the fix.
- The earlier separate-evaluation report audit passed: all three saved models reproduce all 1,572
  prediction rows; 100 local links resolve; all six figures are embedded;
  training/CV source isolation holds. Predictions are byte-identical to the first
  full run; the numerical R² correction changes only the affected metrics.
- The earlier separate evaluation had combined6 main-validation amp R² **0.916** (MAE **3.365**) and phase R²
  **0.891** (MAE **4.205**). On the separate noise controls, noise R² is **−0.306**
  and MAE **22.096**. Adding the samples enables training but does not yield good
  noise-strength estimation with these features at the existing injection levels.
- Reporting revision: 12 regression tests pass, including numerical checks of
  all-three-kind Total/mixture, relative errors, clean exclusion and Quarto output.
  The report refresh preserves models, training features and all 1,572 predictions;
  all 524 validation samples are evaluated together. Audit artifacts are in
  `runs/v1-pooled-report/`.
- Pooled combined6 R²: amp **0.922**, phase **0.868**, noise **0.152**; Total MAE
  **14.205**, mixture MAE **0.161**. Differences from the earlier report reflect
  the changed evaluation population/definitions, not model retraining.
