# Extra-noise strength extension

Decision: 2026-10-26. Retain the existing extra-noise SNR_corr levels and add
**1000, 2000, 3000 and 4000** to both training and validation. Complete: all 484
additional samples were generated on 2026-10-26; all four model setups and the
independent report checks completed on 2026-10-27 (local time). A preservation
check required the recovery described below.
Execution artifacts: `runs/v1-noise-extension-20261026/`.

[Completed four-setup report](../ml/runs/20261028T030413268146Z_combined6_training/report.html)
and [independent validation](../ml/runs/20261028T030413268146Z_combined6_training/validation.json).

## Definition and scope

The full noise list is **5, 10, 15, 20, 30, 40, 50, 100, 1000, 2000, 3000, 4000**.
Amplitude and phase keep **5, 10, 15, 20, 30, 40, 50, 100**.
`SNR_CORR_TARGETS` remains the gain list; `NOISE_SNR_CORR_TARGETS` is the noise
list in `scripts/create_dataset_v1.py`.

The target is still measured extra-noise `SNR_corr = ||eta/sigma0||_2,I`, with
`Y = V + n0 + eta`. The generator uses the same source-dependent baseline noise,
independent deterministic extra-noise seeds, realized-norm normalization,
original reference sigma, valid visibility selection and relative tolerance
`1e-5`. Final visibility weights use the total noise sigma. Clean targets remain
`(0, 0, 0)`; gain targets and the six model features are unchanged.

For `N` valid complex visibility samples,
`sigma_extra/sigma0 = S/sqrt(2*N)` and the equivalent total sigma ratio is
`sqrt(1 + S*S/(2*N))`. Consequently, these fixed SNR levels give different noise
increases across sources. The added range tests stronger, more observable noise;
it does not impose a fixed image-noise ratio or guarantee accurate regression.
See the [original-range audit](noise_identifiability_audit.md).

Using the retained valid visibility counts, the predicted median increase in
total visibility noise sigma is:

| Added SNR_corr | Training median increase | Validation median increase |
| --- | ---: | ---: |
| 1000 | 8.99% | 8.64% |
| 2000 | 32.35% | 31.18% |
| 3000 | 64.05% | 61.91% |
| 4000 | 100.17% | 97.05% |

These are analytical expectations, not measured image-noise changes. For
example, validation increases at S=4000 range from 2.13% to 230.04% across
sources. Values and ranges are saved in
`runs/v1-noise-extension-20261026/expected_noise_increase.json`.

## Data and generation

Use `collect/experiments/dataset_v1_20260921_all_psf/` and its existing sources.

| Population | Original noise samples | Added | Final noise samples | Final pooled rows |
| --- | ---: | ---: | ---: | ---: |
| Training: 100 sources | 800 | 400 | 1,200 | 2,893 |
| Validation: 21 sources | 168 | 84 | 252 | 608 |

Update `dataset_noise_training.json` and `dataset_noise_controls.json` only after
all requested samples pass integrity and partition checks. Existing scientific
products, sample manifests, main `dataset.json`, source partitions and previous
reports are preserved; see the cleanup-audit exception below. No
reserved test sources are generated or used. Baselines are reused, not duplicated.
The dataset's original HTML remains a historical report; generation progress is
recorded in `noise_training_report.json` and `noise_controls_report.json`.

The public resumable commands are:

```sh
bin/run casa scripts/create_dataset_v1.py --noise-training
bin/run casa scripts/create_dataset_v1.py --noise-validation-only
```

Both modes bypass gain generation and PB repairs. An unchanged rerun skips
completed samples. Configuration changes are accepted only for additive noise
levels with all other generation settings unchanged. The validation-only mode
accepts `--noise-control-source SOURCE` for a validation subset and rejects
training/test sources. The ordinary `--noise-controls` mode retains its original
combined dataset-generation behavior.

The execution uses four disjoint CASA source workers, each with an isolated
index and progress report. It recovers existing samples into those indexes,
adds only missing levels, audits every completed sample and merges the indexes.
SHA256, size and modification time snapshots protect pre-existing dataset and
ML-run artifacts; copies of the replaced indexes/report retain prior provenance.

## Four-setup comparison

Repeated the [combined6 comparison](combined6_training.md): `one_per_target`,
`shared_tree`, `one_per_type`, and `shared_by_type`. Used the expanded 2,893 training
rows and 608 validation rows, the exact three source CV assignments from
`ml/runs/20261024T014248110295Z_combined6_training`, seed 42 and the same four
hyperparameter candidates. The filtered per-output counts become 895 amplitude,
898 phase and 1,300 noise rows, each including the same 100 clean examples.
The shared type-specific variant masks each output's loss; shared splits still
use the union of rows. Final validation remains fully pooled for every setup.

Keep all reporting definitions and plot types: All types, Detection / ID,
Per corruption strength, cross-talk and hard examples; train_mean/no_error
baselines; no displayed R². At 1000–4000 the strength slices contain only noise
faults, plus the same clean samples for absolute metrics. Relative metrics
exclude clean rows. The report description states this population explicitly.

The execution wrapper `runs/v1-noise-extension-20261026/fit.py` uses the existing
regression implementation with combined6 only and restores the saved CV source
assignments. It then calls `compare_strategies(..., include_type_specific=True)`.
Each setup retunes on the expanded training data. Larger targets change squared
error loss, target standard deviations in CV scoring, training means and pooled
validation composition. Therefore, direct changes in pooled scores relative to
the old report do not isolate improvements at the original levels; inspect the
strength slices as well.

## Validation

- 38 generator/regression tests pass, including additive resume, unchanged gain
  levels, partition restrictions, no-op resume, reindexing without rewriting
  cleanup audits, and the masked training objective.
- Real-MS paired perturbation and SIGMA/WEIGHT/WEIGHT_SPECTRUM checks passed at
  all four added levels on source `0005+383`. Maximum paired relative norm error
  was `2.74e-11`, below `1e-5`; see
  `runs/v1-noise-extension-20261026/paired-check/validation.json`.
- Final checks cover expected counts, measured targets, fingerprints of existing
  artifacts, source/CV separation and the four-model report.
- All 1,452 noise samples passed manifest integrity, partition and measured-target
  checks. Maximum relative norm error was `6.26e-8`, below `1e-5`.
- All 3,017 original cached feature/target rows match the prior four-setup run
  exactly. The three source CV folds are identical. All 2,432 prediction rows,
  overall and 12-bin metrics for all models/baselines, mixture coverage and
  cross-talk were independently checked. All 16 CV candidates used three folds.
- The report retains the same sections and four plots. All 20 strength axes
  display zero as their lower bound, and all 48 product links resolve. The linear
  strength axis is unchanged; the 5–100 points are compressed by the expanded
  range, so the saved per-bin values in `results.json` are useful for that range.
- A real no-op rerun of both public noise modes completed without adding files
  or changing any of the 59,517 dataset files' sizes/modification times; see
  `runs/v1-noise-extension-20261026/noop_verified.json`.

## Results

All setups selected 100 rounds and learning rate 0.05. The two separate-output
setups selected depth 2; the two shared-tree setups selected depth 4.

| Pooled validation MAE | one_per_target | shared_tree | one_per_type | shared_by_type |
| --- | ---: | ---: | ---: | ---: |
| Amp | 2.731 | 5.128 | 6.574 | 11.795 |
| Phase | 3.691 | 6.273 | 9.974 | 15.594 |
| Noise | 643.838 | 648.877 | 1318.931 | 1680.562 |
| Total | 645.244 | 653.681 | 1328.101 | 1695.230 |
| Mixture | 0.250 | 0.221 | 0.376 | 0.380 |

The stronger injections are visible in the retained noise-scale feature:
validation residual-image MAD increased by medians **8.91%, 30.63%, 65.32% and
100.70%** at SNR 1000, 2000, 3000 and 4000, respectively, relative to each
source's clean sample. Training medians were 8.72%, 29.24%, 62.27% and 103.88%.

Noise regression remains poor. On the 84 added validation noise samples, mean
absolute percentage errors are 67.03%, 66.27%, 59.97% and 54.47%, respectively
for the four setups above. Type-specific setups improve that subset while
producing larger false noise estimates on other samples. All four pooled noise
MAEs exceed the zero baseline (354.720); all four noise RMSEs exceed the training
mean baseline (954.412). Large noise overpredictions also appear at the retained
5–100 levels. Stronger injection alone has not resolved generalization with the
current six features and training protocol. This is a result of this experiment,
not a conclusion that thermal noise is generally unpredictable.

Machine-readable range-specific errors and image-noise changes are in
`runs/v1-noise-extension-20261026/summary.json`. A fresh rerun of the comparison
from the expanded feature cache is:

```sh
ml/.venv/bin/python -m ml.regression_strategies \
  --from-run ml/runs/20261028T030404318702Z_multi_regression \
  --include-type-specific --jobs 4
```

## Recovery and preservation audit (2026-10-27)

All four CASA workers completed successfully and the canonical indexes were
merged: 1,200 training-noise and 252 validation-noise samples. The driver's
preservation assertion then stopped execution before training. Recovering the
968 old noise samples into worker indexes had called cleanup again, replacing
their `cleanup.audit.json` histories with zero-removal audits. This was a
bookkeeping mutation, not regeneration of those samples.

A complete SHA256/size/mtime check of all 55,845 snapshotted files found exactly
those 968 changed cleanup audits; the other 54,877 files were unchanged,
including all original FITS products, QA, simulation metadata, sample manifests,
main index, and prior ML artifacts. The original cleanup audit contents were
not backed up, so their history cannot be restored from the saved hashes.
The exception is recorded in `preservation_differences.json` and
`preservation_verified.json`; preservation is not claimed for those audit files.

The noise-only generator now validates and reindexes a finalized sample directly,
without calling cleanup when its existing cleanup audit is present. A regression
test checks preservation of the audit's bytes and modification time. Training
was completed with `fit.py`, followed by `validate.py` and `summarize.py`;
generation was not repeated. The original `run.log` retains the failure as
provenance; `regression.log` and `experiment.json` record successful completion.

## Training-sample diagnostic

The saved final models were evaluated directly on their cached training rows, without refitting. All saved validation predictions were reproduced exactly, and model/cache/report input hashes were unchanged. These are in-sample scores, not cross-validation estimates.

Noise-only samples at SNR 1000–4000 (400 training rows; 84 validation rows):

| Setup | Training relative MAE | Validation relative MAE |
| --- | ---: | ---: |
| one_per_target | 62.45% | 67.03% |
| shared_tree | 51.78% | 66.27% |
| one_per_type | 50.27% | 59.97% |
| shared_by_type | 39.73% | 54.47% |

Relative MAE averages each absolute noise-output error divided by its requested injected SNR. This table excludes clean, amp and phase samples. All four models have substantial training error as well as worse validation error. The failure is therefore not confined to unseen sources. These results alone do not distinguish limited model capacity, training choices and ambiguity in the current features.

For the original 5–100 noise levels, training noise MAEs are 382.80, 300.08, 505.53 and 354.44 in the setup order above. On clean training samples, mean predicted noise strengths are 411.32, 328.09, 538.01 and 374.44, although the target is zero.

At true training noise strength 4000, mean predictions are 1246.84, 1892.11, 1861.63 and 2464.37. Thus the models learn some strength dependence but substantially underpredict the strongest training injections and overpredict weak/clean samples.

Reproduce with `PYTHONPATH=. ml/.venv/bin/python runs/v1-noise-extension-20261026/check_training_fit.py`. Detailed per-level errors, prediction quantiles and input hashes: `runs/v1-noise-extension-20261026/training_fit.json`; per-sample predictions: `training_noise_predictions.csv` in the same directory.

## Subsequent noise-only experiment

The [single-target combined6 experiment](noise_only_combined6.md) excludes all
zero-injection and gain samples, and provides feature correlations and an actual
CASA CLEAN/model/residual sequence from zero through 4000.
