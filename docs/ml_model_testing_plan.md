# Minimal ML baseline plan

## Question

Can the existing image QA metrics classify a sample as `none` (`0`), `amp`
(`1`), or `phase` (`2`), and how does accuracy change with injected corruption
strength?

The first model is one multinomial logistic regression. `SNR_corr` and
`SNR_corr_target` are ground truth used only for labels and evaluation; they
must never enter the feature vector.

## Deliberate simplifications

- One optional `criterion` string and one labeling function, not a label
  registry or class hierarchy. `None` keeps the dataset's existing stored
  label for backward compatibility.
- Put feature extraction in `ml/logreg.py`; a separate feature package is not
  useful yet.
- Use the existing dataset/dataloader even though the scalar model ignores the
  image tensor. Metadata-only loading is a later performance optimization.
- Reject absent, invalid, or nonfinite QA metrics. Do not add an imputer for a
  problem not present in the inspected dataset.
- Use fixed `C=1.0` and `class_weight="balanced"`. Do not add a search grid,
  cross-validation, or custom sample weighting to the first baseline.
- Do not add a random seed option: `lbfgs` and the nonshuffled loaders make this
  pipeline deterministic.
- Write three artifacts (`model.pkl`, `predictions.csv`, and `report.md`), not a
  reporting framework or duplicate JSON result file.
- Test project logic only. Do not test `uv`, sklearn fitting, pickle, imports,
  report wording, or other library behavior.

## Files

```text
scripts/preprocessing/labels.py       # new
scripts/preprocessing/dataset.py      # minimal wiring change
ml/
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
├── __init__.py
├── logreg.py                         # features, train, and main
├── evaluate.py
├── report.py
└── tests/test_ml.py
```

Add label/dataset assertions to the existing preprocessing test file rather
than creating another preprocessing test module.

## 1. Labeling in the dataset package

Add one result type and function:

```python
@dataclass(frozen=True)
class AssignedLabel:
    id: int
    name: str
    corruption_snr: float | None
    corruption_snr_target: float | None

def assign_label(corruption_reports, criterion):
    ...
```

Implement exactly one derived criterion, `constant_antenna_type`, with these
rules:

| Reports | Valid content | Label | Strength metadata |
| --- | --- | --- | --- |
| none | — | `0`, `none` | measured `0.0`, target `0.0` |
| one | type `amp`; positive finite measured and target SNR | `1`, `amp` | both SNR values |
| one | type `phase`; positive finite measured and target SNR | `2`, `phase` | both SNR values |

Fail on multiple reports, unsupported types, or missing/nonfinite/nonpositive
SNR. Read the measured value from `metrics.SNR_corr`, the target from
`solution.SNR_corr_target`, and the type from `solution.corruption_type`. Do not
guess how future compound corruptions should be collapsed.

Add this constructor option to `FitsSimulationDataset`:

```python
label_criterion: str | None = None
```

When it is `None`, construct the result from the existing `label_id` and
`label_name` already read from `sample.json`; this preserves current callers
without presenting that stored provenance as a labeling criterion. Otherwise
pass the loaded corruption reports and criterion name to `assign_label`.

The dataset already reads corruption reports for metadata. Parse them once,
and return:

```python
{
    "label": assigned.id,
    "label_name": assigned.name,
    "label_metadata": {
        "corruption_snr": assigned.corruption_snr,
        "corruption_snr_target": assigned.corruption_snr_target,
    },
}
```

Collate the two new fields as lists. Apply `target_transform` to the assigned
ID. Do not rewrite `sample.json`; the original nine-class label remains useful
provenance.

## 2. Local Python environment

Create a normal local `uv` project under `ml/` using Python 3.12 and these
direct dependencies only:

```text
numpy
torch
astropy
scikit-learn
```

Commit `.python-version`, `pyproject.toml`, and `uv.lock`. Ignore `ml/.venv/`,
caches, `ml/runs/`, and serialized models. Run ML commands from the repository
root with the ML interpreter, never CASA Python:

```bash
uv run --project ml python -m ml.logreg --dataset /path/to/dataset.json
```

## 3. Features and training

Keep one explicit, ordered allowlist in `ml/logreg.py`:

```text
metrics.clean_peak_jy_per_beam
metrics.dynamic_range_rms
metrics.dynamic_range_scaled_mad
metrics.residual.n_pixels
metrics.residual.area_synthesized_beams
metrics.residual.rms_jy_per_beam
metrics.residual.scaled_mad_jy_per_beam
metrics.residual.residual_abs_peak_jy_per_beam
metrics.residual.residual_min_jy_per_beam
metrics.residual.residual_max_jy_per_beam
metrics.residual.peak_over_scaled_mad
metrics.residual.p99_over_scaled_mad
metrics.residual.p99_5_over_scaled_mad
metrics.residual.rms_over_scaled_mad
```

One helper consumes a nonshuffled dataloader and returns `X`, `y`, sample IDs,
source IDs, and label metadata. Require every value and corresponding
`metric_validity` flag to exist, be valid, and be finite. An error must name the
sample and metric.

The fixed training function is:

```python
def train(X, y):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
        ),
    ).fit(X, y)
```

Treat convergence warnings as errors. Fit the fixed train partition once and
evaluate validation. Add grouped training cross-validation only when there is
an actual hyperparameter or model choice. Never cross-validate on validation.
Evaluate test only when `--evaluate-test` is explicitly supplied after the
baseline has been frozen.

## 4. Evaluation

`ml/evaluate.py` exposes:

```python
def evaluate(dataloader, predictions: Mapping[str, int]) -> dict:
    ...
```

Predictions are keyed by sample ID so ordering cannot silently mismatch. The
function reports:

- sample count;
- accuracy and balanced accuracy;
- macro precision, recall, and F1;
- per-class precision, recall, F1, and support; and
- a fixed 3-by-3 confusion matrix for classes `[0, 1, 2]`.

Compute the same summary for each target-SNR group `0`, `10`, `30`, `50`, and
`100`. Also report clean false-positive rate for group `0`; for positive groups
report corruption-detection recall (`prediction != 0`) and corruption-type
accuracy (`prediction == truth`). These two values are the easiest severity
diagnostics because positive groups contain no true clean samples.

Retain measured SNR in the prediction rows for later continuous analysis, but
do not add bins, regression, plots, confidence intervals, or monotonicity tests
yet.

## 5. Reporting and command

`ml/report.py` creates a new, non-overwriting run directory:

```text
ml/runs/<timestamp>_metric_logreg/
├── model.pkl
├── predictions.csv
└── report.md
```

`predictions.csv` contains sample/source ID, split, truth, prediction, three
class probabilities, measured SNR, and target SNR.

`report.md` briefly records the dataset path, package versions, class map,
fixed model settings, ordered feature list, split sample/source counts, overall
validation metrics, confusion matrix, severity table, and whether test was
evaluated. End with an explicit leakage statement that corruption metrics were
not model inputs. Use only the Python standard library plus `pickle`; do not
add pandas, plotting, templating, or experiment-tracking dependencies.

The `main` in `ml/logreg.py` accepts only:

```text
--dataset
--output (default: ml/runs)
--evaluate-test
```

It constructs train/validation datasets with
`label_criterion="constant_antenna_type"`, extracts features, trains, predicts,
calls `evaluate`, optionally evaluates test, then calls `report`. Print only the
run path and validation accuracy, balanced accuracy, and macro F1.

The checked-in `dataset_v1_20260916T110148` uses the old three-product sample
schema. The real smoke/full run requires regenerated schema-v2 samples that
contain `psf.fits.gz`; do not fabricate PSFs or reinterpret old manifests.

## Tests that matter

Keep four focused checks:

1. A table-driven labeling test covers clean, amp, phase, both SNR outputs, and
   rejection of unsupported/malformed/multiple corruption reports.
2. Extend the existing synthetic FITS dataset test to check that `None` keeps
   the stored sample label and that `constant_antenna_type` reaches both an item
   and its collated batch.
3. One synthetic batch verifies exact feature order and that an invalid or
   nonfinite value fails with sample/metric context.
4. One hand-calculated prediction set spanning clean and two positive strength
   levels verifies the confusion matrix, macro metrics, clean false positives,
   detection recall, and type accuracy.

Then run the command once on a small complete set of source groups and check
that it converges, emits the three artifacts, has exactly one prediction per
evaluated sample, and includes overall plus severity results. This is a smoke
run, not another unit-test framework.

Do **not** add tests for dependency installation/imports, sklearn's optimizer,
standard scaling, pickle round trips, deterministic repetition, report prose,
CLI help, or existing partition logic. Those either test dependencies or are
already covered elsewhere.

## Later models, only if the baseline motivates them

1. Gradient boosting on the same QA metrics tests nonlinear boundaries.
2. Beam-normalized annular/Fourier residual features test ring/spoke structure;
   use logistic regression first, then an RBF SVM if needed.
3. A small residual-only CNN tests whether pixels add information; only then
   compare dirty/clean/residual/PSF channels.
4. Fuse QA metrics with an image embedding only if both contribute separately.

All later models must reuse the same labels, source partitions, and evaluator.

## Definition of done

One documented `uv run` command trains the fixed metric-only model outside
CASA and writes a concise validation report. Labels come from corruption
reports, corruption strength is available for grouped evaluation but absent
from `X`, existing source partitions are unchanged, and test evaluation
requires an explicit flag.
