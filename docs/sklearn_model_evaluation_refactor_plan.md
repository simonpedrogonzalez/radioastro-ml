# Scikit-learn model-evaluation refactor plan

## Goal

Implement the metrics specified in the Obsidian note `Notes/Model Evaluation.md`
with one small, probability-aware, task-agnostic evaluator, then make every ML
report consume its result. Use scikit-learn for the metric definitions instead
of maintaining parallel manual reductions in the evaluator and report modules.

`ml/evaluate.py` must be agnostic to label values, label meanings, and the number
of classes. Its only classification assumptions are a one-dimensional truth
array, an explicit ordered label sequence of length `K`, and an `N x K` score
array whose columns correspond to that sequence. Dataset-specific views and
cohorts are assembled outside this module.

The primary outputs are precision, recall, F1, AUROC, and AUPRC for:

1. the complete multiclass problem (currently `none`, `amp`, `phase`);
2. binary detection (`no error` versus `some error`); and
3. amplitude-versus-phase identification among true corruptions that were
   correctly detected as corrupted.

Also retain the full multiclass confusion matrix and report all five metrics per
class. This follows the first two references in the note: multiclass metrics are
one-versus-rest, macro averages give each class equal weight, and AUROC/AUPRC
must be calculated from continuous scores rather than hard predictions.

## Invariants

- The generic evaluator accepts any hashable class labels and any `K >= 2`.
  Labels need not be numeric, consecutive, sorted, or radio-astronomy-specific.
- Callers must supply the ordered `labels` sequence matching the score columns.
  The current task adapter obtains the order and semantic roles from dataset
  metadata/configuration; `ml/evaluate.py` must not contain `0`, `1`, `2`,
  `none`, `amp`, `phase`, `clean`, or `corrupted` as class assumptions.
- Do not change training data, source partitions, model fitting, seeds, or saved
  model inputs as part of this refactor.
- Require an `N x K` probability array in the explicitly supplied label order.
  Derive hard predictions once by mapping `argmax` column indices through
  `labels`; do not accept a second, possibly inconsistent, hard-prediction array.
- Corruption targets and sample kinds remain evaluation metadata, never model
  features.
- Increased-noise controls remain held-out controls bearing the configured clean
  label in `dataset_noise_controls.json`. Never mix them into the ordinary main
  aggregate or training population.
- Compare models only when sample IDs, truths, partitions, and target strengths
  are identical.

## Generic metric definition

Use these scikit-learn functions in `ml/evaluate.py`:

```python
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
```

For each supplied class label, form its one-versus-rest truth vector directly
with `y_true == label`. This avoids special-casing scikit-learn's two-class
`label_binarize` output and works identically for arbitrary `K`.

The generic result contains:

- `macro`: the unweighted mean of every defined per-class metric;
- `per_class`: precision, recall, F1, AUROC, AUPRC, and support keyed by the
  supplied labels; and
- `confusion_matrix`: calculated in the supplied label order.

Precision, recall, and F1 come from
`precision_recall_fscore_support(..., labels=labels, average=None,
zero_division=0)`. Per-class AUROC and AUPRC use `roc_auc_score` and
`average_precision_score` on each one-versus-rest truth vector and corresponding
score column. Macro AUROC/AUPRC are the arithmetic means of the defined
per-class values. If a class has no positive or no negative examples, its curve
metric is `null`; the macro mean ignores undefined values and is `null` only
when none are defined. This policy is uniform for every number and meaning of
classes.

Do not retain accuracy as a primary metric. Balanced accuracy duplicates macro
recall for single-label multiclass classification, so remove it from the result.
Where CNN checkpoint selection currently uses balanced accuracy, rename that
quantity to `macro_recall`; the numerical selection rule remains unchanged.

## Dataset-specific evaluation views

The following mapping is required by the current experiment, but it does **not**
belong in `ml/evaluate.py`. Put it in a thin task adapter (for example,
`ml/task_evaluation.py`) that receives the class roles from the dataset config
and calls the generic evaluator.

| View | Included rows | Hard prediction | Continuous score | Averaging |
| --- | --- | --- | --- | --- |
| Main | baseline plus gain samples | `argmax` mapped through configured labels | all class probabilities | macro over all configured labels |
| Detection | same rows | binary `argmax(p_clean, sum(p_error))` | sum of configured non-clean score columns | binary, corrupted is positive |
| Error identification | true non-clean rows detected as corrupted by the binary view | configured error-label subset | renormalized configured error-label columns | macro over configured error labels |

“Correctly detected” in the third row means a binary true positive: the sample
is truly corrupted and the collapsed binary view predicts corrupted. It does not mean
that the amplitude/phase subtype was already correct. Record the number and
fraction of true corruptions reaching this conditional evaluation so a model
cannot hide missed detections behind a high subtype score.

The adapter identifies the configured clean role and error roles; it must not
assume particular numeric label IDs. The current dataset happens to assign
`0=none`, `1=amp`, and `2=phase`, but that mapping is input data rather than a
metric definition.

### Main multiclass metrics

Pass all configured class labels and probability columns directly to the generic
evaluator. No three-class branch is permitted.

### Binary detection metrics

Using the configured clean label, map truth and prediction to clean versus
non-clean and sum the score columns for every configured non-clean label. Call
the same generic evaluator with explicit labels `[clean, corrupted]`; expose the
`corrupted` per-class values as detection precision, recall, F1, AUROC, and
AUPRC. Also record sample count, corrupted count, and corrupted prevalence so
the AUPRC baseline is visible.

### Conditional error-identification metrics

Select rows whose truth and prediction are both non-clean. Renormalize the score
columns belonging to the configured error labels, then call the same generic
evaluator with that label subset. This supports two or more error types without
changing the metric module. Return `eligible_corruptions`,
`detected_corruptions`, and `detection_coverage` with these metrics.

If the conditional subset is empty, or a requested AUROC/AUPRC has no positive
or no negative examples, return `null` for that metric and let reports show an
em dash. Do not catch every `ValueError` from scikit-learn; check these known
degenerate cases explicitly so real input errors still fail.

## Corruption-strength cohorts

For every positive target strength, evaluate a matched complete-class cohort
made from:

- every baseline sample at target `0`; and
- samples for every configured error label at that one target strength (amp and
  phase in the current dataset).

This gives main and detection metrics all required classes at every x position.
Evaluating only the positive-strength rows would make detection AUROC/AUPRC
undefined because every truth would be corrupted. The baseline is reused for
calculation but is not duplicated in the dataset or prediction table.

Return:

```text
overall
├── main
├── detection
└── error_identification
by_corruption_snr_target
└── <positive level>
    ├── main
    ├── detection
    └── error_identification
```

Each view exposes the same ordered metric keys:
`precision`, `recall`, `f1`, `auroc`, `auprc`. `main` additionally contains
`per_class` and `confusion_matrix`.

Evaluate increased-noise controls through a separate function/result branch.
For each noise target, report sample count, clean recall/correct-rejection rate,
false-positive rate, and predicted counts keyed by the configured labels. AUROC
and AUPRC are not defined for an all-clean control set and must not be
fabricated.

## Minimal implementation

### 1. Replace `ml/evaluate.py` with a generic module under 200 lines

`ml/evaluate.py` must contain no more than **200 physical lines**, including
imports, type definitions, docstrings, comments, and blank lines. Target roughly
100-150 lines; add a test that fails if the module exceeds 200 lines.

Keep only small input validation, one private curve helper, and one public
function:

```python
def _curve_metrics(binary_truth, score) -> tuple[float | None, float | None]: ...
def evaluate_classification(y_true, probabilities, labels) -> dict: ...
```

Validate only classification inputs: one-dimensional nonempty truth, unique
labels with `K >= 2`, truth values contained in labels, exact `N x K` score
shape, finite nonnegative scores, and row sums close to one. Derive predictions
once and return JSON-serializable Python values.

Do not pass target strength, sample kind, corruption roles, dataloaders, model
objects, paths, or report concerns into this module. Do not add binary or
three-class branches; one-versus-rest per-class computation covers every `K`.

Do not add metric classes, registries, callbacks, pandas dataframes, or a
general experiment framework.

### 2. Add the experiment adapter

Add `ml/task_evaluation.py` for the current experiment's main, detection,
conditional error-identification, corruption-strength cohort, and noise-control
assembly. It reads label roles from the dataset/run configuration and delegates
every actual classification metric calculation to `evaluate_classification`.
The adapter may know the experiment semantics; the evaluator may not.

### 3. Pass probabilities from every model

- `ml/logreg.py`: pass `FeatureSet.y`, `predict_proba`, label order, and task
  metadata to the task adapter; derive the CSV prediction column from the same
  probability array.
- `ml/cnn.py`: make `score()` pass the split labels, probabilities, and metadata
  directly. Use `overall.main.recall` for checkpoint selection in place of the
  duplicate balanced-accuracy field.
- Continue saving every class probability in `predictions.csv`. Write the
  canonical evaluation result to `results.json` for every run type.

### 4. Centralize report rendering

Add small shared helpers in `ml/report.py`; they render metrics but never
recalculate them:

- an overall model-comparison table with rows=model and columns=the five main
  metrics; bold every tied maximum in a column;
- a per-class table containing all five metrics and support;
- a `3 x 5` strength figure: rows are main/detection/error-identification,
  columns are precision/recall/F1/AUROC/AUPRC, x is target strength, and each
  model is one consistently colored line;
- one row of confusion matrices, one column per model, labeled
  `true`/`predicted`, with visible integer annotations, `vmin=0`, a shared
  `vmax=number of cohort samples`, and one shared colorbar.

Use `sklearn.metrics.ConfusionMatrixDisplay` for confusion matrices and
Matplotlib for composition. Keep a single legend for the strength grid. Missing
conditional values remain gaps, not zeros.

Single-run logistic and CNN reports use the same helpers with one model. The
comparison and ablation reports pass multiple named result dictionaries. The
report modules must validate cohort identity before applying a shared confusion
scale or bolding cross-model maxima.

## Code to remove or replace

This is a deliberate replacement, not a compatibility layer.

| File | Remove | Replacement |
| --- | --- | --- |
| `ml/evaluate.py` | Current `_summary()`; the hard-label `evaluate(dataloader, predictions)` API; dataloader/metadata handling; manual NumPy accuracy/FPR/detection/type reductions; the old `by_corruption_snr_target`; fallback sample-kind inference; every fixed class ID/count and corruption-specific branch | Generic `evaluate_classification(y_true, probabilities, labels)` module, at most 200 physical lines |
| `ml/task_evaluation.py` (new) | N/A | Dataset-configured class-role mapping and assembly of main, detection, conditional error-identification, strength, and noise-control views; all metric math delegates to `ml/evaluate.py` |
| `ml/logreg.py` | `validation_by_id`/`test_by_id`, hard-prediction calls to `evaluate`, and console references to accuracy/balanced accuracy | Evaluate the existing probability arrays once and print the five main metrics |
| `ml/cnn.py` | Import of private `_summary`, `score()`'s probability-to-hard-label mapping, and `train_balanced_accuracy`/`val_balanced_accuracy` naming | Probability-aware `score`; use scikit-learn macro recall for the hard-label training diagnostic and `macro_recall` fields with identical checkpoint behavior |
| `ml/report.py` | Old `_metric_lines()` schema, `_conditional_type()`, `_severity_lines()`, `_plot_severity()`, and the ASCII confusion matrix | Shared five-metric tables, strength grid, and heatmap renderer |
| `ml/cnn_report.py` | Imports of private `_summary`/old plot helpers, balanced-accuracy learning-curve naming, and the per-source mini-summary that recomputes incomplete-class macro metrics | Shared canonical report helpers and macro-recall learning history; retain prediction rows for later source analysis |
| `ml/logreg_comparison_report.py` | Reconstruction of a fake dataloader, hard-label-only evaluation, nested `cell()` count reconstruction, two-panel severity plot, and hand-built old metric table | Parse the stored probabilities, call the canonical evaluator, and render shared comparison outputs |
| `ml/ablations.py` | `_conditional_type` dependency, old `balanced_accuracy` result-key checks, the balanced-accuracy/macro-F1 comparison table, and the bespoke two-panel severity plot | Canonical `overall.main.recall`, shared five-metric comparison table, and `3 x 5` strength grid; keep the vectorized source-bootstrap analysis under its macro-recall name |
| `ml/tests/test_ml.py` | Assertions for accuracy, balanced accuracy, `corruption_type_accuracy`, `type_accuracy_among_detected`, and the hard-prediction evaluator API | Probability-based exact fixtures for all three views and per-class scores |
| `ml/tests/test_cnn.py` | Assertions or fixtures using old balanced-accuracy result/history keys | Macro-recall names and canonical result paths |

Do not keep aliases for the removed result keys. Historical run directories
remain immutable artifacts; regenerate their reports from `predictions.csv`
only when probability columns for every configured label and cohort metadata
are present.

## Focused validation

1. Generic evaluator fixtures cover two, three, and four classes, including
   string and nonconsecutive integer labels in a caller-specified order, and
   verify all outputs against direct scikit-learn calls.
2. A source/line-count test asserts `ml/evaluate.py` has at most 200 physical
   lines and contains no task class names or metadata/cohort dependencies.
3. A hand-checkable current-task fixture verifies detection mapping and the
   conditional error-label subset through the adapter.
4. A two-level fixture proves each strength cohort contains the clean baseline
   plus only the selected amp/phase level.
5. A model that detects no corruptions yields `null` conditional
   error-identification metrics without warnings or invented zeros.
6. Invalid probability shapes, labels, sums, nonfinite values, and mixed
   increased-noise/main samples fail clearly.
7. D predictions `[none, amp, phase, none]` still produce 50% false positives,
   with no detection recall or amp/phase-identification claim.
8. A report smoke test checks the `3 x 5` panel count, identical model colors,
   bold maxima, labeled confusion axes, integer cell annotations, one colorbar,
   and common `[0, N]` color limits.
9. Re-evaluate one saved logistic run and one CNN run from their prediction
   tables; both must use the same result schema without re-running a model.

## Not in this pass

- Threshold selection, probability calibration, Brier score, and reliability
  diagrams.
- Source bootstrap confidence intervals or significance tests; retain the
  existing source-group bootstrap separately.
- Monotonicity and first-detected-strength analyses.
- Retraining, hyperparameter changes, or test-set evaluation.
- AUROC/AUPRC for the all-clean increased-noise-control set.

The prediction table retains enough probabilities and metadata to add those
analyses later without changing this metric core.

## Definition of done

- `ml/evaluate.py` is label-meaning-agnostic, supports arbitrary `K >= 2`, and
  is no more than 200 physical lines.
- Every model path supplies probabilities and an explicit label order to the
  one generic evaluator, directly or through the task adapter.
- Every requested overall, per-class, detection, error-identification, and
  strength-resolved metric is produced by scikit-learn with explicit averaging.
- All ML reports use the shared result schema and requested comparison layouts.
- The old hard-label evaluator, duplicate manual report reductions, fallback
  result keys, and ASCII confusion matrices are removed.
- Focused unit tests and one saved-run report smoke pass in the `ml` uv
  environment.
