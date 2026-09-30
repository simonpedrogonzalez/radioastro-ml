# Tuned logistic-regression and XGBoost comparison plan

## Goal

Train both multinomial logistic regression and XGBoost on each dataset-v1
feature view, producing six directly comparable models:

| Model | Inputs |
| --- | --- |
| `qa4` | dynamic range (scaled MAD), residual scaled MAD, peak/scaled MAD, p99.5/scaled MAD |
| `parity2` | log even and odd residual energies |
| `combined6` | all six features |

Use the canonical validation metrics and plots already shared by the neural
models. Evaluate increased-noise samples only as held-out controls.

## Dataset and features

- Read train/validation from `dataset.json` with
  `label_criterion="constant_antenna_type"`; the 17 stored variant labels
  collapse to configured classes `(none, amp, phase)` without code changes.
- Read controls separately from `dataset_noise_controls.json`, validation only.
  Never use them for fitting, scaling, cross-validation, model choice, or
  thresholds.
- Require stable fingerprints of both indexes plus their sample manifests; stop
  if generation changes either input while loading.
- Extract all six features once per sample. For residual `R`, use
  `E=(R+J)/2`, `O=(R-J)/2` on the native paired grid, scale both by the residual
  annular MAD `s`, and record `log1p(mean((E/s)^2))` and
  `log1p(mean((O/s)^2))`. Preserve sample/source IDs and label metadata.
- Dataset-v1 inspection confirms the four QA keys and finite square `256×256`
  products on baseline, new low-strength, and increased-noise samples. The
  eight positive target levels require no fixed-level branches.

## Leakage-safe tuning

- Tune each logistic feature set independently with a pipeline
  `StandardScaler -> LogisticRegression(solver="saga", max_iter=10000)`.
- Use five-fold `StratifiedGroupKFold(shuffle=True, random_state=42)` on the
  training partition, grouping by source so variants of one source never cross
  folds.
- Grid: `C=10**[-4..4]`, `l1_ratio={0,1}` (L2/L1), and
  `class_weight={None,balanced}`. Treat convergence warnings as failures.
- Tune each XGBoost feature set on the identical folds with
  `XGBClassifier(objective="multi:softprob", tree_method="hist",
  eval_metric="mlogloss", random_state=42)`. Use balanced training-only sample
  weights. Grid: `n_estimators={100,300}`, `max_depth={2,4}`,
  `learning_rate={0.03,0.1}`, `min_child_weight={1,5}`,
  `subsample={0.8,1}`, `colsample_bytree={0.8,1}`, and
  `reg_lambda={1,10}`.
- Select maximum mean CV macro F1; break exact ties by macro recall, lower F1
  fold SD, then smaller `C`. Refit the selected pipeline on all training data
  and evaluate validation once. Do not tune on validation or noise controls.
- Save every fold score and selected parameters. CV mean±SD describes tuning
  stability; the final deterministic validation result does not receive
  artificial seed error bars.

## Evaluation and false-positive controls for every model

- For each model, pass validation probabilities and explicit class order to
  `evaluate_task`. Report main, binary detection, conditional error
  identification, per-class metrics, strength cohorts, and confusion matrix.
- Pass noise-control probabilities through the existing all-clean branch. For
  every target and overall, report sample count, correct-rejection rate,
  **false-positive rate** (`predicted != none`), and amp/phase prediction counts.
- Add a matched clean reference using the baseline sample from every source
  represented at each noise target; report baseline FPR and
  `excess FPR = noise-control FPR - matched-baseline FPR`.
- Plot FPR versus noise target for all six tabular models with the matched-baseline
  reference. Do not report AUROC/AUPRC for an all-clean control population and
  do not choose a threshold from these controls.
- Make this a shared artifact contract for logistic regression, ResNet-18, and
  DINOv2: every completed run contains `validation` and `noise_controls` results,
  and its prediction table includes every control sample with class
  probabilities. The combined reports show control FPR by target for every
  model; the per-run artifacts retain sample ID, source, target, predicted
  class, and probabilities so each false positive is inspectable.

## Minimal code changes

- `ml/logreg.py`: replace the fixed single-model path with one six-feature
  loader and a compact six-job runner: three tuned logistic pipelines plus
  three tuned XGBoost classifiers, final fits, predictions, and canonical
  result dictionaries.
- `ml/pyproject.toml` / `ml/uv.lock`: add the maintained `xgboost` package; do
  not implement boosting locally.
- `ml/task_evaluation.py`: expose a small clean-control summary reusable for
  increased-noise and matched-baseline cohorts; retain the current FPR
  definition and schema.
- `ml/nn_common.py` and `ml/run_nn_experiments.py`: load validation controls
  through `dataset_noise_controls.json`, transform them with the corresponding
  main-training normalization statistics, run inference after checkpoint
  selection, and save their probabilities/results. Never include controls in a
  training loader or checkpoint selection.
- `ml/report.py`: replace the single-logreg writer with one comparison writer.
  Reuse `_comparison_lines`, `_metric_lines`, `_plot_strength`, and
  `_plot_confusion_matrices`; add shared noise-FPR tables/plots usable by both
  logistic and NN reports.
- `ml/run_nn_experiments.py`: add the shared control summary and per-target FPR
  plot to its incremental report for every completed ResNet/DINO job.
- Remove `ml/logreg_comparison_report.py`; its saved-run/CNN-specific interface
  is obsolete. Keep report helpers model-agnostic so logistic and NN result
  dictionaries can be combined later after cohort-identity validation.
- `ml/tests/test_ml.py`: replace the fixed-four-feature assumptions with focused
  six-feature, grouped-tuning, control, artifact, and report tests.

## Artifacts and report

Write one run directory containing `config.json`, `cv_results.csv`,
`predictions.csv`, `results.json`, the six fitted `.pkl` estimators, and
`report.qmd`/`report.html`. Record both dataset fingerprints, feature order,
fold source IDs, grid, selected parameters, package versions, and cohort counts.
Use the same result/prediction keys as NN runs so later comparisons do not need
model-specific control logic.

The report contains:

1. one six-model table for precision, recall, F1, AUROC, and AUPRC;
2. one row of fixed-scale confusion matrices;
3. the shared `3×5` corruption-strength figure;
4. per-view and per-class tables for every model;
5. selected hyperparameters and CV macro-F1/recall mean±SD;
6. standardized coefficients/feature importance; and
7. the noisy-control FPR table and plot, including matched-baseline excess FPR.

For every model, `predictions.csv` is the detailed audit: filter
`split=noise_controls` and `predicted_label!=0` to inspect every fooled sample,
including its target and all class probabilities.

## Verification

- Analytical parity fixture verifies E/O geometry, scale invariance, and feature
  order; QA-only, parity-only, and combined matrices share identical rows.
- Fold audit proves no source leakage and scalers are fitted inside each fold.
- Small deterministic tuning fixtures exercise both estimators. Saved
  pipelines/classifiers exactly reproduce stored probabilities and
  `results.json`.
- Controls are absent from training/CV and the fixture predictions
  `[none, amp, phase, none]` give FPR `0.5` with no fabricated curve metrics.
- NN and logistic artifact fixtures both contain the identical control result
  schema and sample-level prediction columns; changing control pixels cannot
  affect training loss, selected hyperparameters, or checkpoint selection.
- Report smoke verifies the five-metric comparison, `3×5` strength grid,
  shared-scale confusion matrices, and per-level noise FPR outputs.
- Run only after dataset generation finishes and both fingerprints remain stable.
