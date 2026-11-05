# Quick recovery-damage regression experiment: Dataset 1 / combined8

Implementation protocol for the original damage-regression request on existing Dataset 1 before generating Dataset v2. No new simulations, mixtures, sampling budgets, presence prediction, L_b, total-error prediction, L_all or classification metrics. Calculating recovery errors and injection-presence metadata is needed to build damage labels, not an additional prediction task.

Current run: [HTML report](../ml/runs/20261104T030653756125Z_recovery_damage_shared_independent_stack/report.html), [saved metrics](../ml/runs/20261104T030653756125Z_recovery_damage_shared_independent_stack/metrics.json), [label provenance](../ml/runs/20261104T030653756125Z_recovery_damage_shared_independent_stack/labels.json), [stacking audit](../ml/runs/20261104T030653756125Z_recovery_damage_shared_independent_stack/stacking_validation.json). Entry point: `python -m ml.recovery_damage`. Only train and validation are used: 3,501 rows in total. Three 2341-351 noise images have a pixel-scale difference of about 0.01–0.03% from their ordinary control. Each image's own WCS and beam was used.

Validation L_d was lowest for XGBoost + XGBoost (5.3244e-6). These are comparisons on the archived-product approximation, with validation used for model selection. Both binders now take the eight combined8 features plus three predictions from independent per-class XGBoost and three from shared multiregression XGBoost. Source-held-out predictions are used on training rows; full-training first-stage models predict validation rows. The stacking audit rebuilt all six training prediction columns exactly. The earlier train/validation run using active-only conditional experts is superseded. The report begins with one train/validation comparison table, bolds the minimum of each column, uses LaTeX loss symbols, and shows a contiguous, globally aligned diagonal grid with tick labels along its bottom and left edges. Mean-by-level plots show ±1 sample standard deviation. The thermal-noise levels extend through 4000 in both splits; at 4000, mean validation damage is 0.00204. The reported test analysis from the initial run is superseded and excluded from the current run.

Scientific source: [V3 setup](</Users/u1528314/Documents/Obsidian Vault/Notes/RadioastroML V3 Setup - Partially Supervised Error Attribution.md>). Its proposed simulation setup and additional tasks are deferred.

## 1. Reuse the existing data and features

Use `collect/experiments/dataset_v1_20260921_all_psf/`, its canonical splits and existing extra-noise indices. Reuse verified input features from the unfiltered `ml/runs/20261028T065537442783Z_combined8_strategies/` run. Join new labels by sample ID. Do not reuse its SNR targets as damage labels or use the below-1000 filtered population.

Preserve `COMBINED8_FEATURES` and feature ordering in `ml/logreg.py`, including both nominal-noise features. Preserve Dataset 1's archived corruption/noise recipes and source splits. Its extra-noise SNR parameter is not the later pilot's total-noise multiplier.

The populations are train 2,893 rows / 100 sources and validation 608 / 21. Both have 12 noise levels, including 1000, 2000, 3000 and 4000. Do not load or evaluate the canonical test split at this stage.

## 2. Build damage labels from saved images

Use class order `(amp, phase, noise)`. For each sample compute:

```text
y = e_TM,2 = ||C - T||_2 / ||T||_2
d_k = y_corrupted - y_matched_control   for the injected type
other d_k = 0
ordinary controls: d = (0, 0, 0)
```

C is the restored fitted model, and T is the true point source convolved with the matching peak-one restoring beam. Evaluate over valid image pixels. Use stored source flux/position and FITS beam/WCS with the existing recovery geometry/metric helpers.

Dataset 1 retains restored/residual images but not raw CLEAN model exports. For this quick experiment use `C = float64(clean) - float64(residual)` under ordinary restoration. Record this as the archived-product approximation: float64 arithmetic does not recover precision already lost in float32 FITS. Do not add raw-model replay or support-reconstruction pipelines.

Before labeling, check that matched controls share the original sky and base-noise realization using simulation metadata. Check image units, restoration convention, valid support and comparable field/beam geometry. Construct each T with its recorded geometry. Small native geometry changes should be documented, not silently described as a fixed-beam pilot. If retained evidence cannot establish pairing or a usable image domain, report affected cases as unlabelable rather than assigning zeros or launching simulations. The resulting target describes recovery under the archived Dataset 1 imaging policy.

Extra-noise cases use the ordinary-noise control, not a noiseless reference. Recognize them from simulation metadata even when their old classification label is `not_corrupted`. Retain signed damage, including zero or negative values for active injections.

Save a small label sidecar keyed by sample ID: source/split, d, o, pi, injection type/presence, y, matched control ID/error, input hashes and label validity. Cache each image's recovery error once and reuse controls. Do not change original manifests.

## 3. Loss for fitting, selection and reporting

With N rows and K=3 outputs:

```text
L_dk = sum over observed rows [(pred_d_nk - d_nk)^2 / pi_nk] / N
L_d  = mean_k L_dk
```

For current isolated data, every valid label is known, including absent-type zeros, so `o=pi=1` everywhere. L_d is ordinary MSE over the entire N×3 matrix. Do not use presence as the observation mask or normalize by active/observed counts or weight sums.

Keep entrywise o/pi support for future mixtures. Unknown targets are masked before subtraction. One uniformly chosen active removal among m types gives pi=1/m, while dense labels give pi=1. Absent types always have o=pi=1. No division by m is needed for controls. Missing current products are labeling failures, not intentional sparse labeling.

Implement matching NumPy and Torch losses. N is minibatch size during Torch optimization and full split size for final scores. Sum numerators and counts across batches before division. Report L_d and all three L_dk on train and validation. L_d is squared error, while sqrt(L_d) is an optional RMS-error diagnostic.

For XGBoost reuse `masked_squared_error()` in `ml/regression_training.py`, replacing its type-specific mask with w=o/pi. Use gradient `w*(prediction-target)` and Hessian `w`, zero for unknown cells. This is the constant-scaled objective `(NK/2)*L_d`, retaining the existing sum-loss regularization convention. Score and select with exact normalized L_d. Avoid dividing tree derivatives by NK while leaving default split/regularization parameters unchanged.

## 4. Compare exactly five models

| Model | Implementation |
| --- | --- |
| MLP | 8 inputs → 64 ReLU → 64 ReLU → 3 linear outputs. Train-only input standardization, Torch L_d |
| Shared XGBoost | `hist`, `multi_output_tree`, three signed outputs |
| Independent XGBoost | Three scalar regressors trained on all observed labels, including absent-type zeros |
| XGBoost + XGBoost | Existing two-stage design, shared-output XGBoost binder |
| XGBoost + MLP | Identical first-stage features, 14 → 64 ReLU → 64 ReLU → 3 linear binder |

Shared trees means vector-output boosted trees, not one boosting round. Preserve signed targets/predictions. Remove old nonnegative clipping, SNR target transforms and scaled-MAE selection from this experiment's path.

The current two-stage contract uses 14 binder inputs: combined8 plus three predictions from the independent per-class XGBoost models and three from the shared multiregression XGBoost model. Both are among the initial models in this comparison. This supersedes the older `ml/combined8_strategies.py` binder inputs, whose second expert was a type-specific positive-only fit. Final binders train on all rows using full L_d. Presence is never an inference feature or gate.

Reuse the source-held-out OOF stacking helpers. Each binder-training row's expert predictions must come from fits excluding its source. Both binders share the same saved expert matrices. Fit final experts on all training sources for validation features.

Keep the existing small XGBoost grid and seed. For a quick comparison, use validation L_d for candidate selection and MLP early stopping. Fix first-stage settings before making OOF features, avoiding nested expert tuning. Use one declared MLP configuration, e.g. Adam, learning rate 1e−3, batch size 64, at most 300 epochs, patience 20. Fit scalers only on training inputs. Do not refit on validation for the reported split comparison.

## 5. Report and minimal changes

Reuse report styling, plotting and serialization from `ml/regression_report.py` and `ml/combined8_strategies.py`. Required outputs:

- One train/validation comparison table with L_d, L_d_amp, L_d_phase, L_d_noise, bold column minima; then validation and train sections in that order.
- Existing true-versus-predicted diagonal plots, five model rows × three outputs for each split. Show signed values, identity/zero lines and all zero-target points. Use one global x/y range across the contiguous grid, existing injection-type colors and shapes, light grid lines, and tick labels only along the bottom and left edges.
- Mean-by-injection-level curves with shaded ±1 sample standard deviation across sources. Injection level remains grouping metadata. Include the true noise-damage scale through 4000.
- Saved split predictions, labels/o/pi, configuration, model/scaler files and label provenance. Report-only reruns must work without fitting. Label binder train loss as measured on its OOF expert-input matrix.

Small patch sequence:

1. Add a label-building helper using saved products and existing FITS/recovery utilities. Write the sidecar and join it to the verified combined8 cache.
2. Add the damage loss/custom objective and a small tabular MLP loop. Adapt existing regression/stacking helpers through narrow callbacks where needed, preserving older experiment defaults.
3. Add a thin damage-comparison entrypoint and reuse report components. No new simulation machinery or generic training framework.
4. Check dense/sparse loss arithmetic, signed/zero labels, source-isolated stacking, and loss recomputation from saved predictions. Run a small existing-data end-to-end case before the full comparison.

Keep d/o/pi arrays and task-specific scoring reusable for Dataset v2. The only future-facing implementation now is the missing-label loss contract. Additional heads, losses, classification reports and mixture generation belong to the later experiment.
