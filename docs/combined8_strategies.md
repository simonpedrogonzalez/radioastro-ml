# Combined8 multi-output strategy comparison

[Report](../ml/runs/20261028T065537442783Z_combined8_strategies/report.html) · [Configuration](../ml/runs/20261028T065537442783Z_combined8_strategies/config.json) · [Predictions](../ml/runs/20261028T065537442783Z_combined8_strategies/predictions.csv)

The models predict measured amp, phase, and extra-noise SNR_corr from the eight features defined in [sigma_na_feature.md](sigma_na_feature.md). All use the same 2,893 training and 608 validation samples, 100/21 disjoint sources, reserved test unused, three training-source CV folds, seed 42, and the same depth 2/4 by 100/300-round grid at learning rate 0.05. CV scores average absolute error divided by the training-fold target standard deviation across all three outputs and validation rows.

| Model | XGBoost output strategy | Training rows |
| --- | --- | --- |
| `1tree_all_data` | `multi_output_tree`: one shared tree structure per boosting round for all three outputs | All 2,893 for every output |
| `per_target_all_data` | `one_output_per_tree`: one tree per output per round | All 2,893 for every output |
| `per_target_only_non0_target` | Three independent scalar regressors | Only positive rows for each output: amp 795, phase 798, noise 1,200. Clean and other zero-target rows are excluded. |
| `2stages` | `multi_output_tree` on 14 features | All 2,893 rows |

`2stages` takes the original combined8 plus six clipped predictions: three from `per_target_all_data`, then three from `per_target_only_non0_target`. Its training features use source-held-out predictions, so each row is predicted by first-stage models fitted without that row's source. The final first-stage models are fitted on all training rows for validation inference. Stage tuning uses an outer source fold; within each outer training fold, its first-stage training features come from inner source-held-out fits. First-stage hyperparameters were selected on the full training partition before stage tuning, so the nested stage CV is useful for model selection but is not an unbiased final performance estimate. The separate 21-source validation partition supplies the final comparison.

| Model | Validation Total MAE | Amp MAE | Phase MAE | Noise MAE |
| --- | ---: | ---: | ---: | ---: |
| `1tree_all_data` | 238.07 | 3.55 | 3.96 | 236.37 |
| `per_target_all_data` | 251.05 | 2.51 | 3.43 | 250.49 |
| `per_target_only_non0_target` | 1072.41 | 13.93 | 15.57 | 1058.62 |
| `2stages` | **202.38** | 3.75 | 3.82 | **200.41** |

Total MAE measures the absolute error of the sum of the three outputs. The positive-only regressors extrapolate poorly to zero-target samples, especially for noise. The two-stage model has the lowest Total and noise MAE in this single validation split. See the report for all earlier metrics, strength slices, cross-talk, hard samples, and true-versus-predicted panels. The solid black line in each panel joins the median true and predicted values at each injected strength for that target.

Reproduce with:

```sh
ml/.venv/bin/python -m ml.combined8_strategies \
  --from-run ml/runs/20261028T060641804439Z_sigma_na_comparison \
  --jobs 4
```

Saved stage training and validation matrices are `stage_train_features.npz` and `stage_validation_features.npz`. An independent audit in `runs/sigma-theory-comparison/validate_strategies.py` checked their 14 columns, all four reloaded model outputs, 2,432 validation predictions, row counts, and target-specific positive-row counts.

A [filtered rerun](combined8_below1000_strategies.md) excludes injected noise levels 1000–4000 from training and evaluation.
