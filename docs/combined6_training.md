# Combined6: output strategy and training population

The results below describe the original eight noise levels, 5–100. The
[noise-strength extension](noise_strength_extension.md) adds 1000, 2000, 3000
and 4000 and repeats these four setups with the same source folds and reporting.
The [completed expanded-range report](../ml/runs/20261028T030413268146Z_combined6_training/report.html)
uses 2,893 training and 608 validation rows. The extension document records its
results, checks and cleanup-audit preservation exception.

[Four-setup report](../ml/runs/20261024T014248110295Z_combined6_training/report.html)
and [validation audit](../ml/runs/20261024T014248110295Z_combined6_training/validation.json).

All setups use combined6 features, the same cached 2,493 training rows, three
saved source CV folds, and 524 validation rows from 21 unseen sources.

| Setup | Tree strategy | Training contributions per output |
| --- | --- | --- |
| one_per_target | Separate scalar-output trees | All 2,493 rows |
| shared_tree | Shared trees with three outputs | All 2,493 rows |
| one_per_type | Three separately fitted scalar regressors | Clean + own fault type |
| shared_by_type | Shared trees with masked squared-error loss | Clean + own fault type |

Filtered counts are 895 amplitude, 898 phase and 900 noise rows, each with 100
clean examples. The separate regressors physically filter their training rows.
The shared variant uses the union of rows for shared splits and histogram bins;
each output's loss includes only clean rows and its own fault type. Other fault
rows have zero gradient and Hessian for that output. Shared splits can still
affect all outputs. This is the per-output-loss interpretation of type-specific
training within a shared tree.

For eligibility mask M, the custom objective is
`L = sum(M * (prediction - truth)^2) / 2`, with gradient
`M * (prediction - truth)` and diagonal Hessian `M`. Initial predictions are each
output's mean among eligible training rows. This uses XGBoost's
[custom-objective interface](https://xgboost.readthedocs.io/en/stable/tutorials/advanced_custom_obj.html).
Saved models retain their intercepts and need no mask or true fault type at inference.

Each setup selects one common parameter configuration for all outputs from the
previous four candidates (depth 2/4, rounds 100/300, learning rate 0.05).
CV scores use every output and every CV validation row; standard deviations
come from all rows in that training fold. Thus only fitting changes. The final
validation population, clipping, metrics, plots and train_mean/no_error baselines
are unchanged. Reserved test data are unused.

All setups select 100 rounds; one_per_type selects depth 2, the others depth 4.

| Pooled validation MAE | one_per_target | shared_tree | one_per_type | shared_by_type |
| --- | ---: | ---: | ---: | ---: |
| Amp | 2.737 | 2.918 | 6.686 | 9.693 |
| Phase | 4.017 | 3.672 | 10.538 | 11.019 |
| Noise | 11.691 | 11.511 | 37.924 | 40.489 |
| Total | 14.205 | 13.928 | 47.196 | 53.362 |
| Mixture | 0.161 | 0.164 | 0.370 | 0.379 |

Type-specific fitting improves amplitude and phase MAE when evaluated only on
their own fault samples, but greatly increases predictions on unrelated fault
types. For example, one_per_type predicts summed noise strength equal to 1.70
times the injected phase strength on phase-fault samples, versus 0.148 for the
all-row counterpart. Removing those zero-target examples worsens pooled results.
Noise-only MAE also worsens. These observations describe this fixed split/seed.

Fifteen tests pass, including finite-difference gradient/Hessian checks,
invariance to excluded labels, and equivalence of the unmasked custom objective
to built-in squared error. The audit verifies all 2,096 prediction rows, absolute
and relative metrics, cross-talk, root Hessian training counts, 20 plot axes,
48 product links, and refreshing all six model files. Both original setups and
their CV results reproduce exactly. All prior artifacts are preserved.

```sh
ml/.venv/bin/python -m ml.regression_strategies \
  --from-run ml/runs/20261024T013136572864Z_combined6_strategies \
  --include-type-specific --jobs 4
```
