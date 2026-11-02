# Combined6: separate versus shared XGBoost trees

A subsequent [four-setup experiment](combined6_training.md) also compares
training each output on clean samples and its own fault type.

[Experiment report](../ml/runs/20261024T013136572864Z_combined6_strategies/report.html)
and [validation audit](../ml/runs/20261024T013136572864Z_combined6_strategies/validation.json).

Both models use the same six features, 2,493 training samples from 100 sources,
and 524 validation samples from 21 unseen sources as the previous experiment.
The cached feature/target files are copied byte for byte. Reserved test data are unused.

The comparison changes XGBoost's `multi_strategy`:

- `one_per_target`: `one_output_per_tree`, separate scalar-output trees for amp,
  phase and noise.
- `shared_tree`: `multi_output_tree`, shared splits and three values per leaf.

These modes follow the [XGBoost multi-output documentation](https://xgboost.readthedocs.io/en/stable/tutorials/multioutput.html).
The installed version is 3.4.1. Both use squared-error training, histogram trees,
seed 42, and the exact three source-grouped CV folds saved in the previous run.
Each mode is tuned independently over depths 2/4, rounds 100/300, and learning
rate 0.05. Selection minimizes mean absolute error divided by training-fold
target standard deviation, averaged over outputs and folds. Targets are fitted
in their original units; scaling applies to CV scoring only.

Both selected depth 4 and 100 rounds. Thus one-per-target has 300 scalar-output
trees; shared-tree has 100 vector-output trees. This compares the modes under
the same parameter grid and boosting rounds, rather than equal parameter counts.
Best CV scores were 0.273602 and 0.268486, respectively.

| Validation metric | One per target | Shared tree |
| --- | ---: | ---: |
| Amp MAE | 2.737 | 2.918 |
| Phase MAE | 4.017 | 3.672 |
| Noise MAE | 11.691 | 11.511 |
| Total MAE | 14.205 | 13.928 |
| Total RMSE | 21.509 | 21.016 |
| Mixture MAE | 0.161 | 0.164 |
| Mixture RMSE | 0.319 | 0.318 |

Shared trees reduce Total MAE by 1.95% on this validation set. Phase and noise
errors improve, while amplitude errors and mixture MAE increase. Noise estimation
remains weak. These are descriptive results from one fixed split and seed.

The same report generator produces the prior sections, metrics, four plot types,
hard samples and training coverage, with train_mean/no_error baselines and no
displayed R². Validation checks cover all 1,048 prediction rows, absolute and
relative errors, model structures, 20 labeled-zero plot axes and 48 product links.
The one-per-target CV scores, predictions and metrics exactly reproduce the prior
combined6 run. All source artifacts remain unchanged. Twelve regression tests pass,
including fit/save/reload checks for both output strategies.

Reproduce into a fresh run directory:

```sh
ml/.venv/bin/python -m ml.regression_strategies \
  --from-run ml/runs/20261022T203038478771Z_multi_regression_report \
  --jobs 4
```
