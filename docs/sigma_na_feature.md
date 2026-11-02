# sigma_NA features and combined6 versus combined8

[Comparison report](../ml/runs/20261028T060641804439Z_sigma_na_comparison/report.html) · [validation](../ml/runs/20261028T060641804439Z_sigma_na_comparison/validation.json)

The additional simple metrics are:

- `sigma_NA`: predicted RMS of an untapered naturally weighted Stokes-I image, in Jy/beam. The thermal precursor calculated and recorded it as `predicted_image_rms_jy_per_beam` for each source. Its calculation uses the theoretical baseline visibility noise and counts unflagged parallel-hand samples in the original Measurement Set: `sigma_simple / sqrt(N_usable)`. The implementation is `scripts.simulation.natural_image_rms_from_simplenoise`.
- `sigma_sNA_units`: the sample's residual scaled MAD (Jy/beam) divided by `sigma_NA`. It is dimensionless. It varies across corruption variants while `sigma_NA` stays constant within each source.

`SIMPLE_FEATURES` now contains the original four QA features plus these two features (`simple6`). `COMBINED8_FEATURES` adds them to the unchanged combined6 columns. The old `sigma_theory` feature was per-visibility `sigma_simple`; it is removed from current ML feature extraction. Historical combined7 artifacts remain as records of that earlier, superseded experiment.

Both scalar XGBoost models predict extra-noise SNR_corr. Both train on all 2,893 training rows and evaluate on all 608 validation rows, using the same source split, three source CV folds, candidate grid, and seed 42. Clean, amplitude, and phase samples have noise target zero. Each model selects its own hyperparameters from the same grid. The reserved test set is unused.

| Model | Validation MAE | Validation RMSE |
| --- | ---: | ---: |
| combined6 | 643.84 | 1026.81 |
| combined8 | 250.49 | 587.41 |

The new features lower validation MAE by 61.09% on this fixed split. See the report for true versus predicted plots, each feature against injected noise, noise ranges, and the original image sequence. The independent audit verifies all 3,501 `sigma_NA` values against the precursor, the two new feature columns, unchanged original six columns and targets, and 7,002 saved model predictions. The 47 relevant tests pass.

Reproduce with:

```sh
ml/.venv/bin/python -m ml.noise_feature_comparison \
  --from-run ml/runs/20261028T030413268146Z_combined6_training \
  --image-run ml/runs/20261028T032841757256Z_noise_only_combined6 \
  --jobs 4
```
