# Combined8 strategies below injected noise SNR 1000

[Comparison report](../ml/runs/20261028T175609834379Z_combined8_below1000_strategies/report.html) · [Configuration](../ml/runs/20261028T175609834379Z_combined8_below1000_strategies/config.json) · [Predictions](../ml/runs/20261028T175609834379Z_combined8_below1000_strategies/predictions.csv)

This rerun uses the four strategies and 14-feature two-stage construction documented in [combined8_strategies.md](combined8_strategies.md). It removes every noise-kind sample with injected SNR_corr **1000 or higher** from both training and validation before fitting, cross-validation, and evaluation. Clean, amplitude, phase, and noise-kind samples at levels 5–100 remain. This removes 400 training and 84 validation rows, leaving 2,493 training rows from 100 sources and 524 validation rows from 21 disjoint sources. The reserved test partition is unused.

The original combined8 values and measured targets for retained samples are unchanged. The same three source folds, grid, seed 42, and evaluation definitions apply, with every model refitted on the filtered population. The first-stage held-out predictions used to train `2stages` are regenerated on the filtered rows. The report includes the same tables, strength slices, cross-talk, hard samples, and median lines in true-versus-predicted plots.

| Model | Validation Total MAE | Amp MAE | Phase MAE | Noise MAE |
| --- | ---: | ---: | ---: | ---: |
| `1tree_all_data` | 13.738 | 2.909 | 3.421 | 11.184 |
| `per_target_all_data` | 13.918 | 2.776 | 3.756 | 10.866 |
| `per_target_only_non0_target` | 64.034 | 12.386 | 13.774 | 45.366 |
| `2stages` | **13.254** | **2.765** | **3.366** | **10.822** |

The positive-only setup still makes large false predictions on zero-target samples. On this validation split, `2stages` has the lowest Total MAE; its lead over `1tree_all_data` is modest. These scores use a different target range and row population from the full-range comparison, so raw MAEs across the two reports are not an isolated measure of what removing the high-noise rows changes.

Reproduce with:

```sh
ml/.venv/bin/python -m ml.combined8_strategies \
  --from-run ml/runs/20261028T060641804439Z_sigma_na_comparison \
  --exclude-noise-at-or-above 1000 \
  --jobs 4
```

The independent audit `runs/sigma-theory-comparison/validate_strategies_below1000.py` checked that both splits exclude all noise-kind rows at or above 1000, the 14 stage columns, target-specific training counts (795 amp, 798 phase, 800 noise), and all 2,096 validation predictions against the reloaded models.
