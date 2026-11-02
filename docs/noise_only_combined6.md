# Noise-only combined6 regression

[Experiment report](../ml/runs/20261028T032841757256Z_noise_only_combined6/report.html)
includes true-versus-predicted plots, six feature-versus-strength panels,
correlation tables and a CLEAN/model/residual image sequence at all 13 levels.

## Protocol

One scalar XGBoost regressor predicts measured extra-noise SNR_corr using the
unchanged combined6 features. Only positive extra-noise variants enter fitting,
cross-validation and evaluation: 1,200 training samples from 100 sources and
252 held-out validation samples from 21 different sources. Each source supplies
12 levels: 5, 10, 15, 20, 30, 40, 50, 100, 1000, 2000, 3000 and 4000.
Zero-injection, amplitude and phase samples are excluded. The reserved test
partition remains unused; “held-out” here means the existing validation split.
The target measures the extra injected perturbation, not absolute thermal sigma.

The feature cache comes from the completed expanded-range four-setup experiment.
The same three source CV folds and four candidates are retained: depth 2/4,
100/300 rounds, learning rate 0.05. Selection now scores the single noise output
on noise-only CV rows. Squared-error loss, histogram trees, seed 42 and zero
clipping are unchanged. Selected depth 2 and 100 rounds. Features are supplied
in their original units; logarithmic plot axes are display choices only.

## Results

| Population | Model MAE | Model RMSE | Training-mean MAE | Training-mean RMSE |
| --- | ---: | ---: | ---: | ---: |
| Training | 782.93 | 1025.26 | 1096.11 | 1329.99 |
| Held-out validation | 1173.10 | 1443.67 | 1096.11 | 1329.99 |

At SNR 1000–4000, mean absolute percentage errors are 48.76% training and 58.32%
validation. At 5–100, MAEs are 550.83 and 1056.58. The single-output model learns
some training signal but does not outperform a constant training-mean predictor
on unseen sources under this protocol and parameter grid.

Held-out Pearson correlations with injected SNR are −0.253 for dynamic range,
+0.359 for residual MAD, +0.045 for peak/MAD, −0.074 for P99.5/MAD, +0.033 for
even parity and −0.075 for odd parity. Residual MAD has the clearest association.
Its median within-source Spearman correlation is 0.699, versus pooled Spearman
0.239. Stronger within-source trends and overlapping pooled distributions are
consistent with source-to-source variation obscuring a common mapping. These
are descriptive correlations with repeated observations per source; weak linear
or rank correlation does not prove absence of nonlinear information.

## Image sequence and verification

Source 2241+098 was chosen before fitting because it has the median validation
total-noise sigma ratio at SNR4000 (1.9705). The sequence includes the baseline
and all 12 positive levels. Zero means no extra injection; baseline thermal noise
remains. Full 256×256 fields use fixed color limits across levels within each
column. CLEAN/model use asinh stretches; residual uses a linear stretch.
CLEAN and residual are Jy/beam; the actual CASA CLEAN component model is Jy/pixel.

Model images were originally discarded. Reproducing this source in an isolated
run directory recovered the actual model images. Of 52 reproduced CLEAN,
residual, dirty and PSF arrays, 48 were pixel-identical; the remaining differences
were at float32 rounding scale (maximum difference/image peak 1.58e-7).
All passed an absolute tolerance of eight float32 epsilons times each original
image peak. All 130 original sample files retained their hashes and mtimes.
The report displays original CLEAN/residual products with these reproduced models.

Independent checks verified all 1,452 predictions, filtered cache rows, source
separation, CV selection, metrics and correlations. The first tree's root cover
is 1,200, confirming the final fitting population. No original dataset, fitted
models or earlier reports were modified. Artifacts are under
`runs/noise-only-combined6-20261027/`; the numerical audit is in the report's
`validation.json`.

Reproduce from the repository root with `ml/.venv/bin/python` and `PYTHONPATH=.`:
run `experiment.py` in that artifact directory, then `report.py` and `validate.py`.
The CASA `image_sequence.py` has already produced the reusable image sequence;
its recorded checks and models are retained beside those scripts. Its scratch
reproductions are separate from the canonical dataset.

## Subsequent feature comparison

The [sigma_NA comparison](sigma_na_feature.md) adds the predicted natural-image RMS and a residual-MAD-to-sigma_NA ratio. It compares combined6 and combined8 on all 2,893 training and 608 validation samples.
