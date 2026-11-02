# Why the current regression does not recover extra-noise strength

The evidence supports a mismatch between the intended task of thermal-noise-level
estimation and the current experiment. The training code includes the noise rows
correctly. The defined target is the global norm of a tiny added random component;
most current features measure shape relative to the image's own noise scale.
These results do not establish that thermal-noise level is generally unpredictable.

Audit outputs: [summary JSON](../runs/noise-identifiability-audit/audit.json),
[all 968 noise samples](../runs/noise-identifiability-audit/samples.json),
[168 held-out FITS pairs](../runs/noise-identifiability-audit/image_checks.json),
[63 scaling controls](../runs/noise-identifiability-audit/scale_checks.json),
[diagnostic figure](../runs/noise-identifiability-audit/noise_scale_diagnostic.png).

## Actual target and injected scale

The simulated data are `Y = V + n0 + eta`. The target is
`S = ||eta / sigma0||`, with clean target zero even though `n0` remains present.
For N valid complex samples, the generator records

```
sigma_extra / sigma0 = S / sqrt(2*N)
sigma_total / sigma0 = sqrt(1 + S*S / (2*N))
```

This is a noise-weighted L2 norm across the dataset, not the per-sample noise
standard deviation. With millions of visibilities, the chosen strengths induce
very small changes in the total noise standard deviation:

| Injected S | Median total σ increase, training | Median total σ increase, validation |
| --- | ---: | ---: |
| 5 | 0.000235% | 0.000225% |
| 20 | 0.00376% | 0.00360% |
| 100 | 0.0939% | 0.0900% |

At S=100, held-out increases range from 0.00134% to 0.309%. N ranges from
808,664 to 185,997,418 among validation sources. A common global norm therefore
also produces different per-visibility noise changes between sources.

Equal perturbation norm does not imply equal inferability from image features:
gain errors have a structured response, whereas the added random component
changes the distribution of an existing random background. This is an
interpretation of this setup, not a proof of impossibility for every estimator.

## Checks of generation and data paths

- All 800 training and 168 validation noise targets match simulation metadata;
  reference/extra/total sigma obey the equations above. Extra and baseline seeds
  differ. Recorded residual MAD values match the cached ML feature column.
- All 168 validation dirty and residual images differ from their corresponding
  clean images. They are not reused clean products.
- Recomputing parity from all 168 noise residuals and 21 clean residuals matches
  the cached features. FITS loading preserves physical pixel scale (float32),
  without per-image normalization in the dataset loader.
- The earlier eight-level CASA scratch check measured the added visibility norm
  before and after baseline-noise addition, with maximum relative discrepancy
  3.1e-9, and verified weights. This audit reads those results; it does not rerun
  CASA or claim a fresh payload-level check of all measurement sets.
- Previous training audits reproduced saved predictions and confirmed noise
  rows entered fitting. No missing-training-row or obvious image-path error was
  found by these checks. Original files were hashed and remain unchanged.

## The available features suppress or lack the needed information

The parity features compute even/odd energy divided by the residual's own MAD
squared. Multiplying a residual by any positive constant multiplies numerator
and denominator by the same squared constant. The features therefore preserve
shape and remove global amplitude information.

As a direct control, each of the 21 clean validation residuals was multiplied
in memory by 1, 1.5 and 2. The MAD followed the multiplier, while the largest
change in either extracted parity feature was only 8.7e-8 (roundoff scale).
This is a feature-path check, not a new noise-injection/imaging experiment.

The QA features include an absolute MAD, so the entire combined6 representation
is not amplitude invariant. But it receives neither sigma0, N, a predicted
reference image-noise level, nor a paired baseline image. Reference visibility
sigma alone spans about 0.078–1.143 Jy among validation sources. Inferring the
extra component from an absolute image scale is therefore confounded by the
observing setup and the pre-existing noise.

Even when compared with the same source's clean residual, the measured MAD
changes are small and variable. At S=100 the median change is 0.0478%, the 10th
to 90th percentiles are -0.355% to +0.373%, and 8 of 21 sources decrease.
Adding an independent random realization can lower a finite-sample scale
estimate; deconvolution adds further variation. These observations alone cannot
separate those contributions. Variance estimation has sampling uncertainty;
the [NIST variance-test discussion](https://www.itl.nist.gov/div898/handbook/eda/section3/eda358.htm)
describes the Gaussian independent-sample case. Image pixels are correlated, so
their count should not be substituted directly for independent observations.

An exactly paired baseline subtraction would cancel the shared n0 and answer
a different, easier question. That baseline image is not an input to these models.

## Implication and next experiment

The current evidence supports treating these levels as weak-noise robustness
controls rather than a useful broad test of thermal-noise-level regression.
Changing XGBoost output strategy or removing zero-target rows cannot restore
information removed or barely expressed by the features.

A meaningful next test should generate a small, separate pilot with known total
noise ratios such as 1, 1.5 and 2, independently realized noise, fixed observing
setups and controlled imaging weights. Verify that measured image noise follows
those levels, first with a direct scale estimator. Include a reference noise
scale as an input; normalize against that reference rather than each corrupted
image's own MAD. Then test held-out-source regression.

If a common SNR_corr label is required, retain it as a recorded target but choose
injection strengths from the desired noise ratios:
`S = sqrt(2*N*(kappa*kappa - 1))`. Merely relabeling the existing samples as kappa
does not enlarge their tiny variation. Such a pilot changes the experimental
range and should be explicit; this audit leaves the dataset and target definition
unchanged.

Reproduce the audit and plot:

```sh
ml/.venv/bin/python -m runs.noise-identifiability-audit.audit
ml/.venv/bin/python -m runs.noise-identifiability-audit.plot
```

## Subsequent experiment decision (2026-10-26)

The user chose to retain the eight existing extra-noise SNR_corr levels and add
1000, 2000, 3000 and 4000 for training and validation. The
[extension protocol](noise_strength_extension.md) preserves the norm definition,
original reference sigma and existing samples. The audit results above concern
the original 5–100 range; they do not establish performance at the new strengths.

