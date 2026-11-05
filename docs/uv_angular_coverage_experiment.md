# Angular uv-coverage experiment

Use dataset 1 source `2241+098` and reproduce its fixed extra-thermal-noise realization at SNR_corr=50 with the original seeds, observing setup, point source, baseline noise, and imaging configuration. After both extra and baseline noise are added and weights initialized, progressively flag a working copy of that Measurement Set. For each cut, flag baselines whose `UVW` orientation falls in a growing angular sector anchored at 0°.

Treat the Fourier plane as conjugate-symmetric: use `atan2(v,u) mod 180°`. At requested removal `p`%, mask orientations `[0, 180p/100)` degrees, equivalent to paired opposite sectors totaling `p`% of 360° by angle. Use 0, 10, …, 90%; 100% has no usable data and cannot form an image. Keep original flags and weights; generate the full SNR_corr=50 noisy sample before applying any new flags. Count the actual fraction of initially usable visibility samples removed, since angular span and sample fraction differ.

Image the progressively flagged working copy with the dataset 1 direct imaging algorithm, pinning cell size and CLEAN mask to the 0% row. Compute all residual statistics in the same 0%-defined annulus in image pixels. Report one row per cut: UVW coverage, dirty, CLEAN, residual, residual scaled MAD (robust sigma), residual absolute peak, and 99.5th percentile tail divided by MAD. Use fixed display scales and inferno colorbars across rows. Compare the 0% image and QA against the retained dataset sample, check nested flags and unchanged unflagged data/weights, and record the source, seeds, wedge definition, actual removal fractions, and CASA warnings.

[Completed report](../runs/uv-angular-2241-noise50-final/report.html) and [flag audit](../runs/uv-angular-2241-noise50-final/validation.json).

The completed run independently reproduced the unmasked noisy Measurement Set and confirmed exact equality of DATA, CORRECTED_DATA, WEIGHT, SIGMA and WEIGHT_SPECTRUM with the final 90%-masked set. The report uses pipeline `inferno` colors, individual colorbars, and a wide scrollable layout.

## Sky-only replication (no regular or extra thermal noise)

The separate [noiseless report](../runs/uv-angular-2241-noiseless/report.html)
uses the same source, 0–90% angular cuts, pipeline settings and five-column
layout. Its grid, CLEAN mask and pixel annulus are pinned to the original
noise50 report. Input point-source flux remains 0.0921665 Jy. Finite nominal
sigma0 weights permit imaging but no noise is added to the visibilities.
Scientific notation exposes the tiny reconstruction/numerical residuals;
residual limits are shared across cuts within each report, not across reports.

Reproduce in a fresh output directory:

```sh
bin/run casa scripts/uv_angular_coverage_experiment.py \
  --output runs/uv-angular-2241-noiseless --no-noise
bin/run casa runs/uv-angular-2241-noiseless/validate.py
ml/.venv/bin/python -m scripts.uv_angular_coverage_report \
  runs/uv-angular-2241-noiseless
```

The generation audit checks zero measured thermal perturbation, exact nested
flags and unchanged DATA/noise-weight column hashes across cuts. The retained
independent `validate.py` checks every stored visibility against the noiseless
sky (including flagged samples), all 30 FITS for finiteness, original grids and
measurement masks, and recalculates annulus MAD/peak/tail statistics.

## Noise-only replication (no source)

The [noise-only SNR_corr=50 report](../runs/uv-angular-2241-no-source-noise50-20261104/report.html)
uses the same `2241+098` template and the original extra-noise target of 50.
The ordinary baseline noise is also present. After generating the paired noisy
sample, subtract its predicted point-source visibilities from `DATA` and
`CORRECTED_DATA`. Keep the original flags, weights, UVW, noise realization,
fixed image grid, CLEAN mask and residual measurement annulus. Image the same
0–90% angular cuts and report the same dirty/CLEAN/residual products and
residual scaled MAD, absolute peak and P99.5/MAD statistics.

```sh
bin/run casa scripts/uv_angular_coverage_experiment.py \
  --output runs/uv-angular-2241-no-source-noise50-20261104 --noise-only
ml/.venv/bin/python -m scripts.uv_angular_coverage_report \
  runs/uv-angular-2241-no-source-noise50-20261104
```
