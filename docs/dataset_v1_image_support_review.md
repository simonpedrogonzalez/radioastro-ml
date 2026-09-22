# Dataset v1 image-support review

## Finding

Dataset v1 contains uniformly shaped `256 x 256` FITS arrays, but their valid
image support differs:

- 106 inspected sources have full-square support.
- 25 have circular support, with invalid pixels outside the circle stored as
  zero.
- All 25 circular cases are VLA D-configuration sources; the inspected A, B,
  and C configurations are square.
- Dirty, clean, and residual share the circular boundary, while the PSF remains
  full-square.

The support is a source-level property. Across the 1,171 completed samples
reviewed, the metric region, selected pixel count, and selected beam area are
constant across all available variants of the same source. It therefore is not
a direct corruption-label shortcut, although it is a strong source and array-
configuration nuisance.

## NaNs and metric calculation

The stored QA metrics are not calculated from the zero-filled FITS pixels.
Metrics are calculated from the CASA images before FITS export. The metric
selection intersects the requested annulus with finite pixels and CASA's
validity mask, and the summarizer filters non-finite values again. Only after
that calculation does FITS export replace remaining NaNs or infinities with
zero.

Therefore the circular zeros visible in retained FITS do not enter the recorded
RMS, scaled MAD, percentile, peak, or dynamic-range metrics.

There is a fallback that uses finite pixels when the CASA validity mask selects
none of the requested region. It still cannot turn NaNs into zeros, but its
mask semantics should be reviewed before a future dataset version.

## Annulus comparability

The annulus is defined from three synthesized-beam major FWHM to one beam before
the square image border. Its outer radius does not account for circular valid or
primary-beam support.

In 18 of the 25 circular sources, the actual metric population is consequently
the nominal annulus clipped by the valid circle. Those sources retain between
45.6% and 99.7% of the nominal annulus pixels. The remaining seven circular
sources contain the full requested annulus inside valid support.

RMS and scaled MAD remain meaningful for the selected pixels, but cross-source
comparability is weakened. Extreme and percentile statistics also depend on the
number of pixels or independent synthesized beams searched.

## Model implications

The logistic-regression feature set currently includes `n_pixels` and
`area_synthesized_beams`. These expose support geometry and array configuration
and should be treated as audit covariates rather than corruption-classification
features.

Pixel models can easily detect the hard zero boundary, especially because the
PSF remains full-square. Source-group splitting prevents simple same-source
label leakage, but a model can still learn configuration-dependent nuisance
features and generalize poorly when support distributions differ.

## Recommended follow-up

1. Keep `n_pixels` and `area_synthesized_beams` for diagnostics, but remove them
   from classifier inputs.
2. Retain an explicit validity or primary-beam mask in future dataset products.
3. Normalize image channels using valid pixels only.
4. Use mask-aware image processing or a common support guaranteed valid for all
   sources before training pixel models.
5. Define future metric regions using both the desired beam-scaled annulus and
   valid/PB support, and always record the actual selected beam area.

The current dataset remains useful for paired within-source analysis and an
initial scalar baseline, but support geometry must be audited before interpreting
image-model performance.
