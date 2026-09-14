# Beam-region image metrics refactoring plan

## Objective

Replace the two partially overlapping image-statistics implementations with one
small imaging-metrics API that:

- selects a full image, central disk, annulus, or exterior region using nullable
  minimum and maximum radii in synthesized-beam units;
- calculates every residual statistic from the same selected pixels;
- is called automatically by both direct and VLA Pipeline imaging;
- is reusable by simulation validation without keeping a second MAD/RMS
  implementation;
- keeps the CLEAN mask and the QA measurement region as independent inputs;
- can draw the exact metric region on an image without duplicating its geometry
  calculation;
- records enough region provenance to make comparisons reproducible.

This document is a refactoring plan only. It does not change the current
experiment or its completed report.

## Decisions resolved

1. Report both conventional `peak / off-source RMS` dynamic range and robust
   `peak / off-source scaled MAD`; use the robust value for simulation matching.
2. Remove `RMSMetrics` rather than retaining an alias, because there is no
   production caller to preserve.
3. Include the exact VLA Pipeline background RMS in the QA schema as a
   specialized estimator, without pretending it is an ordinary beam-region
   statistic.
4. Keep CLEAN masking and QA-region selection as independent arguments. Neither
   implementation may infer or modify the other.
5. Let the imaging plotter display the metric region with an unfilled dashed
   white boundary and black halo, distinct from inferno, the lime-green beam,
   and the solid cyan CLEAN-mask contour.

## Current implementation audit

### Imaging package

`scripts/imaging/qa.py` currently owns `calculate_residual_metrics()`.
It loads the clean and residual CASA images as two-dimensional arrays and
calculates these values over every finite residual pixel:

- scaled MAD;
- residual minimum, maximum, and absolute peak;
- residual peak/scaled-MAD;
- residual p99/scaled-MAD and p99.5/scaled-MAD;
- dynamic range, defined as the global clean-image peak divided by the
  full-image residual scaled MAD.

The implementation is concise, but the pixel region is fixed to the whole
image. It does not report ordinary RMS or the selected pixel count. It also
loads pixel values without explicitly combining the CASA internal pixel mask
with the finite-value mask.

`scripts/imaging/imaging.py::_finalize_result()` calls this function for both
direct imaging and VLA Pipeline imaging, so this is the correct integration
point for an imaging-wide region choice.

The image geometry needed for a beam-scaled selection already exists:

- direct imaging records the measured beam and cell in
  `ResolvedImagingConfig.grid`;
- VLA Pipeline imaging records the same information in
  `effective_imaging_parameters.measured_geometry`;
- the CASA image itself also contains the restoring beam and coordinate
  increments, which allows the metric API to work independently of an
  `ImagingResult`.

### Simulation package

`scripts/simulation/rms_metrics.py` contains a second statistics path:

- `measure_pb_region()` selects pixels using a primary-beam response interval
  and returns ordinary RMS, scaled MAD, and pixel count;
- `vla_pipeline_annulus_rms()` reproduces the Pipeline-specific PB annulus,
  clean-mask exclusion, Chauvenet clipping, axis aggregation, and fallback
  behavior.

`measure_pb_region()` and imaging QA duplicate image validation, pixel
selection, and MAD/RMS concepts but return different models. The exact Pipeline
function is not an ordinary region statistic: its clipping and aggregation are
part of a regression oracle and must remain behaviorally unchanged.

### Existing public and serialized contracts

- `scripts.imaging.ResidualMetrics` is a flat record used by QA text, QA JSON,
  reports, experiments, and tests.
- QA schema version 1 stores that record directly as `metrics`.
- `scripts.simulation.RMSMetrics`, `measure_pb_region()`, and
  `vla_pipeline_annulus_rms()` are exported publicly.
- The extracted thermal comparison currently uses the full-image dynamic range
  both as its simulation target and in its plots.
- The existing completed experiment contains schema-version-1 QA files and
  must remain renderable.

An exact repository-wide search found no production caller that imports or
constructs `RMSMetrics`; only its defining module, package export tests,
integration tests, and documentation use the name. Therefore the refactor will
remove `RMSMetrics` outright and migrate every in-repository caller. A
deprecated alias would add maintenance without protecting a real caller.

The serialized QA contract still requires an explicit schema migration rather
than silently changing what the current flat metric names mean.

## Proposed package boundary

Image-derived statistics belong in `scripts/imaging`, regardless of whether the
image came from real or simulated visibilities. Add one canonical implementation
module:

```text
scripts/imaging/metrics.py
```

It will own CASA plane loading, region-mask construction, and all ordinary
residual-statistics functions. The small immutable records stay in
`scripts/imaging/models.py`, preventing a circular import between the existing
`Beam` model and the functions that consume it. Imaging QA will consume the
implementation directly. Simulation validation will import or re-export it
instead of implementing a second calculator.

Keep the exact Pipeline annulus algorithm as a clearly marked specialized
adapter. It may share CASA path/unit/shape validation, but its Chauvenet result
must not be replaced with the ordinary un-clipped metric engine.

After callers are migrated, `scripts/simulation/rms_metrics.py` should contain
only compatibility imports, or be removed if no external code needs the old
path. It must not retain a second implementation.

## Small public API

### Region selection

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class BeamRegion:
    min_radius_beams: float | None = None
    max_radius_beams: float | None = None
```

The four useful selections require no modes or subclasses:

```python
BeamRegion()                              # all valid pixels
BeamRegion(max_radius_beams=3.0)         # central disk
BeamRegion(3.0, 12.0)                    # annulus
BeamRegion(min_radius_beams=3.0)         # outside the central disk
```

Validation rules:

- supplied bounds must be finite and non-negative;
- when both are supplied, `max_radius_beams` must be greater than
  `min_radius_beams`;
- the lower bound is inclusive and the upper bound is exclusive;
- an empty selection is an error that names the requested bounds and available
  image radius.

Radius will mean angular distance from the image centre divided by the
restoring-beam major-axis FWHM. This circular convention deliberately matches
the existing `central_circle_mask()`, whose six-beam diameter ends at radius
three beam-major FWHM. It avoids introducing an elliptical/position-angle
definition that would not match the mask being excluded.

The primitive must have no hidden edge margin. `max_radius_beams=None` means
all available pixels outside the lower bound. A caller that needs an edge
margin supplies an explicit upper bound and the resolved bounds are recorded.

### Shared metric records

Split residual-distribution statistics from the clean-image peak:

```python
@dataclass(frozen=True)
class RegionMetrics:
    n_pixels: int
    area_synthesized_beams: float
    rms_jy_per_beam: float
    scaled_mad_jy_per_beam: float
    residual_abs_peak_jy_per_beam: float
    residual_min_jy_per_beam: float
    residual_max_jy_per_beam: float
    peak_over_scaled_mad: float
    p99_over_scaled_mad: float
    p99_5_over_scaled_mad: float
    rms_over_scaled_mad: float


@dataclass(frozen=True)
class ImageMetrics:
    region: BeamRegion
    clean_peak_jy_per_beam: float
    residual: RegionMetrics
    dynamic_range_rms: float
    dynamic_range_scaled_mad: float
```

`area_synthesized_beams` is the selected angular area divided by the Gaussian
restoring-beam area. It is important provenance because an extreme such as the
residual peak depends on how many independent beam areas were searched.

There will be one ordinary arithmetic implementation:

```python
def summarize_residual_pixels(values, *, pixel_area_arcsec2, beam) -> RegionMetrics:
    ...
```

It calculates:

```text
scaled MAD = 1.4826 * median(abs(x - median(x)))
RMS        = sqrt(mean(x**2))
```

and derives every normalized residual statistic using that same scaled MAD.
This function contains no CASA calls and is directly unit-testable.

### Image-level entry point

```python
def measure_image_metrics(
    clean_image: str | Path,
    residual_image: str | Path,
    *,
    region: BeamRegion = BeamRegion(),
) -> ImageMetrics:
    ...
```

The function loads the first Stokes/channel plane, combines finite pixels with
the CASA internal mask, constructs the beam-radius selection, and calls the
single residual summarizer. The selection applies to every residual statistic.

The metric region applies only to residual/noise measurements. The clean-image
peak remains global and is deliberately not restricted to the noise region.

### Plot overlay contract

The plotter should accept the same immutable `BeamRegion`; it must not accept a
second pair of raw radii that could diverge from the measured selection:

```python
def casa_image_to_png(
    ...,
    metric_region: BeamRegion | None = None,
) -> None:
    ...


def write_individual_plots(
    ...,
    metric_region: BeamRegion | None = None,
) -> tuple[Path, Path, Path]:
    ...
```

`write_individual_plots()` will show the metric overlay on `residual.png`, where
the statistics are measured. It will not add the overlay to dirty or clean
images by default. A direct call to `casa_image_to_png()` may show it on another
image when explicitly requested.

The plotted boundary must come from the same beam-radius mask builder used by
`measure_image_metrics()`. This guarantees identical image centre, cell-size,
beam-major convention, nullable-bound behavior, and pixel orientation.

Boundary behavior follows `BeamRegion` exactly:

- `BeamRegion()` draws nothing because the complete valid plane is selected;
- a central disk draws its maximum-radius circle;
- an exterior region draws its minimum-radius circle;
- an annulus draws both circles;
- a bound outside the displayed frame is simply not drawn.

Do not shade or fill the selected area because that would alter perception of
the inferno image. Keep the existing beam marker lime green and the CLEAN-mask
contour solid cyan. Draw metric boundaries as dashed pure white lines with a
slightly wider black underlay/halo. The paired black-and-white stroke remains
visible against both dark and bright inferno values and is distinguishable from
the green and cyan annotations. If a metric boundary coincides with the CLEAN
mask, the cyan solid line remains visible through the gaps in the white dashed
line.

Add one compact legend entry labeled `metric region`; do not add separate
legend entries for the inner and outer boundaries. Omit the legend when the
full-image selection produces no boundary.

### Dynamic-range definition

The standard radio-image definition is not an invalid mixture. The
[NRAO Synthesis Imaging Workshop](https://science.nrao.edu/science/meetings/2018/16th-synthesis-imaging-workshop/talks/Wilner_Imaging.pdf)
defines image dynamic range as peak image brightness divided by RMS measured in
a region without emission, and
[CASAdocs](https://casadocs.readthedocs.io/en/v6.2.1/notebooks/image_visualization.html)
likewise recommends comparing the image maximum with RMS from a signal-free
region. The numerator and denominator answer different parts of the definition:
signal strength and off-source noise.

For a restored CLEAN image, using the restored-image peak and the final
residual in a source-free region is appropriate: outside modeled emission, the
restored image contains the residual term, and both products have the same
Jy/beam units. The refactor will use the maximum positive brightness—not the
largest absolute pixel—as the conventional Stokes-I source peak.

Because this project explicitly wants a robust estimator, report two names
rather than calling both simply `dynamic_range`:

```text
dynamic_range_rms = global clean-image maximum
                    / selected residual RMS

dynamic_range_scaled_mad = global clean-image maximum
                           / selected residual scaled MAD
```

`dynamic_range_rms` is the conventional quantity. `dynamic_range_scaled_mad`
is its robust counterpart and is the appropriate matching target for this
experiment. Naming both explicitly avoids presenting a MAD-based estimate as
the conventional RMS definition.

### PB-region compatibility

Move `measure_pb_region()` into the imaging metrics module so it builds a
boolean PB selection and
then calls the same `summarize_residual_pixels()` function. It will return the
richer `RegionMetrics` record; existing callers that access only RMS, scaled
MAD, and pixel count continue to work.

Remove `RMSMetrics` and update every caller to `RegionMetrics`. Remove the
simulation-package exports of all image-metric types and functions; callers
must import them from `scripts.imaging`.

`vla_pipeline_annulus_rms()` can and should be represented in the QA schema,
but it cannot be treated as another `BeamRegion`/`RegionMetrics` calculation:

- its region is defined by primary-beam response, not synthesized-beam radius;
- its PB bounds may be replaced by the Pipeline's image-edge/5% heuristic;
- it first excludes the CLEAN mask, then retries without that exclusion when
  too few pixels remain;
- it runs CASA's iterative Chauvenet algorithm with `maxiter=5`;
- it aggregates selected axes and returns the median of per-plane RMS values.

Those choices are the definition of the compatibility value. Running the
ordinary un-clipped MAD/RMS summarizer on the same pixels would produce a
different statistic. Move this function to the imaging metrics module, preserve
its exact algorithm, and store its result under a clearly specialized QA field
such as:

```json
{
  "pipeline_background": {
    "rms_jy_per_beam": 0.00024,
    "region": "adaptive primary-beam annulus",
    "algorithm": "CASA Chauvenet, maxiter=5"
  }
}
```

It shares the package and serialized report with ordinary metrics, but not the
ordinary estimator implementation.

## Imaging integration

Give both imaging entry points the same optional keyword:

```python
image_ms(..., metric_region=BeamRegion())
image_ms_VLA_pipe(..., metric_region=BeamRegion())
```

Pass the selection through `_finalize_result()` to both
`measure_image_metrics()` and `write_individual_plots()`. The exact same object
therefore controls the calculation and overlay. This guarantees that metrics
are created with every image and prevents experiment scripts from duplicating
QA calculations or plot geometry.

The default empty `BeamRegion()` preserves current full-image behavior for
ordinary callers. Experiments opt into an annulus explicitly.

`QAReport.metrics` becomes `ImageMetrics`, and QA schema version increases from
1 to 2. The JSON structure should be nested rather than encode region names in
field names:

```json
{
  "schema_version": 2,
  "metrics": {
    "region": {
      "min_radius_beams": 3.0,
      "max_radius_beams": null
    },
    "clean_peak_jy_per_beam": 1.2,
    "dynamic_range_rms": 13333.3,
    "dynamic_range_scaled_mad": 17910.4,
    "residual": {
      "n_pixels": 58000,
      "area_synthesized_beams": 1200.0,
      "rms_jy_per_beam": 0.00009,
      "scaled_mad_jy_per_beam": 0.000067
    }
  }
}
```

Update `metric_units`, metric validity, and QA text from the dataclass layout;
do not maintain a separate hand-written list of the same fields in multiple
packages.

## Region policy for the thermal comparison

The general API should not hard-code this experiment's choice. The experiment
should define one named constant and pass it to both original and simulated
imaging calls.

The CLEAN mask and metric region must remain separate settings. The metrics
implementation must never read `mask_nbeams`, infer a metric boundary from the
CLEAN mask, or change the CLEAN mask when a metric region changes. Overlap is
allowed if explicitly requested.

The thermal experiment should therefore declare both choices independently:

```python
IMAGING_CONFIG = DefaultImagingConfig  # independently contains mask_nbeams=6
METRIC_REGION = BeamRegion(min_radius_beams=3.0)
```

The current CLEAN mask has a six-beam diameter, so its boundary happens to be
radius three in the proposed convention. Choosing the same value as the metric
region's lower radius is an explicit experiment policy, not an API coupling.
The caller can instead choose a central region, an overlapping region, a gap,
or no CLEAN mask at all. `min=3.0, max=None` uses every valid pixel outside that
chosen radius and avoids inventing an upper radius that some images cannot
support. If edge diagnostics show a bias, the experiment can adopt an explicit
maximum after checking corpus geometry.

A read-only audit of the current 132 completed original/simulation pairs found
that the nearest image edge spans:

```text
minimum:  4.86 beam-major radii
5th pct: 10.96 beam-major radii
median:  21.25 beam-major radii
maximum: 30.79 beam-major radii
```

Consequently, a fixed annulus beginning at radius six would be empty or clipped
for some samples. If identical finite bounds across the entire corpus are more
important than using all off-source pixels, approximately `3.0 <= r < 4.5`
is the largest plausible common interval and must be validated for sufficient
independent beam area before adoption.

For a new matched-S/N experiment, use the selected-region dynamic range as the
target:

```text
target S/N = original dynamic_range_scaled_mad
           = original global positive clean peak
             / original selected-region residual scaled MAD
```

The original and simulation report must use the same `BeamRegion`. All
residual histograms—scaled MAD, peak/MAD, p99/MAD, and RMS/MAD—must come from
that region. The report should display the bounds and selected beam area once
per sample.

## Backward compatibility and report migration

Do not rewrite existing schema-version-1 QA JSON files. Report readers should
use one compatibility accessor:

- schema 1: read the existing flat `metrics` object and label it `full image`;
- schema 2: read `metrics.dynamic_range_rms`,
  `metrics.dynamic_range_scaled_mad`, and residual values from
  `metrics.residual`, and display the recorded region.

This keeps the completed current report renderable. It does not make its old
full-image measurements equivalent to new annulus measurements, so the two
must not be combined in one population histogram without recomputation.

Changing the matched-S/N target from full-image MAD to annulus MAD changes the
injected source flux. A scientifically clean comparison therefore requires a
new experiment run after this refactor; merely re-rendering the existing report
is insufficient.

## File-by-file changes

### Add

`scripts/imaging/metrics.py`

- CASA plane loading with finite and internal-mask handling;
- beam-radius mask creation reusable by measurement and plotting;
- the pure residual summarizer;
- `measure_image_metrics()` and shared PB-region measurement;
- the preserved exact Pipeline annulus adapter, or shared helpers used by that
  adapter.

### Update

`scripts/imaging/models.py`

- add `BeamRegion`, `RegionMetrics`, and `ImageMetrics` beside the existing
  `Beam` model;
- replace the old overlapping `ResidualMetrics` record;
- change `QAReport.metrics` and `ImagingResult.qa` typing for schema 2.

`scripts/imaging/qa.py`

- remove metric arithmetic and CASA image loading now owned by `metrics.py`;
- keep serialization, validity reporting, text formatting, and tclean-summary
  normalization;
- render the selected region and nested metrics.

`scripts/imaging/imaging.py`

- accept and forward `metric_region`;
- call the canonical measurement function once during finalization.

`scripts/imaging/plot_utils.py`

- accept the same optional `BeamRegion` in both plotting entry points;
- pass it to the residual plot by default;
- contour the canonical region mask without filling it;
- preserve lime green for the beam and solid cyan for the CLEAN mask;
- render the metric boundary as dashed white with a black halo and one compact
  legend key.

`scripts/imaging/vla_pipeline.py`

- accept the identical `metric_region` keyword;
- preserve the separately reported Pipeline weblog background RMS.

`scripts/imaging/__init__.py`

- export the three metric records, `BeamRegion`, and
  `measure_image_metrics()`.

`scripts/simulation/rms_metrics.py` and `scripts/simulation/__init__.py`

- remove `RMSMetrics` and the image-statistics exports;
- delete duplicated ordinary RMS/MAD arithmetic;
- update all callers to import `RegionMetrics`, `measure_pb_region()`, and the
  exact Pipeline adapter from `scripts.imaging`;
- remove `rms_metrics.py` once no imports remain.

`scripts/compare_extracted_thermal_simulations.py`

- define one `METRIC_REGION` constant;
- pass it to both imaging calls;
- keep `DefaultImagingConfig.mask_nbeams` and the metric region independent;
- use `dynamic_range_scaled_mad` for source-flux targeting;
- record the region in the manifest.

`scripts/reporting/vla_original_simulation_comparison.qmd`

- read schema 1 and schema 2 through one helper;
- label new values as beam-region measurements;
- add RMS/MAD and selected beam area;
- build every population plot from the same recorded region.

## Test plan

### Pure unit tests

1. Reject negative, non-finite, reversed, and equal bounds.
2. Verify the four nullable-bound selection cases on a small known grid.
3. Verify that radius is measured from the same centre and beam-major scale as
   `central_circle_mask()`.
4. Verify non-square cell handling.
5. Combine finite pixels and the CASA internal mask correctly.
6. Confirm that all residual metrics use exactly the selected values.
7. Confirm scaled MAD and ordinary RMS on Gaussian-like fixed arrays.
8. Confirm that an extreme outlier moves RMS and peak/MAD much more than scaled
   MAD.
9. Confirm that both dynamic ranges use the global positive clean peak and the
   selected residual RMS or scaled MAD, respectively.
10. Fail clearly for empty or too-small selections.
11. Record `n_pixels` and `area_synthesized_beams` correctly.
12. Verify that measurement and plotting use the same boolean region mask.
13. Verify that full, central, exterior, and annular selections produce zero,
    one, one, and two visible boundaries respectively when those bounds fall in
    the frame.

### Compatibility tests

1. `BeamRegion()` reproduces the existing full-image metrics on an unmasked
   fixture, apart from newly added fields.
2. Existing `measure_pb_region()` RMS and scaled MAD results remain within
   numerical tolerance after switching to the shared summarizer.
3. The fixed 0012-399 `vla_pipeline_annulus_rms()` regression remains unchanged
   at its current CASA-version tolerance.
4. The schema-1 report fixture still renders.
5. A schema-2 fixture renders the region, nested metrics, and all histograms.

### CASA integration tests

1. Direct imaging writes schema-2 full-image QA by default.
2. Direct imaging with an exterior/annulus selection records nonzero pixels,
   valid scaled MAD, and the requested bounds.
3. VLA Pipeline imaging accepts the same region object without changing its
   Pipeline weblog RMS.
4. Original and simulated images in a one-sample comparison use identical
   region bounds.
5. The canonical extracted MS tree remains unchanged.
6. The residual PNG displays the selected metric region, while dirty and clean
   PNGs remain unchanged unless explicitly requested.

## Acceptance criteria

- Ordinary RMS and MAD arithmetic exists in exactly one module.
- One `BeamRegion(min, max)` type expresses full, central, annular, and exterior
  selections.
- All normalized residual statistics in one QA record use the same pixels.
- Conventional dynamic range uses the global positive clean peak and selected
  residual RMS; robust dynamic range uses the same peak and selected residual
  scaled MAD. Both definitions are printed in QA text and the report.
- Both imaging entry points calculate the selected metrics automatically.
- CLEAN masking and metric-region selection are independent parameters with no
  implicit coupling.
- The plotter accepts that same metric-region object and draws its exact
  boundaries only, without a fill or a second geometry implementation.
- Metric-region styling remains legible over inferno and visually distinct
  from the lime-green beam and solid cyan CLEAN mask.
- Simulation callers reuse the imaging implementation.
- `RMSMetrics` and `scripts/simulation/rms_metrics.py` are removed after all
  in-repository callers move to the canonical imaging API.
- The exact Pipeline regression statistic remains numerically unchanged.
- Old reports remain renderable and are never mixed silently with new-region
  population metrics.
- The implementation adds one focused metrics module and removes more duplicate
  code than it introduces.
