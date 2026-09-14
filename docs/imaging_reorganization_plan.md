# Imaging Package Reorganization

## Goal

Create a small reusable imaging package with two public imaging entrypoints:

```python
direct_result = image_ms(ms, config, output_dir, imsize=(256, 256))
vla_result = image_ms_VLA_pipe(
    ms,
    output_dir,
    imsize=(256, 256),
    mask_nbeams=6.0,
)
```

`ms` may be either a visibility ID such as `"0012-399"` or a path to an
existing Measurement Set. `image_ms()` uses an explicit `ImagingConfig`.
`image_ms_VLA_pipe()` uses the CASA VLA Imaging Pipeline as its final imaging
engine and therefore takes no `ImagingConfig`. Its wrapper prepares a
Pipeline-compatible input, selects a target field and continuum mode, fixes the
requested image dimensions, and optionally supplies a central mask; the
Pipeline derives the remaining imaging parameters. Both functions share the
same ID/path resolution, individual plotting, QA calculation, result model,
and self-contained output-directory contract. Both default to a 256x256 image
and accept either a scalar square size or an `(x, y)` override.

The first implementation should stop at individual image PNGs and the latest
QA report. Multi-panel plot composition is deliberately outside this phase.

## Preliminary design decisions

1. `image_ms()` is the direct-imaging public entrypoint. Callers pass only an
   ID or MS path, an `ImagingConfig`, and an output directory.
2. `MeasurementSetMetadata` is renamed to `MSBandMeta`. It describes the band
   inferred from the Measurement Set, not the whole MS.
3. Configuration objects own resolution of their derived values. The outer
   `ImagingConfig.resolve()` coordinates the smaller `resolve()` methods.
4. `GridConfig.resolve()` owns the two-step grid estimate and returns the
   measured `Beam` and computed `ImageGrid` together.
5. CSV-derived metadata is found internally. Callers do not construct or pass
   `CSVMeta` or `MSBandMeta` to either entrypoint.
6. The calibrator-metadata location and extracted-MS root are direct-imaging
   configuration attributes with repository defaults; the VLA entrypoint uses
   those package defaults.
7. Direct imaging uses one explicit `datacolumn` value: `"corrected"` or
   `"data"`. The VLA wrapper has no data-column argument: it validates and
   prefers `CORRECTED_DATA`, then falls back to validated `DATA` with a QA
   warning.
8. `CleanIterationsConfig` has no `mode`. Its populated fields directly state
   the stopping controls, so a separate mode would duplicate that information.
9. The default CLEAN iteration ceiling is `1_000_000`. It is a ceiling, not a
   request to perform all one million iterations; CASA may stop earlier when a
   threshold or `nsigma` criterion is met.
10. A package-level `DefaultImagingConfig` contains the canonical defaults now
    used by the imaging workflow.
11. The output directory contains CASA products, individual PNGs, and the QA
    report in both text and JSON formats.
12. The QA report is the authoritative final product of this phase. Plot
    composition can later consume the saved images and JSON without being part
    of imaging itself.
13. `ImagingResult` includes a normalized summary of the final imaging
    `tclean` execution: iterations, major cycles, stopping code and description,
    convergence values, and per-cycle history.
14. `image_ms_VLA_pipe()` runs the VLA pipeline as the final imaging engine. It
    has Pipeline-specific input preparation and parameter setup, then shares
    the product plotting, QA, and result finalization used by `image_ms()`.
15. Existing scripts are reference implementations, not code to copy blindly.
    New code should preserve verified behavior while simplifying globals,
    branching, and duplicated flows.


These design decisions should be followed unless specific evidence encountered during
implementation provides a concrete reason for a different choice.

## Package structure

```text
scripts/imaging/
├── __init__.py       # Both entrypoints, ImagingConfig, DefaultImagingConfig
├── imaging.py        # image_ms() and shared setup/finalization
├── vla_pipeline.py   # image_ms_VLA_pipe() and VLA pipeline invocation
├── config.py         # User configs, resolved configs, and defaults
├── metadata.py       # ID/path, CSVMeta, MSBandMeta, data-column resolution
├── models.py         # Beam, ImageGrid, ImagingResult, and QA dataclasses
├── qa.py             # Metric calculation and TXT/JSON serialization
└── plot_utils.py     # Individual dirty, clean, and residual PNGs
```

There is no plot composer in the initial package. There is also no generic CASA
runner abstraction: direct imaging should call `tclean`, `imhead`, and `imstat`
where their results are used. The two entrypoints should share only small,
concrete operations such as path resolution, output setup, product discovery,
plotting, QA, and result serialization; do not build an abstract multi-engine
framework.

## Public API

```python
from pathlib import Path

from scripts.imaging import DefaultImagingConfig, image_ms, image_ms_VLA_pipe

result = image_ms(
    "0012-399",
    DefaultImagingConfig,
    Path("experiments/0012-399"),
)

vla_result = image_ms_VLA_pipe(
    "0012-399",
    Path("experiments/0012-399-vla"),
)
```

The initial signature should remain explicit:

```python
def image_ms(
    ms: str | Path,
    config: ImagingConfig,
    output_dir: str | Path,
    *,
    imsize: int | Sequence[int] = (256, 256),
) -> ImagingResult:
    ...
```

A convenience default for `config` may be added later, but the underlying API
must retain an explicit `ImagingConfig` path for reproducible experiments.

The VLA-pipeline entrypoint is intentionally smaller:

```python
def image_ms_VLA_pipe(
    ms: str | Path,
    output_dir: str | Path,
    *,
    imsize: int | Sequence[int] = (256, 256),
    mask_nbeams: float | None = 6.0,
) -> ImagingResult:
    ...
```

It does not accept an `ImagingConfig`. `mask_nbeams` is the diameter of a
centered circular mask in synthesized beams; the default is six beams and
`None` disables the mask. The wrapper passes the requested dimensions to
`hif_makeimlist(hm_imsize=...)`, prepares the first MS field as a continuum
`regcal` target, and fixes the Pipeline cycle factor at `3.0`. The VLA Pipeline
derives the spectral-window selection, cell size, weighting, deconvolution,
and stopping controls. The wrapper records the effective values from the
Pipeline's generated `tclean` call and returns them in the same result shape as
direct imaging.

### ID and path resolution

```python
def resolve_path(ms: str | Path, extracted_root: Path) -> ResolvedMS:
    ...
```

Resolution rules:

- An existing path is used directly.
- Otherwise the value is treated as a visibility ID and searched for below the
  configured extracted-MS root.
- Direct imaging obtains that root from `ImagingConfig`; VLA-pipeline imaging
  uses the package's `DEFAULT_EXTRACTED_MS_ROOT` because it has no imaging
  configuration.
- Resolution must identify exactly one MS. Zero or multiple matches produce a
  useful error listing the searched location and candidates.
- `ResolvedMS` records both the resolved absolute path and the visibility ID,
  when one can be inferred.

## Metadata models and discovery

Use precise immutable result types:

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CSVMeta:
    visibility_id: str
    array_configuration: str | None
    catalog_band_codes: tuple[str, ...]
    catalog_frequency_ghz: float | None


@dataclass(frozen=True)
class MSBandMeta:
    selected_band: str
    representative_frequency_ghz: float
    frequency_reference_spw: int
    catalog_band_matches: bool | None


@dataclass(frozen=True)
class Beam:
    major_arcsec: float
    minor_arcsec: float
    position_angle_deg: float


@dataclass(frozen=True)
class ImageGrid:
    imsize: tuple[int, int]
    cell_arcsec: tuple[float, float]
    field_of_view_arcsec: tuple[float, float]
```

`CSVMeta` and `MSBandMeta` are resolved internally:

- `CSVMeta` is looked up by visibility ID in the configured calibrator metadata
  CSV.
- `MSBandMeta` is inferred from the MS spectral-window metadata and may compare
  itself with the catalog data when `CSVMeta` is available.
- A path-only MS with no matching CSV row can still be imaged when the catalog
  fields are not essential. The missing lookup and any fallback must be
  recorded in QA.
- Neither public entrypoint accepts a metadata object.

Default locations should be repository-relative configuration values rather
than hard-coded absolute user paths:

```python
calibrator_meta_csv = Path("collect/small_subset/small_selection.csv")
extracted_ms_root = Path("collect/extracted")
```

The exact existing CSV filenames should be confirmed during implementation.
Path resolution should be anchored to the repository/package location, not the
process's current working directory.

## Configuration and resolution

Configuration types contain user intent. Separate resolved types contain
measured or computed values and are immutable.

```python
@dataclass(frozen=True)
class ResolvedGrid:
    beam: Beam
    image_grid: ImageGrid
    first_pass_imagename: Path


@dataclass(frozen=True)
class ResolvedCleanControls:
    niter: int
    threshold: str | None
    nsigma: float | None
    cycleniter: int | None


@dataclass(frozen=True)
class ResolvedImagingConfig:
    ms_path: Path
    visibility_id: str | None
    csv_meta: CSVMeta | None
    ms_band_meta: MSBandMeta
    datacolumn: str
    grid: ResolvedGrid
    # All other effective CASA parameters also belong here.
```

### Resolution hierarchy

Conceptually, resolution proceeds as follows:

```text
ImagingConfig.resolve(ms, workspace)
├── resolve_path(ms)
├── resolve CSVMeta from calibrator_meta_csv
├── resolve MSBandMeta from the Measurement Set
├── resolve and validate datacolumn
└── GridConfig.resolve(ms)
    ├── run the first-pass image
    ├── measure Beam
    └── compute ImageGrid
```

The outer resolver should create and cache a small internal resolution context
so nested configs do not repeat MS or CSV reads. A nested resolver may receive
that private context, but public callers never provide `CSVMeta` or
`MSBandMeta` themselves. This preserves the simple conceptual API
`GridConfig.resolve(ms)` while allowing `ImagingConfig.resolve()` to coordinate
and reuse metadata.

Some CLEAN values cannot be resolved until the dirty image exists. In
particular, a threshold expressed as a fraction of the dirty peak requires a
staged resolution:

```text
resolve MS, metadata, datacolumn, and grid
run dirty imaging
CleanIterationsConfig.resolve(dirty_residual)
run clean imaging
```

This is still one `image_ms()` operation. The returned result and QA files must
contain the final effective values, not only the unresolved inputs.

### `GridConfig`

```python
@dataclass(frozen=True)
class GridConfig:
    pixels_per_beam: float = 4.0
    field_of_view_in_beams: float = 64.0
    min_imsize: int = 128
    max_imsize: int = 1024
    min_cell_arcsec: float = 0.02
    max_cell_arcsec: float = 50.0

    def resolve(self, ms: ResolvedMS, *, workspace: Path) -> ResolvedGrid:
        ...
```

`GridConfig.resolve()` absorbs the current `estimate_grid_2step` behavior:

1. Choose a safe provisional grid from internally resolved MS/band metadata.
2. Run a zero-iteration first-pass `tclean` in the output workspace.
3. Read and validate the synthesized beam.
4. Compute the final cell size and apply the requested image dimensions
   (256x256 by default).
5. Return both `Beam` and `ImageGrid` as one `ResolvedGrid`.

`estimate_grid_2step` should not remain part of the public API. A small private
helper is acceptable if it keeps `GridConfig.resolve()` readable.

### Data-column validation

`ImagingConfig.datacolumn` is a single explicit direct-imaging parameter:

```python
datacolumn: Literal["corrected", "data"] = "corrected"
```

There is no automatic selection in `image_ms()`: a direct run uses exactly the
column requested by its configuration.

Resolution must map CASA names to actual table columns and validate the chosen
column using chunked reads so a large MS is not loaded into memory. At minimum,
it must confirm that:

- the table column exists;
- the selected rows contain unflagged samples;
- at least one unflagged value is finite; and
- the usable data are not entirely empty or zero-valued.

For direct imaging, failure raises an error naming the requested and available
columns and explaining why the requested column was unusable; it does not fall
back to a different column. The VLA entrypoint instead prefers usable
`CORRECTED_DATA`, falls back to usable `DATA`, and records the fallback reason
as a QA warning. It uses CASA `split` to materialize the selected column as the
`DATA` column of a Pipeline-specific input MS.

The resolved data-column name and its validation summary must appear in both QA
reports.

### `CleanIterationsConfig`

```python
@dataclass(frozen=True)
class CleanIterationsConfig:
    niter: int = 1_000_000
    threshold: str | None = None
    dirty_peak_fraction: float | None = None
    nsigma: float | None = None
    cycleniter: int | None = None

    def resolve(self, dirty_residual: Path) -> ResolvedCleanControls:
        ...
```

There is no `mode` field. Resolution and validation follow directly from the
fields:

- `threshold` and `dirty_peak_fraction` are mutually exclusive.
- `dirty_peak_fraction`, when present, is converted into an absolute threshold
  after measuring the dirty residual peak.
- `nsigma` may be used as CASA's additional stopping criterion.
- At least one meaningful stopping control should be present in the default
  configuration; otherwise `niter=1_000_000` risks unnecessary work.
- Negative thresholds, fractions, `nsigma`, or iteration counts are rejected.

The report must distinguish the requested ceiling from the iterations CASA
actually performed and must record the CASA stop code/reason when available.

### Final `tclean` execution summary

The return value from the last imaging `tclean` call is part of the result, not
temporary logging. The
[CASA returned-dictionary documentation](https://casadocs.readthedocs.io/en/v6.6.0/notebooks/synthesis_imaging.html#returned-dictionary)
defines these useful top-level fields:

- `iterdone`: total minor-cycle iterations;
- `nmajordone`: total major cycles, including the initial residual calculation;
- `stopcode` and `stopDescription`: the global termination condition;
- `summarymajor`: total iteration counts at major-cycle boundaries; and
- `summaryminor`: per-field/channel/Stokes convergence history, including
  `iterDone`, `peakRes`, `modelFlux`, `cycleThresh`, and minor-cycle stop codes.

Direct imaging should request `fullsummary=True` for its final CLEAN call and
normalize the returned CASA record immediately. The normalizer must tolerate
CASA-version differences in key capitalization and array/scalar types. It must
retain useful unknown keys in a JSON-safe `raw` mapping rather than discard
them. `fullsummary` is an execution-reporting switch, not an imaging decision,
so it need not be listed among the scientific imaging parameters in the report.

```python
@dataclass(frozen=True)
class TcleanRunSummary:
    source: Literal["return_record", "pipeline_context", "pipeline_log"]
    iterdone: int | None
    nmajordone: int | None
    stopcode: int | None
    stop_description: str | None
    major_cycle_iteration_counts: tuple[int, ...]
    final_peak_residual_jy_per_beam: float | None
    final_model_flux_jy: float | None
    final_cycle_threshold_jy_per_beam: float | None
    minor_cycle_history: dict[str, object] | None
    raw: dict[str, object]
```

The summary is stored as `ImagingResult.tclean_summary` and serialized in full
in `qa.json`. `qa.txt` includes its iterations, major-cycle count, stop code,
and stop description. A missing or malformed return record must not erase the
image products: record nulls and a QA warning with the original keys/types that
could not be normalized.

For VLA-pipeline imaging, capture the corresponding final imaging-stage summary
from the pipeline result/context when it is exposed. If that CASA pipeline
version does not retain the inner `tclean` return record, inspect the standard
pipeline result and log artifacts for explicit values. Never infer a stop
condition merely from the requested parameters. Unavailable fields remain
null, state their provenance, and generate a warning.

### `DefaultImagingConfig`

The package root exports one canonical, reusable default value:

```python
DefaultImagingConfig = ImagingConfig(
    datacolumn="data",
    specmode="mfs",
    stokes="I",
    gridder="standard",
    deconvolver="mtmfs",
    nterms=1,
    weighting="briggs",
    robust=0.5,
    gain=0.1,
    mask_nbeams=6.0,  # centered circular mask, diameter in synthesized beams
    grid=GridConfig(
        pixels_per_beam=4.0,
        field_of_view_in_beams=64.0,
        min_imsize=128,
        max_imsize=1024,
        min_cell_arcsec=0.02,
        max_cell_arcsec=50.0,
    ),
    clean=CleanIterationsConfig(
        niter=1_000_000,
        dirty_peak_fraction=1e-7,
        nsigma=3.0,
        cycleniter=None,
    ),
    calibrator_meta_csv=DEFAULT_CALIBRATOR_META_CSV,
    extracted_ms_root=DEFAULT_EXTRACTED_MS_ROOT,
)
```

These defaults use raw `DATA`, single-term multi-frequency synthesis, a central
mask six synthesized beams in diameter, and a CLEAN ceiling of one million
iterations. The dirty-image absolute peak is multiplied by `1e-7` to produce
an absolute threshold, while `nsigma=3.0` supplies an additional stopping
criterion. `cycleniter=None` means the wrapper does not pass `cycleniter` and
CASA uses its own default.

The exported object is immutable. Experiments should use
`dataclasses.replace(DefaultImagingConfig, ...)` rather than mutate global
state.

## `image_ms()` workflow

`image_ms()` owns this sequence:

1. Create and validate `output_dir`.
2. Resolve the ID/path and the pre-dirty portion of `ImagingConfig`.
3. Run the first-pass beam measurement through `GridConfig.resolve()`.
4. Run the final-grid zero-iteration dirty image.
5. Resolve dirty-image-dependent CLEAN controls.
6. Build the mask using the resolved beam and grid.
7. Run the configured final CLEAN and retain its returned summary record.
8. Normalize the final `tclean` execution summary.
9. Compute QA metrics from the final products.
10. Write individual dirty, clean, and residual PNGs.
11. Write a concise human-readable TXT summary and a complete machine-readable
    JSON report.
12. Return an `ImagingResult` containing resolved configuration, product paths,
    structured metrics, and the final `tclean` summary.

All CASA image prefixes live below `output_dir`; the function must not depend on
or overwrite products elsewhere.

## `image_ms_VLA_pipe()` workflow

The VLA entrypoint uses the same outer flow but replaces the direct dirty/CLEAN
steps with the pipeline sequence already exercised by the VLA comparison test
scripts:

1. Create and validate `output_dir`.
2. Resolve the visibility ID or MS path using the same `resolve_path()` helper.
3. Create all pipeline working directories below `output_dir` and initialize
   the required CASA pipeline context.
4. Validate the source data. Prefer `CORRECTED_DATA`, fall back to `DATA` with
   a warning, and split the selected column into a Pipeline-specific input MS.
5. Select the first field as the target and mark the copied MS as target data.
6. When `mask_nbeams` is not `None`, run a zero-iteration direct-imaging probe
   to measure the synthesized beam, then create a centered circular mask with
   the requested beam-scaled diameter. Skip the probe when masking is disabled.
7. Run the VLA Pipeline import/setup and imaging stages. Request continuum
   `regcal` imaging for the selected field and exact image dimensions, use the
   manual central mask by default (or no mask when disabled), and set the
   Pipeline cycle factor to `3.0`. The Pipeline derives spectral-window
   selection, cell size, weighting, deconvolution, and stopping controls.
8. Parse `pipeline-*/html/casa_commands.log`. Select the last positive-`niter`
   `tclean` call as the final clean and the preceding zero-iteration call as
   the dirty run, then construct product paths from their exact `imagename`
   values. `ImagingResult.dirty_image` is the dirty run's residual product for
   this engine.
9. Record the effective final pipeline imaging parameters and capture the last
   imaging `tclean` execution summary where the pipeline exposes it.
10. Pass those products through the exact same individual plotting and QA
   functions used by `image_ms()`.
11. Write the same JSON report schema and TXT summary format and return the
    same `ImagingResult` type, with `engine="vla_pipeline"` and no direct
    `ImagingConfig`.
12. It is a fundamental requirement that the VLA imaging pipeline computes and writes down in the result the background noise estimate:
must report the background image RMS measured by the VLA Imaging Pipeline.

The VLA pipeline computes a robust residual RMS from a noise annulus between approximately the 0.2 and 0.3 primary-beam response levels and uses it internally for nsigma-based cleaning.

During implementation, read only the explicit `non-pbcor image RMS` value from
the final imaging stage's `t2-4m_details.html` weblog page. Normalize its stated
Jy/beam unit to Jy/beam. Do not inspect Pipeline context/result attributes and
do not use fuzzy key matching or an independently calculated substitute. If
that exact weblog value is absent or unreadable, store `None` and add a warning.

Store the normalized value in ImagingResult/qa.json as:
vla_background_rms_jy_per_beam: float | None

This value is only required for image_ms_VLA_pipe(). Direct imaging does not need to provide it.

The VLA function does not call `image_ms()` internally and does not replace the
Pipeline's final imaging with a hand-written `tclean` call. Its only direct
`tclean` operation is the zero-iteration beam probe used to size the optional
central mask. The commonality belongs in small setup/finalization helpers after
each imaging engine has produced its artifacts.

Take into account that /Users/u1528314/Applications/CASA.app/Contents/MacOS/casa this casa has no access to VLA pipeline while
/Users/u1528314/Applications/CASA \2.app/... (or similar) does have access to the VLA pipeline. This will be important when executing.

## Output directory contract

The exact CASA suffixes depend on the deconvolver, but the stable layout should
be:

```text
output_dir/
├── firstpass.*       # direct imaging only
├── dirty.*           # direct imaging only
├── clean.*           # direct imaging only
├── mask_probe/        # VLA-only direct beam probe when masking is enabled
├── pipeline/          # VLA-pipeline work products only
├── dirty.png
├── clean.png
├── residual.png
├── qa.txt
└── qa.json
```

`qa.json` is the complete structured report. It contains:

- input value, resolved absolute MS path, and inferred visibility ID;
- data-column choice and validation result;
- resolved `CSVMeta` and `MSBandMeta`, including missing-data warnings;
- measured beam and computed image grid;
- every effective CASA imaging and CLEAN parameter;
- paths to dirty, clean, residual, model, mask, PSF, and PNG products;
- requested CLEAN controls and the normalized final `tclean` return summary,
  including iterations, major cycles, stop code/description, convergence
  values, and summary provenance;
- all QA metrics, units, validity flags, and warnings.

`qa.txt` is a concise operational summary. It includes resolved input and
metadata, principal grid/beam and imaging parameters, the requested CLEAN
controls, final iteration/major-cycle/stop status, all metric values, the VLA
background RMS when available, and warnings. It intentionally omits verbose or
machine-oriented content such as product paths, the full effective-parameter
mapping, metric-validity flags, detailed convergence histories, and raw
`tclean` records.

`qa.json` should use a versioned schema so later plot or batch tools can consume
it safely. JSON values should be native numbers, strings, booleans, arrays, or
`null`; CASA and NumPy scalar objects must be normalized before serialization.

No composed comparison or multi-panel PNG is generated in this phase.

## QA model

Keep raw metrics separate from presentation:

```python
@dataclass(frozen=True)
class ResidualMetrics:
    scaled_mad_jy_per_beam: float
    residual_abs_peak_jy_per_beam: float
    residual_min_jy_per_beam: float
    residual_max_jy_per_beam: float
    peak_over_scaled_mad: float
    p99_over_scaled_mad: float
    p99_5_over_scaled_mad: float
    dynamic_range: float


@dataclass(frozen=True)
class QAReport:
    schema_version: int
    metrics: ResidualMetrics
    tclean_summary: TcleanRunSummary | None
    warnings: tuple[str, ...]
```

The implementation should reuse the current metric definitions, including the
exact pixel selection and finite-value handling, so reorganizing the code does
not silently redefine success. Any intentionally changed metric must be called
out in the migration notes and tested against a fixed image.

`qa.txt` is a readable summary of the structured report. `qa.json` is the
complete source for automation and later plot composition.

## Result object

```python
@dataclass(frozen=True)
class ImagingResult:
    engine: Literal["direct", "vla_pipeline"]
    ms_path: Path
    visibility_id: str | None
    output_dir: Path
    resolved_config: ResolvedImagingConfig | None
    effective_imaging_parameters: dict[str, object]
    dirty_image: Path
    clean_image: Path
    residual_image: Path
    model_image: Path | None
    mask_image: Path | None
    psf_image: Path | None
    dirty_png: Path
    clean_png: Path
    residual_png: Path
    qa_text: Path
    qa_json: Path
    tclean_summary: TcleanRunSummary | None
    qa: QAReport
```

The result returns paths rather than loading large CASA images into memory.
For `image_ms()`, `resolved_config` is populated. For `image_ms_VLA_pipe()`, it
is `None` because there was no user imaging configuration; the parameters
actually used by the pipeline are recorded in `effective_imaging_parameters`.

## Reference-first implementation policy

Before implementing any operation, search the repository for an existing,
working version and understand its assumptions and outputs. The first
references to inspect are:

- `scripts/image_extracted.py` for MS metadata, grid estimation, direct CASA
  imaging, masks, and QA definitions;
- `scripts/plot_extracted.py` and `scripts/img_utils.py` for CASA image reading,
  units, orientation, color scaling, and individual plots;
- `scripts/vla_imaging_pipeline_test.py` for VLA pipeline setup, task sequence,
  product discovery, parameter reporting, and metric comparison;
- `scripts/recreating_vla_pipeline_test.py` for direct `tclean` parameter
  capture and final return-summary handling; and
- `scripts/experiment_outputs.py` for output-directory and JSON-writing
  conventions.

Other scripts should be inspected when they contain a closer implementation of
the specific behavior being migrated. Repository search is a required first
step, not an assumption that the files listed above are exhaustive.

Reference code is evidence, not the desired architecture. For each migrated
piece:

1. Identify the minimum behavior and numerical conventions that must remain
   equivalent.
2. Separate genuine CASA requirements from experiment-only globals, fixed IDs,
   path assumptions, presentation code, and historical workarounds.
3. Implement one short, linear code path in the new design.
4. Share a helper only when both entrypoints truly perform the same domain
   operation; avoid boolean switches that create several workflows inside one
   function.
5. Prefer a few explicit dataclasses and functions over registries, adapters,
   inheritance hierarchies, or generic pipeline-engine abstractions.
6. Do not modify legacy code, the implementation should only be done in the new package

Code size and readability are design requirements. The package should be
smaller and easier to trace than the scripts it replaces, with no duplicated
or hidden imaging flows.

## Migration boundary

Implement the package additively. Do not modify `image_extracted.py` or
`simulations.py` until the new path has been validated on a known MS.

Initial migration steps:

1. Capture the current canonical defaults and QA definitions in regression
   fixtures.
2. Implement ID/path and internal metadata resolution.
3. Implement resolved configuration types and data-column validation.
4. Move the two-step beam/grid calculation behind `GridConfig.resolve()`.
5. Implement dirty imaging and staged CLEAN-control resolution.
6. Implement final direct imaging and normalize its full `tclean` return
   summary.
7. Implement shared individual PNG output and TXT/JSON QA reporting.
8. Compare direct products and metrics with the existing workflow on one real
   MS . (for this step use 0012-399, in some kind of test script)
9. Implement `image_ms_VLA_pipe()` from the proven VLA comparison-script task
   sequence, including input normalization, target selection, beam-scaled
   masking, and the explicitly requested Pipeline controls.
10. Confirm both entrypoints produce the same report schema, metric definitions,
    orientation, units, and plot conventions from equivalent products, by writing test files over 0012-399.

Batch execution, simulation generation, and plot composition remain outside
this package. VLA pipeline orchestration is in scope only through
`image_ms_VLA_pipe()`; the core direct-imaging workflow must not depend on it.
