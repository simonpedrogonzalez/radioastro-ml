# Preprocessing package and ML-ready simulation retention plan

## Objective

Add a `scripts.preprocessing` package that turns completed simulations into
small, self-describing samples and loads those samples with PyTorch. A retained
sample must contain:

- the dirty, final CLEAN, and final residual pixel arrays as FITS files;
- an explicit training label;
- the imaging configuration and QA measurements;
- the simulation configuration, including injected components, noise, and
  random seeds;
- an ordered list containing one JSON/TXT report pair for every applied
  corruption;
- enough WCS, beam, unit, and plotting information to recreate the scientific
  PNG views without retaining PNG files; and
- stable relative references tying all of those records together.

The raw/corrupted Measurement Set and CASA work products are temporary build
artifacts. They may be removed only after the retained sample has been written
and validated. The existing per-experiment and per-sample directory hierarchy
must remain intact.

## Current repository state

Some of the required provenance already exists and should be extended rather
than replaced:

- `scripts.simulation.simulate_ms()` writes `<name>.simulation.json` and returns
  its path in `SimulationResult.metadata_json`.
- That JSON already records input/output MS paths, CASA version, component
  records (including source flux), requested and resolved noise parameters,
  the noise seed, weight initialization, and created paths.
- `scripts.imaging` writes `qa.json` and `qa.txt`. `qa.json` contains the
  effective imaging parameters, measured grid and beam, QA metrics, tclean
  summary, warnings, and product paths.
- Direct and VLA Pipeline imaging currently retain CASA image tables and write
  `dirty.png`, `clean.png`, and `residual.png`. The plotting code temporarily
  exports FITS but deletes it after rendering.
- `scripts.corruption.write_corruption_reports()` now writes a versioned,
  strict-JSON `corruption.json` and a human-readable `corruption.txt` for any
  corruption configuration implementing `to_report_dict()` and
  `to_report_text()`. The JSON envelope contains `schema_version`, caller-owned
  `context`, and the configuration-owned `configuration` object. The current
  amplitude/phase experiment uses this API after applying its corruption.
- The repository's import convention is `scripts.<package>`, and there is no
  root Python packaging configuration. The new package should therefore be
  `scripts/preprocessing/`, imported as `scripts.preprocessing`, rather than a
  disconnected top-level package.

The existing JSON records contain absolute and temporary paths. Cleanup would
make some of those paths stale, so retained manifests must use paths relative to
the sample directory and must distinguish source provenance from retained
products.

## Retained sample contract

Use one manifest per sample as the authoritative loader entrypoint. Do not make
the dataset infer labels or product roles from directory/file names.

```text
<dataset-root>/
├── dataset.json                         # ordered sample index and label map
└── <sample-id>/
    ├── sample.json                      # label and relative references
    ├── corruptions/                     # omitted when uncorrupted
    │   ├── 000/
    │   │   ├── corruption.json
    │   │   └── corruption.txt
    │   └── 001/                         # present for a second applied corruption
    │       ├── corruption.json
    │       └── corruption.txt
    └── simulation/
        ├── <sample-id>.simulation.json
        ├── <sample-id>.simulation.txt
        └── default_imaging/
            ├── dirty.fits.gz
            ├── clean.fits.gz
            ├── residual.fits.gz
            ├── qa.json
            └── qa.txt
```

Existing experiment/sample/variant hierarchy may wrap these sample directories;
`dataset.json` points to each `sample.json` explicitly. Original, baseline, and
corrupted variants are separate ML samples when all are wanted. One
`sample.json` selects exactly one image triplet; the loader must never guess
which nested imaging directory is intended.

The baseline retained contract is exactly three FITS image products. PNGs are
not retained. `qa.txt` and `simulation.txt` are kept because they are tiny and
useful to humans, but JSON is authoritative.
No auxiliary-product retention option is part of this design: masks,
primary-beam images, PNGs, and other imaging work products are not retained in
the compact sample.

### Fixed data partitions

`scripts/preprocessing/partitions.py` contains literal, version-controlled ID
lists for the `train`, `test`, and `val` partitions. The initial assignment is
a contiguous split of the 133 sorted directories in `collect/extracted`: 93
training IDs, followed by 20 testing IDs and 20 validation IDs. It is not
rediscovered or randomized at runtime. All variants of a source ID remain in
the source's partition; for example, `0012-399_phase_only` follows
`0012-399`.

`FitsSimulationDataset` requires a `partition` keyword. It reads manifests only
far enough to identify their source ID, then fully validates and exposes only
samples assigned to the requested partition. Unknown partition names and
sample IDs that cannot be mapped to exactly one hard-coded source ID are
errors, preventing accidental cross-partition leakage.

### `sample.json`

Use a versioned schema with all file references relative to `sample.json`:

```json
{
  "schema_version": 1,
  "sample_id": "0012-399_phase_only",
  "label": {
    "id": 2,
    "name": "phase_corruption"
  },
  "products": {
    "channel_order": ["dirty", "clean", "residual"],
    "dirty": "simulation/default_imaging/dirty.fits.gz",
    "clean": "simulation/default_imaging/clean.fits.gz",
    "residual": "simulation/default_imaging/residual.fits.gz"
  },
  "metadata": {
    "imaging_qa": "simulation/default_imaging/qa.json",
    "imaging_text": "simulation/default_imaging/qa.txt",
    "simulation": "simulation/0012-399.simulation.json",
    "simulation_text": "simulation/0012-399.simulation.txt",
    "corruptions": [
      {
        "corruption": "corruptions/000/corruption.json",
        "corruption_text": "corruptions/000/corruption.txt"
      },
      {
        "corruption": "corruptions/001/corruption.json",
        "corruption_text": "corruptions/001/corruption.txt"
      }
    ]
  },
  "integrity": {
    "algorithm": "sha256",
    "files": {}
  }
}
```

`metadata.corruptions` is always present and is an ordered list. Its order is
the application order, so zero corruptions are represented by `[]`, one by one
report pair, and a composed corruption pipeline by multiple report pairs. Each
entry requires both relative paths; the JSON file is authoritative and the TXT
file is its human-readable companion. Numbered directories prevent report-name
collisions without encoding scientific meaning in filenames. The referenced
corruption JSON already owns the type, strength, selection, time behavior, and
caller-provided context, so `sample.json` and the simulation report must not
copy those fields.

The label is supplied explicitly by the experiment driver. It must not be
derived from a filename. `dataset.json` owns the stable mapping from label names
to integer IDs so independently produced runs do not silently assign different
IDs. Continuous targets such as corruption strength remain authoritative in the
referenced corruption JSON. A later task configuration may select one as a
regression target without rewriting the images or copying it into
`sample.json`.

The integrity section should contain the byte size and SHA-256 of every retained
file after finalization. A sample is complete only after those values have been
verified and `sample.json` has been atomically installed.

## Simulation reporting changes

### Preserve and version the existing report

Move construction and serialization of the current simulation payload behind
shared reporting helpers, for example:

```python
render_simulation_text(report) -> str
write_simulation_reports(report, json_path, text_path) -> None
```

`simulate_ms()` should write both `<name>.simulation.json` and
`<name>.simulation.txt` on every successful simulation. Extend
`SimulationResult` with `metadata_text`. JSON remains the machine-readable
source of truth; the text report is a concise rendering of the same data.

Use atomic writes and strict JSON (`allow_nan=False`). Increment the schema when
the contract changes and retain a reader for schema version 1 so existing
simulation output can still be prepared.

### Required simulation fields

The normalized report should include:

- schema version, sample ID, creation timestamp, repository commit, and CASA
  version;
- input visibility identity plus a durable source/archive locator and checksum
  where available (an absolute local path alone is not rerunnable after
  cleanup);
- output MS and generated-artifact provenance, clearly marked as temporary when
  it will be deleted;
- ordered operation stages, such as `predict -> corrupt -> noise -> weights`;
- every injected component with normalized source type, position, spectrum,
  polarization, flux and units;
- requested and resolved noise model/parameters, including the realized
  `simplenoise_jy`, physical assumptions, and seed;
- weight initialization/preservation policy;
- warnings, clipping/validation results, and failure status if partial metadata
  is retained; and
- relative references to downstream imaging reports once imaging completes.

The current component and thermal-noise payloads already provide much of this.
The main gaps are the text rendering, durable source identity, operation-level
lineage, and repository version. Applied-corruption configuration belongs to
the corruption reports and is linked from `sample.json`, rather than being
folded into the simulation schema.

### Corruption integration

Do not scrape corruption settings from log text or duplicate them in simulation
metadata. After each successful application, the experiment driver should call
`write_corruption_reports()` with a unique JSON/TXT destination and context
containing the run/sample identity, application index, and actual seed values
owned by the caller. It then appends that returned report pair to
`metadata.corruptions` in application order.

Validation must require both files in every pair, validate the corruption JSON
schema independently, and reject duplicate paths or inconsistent sample
identity/order in context. A no-corruption sample still contains
`"corruptions": []`; a missing key is invalid. If one of several corruption
applications or report writes fails, the sample must not be finalized.

## Imaging retention changes

Both imaging entrypoints should gain an explicit option named along the lines
of:

```python
keep_intermediate_products: bool = False
```

The default `False` means the final durable imaging products are the three
compressed FITS files plus QA metadata. `True` retains the existing CASA image
tables and ancillary products for debugging. This option governs products owned
by the imaging output directory only; it must never delete the input MS.

Finalization order is important:

1. Complete tclean/Pipeline imaging.
2. Calculate metrics and render any requested PNGs while CASA mask and image
   tables still exist.
3. Export dirty, CLEAN, and residual images to temporary FITS paths.
4. Validate FITS shape, finite-data policy, `BUNIT`, celestial WCS, beam header,
   and agreement among all three grids.
5. Compress and atomically rename the FITS products.
6. Write QA JSON/text with retained relative FITS references and separate
   provenance for temporary CASA products.
7. Reopen and checksum the complete retained set.
8. Only then remove owned CASA work products when
   `keep_intermediate_products=False`.

The initial implementation should export the same image planes currently used
by plotting and QA. FITS must preserve the native physical values, normally
Jy/beam; image scaling for ML is a loader/transform concern, not an export-time
mutation.

`ImagingResult` should expose explicit `dirty_fits`, `clean_fits`, and
`residual_fits` paths. CASA table fields should become optional when
intermediates are not retained. Existing callers and tests that inspect CASA
tables must request `keep_intermediate_products=True`; reporting callers should
switch to the retained FITS/QA paths.

### Reconstructing plots

FITS provides pixels, celestial WCS, physical units, and beam headers, but it is
not by itself a complete Matplotlib rendering recipe. Store the following in
the imaging JSON at plot creation time:

- selected FITS plane and array orientation;
- Jy/beam-to-display-unit conversion;
- exact numeric `vmin` and `vmax`, percentile rule, and symmetric/asymmetric
  policy;
- colormap name and library/version;
- origin, interpolation, figure size, DPI, margins, and colorbar label/ticks;
- title and celestial axis labels/formatting;
- beam ellipse position and style;
- metric-region geometry and style; and
- analytic overlay geometry and style, if shown.

Also record Python, NumPy, Astropy, and Matplotlib versions. This is sufficient
to recreate the intended scientific plot in a pinned environment when every
overlay can be derived from retained pixels or recorded analytic geometry. A
plot that depends on an automatic/non-analytic mask cannot be exactly
reconstructed under this compact contract and should omit that overlay when
rendered later. Literal pixel-for-pixel reproduction across arbitrary
environments is outside this contract because font rendering and layout can
vary by platform.

## `scripts.preprocessing` package

Proposed layout:

```text
scripts/preprocessing/
├── __init__.py       # small public API
├── schema.py         # sample/dataset manifest parsing and validation
├── cleanup.py        # retention planning, verification, and safe deletion
├── fits.py           # FITS validation and tensor-plane loading
├── partitions.py     # fixed train/test/val source-dataset ID lists
├── dataset.py        # PyTorch Dataset and collator
└── loader.py         # DataLoader construction helper
```

The package used for training must not import CASA. CASA-dependent FITS export
belongs to imaging finalization; preprocessing only validates/loads the exported
FITS and manages already-declared artifacts. Runtime dependencies for this
package are PyTorch, NumPy, and Astropy and should be declared explicitly when
the repository adds its Python environment/package configuration.

### Cleanup API

Provide a function such as:

```python
cleanup_simulation_sample(
    sample_dir,
    *,
    manifest="sample.json",
    dry_run=True,
) -> CleanupReport
```

The default dry run is intentional because this operation is destructive. The
experiment driver may call it with `dry_run=False` only after it has created a
complete sample manifest.

Cleanup should:

- resolve and validate all manifest references beneath the exact sample root;
- reject symlinked sample roots and any path traversal/out-of-root target;
- verify the three FITS files can be opened, have compatible shapes/WCS/units,
  and match stored checksums;
- verify label, QA, simulation metadata, every ordered corruption report pair,
  and all schema versions;
- build a deterministic deletion plan and report reclaimed bytes before doing
  anything;
- remove only recognized generated artifacts, never broad wildcard matches;
- leave unexpected files untouched and either report them or fail in strict
  mode;
- preserve the directory structure containing the retained sample;
- write an audit record listing retained/removed paths and byte counts;
- be idempotent, so a second run validates the compact sample and deletes
  nothing; and
- stop without deletion if validation fails.

Recognized removable artifacts include the simulated/corrupted MS copies,
component lists after their contents are captured in metadata, gain tables after
their normalized configuration is captured, CASA image/model/residual/PSF/mask/
sumwt/PB tables, first-pass/probe products, Pipeline contexts and weblog assets,
large run logs, PNGs, temporary FITS, flag versions, and `table.lock` files.
There is no retention switch for these products. Original external input data
is never a cleanup target.

Cleanup needs to update or supersede stale product paths in old `qa.json` and
simulation JSON. Provenance may state that a temporary artifact was removed,
but active `products` references must point only to retained files.

### Dataset contract

Provide a map-style dataset with an API similar to:

```python
FitsSimulationDataset(
    root,
    *,
    partition="train",
    index="dataset.json",
    channels=("dirty", "clean", "residual"),
    transform=None,
    target_transform=None,
    validate=True,
    load_metadata=True,
)
```

Each item should have a stable structure:

```python
{
    "image": Tensor,          # [3, height, width], channel order from manifest
    "label": int,
    "sample_id": str,
    "partition": str,
    "qa": dict,
    "metadata": {
        "simulation": dict,
        "imaging": dict,
        "corruptions": list[dict],  # ordered decoded corruption JSON reports
    },
    "paths": dict,           # resolved paths for traceability/debugging
}
```

Loader behavior:

- Open FITS with Astropy and return `torch.float32` by default.
- Select/squeeze only declared singleton frequency/Stokes axes, then require a
  two-dimensional plane. Do not silently flatten a real cube.
- Enforce the declared channel order `dirty`, `clean`, `residual`.
- Require equal shapes and compatible celestial WCS for all three images.
- Preserve raw physical pixel values by default. Normalization, clipping,
  augmentation, and channel selection are explicit transforms.
- Define a configurable NaN/Inf policy. Validation should fail by default;
  replacing invalid pixels with a fill value must be explicit and reported.
- Validate FITS `BUNIT` and either require one unit across channels or perform an
  explicit recorded conversion.
- Resolve all manifest references safely beneath the sample root.
- Require `partition="train"`, `"test"`, or `"val"`; expose and fully
  validate only samples whose source ID occurs in that hard-coded list.
- Load and validate corruption JSON reports in manifest order; return `[]` for
  an uncorrupted sample. The text companions remain traceability paths and are
  not parsed as machine metadata.
- Use `dataset.json` ordering for reproducibility. Optional recursive discovery
  may help build that index, but training should use a frozen index.

Variable-size metadata dictionaries do not work well with PyTorch's default
collation. Supply a collator that stacks image tensors and numeric labels while
keeping `sample_id`, QA, metadata, and paths as per-sample lists. A helper such
as `make_simulation_dataloader()` can construct the standard PyTorch
`DataLoader` with that collator and normal options including batch size,
shuffle, workers, pinning, and deterministic generator seed.

Do not make the loader import experiment scripts, consult the current working
directory, or require access to deleted Measurement Sets.

## Expected file-level changes

The implementation is expected to touch the following areas:

| Path | Planned responsibility |
| --- | --- |
| `scripts/simulation/reporting.py` | Versioned simulation model validation, JSON serialization, and text rendering. |
| `scripts/simulation/simulations.py` | Build the expanded report, write JSON/TXT, and return both metadata paths. |
| `scripts/simulation/__init__.py` | Export the intended public reporting/result API. |
| `scripts/imaging/plot_utils.py` | Return the resolved plot recipe rather than keeping color-limit decisions implicit. |
| `scripts/imaging/models.py` | Represent retained FITS separately from optional CASA work products. |
| `scripts/imaging/imaging.py` | Export/validate FITS and apply the direct-imaging retention option. |
| `scripts/imaging/vla_pipeline.py` | Apply the same retained-product contract to VLA Pipeline output. |
| `scripts/imaging/qa.py` | Serialize retained relative product references and plot provenance. |
| `scripts/preprocessing/` | Add manifest validation, cleanup, FITS loading, Dataset, collator, and DataLoader helper. |
| `scripts/preprocessing/partitions.py` | Own the literal, non-overlapping train/test/val source-ID lists. |
| experiment drivers under `scripts/` | Assign labels, write/reference ordered corruption report pairs, write sample manifests, and invoke cleanup. |
| `tests/` | Add pure unit tests plus narrowly opt-in CASA integration coverage. |

No simulation, imaging, or cleanup logic should live in the runnable experiment
drivers beyond orchestration and experiment-specific label/configuration
selection.

## Experiment-driver integration

For every completed simulated variant, drivers should follow this sequence:

```text
simulate and write simulation JSON/TXT
            ↓
apply each corruption and write its JSON/TXT report pair
            ↓
image and export the three retained FITS plus QA JSON/TXT
            ↓
write label, core metadata references, and ordered corruption pairs to sample.json
            ↓
validate and checksum the retained sample
            ↓
optionally clean temporary simulation artifacts
            ↓
atomically add the sample to dataset.json
```

Do not add a sample to the dataset index before cleanup succeeds. Interrupted
samples remain resumable work directories and are not visible to training.
Experiment-level `report.json` can continue to drive Quarto reports, but it
should reference the same sample manifests rather than duplicate scientific
configuration.

Existing experiment verification currently requires PNG and CASA products. It
must be migrated to verify sample manifests, three FITS files, metadata, and
checksums instead. Any report that needs images should render them on demand
from FITS and the stored plot recipe.

## Compatibility and migration

Implement readers for the current simulation schema and QA schema version 2.
Validate corruption report schema version 1 through the public corruption
report contract. Existing runs that already have a single
`corruption.json`/`corruption.txt` pair at the variant root may reference those
files in a one-entry `metadata.corruptions` list; migration does not need to
rename them. New runs use numbered destinations so several applications cannot
collide.

Provide a preparation command/function that can compact an existing completed
run without rerunning simulation when its CASA image tables are still present.
If only PNGs remain, it cannot create scientifically equivalent FITS and must
fail clearly rather than treating PNG RGB values as image data.

Because `keep_intermediate_products=False` changes existing imaging behavior,
migrate internal diagnostic scripts that need CASA tables to pass `True`
explicitly. Run the compact-retention path only for newly completed samples or
after an existing sample passes the new verification. Never bulk-clean old
experiments merely because they resemble the expected layout.

## Tests required

### Simulation reporting

- Existing `simulate_ms()` metadata remains readable after the schema update.
- Successful simulation writes matching JSON and text reports.
- Components, requested/resolved noise, seeds, weights, and versions survive
  serialization.
- Reports are strict JSON and are installed atomically.

### Corruption-report references

- An uncorrupted manifest requires `metadata.corruptions: []`.
- One and several corruption report pairs load in declared application order.
- Missing JSON/TXT companions, duplicate or escaping paths, unsupported report
  schemas, and context inconsistent with the sample/order fail validation.
- The decoded JSON report retains its `schema_version`, `context`, and
  configuration-owned `configuration` without copying those fields into the
  simulation report or `sample.json`.

### Imaging retention

- Dirty/CLEAN/residual roles map to the correct FITS files for direct and VLA
  Pipeline imaging.
- FITS headers preserve WCS, unit, beam, and plane information.
- Plot scale/recipe values reproduce the renderer's selected color limits.
- `keep_intermediate_products=False` deletes only owned work products after
  verification; `True` preserves existing behavior.
- An export or validation failure leaves CASA products intact.

### Cleanup

- Dry run returns the exact deletion plan and changes nothing.
- Compacting a realistic fixture preserves required files and directory
  hierarchy, records reclaimed bytes, and is idempotent.
- Missing/corrupt/mismatched FITS, missing metadata, bad checksums, symlinks,
  path traversal, and unexpected files prevent unsafe deletion.
- External original Measurement Sets are never selected for removal.

### Dataset and DataLoader

- The fixed partitions contain every current source dataset ID exactly once;
  variants remain with their source ID and invalid names fail explicitly.
- Constructing a dataset for one partition excludes and avoids full retained-
  file validation of samples in the other two partitions.
- A tiny synthetic FITS fixture loads as `[3, H, W]` in the declared order.
- Labels, imaging/simulation metadata, and the ordered corruption reports are
  attached to the correct sample.
- Shape, WCS, unit, non-singleton cube, missing channel, and invalid-pixel
  failures are explicit.
- Custom transforms operate without changing stored data.
- Batching stacks tensors/labels and preserves variable metadata as lists.
- Deterministic index ordering works with zero and multiple workers.

### End-to-end integration

Add an opt-in CASA integration test covering simulation, two sequential
corruptions and their report pairs, imaging, FITS export, cleanup, and loading
one resulting sample without CASA. The last loading step should run in ordinary
Python to prove the retained dataset has no CASA dependency.

## Suggested implementation order

1. Freeze `sample.json`, `dataset.json`, simulation-report, corruption-reference,
   and plot-recipe schemas with pure-Python validators.
2. Extend simulation reporting to JSON plus text, and reuse the corruption
   package's existing JSON/TXT reporting API without duplicating its payload.
3. Make imaging retain and validate the three compressed FITS products, then
   add `keep_intermediate_products` without deleting anything initially.
4. Implement cleanup planning, dry-run reporting, guarded deletion, and audit
   records.
5. Update experiment drivers to assign labels, write sample manifests, invoke
   cleanup, and update the dataset index atomically.
6. Implement the FITS dataset, custom collator, and DataLoader helper.
7. Migrate old experiment verification/reporting and enable the default compact
   retention behavior after end-to-end tests pass.

## Acceptance criteria

The work is complete when a newly generated simulation can be compacted to the
three compressed FITS files and versioned metadata, its original directory
hierarchy remains recognizable, and ordinary Python can load it as a
three-channel PyTorch tensor with the correct label, QA, imaging configuration,
simulation provenance, and ordered configured-corruption provenance. Re-running
cleanup must be a no-op, and no temporary product may be deleted before the
compact sample passes validation and checksum verification.
