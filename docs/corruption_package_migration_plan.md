# Compatibility-preserving corruption package migration


## Goal

Move the reusable corruption implementation into a proper
`scripts.corruption` package while preserving the interface used by current
experiments. The migration should make corruption types and reporting helpers
easy to import, without moving the runnable experiment scripts or changing the
scientific behavior of existing calls.

Add a versioned JSON and human-readable text report for each configured
corruption. The report must state which error was configured, its strength,
selection, time behavior, and seed/context values supplied by the experiment.
Keep this report intentionally small; it describes the requested configuration
and does not attempt to reverse-engineer or summarize the resulting gain table.

This is a focused packaging and provenance migration. The broader behavioral
redesign and additional physical corruption types remain described in
[`corruption_framework.md`](corruption_framework.md). They should not be mixed
into this migration unless a minimal correction is required to preserve an
already-working caller.

## Scope

Move these reusable modules into the new package:

- `scripts/corruption.py`;
- `scripts/corrfn.py`;
- `scripts/corrtab_utils.py`;
- `scripts/timegrid.py`; and
- the corruption-specific plotting helpers currently in
  `scripts/plot_utils.py`.

Keep the runnable/research scripts at their present paths, and untouched (as historical / legacy things) including:

- `scripts/corruption_pipeline.py`;
- `scripts/corruption_gaindrift.py`;
- `scripts/corruption_noise.py` and `scripts/corruption_noise2.py`;
- `scripts/corruption_trop.py` and `scripts/corruption_trop2.py`;
- `scripts/corruption_leakage.py`;
- `scripts/corruption_pointing.py`;
- `scripts/corruption1.py`; and
- experiment drivers such as
  `scripts/single_amp_vs_phase_corruption_simulation.py`.

Those files remain external callers.

## Compatibility contract

The currently used construction and chaining interface must continue to work:

```python
corruption = AntennaGainCorruption(
    timegrid=TimeGrid(solint="10m", interp="linear"),
    amp_fn=amplitude_function,
    phase_fn=phase_function,
    query=(
        GTabQuery()
        .where_in(GCOLS.ANTENNA1, antenna_ids)
        .group_by([GCOLS.ANTENNA1])
    ),
)

corruption.build_corrtable(
    ms,
    gain_table,
    seed=seed,
).apply_corrtable(
    ms,
    gain_table,
    seed=seed,
)
```

Preserve these signatures and behaviors:

```text
AntennaGainCorruption(timegrid, amp_fn=None, phase_fn=None, query=None)
build_corrtable(ms, corrtab, *, seed=0) -> self
apply_corrtable(ms, corrtab, seed=0) -> self
TimeGrid(solint="int", interp="linear")
GTabQuery.where_in(...).where_eq(...).where_between(...)
GTabQuery.sort_by(...).group_by(...)
```

In particular, returning `self` from both corruption methods is required for
the current chained calls. Preserve current parameter names, positional versus
keyword behavior, default values, accepted path types, and phase/amplitude
interpretation during this migration.

The current implementation interprets `amp_fn` output as the complete,
dimensionless gain magnitude and `phase_fn` output as radians. Reporting must
describe that actual behavior; it must not silently reinterpret amplitude as a
fractional offset from unity. A future version can introduce safer amplitude
semantics as a separately tested API change.

### Import compatibility

The preferred imports after migration should be available from one place:

```python
from scripts.corruption import (
    AntennaGainCorruption,
    Corruption,
    GainCorruption,
    CorrFn,
    MagnitudeSpec,
    MaxLinearDrift,
    MaxSineWave,
    RandomPhaseMaxSineWave,
    fBM,
    TimeGrid,
    GCOLS,
    GTab,
    GTabQuery,
    get_unflagged_antennas,
    write_corruption_reports,
)
```

Existing callers also import from the old module paths:

```python
from scripts.corrfn import fBM
from scripts.corrtab_utils import GCOLS, GTabQuery
from scripts.timegrid import TimeGrid
```

Keep `scripts/corrfn.py`, `scripts/corrtab_utils.py`, and
`scripts/timegrid.py` as thin compatibility shims that re-export the same class
objects from `scripts.corruption`. Do the same for the corruption-specific
functions in `scripts/plot_utils.py` if they are moved. The shims must contain
no duplicate implementation and should be marked as compatibility modules, not
deprecated until all experiment notebooks and external users have migrated.

`scripts/corruption.py` cannot remain beside a `scripts/corruption/` directory
with the same import name. Replace it atomically with the package and make
`scripts/corruption/__init__.py` expose the old module's public classes.

## Proposed package layout

```text
scripts/corruption/
├── __init__.py       # stable public imports
├── core.py           # Corruption, GainCorruption, AntennaGainCorruption
├── functions.py      # CorrFn and existing curve implementations
├── tables.py         # GCOLS, GTab, GTabQuery, gain-table helpers
├── timegrid.py       # TimeGrid
├── diagnostics.py    # corruption-function plotting helpers
└── reporting.py      # generic JSON/TXT validation and output
```

The package root should deliberately re-export the commonly configured pieces
so experiment code no longer needs to know which internal module owns them.
Internal modules may change later without changing user imports.

Move CASA, Matplotlib, and optional stochastic-process imports as close as
possible to the operation that needs them. Importing
`scripts.corruption.reporting`, configuration classes, or the package's public
names in ordinary Python should not initialize CASA tools or plotting. This is
important because the preprocessing and metadata readers must work outside
CASA.

## Corruption reporting API

### Configuration objects own their representation

The reporting writer must not know the fields or inner workings of
`AntennaGainCorruption`, `TimeGrid`, `GTabQuery`, or individual curve types.
Otherwise every new corruption class would require type-specific branches in
the writer.

Define a small reporting protocol implemented by every top-level corruption
configuration:

```python
class ReportableCorruption(Protocol):
    def to_report_dict(self) -> dict[str, object]:
        """Return the complete JSON-safe configuration owned by this object."""

    def to_report_text(self) -> str:
        """Return a concise human-readable rendering of that configuration."""
```

`Corruption` should define this interface. `AntennaGainCorruption` implements
it by composing representations supplied by its child configuration objects.
`TimeGrid`, `GTabQuery`, and each built-in `CorrFn` should likewise know how to
represent their own public configuration. Adding a future
`BandpassCorruption`, for example, requires implementing the protocol on that
class; `write_corruption_reports()` remains unchanged.

Use deterministic, concise `__repr__` methods on configuration classes as a
useful default for interactive use and text reports. They should include only
constructor/configuration values, never memory addresses, large sampled arrays,
CASA tool objects, or private caches. A class may implement
`to_report_text()` simply as `repr(self)` when that representation is clear.

Do not use `repr()` as the JSON format. It is not reliably machine-readable or
schema-stable. `to_report_dict()` is the structured contract; `__repr__` or
`to_report_text()` is only the human-readable presentation.

The ownership chain is:

```text
AntennaGainCorruption.to_report_dict()
├── TimeGrid.to_report_dict()
├── amplitude CorrFn.to_report_dict(), or null
├── phase CorrFn.to_report_dict(), or null
└── GTabQuery.to_report_dict(), or an explicit unfiltered value

write_corruption_reports()
└── asks the top-level object for its dict/text and writes them
```

The writer must not inspect `corruption.tg`, `amp_fn`, `phase_fn`, query private
fields, dataclass fields, or class names to decide what to output.

### Generic writer

Add one public writer for any object implementing the protocol:

```python
write_corruption_reports(
    corruption,
    *,
    json_path,
    text_path,
    context=None,
) -> CorruptionReportPaths
```

The intended compatible usage is:

```python
corruption.build_corrtable(ms, gain_table, seed=seed).apply_corrtable(
    ms, gain_table, seed=seed
)

reports = write_corruption_reports(
    corruption,
    json_path=variant_dir / "corruption.json",
    text_path=variant_dir / "corruption.txt",
    context={
        "name": "single_amp_vs_phase",
        "sample_id": "0012-399",
        "variant": "phase_only",
        "seed": seed,
    },
)
```

This is additive: the existing build/apply call is unchanged. Training-data
drivers should call it after successful application so the file identifies the
configuration that was used. It does not inspect the MS or gain table.

The writer's responsibilities are deliberately generic:

- require the reporting protocol;
- call `corruption.to_report_dict()` and `corruption.to_report_text()`;
- validate that the returned dictionary and optional context are strict,
  finite JSON values;
- add only the report schema version and caller-provided context envelope;
- atomically write JSON and text; and
- return their paths.

It must not calculate error strengths, query selections, convert units, inspect
gain tables, or contain an `isinstance()` chain for known corruption classes.
Both files must be written atomically with strict JSON (`allow_nan=False`). If
either cannot be completed, do not leave one final path that makes the run
appear fully reported.

The returned path record should be small and immutable:

```python
@dataclass(frozen=True)
class CorruptionReportPaths:
    json_path: Path
    text_path: Path
```

Do not make reporting dependent on a hard-coded current working directory.
Callers select both output paths. Include paths only when an object owns them or
when the caller deliberately provides them as simple context.

## JSON report schema

Use a small versioned envelope around the configuration-owned dictionary:

```json
{
  "schema_version": 1,
  "context": {
    "name": "single_amp_vs_phase",
    "sample_id": "0012-399",
    "variant": "phase_only",
    "seed": 20260909
  },
  "configuration": {
    "type": "antenna_gain",
    "time_grid": {
      "solint": "10m",
      "interp": "linear"
    },
    "selection": {
      "filters": [
        {"operation": "eq", "column": "ANTENNA1", "value": 0}
      ],
      "group_by": ["ANTENNA1"]
    },
    "amplitude": null,
    "phase": {
      "type": "constant",
      "value": 0.7853981633974483,
      "unit": "radian"
    }
  }
}
```

The exact numeric values above are illustrative, not defaults.

### Configuration fields

For the current antenna-gain object, its own representation should include only
values it already owns and can report directly:

- a stable corruption type name;
- `TimeGrid.solint` and `TimeGrid.interp`;
- configured query filters and grouping in call order;
- whether amplitude and phase functions are absent;
- function type and public constructor/configuration parameters; and
- units determined by the function's role: amplitude is the dimensionless full
  gain magnitude and phase is radians under the current interface.

Do not include inferred application behavior, implicit CASA defaults, created
artifact lists, software versions, selected row counts, realized identifiers,
or numeric statistics unless a future configuration/run object explicitly owns
and exposes them. The first report should favor a small truthful record over a
large record assembled through separate computation.

Add stable `to_report_dict()` and concise representations to `TimeGrid`,
`GTabQuery`, and each built-in `CorrFn`. Do not serialize private implementation
state by blindly dumping `__dict__`. In particular, the fBM sampled `t_grid`
and `x_grid` arrays are mutable sampled state, not constructor configuration and
must not be copied into the configuration report.

Current experiments also use duck-typed functions such as a constant curve.
They should implement the same child-configuration protocol. The current
constant curve can provide, for example, `{"type": "constant", "value":
...}` plus a concise `repr`. An unknown object without a structured
representation should produce an explicit reporting error; `repr()` alone is
acceptable for text but is not enough to create trustworthy JSON.

## Text report

`corruption.txt` should use the top-level configuration object's
`to_report_text()` result. A useful compact result is:

```text
Corruption Configuration
Experiment: single_amp_vs_phase / 0012-399 / phase_only
Seed: 20260909

AntennaGainCorruption(
  timegrid=TimeGrid(solint='10m', interp='linear'),
  amplitude=None,
  phase=ConstantCurve(value=0.7853981633974483, unit='radian'),
  query=GTabQuery(filters=[ANTENNA1 == 0], group_by=['ANTENNA1'])
)
```

The generic writer may add the title and context header, but the configuration
body comes directly from the configuration object. JSON remains the complete
machine-readable detail.

## Relationship to simulation and preprocessing metadata

The corruption report should be usable in either of two ways:

1. retained as `corruption.json`/`corruption.txt` and referenced from the
   simulation/sample manifest; or
2. embedded under the simulation report's ordered `corruptions` list.

Use the same schema content in both cases. Do not maintain an experiment-only
second description with different field names. The preprocessing sample
manifest described in
[`preprocessing_package_plan.md`](preprocessing_package_plan.md) should point to
the corruption report or to the simulation report containing it.

The experiment's class label is related to, but not a replacement for, the
corruption configuration. For example, `phase_corruption` can be the label while
the report carries the selected antenna, `+45 deg` magnitude, solution interval,
and seed.

Before preprocessing deletes a gain table, it must verify that its corruption
JSON/TXT exists, its configuration is valid, and the retained simulation/sample
metadata references the report. A gain table must not be deleted if it is the
only surviving record of the injected error.

## Migration steps

### 1. Freeze current behavior

Add characterization tests for public signatures, chaining, import paths,
query serialization inputs, time-grid configuration, selected gain-table rows,
and current amplitude/phase units. These tests define what this migration may
not accidentally change.

The known issues documented in `corruption_framework.md`, including broken
`solint="int"` handling for the current builder and inconsistent `CorrFn`
protocols, should be marked as known limitations rather than silently changed
as part of moving files.

### 2. Create the package atomically

Create `scripts/corruption/`, move each implementation once, update only
internal imports, and replace `scripts/corruption.py` with the directory in one
commit. Add the old-path compatibility shims in the same commit so the tree is
never left with broken experiment imports.

### 3. Define the public exports

Make `scripts/corruption/__init__.py` the supported access point. Test that
objects imported through old shims are identical to those imported through the
new package, not duplicate classes with incompatible `isinstance()` behavior.

### 4. Add configuration serialization

Add the reporting protocol to `Corruption` and implement
`to_report_dict()`/`to_report_text()` or stable `__repr__` behavior on the
corruption object, `TimeGrid`, `GTabQuery`, and built-in functions. Validate
that all structured values are finite, JSON-safe, unit-labeled, and stable
across runs.

### 5. Add the generic writer

Implement a writer that knows only the reporting protocol, validates the
configuration/context envelope, atomically writes JSON/TXT, and returns both
paths. Reporting failure must not remove or alter the gain table or corrupted
MS.

### 6. Integrate active data-producing drivers

Keep their existing build/apply calls. Add the report-writing call immediately
after successful application, then reference the resulting JSON/TXT from the
experiment and preprocessing manifests. Diagnostic scripts that do not create
dataset samples may opt in later.

### 7. Defer behavioral redesign

After the package migration and report contract are stable, separately address
safe copied-output APIs, immutable corruption functions, seed hierarchy,
amplitude-error conventions, optional diagnostics, and additional corruption
families from `corruption_framework.md`.

## Expected file-level changes

| Path | Planned change |
| --- | --- |
| `scripts/corruption.py` | Replaced atomically by the package directory. |
| `scripts/corruption/__init__.py` | Re-export the stable current interface plus reporting helpers. |
| `scripts/corruption/core.py` | Own current corruption classes and chained methods. |
| `scripts/corruption/functions.py` | Own current corruption-function classes. |
| `scripts/corruption/tables.py` | Own gain-table models, queries, and CASA helpers. |
| `scripts/corruption/timegrid.py` | Own `TimeGrid`. |
| `scripts/corruption/diagnostics.py` | Own corruption-specific plotting. |
| `scripts/corruption/reporting.py` | Validate and atomically write configuration-owned JSON/text representations. |
| `scripts/corrfn.py` | Compatibility re-export shim. |
| `scripts/corrtab_utils.py` | Compatibility re-export shim. |
| `scripts/timegrid.py` | Compatibility re-export shim. |
| `scripts/plot_utils.py` | Compatibility exports if plotting helpers move. |
| active simulation/corruption drivers | Add report call and manifest references; retain location and build/apply interface. |
| `tests/` | Add import, compatibility, schema, rendering, and opt-in CASA tests. |

## Tests required

### Import and API compatibility

- Existing import statements in repository scripts still import successfully.
- Old and new paths return the exact same class/function objects.
- Constructor and method signatures/defaults remain unchanged.
- `build_corrtable(...).apply_corrtable(...)` still chains by identity.
- The package can expose configuration/reporting utilities outside CASA.

### Configuration reporting

- `None`, constant, sine, random-phase sine, drift, and fBM amplitude/phase
  configurations serialize with correct type, parameters, and units.
- Every configuration type owns its dictionary and text/repr representation;
  the generic writer contains no corruption-specific field extraction.
- Query filters preserve call order and convert NumPy values to strict JSON.
- Time-grid input values are recorded by `TimeGrid` itself.
- A new fake corruption class can write reports by implementing the protocol,
  without changing the writer.
- Custom duck-typed functions either implement the child protocol or fail
  explicitly for structured JSON.
- JSON is versioned and strict; text deterministically reflects its contents.
- Temporary-file or serialization failure does not leave a partial final pair.

### CASA integration

- A constant one-antenna phase experiment reports its configured antenna and
  phase without inspecting the gain table.
- A constant `1.5` amplitude reports the configured full gain magnitude `1.5`
  without adding separately computed statistics.
- The existing amplitude-versus-phase experiment writes one report per variant
  and references it from its manifest.

## Acceptance criteria

The migration is complete when current experiment scripts run with their
existing imports and unchanged `AntennaGainCorruption` build/apply chains, the
reusable implementation lives only under `scripts/corruption/`, old secondary
module paths are compatibility shims, and every training-data corruption can
write validated JSON/TXT from representations owned by its configuration
objects. Adding a new top-level corruption type must require no changes to the
generic writer. No runnable corruption experiment or direct external CASA call
needs to be moved into the package.
