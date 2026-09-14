# Measurement Set corruption framework and package plan

## Purpose

This document has two jobs:

1. describe the corruption code that currently exists in this repository; and
2. define the changes needed to turn it into a safe, reproducible package that
   composes with `scripts.simulation`.

The current code is useful experimental work, but it is not yet a supported
package. Only the antenna-gain path has the beginnings of a reusable
abstraction. Noise, troposphere, leakage, and pointing experiments are still
stand-alone scripts with hard-coded inputs and outputs.

## Current code inventory

### Reusable prototype

| File | Current responsibility |
| --- | --- |
| [`scripts/corruption.py`](../scripts/corruption.py) | `Corruption`, `GainCorruption`, and the concrete `AntennaGainCorruption` builder/applier. |
| [`scripts/corrfn.py`](../scripts/corrfn.py) | Time-dependent scalar models: linear drift, sine waves, random-phase sine waves, and fractional Brownian motion (fBM). |
| [`scripts/timegrid.py`](../scripts/timegrid.py) | Converts a solution interval such as `"10m"` into gain-table sampling knots and declares interpolation behavior. |
| [`scripts/corrtab_utils.py`](../scripts/corrtab_utils.py) | Builds an identity CASA gain table, represents its row metadata as `GTab`, and filters/groups rows with `GTabQuery`. |
| [`scripts/plot_utils.py`](../scripts/plot_utils.py) | Plots the requested corruption function, sampled knots, and gains written at gain-table row times. |
| [`scripts/time_utils.py`](../scripts/time_utils.py) | Converts CASA/MJD time values for diagnostics. |

### Experimental drivers and diagnostics

| File | What it explores | Package status |
| --- | --- | --- |
| [`scripts/corruption_gaindrift.py`](../scripts/corruption_gaindrift.py) | CASA `setgain`, table repair, selected antennas, recovery with `gaincal`, closure tests, and plots. | Research notebook in script form; not reusable as a library. |
| [`scripts/corruption_pipeline.py`](../scripts/corruption_pipeline.py) | Batch flow: copy extracted MS, recalibrate, inject fBM phase gains into two antennas, image, and compare. | Useful end-to-end behavioral reference. |
| [`scripts/corruption_noise.py`](../scripts/corruption_noise.py), [`scripts/corruption_noise2.py`](../scripts/corruption_noise2.py) | Obvious fixed-`simplenoise` tests and before/after images. | Superseded for production use by `scripts.simulation` noise handling. |
| [`scripts/corruption_trop.py`](../scripts/corruption_trop.py), [`scripts/corruption_trop2.py`](../scripts/corruption_trop2.py) | CASA `settrop` experiments using intentionally extreme parameters. | Unvalidated experiment; parameters are not realistic defaults. |
| [`scripts/corruption_leakage.py`](../scripts/corruption_leakage.py) | Constant CASA `setleakage` experiment with IQUV imaging. | Unvalidated experiment; not part of the reusable core. |
| [`scripts/corruption_pointing.py`](../scripts/corruption_pointing.py) | A primary-beam attenuation calculation followed by amplitude `setgain`. | A pointing-like amplitude proxy, not a true pointing-offset implementation. |
| [`scripts/corruption1.py`](../scripts/corruption1.py) | Minimal early `settrop` trial. | Historical prototype. |
| [`scripts/check_noop_tables.py`](../scripts/check_noop_tables.py), [`scripts/check_noop_imaging.py`](../scripts/check_noop_imaging.py) | Characterize what `simulator.reset(); simulator.corrupt()` changes even with no corruption terms. | Valuable source for integration-test expectations. |
| [`scripts/closure.py`](../scripts/closure.py) | Antenna- and baseline-phase experiments plus closure diagnostics. | Valuable validator; baseline mutation should remain separate from antenna-gain corruption. |

The older [`scripts/sim_utils.py`](../scripts/sim_utils.py) is not a suitable
integration point. The supported simulation API is the
[`scripts/simulation/`](../scripts/simulation/) package.

## Current antenna-gain flow

The working concept is a two-step operation:

```python
AntennaGainCorruption(...).build_corrtable(
    ms,
    gain_table,
    seed=seed,
).apply_corrtable(
    ms,
    gain_table,
    seed=seed,
)
```

Both methods operate on caller-provided paths. `apply_corrtable()` modifies the
MS in place; it does not make a protective copy.

### 1. Create an identity gain-table template

`make_template_gain_corrtab()` asks CASA to create a G-type gain table with:

```python
sm.setgain(mode="random", table=gain_table, amplitude=0)
```

It then explicitly overwrites every `CPARAM` value with `1+0j` and verifies
that the table is an identity corruption. This uses CASA to establish the
correct gain-table schema and row layout for the input MS while avoiding
CASA's problematic random/fBM gain values.

The function currently calls `rmtables(gain_table)` first, so an existing table
at that path is deleted without a package-level overwrite policy.

### 2. Read and select gain-table rows

`GTab.from_casa_table()` loads these row-level columns:

```text
TIME
FIELD_ID
SPECTRAL_WINDOW_ID
ANTENNA1
ANTENNA2
INTERVAL
SCAN_NUMBER
OBSERVATION_ID
```

It also stores `ROWID`, the row index used to write generated gains back into
the original `CPARAM` array.

`GTabQuery` can:

```text
where_in(column, values)
where_eq(column, value)
where_between(column, lower, upper)
sort_by(columns)
group_by(columns)
```

The common experiment selects a few `ANTENNA1` IDs and groups by `ANTENNA1`.
In a CASA gain table, this is effectively a per-antenna selection. Unselected
rows stay at the identity gain.

### 3. Construct a time grid

For a numeric solution interval, `TimeGrid.full_grid()` creates uniform knots
from the global minimum gain-table time through the global maximum, plus one
extra knot. Examples are `"10s"` and `"10m"`.

The corruption function is sampled at those knots. Its values are then mapped
to the actual row times with either linear or nearest interpolation.

The intended `solint="int"` behavior is one generated value per gain-table row
time. It is not currently functional because `build_corrtable()` calls
`full_grid()` before checking for `"int"`, and `full_grid()` deliberately
rejects `"int"`.

### 4. Sample an independent curve for each group

For each query group, the builder creates a pseudo-random generator derived
from the top-level seed and group key. Amplitude and phase curves are sampled
separately.

The intended conventions are:

```text
amplitude curve: dimensionless gain magnitude
phase curve: radians
complex gain: amplitude * exp(1j * phase)
```

If no amplitude function is supplied, the magnitude is one. If no phase
function is supplied, the phase is zero.

The resulting one-dimensional gain is broadcast over all correlations and all
channels represented by each selected gain-table row:

```python
CPARAM[:, :, selected_row_ids] = gain[None, None, :]
```

Consequently, the current reusable prototype supports time-variable,
antenna-based gains, but not different curves per correlation or channel.

### 5. Apply the gain table

`apply_corrtable()` opens the MS in CASA and configures:

```python
sm.setapply(
    table=gain_table,
    type="G",
    interp="linear",
    calwt=False,
)
sm.corrupt()
```

For antennas `i` and `j`, the intended visibility operation is:

```text
V_out(i,j) = g_i * conjugate(g_j) * V_in(i,j)
```

An antenna-based corruption should therefore preserve closure phase. The
repository's closure experiments support that expectation. A direct
single-baseline phase edit is a different error family and can violate closure.

`calwt=False` means the corruption does not modify visibility weights. This is
appropriate when modeling an uncorrected calibration error whose nominal
thermal weights are to remain unchanged, but it must be recorded as an
explicit semantic choice.

### 6. Diagnostic plot

`build_corrtable()` always writes:

```text
images/corruption_function.png
```

The location is hard-coded, the directory is not created, and repeated runs
overwrite the same file. Plot generation is currently part of execution rather
than an optional diagnostic.

## Other corruption experiments

The stand-alone scripts establish useful candidates for later package effects,
but they do not yet share configuration, path safety, metadata, or tests.

### Additive thermal noise

The noise scripts call `setnoise(mode="simplenoise")` followed by `corrupt()`.
The supported simulation package now does this more carefully: it calculates
or validates `sigma_simple`, records the seed and physical assumptions, and
initializes `SIGMA`, `WEIGHT`, and `WEIGHT_SPECTRUM` consistently. Thermal
noise should remain owned by `scripts.simulation`, not be duplicated in the
new corruption package.

### Tropospheric corruption

The troposphere scripts use CASA `settrop`, sometimes with a field/SPW
selection from `setdata()`. The checked-in values include extremely large PWV,
fractional fluctuation, and wind-speed settings chosen to make an effect
visually obvious. They must not become defaults. This effect needs a defined
VLA physical model, units, supported CASA versions, and weight semantics before
promotion.

### Polarization leakage

The leakage script uses constant `setleakage` terms and IQUV imaging. Before
promotion it needs a full-polarization MS contract, a clear Jones/D-term
parameter convention, and analytic checks for the expected leakage between
Stokes products.

### Pointing-like corruption

The current pointing script converts a fixed angular offset through a Gaussian
primary-beam approximation and feeds the resulting attenuation to `setgain`.
That is an amplitude-gain proxy. It does not apply an antenna-dependent sky
direction offset and should not be called a physical pointing-error simulator.

### CASA-generated gain corruption

`corruption_gaindrift.py` contains repairs for CASA-generated random and fBM
gain tables: shifting random gains toward unity, clipping extreme values,
neutralizing unselected antennas, and forcing phase-only gains. These are
useful evidence for why the custom Python gain generator was started. They
should remain diagnostic/migration references rather than enter the public API
unchanged.

## Problems to resolve before packaging

### Correctness blockers

1. **The corruption-function protocol is inconsistent.** `CorrFn` declares
   `sample()` and `eval(..., rng=...)`, but the subclasses implement different
   combinations. `AntennaGainCorruption` always calls
   `fn.sample(...).eval(...)`. `MaxLinearDrift` and `MaxSineWave` inherit the
   base `sample()` that raises `NotImplementedError`; their `eval()` requires
   an RNG that the caller does not pass. The plotting code calls `eval()` with
   no RNG. In practice, only the stateful random-phase sine and fBM shapes fit
   the builder's current calling convention.
2. **`solint="int"` is broken.** The builder requests a numeric full grid before
   reaching its integration-time branch.
3. **Amplitude semantics are unsafe.** The builder treats an amplitude model's
   output as the full gain magnitude. A zero-centered drift therefore produces
   gains around zero, not deviations around unity. The new API must choose and
   enforce either `gain_amplitude = 1 + fractional_error` or a log-amplitude
   convention.
4. **The fBM implementation mutates model objects and NumPy's global RNG.** A
   sampled model cannot safely be reused across groups or concurrent jobs, and
   global `numpy.random.seed()` can affect unrelated code.
5. **The group seed derivation is collision-prone.** Summing the bytes of
   `repr(group_key)` can give different group keys the same seed. It should use
   a stable cryptographic digest or `SeedSequence` hierarchy.
6. **The application column and CASA side effects are implicit.** The code
   assumes the simulator's `DATA` behavior and does not validate which columns
   changed. CASA may add `MODEL_DATA` and `CORRECTED_DATA` even for a no-op
   `corrupt()`. The package contract must say exactly which visibility column
   is the input and output.

### Safety and usability blockers

1. `apply_corrtable()` modifies its input MS in place. There is no default
   copy, output-path validation, or overwrite refusal.
2. Gain-table creation deletes an existing target with `rmtables()`.
3. CASA tools are imported at module import time, preventing ordinary Python
   code from importing and testing the pure configuration/model pieces.
4. CASA return values are not consistently checked and tool cleanup is not
   consistently protected by `try/finally`.
5. Plotting is mandatory and writes to one hard-coded path.
6. There is no structured result, JSON provenance record, schema version, or
   list of created artifacts.
7. The mutable query builder is modified again by `build_corrtable()` when it
   adds sorting. Reusing the same query can carry hidden state between runs.
8. Units live in comments: phase is radians, periods and times are seconds, and
   amplitude is dimensionless. Configuration should encode these names or
   validate explicit quantity strings.
9. The prototype broadcasts one gain to every correlation and channel. That
   limitation is not validated or exposed to the caller.
10. External dependencies (`stochastic`, the unused `fbm` import, Astropy,
    Matplotlib, and CASA) are not separated into required versus optional
    runtime dependencies.

### Scientific-contract blockers

1. “Corrupt a calibrated real MS,” “corrupt an ideal synthetic sky,” and
   “simulate a physical observation” are different operations. They need
   explicit input and ordering semantics.
2. Amplitude corruption and thermal noise interact with weights. Preserving
   weights with `calwt=False` is a reasonable residual-calibration model, but
   it is not the same as deriving weights from the post-gain noise variance.
3. Sampling one curve across global elapsed time also spans scan/field gaps.
   The package needs a declared reset policy: continuous across gaps, restart
   per scan, restart per field, or group explicitly.
4. Generated values are identical across correlations and channels. Later
   support for polarization- or frequency-dependent errors must be deliberate,
   not an accidental broadcast.
5. “Realistic” parameters are not yet established. The current code proves
   controllability and some closure/recoverability properties; it does not yet
   establish a population model for VLA calibration errors.

## Proposed package

Replace the module `scripts/corruption.py` with a package of the same import
name. The file and directory cannot coexist reliably, so this must be one
atomic migration with existing imports updated and tested.

```text
scripts/corruption/
├── __init__.py          # small documented public API
├── models.py            # immutable effect, selection, time-grid, and result records
├── functions.py         # deterministic pure time-series models
├── gains.py             # antenna-gain table construction and validation
├── runner.py            # safe copy, ordered application, metadata, cleanup
├── diagnostics.py       # optional plots and realized-effect summaries
└── _casa.py             # lazy CASA imports and narrow adapter functions
```

Do not move thermal-noise calculation into this package. Do not initially move
the large experimental drivers into it. Keep those scripts as callers until
the package API has replaced their reusable portions.

### Initial public API

The first release should support only the best-understood operation:
antenna-based complex gain corruption.

```python
from scripts.corruption import (
    AntennaGain,
    CorruptionResult,
    FractionalBrownianMotion,
    GainSelection,
    SineWave,
    TimeGrid,
    corrupt_ms,
)
```

An intended call should look like:

```python
effect = AntennaGain(
    selection=GainSelection(antenna_ids=(0, 1)),
    time_grid=TimeGrid(interval="10m", interpolation="linear"),
    phase=FractionalBrownianMotion(rms=0.15 * math.pi, hurst=0.05),
    amplitude=None,
)

result = corrupt_ms(
    input_ms="simulation.ms",
    output_ms="simulation.phase-corrupted.ms",
    effects=[effect],
    seed=12345,
)
```

`corrupt_ms()` should:

1. resolve and validate the input as an MS;
2. require an output ending in `.ms` and refuse existing outputs;
3. copy the input, preserving the source unchanged;
4. generate gain tables under an output-specific artifact directory;
5. apply effects in the declared order;
6. validate expected column and row invariants;
7. write metadata atomically; and
8. return a frozen `CorruptionResult`.

Suggested result fields are:

```python
@dataclass(frozen=True)
class CorruptionResult:
    ms_path: Path
    metadata_json: Path
    gain_tables: tuple[Path, ...]
    diagnostic_paths: tuple[Path, ...]
    seed: int
```

### Effect and model contracts

Use immutable specifications. Sampling must return a new realized curve rather
than mutate the specification.

Every time-series model should have one interface:

```python
values = model.sample(times_seconds, rng)
```

It should return one finite value per requested time. Interpolation belongs to
the time-grid/runner layer, not inside each model.

Use explicit gain conventions:

```text
phase model output: radians
amplitude-error output: fractional deviation from unity
gain amplitude: 1 + amplitude_error
complex gain: (1 + amplitude_error) * exp(1j * phase)
```

Reject non-positive gain amplitudes unless a separately configured clipping
policy is present. Record clipping counts; never clip silently.

`GainSelection` should use typed fields rather than exposing raw gain-table
column strings:

```python
GainSelection(
    antenna_ids=(0, 1),
    field_ids=None,
    spw_ids=None,
    scan_numbers=None,
    time_range_s=None,
    group_by=("antenna",),
    reset_at="observation",  # later: "scan" or "field"
)
```

The implementation may still use `GTab` internally. Empty selections should
be errors by default, not successful no-ops.

### Reproducibility

Derive one RNG stream per effect and selection group from:

```text
top-level seed
effect index
effect type
canonical group key
```

Use `numpy.random.SeedSequence` with stable digest-derived integers. Never use
Python's salted `hash()`, a byte sum, or the global NumPy RNG.

Metadata must include:

```text
schema version
input and output MS paths
CASA version
top-level seed and deterministic child-stream identifiers
ordered effect specifications
selected antenna/field/SPW/scan IDs
time-grid and interpolation policy
gain-table paths
realized amplitude and phase summaries
clipping or validation warnings
weight policy
created artifacts
```

### Weight behavior

The corruption package should default to:

```text
weight_policy = "preserve"
CASA setapply calwt = False
```

It must not call `initweights()` automatically. The simulation package already
owns the synthetic thermal-noise weight contract.

For phase-only gains, rotating circular complex Gaussian noise leaves its
variance unchanged, so preserving synthetic `SIGMA`/`WEIGHT` is consistent.

For amplitude gains applied after noise, the data and noise amplitudes are
scaled while weights remain nominal. That models an uncorrected residual gain
error. It does not model a fully self-consistent change in receiver
sensitivity. The metadata should state this distinction.

## Composition with `scripts.simulation`

### Simple supported chain

Because `SimulationResult.ms_path` is a `Path`, the independent APIs can
compose directly:

```python
from scripts.simulation import phase_center_point_source, simulate_ms
from scripts.corruption import AntennaGain, corrupt_ms

source = phase_center_point_source(template_ms, flux_jy=0.1)
simulation = simulate_ms(
    template_ms,
    [source],
    "ideal-plus-noise.ms",
    noise_model="vla-thermal",
    noise_parameters={"band": "C", "sampler": "8bit"},
    seed=100,
)

corrupted = corrupt_ms(
    simulation.ms_path,
    "ideal-plus-noise.phase-corrupted.ms",
    effects=[phase_effect],
    seed=200,
)
```

This gives an explicit lineage and leaves both the ideal simulation and the
corrupted derivative available.

### Ordering is part of the scientific model

The simple chain above performs:

```text
simulate source and noise:       DATA = sky + thermal_noise
apply multiplicative gains:      DATA = G(sky + thermal_noise)
```

For phase-only unit-magnitude gains and circular Gaussian noise, rotating the
noise does not change its distribution. This chain is suitable for the initial
phase-corruption use case.

For amplitude gains, this chain scales both the sky and already-added noise.
If the desired measurement equation is instead:

```text
DATA = G(sky) + thermal_noise
```

then the operations must be:

```text
predict ideal sky
apply multiplicative corruption
add thermal noise
initialize synthetic weights
```

The current public APIs cannot express that sequence cleanly because noise
injection and final weight initialization are internal to `simulate_ms()`.
Do not conceal this by claiming that all chaining orders are equivalent.

### Recommended composition extension

After the independent gain package is stable, add a high-level observation
builder that owns the ordered transaction. A suitable name is
`simulate_observation()` in `scripts.simulation`, with corruption specs as an
optional stage:

```python
result = simulate_observation(
    template_ms,
    components=[source],
    output_ms="source-gain-noise.ms",
    corruptions=[phase_effect, amplitude_effect],
    noise_model="vla-thermal",
    noise_parameters={"band": "C", "sampler": "8bit"},
    seed=12345,
    order=("predict", "corrupt", "noise", "weights"),
)
```

The implementation should copy the template only once and call narrow internal
operations shared by `simulate_ms()` and `corrupt_ms()`:

```text
copy template
zero/prepare visibility columns
predict components
build and apply gain corruptions
add thermal noise
initialize SIGMA/WEIGHT/WEIGHT_SPECTRUM
write one parent manifest that links all gain tables and component lists
```

Keep `simulate_ms()` as the small backward-compatible source/noise API. The
high-level orchestrator may import both packages, but the low-level corruption
package must not import experiment drivers or imaging code.

An alternative short-term flow for the physical order is:

```text
simulate_ms(..., noise_model=None)
corrupt_ms(...)
add thermal noise with a new narrowly scoped simulation API
```

If that route is chosen, expose a safe copy-producing function such as
`add_noise_ms()` rather than exposing the current private `_apply_noise()`.

### Seeds

Source prediction is deterministic, but noise and corruptions consume random
numbers. Derive independent child seeds from one experiment seed and record
them by stage:

```text
experiment seed
├── thermal-noise seed
├── corruption 0 seed
├── corruption 1 seed
└── diagnostic/resampling seed, if any
```

Changing the number of corruption effects must not silently change the thermal
noise realization.

## Migration plan

### Phase 1: stabilize pure behavior

1. Add tests that currently expose the function-protocol and `solint="int"`
   failures.
2. Replace `CorrFn.sample()/eval()` with one immutable `sample(times, rng)`
   contract.
3. Fix and validate `TimeGrid`, including integer integration times, numeric
   intervals, nearest/linear interpolation, repeated times, gaps, and empty
   inputs.
4. Define fractional-amplitude and phase-radian conventions.
5. Replace global RNG use and collision-prone group seeds.
6. Move pure modules so they import without CASA, Matplotlib, Astropy, or the
   optional fBM library.

### Phase 2: package the gain-table path

1. Replace `scripts/corruption.py` with `scripts/corruption/` atomically.
2. Move gain-table internals behind lazy CASA adapters.
3. Implement safe `corrupt_ms()` copy/refuse-overwrite behavior.
4. Make diagnostics optional and caller-directed.
5. Add structured results and atomic JSON metadata.
6. Update `corruption_pipeline.py` and the active portion of
   `corruption_gaindrift.py` to use the public API.

### Phase 3: validate with real CASA

1. Identity gain: `DATA` values remain unchanged, allowing documented CASA
   schema side effects only.
2. Constant phase on one antenna: affected baselines match the analytic complex
   multiplier; unrelated baselines are unchanged.
3. Antenna-based phase: closure phase remains unchanged.
4. Baseline-only phase test: closure changes, proving the validator is
   sensitive to the distinction.
5. Time interpolation: gain-table row values match expected sine/linear values.
6. Selection: only requested antennas, fields, SPWs, scans, and time ranges are
   changed.
7. Reproducibility: same seed gives identical tables/data; a different seed
   changes stochastic curves.
8. Input safety: source MS tree hash remains unchanged after success and
   failure.
9. Composition: simulation weights survive phase-only corruption exactly;
   flags and sampling metadata remain unchanged.

### Phase 4: ordered observation simulation

1. Refactor source prediction, corruption application, additive noise, and
   weight initialization into explicit internal stages.
2. Implement `simulate_observation()` or a safe `add_noise_ms()` bridge.
3. Test both declared equations: `G(sky + noise)` and `G(sky) + noise`.
4. Compare natural-weight noise-only images against the existing theoretical
   VLA validation before testing Briggs-weighted science images.

### Phase 5: promote additional effects selectively

Promote troposphere, leakage, bandpass, or true pointing corruption only after
each has:

```text
a typed physical configuration
documented units and CASA-version support
a declared ordering and weight policy
an analytic or trusted-reference validation
realistic parameter ranges
seed reproducibility tests
```

The existing obvious-effect scripts should remain experiments until those
requirements are met.

## Definition of done for the first package release

The initial corruption package is ready when all of the following are true:

- one public call safely creates a corrupted copy of an MS;
- phase-only and fractional-amplitude antenna gains have unambiguous units and
  equations;
- selection and time-grid behavior are deterministic and tested;
- the original MS is never modified;
- no output is silently overwritten;
- thermal weights and flags have an explicit preservation contract;
- the returned result and JSON record make the corruption fully reproducible;
- diagnostics are optional and write only beneath caller-selected output;
- the package imports under ordinary Python without CASA until an operation
  actually needs CASA; and
- a `SimulationResult.ms_path` can be passed directly to `corrupt_ms()` with a
  documented statement of whether the result represents `G(sky + noise)` or
  `G(sky) + noise`.
