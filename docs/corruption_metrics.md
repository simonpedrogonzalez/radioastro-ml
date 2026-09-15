# Detectability-controlled constant gain corruption plan

## Goal

Add a small, self-contained capability to `scripts.corruption` for creating a
constant amplitude-only or phase-only error on one antenna at a requested
visibility-space detectability.

The caller should be able to provide the existing simulated Measurement Set,
its thermal-noise level, the antenna, corruption type, sign, and target
detectability. The corruption package will calculate:

1. `epsilon_g=|g-1|` (stored as `gain_error_magnitude`);
2. `epsilon_vis`, the fractional signal perturbation across the full MS; and
3. `rho_corr`, the corruption signal relative to thermal noise.

It will then choose the required constant amplitude gain or phase offset,
construct the existing `AntennaGainCorruption`, and include the three metrics
and their readable explanation in `corruption.json` and `corruption.txt`.

Keep this implementation inside `scripts.corruption`. Do not change the
simulation, preprocessing, or imaging packages. Do not increment any schema
versions. The existing report schema simply gains an optional `metrics` field.

## Existing behavior to preserve

- `AntennaGainCorruption(timegrid, amp_fn=None, phase_fn=None, query=None)` and
  its existing build/apply methods remain compatible.
- Amplitude functions return the complete dimensionless gain magnitude:
  `1.0` is identity and `1.10` is a positive 10% amplitude error.
- Phase functions return radians.
- The selected antenna is represented in the gain-table query by
  `GCOLS.ANTENNA1 == antenna_id`. Applying that gain table affects every MS
  baseline containing the antenna.
- `write_corruption_reports()` remains the generic writer. It does not contain
  formulas for a specific corruption type.
- Corruption reports retain their current `schema_version`. Existing reports
  without metrics remain valid.
- Current simulation reports already contain `output_ms` and
  `noise.simplenoise_jy`; those values can be consumed without changing the
  simulation report.

The current constant amplitude/phase experiment has a private
`_ConstantCurve`. Replace that pattern with a public reportable `Constant`
corruption function owned by `scripts.corruption`.

## Fixed whole-MS metric semantics

A public `VisibilityMetricSelection` is not needed for this use case. The
metric definition is fixed so two reports always mean the same thing.

Every calculation uses the entire MS with these rules:

- read `DATA`;
- include all rows, fields, spectral windows, and channels;
- exclude autocorrelation rows;
- exclude `FLAG_ROW` rows and individually flagged values;
- include the parallel-hand correlations used for Stokes-I imaging: `RR`,
  `LL`, `XX`, and `YY` (CASA correlation codes 5, 8, 9, and 12);
- exclude non-finite values and samples without finite positive weights;
- use `WEIGHT_SPECTRUM` when valid, otherwise broadcast `WEIGHT` over channels;
  and
- identify affected rows with
  `(ANTENNA1 == antenna_id) | (ANTENNA2 == antenna_id)`.

There is no caller-configurable data, field, SPW, channel, or correlation
selection in the initial API. If selection is needed later, it should be a
separate extension rather than part of this task.

## Why the thermal-noise value is needed

The existing simulated MS `DATA` contains predicted sky plus thermal noise. A
raw sum of `|DATA|^2` would therefore treat noise power as source-signal power
and make the requested corruption depend incorrectly on how many noisy samples
exist.

For the current simulation, `simplenoise_jy = sigma` is the standard deviation
of each real and imaginary visibility component. Therefore one complex noise
sample contributes expected power

$$
E[|n|^2] = 2\sigma^2.
$$

Use the existing simulation noise value to estimate signal power by subtracting
that expected contribution from the aggregate observed power. Do not clip
individual samples; positive and negative fluctuations must cancel in the full
sum.

For all valid parallel-hand samples `M`, define

$$
P_{\rm all}
=
\sum_{m\in M}\left(|V_m^{\rm DATA}|^2-2\sigma^2\right).
$$

For samples `A_k` on baselines containing antenna `k`, define

$$
P_A
=
\sum_{m\in A_k}\left(|V_m^{\rm DATA}|^2-2\sigma^2\right)
$$

and

$$
P_{A,w}
=
\sum_{m\in A_k}
w_m\left(|V_m^{\rm DATA}|^2-2\sigma^2\right).
$$

The implementation must accumulate these in float64/complex128 and in row
chunks rather than loading the whole MS into memory. Reject the calculation if
`P_all`, `P_A`, or `P_A,w` is not finite and positive. A non-positive result
means the simulated source is too weak for this estimator; do not silently
clip it to zero or fall back to a different definition.

For current sigma-initialized simulations, verify that the selected MS weights
are consistent with `1 / sigma**2` within relative tolerance `1e-6`. Reject a
different weight convention in this first implementation.

This noise-debiased aggregate keeps the implementation entirely in the
corruption package and works with the existing simulation outputs. It is an
estimate from the recorded noise realization, so the report must identify the
method as `noise_debiased_data_power` and include the counts and accumulated
powers used.

## The three metrics

For complex antenna gain `g`, define

$$
\boxed{\epsilon_g=|g-1|}.
$$

This is `gain_error_magnitude`, the physical fractional displacement applied to
each affected signal visibility.

The fractional perturbation across the full MS is

$$
\boxed{
\epsilon_{\rm vis}
=
\epsilon_g\sqrt{\frac{P_A}{P_{\rm all}}}
}.
$$

The unit-gain detectability for the selected antenna is

$$
A_k=\sqrt{P_{A,w}},
$$

and corruption detectability is

$$
\boxed{
\rho_{\rm corr}=\epsilon_g A_k
}.
$$

Consequently the physical gain displacement needed for a target is

$$
\boxed{
\epsilon_g=\frac{\rho_{\rm target}}{A_k}
}.
$$

All three quantities are dimensionless and must remain separate:

- `gain_error_magnitude` describes the antenna gain;
- `epsilon_vis` describes the fractional perturbation of the full visibility
  dataset; and
- `rho_corr` describes the perturbation relative to thermal noise.

## Compact object-oriented API

Do not add four or five public functions for this one special case. Add one
parameter object, one metrics object, a public constant function, and class
constructors on the relevant existing objects.

### Parameters

Add this immutable model in `scripts/corruption/metrics.py`:

```python
@dataclass(frozen=True)
class ConstantGainParameters:
    ms: Path
    antenna_id: int
    corruption_type: Literal["amp", "phase"]
    target_rho_corr: float
    thermal_noise_jy: float
    sign: Literal[-1, 1] = 1

    @classmethod
    def from_simulation_report(
        cls,
        simulation_report,
        *,
        antenna_id,
        corruption_type,
        target_rho_corr,
        sign=1,
    ) -> "ConstantGainParameters": ...
```

The normal constructor supports callers that already have a
`SimulationResult`:

```python
parameters = ConstantGainParameters(
    ms=simulation.ms_path,
    antenna_id=antenna_id,
    corruption_type="phase",
    target_rho_corr=5.0,
    thermal_noise_jy=simulation.simplenoise_jy,
    sign=-1,
)
```

`from_simulation_report()` is only a convenience adapter. It uses the existing
simulation-report reader and extracts `output_ms` and
`noise.simplenoise_jy`. It performs no simulation calculation and requires no
change to `scripts.simulation`.

Validate finite positive noise and target values, a nonnegative integer antenna
ID, corruption type exactly `"amp"` or `"phase"`, sign exactly `-1` or `1`,
and an existing `.ms` directory.

### Metrics and readable calculation text

Add:

```python
@dataclass(frozen=True)
class DetectabilityMetrics:
    corruption_type: Literal["amp", "phase"]
    antenna_id: int
    sign: Literal[-1, 1]
    gain_error_magnitude: float
    amplitude_gain: float | None
    phase_offset_rad: float | None
    phase_offset_deg: float | None
    epsilon_vis: float
    rho_corr: float
    target_rho_corr: float
    unit_gain_detectability: float
    thermal_noise_jy: float
    valid_sample_count: int
    affected_sample_count: int
    total_signal_power: float
    affected_signal_power: float
    weighted_affected_signal_power: float
    power_estimator: str = "noise_debiased_data_power"

    def to_report_dict(self) -> dict[str, object]: ...
    def to_report_text(self) -> str: ...
    def __repr__(self) -> str: ...
```

`repr(metrics)` should be concise and diagnostic. `to_report_text()` should be
the human explanation used by the TXT report. It must show the actual formulas,
substituted aggregate values, and interpretations, for example:

```text
Detectability metrics
--------------------------------------------------------------------------------
Power estimator: noise-debiased DATA over the full unflagged MS
Thermal sigma: 0.0745356 Jy per real/imaginary component
Valid samples: 100000; affected samples: 7400

epsilon_g=|g-1|=0.100000
  Physical complex-gain displacement on affected baselines.

epsilon_vis=epsilon_g*sqrt(P_A/P_all)
            = 0.100000 * sqrt(12.5 / 171.5)
            = 0.027000
  Fractional signal perturbation across the full visibility dataset.

rho_corr=epsilon_g*sqrt(P_A,w)
         = 0.100000 * sqrt(2500.0)
         = 5.000000 (requested 5.000000)
  Injected signal perturbation relative to combined thermal noise.
```

The exact example numbers are illustrative. Keep formula names stable and
render all values with enough precision to reproduce the calculation.
`str(metrics)` may delegate to `to_report_text()` so callers can print the
explanation directly.

### Constant function

Add the public corruption function:

```python
@dataclass(frozen=True)
class Constant(CorrFn):
    value: float
    metrics: DetectabilityMetrics | None = None

    @classmethod
    def from_detectability(
        cls, parameters: ConstantGainParameters
    ) -> "Constant": ...

    def sample(self, rng, **kwargs):
        return self

    def eval(self, times): ...
    def to_report_dict(self) -> dict[str, object]: ...
    def to_report_text(self) -> str: ...
```

`Constant.from_detectability()` owns the full-MS scan, noise debiasing, target
solution, and construction of `DetectabilityMetrics`. It returns the value
expected by the current gain implementation:

- for `corruption_type="amp"`, `value` is the full gain magnitude;
- for `corruption_type="phase"`, `value` is radians.

A regular `Constant(value)` remains available for fixed-value corruptions and
has `metrics=None`.
`Constant.to_report_dict()` continues to describe only the constant
configuration (`type` and `value`); metrics are emitted once at the report
envelope level and are not duplicated inside `configuration`.

Keep CASA/table imports inside the detectability calculation path. Importing
`Constant`, `DetectabilityMetrics`, or `scripts.corruption.reporting` must not
require a CASA environment.

### Amplitude conversion

For amplitude corruption:

$$
\boxed{g_{\rm amp}=1+s\epsilon_g}.
$$

Require `g_amp > 0`; reject the negative branch when `epsilon_g >= 1`. Never
clip the gain or change the requested sign.

### Phase conversion

For phase corruption:

$$
|e^{i\phi}-1|=2\sin(|\phi|/2),
$$

so

$$
\boxed{
\phi=s\,2\arcsin(\epsilon_g/2)
}.
$$

Require `epsilon_g <= 2`, store radians in `Constant.value`, and report the
corresponding degrees. Never use the small-angle approximation in code.

### AntennaGainCorruption constructor

Add one class constructor:

```python
@classmethod
def from_detectability(
    cls,
    timegrid,
    parameters: ConstantGainParameters,
) -> "AntennaGainCorruption": ...
```

It delegates the calculation to `Constant.from_detectability(parameters)`,
puts the returned function in `amp_fn` or `phase_fn`, leaves the other `None`,
and builds the existing one-antenna query:

```python
GTabQuery().where_eq(GCOLS.ANTENNA1, parameters.antenna_id).group_by(
    [GCOLS.ANTENNA1]
)
```

The existing `__init__` signature does not change. Set
`corruption.metrics = constant.metrics` inside the class constructor; ordinary
instances have `metrics=None`.

This gives the driver one operation:

```python
parameters = ConstantGainParameters.from_simulation_report(
    source.simulation_metadata,
    antenna_id=antenna_id,
    corruption_type="phase",
    target_rho_corr=5.0,
    sign=-1,
)
corruption = AntennaGainCorruption.from_detectability(
    TimeGrid(solint="10m", interp="linear"),
    parameters,
)

print(corruption.metrics)
corruption.build_corrtable(ms, gain_table, seed=seed).apply_corrtable(
    ms, gain_table, seed=seed
)
write_corruption_reports(
    corruption,
    json_path=corruption_json,
    text_path=corruption_text,
    context=context,
)
```

No experiment should repeat the amplitude/phase formulas or implement another
private constant curve.

## Reporting without a schema-version change

Keep the current corruption report `schema_version` unchanged. Add one sibling
field to the existing envelope:

```json
{
  "schema_version": 1,
  "context": {},
  "configuration": {
    "type": "antenna_gain",
    "amplitude": null,
    "phase": {
      "type": "constant",
      "value": 0.1000417136,
      "unit": "radian"
    }
  },
  "metrics": {
    "type": "constant_one_antenna_detectability",
    "power_estimator": "noise_debiased_data_power",
    "corruption_type": "phase",
    "antenna_id": 3,
    "sign": 1,
    "thermal_noise_jy": 0.07453559925,
    "valid_sample_count": 100000,
    "affected_sample_count": 7400,
    "total_signal_power": 171.5,
    "affected_signal_power": 12.5,
    "weighted_affected_signal_power": 2500.0,
    "unit_gain_detectability": 50.0,
    "gain_error_magnitude": 0.1,
    "amplitude_gain": null,
    "phase_offset_rad": 0.1000417136,
    "phase_offset_deg": 5.73196797,
    "epsilon_vis": 0.027,
    "rho_corr": 5.0,
    "target_rho_corr": 5.0
  }
}
```

The numbers illustrate structure only. `DetectabilityMetrics.to_report_dict()`
owns the metrics representation. `write_corruption_reports()` should obtain it
from `corruption.metrics` when present and otherwise write `"metrics": null`.
The generic writer must not know the formulas or inspect amplitude/phase
internals.

For TXT, append `corruption.metrics.to_report_text()` after the existing
configuration section. A fixed-value corruption with no metrics prints a short
`Detectability metrics: not calculated` line. This makes every new report
self-explanatory without altering the schema version or changing another
package.

The existing preprocessing validator already tolerates additional top-level
fields while validating `context` and `configuration`, so it does not need to
change for this addition.

## Interpretation limitation

This `rho_corr` is the optimal aggregate visibility-space S/N of the estimated
deterministic sky-signal perturbation. It is not an image-domain artifact S/N:
in particular, `rho_corr=10` does not imply `max=max(|residual|)/sigma=10`.
It does not guarantee equal image-domain or classification difficulty.
Deconvolution and source morphology may respond differently to amplitude and
phase corruption. In the current gain-after-noise flow, amplitude corruption
also rescales noise on affected baselines while phase corruption only rotates
circular noise. The report should state this limitation in its explanatory
text.

The usefulness of the metric is that targets such as `rho_corr = 1, 2, 5, 10`
have consistent calculation semantics across observations. The resulting
physical gain values are allowed to differ between sources and antennas.

## Validation and failure behavior

Reject before creating a corruption object when:

- the MS, antenna, `DATA`, flags, correlation mapping, or usable weights are
  missing;
- the thermal sigma or target is non-finite or non-positive;
- the MS weights disagree with the supplied thermal sigma;
- no valid full-MS or affected samples remain;
- any aggregate signal power is non-finite or non-positive;
- `corruption_type` or `sign` is invalid;
- an amplitude solution would be non-positive; or
- a phase target requires `gain_error_magnitude > 2`.

Never clip a target, silently switch signs, use flagged samples, include
autocorrelations, change the fixed correlation convention, or replace a failed
noise-debiased estimate with raw noisy power.

## File changes

| File | Change |
| --- | --- |
| `scripts/corruption/metrics.py` | Add `ConstantGainParameters`, `DetectabilityMetrics`, simulation-report convenience loading, and private chunked full-MS power accumulation. |
| `scripts/corruption/functions.py` | Add `Constant` and `Constant.from_detectability()`. |
| `scripts/corruption/core.py` | Add `AntennaGainCorruption.from_detectability()` and optional `metrics` state without changing `__init__`. |
| `scripts/corruption/reporting.py` | Add the same-schema `metrics` field and delegate readable rendering to the metrics object. |
| `scripts/corruption/__init__.py` | Lazily export the new public models and `Constant`. |
| `tests/test_corruption_metrics.py` | Add calculation, constructor, and text/JSON representation tests. |
| `tests/test_corruption_package.py` | Preserve existing API/report compatibility and cover lazy exports. |

No simulation, imaging, or preprocessing package file is part of this change.
The driver example above documents the intended later call-site migration, but
modifying an experiment driver is not part of this implementation task.

## Tests

### Pure metric tests

- A small synthetic MS-table fixture with known complex values, flags,
  correlations, weights, and antenna columns produces the hand-calculated
  `P_all`, `P_A`, `P_A,w`, `A_k`, `gain_error_magnitude`, `epsilon_vis`, and
  `rho_corr`.
- Thermal noise subtraction is `2 * sigma**2` per complex value, occurs after
  summation rather than per-sample clipping, and rejects non-positive totals.
- Autocorrelations, flags, cross-hands, non-finite values, and invalid weights
  are excluded exactly once.
- `WEIGHT_SPECTRUM` and broadcast `WEIGHT` agree for uniform current
  simulations.
- Chunked accumulation matches a one-array reference calculation.
- Target values `1`, `2`, `5`, and `10` round-trip through the solver.
- Equal `gain_error_magnitude` gives equal metrics for amplitude and phase.
- Both signs work, and amplitude/phase domain failures do not clip.

### Object and reporting tests

- `Constant(value)` remains a normal fixed constant with no metrics.
- `Constant.from_detectability(parameters)` stores the correct amplitude
  magnitude or phase radians and owns a `DetectabilityMetrics` instance.
- `AntennaGainCorruption.from_detectability()` sets exactly one of `amp_fn` and
  `phase_fn` and selects exactly one antenna.
- `repr(metrics)` is useful at the console; `to_report_text()` includes all
  three formulas, substituted values, units, and interpretations.
- The unchanged-schema JSON contains strict finite metrics, and TXT contains
  the readable calculation.
- Existing callers without metrics still write a valid report with
  `metrics: null`.
- Reporting remains importable without CASA.
- Invalid calculation/report values do not leave a partial JSON/TXT pair.

### Opt-in CASA test

Use one current simulated MS and its recorded `simplenoise_jy`, construct both
an amplitude and phase corruption for the same target, and verify the generated
gain-table constants match their calculated values. Confirm that report JSON
and text contain the same three metrics and calculation explanation.

## Acceptance criteria

The task is complete when an experiment can create
`ConstantGainParameters` directly or from an existing simulation report, call
`AntennaGainCorruption.from_detectability()`, apply the returned corruption,
print a readable derivation through `corruption.metrics`, and write the same
three quantities to the existing-version corruption report. The implementation
must use the fixed full-MS convention, remain bounded-memory, preserve current
constructors and reports, and require no production-code change outside
`scripts.corruption`.
