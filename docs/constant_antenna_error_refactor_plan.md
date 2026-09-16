# Constant antenna phase/amplitude simulation refactor plan

## Purpose and status

This document is an implementation plan, not an implementation. It reconciles
the current repository code with the updated scientific definition in:

- `/Users/u1528314/Documents/Obsidian Vault/Notes/Constant Antenna Phase and Amplitude Error Simulation.md`

The one-source restriction below was a validation stage. After that run passed,
the dataset driver was expanded to every completed source in all three fixed
preprocessing partitions.

The inspected implementation includes:

- `scripts/corruption/`, especially `metrics.py`, `functions.py`, `core.py`,
  `reporting.py`, `tables.py`, `timegrid.py`, and `diagnostics.py`;
- `scripts/simulation/`, especially `simulations.py`, `noise.py`, and
  `reporting.py`;
- `scripts/create_dataset_v1.py` (the repository does not contain a file named
  `create_Dataset_1.py`);
- `scripts/reporting/create_dataset_v1.qmd`;
- the associated corruption and simulation tests; and
- the existing corruption design documents.

The Obsidian note is treated as the updated experiment specification. In this
document, **clean visibility** always means the noiseless, uncorrupted sky
visibility, not a CLEAN image and not `DATA` after thermal noise has been added.

## Updated scientific contract

For every valid visibility sample $m$, retain the following distinction:

\[
V_m = \text{noiseless, uncorrupted sky visibility},
\]

\[
V_m^{\mathrm{corr}}=C(V_m),
\qquad
\Delta V_m=V_m^{\mathrm{corr}}-V_m,
\]

and only then add thermal noise:

\[
V_m^{\mathrm{obs}}=V_m+n_m,
\qquad
V_m^{\mathrm{obs,corr}}=V_m^{\mathrm{corr}}+n_m.
\]

The two canonical corruption metrics are:

\[
\epsilon_{\mathrm{vis}}
=\frac{\lVert\Delta\mathbf V\rVert_2}{\lVert\mathbf V\rVert_2}
=\sqrt{\frac{\sum_m|\Delta V_m|^2}{\sum_m|V_m|^2}},
\]

\[
\mathrm{SNR}_{\mathrm{corr}}
=\left\lVert\frac{\Delta\mathbf V}{\boldsymbol\sigma}\right\rVert_2
=\sqrt{\sum_m\frac{|\Delta V_m|^2}{\sigma_m^2}}.
\]

For one antenna $k$ with a constant gain $g$, let $A_k$ be the set of
valid visibility samples on baselines containing that antenna and define

\[
\epsilon_g=|g-1|.
\]

Then

\[
\mathrm{SNR}_{\mathrm{corr}}
=\epsilon_g
\sqrt{\sum_{m\in A_k}\frac{|V_m|^2}{\sigma_m^2}},
\]

so the requested gain-error magnitude is

\[
\epsilon_g
=\frac{\mathrm{SNR}_{\mathrm{target}}}
{\sqrt{\sum_{m\in A_k}|V_m|^2/\sigma_m^2}}.
\]

The physical constant is then

\[
g_{\mathrm{amp}}=1+s\epsilon_g,
\qquad
\phi=s\,2\arcsin(\epsilon_g/2),
\qquad s\in\{-1,+1\}.
\]

The implementation must follow the note's convention that `simplenoise` is
$\sigma_m$, the standard deviation of each real and imaginary component. No
extra factor of two belongs in the stated `SNR_corr` equation.

## Current implementation versus required behavior

| Concern | Current implementation | Required behavior |
| --- | --- | --- |
| Metric input | Reads noisy `DATA` and estimates sky power by subtracting $2\sigma^2$ per complex sample. | Read the exact noiseless $V$; do not estimate it from $V+n$. |
| Corruption order | Copies a source-plus-noise MS and applies the gain, producing $g(V+n)$ on affected baselines. | Apply corruption to $V$, then add noise, producing $gV+n$. |
| Baseline branch | Reuses images and QA from an already noisy simulation. | Rebuild from the same noiseless $V$, then add the paired noise realization. |
| `epsilon_vis` | Uses `epsilon_g*sqrt(P_A/P_all)` where both powers are noise-debiased estimates. | Compute the norm ratio from exact $V$ and $\Delta V$. |
| Corruption S/N | Named `rho_corr`; based on noise-debiased weighted source power and forced to equal its target by construction. | Name `SNR_corr`; use $\sqrt{\sum |\Delta V_m|^2/\sigma_m^2}$, and independently verify it after applying the corruption. |
| Noise behavior | Amplitude gain rescales affected noise and phase gain rotates it. | The same thermal-noise process is added after either corruption, so noise itself is not corrupted. |
| Provenance | Reports `noise_debiased_data_power` and the gain-after-noise limitation. | Report exact noiseless/difference norms and the explicit `predict -> corrupt -> noise -> image` order. |

# A. Functional refactor (required)

## A1. Split the simulation into explicit scientific stages

The simulation package needs public operations that make the order impossible
to misunderstand:

1. Construct a source-only MS containing exact $V$.
2. Optionally copy it and apply $C(V)$, producing exact
   $V^{\mathrm{corr}}$.
3. Add thermal noise to the selected source-only or corrupted MS.
4. Initialize `SIGMA`, `WEIGHT`, and `WEIGHT_SPECTRUM` for that thermal-noise
   model.
5. Image only the final observed MS.

`simulate_ms(..., noise_model=None)` already provides most of stage 1, but
noise application is currently a private helper tied to the open simulator
inside `simulate_ms`. Extract a deliberately named public operation such as
`add_thermal_noise_inplace(...)`, or a copy-producing equivalent. If an
in-place API is selected, its name and documentation must make mutation
explicit.

Do not move thermal-noise generation into the corruption package. Noise remains
owned by `scripts.simulation`; the experiment driver owns the ordering of the
simulation and corruption stages.

## A2. Build every branch from one noiseless source MS

For the one-source development run, create one temporary source-only MS per
astronomical source. Treat the existing matched-thermal run as a recipe for:

- the original sampling/template MS;
- the component definition and source flux;
- the selected thermal-noise model and resolved `simplenoise_jy`; and
- the imaging and metric-region configuration.

Do **not** use its already noisy `simulation_ms` as the new corruption input.
The current retained thermal MS cannot recover the exact $V$ by subtracting an
expected noise power.

The baseline must also be regenerated and re-imaged. Copying its old FITS/QA
products would compare a different execution path and potentially a different
noise realization with the new variants.

## A3. Use paired thermal noise

The note writes the same $n_m$ in both observed equations. Implement that as
a common-random-number design:

- derive one `thermal_noise_seed` from the source ID;
- use that same seed and identical noise parameters for the baseline and every
  amplitude/phase variant of that source; and
- keep `corruption_seed` separate from `thermal_noise_seed`.

This ensures

\[
V^{\mathrm{obs,corr}}-V^{\mathrm{obs}}=\Delta V
\]

up to numerical/CASA execution effects, rather than mixing corruption with a
difference between independent noise realizations. Add a CASA integration test
that explicitly verifies the equality of the recovered noise arrays:

\[
V^{\mathrm{obs}}-V
=V^{\mathrm{obs,corr}}-V^{\mathrm{corr}}.
\]

If CASA does not reproduce the same noise array solely from the same seed,
generate one noise realization once, retain it transiently, and add that array
to every branch in bounded chunks.

## A4. Replace noise-debiased power estimation with exact visibility norms

Remove the current calculation based on

```text
sum(|DATA|^2 - 2*sigma^2)
```

and remove the `noise_debiased_data_power` estimator label. The target solver
must scan the source-only $V$, applying one documented validity mask:

- unflagged samples only, including `FLAG_ROW`;
- cross-correlations only unless a future experiment explicitly includes
  autocorrelations;
- supported parallel hands only for the current Stokes-I experiment;
- finite complex values only; and
- only rows whose baseline contains antenna $k$ for the affected sum.

Accumulate quantities named directly after the equation. The public/result
values should be the norms, while a chunked implementation may use `_sq`
accumulators internally:

```text
V_L2_sq = sum_all(|V|^2)
V_L2 = sqrt(V_L2_sq) = ||V||_2

V_Ak_L2_sq = sum_Ak(|V|^2)
V_Ak_L2 = sqrt(V_Ak_L2_sq) = ||V_Ak||_2

V_Ak_over_sigma_L2_sq = sum_Ak(|V / sigma|^2)
V_Ak_over_sigma_L2
    = sqrt(V_Ak_over_sigma_L2_sq)
    = ||V_Ak / sigma_Ak||_2
```

For the current homogeneous thermal-noise model, $\sigma_m$ is the recorded
`simplenoise_jy`. The general calculation should accept a scalar now without
inventing a varying-noise abstraction; it can later accept an array/provider
when a validated heterogeneous-noise model exists.

## A5. Solve the constant gain from the updated equation

Use

```text
eps_g = SNR_corr_target / V_Ak_over_sigma_L2
```

Then preserve the existing physical conversions and domain checks:

- amplitude: `g_amp=1 + sign*eps_g`, requiring a positive gain
  under the current magnitude convention;
- phase: `phi_rad=sign*2*asin(eps_g/2)`, requiring `eps_g <= 2`, with
  `phi_deg=degrees(phi_rad)` only as a display convenience; and
- amplitude and phase variants at the same `SNR_corr_target` must have the same
  $\epsilon_g=|g-1|$.

For the constant one-antenna special case, the predicted fractional
perturbation is exactly

\[
\epsilon_{\mathrm{vis}}
=\epsilon_g
\sqrt{\frac{\sum_{m\in A_k}|V_m|^2}{\sum_m|V_m|^2}}.
\]

The generic definitions should nevertheless remain expressed through
$\Delta V$, so they extend correctly to time-variable gains, multiple
antennas, and other corruption families.

## A6. Independently measure the applied corruption

The current `rho_corr` always equals its target because both are calculated
from the same analytic expression. That is a solver round-trip, not evidence
that CASA applied the requested corruption correctly.

Before adding noise, compare the source-only and corrupted MSs chunk by chunk
and calculate:

```text
Delta_V = V_corr - V
Delta_V_L2 = sqrt(sum(|Delta_V|^2))
Delta_V_over_sigma_L2 = sqrt(sum(|Delta_V / sigma|^2))

eps_vis = Delta_V_L2 / V_L2
SNR_corr = Delta_V_over_sigma_L2
```

Report both requested/predicted and measured values, plus absolute or relative
closure errors such as:

```text
SNR_corr_target
SNR_corr_expected
SNR_corr
SNR_corr_relerr
eps_vis_expected
eps_vis
```

The run should fail if the measured result differs from the predicted result
beyond a tight, explicitly tested numerical tolerance. This check will also
catch antenna selection, conjugation, interpolation, flag, and correlation
mapping errors.

## A7. Adopt the updated names and notation everywhere

Use the note's notation in Python, JSON, text, HTML, plot labels, sample IDs,
tests, and documentation. Because the semantics also change, this should be a
clean schema break rather than aliases that make old and new results look
equivalent.

| Current name | Updated name |
| --- | --- |
| `rho_corr` | `SNR_corr` |
| `target_rho_corr` | `SNR_corr_target` |
| `unit_gain_detectability` | `V_Ak_over_sigma_L2` |
| `TARGET_DETECTABILITIES` | `SNR_CORR_TARGETS` |
| `DetectabilityMetrics` | `CorruptionMetrics` |
| `DetectabilityMetricDefinition` | `CorruptionMetricDefinition` |
| `detectability_metric_definitions()` | `corruption_metric_definitions()` |
| `from_detectability(...)` | a dedicated `solve_constant_gain(...)` function |
| `amp_rho_10` / `phase_rho_10` | `amp_snr_10` / `phase_snr_10` |
| `P_all` | `V_L2_sq` or `V_L2`, as appropriate |
| `P_A` | `V_Ak_L2_sq` or `V_Ak_L2`, as appropriate |
| `P_A,w` | `V_Ak_over_sigma_L2_sq` or `V_Ak_over_sigma_L2` |
| `gain_error_magnitude` | `eps_g` |
| `epsilon_vis` | `eps_vis` |
| `amplitude_gain` | `g_amp` |
| `phase_offset_rad` / `phase_offset_deg` | `phi_rad` / `phi_deg` |

The casing is intentional: `V`, `Delta_V`, and `SNR_corr` visually match the
mathematical symbols, while suffixes identify operations or qualifiers:
`_L2`, `_sq`, `_target`, `_expected`, and `_relerr`. In prose and equations,
use $A_k$ only for the **set of affected samples**, never as a power or norm.

## A8. Update report schemas and provenance

Bump the corruption report schema version because both field names and
scientific meanings change. Bump the simulation report schema if the staged
operation/provenance structure changes.

The machine-readable report should record at least:

- exact canonical metric definitions and LaTeX;
- `visibility_source="noiseless_predicted_DATA"`;
- the validity/selection policy and sample counts;
- the three descriptive source norms from A4;
- requested, predicted, and measured corruption metrics from A6;
- the physical gain/phase solution;
- separate corruption and thermal-noise seeds;
- the explicit operation order;
- whether the same noise realization was verified across branches; and
- the source-only, corrupted-pre-noise, and final-observed artifact roles and
  lifecycles.

Delete the obsolete report warning that amplitude corruption rescales noise
and phase corruption rotates noise. Under the corrected flow neither statement
describes the generated observed data.

## A9. Update `create_dataset_v1.py`

Change the driver flow for its one selected source to:

```text
reconstruct/predict source-only V once
|
+-- baseline: copy V -> add common n -> image -> finalize
|
+-- each variant:
    copy V -> solve gain from V -> build/apply corruption -> measure Delta V
    -> add common n -> image -> finalize
```

Specific changes:

1. Replace `ThermalSource.simulation_ms` as the working input with a
   source-only MS and retain the thermal run only as configuration/provenance.
2. Rebuild `_finalize_baseline()` instead of exporting the existing thermal
   run's FITS and QA files.
3. In `_finalize_variant()`, solve from the source-only MS, corrupt before
   calling the simulation package's noise stage, and report metrics before
   cleanup.
4. Use one source-level `thermal_noise_seed`; do not derive it from the variant
   suffix.
5. Rename the target constants, variant fields, label names, sample suffixes,
   console output, manifest keys, plot titles, and diagnostic captions.
6. Record the corrected simulation policy as
   `source-only -> optional corruption -> shared thermal noise -> imaging`.
7. Keep the current one-source restriction, imaging configuration, shared
   per-channel color scales, annular image metrics, and corruption diagnostic
   plots.
8. Reassess the numeric target ladder only after the corrected one-source run.
   `SNR_corr` is still an aggregate visibility-domain metric and is not expected
   to equal an image residual statistic such as
   `max=max(|residual|)/sigma`.

## A10. Update the Quarto report

In `scripts/reporting/create_dataset_v1.qmd`:

- replace every `rho_corr` label/key with `SNR_corr`;
- show `SNR_corr_target`, `SNR_corr_expected`, and the independently measured
  `SNR_corr` distinctly;
- show `eps_vis_expected` and the independently measured `eps_vis`;
- display the canonical equations directly from package-owned definitions;
- explain that $V$ is noiseless sky visibility and
  $\Delta V=V^{\mathrm{corr}}-V$, both evaluated before noise;
- state the branch equations $V+n$ and $V^{\mathrm{corr}}+n$;
- state whether a common noise realization was verified;
- retain the current imaging-package plots, shared color scales, beam glyphs,
  and metric annuli; and
- retain the gain-function diagnostic plots, but label them with
  `SNR_corr_target`.

## A11. Replace the tests that encode the old semantics

The current hand-calculated tests deliberately construct noisy `DATA`, subtract
$2\sigma^2$, and assert the `P_A` formulas. Replace them with exact noiseless
fixtures and analytical checks for:

1. direct $\epsilon_{\mathrm{vis}}$ and `SNR_corr` from $V$ and
   $\Delta V$;
2. the constant one-antenna reduction;
3. identical amplitude/phase $\epsilon_g$ at a shared target;
4. phase conversion at finite angles, not only the small-angle limit;
5. flags, `FLAG_ROW`, autocorrelations, cross-hands, DDIDs, chunk boundaries,
   and antenna selection;
6. scalar $\sigma_m$ and a future-ready toy nonuniform-$\sigma_m$ calculation
   if that interface is introduced;
7. predicted-versus-measured agreement after writing/applying a CASA gain
   table;
8. corruption-before-noise stage order;
9. identical paired noise in the baseline and corrupted branches;
10. unchanged `SIGMA`/weights across variants after the common noise stage;
11. updated schema and canonical formula text; and
12. one-source end-to-end creation, imaging, plotting, finalization, and
    cleanup.

Add a regression assertion demonstrating the old bug:

```text
old: g*(V+n) = gV + gn
new: g*V+n
```

For nonzero noise and $g\ne1$, the test must prove these arrays differ and
that the new pipeline produces the second expression.

## A12. Invalidate old generated results

Existing dataset-v1 calibration artifacts were created with the old metric
estimator, old naming, and gain-after-noise order. They must not be mixed with
new samples or presented as comparable results. Mark their manifests/reports
as superseded or generate the corrected experiment in a new output directory
with bumped schemas and new sample labels. Do not silently resume an old output
directory under the new code.

Update or supersede `docs/corruption_metrics.md`, whose current formulas and
implementation instructions describe the obsolete noise-debiased approach.

# B. Code refactor

## Section 1 — Required cleanup of the recently added metrics, reporting, and constant-gain path

### B1.1. Keep scientific calculation separate from CASA I/O and presentation

`scripts/corruption/metrics.py` currently combines parameter validation,
simulation-report loading, correlation mapping, CASA table scanning, formula
definitions, metric storage, and a long text renderer. This makes it hard to
see which equation is actually being implemented.

Refactor into three small responsibilities:

- a bounded-memory visibility reader/accumulator that returns explicitly named
  exact norms;
- pure functions that solve a constant gain and calculate metrics from arrays
  or accumulated norms; and
- report adapters that serialize already calculated configuration, metrics,
  and provenance.

The pure solver should be testable without CASA. Its code should visually
match the equation in the note.

### B1.2. Remove experiment I/O from `Constant`

`Constant.from_detectability()` makes a generic scalar corruption function
open an MS, calculate source power, solve an experiment target, and construct a
metrics object. That is too much responsibility for `Constant`.

Keep `Constant(value)` as a small corruption-function configuration. Move the
target-S/N solve to a dedicated constant-gain service/module, for example:

```text
measure_constant_gain_basis(source_ms, antenna_id, sigma)
solve_constant_gain(SNR_corr_target, V_Ak_over_sigma_L2, corruption_type, sign)
```

The result can then be passed explicitly into `AntennaGainCorruption`.

### B1.3. Replace mutable, hidden metrics attachment with an explicit result

`AntennaGainCorruption.__init__()` sets `self.metrics=None`, and the alternate
constructor later mutates it. `write_corruption_reports()` then discovers that
state with `getattr`. This implicit side channel makes configuration and run
results easy to confuse.

Keep the scientific values explicit without creating a hierarchy of run-result
wrappers. One small immutable solution object and one metrics object are
enough:

```text
ConstantGainSolution
    eps_g
    g_amp or phi_rad
    SNR_corr_target
    SNR_corr_expected
    eps_vis_expected

CorruptionMetrics
    SNR_corr
    eps_vis
```

Pass these objects explicitly between the driver, corruption construction, and
report writer. Do not attach them to `Constant` or discover them through
`getattr`. Paths are already owned by the caller and do not need another result
object. A fixed `Constant` with no target should remain reportable without
pretending that metrics were calculated.

### B1.4. Make canonical metric definitions data, not hand-built prose

Retain the useful idea of one package-owned definition per metric, but change
the definitions to the updated formulas. The stable representation should be:

```text
eps_g=|g-1|
eps_vis=||Delta_V||_2/||V||_2
SNR_corr=||Delta_V/sigma||_2
```

Keep plain-text formula, LaTeX, one-sentence description, and unit in each
definition. JSON and Quarto should consume these definitions. Do not manually
restate alternate formulas in each renderer.

The constant-gain reductions may be reported as derivation details, but must
not replace the general canonical definitions.

### B1.5. Make report rendering a reporting concern

`CorruptionMetrics.to_report_text()` currently owns layout, headings,
substituted numerical derivations, interpretation prose, and formatting. Keep
the metrics object as immutable scientific data with a concise `repr`; move the
full text layout to `scripts/corruption/reporting.py` or a dedicated renderer.

Likewise, make `write_corruption_reports()` accept explicit configuration,
metrics, and context/result objects rather than inspecting an optional
`.metrics` attribute. The generic strict-JSON and atomic-write behavior should
be retained.

### B1.6. Use one vocabulary for configuration, prediction, measurement, and output

Do not overload `detectability` for a target, a norm, and a measured metric.
Use:

- `SNR_corr_target` for the requested control variable;
- `V_Ak_over_sigma_L2` for the solver denominator;
- `SNR_corr_expected` and `eps_vis_expected` for analytic values;
- `SNR_corr` and `eps_vis` for values measured from
  $\Delta V=V^{\mathrm{corr}}-V$; and
- `image_*` or the existing image metric names for post-imaging outcomes.

This also avoids the current report ambiguity where the displayed
`rho_corr` is merely the target reconstructed from its own solution.

### B1.7. Remove obsolete complexity instead of preserving it as fallback

Once all callers use exact noiseless visibilities, delete:

- aggregate thermal-noise subtraction;
- `total_observed_power` and `affected_observed_power` intermediates;
- the positive-after-debiasing failure mode;
- exact equality checks between every MS weight and
  `1/thermal_noise_jy**2` as a prerequisite for source-power estimation;
- `power_estimator="noise_debiased_data_power"`; and
- the gain-after-noise interpretation warning.

Weights may still be validated as output metadata, but they should not be used
to reconstruct a signal that the pipeline can retain exactly.

### B1.8. Make the simulation report express stage order

The current simulation report has a flat operations list, and each corruption
sample copies the source run's simulation report even after modifying the MS.
That copied report no longer describes the actual sample history.

Create a sample-specific provenance chain containing the source prediction,
optional corruption application, thermal-noise addition, weight
initialization, and imaging input. Report each stage's input/output role and
seed. Do not present the original thermal-run report as if it described the
new variant MS.

### B1.9. Recommended file boundary

A small, direct layout is preferable to adding generic frameworks:

```text
scripts/corruption/
    metrics.py          canonical definitions and pure metric calculations
    visibility.py       bounded CASA reads and exact V/Delta-V accumulators
    constant_gain.py    target-SNR basis and pure amplitude/phase solver
    core.py             corruption configuration and gain-table execution
    reporting.py        schemas and JSON/text renderers
```

If `visibility.py` would contain only one short reader, keep it in `metrics.py`.
The goal is visible scientific logic, not the maximum number of modules.

## Section 2 — Optional cleanup of the core and corruption functions

These are the optional changes worth implementing after the functional
correction and its regression tests. They must preserve the generalized core:
arbitrary `CorrFn` implementations, `TimeGrid`, flexible `GTabQuery`
selection/grouping, multiple antenna groups, and extension to additional
corruption families.

### B2.1. Make corruption functions immutable specifications

`RandomPhaseMaxSineWave.sample()` and `fBM.sample()` mutate and return the same
object. Reusing one instance across groups can leak sampled state. Prefer:

```text
specification.sample(rng, times) -> immutable realization
realization.evaluate(times) -> values
```

Use only the passed `numpy.random.Generator`; remove the global
`np.random.seed(...)` side effect in the fBM path.

### B2.2. Normalize the `CorrFn` protocol

The base class and subclasses currently disagree about whether `eval` accepts
`rng`, while diagnostics call it without `rng`. Define one signature for
sampling and one for evaluating a sampled realization, then make every built-in
function follow it. Add protocol/conformance tests.

### B2.3. Fix `TimeGrid` before expanding its use

The existing design document correctly notes that `solint="int"` is broken:
`build_corrtable()` calls `full_grid()` before reaching its special-case code,
and `full_grid()` rejects `"int"`. Also, the type annotation declares only
`Literal["linear"]` even though runtime code accepts `"nearest"`.

Make `TimeGrid` frozen and validated, represent parsed intervals explicitly,
support both interpolation modes in its type, and test boundary knots and
per-integration behavior.

### B2.4. Make mutation steps explicit without adding result wrappers

Both `build_corrtable()` and `apply_corrtable()` mutate external state and
return `self`. The fluent call hides the boundary between table creation and MS
mutation, but a new result-object hierarchy would be unnecessary bloat.

Use the minimal change:

```python
corruption.build_corrtable(ms, gain_table, ...)
corruption.apply_corrtable(ms, gain_table, ...)
```

Stop chaining in supported callers, change the two methods to return `None`
when old callers have been migrated, and let the caller continue to own the
paths it supplied. The solution and metrics are already represented by the
small scientific objects described in Section 1; do not duplicate paths,
configuration, or metrics in another execution-result wrapper.

### B2.5. Remove incidental output

Remove incidental console output such as
`Grup:`. Progress messages should come from the experiment driver.

## Recommended implementation order and gates

### Required file-impact checklist

| File | Required change |
| --- | --- |
| `scripts/simulation/noise.py` | Expose a supported post-corruption thermal-noise operation while keeping noise normalization and weight initialization in the simulation package. |
| `scripts/simulation/simulations.py` | Support an explicit source-only stage and record the corrected ordered provenance; avoid forcing prediction and noise into one inseparable operation. |
| `scripts/simulation/reporting.py` | Render the ordered stages and bump the schema if its structured representation changes. |
| `scripts/simulation/__init__.py` | Export the selected public staged-simulation/noise API. |
| `scripts/corruption/metrics.py` | Replace noise debiasing and `P_*` aggregates with exact $V$/$\Delta V$ norms, equation-shaped names, canonical definitions, and expected/measured result records. |
| `scripts/corruption/functions.py` | Remove the MS-reading target solver from `Constant`; keep it as a scalar function specification. |
| `scripts/corruption/core.py` | Accept an explicit solved constant, stop owning metrics through mutable hidden state, and keep the generalized corruption scheme. |
| `scripts/corruption/reporting.py` | Bump the report schema and render explicit configuration plus predicted/measured results without `getattr(..., "metrics")`. |
| `scripts/corruption/__init__.py` | Replace old detectability exports with the updated metric/solver/result API. |
| `scripts/create_dataset_v1.py` | Rebuild the baseline and variants from source-only $V$, corrupt before noise, reuse one noise seed, rename labels/keys, and measure the applied corruption. |
| `scripts/reporting/create_dataset_v1.qmd` | Use the new formulas, names, provenance, predicted/measured values, and shared-noise statement. |
| `tests/test_corruption_metrics.py` | Replace noise-debiased fixtures with exact analytical norms and add independent applied-corruption validation. |
| `tests/test_corruption_package.py` | Update public API/schema contracts and remove tests that require hidden mutable metrics attachment. |
| `tests/test_simulation_package.py` | Test the staged noise API, operation ordering, weights, and deterministic paired noise. |
| `docs/corruption_metrics.md` | Supersede the old plan or rewrite it to match this document and the Obsidian note. |

The legacy research scripts should not be mass-edited during the scientific
correction. Update only callers that use the supported API, and handle the
remaining legacy scripts in a separate compatibility decision after the new
one-source run passes.

1. **Lock the equations:** add pure analytical tests using exact $V$,
   $\Delta V$, and scalar $\sigma$.
2. **Stage the simulation:** expose source-only prediction and post-corruption
   noise application; verify common noise with CASA.
3. **Replace the solver:** calculate the constant gain from exact noiseless
   norms and update names.
4. **Measure the application:** compare clean and corrupted pre-noise MSs and
   fail on target mismatch.
5. **Update schemas/reporting:** bump versions and render only canonical names
   and formulas.
6. **Update the one-source driver:** regenerate baseline and variants through
   the corrected stage order.
7. **Run one-source validation:** inspect numerical target agreement, noise
   equality, gain plots, shared-scale images, and image metrics.
8. **Choose the production target ladder:** adjust only after seeing the
   corrected images and metrics.
9. **Apply the Section 2 cleanup:** keep each change separately tested and
   behavior-preserving.

The corrected one-source run is complete only when the report demonstrates all
of the following:

```text
V is exact and noiseless
Delta_V is measured before noise
SNR_corr ~= SNR_corr_target
eps_vis ~= eps_vis_expected
noise_baseline == noise_variant
operation order == predict -> corrupt (optional) -> noise -> image
all image variants use the same per-channel display scale
```
