# Simulation Package Implementation Plan

## Goal

Create a small reusable package for controlled visibility simulation on the
sampling of an existing Measurement Set (MS). The first supported experiment is
a constant-spectrum, unpolarized point source at the phase centre, optionally
with VLA-like Gaussian visibility noise. The package must also measure image
noise and provide reproducible validation against both the VLA sensitivity
equation and the stored 0012-399 VLA Pipeline result.

The package is intentionally narrow. It is not a general sky-model framework,
an atmospheric simulator, or a replacement for the imaging package.

## Decisions validated before implementation

The following decisions are based on the repository code, the referenced vault
notes, the 0012-399 products, CASA documentation, and the installed VLA Pipeline
implementation.

1. Use the package name `scripts/simulation/`, singular. The repository already
   contains [scripts/simulations.py](../scripts/simulations.py), so creating a
   `scripts/simulations/` package would introduce an ambiguous module/package
   name and could break existing imports.
2. A point-source simulation does not require an image model. The source of
   truth is a CASA component-list table (`*.components.cl`), and
   `simulator.predict(complist=...)` consumes it directly. A later `tclean`
   `.model` or `.model.tt0` is a reconstructed imaging product, not the input
   sky model. `MODEL_DATA` is not the stored truth either: CASA documents that
   `predict()` sets it to one after writing the prediction into `DATA`.
3. Do not hide imaging inside source construction. Image RMS cannot be computed
   from an MS alone; it requires an image and a defined region or mask. The S/N
   source builder therefore receives an already measured RMS as a number.
4. Do not add a source-configuration class. Two concise constructors with the
   shared `phase_center_point_source...` prefix are enough for the initial fixed
   source family.
5. Do not add `utils.py` initially. Path resolution already exists in
   [scripts/imaging/metadata.py](../scripts/imaging/metadata.py), and the
   remaining operations are clearest as direct CASA calls in `simulations.py`.
   Add a utility module only when a real second caller appears.
6. Select noise with a short string and one parameter mapping; do not introduce
   a noise-model protocol or class hierarchy. The initial supported selectors
   are `"vla-thermal"`, which performs the package's VLA calculation and then
   uses CASA `simplenoise`, and `"simplenoise"`, which passes a caller-supplied
   sigma to CASA. Reserve the CASA spellings `"tsys-atm"` and `"tsys-manual"`
   for later support without changing the `simulate_ms()` signature. Do not
   implement gain corruption or frequency-dependent SEFD interpolation in this
   phase.
7. Include a versioned copy of the NRAO 2026A OSS fiducial SEFD table and the
   documented sampler efficiencies in `noise.py`. The `"vla-thermal"` branch
   may use a band name and the unambiguous 8-bit value. It must still accept
   explicit
   `sefd_jy` and `eta_c` overrides because the MS does not encode weather,
   elevation, sampler choice reliably, or the Exposure Calculator's complete
   frequency-dependent SEFD curve. For 3-bit data, whose documented efficiency
   is a range, require an explicit efficiency instead of silently choosing one.
8. The first theoretical-noise implementation supports one homogeneous SPW.
   If `EFFECTIVE_BW` varies by channel, `EXPOSURE` varies by selected row, an
   autocorrelation is present, or more than one SPW is selected, fail with a
   useful error. Do not silently average physically different samples. Per-SPW
   or per-row noise grouping is a later feature.
9. Replace weights inherited from the copied observation in both noisy and
   noiseless outputs. For constant `simplenoise`, write only
   `SIGMA = sigma_simple` in bounded chunks and then call CASA
   `initweights(vis=..., wtmode="sigma", dowtsp=True)`. CASA, rather than our
   code, derives `WEIGHT` and uniform `WEIGHT_SPECTRUM`. For a noiseless output,
   call `initweights(..., wtmode="ones", dowtsp=True)` to establish neutral,
   finite bookkeeping weights. Do not encode zero uncertainty as `SIGMA=0`,
   because that would imply infinite weights. CASA removes `SIGMA_SPECTRUM` in
   both non-channelized modes; accept that documented schema instead of manually
   recreating the column. Preserve the original flags.
10. Keep two explicitly different image-noise operations:

    - a classical PB-region RMS/scaled-MAD measurement for validating Gaussian
      noise; and
    - a VLA-Pipeline-compatible annulus RMS for regression against the stored
      Pipeline value.

    They must not share one ambiguous name because the Pipeline statistic is
    not simply the raw `0.2 <= PB <= 0.3` RMS described in the vault note.
11. Keep the simulation flow linear: copy, write components, predict, select
    the noise branch, call CASA, close the simulator, initialize weights, and
    write run metadata. Keep the short selector dispatch together in
    `noise.py`; do not introduce registries, runners, backends, protocols, or
    model objects around the direct CASA calls.

### Corrections to assumptions in the draft

These corrections are important implementation requirements, not optional
refinements.

- `point_source_for_snr(ms, rms_metric_function=...)` is not a sound API. An MS
  is insufficient input to an image RMS function. Use
  `phase_center_point_source_from_snr(ms, snr, image_rms_jy_per_beam)`. This
  shares a prefix with `phase_center_point_source()` because both construct the
  same source; only the way flux is supplied differs.
- The 0012-399 Pipeline regression fixture needs the non-PB-corrected image,
  matching PB, and clean mask. Copying only the image and `qa.json` cannot
  reconstruct the Pipeline's pixel selection.
- The stored JSON contains the rounded Pipeline value `0.00024 Jy/beam`. The
  producing log contains the full value
  `0.00024389197254258183 Jy/beam`; preserve both.
- The vault note is partly wrong where it says the VLA Pipeline statistic is
  the ordinary un-clipped RMS over a fixed 0.2--0.3 annulus. Those PB limits are
  Pipeline defaults and remain a useful generic diagnostic, but the exact
  Pipeline operation also uses a clean-mask exclusion and CASA Chauvenet
  rejection, and it adapts the PB limits for small images. For the stored 256 by
  256 image, the adaptive heuristic selected approximately
  `0.9934094548 < PB < 0.9947110415`, excluded pixels inside the clean mask,
  and ran CASA's Chauvenet statistic for up to five iterations. The generic
  metric and exact compatibility metric therefore remain separate.
- [scripts/sim_utils.py](../scripts/sim_utils.py) is a behavioral reference,
  not reusable package code. It overwrites existing destinations through the
  legacy `copy_ms`, zeroes more columns than the new flow needs, and currently
  calls `_write_component_list(zv, ...)` with an undefined name. Do not import
  its `simulate_ms()` into the new package.

## Package and test layout

```text
scripts/simulation/
├── __init__.py          # Small public API only
├── components.py        # Phase-centre metadata and the two source builders
├── noise.py             # Noise selector, equations, and VLA reference snapshot
├── rms_metrics.py       # Classical and Pipeline-compatible image statistics
└── simulations.py       # SimulationResult and the linear simulate_ms flow

tests/
├── test_simulation_package.py
├── test_simulation_0012_399_integration.py
├── validate_vla_noise_0012_399.py  # Fixed, directly runnable validation
├── vla_noise_validation_report.qmd # Small Quarto report template
└── fixtures/simulation/vla_pipeline_0012_399/
    ├── image/            # Copied iter1.image.tt0 CASA table
    ├── pb/               # Copied iter1.pb.tt0 CASA table
    ├── clean_mask/       # Copied central_clean_0.mask CASA table
    ├── qa.json           # Unmodified producing QA JSON
    └── reference.json    # Exact value, source paths, thresholds, versions
```

Do not create `config.py`, `models.py`, or `utils.py` for this first package.
The selector and calculations belong in `noise.py`; the result dataclass belongs
beside `simulate_ms()`.

## Public API

Export only the following from `scripts/simulation/__init__.py`:

```python
from scripts.simulation import (
    RMSMetrics,
    SimulationResult,
    get_phase_center,
    get_reference_frequency,
    measure_pb_region,
    phase_center_point_source,
    phase_center_point_source_from_snr,
    simulate_ms,
    simplenoise_from_image_rms,
    theoretical_vla_simplenoise,
    vla_pipeline_annulus_rms,
)
```

The intended first end-to-end use is:

```python
from scripts.simulation import (
    phase_center_point_source,
    simulate_ms,
)

ms = "collect/extracted/0012-399/0012-399/0012-399.ms"
source = phase_center_point_source(ms, flux_jy=0.01)
result = simulate_ms(
    ms,
    [source],
    "experiments/0012-399-simulation/simulated.ms",
    noise_model="vla-thermal",
    noise_parameters={"band": "C", "sampler": "8bit"},
    seed=12345,
)
```

The examples must use keyword arguments for physical quantities with easily
confused units. Use `noise_parameters`, not free `**kwargs` on `simulate_ms()`,
so noise-specific names do not pollute its signature and the complete noise
configuration is easy to validate and record.

## `components.py`

### API

```python
def get_phase_center(ms: str | Path, *, field_id: int = 0) -> str:
    """Return a CASA direction string such as 'J2000 <ra>rad <dec>rad'."""


def get_reference_frequency(ms: str | Path, *, spw_id: int = 0) -> str:
    """Return a CASA frequency string such as 'TOPO <frequency>Hz'."""


def phase_center_point_source(
    ms: str | Path,
    flux_jy: float,
    *,
    field_id: int = 0,
    spw_id: int = 0,
) -> dict[str, object]:
    """Build addcomponent parameters for the one supported source family."""


def phase_center_point_source_from_snr(
    ms: str | Path,
    snr: float,
    image_rms_jy_per_beam: float,
    *,
    field_id: int = 0,
    spw_id: int = 0,
) -> dict[str, object]:
    """Set flux_jy = snr * image_rms_jy_per_beam and build the point source."""
```

`phase_center_point_source()` returns exactly the arguments needed by
`componentlist.addcomponent()`:

```python
{
    "flux": [flux_jy, 0.0, 0.0, 0.0],
    "fluxunit": "Jy",
    "polarization": "Stokes",
    "dir": get_phase_center(ms, field_id=field_id),
    "shape": "point",
    "freq": get_reference_frequency(ms, spw_id=spw_id),
    "spectrumtype": "constant",
}
```

`phase_center_point_source_from_snr()` performs only the multiplication and
then calls `phase_center_point_source()`. The shared prefix makes the common
source family visible in autocomplete and documentation.

This fixes all choices except flux. There is no boolean matrix for point versus
Gaussian, constant versus spectral-index, or polarized versus unpolarized.

### Metadata rules

- Read `FIELD.PHASE_DIR[field_id]` and its units.
- Resolve the direction frame from the column `MEASINFO`. If it names a
  `VarRefCol`, read the row's code from that column and map it through
  `TabRefCodes`/`TabRefTypes`. Use a fixed `MEASINFO.Ref` only when present.
  Do not hard-code `J2000` as the fallback for an unknown frame.
- Read `SPECTRAL_WINDOW.REF_FREQUENCY[spw_id]` and resolve its frame in the
  same way. Do not maintain a private numeric frequency-frame dictionary when
  the MS already carries its code-to-name mapping.
- Require a static field for this initial source model:
  `NUM_POLY == 0` and no active ephemeris. A moving or polynomial phase centre
  cannot be represented correctly by one fixed component direction.
- Validate that IDs exist and all numeric inputs are finite and positive.

The verified 0012-399 reference MS has one field with a J2000 phase direction
and one SPW with `REF_FREQUENCY = 7.228e9 Hz` in TOPO. The existing
[scripts/components.py](../scripts/components.py) produced the intended source
shape, but its `MEASINFO.Ref` fallback happens to work for this MS only because
the row's variable reference code is zero (J2000). The new code must implement
the general row-reference lookup above.

## `noise.py`

### Selector and calculations

The simulation entry point receives a selector string and a parameter mapping:

```python
noise_model="vla-thermal"
noise_parameters={"band": "C", "sampler": "8bit"}
```

Keep one internal dispatcher in `noise.py`:

```python
def _apply_noise(
    simulator: object,
    ms: str | Path,
    *,
    noise_model: str,
    noise_parameters: Mapping[str, object],
    seed: int,
) -> dict[str, object]:
    """Apply noise and return normalized metadata and weight information."""
```

It contains a short, explicit branch per supported selector; do not create a
registry or a class for this. The initial behavior is:

- `"vla-thermal"`: accept `band`, `sampler`, and optional `sefd_jy`/`eta_c`
  overrides; calculate `sigma_simple` with
  `theoretical_vla_simplenoise()`; then call
  `simulator.setnoise(mode="simplenoise", simplenoise=f"{sigma}Jy")`.
- `"simplenoise"`: require the CASA quantity parameter `simplenoise`, normalize
  it to a positive finite Jy value with `casatools.quanta`, and pass it to
  `simulator.setnoise(mode="simplenoise", ...)`.

Both branches call `simulator.setseed(seed)` before `setnoise()` and call
`simulator.corrupt()` once. The returned plain dictionary contains the public
selector, actual CASA mode and parameters, resolved scalar sigma in Jy, seed,
reference source/overrides, and required weight-initialization mode. It is an
internal result, not another public configuration type.

Reject unknown keys rather than forwarding arbitrary mappings. In particular,
`mode` and `seed` cannot appear inside `noise_parameters`, because the named
arguments are the single source of truth. CASA errors are useful after this
small boundary validation and should not be hidden or translated repeatedly.

Reserve `"tsys-atm"` and `"tsys-manual"` as future selectors. Their eventual
branches can call `simulator.setnoise(mode=noise_model,
**noise_parameters)` directly, using CASA's own parameter names. Do not accept
them in the first implementation merely as pass-throughs: CASA calculates
non-constant noise internally but does not update `SIGMA` or `WEIGHT` during
`corrupt()`. A CASA 6.7.6 probe on a copied 0012-399 MS confirmed that both
columns remained exactly unchanged after `tsys-manual` corruption. Supporting
these selectors correctly therefore also requires a validated varying-weight
policy; reusing the copied observation's weights or assigning one constant
sigma would be misleading. Until that work exists, raise a focused
`NotImplementedError` explaining this constraint.

Retain the two public calculation functions as independently testable building
blocks:

```python
def theoretical_vla_simplenoise(
    ms: str | Path,
    *,
    sefd_jy: float,
    eta_c: float,
) -> float:
    """Return sigma_simple in Jy for one homogeneous SPW."""


def simplenoise_from_image_rms(
    sigma_na_jy_per_beam: float,
    *,
    nchan: int,
    npol: int,
    nbaselines: int,
    nintegrations: int,
) -> float:
    """Convert a known ideal natural-image RMS to simplenoise in Jy."""
```

CASA documents the approximate natural-image relationship

\[
\sigma_{NA} \simeq
\frac{\sigma_{\rm simple}}
{\sqrt{n_{\rm ch} n_{\rm pol} n_{\rm baselines} n_{\rm integrations}}}.
\]

Therefore the second function is

\[
\sigma_{\rm simple} = \sigma_{NA}
\sqrt{n_{\rm ch} n_{\rm pol} n_{\rm baselines} n_{\rm integrations}}.
\]

For an ideal complete array,

\[
n_{\rm baselines}=\frac{N(N-1)}{2},\quad
n_{\rm integrations}=\frac{t_{\rm int}}{\delta t},\quad
n_{\rm ch}=\frac{\Delta\nu}{\delta\nu}.
\]

Combining that relationship with the NRAO VLA natural-image sensitivity
equation gives the Option A2 result

\[
\boxed{
\sigma_{\rm simple} =
\frac{\mathrm{SEFD}}
{\eta_c\sqrt{2\,\delta\nu\,\delta t}}
}
\]

where `delta_nu` is `SPECTRAL_WINDOW.EFFECTIVE_BW` and `delta_t` is
the MAIN-table `EXPOSURE` for the selected samples. `EFFECTIVE_BW`, rather than
`CHAN_WIDTH`, is required because the MS standard defines it as the effective
noise bandwidth.

### Required checks

`theoretical_vla_simplenoise()` must inspect the MS and:

- map MAIN `DATA_DESC_ID` through `DATA_DESCRIPTION.SPECTRAL_WINDOW_ID`;
- require exactly one used SPW in this initial implementation;
- require one positive finite `EFFECTIVE_BW` value across all its channels;
- require one positive finite `EXPOSURE` across its MAIN rows;
- reject autocorrelation rows because the factor of two above is the
  cross-correlation expression;
- reject non-positive or non-finite `sefd_jy` and `eta_c`; and
- return a float in Jy. `_apply_noise()` is responsible for formatting the CASA
  quantity string.

Keep the homogeneous-sampling inspection in one private function in
`noise.py`; it is used by `theoretical_vla_simplenoise()`, which the
`"vla-thermal"` branch calls. The direct `"simplenoise"` selector represents a
deliberately supplied CASA sigma and does not pretend it was derived from VLA
theory.

The function must not inspect flags to change per-sample sigma. A flag removes a
sample from the later image; it does not change the thermal sigma assigned to
each remaining sample. Flags do matter when predicting the achieved image RMS.

### Versioned VLA reference values and overrides

The NRAO 2026A OSS supplies the following fiducial SEFD snapshot:

| Frequency / band | Fiducial SEFD (Jy) |
| --- | ---: |
| 0.39 GHz / P | 2790 |
| 1.5 GHz / L | 420 |
| 3.0 GHz / S | 370 |
| 6.0 GHz / C | 310 |
| 10.0 GHz / X | 250 |
| 15 GHz / Ku | 320 |
| 20 GHz / K | 500 |
| 33 GHz / Ka | 600 |
| 45 GHz / Q | 1300 |

Put these values in `noise.py` under names that include the source edition, for
example `VLA_OSS_2026A_SEFD_JY`. Also store the source URL and edition in the
model's metadata. This is a useful, reviewable default table; there is no reason
to force every caller to retype `310.0` for the same documented C-band
fiducial.

The same OSS gives `eta_c ~= 0.93` for 8-bit samplers and `0.78--0.83` for
3-bit samplers. Include `0.93` as the 8-bit default and the 3-bit range as
reference data. Do not choose a midpoint for 3-bit: the `"vla-thermal"` branch
must require `eta_c` for that sampler until the package supports a more specific
sampler configuration.

These embedded numbers are still only fiducial points. NRAO states that the
Exposure Calculator uses a frequency-dependent SEFD curve, including zenith
atmospheric emission; at high frequencies actual sensitivity also depends on
weather, elevation, and pointing. Therefore:

- `band="C", sampler="8bit"` is a convenient, documented fiducial model;
- explicit `sefd_jy` and `eta_c` override the snapshot for an experiment; and
- the metadata must say whether each value came from the 2026A table or an
  override.

The branch must also verify that all selected channel frequencies fall within
the named receiver band. This catches a mistaken band label; it does not turn
the fiducial point into a frequency-dependent interpolation.

The authoritative places to obtain values are the NRAO OSS Sensitivity page
(equation, curve, fiducial table, sampler efficiencies) and the VLA Exposure
Calculator and its guide (frequency-specific experiment estimate). Do not copy
numeric points by reading pixels from the published curve image. If an exact
calculator run matters, save its output alongside the validation report and
record the chosen conditions.

### Verified 0012-399 worked example

Direct inspection of the reference MS gave:

| Quantity | Value |
| --- | ---: |
| SPWs | 1 |
| antennas | 27 |
| cross baselines | 351 |
| unique integration times | 235 |
| exposure per integration | 5 s |
| nominal on-source duration | 1175 s |
| channels | 64 |
| effective bandwidth per channel | 2 MHz |
| nominal total bandwidth | 128 MHz |
| correlations in the MS | RR, RL, LR, LL |
| products contributing to Stokes I | 2: RR and LL |

With the fiducial C-band `SEFD = 310 Jy` and the explicit 8-bit assumption
`eta_c = 0.93`:

```text
sigma_simple = 0.074535599249993 Jy
nominal sigma_NA = 2.29388305498282e-05 Jy/beam
```

Only about 54.35% of each correlation's samples are unflagged in this extracted
MS. Counting unflagged RR and LL samples gives an effective sample count of
5,738,686 and the first-order flag-adjusted expectation

```text
sigma_NA,flagged ~= sigma_simple / sqrt(5,738,686)
                   = 3.11141195632344e-05 Jy/beam
```

This corresponds to about 69.57 MHz of effective usable bandwidth if expressed
as a single bandwidth at 1175 s. The validation script must print both nominal
and flag-adjusted expectations. It must not claim that the nominal 22.94
microJy value is the expected image RMS after preserving these flags.

The fixed 0012-399 Exposure Calculator validation uses a separate,
experiment-specific override. A manual ECT run at 7.291 GHz with B
configuration, 27 antennas, dual polarization, natural weighting, 8-bit
sampling, Zenith/Winter, 1175 s on source, and an accidentally entered
69.7525 MHz bandwidth returned 23.7264 microJy/beam. Solving the same
sensitivity equation for SEFD gives 236.6994 Jy, rounded to `236.7 Jy` for the
test. Correcting the bandwidth to the MS-derived 69.5724798 MHz gives the
calculator target

```text
sigma_NA,ECT ~= 23.7571 microJy/beam
sigma_simple = 0.0569115365886237 Jy
```

This `236.7 Jy` value is an effective representation of that exact calculator
case, including its frequency-dependent sensitivity at 7.291 GHz. It is not a
new C-band constant and does not replace the package's documented 310 Jy
default at the 6 GHz fiducial point. The test must pass it through the existing
`sefd_jy` override and record its origin.

Also note that `npol=2` in the image-noise equation even though the MS stores
four correlation products. Counting all four would incorrectly include the
cross-hands in a Stokes-I sensitivity calculation.

## `rms_metrics.py`

### Generic metric

```python
@dataclass(frozen=True)
class RMSMetrics:
    rms_jy_per_beam: float
    scaled_mad_jy_per_beam: float
    n_pixels: int


def measure_pb_region(
    image: str | Path,
    pb: str | Path,
    *,
    pb_min: float,
    pb_max: float | None = None,
    exclude_mask: str | Path | None = None,
) -> RMSMetrics:
    ...
```

Use the final non-PB-corrected restored image. Select finite pixels satisfying
`PB >= pb_min` and, when supplied, `PB <= pb_max`. When `exclude_mask` is
supplied, also require `mask < 0.1`. Return:

\[
\mathrm{RMS}=\sqrt{\frac{1}{N}\sum_i x_i^2},\qquad
\mathrm{scaled\ MAD}=1.4826\,\mathrm{median}(|x_i-\mathrm{median}(x)|).
\]

Implement this with one direct `casatools.image.statistics(..., robust=True,
stretch=True)` call and the corresponding CASA mask expression. Do not load an
entire cube into Python merely to reproduce statistics CASA already provides.
Check that the image brightness unit is Jy/beam, inputs exist, PB bounds are
ordered, the coordinate systems/shapes can be stretched compatibly, and at
least one pixel is selected.

The vault diagnostics become explicit calls rather than more wrappers:

```python
central = measure_pb_region(image, pb, pb_min=0.5)
annulus = measure_pb_region(image, pb, pb_min=0.2, pb_max=0.3)
```

This is the classical statistic used for Gaussian-noise validation.

### VLA Pipeline compatibility metric

```python
def vla_pipeline_annulus_rms(
    image: str | Path,
    pb: str | Path,
    clean_mask: str | Path | None,
) -> float:
    """Reproduce the Pipeline non-PB-corrected, non-clean-mask RMS."""
```

Reproduce the installed Pipeline behavior directly, without importing private
Pipeline modules:

1. Start with PB limits 0.2 and 0.3.
2. Apply the Pipeline's small-image PB-limit adjustment from
   `ImageParamsHeuristics.pblimits()`. This is necessary when the image does not
   extend to PB 0.2.
3. When a clean mask exists, first select `clean_mask < 0.1` in that annulus.
4. Call `image.statistics()` with `robust=True`, `axes=[0, 1, 2]`,
   `algorithm="chauvenet"`, `maxiter=5`, and `stretch=True`.
5. If the median selected point count per channel is below 10, repeat on the
   full PB annulus without excluding the clean mask, matching the Pipeline
   fallback.
6. For a cube, return the median of the per-channel RMS values. For the stored
   continuum `.tt0` image there is one spectral plane.

Do not implement Chauvenet rejection in Python. Calling CASA directly is both
shorter and more likely to remain behaviorally compatible.

### Why the 0012-399 Pipeline limits are so close to one

The limits are derived from code and stored data, not inferred from the reported
RMS:

- the producing log records the exact Pipeline RMS and CASA/Pipeline versions;
- `cleanbox.py::analyse_clean_result()` records the mask expression and CASA
  Chauvenet arguments;
- `imageparams_base.py::pblimits()` contains the small-image rule; and
- reading the matching stored PB table gives the numerical limits below.

The 256 by 256 image covers only 38.4 arcsec. Its PB at the centre of the top
edge is approximately 0.99197, so a 0.2--0.3 PB annulus is entirely outside the
image. In this situation the Pipeline finds the first usable row at least 5% of
the image height in from an edge and makes the second boundary another 5% of
the height inward. Here those boundaries are at offsets 12 and 24 pixels and
map to PB values 0.9934094548 and 0.9947110415.

The small numerical PB width is therefore not a one-pixel annulus: the PB is
very flat near the pointing centre. Spatially it is about 12 pixels wide and
selects approximately 8,102 pixels before Chauvenet rejection (8,089 after it
in the CASA 6.7.6 check). That is enough pixels for a stable background sample,
but its interpretation is narrower than the generic vault diagnostic: it is a
Pipeline fallback for an undersized image, not a measurement of the sky near
the 0.25-power point. Use it only for compatibility/regression. For a physical
0.2--0.3 annulus validation, image a sufficiently large field.

Accordingly, retain the note's ordinary 0.2--0.3 calculation as a generic
Gaussian-noise diagnostic, but do not label it as an exact VLA Pipeline
statistic. The implementation and tests must retain both named operations and
explain which one is being reported. Correcting the vault note itself is outside
this repository change; this plan is the authoritative implementation
specification.

### 0012-399 regression fixture

Copy, without modifying, these producing artifacts:

- `collect/experiments/compute_background_RMS_20260903T133414/0012-399/`
  `vla_pipeline/pipeline/`
  `oussid.s5_0.J0012-3954_sci.C_band.cont.regcal.I.iter1.image.tt0`
- the matching `iter1.pb.tt0`;
- `central_clean_0.mask`, which is the mask passed to the Pipeline statistic;
  and
- `collect/experiments/compute_background_RMS_20260903T133414/0012-399/`
  `vla_pipeline/qa.json`.

Do not copy the whole 221 MB Pipeline directory. These three CASA tables and the
JSON are approximately 1.6 MB in total.

Add `reference.json` with:

```json
{
  "pipeline_rms_jy_per_beam_exact": 0.00024389197254258183,
  "qa_json_rms_jy_per_beam_rounded": 0.00024,
  "casa_version": "6.6.6.18",
  "pipeline_version": "2025.1.0.36",
  "pb_min_for_this_image": 0.9934094548225403,
  "pb_max_for_this_image": 0.9947110414505005,
  "statistic": "CASA image.statistics RMS after Chauvenet rejection, maxiter=5",
  "clean_mask_rule": "central_clean_0.mask < 0.1"
}
```

The exact RMS comes from the producing stage-5 `casapy.log`; `qa.json` contains
only the rounded weblog value. Record the producing artifact paths in the
manifest too so the fixture remains auditable after its copies are renamed.

The regression assertion is:

```python
np.isclose(actual, 0.00024389197254258183, rtol=5e-3, atol=0.0)
```

The 0.5% relative allowance is version compatibility, not statistical
uncertainty: CASA 6.7.6.14 measured approximately `0.00024458 Jy/beam` from the
same stored tables and algorithm, about 0.28% above the CASA 6.6.6.18 result.
When the test runs under the producing CASA version, additionally require a
much tighter `rtol=1e-6` result.

For comparison, a raw NumPy RMS over the un-clipped Pipeline pixel selection is
approximately `0.000247886 Jy/beam`, about 1.6% above the producing Pipeline
value. This confirms that an ordinary annulus function is not a valid Pipeline
regression implementation.

## `simulations.py`

### API and outputs

```python
@dataclass(frozen=True)
class SimulationResult:
    ms_path: Path
    component_list: Path | None
    metadata_json: Path
    simplenoise_jy: float | None
    seed: int | None


def simulate_ms(
    ms: str | Path,
    components: Sequence[Mapping[str, object]],
    output_ms: str | Path,
    *,
    noise_model: str | None = None,
    noise_parameters: Mapping[str, object] | None = None,
    seed: int = 185349251,
) -> SimulationResult:
    ...
```

`components` may be empty only when `noise_model` is supplied; that is the
noise-only validation case. At least one of source or noise must be present.
`None` means no noise and requires `noise_parameters` to be absent or empty.
The selector chooses behavior; `simulate_ms()` never infers a model from the
presence of a scalar.

The output path is explicit and must end in `.ms`. Relative paths are resolved
from the caller's current working directory. Never place output silently beside
the input MS. Refuse to proceed when the output MS, component-list path, or
metadata path already exists. Do not expose an `overwrite=True` shortcut in
this first API.

For `output_ms = simulated.ms`, the output contract is:

```text
simulated.ms/                 # Copied sampling plus simulated DATA
simulated.components.cl/      # CASA table; absent for noise-only runs
simulated.simulation.json     # Reproducibility metadata and product paths
```

The JSON records the resolved original MS, component dictionaries, source
field/SPW IDs when present, requested selector and parameters, normalized CASA
mode and parameters, resolved sigma when applicable, seed,
weight-initialization mode, CASA version, and created paths. The CASA
component-list table remains the sky model consumed by `predict()`.

Here "provenance" means the short record of how a generated artifact was made.
Use the plainer API name `metadata_json`, but keep the information. An output MS
does not say that `DATA` came from a particular component list, which random
seed was used, whether 310 Jy was a 2026A fiducial or an override, or which CASA
version generated it. Without this small JSON file, two visually identical MS
directories can represent different experiments and their results cannot be
reproduced reliably.

### Linear implementation flow

Implement the body in this order:

1. Resolve the input with `scripts.imaging.metadata.resolve_path()` so both an
   existing path and an ID such as `0012-399` work.
2. Validate inputs and ensure the resolved output is not the input. Validate
   the selector/parameter combinations before copying anything.
3. Copy the closed MS with `shutil.copytree()`. Because pre-existing outputs
   were rejected, this copy cannot delete user data.
4. If components are present, create the CASA component list directly with
   `componentlist.addcomponent(**component)` and `rename()`.
5. If there are no components, zero `DATA` and `CORRECTED_DATA` if present, in
   bounded row chunks. Do not zero flags, UVW, weights, metadata, or unrelated
   columns.
6. Open one `casatools.simulator` on the copied MS.
7. If components exist, call
   `predict(complist=..., incremental=False)`. This replaces copied observed
   data; zeroing first is unnecessary in this branch.
8. If noise is requested, call `_apply_noise()` once. It performs the selected
   calculation, calls `setseed()`, `setnoise()`, and `corrupt()`, and returns a
   normalized plain metadata dictionary.
9. Close the simulator in `finally`.
10. For either initially supported noise selector, take the resolved constant
    sigma returned by `_apply_noise()`, write it to `SIGMA` in bounded chunks,
    close the table, then call
    `casatasks.initweights(wtmode="sigma", dowtsp=True)`. If no noise was added,
    call `casatasks.initweights(wtmode="ones", dowtsp=True)` directly.
11. Write the metadata JSON only after the CASA operations succeed, then return
    the result.

Two private chunked table operations are acceptable: `_zero_visibility_data()`
in `simulations.py` and `_set_constant_sigma()` in `noise.py`. The latter writes
only the scalar that CASA cannot accept as an `initweights` argument. Do not
manually compute/write `WEIGHT`, and do not split the CASA simulator lifecycle
into a chain of one-line helpers.

`statwt` is not an alternative here: it estimates weights from scatter in the
visibilities, so a source can bias it and a noiseless source-only dataset is
degenerate. `initweights(wtmode="nyq")` is also not the desired operation: it
uses bandwidth and exposure for normalized raw data but does not apply this
simulation's SEFD-to-Jy scale. `wtmode="sigma"` is the CASA operation that
matches a known synthetic sigma.

After either non-channelized initialization, expect `SIGMA`, `WEIGHT`, and a
uniform `WEIGHT_SPECTRUM` to exist and agree. CASA deliberately removes
`SIGMA_SPECTRUM` in `sigma` and `ones` modes; tests must verify that documented
result rather than requiring the input schema to be preserved. In the no-noise
case, values of one mean "uniform neutral imaging weights," not a physical
claim of 1 Jy thermal uncertainty. Exact zero uncertainty cannot be represented
by finite inverse-variance weights.

The original MS must only ever be opened read-only. On failure, report the
partial output paths clearly; removal can be performed only for paths created
by that call.

### Model semantics

- The component list is stored as a CASA table directory, containing files such
  as `table.dat` and `table.f0`; it is not a Python-list file.
- No model image is created before simulation.
- `simulator.predict()` can start directly from the component list.
- A model image is needed only for a different workflow such as supplying
  `tclean(startmodel=...)`. The repository's
  [zero_residual_with_skymodel_test.py](../scripts/zero_residual_with_skymodel_test.py)
  materializes an image for that special test because `tclean` cannot use its
  lazy component-list image directly. That behavior must not be copied into the
  normal simulation path.
- To inspect exact predicted visibilities without replacing observed `DATA`, a
  validation test may use `ft(..., complist=..., usescratch=True,
  incremental=False)` on a separate temporary copy, as demonstrated by
  [model_removal_test.py](../scripts/model_removal_test.py). This is a test
  oracle, not the production simulation flow.

## Fixed VLA calculator validation and report

Implement `tests/validate_vla_noise_0012_399.py` as one directly runnable,
fixed validation case, not package API and not a parameterized command-line
tool. It must accept no arguments. Keeping `validate_...` rather than
`test_...` prevents ordinary test discovery from launching an expensive CASA
simulation, while its assertions and nonzero failure exit still make it an
integration test.

Put the chosen constants together at the top of the file: the 0012-399 input
MS, C band, 8-bit sampler, the ECT-equivalent 236.7 Jy SEFD override and 0.93
efficiency, a fixed random seed, and the natural-weight dirty-image
configuration. Keep the 310 Jy 2026A fiducial as the package default; the fixed
validation must supply 236.7 explicitly. Create a new timestamped directory
under `experiments/vla_noise_validation_0012_399/` on each run so the program
needs no output argument and never overwrites a prior validation.

Run it from the repository root with:

```text
casa --nogui --nologger -c tests/validate_vla_noise_0012_399.py
```

It must print, before running CASA:

- representative frequency: mean of `CHAN_FREQ`, about 7.291 GHz;
- nominal and flag-adjusted usable bandwidth: 128 MHz and about 69.57 MHz;
- 27 antennas, 351 cross baselines, 235 integrations, 5 s exposure, and
  1175 s nominal time;
- Stokes I / dual polarization, natural weighting, no taper, and `niter=0`;
- the 236.7 Jy ECT-equivalent SEFD and its derivation, `eta_c`, source edition,
  sampler, declared Zenith/Winter assumptions, and seed;
- calculated `sigma_simple`, nominal image RMS, and flag-adjusted image RMS.

Then it must:

1. call `simulate_ms(..., noise_model="vla-thermal",
   noise_parameters={"band": "C", "sampler": "8bit", "sefd_jy": 236.7,
   "eta_c": 0.93})` and make a noise-only simulated MS through the package;
2. image it non-PB-corrected with natural weighting, Stokes I, no taper, and
   `niter=0`;
3. report central and annulus classical RMS/scaled-MAD values where the PB
   product covers those regions;
4. write `results.json` plus a dirty-image PNG, pixel-distribution PNG, and any
   usable PB-region comparison plot;
5. print ratios of measured values to both the flag-adjusted prediction and
   the corrected 23.7571 microJy/beam ECT target, and assert the same 10%
   acceptance criteria as the noise-only integration test; and
6. render a small self-contained HTML report before exiting, including when a
   scientific assertion fails.

The program may call `tclean` directly for this one validation image so it can
request and retain the matching PB product explicitly. Wrapping this single,
fixed CASA call in another imaging abstraction would obscure the parameters
being validated. The separate full-flow integration test below is the test
that must exercise `scripts.imaging.image_ms()`.

Store `tests/vla_noise_validation_report.qmd` as the report template. Copy it
into the run directory after the result JSON and PNGs exist, then use the
existing `scripts.reporting.QuartoReporter` with `every=1`; call
`sample_completed()` once and `finish()` before applying the final pass/fail
exit. The reporter already owns the `quarto render` subprocess and background
thread, so do not duplicate that machinery. Verify that the HTML file exists;
if Quarto is unavailable, fail with the JSON/PNG directory printed clearly.

The terminal output must give the essential values, and the report must contain
a row-by-row table in this order:

| Step | Calculator row | Value |
| ---: | --- | --- |
| 1 | Representative Frequency | 7291 MHz, then Tab |
| 2 | Frequency Bandwidth | 69.5725 MHz, then Tab |
| 3 | Purpose of Calculation | Noise Simulation Check |
| 4 | Array Configuration | B |
| 5 | Number of Antennas | 27 |
| 6 | Polarization Setup | Dual |
| 7 | Type of Image Weighting | Natural |
| 8 | Receiver Band | output only; C |
| 9 | Approximate Beam Size | output only; approximately 1.318 arcsec |
| 10 | Number of Frequencies | 1 |
| 11 | Session Includes Pointing | False |
| 12 | Digital Samplers | 8 bit |
| 13 | Elevation | Zenith (90 degrees) |
| 14 | Average Weather | Winter |
| 15 | Calculation Type | Noise/Tb |
| 16 | Number of Sources | 1 |
| 17 | Time on Source | 1175s or 19m35s |
| 18 | Total On-Source Time | output only; 19m35s |
| 19 | Total Time | output only; approximately 37m22s |
| 20 | Line Velocity Width | output only; approximately 2860.69 km/s |
| 21 | RMS Noise | output only; approximately 23.7571 microJy/beam |
| 22 | RMS Brightness | output only; approximately 0.3144 K |
| 23 | Confusion Level | output only; informational |
| 24 | Overhead Explanation | output only; informational |

The first two rows must be submitted before the calculator enables the other
controls. The report must explicitly warn not to transpose the bandwidth to
69.7525 MHz. Compare the calculator RMS to the reported non-PB-corrected dirty
image RMS, not to the real-data Pipeline background RMS.

The program cannot query the external calculator automatically. It instead
records the manual result and the declared sampler, elevation, and weather
choices that produced the effective 236.7 Jy value. These are validation
inputs, not claims about observing conditions recovered from 0012-399.

## Automated validation

### Fast tests

`tests/test_simulation_package.py` runs without a full simulation and covers:

- `simplenoise_from_image_rms()` on a hand-computed sample-count case;
- equivalence of the two noise equations for the ideal 0012-399 counts;
- the Option A2 result `0.074535599249993 Jy` from 2 MHz, 5 s, 310 Jy, and
  0.93 using a minimal mocked table interface;
- the 0012-399 ECT-equivalent override result `0.0569115365886237 Jy` from the
  same sampling with 236.7 Jy, plus its 23.7571 microJy/beam flag-adjusted
  target;
- rejection of invalid values, nonuniform bandwidth/exposure, multiple SPWs,
  and autocorrelations;
- VLA snapshot lookup, 8-bit default, required explicit 3-bit efficiency, and
  explicit SEFD/efficiency overrides;
- exact source record fields and `flux = snr * image_rms` through
  `phase_center_point_source_from_snr()` using mocked MS metadata;
- selector validation; exact CASA call construction for `"vla-thermal"` and
  `"simplenoise"`; rejection of conflicting/unknown parameters; and the focused
  `NotImplementedError` for the reserved `tsys-*` selectors;
- `SimulationResult` and public exports.

CASA imports should occur inside CASA-dependent functions so the pure formula
tests and package import do not require launching CASA.

### CASA fixture regression

Under `RUN_CASA_SIMULATION_INTEGRATION=1`, verify that:

- `vla_pipeline_annulus_rms()` matches the exact stored 0012-399 result within
  the version-aware tolerance above;
- the copied `qa.json` still contains `0.00024`;
- `reference.json` points to CASA 6.6.6.18 / Pipeline 2025.1.0.36; and
- the generic metric and Pipeline-compatible metric are treated as different
  named results.

### Noise-only integration test

Using a temporary output directory and a fixed seed:

1. Create a noise-only copy of 0012-399 with the Option A2 value using the
   explicit 236.7 Jy ECT-equivalent override.
2. Assert the original MS hashes are unchanged.
3. Assert simulated unflagged real and imaginary samples have means close to
   zero and standard deviations close to `sigma_simple`.
4. Assert `SIGMA` equals `sigma_simple`, CASA-derived `WEIGHT` equals
   `1 / sigma_simple**2`, `WEIGHT_SPECTRUM` is uniform and consistent, and
   `SIGMA_SPECTRUM` is absent as documented for `initweights(wtmode="sigma")`.
5. Image with natural weighting, no taper, Stokes I, and `niter=0`.
6. Assert ordinary RMS and scaled MAD agree within 10% in each usable PB
   region.
7. Assert central and annulus RMS agree within 10% when both regions exist.
8. Assert the measured RMS agrees with the flag-adjusted theoretical value
   within 10%.

The 10% limits are scientific acceptance tolerances for one random realization
and imperfect finite-image sampling. They are not appropriate for the
deterministic formula unit tests.

### Component and full-flow integration test

Still under the opt-in CASA flag:

1. Build a 1 Jy phase-centre component and predict it without noise.
2. On a separate temporary copy, write the same component through `ft()` and
   compare its `MODEL_DATA` with the simulated `DATA` for unflagged samples.
3. Verify the original is unchanged, the returned component-list table and
   metadata JSON exist, the noiseless MS has `SIGMA=WEIGHT=1`, and its uniform
   `WEIGHT_SPECTRUM` exists while `SIGMA_SPECTRUM` does not.
4. Run the noise-only pilot, measure its image RMS, and construct a 100-sigma
   source with `phase_center_point_source_from_snr()`.
5. Simulate the 100-sigma source plus the same 236.7 Jy ECT-equivalent noise
   model with a fixed seed.
6. Image both the original 0012-399 MS and this simulated MS through
   `scripts.imaging.image_ms()` using the same explicit configuration.
7. Require all image/QA products, require the simulated source peak-to-RMS to
   be close to 100 within a documented 10% tolerance, and record both results.

For step 7, define peak-to-RMS as the recovered simulated-image absolute peak
divided by the independent noise-only pilot RMS used to construct the source.
Do not divide by the source-plus-noise dirty residual's full-image MAD: with
`niter=0`, a 100-sigma point source contributes dirty-beam sidelobes throughout
that image and biases the MAD upward. The source-plus-noise QA MAD remains a
useful recorded diagnostic, but it is not an independent thermal-noise
denominator.

Use one explicit validation imaging configuration for both calls: `data`,
natural weighting, Stokes I, `hogbom`, no clean mask, and
`CleanIterationsConfig(niter=0)`. The existing `ImagingConfig` has no taper
field, so omitting `uvtaper` preserves the required no-taper CASA default. Do
not use `DefaultImagingConfig` unchanged because it requests Briggs weighting
and deep cleaning, which would invalidate a comparison with the theoretical
natural-image equation.

Do not assert that the real original and controlled simulation have equal RMS.
The original image contains calibration residuals, source structure, weighting
effects, and other non-thermal contributions. Its stored Pipeline annulus RMS
is about 0.24 mJy/beam, roughly an order of magnitude above the nominal
theoretical floor; it is a comparison product, not the target value for the
Option A2 noise-only test.

Use `tempfile.TemporaryDirectory()` for generated MSs and images. The checked-in
fixture is read-only, and the expensive tests remain opt-in, following the
pattern in
[tests/test_imaging_0012_399_integration.py](../tests/test_imaging_0012_399_integration.py).

## Implementation order and completion criteria

1. Add the package skeleton and pure noise calculation tests.
2. Implement component metadata/frame handling and source-record tests.
3. Copy the minimal Pipeline fixture and implement both RMS operations.
4. Implement the linear simulation flow and weight updates.
5. Add the noise-only and component prediction integration tests.
6. Add the fixed Exposure Calculator validation program, plots, and Quarto
   report.
7. Add the full original-versus-simulated 0012-399 integration test.

The package is complete when all fast tests pass outside CASA, all opt-in tests
pass in the supported CASA environment, the Pipeline fixture regression meets
its version-aware tolerance, and the fixed validation program produces its
JSON, plots, HTML report, and enough explicit inputs for another person to
reproduce the calculator comparison.

## Sources

### Repository and vault sources

- [Legacy source constructors](../scripts/components.py)
- [Legacy simulation helpers](../scripts/sim_utils.py)
- [Legacy comparison-heavy simulation script](../scripts/simulations.py)
- [Imaging package design and behavior](imaging_reorganization_plan.md)
- [Imaging integration-test pattern](../tests/test_imaging_0012_399_integration.py)
- [Existing Quarto report runner](../scripts/reporting/quarto.py)
- [Visibility-domain component test](../scripts/model_removal_test.py)
- [Image-model special-case test](../scripts/zero_residual_with_skymodel_test.py)
- [Vault: CASA Thermal Noise Simulation](<../../Obsidian Vault/Inbox/CASA Noise Simulation.md>)
- [Vault: Measuring image RMS in clean simulations](<../../Obsidian Vault/Notes/Measuring image RMS in clean simulations.md>)
- [Vault: CASA Visibility Simulation Flow](<../../Obsidian Vault/Inbox/CASA Visibility Simulation Flow.md>)
- [0012-399 QA JSON](../collect/experiments/compute_background_RMS_20260903T133414/0012-399/vla_pipeline/qa.json)
- [Producing Pipeline stage-5 log](../collect/experiments/compute_background_RMS_20260903T133414/0012-399/vla_pipeline/pipeline/pipeline-20260903T193606/html/stage5/casapy.log)

The exact Pipeline algorithm was also checked in the installed
CASA 6.6.6.18 / Pipeline 2025.1.0.36 source:

```text
pipeline/hif/heuristics/cleanbox.py::analyse_clean_result
pipeline/hif/heuristics/imageparams_base.py::pblimits
pipeline/hif/tasks/tclean/basecleansequence.py::iteration_result
```

### Primary external references

- [CASA simulator tool](https://casadocs.readthedocs.io/en/latest/api/tt/casatools.simulator.html):
  `predict`, `setnoise`, `setseed`, and `simplenoise` semantics.
- [CASA component-list tool](https://casadocs.readthedocs.io/en/v6.6.1/api/tt/casatools.componentlist.html):
  `addcomponent` fields and constant-spectrum support.
- [CASA image tool](https://casadocs.readthedocs.io/en/stable/api/tt/casatools.image.html):
  RMS, MAD, cursor axes, and Chauvenet statistics.
- [CASA data weights](https://casadocs.readthedocs.io/en/latest/notebooks/data_weights.html):
  `SIGMA`, `WEIGHT`, effective bandwidth, and exposure relationships.
- [CASA `initweights`](https://casadocs.readthedocs.io/en/latest/_modules/casatasks/calibration/initweights.html):
  `sigma` and `ones` modes, `WEIGHT_SPECTRUM` initialization, and documented
  removal of `SIGMA_SPECTRUM` in non-channelized modes.
- [MeasurementSet v2 definition](https://casacore.github.io/casacore-notes/229.html):
  `PHASE_DIR`, `REF_FREQUENCY`, `CHAN_FREQ`, `MEAS_FREQ_REF`, and
  `EFFECTIVE_BW` meanings.
- [VLA OSS 2026A sensitivity](https://science.nrao.edu/facilities/vla/docs/manuals/oss2026a/performance/sensitivity):
  image-sensitivity equation, fiducial SEFD values, and sampler efficiencies.
- [VLA Exposure Calculator guide](https://science.nrao.edu/facilities/vla/docs/manuals/propvla/determining):
  required calculator inputs and interpretation.
- [CASA/VLA Pipeline release information](https://science.nrao.edu/facilities/vla/data-processing/pipeline/vipl_666):
  the CASA and Pipeline versions that produced the stored reference.
