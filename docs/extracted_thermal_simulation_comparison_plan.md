# Extracted data versus matched-dynamic-range thermal simulation experiment

## Goal

For every canonical extracted Measurement Set, compare a direct CASA image of
the real data with a point-source-plus-thermal-noise simulation on exactly the
same sampling and flags. The simulation source strength is set separately for
each sample so that its requested signal-to-noise ratio equals the dynamic
range measured from the real image.

Here, the reported image S/N and dynamic range are the same estimator:

```text
dynamic range = max(abs(clean image)) / scaled MAD(clean residual)
scaled MAD = 1.4826 * median(abs(residual - median(residual)))
```

This makes the simulated and original images comparable in peak-to-error scale
while leaving their residual structure different: the simulation contains
random thermal noise, whereas the original may contain calibration and imaging
artifacts.

## Imaging engine

Both sides use the repository's ordinary direct imaging entry point:

```python
from scripts.imaging import DefaultImagingConfig, image_ms

result = image_ms(ms_path, DefaultImagingConfig, output_dir)
```

This invokes CASA `tclean` directly. It does not import or call
`image_ms_VLA_pipe`, does not run `hifv`, and does not require CASA's
`--pipeline` option. `DefaultImagingConfig` currently means Stokes I MFS,
standard gridder, MT-MFS with `nterms=1`, Briggs `robust=0.5`, a central
six-beam mask, and the default deep-clean controls.

## Inputs and fixed assumptions

Only canonical inputs are selected:

```text
collect/extracted/<id>/<id>/<id>.ms
```

The batch constants are:

```python
SAMPLE_IDS = None
REPORT_EVERY = 1
SAMPLER = "8bit"
ETA_C = 0.93
BASE_SEED = 20260907
DISK_SPACE_FACTOR = 4.0
MIN_HEADROOM_GIB = 5.0
EXPERIMENT_DIR = None
```

`SAMPLE_IDS = None` selects all canonical samples. `EXPERIMENT_DIR = None`
creates a new timestamped run; assigning an existing compatible directory is
an intentional resume. The 8-bit sampler and `eta_c=0.93` are declared
experimental assumptions, not metadata recovered from the observations.

The per-band VLA OSS 2026A SEFD table is used. A stable SHA-256-derived seed
from `BASE_SEED` and the sample ID makes each simulation repeatable.

## Per-sample flow

### 1. Image the original MS

```python
original = image_ms(
    ms_path,
    DefaultImagingConfig,
    sample_dir / "original" / "default_imaging",
)
```

The selected band comes from
`original.resolved_config.ms_band_meta.selected_band`; it is not inferred from
a directory name.

### 2. Measure the original dynamic range and calculate thermal noise

The original QA result supplies the target:

```python
target_dynamic_range = original.qa.metrics.dynamic_range
```

The simulation noise is still based on the theoretical visibility noise and
the retained flags:

```python
simplenoise_jy = theoretical_vla_simplenoise(
    ms_path,
    sefd_jy=VLA_OSS_2026A_SEFD_JY[band],
    eta_c=ETA_C,
)
predicted_image_rms_jy_per_beam = natural_image_rms_from_simplenoise(
    ms_path,
    simplenoise_jy,
)
component = phase_center_point_source_from_snr(
    ms_path,
    target_dynamic_range,
    predicted_image_rms_jy_per_beam,
)
```

Thus the phase-centre point-source flux is:

```text
original measured dynamic range * flag-adjusted predicted natural image RMS
```

The target is no longer an arbitrary fixed 100. The prediction is a natural
weighting estimate, while final images use Briggs weighting and deconvolution,
so the achieved simulated dynamic range need not equal the target exactly. The
report shows the target and both measured results so that this difference is
visible rather than hidden.

### 3. Simulate source plus noise

```python
simulation = simulate_ms(
    ms_path,
    [component],
    sample_dir / "simulation" / f"{sample_id}_matched_snr.ms",
    noise_model="vla-thermal",
    noise_parameters={"band": band, "sampler": SAMPLER},
    seed=sample_seed,
)
```

The simulation copies the input MS, predicts a constant-spectrum unpolarized
phase-centre component, adds constant CASA `simplenoise`, preserves flags, and
initializes synthetic SIGMA/WEIGHT values. The driver verifies that the
simulation's recorded `simplenoise_jy` is identical to the value used to set
the source flux.

### 4. Image the simulation

```python
simulated = image_ms(
    simulation.ms_path,
    DefaultImagingConfig,
    sample_dir / "simulation" / "default_imaging",
)
```

The driver checks that user-controlled imaging settings match between the
original and simulation: requested image size, mask diameter, gridder,
weighting, robust value, Stokes, deconvolver, and `nterms`. Data-derived cell
size and clean controls may differ because the direct imager resolves them from
each MS's synthesized beam and dirty image.

### 5. Commit atomically and continue

A sample enters `report.json` only after both image results, both QA files, all
six PNGs, the simulated MS, component list, and simulation metadata exist.
The manifest is written atomically before the asynchronous reporter is
notified. A failure records the sample, stage, and exception and the batch
continues.

An uncommitted interrupted sample tree is removed before retrying, but only
after verifying it is the exact per-sample directory beneath the experiment
and contains only `original` and/or `simulation`. Completed samples are
verified and skipped during resume. Canonical input MSs are never modified.

## Output layout

```text
collect/experiments/extracted_thermal_simulation_comparison_<timestamp>/
  report.qmd
  report.json
  report.html
  <id>/
    original/
      default_imaging/
        qa.json
        qa.txt
        dirty.png
        clean.png
        residual.png
        ... CASA image products ...
    simulation/
      <id>_matched_snr.ms/
      <id>_matched_snr.components.cl/
      <id>_matched_snr.simulation.json
      default_imaging/
        qa.json
        qa.txt
        dirty.png
        clean.png
        residual.png
        ... CASA image products ...
```

Each manifest sample records `source_snr`, `target_dynamic_range`, and
`source_snr_basis="original_clean_peak_over_residual_scaled_mad"`, plus the
predicted thermal RMS and paths to the authoritative QA/simulation records.

## Report

The report starts with four two-population comparison histograms. Each plot
uses common bin edges and a common count scale for original and simulation:

1. dynamic range / estimated image S/N;
2. residual scaled MAD in Jy/beam;
3. residual peak / scaled MAD;
4. residual p99 / scaled MAD.

Dynamic range and scaled MAD use log-spaced bins because their populations can
span orders of magnitude. The histograms update from all completed pairs each
time the incremental report is rendered.

Every completed sample remains a compact four-row comparison with six linked
PNGs: original metadata/metrics, original dirty-clean-residual, simulation
metadata/metrics, and simulation dirty-clean-residual. Both metric cells show
`estimated S/N (clean peak/scaled MAD)`. The simulation row additionally shows
the target copied from the original and the achieved simulated S/N.

The PNG writer scales each image independently, so similar numeric colorbar
ranges are an expected consequence to inspect, not a forced plotting limit.

## Validation

Fast tests cover canonical discovery, deterministic seeds, flag-aware natural
RMS calculation, atomic manifests, interrupted-output cleanup, commit ordering,
and Quarto rendering with four histograms, four sample rows, and six images.

The opt-in 0012-399 integration test verifies:

1. the original MS tree hash is unchanged;
2. both QA records say `engine="direct"`;
3. source flux equals original measured dynamic range times predicted RMS;
4. simulation `simplenoise`, SIGMA, and WEIGHT satisfy the noise contract;
5. both image/QA product sets exist and stable settings match;
6. the report contains the four population plots and one complete sample pair.

## Run command

Use standard CASA without Pipeline mode:

```bash
'/Users/u1528314/Applications/CASA.app/Contents/MacOS/casa' \
  --nogui --nologger \
  -c scripts/compare_extracted_thermal_simulations.py
```

The driver performs a conservative disk-space preflight, updates the report
after each completed pair, continues past per-sample failures, and prints the
experiment, manifest, and HTML paths at completion.
