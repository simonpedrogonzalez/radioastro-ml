# Single amplitude versus phase corruption simulation plan

## Goal

Use the current package-like antenna-gain corruption code to make a small,
controlled comparison from one existing thermal simulation:

```text
uncorrupted 0012-399 thermal simulation
constant amplitude error on one antenna only
constant phase error on the same antenna only
```

Each corrupted case is an independent copy. The two errors are never active in
the same MS. Both copies and all new imaging/report products live beneath a new
experiment directory; the source thermal experiment and canonical extracted
MS are never modified.

The required report is one three-row comparison for `0012-399`. Its rows are
the clean thermal-noise simulation, the one-antenna amplitude-only case, and
the one-antenna phase-only case; its columns are dirty, CLEAN, and residual.
The corruption magnitudes are intentionally strong enough to be visually
significant.

The experiment driver is
[`scripts/single_amp_vs_phase_corruption_simulation.py`](../scripts/single_amp_vs_phase_corruption_simulation.py).

## Checked source simulation

At plan creation time, the newest thermal-comparison run containing a complete,
already-imaged `0012-399` simulation is:

```text
collect/experiments/extracted_thermal_simulation_comparison_20260909T005739
```

The source products are:

```text
0012-399/simulation/0012-399_matched_snr.ms
0012-399/simulation/0012-399_matched_snr.simulation.json
0012-399/simulation/default_imaging/qa.json
0012-399/simulation/default_imaging/dirty.png
0012-399/simulation/default_imaging/clean.png
0012-399/simulation/default_imaging/residual.png
```

The source manifest records:

```text
image size: 256 x 256 pixels
metric annulus: 3.0 <= radius < 6.719983535034756 beam-major FWHM
source model: phase-centre point source plus VLA thermal simplenoise
```

The driver discovers the newest qualifying run dynamically. A run qualifies
only when its `report.json` contains exactly one completed `0012-399` entry and
the simulated MS, simulation metadata, QA JSON, and all three PNGs exist. A
specific run can be pinned with `--source-run`.

## Fixed experiment choices

### Sample and antenna

Only `0012-399` is processed. Exactly one antenna is selected. By default this
is the lowest numeric antenna ID that occurs in at least one unflagged MS row.
The driver prints and records both its ID and name. `--antenna eaNN` or
`--antenna ID` can override it. The same antenna is used in both variants.

### Constant amplitude case

The selected antenna receives:

```text
gain amplitude = 1.50
gain phase = 0 radians
```

This is a constant +50% antenna gain-amplitude error. All other antenna gains
remain `1+0j`. For a baseline containing the selected antenna exactly once,
the ideal visibility amplitude is multiplied by 1.50. Baselines not containing
that antenna are unchanged.

### Constant phase case

The selected antenna receives:

```text
gain amplitude = 1.0
gain phase = +45 degrees
```

A baseline containing the selected antenna receives a positive or negative
45-degree visibility phase change depending on whether the antenna is the
first or second member of the baseline. Baselines not containing it are
unchanged. Because this is an antenna-based error, closure phase should be
preserved.

### Time behavior and seeds

The current framework requires a numeric `TimeGrid` to avoid its broken
`solint="int"` path. The script uses:

```text
solint = 10m
interpolation = linear
```

The function is constant, so knot spacing does not change the realized gain.
It only satisfies the current gain-table builder contract.

The base seed is `20260909`:

```text
amplitude variant seed = 20260910
phase variant seed = 20260911
```

The curves are constant, but seeds are still supplied to CASA and recorded for
consistent provenance.

## Use of the current corruption framework

The driver uses the reusable prototype, not any function from the exploratory
`corruption_*` scripts:

```python
from scripts.corruption import AntennaGainCorruption
from scripts.corrtab_utils import GCOLS, GTabQuery
from scripts.timegrid import TimeGrid
```

It supplies a small duck-typed constant curve because the current
[`scripts/corrfn.py`](../scripts/corrfn.py) has no compatible constant model.
The gain-table query is:

```text
ANTENNA1 == selected antenna ID
group by ANTENNA1
```

The framework first creates an identity G table, replaces only the selected
antenna rows, and applies it with:

```text
type = G
interpolation = linear
calwt = False
```

The builder always writes `images/corruption_function.png` relative to the
working directory. The driver temporarily enters each variant directory so
that this side effect remains inside the new experiment and the two plots do
not overwrite one another.

## Copy and mutation safety

For each variant the script performs:

```text
source simulated MS
        |
        +-- copy --> constant_amplitude/0012-399_constant_amplitude.ms
        |
        +-- copy --> constant_phase/0012-399_constant_phase.ms
```

`AntennaGainCorruption.apply_corrtable()` modifies only those copies. The
driver refuses an existing experiment output directory and creates a new
timestamped directory by default.

The underlying current helper deletes an existing gain-table path, but the
driver always supplies a new path inside a newly created variant directory.

## Noise and weight semantics

The source MS already contains the simulated point source and thermal noise.
This experiment therefore applies:

```text
DATA_before = sky + thermal_noise
DATA_after = G(DATA_before)
           = G(sky + thermal_noise)
```

This is not the alternative physical ordering `G(sky) + thermal_noise`.

The current framework uses `calwt=False`, so it preserves the source
simulation's `SIGMA`, `WEIGHT`, and `WEIGHT_SPECTRUM` values.

For the phase-only case, a unit-magnitude rotation preserves the distribution
and variance of circular Gaussian thermal noise. The retained weights remain
consistent.

For the amplitude case, gains scale both source and noise on affected baselines
by 1.50, but the statistical weights remain unchanged. Interpret this as an
uncorrected residual amplitude-calibration error with nominal weights, not as
a new self-consistent receiver-noise model.

## Imaging

Both corrupted copies use the current imaging package:

```python
from scripts.imaging import DefaultImagingConfig, image_ms
```

The driver supplies:

```text
DefaultImagingConfig
image size copied from the thermal experiment: 256 x 256
exact source metric region: 3.0 to 6.719983535034756 beams
```

This retains Stokes I MFS, standard gridding, MT-MFS with `nterms=1`, Briggs
`robust=0.5`, and the central six-beam-diameter CLEAN mask. Reusing the exact
metric region makes baseline, amplitude, and phase QA directly comparable.

Each result gets its own dirty, CLEAN, and residual CASA images, PNGs,
`qa.json`, and `qa.txt` beneath its variant directory.

## Reporting

The driver uses `scripts.reporting.QuartoReporter` and:

[`scripts/reporting/single_amp_vs_phase_corruption_simulation.qmd`](../scripts/reporting/single_amp_vs_phase_corruption_simulation.qmd)

The main report comparison is one matrix with exactly three rows for the same
`0012-399` dataset and three image columns:

| Row | Dataset/case | Dirty | CLEAN | Residual |
|---:|---|:---:|:---:|:---:|
| 1 | `0012-399` clean simulation with thermal noise | image | image | image |
| 2 | `0012-399` with constant amplitude-only error on one antenna | image | image | image |
| 3 | `0012-399` with constant phase-only error on the same antenna | image | image | image |

The report also includes:

- copied uncorrupted baseline dirty/CLEAN/residual PNGs;
- amplitude-corrupted dirty/CLEAN/residual PNGs;
- phase-corrupted dirty/CLEAN/residual PNGs;
- the realized gain-function plot for each variant;
- the selected antenna and exact errors;
- clean peak, residual RMS, residual scaled MAD, robust dynamic range, and
  metric-region bounds; and
- source paths, operation ordering, seeds, and weight policy.

At the very end, after the gain diagnostics and other report content, it repeats
only the three CLEAN images in one side-by-side row ordered as:

```text
thermal simulation | amplitude-only error | phase-only error
```

This final panel is a visual comparison aid; it does not replace the complete
three-row dirty/CLEAN/residual matrix above.

Baseline PNGs and QA are copied into the new experiment. Nothing is written to
the source thermal result directory.

## Output layout

```text
collect/experiments/single_amp_vs_phase_corruption_simulation_<timestamp>/
├── report.json
├── report.qmd
├── report.html
├── baseline/default_imaging/
│   ├── dirty.png
│   ├── clean.png
│   ├── residual.png
│   ├── qa.json
│   └── qa.txt
├── constant_amplitude/
│   ├── 0012-399_constant_amplitude.ms/
│   ├── 0012-399_constant_amplitude.G/
│   ├── images/corruption_function.png
│   └── default_imaging/
│       ├── dirty.png
│       ├── clean.png
│       ├── residual.png
│       ├── qa.json
│       └── CASA image products
└── constant_phase/
    ├── 0012-399_constant_phase.ms/
    ├── 0012-399_constant_phase.G/
    ├── images/corruption_function.png
    └── default_imaging/
        ├── dirty.png
        ├── clean.png
        ├── residual.png
        ├── qa.json
        └── CASA image products
```

## Running the experiment

From the repository root:

```bash
/Users/u1528314/Applications/CASA.app/Contents/MacOS/casa \
  --nogui --nologger \
  -c scripts/single_amp_vs_phase_corruption_simulation.py
```

To pin the paths and antenna:

```bash
/Users/u1528314/Applications/CASA.app/Contents/MacOS/casa \
  --nogui --nologger \
  -c scripts/single_amp_vs_phase_corruption_simulation.py \
  --source-run collect/experiments/extracted_thermal_simulation_comparison_20260909T005739 \
  --output-dir collect/experiments/my_amp_phase_check \
  --antenna ea01
```

The script prints its resolved source, simulated MS, selected antenna, output
directory, per-variant status, and final report path.

## Validation requirements

Fast tests should verify discovery, rejection of missing products, constant
curve behavior, output overwrite refusal, and manifest paths. A real-CASA run
should additionally verify:

1. the source simulation remains unchanged;
2. both copies retain flags, UVW, row count, channel layout, and weights;
3. only baselines touching the selected antenna change;
4. amplitude-case visibility ratios have magnitude 1.50;
5. phase-case ratios have magnitude 1 and phase magnitude 45 degrees;
6. the phase-only case preserves closure phase;
7. all six corrupted-image PNGs, both gain plots, both QA records, and the HTML
   report exist; and
8. all three QA records use identical metric-region bounds.

## Known limitations

- This exercises the current corruption prototype rather than fixing its API.
- It broadcasts the same gain across correlations and channels.
- It does not test simultaneous amplitude and phase errors.
- It does not implement `G(sky) + thermal_noise` ordering.
- The deliberately strong +50% amplitude and +45-degree phase errors are
  controlled, visually significant diagnostics, not a fitted VLA error
  population or recommended realistic defaults.
- It compares imaging outcomes but does not attempt recovery with `gaincal`.
