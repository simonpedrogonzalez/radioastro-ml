# V2 section 1: make the features visible

Implement only section 1 of the vault note **RadioastroML V2 Setup – Directional
Features and Corruption Regression**. Background read: **Imaging,
Self-Calibration, and Directional Features**, **Jeff Directional Fourier Features
– Assessment and Literature Review**, **Visibilities as a Dataset**, and the
historical **RadioastroML Dataset 1** note. The current V1 index is authoritative.

## Scope and implementation

1. `scripts/report_fourier_toy.py`: a 256 × 256 cosine, sine, their sum, doubled
   amplitude, and half-bin example. Show image, real/imaginary Fourier response,
   expected ±uv coordinates, direct dot-product agreement, and amplitude/power
   scaling. Use signed angular WCS increments and the explicit phase-centre
   factor with `norm="ortho"`; test a noninteger centre and rotated grid too.
2. `scripts/report_v2_pilots.py`: select three **indexed training** observations
   with distinct sampling: `0005+383`, `0012-399`, `0201-115` (0.094,
   0.186, 8.290 arcsec pixels). Their parent MSs and observation project IDs
   are distinct. Reuse existing
   simulation, constant-gain feasibility/measurement, FITS, and imaging code.
   Generate five new variants per source and seed: no gain, amplitude, phase,
   combined amplitude/phase, and twice the thermal standard deviation. Use
   three independent seeds and one fixed antenna per source. Aim for each
   pure gain component's SNR_corr = 30; record actual gains and feasibility.
   Add one single-baseline diagnostic on the first source.
3. Produce one HTML report per observation. Each has residual → Fourier power
   with uv coverage → baseline score matrix, antenna ranking, noise/coverage
   controls, cross-seed comparison, and a candidate selector showing measured
   visibility amplitude/phase versus time. Mark injected tracks independently
   of score colours. The revised report removes dirty-image differences and
   identifies the known injection in plain text.

Use the minimal geometry from section 3 solely to draw section 1: resolve
DDID → SPW/channel frequency and correlation type; retain valid unflagged
parallel-hand cross-correlations; bilinearly associate samples with FFT cells,
fold conjugates, exclude DC/Nyquist, and report out-of-domain coverage. Baseline
score is coverage-weighted mean Fourier power; antenna score is the median of
partner scores. These are inspection scores, never probabilities or fitted gains.
Do not build the full V2 feature bank, extend the learning dataset, fit regressions,
or implement physical gain fitting. Figure 3 is explicitly deferred because it
requires the later regression stage; no fabricated prediction plot.

## Invariants and products

- Read V1 through `dataset.json` and parent provenance, never recursive sample discovery.
- All selected sources must be training sources; inspect shared-parent provenance.
- Write new products under `collect/experiments/v2_visible_features/`; preserve V1,
  original MSs, source partitions, and the vault.
- Preserve native 256 × 256 pixels, phase centre, original reference image-noise
  scale, flags and imaging selection. Keep weights fixed even for doubled noise.
- Save seeds, geometry, actual settings, injection checks, visibility evidence,
  and measured timings. Any display subsampling must be labelled; scores use
  all valid samples. A time plot shows baseline measurements, not antenna gains.
- Reports explain equations, colour scales, controls, limitations, and whether
  localization patterns repeat. Shared uv cells can implicate several baselines.

## Validation and run

### Pilot report revision requested after inspection

- Rerun only `0005+383` into `collect/experiments/v2_visible_features_revised/`.
  Scientific scores and injections stay as defined above.
- Reuse `scripts.imaging.plot_utils.casa_image_to_png` and
  `shared_fits_display_limits`: residual in mJy/beam, inferno palette, RA/Dec,
  one fixed display range across all cases.
- Enlarge Fourier plots and remove grey sample overlays. Use standard `Reds`
  for Fourier power, baseline matrix, tracks and visibility points. Share
  limits and display `ln(1 + value)` everywhere: pixel power P and its baseline
  average B are different values, explicitly labelled.
- Use one candidate selector, show only its tracks with uniform larger dots,
  and identify injection truth in text. Retain actual coordinates; thin only
  the displayed uv points to reduce crowding. All samples still enter scores.
- Fix amplitude and phase y limits from the pooled displayed visibility
  minima/maxima across all cases and candidates; mark those global bounds.
  Keep actual times, including empty gaps.
- Show antenna score A directly in the outcome summary, instead of a ratio
  to a no-gain case. Remove its horizontal reference lines. Remove dirty
  differences, model subtraction selector, seed IDs and reproducibility text.
- Write a Vault note explaining R, s₀, Z, P, coverage H, baseline score B,
  antenna score A, ranking and the display logarithm, with every symbol defined.
- Check plotting recipes, unchanged score calculations, fixed y limits,
  consistent colour mapping and candidate interactions; inspect exported plots.
- Compact layout: retain full-resolution images but fit the residual, Fourier
  power and baseline matrix in one HTML grid row. Below, show four vector
  ranked bar plots (antenna raw/log and baseline raw/log), updated by the case
  selector. Mark injected antennas/affected pairs in teal; controls have no
  marks, and a single-pair injection marks no antenna. Ranked score axes start
  at zero and fit the selected case; visibility axes remain fixed across cases.
  Click-to-enlarge preserves readability; small phone screens stack.
- Fit the uv and paired time plots into one HTML row as well. This layout and
  ranking addition uses retained evidence; no new simulation is needed.

```sh
/Users/u1528314/Applications/CASA.app/Contents/MacOS/casa --nogui --nologger -c scripts/report_v2_pilots.py --generate-only --source 0005+383 --output collect/experiments/v2_visible_features_revised
ml/.venv/bin/python scripts/report_v2_pilots.py --report-only --source 0005+383 --output collect/experiments/v2_visible_features_revised
node tests/check_v2_report_interaction.cjs collect/experiments/v2_visible_features_revised 0005+383
```

Revision executed: all 16 cases were freshly generated for this MS. Seven
analytical/display tests and all 16 case interactions passed; fixed visibility
ranges, shared colours and imaging plot recipes were checked. The new FFTs
match the original run, and the injected antenna remains first in all nine
gain cases. Static plots were inspected; browser layout remains unverified
because no browser connection was available. Open
`collect/experiments/v2_visible_features_revised/pilot_0005+383.html`.
Score definitions are in the Vault note **Pilot Scores - From Residual Pixels
to Antenna Ranks**. Earlier reports remain archived in the original folder.

Run analytic FFT/sign/WCS checks before CASA. Check injection strength, shared
noise, fixed geometry/weights, valid coverage, and training-only membership.
Generate actual CLEAN residuals for all pilot cases; report failures of the
localization hypothesis honestly. Inspect report figures and interactive
selectors. Provide HTML paths and exact rerun commands after execution.

### Commands

Run from the repository root. CASA generates/resumes the simulations; the ML
environment renders the HTML. Completed cases are reused after checking the
configuration, so rendering changes do not require rerunning the simulations.

```sh
ml/.venv/bin/python scripts/report_fourier_toy.py
/Users/u1528314/Applications/CASA.app/Contents/MacOS/casa --nogui --nologger -c scripts/report_v2_pilots.py --generate-only
ml/.venv/bin/python scripts/report_v2_pilots.py --report-only
ml/.venv/bin/python -m unittest discover -s tests -p test_v2_visible_features.py -v
node tests/check_v2_report_interaction.cjs collect/experiments/v2_visible_features
```

Use `--source 0005+383` to generate/render just one pilot. CASA's `--audit-only`
mode repeats the exact parent-versus-model sampling check. Use `--output PATH`
on both report scripts for a separate experiment. Each pilot retains its
noiseless `model.ms`, geometry, JSON checks, per-case FITS and compact evidence;
temporary variant/noise MS copies are removed after those products are saved.

### Executed result

- **46 cases completed**: 45 controlled cases plus one single-baseline example.
  The injected antenna ranks first in all 27 gain cases; the injected single
  pair ranks first among its 300 baselines. These are strong training-only
  demonstrations, not detection accuracy or gain estimates.
- Doubling noise raises the reference antenna's score by approximately 4×
  (3.98–4.08× here) without changing its rank. Noise still creates candidates.
  Coverage-only reference ranks are 1, 20 and 10 for the three sources.
- Six analytical/selection tests passed. Parent flags, rows, UVW, times,
  channel mappings and phase centres match exactly. All 46 saved FFTs/scores
  were independently recomputed; generation checked strengths, noise checksums,
  fixed weights, native WCS, finite support and matching PSFs.
- All 46 case selectors, antenna/pair selections, measurement modes and hover
  lookups passed a JavaScript DOM/canvas test. Static figures were visually
  inspected and embedded resources verified. **Browser layout testing remains
  unverified:** the browser connection was unavailable.
- CASA's empty-model notices come from zero-iteration dirty imaging; Astropy's
  date/observatory WCS notices normalize inherited metadata. Scientific checks
  passed; changing those existing library notices is outside this task.

Open `collect/experiments/v2_visible_features/index.html` first. It links to
`toy_fourier.html`, `pilot_0005+383.html`, `pilot_0012-399.html` and
`pilot_0201-115.html`. The four reports are self-contained and work offline;
keep the five HTML files together to preserve the index links. Figure 3 remains
deferred until the later fitting stage is requested.
