# Dataset v2: proposed update

**Status: proposal only; no generator changes or simulation launch.** This revises the sampling, retention, parallel execution, and candidate targets/features in [the existing design](dataset_v2_three_task_design.md). Preserve historical runs and use a fresh versioned run for implementation.

**Motivation:** the [snapshot audit](../ml/runs/20261105_damage_audit/report.html) reduced pooled validation RMSE from 0.1447 to 0.1020 using saved CLEAN diagnostics and target scaling. This is exploratory validation evidence. New sampling and dirty-image targets remain untested. Pooled RMSE mixes active and absent rows; reports for this plan always also give active-row and absent-row RMSE per damage type (§5).

## 1. Generation changes

- **Matched combination groups:** sample one `(A, phase, r, antenna)` recipe, then generate all eight presence masks with identical parameters wherever active. Reuse these cases for every matched removal. Gains still affect noiseless visibilities before additive noise; amp and phase use the same antenna.
- **Budget (configurable, increased):** `recipes_per_source = R` with one shared noise realization gives `7R + 1` cases per source before deduplication (the control is shared), plus the noiseless reference. Every case is an example with all three damage labels observed (`4R` active labels per output); no inverse-observation weighting is needed. Because the eight masks of one recipe share parameters, rows within a recipe are strongly correlated and each source has only `R` distinct strengths per type. Start from a larger budget than the earlier six recipes (default `R = 12`, i.e. 85 cases per source) and keep `R` a run parameter so it can be raised further.
- **Coverage:** use stratified/LHS recipes across sources, balancing attenuation/amplification and phase signs. Inspect pooled coverage of distinct parameter values per type. Keep large phase errors, add weak injections near phase zero and gain one, and report severity bands separately. Freeze new bounds from a training-only pilot; CLEAN-derived bounds do not automatically suit dirty-image targets.
- **Keep negative damage:** retain cases where removing a corruption worsens recovery. Do not clip labels or filter examples by outcome. Preserve source-disjoint splits; all recipes, combinations, antennas, and seeds from a source stay together.
- **Retain all imaging products:** keep both native CASA image tables and their FITS exports, including intermediate imaging products (dirty, raw CLEAN model, residual, restored, PSF, primary beam and weight images where produced), with WCS, beam, units, support, QA, and convergence records. **No FITS image or other imaging subproduct is deleted after commit**, so further features can be computed later without re-simulating. Only disposable visibility scratch (per-case Measurement Set copies) and temporary CASA working lattices that are not retained image products may be removed after a verified commit. Fixed per-source reference products are stored once per source. Measure retained bytes, peak scratch and memory per pilot unit before choosing concurrency and checking the full-run storage budget.
- **Collected data stay unchanged:** the generator only reads the collected source Measurement Sets and must work on copies; it must never modify or delete anything under `collect/`. No additional verification step is required.

## 2. Parallel execution

Generation must run as many independent processes, on one machine or across cluster jobs (CHPC via Slurm; scheduler details deferred).

- **Work unit:** one sampling template, i.e. one `(source, recipe)` group: its seven corrupted mask cases. The per-source control and noiseless reference are a separate per-source unit. Units share no mutable state.
- **Stages:** (1) per-source preparation (reference, masks, nominal noise), parallel over sources; (2) planning, a fast deterministic step that commits the full immutable unit list and the shard assignment; (3) unit execution, parallel over units; (4) a single join/verify step that builds the tabular split after all units commit. Stage 3 starts only after both the source's preparation and the full plan/shard assignment are committed.
- **Partitioning parameters:** `--num-shards N --shard i` selects a deterministic, disjoint subset of units (e.g. one Slurm array task per shard), and `--parallel P` runs up to `P` unit processes inside that shard. Sharding is size-balanced by the parent Measurement Set size so shards finish at similar times.
- **Independence and restart:** each worker owns whole recipe groups, including all seven cases, and each unit uses its own scratch directory (`--scratch-root`, e.g. node-local storage). Outputs are written atomically and sealed on commit. Rerunning a shard skips sealed units and redoes unsealed ones. Use no global or per-unit locks: allow only one live attempt per shard, confirm the previous attempt and its child workers have stopped before an overlapping retry, and stop all existing workers before changing shard assignments. Repartitioning changes only the assignment, not the immutable simulation plan. A commit seal marks completion; it does not prevent overlapping live attempts.
- **Determinism:** unit outputs depend only on the simulation plan and unit identity, not on shard count, ordering or `P`. Derive noise seeds from `(run_seed, source, noise_slot, purpose)`, **never recipe**; every recipe and the shared control use the same nominal noise realization within a source/slot. Derive parameter/antenna sampling seeds from `(run_seed, source, recipe, purpose)`. The default has one noise slot; additional pilot slots each require their own shared control.

## 3. Targets to retain or add

Use fixed support and imaging weights/grid across each source's cases. Let `D0` be its noiseless uncorrupted **dirty image**, `D` the observed dirty image, `C` the convolved CLEAN model, and `T` the restoring-beam-convolved true sky. Define `eC = ||C − T||₂ / ||T||₂` and `eD = ||D − D0||₂ / ||D0||₂`. Subscript `−k` means the matched case with corruption `k` removed; noise removal returns to nominal noise using the same realization.

| Target | Definition | Purpose |
|---|---|---|
| Presence, retain | Three injection indicators `b_k` | Detect each corruption independently |
| **CLEAN recovery, primary** | `eC`, `dC_k = eC_all − eC_−k` | Total recovery error and signed benefit of removing each type |
| Dirty perturbation, computed and stored | `eD`, `p_k = ‖D_all − D_−k‖₂ / ‖D0‖₂` | Total dirty error and nonnegative per-type image change, for later experiments |

**Primary severity target: signed model damage `dC_k`.** The dirty targets `eD` and `p_k` are computed and retained for every row but are not the first modelling target. `p_k` measures change, not improvement; its components need not add to total error. `eD` includes nominal thermal noise, so controls need not have zero total error. Optionally retain signed `D − D0` as a future image-prediction target.

## 4. Features

Inputs must use only the full observed case and inference-available metadata. Truth, injected parameters, selected antenna identity, controls, and removal images are **label/audit information only**.

The feature set has three groups: **combined8** (baseline; residual summaries, restored peak and nominal-noise reference), **dirty6** (the first six features evaluated using the dirty image as specified below), and **model6** (CLEAN-model flux).

### 4.1 combined8 (unchanged baseline)

The first six columns (**combined6**) summarize images: column 1 combines the restored-image peak with residual MAD, and columns 2–6 use the residual. Column 7 is nominal-noise metadata; column 8 combines measured residual MAD with that reference.

| # | Feature | Calculation |
|---|---|---|
| 1 | `dynamic_range_scaled_mad` | max of the restored image (valid pixels) / residual scaled MAD in the QA region |
| 2 | `residual.scaled_mad_jy_per_beam` | `1.4826 × median(abs(r − median(r)))` over the QA region |
| 3 | `residual.peak_over_scaled_mad` | `max(abs(r))` / scaled MAD |
| 4 | `residual.p99_5_over_scaled_mad` | 99.5th percentile of `abs(r)` / scaled MAD |
| 5 | `log1p_even_energy_over_mad2` | `log1p(mean(even(r)²) / s²)`, `even(r) = (r + r_flipped)/2` about pixel (128, 128), `s` = scaled MAD in the 32–72 px annulus |
| 6 | `log1p_odd_energy_over_mad2` | same with `odd(r) = (r − r_flipped)/2` |
| 7 | `sigma_NA` | nominal natural-weight image noise from the source Measurement Set (not an image measurement) |
| 8 | `sigma_sNA_units` | residual scaled MAD / `sigma_NA` |

**Parity implementation:** reuse [the existing function](../ml/nn_common.py) exactly on native 256×256 images. It pairs `image[1:, 1:]` with its flip on both axes, reflecting this 255×255 interior about original pixel `(128, 128)`; energy means also exclude the first row and column. Flipping the full 256×256 array would instead reflect about `(127.5, 127.5)` and change the feature. Annulus pixels satisfy `32 <= radius < 72` about `(128, 128)`.

`sigma_NA` reuses `predicted_image_rms_jy_per_beam`: `sigma0 / sqrt(N_valid)`, where `sigma0` is the nominal per-real/imaginary visibility-component noise STD and `N_valid` counts parallel-hand correlation/channel/row samples with both `FLAG` and `FLAG_ROW` false ([implementation](../scripts/simulation/noise.py)). It is a natural-weight reference under the constant-noise assumption, not the measured RMS of the Briggs-weighted corrupted image; do not multiply it by the injected noise factor.

### 4.2 dirty6 (new)

Recompute features 1–6 with the observed dirty image `D` in place of the residual, using the same functions, QA region, annulus and parity centre:

| Feature | Calculation |
|---|---|
| `dirty.dynamic_range_scaled_mad` | max of `D` (valid pixels) / scaled MAD of `D` in the QA region |
| `dirty.scaled_mad_jy_per_beam` | scaled MAD of `D` in the QA region |
| `dirty.peak_over_scaled_mad` | `max(abs(D))` / scaled MAD of `D` |
| `dirty.p99_5_over_scaled_mad` | 99.5th percentile of `abs(D)` / scaled MAD of `D` |
| `dirty.log1p_even_energy_over_mad2` | parity energy of `D`, scaled by the annulus MAD of `D` |
| `dirty.log1p_odd_energy_over_mad2` | same, odd part |

Rationale: CLEAN can absorb coherent corruption into its model; dirty6 tests whether information lost from the residual helps predict damage. For a point source centered on the parity centre, the complex-gain perturbation is `A*exp(i*phi) − 1 = (A*cos(phi) − 1) + i*A*sin(phi)`: the real term contributes even structure and the imaginary term odd structure. A pure amplitude error is even, but a phase error generally contributes **both**. Phase errors are predominantly odd only in the small-angle limit at `A = 1`; at ±180° their perturbation is entirely real/even. Dirty even energy also includes the uncorrupted source and PSF, and squared energies discard signs. Treat dirty6 as an untested feature ablation, not a guaranteed amplitude/phase separator. Its first and third features can be similar; retain both for the shared six-feature schema.

### 4.3 model6 (new)

Let `M` be the raw Stokes-I CLEAN model in **Jy/pixel**, before restoring-beam convolution. All sums use the fixed valid support. **Total CLEAN model flux means `F = sum(M)` in Jy, not the pixel mean.** The existing `final_model_flux_jy` comes from CASA's `modelFlux` summary; calculate these new sums directly from the retained model and cross-check that summary.

| Added feature | Exact calculation |
|---|---|
| Positive model flux | `F_pos = sum(max(M, 0))` |
| Negative model flux magnitude | `F_neg = sum(max(-M, 0))`, reported as a nonnegative number |
| Absolute model flux | `F_abs = sum(abs(M))` |
| Model flux inside beam aperture | `F_in = sum(M[inside])`, signed |
| Model flux outside beam aperture | `F_out = sum(M[outside])`, signed |
| Total model flux / restored peak | `F / P`, where `P = max(restored_image)` |

These sums are deliberately simple and partly redundant: `F = F_pos − F_neg = F_in + F_out` and `F_abs = F_pos + F_neg`. No beam-area conversion is applied to the raw model. This direct sum would **not** be the integrated flux of a restored image in Jy/beam. Empty models have zero flux sums. Use a missing ratio when `P <= 0` or nonfinite; do not discard the case. The existing features also give `P = dynamic_range × residual_MAD`.

**Beam aperture:** a Gaussian has no sharp edge, so use its half-maximum (FWHM) ellipse centered at the **observation phase centre**. Obtain east/north angular offsets `(E, N)` from WCS. With `theta = radians(BPA)`, set `u = E*sin(theta) + N*cos(theta)`, `v = E*cos(theta) − N*sin(theta)`, then `inside = (u / (BMAJ/2))² + (v / (BMIN/2))² <= 1`. All angular lengths must use the same units. `outside` is the remaining valid support. Use the recorded beam/phase centre, never a fitted source centroid or the simulated truth position.

No PSF-sidelobe, centroid, moment or generic observation-context features are proposed for the first implementation. Because all imaging products are retained (§1), further features can be added later without re-simulating. The earlier eight-diagnostic audit result is not a measured result for this revised feature set.

## 5. Small validation sequence

1. Pilot matched groups on several training sources spanning SNR/UV coverage; repeat noise seeds on a subset. Verify gain/noise pairing, complete removal joins, absent-type zeros, fixed geometry, retained products, and that sharded runs with different `N`/`P` in separate output directories give identical scientific unit outputs. Measure retained storage, peak scratch and memory before setting production concurrency.
2. Compare combined8, combined8 + dirty6, combined8 + model6, and combined8 + dirty6 + model6 (20 features) using source-grouped training folds, on the primary target `dC_k`. Fit regression targets in training-only RMS units and decode predictions for reporting.
3. Report, per damage type and in damage units (RMSE): active-row error (injection present), absent-row error (injection absent, true damage 0), pooled error, training-constant baselines, mixture/severity slices, and source-bootstrap uncertainty. Include the **leakage metric**: predicted damage of one type on rows where that type is absent but another is present, in particular predicted amp damage on rows with a phase error and no amp error, and predicted phase damage on rows with an amp error and no phase error. Freeze features/ranges/model selection before reserved-test evaluation. Scale generation only after the pilot establishes useful target coverage and affordable retention.
