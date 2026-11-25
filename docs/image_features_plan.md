# Image features (energy, radial, FFT): implementation and experiment plan

Source of the ideas: vault note `Inbox/Features to try Summary.md`, Conclusion items 1–3 (item 3 is the
2026-11-24 version, **without** the optional smooth fit of the reference). This plan covers:
(a) extracting the three feature blocks, (b) a per-case report that shows what the features see, and
(c) one experiment: all22 + each block, single `xgb_mlp`, on the original v2 train/val.

Status: **plan only, nothing implemented.**

## 0. What exists and what is reused

| Need | Existing code | Use |
|---|---|---|
| Per-case planes and paths | `cases/<id>/case.json` → `products` (`model`, `clean`, `residual`, `dirty`, `psf` FITS) | read only |
| Fixed support Ω, truth, beam radius ρ | `sources/<sid>/recovery/reference_masks.npz` (`omega`, `rho`, `truth`, `source`) | `rho` is already centred on the **known source position** in elliptical beam units, ρ = 1 at one FWHM (`recovery_products.py` → `geometry`). In v2, Ω is the full 256×256 plane. |
| Convolved model C | `scripts/clean_recovery_noise_sweep.convolve_model(model, header)` | same C as eC uses |
| Raw model flux sums (Jy) | `scripts/dataset_v2_matched_extract.model6` | first 5 values divided by s |
| Nominal RMS | `plan.json` → `prepared_sources[].nominal_rms` | lower bound of s |
| Header-based FFT and (u, v) | `scripts/report_fourier_toy.phase_centered_fft` (tested in `tests/test_v2_visible_features.py`) | transform and uv grid. No MS metadata. |
| Loading/eC check pattern | `ml/val_error_cases.case_images` | pattern for the eC reproduction check |
| Training families | `ml/multitask.py` (`load_split`, `_fit_tree_family`, `fit_mlp`, `decode`, `task_metrics`) | reused. Only change: extract the expert/OOF step from `fit_data` into a function (§3.1). |
| Bootstrap helpers | `ml/compare_val_by_ec.bootstrap_weights`, `weighted_metrics` | paired source bootstrap |

**Invariants.**
- No existing `<split>.npz`, case file or image is modified. v2.1 manifests hash `train.npz`, so the new features go into a **separate table** joined by `case_ids`.
- Targets, loss scales, the feature view (all22), `--with-dirty` and the model settings stay unchanged.
- Features never use the truth image. ρ only encodes the known source position, which is available at inference.

## 1. Feature definitions (exact)

**Shared extraction**, per case:
- R = residual, I = restored (`clean`), C = `convolve_model(model, header)`.
- Scale s = max(max |I| over Ω, nominal RMS). C and R are divided by s.
- All statistics are taken over Ω.

**Block `energy`** (16 features)
- `energy.log10_s`
- For x ∈ {C/s, R/s}, with names `energy.{C,R}.<stat>`:
  - `rms`;
  - `mean_abs`;
  - `pos_ms` = mean(x²·[x > 0]);
  - `neg_ms` = mean(x²·[x < 0]);
  - `max_abs`.
- The 5 Jy-valued `model6` sums divided by s: `energy.model.{flux_positive, flux_negative_magnitude, flux_absolute, flux_inside_beam, flux_outside_beam}_over_s`.
- An empty model gives zeros, not missing values.

**Block `radial`** (25 features; option B: regions differ per channel)
- Regions are in ρ (elliptical beam units from `reference_masks.npz`, ρ = 1 at one FWHM, centred on the known source), intersected with Ω.
- **C channel** (convolved model), regions [0, .5), [.5, 1), [1, 2), [2, 4), [4, edge). Names `radial.C.r<k>.<stat>`:
  - `pos_ms` and `neg_ms`, the mean of x²·[x > 0] and x²·[x < 0] over the region's pixels;
  - `frac`, the region's share of C's total squared energy, Σ_r x² / Σ_Ω x².
  - 15 features.
- **R channel** (residual), regions [0, 2) and [2, 4) inside the CLEAN search mask, [4, 8) just outside it, [8, 16) and [16, edge) for the far field. Names `radial.R.r<k>.<stat>`:
  - `pos_contrast` = mean_r(x²·[x > 0]) / mean_Ω(x²);
  - `neg_contrast` = mean_r(x²·[x < 0]) / mean_Ω(x²).
  - This is energy per pixel relative to the whole field, so the value does not depend on the region's area.
  - The field level itself is block 1's R RMS, so it is not repeated.
  - 10 features.
- **Why these choices** (measured on the local smoke data, one source 0449+113, 84 + 28 corrupted cases against its control):
  - **C only changes inside ρ ≈ 4.** The CLEAN search mask reaches ρ ≈ 4–5 (0449+113: 5.3; 2241+098: 4.2), so model components cannot lie farther out. The change in C relative to the control is exactly 0 from ρ = 4 outward.
    - Noise-only cases put 78% of the model change in [0, .5) and 21% in [.5, 1).
    - Gain-only cases spread it: .34 / .19 / .17 / .13 over the first four regions, and up to 30% in [1, 2) for strong gains.
    - The C tail region [4, edge) is kept only as a check that it stays near zero.
  - **R's energy share per region equals the region's pixel share.** For example, 8–16 / 16–24 / 24–32 / 32+ beams hold .27 / .33 / .26 / .06 of R's energy and .27 / .33 / .25 / .065 of the pixels. A `frac` feature for R would encode geometry, not corruption, so R uses the density contrast instead.
  - **The contrast differs near the source.** In [2, 4), gain-only cases have about 1.4× the field's energy per pixel and noise-only cases about 0.66× (CLEAN removes the noise inside the mask). Beyond 4 beams both are about 1×.
  - The images span about 40–64 beams across (edge midpoints at ρ 19–32, corners at 41–45; 7 local sources), so R's far field is split at 16.
  - The far-field regions are cut by the square image, so their shape differs per source; per-pixel quantities are still comparable.
- These measurements come from one source and are re-checked on the full training set before training (§2.1).
- Missing (NaN) values: an empty region gives NaN, and so do `frac` and the contrasts for a channel with zero total energy. XGBoost handles NaN natively; `fit_mlp` already imputes it with the training mean.

**Block `fft`** (16 model features plus stored curves)
- **Transform.** For x ∈ {C/s, R/s}: y = x − mean_Ω(x), then `z, u, v = phase_centered_fft(y, header)` and P = |z|².
- **Cells.** Keep the independent cells: one of each Hermitian pair, self-conjugate cells once, DC excluded. That is N²/2 + 1 cells for an even N×N grid. Normalize p = P / ΣP and sort it in decreasing order.
- **Cumulative top-k shares** S_k = Σ of the k largest p, for k ∈ {1, 3, 10, 30, ⌈0.001 N_cells⌉, ⌈0.01 N_cells⌉}. S_1 is peak/total and S_{1%} is the top-1% share.
- **Stored per case, not fed to the model:**
  - `log10 S_k` (6 values per channel);
  - a ranked curve `log10 S` at 24 log-spaced rank fractions, for the gate and report plots.
- **Excess, computed at training time** from the training rows with mask 0 (controls) only, per channel and k:
  - `fft.{C,R}.excess_z.k<k>` = (log10 S_k − median_ctrl) / (1.4826 · MAD_ctrl).
- **Supporting features:**
  - `fft.{C,R}.peak_cycles_per_beam` = |(u, v)| of the top cell × √(BMAJ·BMIN) in radians;
  - `fft.{C,R}.angular_concentration` = |Σ P e^{2iθ}| / Σ P with θ = atan2(v, u). Over the half-plane, e^{2iθ} is identical for both members of a Hermitian pair, so this is well defined.
- A zero-power channel (empty model) gets all of its `fft.*` features as NaN.
- No window, no local-excess feature, no raw phase. Total power is not added, because it equals the RMS² of block 1 (Parseval's theorem).

**PSF-whitened residual variant (`fft.Rq`, decided by the gate).**
- The PSF is the image of the gridded, weighted uv sampling, so P_PSF = |FT(PSF)|² is the case's own coverage map. It is an archived imaging product (`psf.fits.gz`), not MS metadata.
- The residual satisfies FT(R) = S·(V − V_model). Its power therefore sits on the sampled tracks even for normal data, which is why raw P_R is spiky.
- The variant ranks q(k) = P_R(k) / P_PSF(k) instead of P_R, over the independent cells with P_PSF ≥ 10⁻³ · max P_PSF.
  - The same PSF FT is used without mean subtraction.
  - The threshold is fixed, not tuned. Cells outside it are dropped and their count is stored.
- The same S_k, `excess_z` (training-control reference) and supporting features are computed from q, normalized by Σq.
- Only for R. The C spectrum is the model's FT times the restoring Gaussian, not the sampling.
- Expected limits:
  - With Briggs robust 0.5 weighting, noise power per cell scales roughly with Σw², while P_PSF scales with (Σw)², so q is not exactly flat for normal data. The training controls still calibrate it.
  - CLEAN major cycles, gridding and MT-MFS make FT(R) ⊂ support(S) only approximately true.
- Stored per case:
  - `log10 S_k` and the ranked curve for q;
  - the number of kept cells.
- Cost: one extra FITS read and FFT per case.

> **Caveat to keep in mind when reading results.** With one pooled control reference, `excess_z` is an affine
> transform of `log10 S_k` for each feature. Trees and the standardized MLP therefore gain nothing from the reference
> itself: the experiment tests the ranked-concentration information, while the reference makes the values interpretable
> (report, gate). The reference can only add information if it is **conditioned** per case. The PSF-whitened `fft.Rq`
> variant above does that: each case is referenced to its own coverage, and the gate (§2.1) decides whether it replaces `fft.R`.

Feature totals: energy 16, radial 25, fft 16. A combined set is all22 + 57.

## 2. Files to add (small, one job each)

New package `scripts/features/`, alongside `scripts/imaging` and `scripts/corruption`:

| File | Content | ~lines |
|---|---|---|
| `__init__.py` | empty | 0 |
| `planes.py` | `load_case(run, case_id, prepared, masks) -> dict(C, R, I, omega, rho, header, s, model)`, which also returns eC recomputed from C (checked against the stored value) | 40 |
| `energy.py` | `energy(planes) -> dict[name, float]` | 25 |
| `radial.py` | `C_REGIONS`, `R_REGIONS`, `radial(planes) -> dict` | 30 |
| `fourier.py` | `power(x, header)`, `spectrum(power, uv, header) -> (features dict, curve)`, `independent_cells(shape)`, `excess(raw_log_s, controls) -> z`; the q variant is `spectrum(P_R / P_PSF, ...)` on the kept cells | 70 |
| `extract.py` | CLI: one verified run plus a split → a table npz | 70 |

The table, e.g. `runs/image-features-v2/train.npz`, contains:
- `case_ids` and `row_ids`, in `<split>.npz` row order;
- `names` and `X`, with the energy, radial and fft raw columns;
- `curve_C`, `curve_R` and `curve_Rq`;
- `split_npz_sha256` and `plan_sha256`, checked on load.

`extract.py --jobs` uses a process pool. There is no resume: about 10k cases at roughly 0.2 s each is about 30 min serial.

Other files:
- **Report:** `scripts/report_image_features.py` (~200 lines, §4).
- **Experiment:** `ml/image_feature_experiment.py` (~150 lines, §3).
- **Tests:** `tests/test_image_features.py`, unittest (§5).

### 2.1 Gate (before training)

This is the first section of the per-case report:
- Ranked curves of the training controls (median, 5–95% band) against corrupted training rows, grouped by number of corruptions, for both C and R.
- Plus the ratio of the control MAD of `log10 S_k` to the all-row MAD.
- The same panels for the PSF-whitened residual q = P_R/P_PSF, next to the raw P_R panels, with the same control-band tightness ratio.
- **Decision:** if the q control band is clearly tighter than the raw-P_R band (smaller control MAD of `log10 S_k` and better separation of corrupted rows), the experiment's `fft` set uses `fft.Rq` in place of `fft.R`. Otherwise it keeps raw P_R. `fft.C` is unchanged either way.
- The report also shows P_PSF and q maps for the selected pairs (§4), to check visually that the residual power lies on the PSF's support.
- **Tight control band:** keep the pooled reference.
- **Radial check (option B).** Per region, the median share of squared energy against the region's share of pixels, for ΔC and R (relative to each source's control), by corruption type (control, noise-only, gain-only, mixed), over all training sources. Confirms or corrects the one-source findings:
  - C changes stay inside ρ < 4;
  - R's energy share follows the area;
  - R's contrast in [2, 4) separates gain-only from noise-only.

  If a finding fails, the regions are revised before training.
- **Wide band:** flag it, and decide together whether to condition the reference on the PSF spectrum. That would be a separate small change.

## 3. Experiment: does each block, by itself, improve all22 + xgb_mlp?

### 3.1 Only change to `ml/multitask.py`

Move the block of `fit_data` that builds the OOF expert matrix and `stage_train`/`stage_val` into:

```python
def expert_stage(train, val, fitted, jobs):   # fitted holds xgb_independent and xgb_shared
    ...same code, unchanged...
    return stage_train, stage_val, oof
```

`fit_data` then calls it. Behaviour must be identical. Check: save the predictions of the synthetic `ml/tests/test_multitask_fit.py` fit before the change and compare them after (exact equality). The existing tests must still pass.

### 3.2 `ml/image_feature_experiment.py`

```
python -m ml.image_feature_experiment --train runs/dataset-v2-matched-train --val runs/dataset-v2-matched-val \
  --features-train runs/image-features-v2/train.npz --features-val runs/image-features-v2/val.npz \
  --output ml/runs/image-features-xgb_mlp --sets base energy radial fft --seeds 42 43 44 --jobs 8
```

- **Data.** `load_split(..., "all22", dirty=True)` for both splits, exactly as `matched-v2-full-all22-dirty`. Then join the table columns by `case_ids`; every row must be present and the hashes must match.
  - A set is `base` or a `+`-joined list of blocks; e.g. `energy+radial` is allowed later.
  - `fft` adds `excess_z` (reference from the training controls) plus the 4 supporting features, and never the raw `log10 S_k`.
- **Fit, per set.**
  - Fitted once (deterministic): `_fit_tree_family` for `xgb_independent` and `xgb_shared`, then `expert_stage`.
  - Per seed: `fit_mlp(stage_train.X, ..., seed=s)`.
  - Same grid, early stopping and scales as the baseline. Selection uses validation J, as in all earlier runs; this protocol is equally optimistic for every set. Test is untouched.
- **Outputs.**
  - `<output>/<set>/seed<s>/predictions_val_xgb_mlp.npz` plus a `results.json` with `train_rows`, `feature_names`, `best_epoch` and the expert `params`. This is the same layout as multitask runs, so `ml.compare_val_by_ec` can compare the sets directly.
  - `<output>/summary.json` and `report.html`.
- **Report** (one page):
  1. **Per set:** mean over seeds [min, max] of:
     - eC RMSE overall, on eC > 1 and on eC < 0.1;
     - eC R²;
     - d_amp, d_phase and d_noise R² on active rows;
     - presence AUROC.

     All of these come from `task_metrics`.
  2. **Paired Δ against `base`:** the per-seed difference averaged over the seeds, with a 95% source bootstrap (`bootstrap_weights`, 1000 resamples of the 21 validation sources). Shown green or red when the interval excludes 0.
  3. **Sanity check:** `base`/seed 42 must reproduce `ml/runs/matched-v2-full-all22-dirty/predictions_val_xgb_mlp.npz`; report the maximum absolute difference. A mismatch is a finding to explain, not a failure.
- **Reading it.**
  - Seeds only vary the MLP initialisation; the experts are deterministic. So the seed spread understates model variance; the source bootstrap is the main uncertainty.
  - A block "helps" only if its Δ interval excludes 0 on eC RMSE overall, or on the eC > 1 slice without hurting eC < 0.1.

## 4. Per-case report (`scripts/report_image_features.py`)

```
python -m scripts.report_image_features --run runs/dataset-v2-matched-val --table runs/image-features-v2/val.npz \
  --train-table runs/image-features-v2/train.npz --output runs/image-features-v2/case_report
```

The report:
- recomputes the planes for the chosen cases from the images (same `planes.py`);
- uses the train table only for the control reference band and for standardization;
- has the gate (§2.1) as its first section.

**Pair selection, per block** (automatic, overridable with `--pair CASE_A:CASE_B`; repeatable):
- Within one validation source, the source's control is compared with the corrupted case farthest from it in that block's standardized features (z-scored on training rows). Using one source fixes geometry and coverage, so the differences come from the corruption.
- Two pairs per block:
  - the source with the largest such distance;
  - a different source at the 75th percentile, i.e. one extreme and one typical large change.
- That gives 6 pairs (3 blocks × 2). Every pair shows both cases' recipe (mask, A, φ, r, antenna), eC, eD and the all22 values most relevant to that block.

**Per pair**, one section per block:
- **All blocks:**
  - C/s and R/s for both cases, on one shared colour scale per channel;
  - the ρ = 0.5, 1, 2, 4, 8 and 16 ellipses overlaid.
  - The text lists the 3–5 largest standardized changes, e.g. "R neg_ms in [1, 2) beams ×14".
  - I add a written interpretation after the first CHPC run ("they change because …"), tied to the recipe. It is not generated automatically.
- **energy:** side-by-side bars of the 16 values (log y) for case A and case B.
- **radial:** two heat-map grids, shown for A, B and log10(B/A):
  - C: 5 regions × {pos_ms, neg_ms, frac};
  - R: 5 regions × {pos_contrast, neg_contrast}.

  Plus fine-grained energy-per-pixel-vs-ρ profiles of both cases, C and R, overlaid on one axis.
- **fft:**
  - log power of both cases over the half-plane, in cycles per beam, with the top-k cells marked; for R also P_PSF and q;
  - the ranked curve (log S vs rank fraction, log–log) of both cases over the control band;
  - bars of `excess_z` per k, with the supporting features printed.

## 5. Validation

- **Unit tests** (`tests/test_image_features.py`, synthetic, no CASA):
  - energy statistics on a known ±image;
  - radial: regions from a toy ρ; the empty region gives NaN; C's `frac` sums to 1; a uniform R gives every contrast = its sign share (pos + neg = 1 per region);
  - fft: the independent-cell count is N²/2 + 1; Parseval's theorem holds; an integer-period cosine fringe gives S_1 ≈ 1, its frequency and angle are recovered, and `angular_concentration` ≈ 1; isotropic noise gives a small `angular_concentration`; the control median gives `excess_z` = 0; q of a residual equal to the PSF itself is flat (all kept cells equal).
- **Smoke run** on `runs/dataset-v2-matched-smoke-20261105`:
  - extraction runs;
  - eC recomputed from C matches the stored eC (rtol 1e-9);
  - `energy.model.*_over_s` · s equals the npz `model.*` columns;
  - no infinite values; NaN only where defined above.
- **multitask refactor:** identical predictions (§3.1), existing ML tests pass.
- **Experiment:** base/seed 42 reproduces the baseline run (§3.2).

## 6. CHPC commands (in order)

```bash
bin/submit ml-cpu python -m scripts.features.extract --run runs/dataset-v2-matched-train --output runs/image-features-v2/train.npz --jobs 8
bin/submit ml-cpu python -m scripts.features.extract --run runs/dataset-v2-matched-val   --output runs/image-features-v2/val.npz   --jobs 8
bin/submit ml-cpu python -m scripts.report_image_features --run runs/dataset-v2-matched-val \
  --table runs/image-features-v2/val.npz --train-table runs/image-features-v2/train.npz --output runs/image-features-v2/case_report
# look at the gate first; then:
bin/submit ml-cpu python -m ml.image_feature_experiment --train runs/dataset-v2-matched-train --val runs/dataset-v2-matched-val \
  --features-train runs/image-features-v2/train.npz --features-val runs/image-features-v2/val.npz \
  --output ml/runs/image-features-xgb_mlp --sets base energy radial fft --jobs 8
```

## 7. Radial regions: alternatives for later (not now)

Option B above is the one implemented. These are recorded for later, if the radial check or the experiment shows B is insufficient:
- **A. One set of beam radii for both channels.** The original [0, .5) … [8, 16), [16, edge) for C and R alike, with R's `frac` replaced by the density contrast. Simpler, but gives C regions with zero energy and R inner regions too small to estimate well.
- **C. Equal-area rings per source** (pixel-count quantiles of ρ). These compare cleanly across sources, but the first ring would reach ρ ≈ 14 and erase the near-source detail the model channel needs. Only sensible for R's outer field.
- **D. Field-relative radius** ρ / ρ_edge for R's far field. With fields of 40–64 beams, this makes the outer rings mean the same fraction of the field everywhere. A refinement of B's [8, 16) and [16, edge).
- **E. PSF-derived radii** (first null and first sidelobe distances from each case's PSF profile). The most physical choice for sidelobe-type artifacts, but it adds a profile-fitting step.

## 8. Out of scope (deliberately)

- Angular harmonics, autocorrelation and Gabor (items 4–6).
- A smooth or parametric reference fit.
- Raw Fourier phase.
- The v2.1 or extension data.
- Feeding the features into the dataset npz or `FEATURE_VIEWS`. That would only be worth doing for a block that wins.
