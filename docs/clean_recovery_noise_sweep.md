# CLEAN recovery metrics: implementation plan

Implement the experiment defined in [the vault note](</Users/u1528314/Documents/Obsidian Vault/Notes/CLEAN Recovery Metrics and Thermal Noise Sweep.md>). This document is a plan only. Use a fresh run directory and preserve existing experiments and ML targets.

## Reuse and required changes

| Existing code | Use for this experiment |
| --- | --- |
| `scripts/noise_uv_experiments.py` | Extend its existing `source_run` with an optional recovery mode and explicit seeds. Reuse source lookup, sky construction, paired noise scaling, checkpoint records and scratch-MS cleanup. Default behavior stays unchanged. |
| `scripts/create_dataset_v1.py` | Reuse `_noise_request`, `_simulation_payload`, `_prepare_V_ms`, `_rescale_visibility_delta` and `_cleanup_V_work`. |
| `scripts/uv_angular_coverage_experiment.py::FixedGrid` | Freeze the grid and CLEAN-mask reference beam across all seeds and levels. |
| `scripts/imaging/imaging.py::image_ms` | Use `keep_intermediate_products=True` until raw model and valid support are exported. Current default cleanup deletes these products. |
| `scripts/imaging/config.py::ImagingConfig` | Add an optional restoring-beam setting, default unset, forwarded to CASA. `FixedGrid` alone does not fix the actual restoring beam. |
| `scripts/imaging/fits.py::export_casa_fits` | Reuse compressed atomic export for the model. First extend the single-plane loader in `scripts/preprocessing/fits.py` to accept `Jy/pixel`, which it currently rejects. Keep image-triplet checks requiring `Jy/beam`. |
| `scripts/noise_uv_report.py` | Reuse HTML/table and FITS plotting patterns in a small dedicated recovery report module. Its current plots are hard-coded to uv cuts, different metrics and independently scaled panels. |

Add one recovery metrics/report module using NumPy, Astropy, Matplotlib and **[kneed](https://github.com/arvkevi/kneed)**. Add `kneed` to the existing ML environment's dependencies in `ml/pyproject.toml` and record the installed version. Use the package implementation, not a custom knee detector.

## Generate only the required runs

1. Start with `2241+098`, original flags and `cuts=[0]`. Use its stored point-source flux and phase-centre coordinates. Run levels `[0,1,2.5,5,10,15,20,30,40,50,100,200,400,700,1000,2000,3000,4000]` and noise seeds `0..9`. One noiseless run is shared by all seeds: 171 unique images initially.
2. Reuse the generator's `V + a*n_seed` scaling. The current script uses one source-derived seed and no seed directory, so add explicit seed handling and seed-aware checkpoints. Build one `regular.ms` per seed, and reuse its noise realization across levels. In recovery mode initialize every MS with the same nominal `sigma0` weights, including zero noise. The current uv sweep instead scales absolute weights with `a`.
3. Image the noiseless source first. Save its resolved grid, central CLEAN mask, final restoring beam, metric annulus and valid image support. Reuse them for every remaining run. Preserve `DefaultImagingConfig`: MTMFS with `nterms=1`, Briggs 0.5, gain 0.1, six-beam mask diameter, and its existing `nsigma`/dirty-peak stopping policy. Record each resolved threshold and convergence summary.
4. Save `dirty`, `clean`, `residual`, `psf`, **raw model** (`.model.tt0`, Jy/pixel) and reference support masks before deleting temporary CASA products. Obtain validity from CASA masks and normalization support, including the configured PB cutoff. Do not infer support from nonzero pixels or the CLEAN search mask. Current zero-filled FITS cannot distinguish invalid support from genuine zeros.
5. Write source/seed/level records atomically, including product paths, exact flux, sigma0, seed, actual imaging configuration and geometry. A completed record must include the required model/support products and matching settings. Existing uv-sweep records alone do not satisfy this experiment. Resume only matching complete runs.

Use the existing CASA launcher (`bin/run casa`) for generation and `ml/.venv/bin/python` for report generation. The current `run_noise_uv_experiments.py` automatically queues training sources and the old report, so do not use that queue unchanged for the single-source pilot. Extend its configuration only when launching the later training-source batch.

## Compute the specified metrics

Load matching planes with Astropy. Use the full fixed valid image region `Omega`. Use celestial WCS and beam position angle to obtain major/minor-axis offsets `u,v` in arcsec. With beam FWHM `bmaj,bmin`, compute `rho²=(u/bmaj)²+(v/bmin)²`, source region `S = Omega & (rho<=1)`, and true restored image `T=F*exp(-4*ln(2)*rho²)`. The source centre is the stored true position, not the brightest pixel.

The direct imaging code uses ordinary restoration and exports image/residual without residual rescaling, so obtain the smoothed model as `C=I-R`. Keep the raw model `M` for flux sums. All norms below use `Omega` and denominators from `T`:

| CSV keys / note symbols | Calculation |
| --- | --- |
| `e_TI_1`, `e_TI_2` / $e_{TI,p}$ | `D_p(I,T)` |
| `e_TM_1`, `e_TM_2` / $e_{TM,p}$ | `D_p(C,T)` |
| `e_II_1`, `e_II_2` / $e_{II,p}$ | `D_p(I,I_regular_same_seed)` |
| `e_MM_1`, `e_MM_2` / $e_{MM,p}$ | `D_p(C,C_regular_same_seed)` |
| `e_Fin` / $e_{\text{Fin}}$ | `abs(sum(M[S])-F)/F`, also save signed flux error |
| `e_Fout` / $e_{\text{Fout}}$ | `sum(abs(M[Omega & ~S]))/F` |
| `e_pos` / $e_{\rm pos}$ | `rho` at the brightest `C` pixel in `Omega`, using the note's tie and missing-detection rules |

Here `D_1(X,Y)=sum(abs(X-Y))/sum(abs(T))` and `D_2(X,Y)=sqrt(sum((X-Y)**2)/sum(T**2))`. Use float64 arithmetic. Compute the shared noiseless reconstruction once, but pair it with each seed's regular image when calculating `e_II`/`e_MM`. Reuse `beam_region_mask` and `summarize_residual_pixels` for the existing annulus robust RMS. Save all metrics, masks and missing/failed-run status. Newly nonfinite pixels inside `Omega` invalidate the run.

## Report and knee markers

Produce `report.html`, `metrics.csv` and `knees.json`. For each metric plot individual seeds, the median and the 10th–90th percentile band, with **log noise multiplier on x and untransformed error on linear y**. Show zero noise separately. Report seed counts at each level and missing position detections. For image comparisons use shared scales within each product, with separate Jy/pixel scales for raw models.

Run `kneed.KneeLocator(log10(a), metric, curve="convex", direction="increasing", S=1.0, interp_method="interp1d", online=False)` on the median and each seed, using finite metric values at `a>1`. Keep zero error values. Require three usable levels and variation in both axes. Convert `.knee` back with `10**locator.knee`. Record `None` as no knee, and flag curves whose shape is unsuitable for the increasing-convex assumption.

**Mark every detected median knee** with a prominent point, dashed vertical line and numeric noise-multiplier label on its metric plot. Mark seed knees as smaller points on their own curves. If no knee is found, say so on the plot. Include linked image/model/residual panels at regular noise, detected knees and the largest noise. Cite [Kneedle (Satopää et al., 2011)](https://raghavan.usc.edu/papers/kneedle-simplex11.pdf) and [kneed](https://github.com/arvkevi/kneed) in the report.

The axes and range define the knee being sought. On log-x/linear-y axes even a smooth power law can be curved. Report candidate knees, not proven CLEAN failures. After the pilot, reuse the same implementation per training source without changing source splits or fitting ML models as part of this experiment.
