# CLEAN recovery metric comparison

[Completed original-flux report](../runs/clean-recovery-original-flux-20261104/report.html): 171/171 images, 180 paired metric rows, no invalid cases.

The current experiment is the original-flux thermal-noise comparison. The earlier [2026-11-02 pilot](../runs/clean-recovery-pilot-20261102/report.html) used the Dataset v1 SNR-preserving simulated flux and is retained as a historical run. It must not be resumed with the new protocol. The stopped training run is also historical.

## Scientific setup

For each source, take the point-source position from the existing thermal simulation metadata and the **flux from the original CASA image's `tclean_summary.final_model_flux_jy`** in `collect/experiments/extracted_thermal_simulation_comparison_20260909T005739/<source>/original/default_imaging/qa.json`. This is the integrated CLEAN model estimate of the original point-source flux. Record its original restored-image peak as a cross-check. For 2241+098, these are 0.1498904526 Jy and 0.1493805349 Jy/beam, respectively; the old SNR-preserving simulation used 0.0921665222 Jy. Predict fresh noiseless visibilities from the *original input Measurement Set*, with the original flux and position. The original data remain read-only.

Keep the theoretical ideal thermal-noise request and nominal visibility weights from the prior protocol. One noiseless image is shared. For each of ten independent seeds, make a regular-noise realization at `a=1`, then scale that same additive realization to `a=2.5,5,10,15,20,30,40,50,100,200,400,700,1000,2000,3000,4000`. Image the noiseless case first and freeze its grid, beam, CLEAN search mask and valid evaluation support. Store settings, flux provenance, seed map, products and hashes in run-owned checkpoints. A changed settings hash requires a new run directory.

Define the thermal coordinate as in the [simulation parametrization note](</Users/u1528314/Documents/Obsidian Vault/Notes/Simulation Parametrization Pilot.md>):

$$\mathrm{SNR}_0=F/\sigma_{\mathrm{img},0},\qquad r=a/\mathrm{SNR}_0.$$

Here $F$ is the original imaging model flux estimate and $\sigma_{\mathrm{img},0}$ is the stored *predicted nominal image RMS*. `r` and `a` are both saved per metric row. The noiseless image has `a=r=0`; the regular thermal control has `a=1` and `r=1/SNR0`.

## Metrics and report

Let $I$ be restored image, $R$ residual, $C=I-R$ smoothed CLEAN model, $M$ raw CLEAN model, and $T$ the analytic point source restored by the fixed peak-one beam. Evaluate on the fixed valid pixel support $\Omega$. Use $D_1(X,Y)=\sum_\Omega|X-Y|/\sum_\Omega|T|$ and $D_2(X,Y)=\sqrt{\sum_\Omega(X-Y)^2/\sum_\Omega T^2}$. For same-seed regular images $I_1,C_1$, save $e_{TI,p}=D_p(I,T)$, $e_{TM,p}=D_p(C,T)$, $e_{II,p}=D_p(I,I_1)$ and $e_{MM,p}=D_p(C,C_1)$ for $p=1,2$. The CSV also retains raw-model flux, position and residual-RMS diagnostics. The noiseless paired metrics are repeated for each seed because the regular comparator differs.

The report has one contiguous seven-row, two-column figure. The first four rows show TI, TM, II and MM with $L_2$ at left and $L_1$ at right. The fifth row shows $e_{\mathrm{Fin}}$ and $e_{\mathrm{Fout}}$. Position error occupies the sixth row at left, and robust residual RMS occupies a dedicated seventh row at left; unused right-side panels are hidden. Axes share the thermal coordinate and have a light grid. Each visible panel has its own labeled y scale; right-column y ticks appear on the right. Interior x tick labels are hidden. Bottom ticks label `r` with the corresponding `a` in parentheses; zero appears on the same symmetric-log x axis. Thin lines show seeds, and the dark mean with a one-standard-deviation band summarizes each injection level. Position values flagged as unreliable for empty raw models remain in the CSV but are omitted from the plot and its summary. No knee detection, extra zero-point subplot or image gallery is generated. Full image/model products remain linked through checkpoint records. `metrics.csv`, `progress.json` and `report_status.json` document data and missing cases.

## Run

Use a fresh output directory; the coordinator writes `recovery_flux_mode.json` before launching CASA, which prevents accidentally using the old SNR-preserving flux path. It refreshes the report during generation:

```sh
ml/.venv/bin/python -m scripts.run_clean_recovery \
  --output runs/clean-recovery-original-flux-20261104
```

A smaller validation run can be made by writing `{"mode":"original_imaging_model_flux"}` to `recovery_flux_mode.json` in a fresh run directory, then invoking `bin/run casa scripts/noise_uv_experiments.py --recovery --output <run> --source 2241+098 --levels 0 1 10 --seeds 0`, followed by `ml/.venv/bin/python -m scripts.clean_recovery_noise_sweep --output <run>`. Never modify an existing run's flux mode. Run the analytical checks with `ml/.venv/bin/python -m unittest tests.test_clean_recovery_noise_sweep -q`.
