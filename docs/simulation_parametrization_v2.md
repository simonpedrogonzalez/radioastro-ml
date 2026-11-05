# Simulation parametrization v2

This is a new isolated run and one combined report. The [2026-11-02 bounds pilot](../runs/injection-bounds-pilot-20261102/report.html) and [old slider report](../runs/injection-range-visual-20261104/report.html) remain historical and are not changed. The vault's [Simulation Parametrization Pilot](</Users/u1528314/Documents/Obsidian Vault/Notes/Simulation Parametrization Pilot.md>) supplies the table layout and the definitions of recovery error and thermal $r$.

## Source and injection protocol

Rank the exact 100 training sources by $\mathrm{SNR}_0=F/\sigma_{\mathrm{img},0}$, breaking ties by source ID and choosing ranks 0, $n//2$, and $n-1$. Here $F$ is `tclean_summary.final_model_flux_jy` from each **original CASA image** in the extracted thermal experiment; its `clean_peak_jy_per_beam` is retained as a cross-check. $\sigma_{\mathrm{img},0}$ is the stored predicted nominal image RMS, which depends on the observation and ideal thermal-noise request. The current selected low/middle/high templates are `1119-030`, `0735+331`, and `0744-064`, with nominal SNR about 560, 17,340, and 136,588. Recompute the selection when the run is first planned; the immutable plan records all ranked sources and QA hashes.

Predict fresh noiseless visibilities at the original model flux from each original input Measurement Set. Choose exactly two antennas with the smallest and largest **positive** $\sqrt{\sum_{l\in A_j}|V_l^0/\sigma_{0,l}|^2}$, breaking ties by antenna ID. A single gain affects one selected antenna's noiseless visibilities, followed by the same ideal additive thermal-noise realization for that source/seed. Thermal cases scale that same ordinary realization. Keep the previous CASA noise request, three seeds, nominal weights, flags, UV coverage, grid, final restoring beam, search mask and valid support.

Sweep phase on both signed branches and amplitude on both sides of one, retaining the existing pilot's initial physical grids, two refinement rounds and caps. Thermal noise now uses a common grid in $r=a/\mathrm{SNR}_0$: `0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2`. Decode each level as $a=r\mathrm{SNR}_0$, after the ordinary $a=1$ control. This reaches comparable recovery regimes across sources with very different original fluxes. The noiseless case is $a=r=0$. All cases are checkpointed by physical parameters; interrupted runs resume only matching immutable settings.

## Measurements and report

Convolve the raw CLEAN model by the fixed peak-one restoring Gaussian. Define $y=e_{TM,2}=\lVert C-T\rVert_2/\lVert T\rVert_2$ on fixed valid support and $d=y-y_0$ relative to the same-seed ordinary control. The report derives physical parameter ranges from sampled 0.05 and 0.9 recovery-error crossings. It shows **full**, **summary**, and **conclusion ranges** tables in the vault note's style, without context-decision or internal $q$ tables. Cases lacking complete valid seeds cannot establish a bound. A high tested endpoint with no sustained 0.9 crossing is labeled as coverage, not a confirmed severe transition.

For each of amplitude, signed phase and thermal noise, a two-column plot shows $y$ and signed $d$ for all sources. Blue/orange/green mean low/middle/high SNR, with no source IDs in the plot legend. Large-contribution antennas use solid lines; small-contribution antennas use dashed lines. Transparent spans connect the two antenna means for each source; thermal seed curves are solid with a min–max band. Horizontal dashed guides mark 0.05 and 0.9. On each signed phase or amplitude branch, the blue vertical line is the **earliest first outward 0.05 crossing of any individual curve**; the red line is the **furthest first outward 0.9 crossing**, using the tested endpoint for each curve that never reaches 0.9. Phase and amplitude have up to four lines, thermal noise two. Crossings are interpolated linearly in signed phase and logarithmically in amplitude or thermal $r$, matching each plot's x coordinate. The original conclusion ranges table remains unchanged; an adjusted conclusion ranges table gives these interpolated plot bounds and identifies endpoint use. Values are written beside the lines; thermal lines show both $r$ and low/middle/high-source $a=r\mathrm{SNR}_0$. The same $y$-based markers appear on the $d$ panel for comparison; they are not thresholds on $d$.

Below each corruption's combined plot, show low, mid and high SNR templates with that role before each source ID. Each has one slider over the **collected levels available for its selected antenna and seed**, the same six-image horizontal strip and per-source line plot as the old visual report. The six channels are analytic truth, convolved raw CLEAN model, signed difference, dirty, restored, and residual. Every frame uses a fixed source/channel color limit across all cases. The truth/recovered and dirty/restored limits are anchored to the known source peak; error uses 20% of that peak, while residual uses the pooled robust scale. This avoids a nearly zero percentile colorbar for a point source occupying far below 0.5% of the image pixels. The HTML modal retains horizontal and vertical scrolling, zoom and pan. Each slider line plot marks the same interpolated branch bounds as the combined plot; they can lie between slider stops, whose images let the user inspect the crossing bracket. Thermal slider plots label both their source-specific $a$ and $r$. The six image frames preload before the row swaps and reserve their display height during loading. The noiseless thermal image is a slider stop but lies outside the positive log line plot. No bottom image gallery is generated.

## Execution

Use a fresh directory; `--v2` is mandatory and rejects a mismatched existing plan:

```sh
bin/run casa scripts/injection_bounds_pilot.py --v2 --smoke \
  --output runs/simulation-parametrization-v2-smoke-20261104
bin/run casa scripts/injection_bounds_pilot.py --v2 \
  --output runs/simulation-parametrization-v2-20261104
```

The CASA runner refreshes the tables and combined plots during generation. When all cases finish, the same report module renders the complete sliders and writes `report.html`, `metrics.csv`, `bounds.json`, `display.json`, `render_data.json`, and the case manifest. A report-only final rerender is `ml/.venv/bin/python -m scripts.simulation_parametrization_v2_report --output <run> --final`. The old runs are never reused because their source flux and template selection differ.

## Dataset v2 handoff

The pilot completed 861/861 images. Dataset generation adopts the vault's **V2 adjusted** column: phase `[-180,-23.79]` / `[23.743,180]` degrees, attenuation `[0.01,0.66391]`, amplification `[1.4309,20.276]`, thermal `r=[0.0097046,0.51759]`. Sample phase and attenuation linearly, amplification and r logarithmically. Log interpolation of attenuation crossings does not change its linear sampling policy. Absence is `phi=0, A=1, a=1` (`r=1/SNR0`), not noiseless `a=0`.

See the [Dataset v2 design](dataset_v2_three_task_design.md) for the minimal generator patch and validation. Use original model flux for every dataset source with unchanged nominal noise. Existing old-flux smoke runs remain historical; pilot template/extrema selection does not replace dataset source/antenna sampling.
