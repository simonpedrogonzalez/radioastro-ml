# Corruption metrics and noise versus uv coverage

Status (Nov 2): STD-multiplier convention selected; truly noiseless rows required.
Nine-cell pilot passed. Full example and 100-training-source imaging sweeps
launched under `scripts/run_noise_uv_experiments.py`; completion is recorded in
`runs/noise-uv-experiments-20261102/progress.json` and `driver_status.json`.
This is an experiment change, not a library refactor or ML training run.

Nov 2 supersedes the initial pending choice below: use the previous numerical
levels as total thermal STD multipliers, adding 0,1,2.5. Thus Y=V+a*n0,
with the same realization paired across levels. No extra independent noise
is added. a=0 contains only the predicted sky; a=1 is regular noise.
Levels are 0,1,2.5,5,10,15,20,30,40,50,100,1000,2000,3000,4000.
SNR_corr_actual=||Y−V||/sigma0 reports the total thermal perturbation;
SNR_difference_from_regular_actual=||Y−Y_regular||/sigma0 preserves the
previous baseline comparison explicitly. Both are freshly measured after each
cut, with an additional pre-cut norm. Never label a multiplier as SNR_corr.

Input source flux is fixed across every noise/UV/corruption variant within a
source. It differs across sources: the precursor sets flux=original robust
dynamic range × predicted natural-image RMS. An audit of all2,893 retained
training variants found identical input flux within every source; across100
sources flux spans0.004321–0.668421Jy. Example2241+098 uses0.0921665Jy.
Amplitude errors affect apparent visibilities/recovered flux, not this input
sky definition; phase errors can redistribute flux and reduce image peaks.

## Initial alternatives (historical; choice resolved above)

The request describes level 1 as regular thermal-noise injection, but also
calls higher levels extra-noise SNR_corr. Those are different parameters:

- A noise multiplier a gives per-component sigma = a sigma0, with a=0
  noiseless and a=1 regular noise. Values 1000–4000 are extremely noisy.
- The existing extra-noise convention gives Y=V+n0+eta and
  SNR_corr=||eta/sigma0||. Regular noise has extra SNR_corr=0.
  For N complex samples its total sigma is sigma0 sqrt(1+SNR_corr^2/(2N)).

The final choice applies to both new noise sweeps. Level-zero is truly
noiseless in both experiments. Gain comparisons retain existing SNR targets.

## Running and validation

```sh
ml/.venv/bin/python scripts/run_noise_uv_experiments.py \
  --output runs/noise-uv-experiments-20261102
```

This uses two disjoint CASA processes: example then training shard0, and
training shard1. Each source/level/cut writes an atomic record, FITS products
and original pipeline QA. Completed cells are skipped on resume; partial
imaging directories are archived before retry. Generated scratch MS copies
are removed after processing, and original data are never overwritten.
`scripts/noise_uv_report.py` refreshes the HTML/CSV every five minutes; counts
identify unfinished work. Global UV medians use identical source membership
across all150cells and are labeled provisional until100sources complete.
The full task comprises150example and15,000training imaging cells.

[Live report](../runs/noise-uv-experiments-20261102/report.html).
Pilot: multipliers0,1,2.5 at cuts0,50,90. Regular uncut dirty/CLEAN/residual
FITS were pixel-identical to original dataset products. Noiseless thermal
norm/sigma exactly0. All pixels finite, grids/annulus and sky flux identical,
sigma_NA equation and paired-noise norms checked. Records retain CLEAN
convergence summaries; the standard empty-model warning from dirty niter0
imaging is expected. CASA also emits an existing external data-update-lock
warning; this audit did not modify the external CASA data installation.
Typical pilot imaging6–11seconds/cell excluding MS reconstruction/I/O;
full execution will take many hours and large sources can be slower.

## Shared protocol

Reuse source 2241+098 for image tables, matching the previous noise sequence
and angular-coverage experiment. Training comparisons use exactly the existing
100 training sources; no validation/test sources enter population summaries.
Source 2241+098 is a validation source used only for the illustrative table.

Reuse the standard dataset direct CASA imaging configuration (Briggs robust
0.5), source reconstruction, observing setup, flags and seeds. Baseline and
noise controls must be paired within each source. Do not alter original data,
indices, reports, features or models. Write new products under runs/.

Experiment 2 uses the previous conjugate-paired angular cuts 0,10,...,90%:
atan2(v,u) modulo 180 degrees, removing [0,180p/100). Report actual sample loss
separately. Generate noise before cuts, preserve surviving DATA and weights,
and fix each source's 0%-reference grid, CLEAN mask and pixel annulus across
cuts and noise levels. 100% is unimageable. Compute annulus scaled MAD
(the requested robust residual RMS), retain CLEAN convergence diagnostics,
and show the noiseless curve separately to distinguish reconstruction residuals
from noise. Lost samples alone increase predicted natural-weight image noise;
Briggs weighting and CLEAN mean sigma_NA is not an exact residual prediction.

Experiment 1: single-source noise table with dirty/CLEAN/residual, noise
sigma, reference and updated sigma_NA, measured SNR_corr and annulus scaled
MAD; pairwise matrices for that source and all training sources. Global amp
and phase matrices use retained gain samples up to SNR_corr100.
Experiment 2: single-source image tables per noise level and line plot across
cuts, plus a 100-training-source median curve per noise group. Medians must
use identical source membership for every cut; report any failures explicitly.
Use resumable per-source/per-level/per-cut outputs and time a pilot before
starting the approximately 15,000-image training sweep.

## Implemented retained-data comparison

Command:

```sh
ml/.venv/bin/python scripts/corruption_metric_relationships.py \
  --output runs/corruption-metric-relationships-20261101/existing-training
```

[Partial report](../runs/corruption-metric-relationships-20261101/existing-training/report.html)
contains noise/amp/phase matrices and a CSV of all 2,893 existing training rows.
Each type includes its positive examples and the same 100 regular-noise clean
baselines. The script cross-checks every residual MAD against the saved feature
cache. Reference sigma_NA is the prior ML feature; total sigma_NA multiplies it
by sigma_total/sigma_reference. These are natural-image predictions, while the
actual images use Briggs weighting. The script does not simulate new cases.

## Actual SNR verification (Nov 2)

`scripts/noise_snr_audit.py` reconstructs source 2241+098 in isolated scratch
and evaluates `SNR_corr_actual = sqrt(sum(abs(Y-Y_reference)**2))/sigma0`
from stored complex DATA. Sigma0 stays fixed at the original reference, not
the increased sigma. The valid selection matches the existing definition.
Noiseless and regular-noise controls are included, and zero deltas are valid.

The audit compares STD multipliers 0,1,2.5,5,100,1000,2000,4000 of one fixed
realization with all existing extra-noise SNR targets plus 1 and 2.5.
It reports both the total thermal norm relative to noiseless sky and the
perturbation norm relative to regular-noise data. With fixed realization n0,
the former is a||n0/sigma0|| and the latter is |a-1|||n0/sigma0||.
Consequently, the noiseless control has total thermal norm zero, but differs
from regular-noise data by removal of n0; that difference must not be called
an injected extra-noise amount. Independent additive extra noise has a
different pairing: sigma_extra=sigma0 sqrt(a²-1) for total STD multiplier a>=1.
For the existing injections, the paired measured norm is checked against its
target at relative tolerance 1e-5 and against the library metric at 1e-12.

```sh
bin/run casa scripts/noise_snr_audit.py \
  --output runs/noise-snr-actual-audit-20261102
```

[Direct MS audit report](../runs/noise-snr-actual-audit-20261102/report.html).
Temporary copies are deleted after verification; original MSs are not modified.
This checks injection arithmetic, not imaging or the full training sweep.

The updated retained-data comparison is at
[Nov 2 report](../runs/corruption-metric-relationships-20261102/existing-training/report.html).
Its CSV now includes `SNR_corr_actual`, provenance and measured-minus-requested.
Gain values use retained measured Delta_V_L2/sigma0; noise values use retained
generation-time measurements. These are checked against saved targets, but
must not be described as fresh MS rereads for all 2,893 samples.
Future uv-cut rows should report the surviving-sample norm as well as the
pre-cut injection norm, without renormalizing the noise after applying cuts.
