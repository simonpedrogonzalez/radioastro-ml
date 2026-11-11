# VLA pipeline QA experiment

Source: vault `Notes/Week 10 RadioastroML Update 2026-11-04.md`, Jeff Feedback, Idea 2.
Status: DRAFT specifications, updated 2026-11-09. Recipe comparisons and evaluation conventions below supersede conflicting wording in the earlier draft. Unresolved choices are listed at the end. No implementation or run is authorized by this document alone.

## Scientific question and scope

Assess how well specified gain-calibration and automated target-self-calibration recipes repair the residual corruptions in matched Dataset v2: constant antenna amplitude errors, constant antenna phase errors, additional Gaussian thermal noise, and their mixtures.

RFI is out of scope. Do not add RFI injection or automatic RFI detection to this experiment. Additional Gaussian noise is not an RFI simulation.

The available cases are single-field simulated observations based on already-calibrated observational templates, with corruption injected afterwards. They lack the separate flux, bandpass and phase-calibrator scans needed for the full standard VLA calibration recipe. Conclusions must name the tested recipe and conditions. A surviving error is not evidence that every VLA processing route cannot correct it.

## A. Known-model gain-recovery benchmark

Run `gaincal` and `applycal` against the exact injected point-source sky model, then evaluate with matched-v2 imaging. This tests whether a predefined gain-solving procedure can recover the source with the correct sky supplied. It is a favorable diagnostic benchmark, not a full pipeline or calibrator-to-target transfer experiment.

Use fixed phase-only and phase-then-amplitude variants as appropriate to the recipe table. Freeze solution intervals, `minsnr`, `minblperant`, gain type, amplitude normalization, interpolation and reference-antenna policy after the pilot. Do not tune each case using its true injection or measured recovery score. Select the reference antenna using a fixed rule based on available metadata/data quality, not the injected bad-antenna identity. A corrupted reference antenna is not automatically unusable because phase solutions are relative.

Record whether amplitude normalization preserves an external flux scale or permits adjustment to the known model. Supplying exact flux in A gives it information unavailable to ordinary target self-calibration. A versus B differences include this model advantage and the different recipe settings, not solely pipeline heuristics.

## B. Automated target self-calibration

Use the NRAO imaging pipeline's self-calibration machinery through `scripts/vla_pipe.py`, with the simulated field registered as a target and a pinned CASA/pipeline version. This asks whether the automated procedure repairs the source using its own image-derived model and acceptance rules.

The pipeline must create its own internal images, masks and model as required for self-calibration. Do not give B the simulation truth or force the external evaluation mask into its internal model-building. Truth is reserved for evaluation, while A explicitly uses it as a benchmark.

### Recipe comparisons

[USER] 2026-11-09 SUPERSEDES the table below: only `phase_amp` is run, in A and B, each with `flagging_off`
and `flagging_on`. Phase-only is not worth accounting for (amplitude correction is the sensible standard
choice). B `default` is dropped as well, since in pipeline 2025.1.0.36 it equals phase-only
(`amplitude_selfcal=False`). The table is kept for the record.

[USER] Include multiple named recipes, with phase-only/default comparisons and flagging-enabled/disabled variants. Amplitude correction must be tested before interpreting amplitude survivors as a correction limitation.

| Gain recipe | Definition | Allowed interpretation |
| :-- | :-- | :-- |
| `default` | Unmodified self-calibration settings for the pinned installed pipeline version. Record the actual resolved settings. | Performance of that default recipe. |
| `phase_only` | Explicitly disable amplitude self-calibration. | Phase-only repair, not amplitude-correction capability. |
| `phase_amp` | Enable the supported amplitude-self-calibration attempt after phase calibration. Keep the recipe's acceptance checks. | Automated phase-plus-amplitude policy. Enabling an attempt does not prove amplitude solutions were accepted or applied. |

For each distinct gain recipe, compare `flagging_off` and `flagging_on`, starting from identical independent input copies. If the installed default equals phase-only, run it once and display both labels rather than duplicating it. Settings other than the intended flagging intervention should be the same within each pair, with all resolved differences recorded.

- `flagging_off`: preserve the input visibility flags throughout the evaluated recipe. Do not reset pre-existing flags. Calibration tables may mark solutions invalid, but these must not become new visibility flags. Record samples left uncorrected because no valid solution exists. Verify the final MS flags, including split/working copies, rather than assuming one `applycal` parameter controls every stage.
- `flagging_on`: allow the specified calibration-solution/application flags and the recipe's documented automatic internal flags. Record their origin and extent. Do not supply a hand-written flag template informed by injected truth.
- If the pipeline cannot run its full internal procedure without changing flags, label a controlled no-flag variant as a modified recipe. Do not claim it is an untouched default. Resolve this in the smoke test.

`hifv_flagtargetsdata` applies a target flag template in the documented imaging recipe. Its presence is not an automatic RFI test. Here, the flagging comparison concerns the selected calibration/self-calibration procedures and their treatment of invalid solutions.

A uses the same explicit flagging distinction where applicable. Decide its exact solver settings before main runs. B's default settings must not be inferred from wrapper overrides, especially the current explicit `amplitude_selfcal=False`.

## Regeneration and input provenance

Dataset v2 deletes `observed.ms` after imaging. Regenerate cases from the immutable plan using the same generation path as `create_dataset_v2_matched.execute_unit`: noiseless signal, nominal noise seed, gain injection and paired noise.

Verify stored `v_sha256` and `nominal_noise_sha256`, the case parameters, gain application, noise scale, and code/configuration provenance. Those two hashes alone do not prove the final corrupted MS is correct. On the smoke panel, also reproduce selected committed before images/metrics to check the complete reconstruction and imaging path. Reuse committed before measurements only after that agreement is established.

Make the visibility-column contract explicit. The current wrapper prefers `CORRECTED_DATA` when it exists, whereas matched-v2 evaluation uses `DATA`. Verify which column contains the regenerated corruption, deliberately copy/select it for recipe input, and verify the final applied output column. Do not infer correctness from a column's existence or a task returning successfully. Avoid carrying an exact simulation model into B's model-building path through `MODEL_DATA` or virtual-model metadata.

Keep source runs and observational data read-only. Each experiment/recipe writes to its own new output directory. Record CASA/pipeline versions, code revision, parameters, input/output flags and weights, applied calibration tables and selected columns.

## Internal imaging versus evaluation imaging

[USER] The pipeline may use its own imaging internally to build and evaluate self-calibration models. All reported matched-v2 eC/eD and frozen-model features come from the separate, common evaluation imaging.

For external evaluation, keep the per-source sky, grid, pixel scale, field of view, reference restoring beam, CLEAN search mask, metric regions, weighting algorithm and CLEAN settings fixed across recipe comparisons. Use each recipe's actual output flags and weights as its data inputs. Do not reset them solely to pass old invariants. A common weighting algorithm does not imply identical effective imaging weights after flagging/reweighting.

Keep pipeline-native images and native QA as secondary diagnostics. Do not mix their scores/features with matched-v2 metrics or frozen-model predictions.

Changing flags changes the dirty PSF but does not automatically require changing the chosen restoring beam. Keep the reference beam for the initial comparison. The policy for cases with substantial data loss or materially degraded resolution is deliberately unresolved pending discussion. Do not silently broaden the beam, replace truth, exclude cases or call them successful.

## Flags, weights and dirty references

[USER] Compute dirty error even in flagging-enabled recipes. Compute the after dirty image and its noiseless reference using the final flagging state of that particular recipe.

Let I_r be the dirty-imaging operator with recipe r's output sampling, flags and effective weights. Let V0 be the original noiseless sky visibilities, and Vout,r the recipe's corrected output. Then:

    D_after,r = I_r(Vout,r)
    D_truth,r = I_r(V0)
    eD_after,r = ||D_after,r - D_truth,r||_Omega / ||D_truth,r||_Omega

Generate the reference with the same selected rows/channels, imaging settings, flags and weights as the after image. Do not apply the estimated gains to V0: that would move the truth toward the fitted answer. Match row identities explicitly if a pipeline split/reorders data. If the pipeline averages samples, apply the same sampling/averaging operation to the truth or declare that output unsupported until the mapping is implemented.

If weights change, rebuild the dirty reference even without new flags. Log whether the change came from calibration weight scaling, statistical reweighting or another stage. Weight policy must be specified per recipe, not inferred from defaults. Keep it fixed within a flagging comparison wherever the pipeline permits this.

Replace the old all-columns-identical invariant with explicit allowed changes. Sky truth and input noise seed remain fixed. Flag and weight changes are allowed only as declared by the recipe. Geometry and sample mapping must remain traceable. The dirty reference is an observation-dependent representation of the same sky, so rebuilding it does not change that sky truth.

Retain the original committed before comparison as the main end-to-end result. An additional diagnostic may image the uncorrected case with final flags/weights and compare it to the corrected output on those same retained data. Label that as conditional on the output selection. It must not replace the original before result or hide discarded data.

## Controls, noise semantics and measurements

Process a nominal-noise, uncorrupted control independently through every recipe for each source/noise seed. This measures recipe-induced bias, noise fitting and flagging in otherwise clean data. Compare the corrupted output with its own before measurement and the processed control. Do not apply gains fitted on the corrupted case to the main control.

The current simulation applies gains to the signal, then adds noise. For baseline gain G, it produces Vobs = G V0 + n. Even exact inversion gives V0 + n/G, not V0 + n. Amplitude attenuation can therefore leave amplified noise after correct gain removal. Keep this simulation convention explicit. A known-inverse diagnostic on a few gain cases can check the implementation and illustrate the changed noise, but is not a guaranteed minimum-error estimator. No operation may subtract the known noise realization as part of a tested recipe.

Continue computing eC, eD and before/after features. eC uses the convolved recovered model against the fixed true model at the common restoring beam. eD includes noise because its reference is noiseless. Record source-flux error, residual RMS and newly flagged fractions alongside these scores, to expose cases with a good model estimate but poor sensitivity or extensive data loss.

The earlier joint rule `eC <= 0.05 and eD <= 0.05` remains provisional, not an established scientific success criterion. Recommendation pending approval: use continuous eC as the primary reconstruction outcome, eD as a diagnostic, and choose any absolute acceptance threshold after checking processed controls on pilot sources. Freeze it before the main evaluation. Do not publish success rates under an undeclared or post-hoc threshold.

Frozen matched-v2 model predictions remain secondary exploratory outputs. Record the exact saved model and feature view. Calibration, flags and weights can shift the feature distribution. Existing per-corruption damage labels do not automatically remain valid after processing. Do not score after-processing damage predictions against old labels: obtaining those labels requires processing the corresponding removal/control cases under a clearly specified comparison.

## Outcomes and reporting

### Cases with substantial data loss

[USER] Keep the strong-corruption samples in the experiment and run every selected recipe on them. Do not exclude a case because substantial flagging, failed solutions or unusable output is expected. Observing how the recipe handles such cases is an intended result.

Record the actual disposition and the pipeline's stated reason: correction applied, correction rejected, no valid calibration solutions, or no usable output after flagging. Report newly flagged fractions and remaining usable data. Use “this recipe did not recover a usable result” rather than “the data are irrecoverable,” unless quoting an explicit pipeline message (which is still that recipe's assessment, not proof that no other method could recover the data).

If an image can be formed, compute the agreed metrics using its actual flags and weights. If no valid image/reference can be formed, record the metrics as unavailable with a reason, never as zero and never as a silently omitted case. Keep these cases in the overall disposition counts. Present metric summaries for measurable cases with the measurable/attempted counts alongside them. Runtime failures and unsupported input remain separate from a valid scientific rejection.

No blanket flagging-fraction cutoff is selected here. How to evaluate resolution when a heavily flagged case still produces an image remains open. Retaining the sample does not require declaring its image acceptable.

Separate execution status, correction applied, and scientific outcome:

- Correction accepted/applied, including phase and amplitude separately.
- Valid decision to skip or reject a correction, with reason.
- Completed but reconstruction remains poor under the chosen evaluation criterion.
- Unsupported input or metadata/adapter incompatibility.
- Runtime/software failure.

Unsupported inputs and crashes are not scientific proof of uncorrectable corruption. Conversely, a valid no-op decision on a damaged image is relevant to the automatic recipe's usefulness. Keep both visible, with explicit denominators in summary rates.

The wrapper's current `did_selfcal=True` only means the call returned. Verify accepted tables and actual application, including whether the selected returned MS is the one the task corrected. A recipe enabling amplitude correction may still reject it.

Report a compact table per recipe: attempted cases, valid completions, corrections applied, skips, incompatibilities, failures, eC before/after, eD before/after, flux error, residual RMS and additional flagging. Show outcomes by corruption type, strength and template S/N, rather than interpreting all extra-noise cases as calibration failures. Retain provenance for paired comparisons.

## Run sequence

1. Keep the existing 0449+113 smoke source and its 85 cases for technical validation of regeneration, columns, recipe execution and metric references. Use a few representative B cases first. One source is not an effectiveness assessment.
2. Before scaling, select a common multi-template panel for all recipe variants covering S/N, observing geometry, pure amplitude/phase/noise cases and a few mixtures. Choose it independently of A outcomes. Proposed size: 6–10 templates and three paired noise seeds, subject to runtime approval.
3. A-survivor-only B runs may be additional diagnostics but must not supply B's general success rate. Cases sharing a template, noise seed or matched recipe are correlated. Summarize across templates and use template-level resampling if uncertainty intervals are reported.
4. Full train/validation CHPC evaluation remains a later step after the scientific choices and smoke/panel checks are settled. Nothing is launched yet.

## Open decisions

- Substantial-loss cases remain included and their disposition is reported [USER]. Still open: image acceptability and common-resolution policy when heavily flagged data produce a measurable image.
- Exact A solver settings and pinned B pipeline version, plus identification of default versus phase-only equivalence.
- How to enforce flagging-off throughout B's internal steps. Record any required recipe modification.
- Exact weight policy per recipe. Calibration scaling and statistical reweighting are distinct operations.
- Final reconstruction acceptance rule and treatment of eD in it. The old 0.05 conjunction is provisional.
- Final multi-template panel, noise seeds, saved ML model and output layout.

## Documentation checked

- NRAO VLA imaging pipeline and self-calibration options: https://science.nrao.edu/facilities/vla/data-processing/pipeline/vipl_666
- CASA gaincal, reference antennas and amplitude normalization: https://casadocs.readthedocs.io/en/stable/api/tt/casatasks.calibration.gaincal.html
- CASA applycal, output columns, weight calibration and application flag modes: https://casadocs.readthedocs.io/en/stable/api/tt/casatasks.calibration.applycal.html
- Local implementation: scripts/vla_pipe.py, scripts/create_dataset_v2_matched.py, scripts/create_dataset_v2.py, scripts/recovery_products.py and scripts/dataset_v2_matched_extract.py.

## Implementation status (2026-11-09)

Code: `scripts/vla_pipeline_qa.py` (stages `--plan`, `--execute`, `--report`), plus an opt-in
`require_same_samples=False` mode in `scripts/recovery_products.py` (flags/weights may differ from V and are
recorded instead of rejected). Defaults above are unchanged for every existing caller.

- Regeneration verifies case parameters, `v_sha256`, `nominal_noise_sha256`, the gain and the paired-noise hash.
  (Superseded 2026-11-10: opt-in with `--verify-regeneration`, reproduction default `none`.)
  Optional reproduction re-images the regenerated case and compares eC/eD with the committed values.
- A (provisional, not frozen): `gaincal` p (solint int) then ap (solint inf, per scan), minsnr 3, minblperant 4,
  solnorm False, MODEL_DATA = noiseless V0, refant = most unflagged visibilities (ties lowest id),
  applycal calwt False, `calonly` (flagging_off) or `calflag` (flagging_on).
- B: pipeline 2025.1.0.36 / CASA 6.6.6.18 locally. hifv_importdata -> [hifv_flagtargetsdata if flagging_on]
  -> hif_checkproductsize -> hif_makeimlist -> hif_selfcal(apply=True). (No hif_makeimages, see decision below.)
  flagging_off = `apply_cal_mode_default="calonly"` and no flag template, labelled a modified recipe.
  Default `amplitude_selfcal` is False in this version, so `default` is run once as `phase_only`.
  The corrected MS is found by checking every MS for CORRECTED_DATA != DATA (superseded 2026-11-10: pipeline
  application records and flag summaries, see Implementation notes). Pipeline log decision lines are kept.
- Evaluation: matched-v2 imaging on the recipe output (its flags/weights). The dirty reference I_r(V0) is
  rebuilt only if FLAG/FLAG_ROW/WEIGHT/WEIGHT_SPECTRUM changed. `split` rewrites SIGMA*, which imaging does
  not use, recorded as `changed_columns`.
- Dispositions: applied, not_applied, no_valid_solutions, no_usable_output, unsupported, runtime_failure.
- Frozen-model predictions: `--model <ml.multitask output> --model-names xgb_xgb`. NOT tested locally (no
  saved all22 model on this machine).

Local technical checks (smoke source 0449+113):
- A, 3 cases x 4 recipes (`runs/vla-pipeline-qa-A-test-20261109`): reproduction exact (eC/eD equal to all
  digits). Amp case fixed only by phase_amp, phase case by both, control unchanged.
- B, 1 phase case x phase_only on/off (`runs/vla-pipeline-qa-B-test-20261109`): hif_selfcal completed,
  `sc_success=False` because auto-multithresh found no mask regions (pipeline image 7200x7200 at 0.095",
  rms 5.4 mJy/beam, peak S/N 129). Recorded as not_applied. About 18 min per B recipe on the Mac.

[USER] 2026-11-09: B keeps the pipeline's own image sizes (no `maximsize` reduction). Its decisions (PSF
sidelobe level, rms, per-solint S/N, mask) are computed from those images, so shrinking them could change the
outcome being measured. Compute cost is accepted (CHPC).

[USER] 2026-11-09: B never runs the regcal `hif_makeimages` before `hif_selfcal` (always, not an option).
Those images are the pipeline's deliverable and are used neither by `hif_selfcal` nor by this experiment.
Every B result is therefore labelled a modified recipe ("no regcal hif_makeimages").

## Required fixes: use pipeline bookkeeping for application and flagging

[USER] 2026-11-10. These are implementation requirements for experiment B, verified against the installed pipeline 2025.1.0.36 source. They supersede the output-discovery rule in the implementation snapshot above. Implemented 2026-11-10 (see Implementation notes below). Do not infer calibration application or flag changes by comparing MS columns.

### 1. Identify applied calibration and its output from the task result

Change `recipe_b()` in `scripts/vla_pipeline_qa.py` (the `changed = {}` loop after `hif_selfcal`, currently around lines 628–657). Remove the scan of working-directory MS files, the first-20,000-row `CORRECTED_DATA != DATA` test and the alphabetical selection of the first changed file.

Read the actual `SelfcalResults` returned by `hif_selfcal` (unwrap a task result container if needed). For this continuum/continuum-line experiment, inspect `result.applycal_result_contline`. For each contained applycal result `r`, iterate `r.applied`. Each calibration application `calapp` has:

- `calapp.calto.vis`: the MS to which calibration was applied.
- `calapp.calto.field`, `calapp.calto.spw` and `calapp.calto.intent`: the application selection.

Match these records to the registered input target and its spectral windows. Multiple calibration tables can refer to the same MS, so deduplicate by MS and selection rather than expecting one application record. Resolve field selections through the context, since they can be names, IDs or empty selections. Do not select an arbitrary intermediate MS.

Cross-check the context metadata with `context.observing_run.get_ms(vis)`. The pipeline registers the relevant selection as `DataType.SELFCAL_CONTLINE_SCIENCE` mapped to `CORRECTED_DATA` (or `SELFCAL_CONT_SCIENCE` for a continuum-only input). Query `ms.get_data_column(dtype, source=field_name, spw=spw_selection)` for that selection. Evaluate the identified MS and registered column. This is a metadata lookup, not a column-content comparison.

Exact upstream locations (relative to the installed `pipeline/` package):

- `hif/tasks/selfcal/selfcal.py`: `SelfcalResults.__init__`, `merge_with_context` and `_register_datatype` (lines 33–118). `_register_datatype` itself follows `r.applied -> calapp.calto` and registers the corrected column.
- The same file, `Selfcal._apply_scal` (around line 874): collects final gain tables only for successful targets and applies them to the registered science MSes.
- `domain/measurementset.py`: `get_data_column` (around line 1486), with source/SPW selection support.

Keep target `sc_success` and rejection reasons as diagnostics, but do not use solver success alone as proof that final solutions were applied. Empty application records mean no recorded application only when the task completed normally. Exceptions or inconsistent records are unavailable/implementation failures. If no calibration was applied, retain the explicitly registered target MS and its DATA column as the candidate output (flagging may still have changed it).

### 2. Detect flag additions from task summaries

Capture the return value of `hifv_flagtargetsdata()` in `recipe_b()` (currently discarded around line 615). Its `FlagtargetsdataResults` contains `mses` and `summaries`. Each entry in `summaries` corresponds to an MS and contains summary dictionaries named `before` and `template`. Read their `flagged` and `total` counts, matching summaries by name rather than assuming fixed list positions.

Also inspect `r.summaries` on each final applycal result from `result.applycal_result_contline`. With `flagsum=True` (the installed default), these dictionaries are named `before` and `applycal`. This captures flags added while applying calibration, including failed-solution flagging. Checking only the earlier target-flagging task would miss these additions.

Exact upstream locations:

- `hifv/tasks/flagging/flagtargetsdata.py`: `FlagtargetsdataResults` stores `mses`, `summaries` and executed commands. `Flagtargetsdata.prepare` builds the per-MS summaries (around lines 266–295). `_get_flag_commands` requests the `before` and `template` summaries (around lines 310–328).
- `h/tasks/applycal/applycal.py`: `ApplycalInputs.flagsum` defaults to True (line 65). `SerialApplycal.prepare` schedules summaries named `before` and `applycal`, then stores them in `result.summaries` (around lines 293–330).

For each relevant stage, use `after['flagged'] - before['flagged']`. Under a verified add-only recipe, unchanged counts with identical sample scope and total mean no flags changed. Record the counts and stage names in `recipe.json`, replacing the flag-array hash comparison as the source of the flag-change decision. Verify executed template commands do not unflag or restore flags. Equal counts are not sufficient if flags can be removed or restored. Missing summaries, mismatched scopes or unexpected count decreases mean unknown, not unchanged.

Reuse the verified original metrics only when there is no recorded calibration application, all relevant flagging stages are unchanged (or explicitly skipped), and the recipe guarantees unchanged weights and sampling. The installed self-calibration application uses `calwt=False`. If flags were added, evaluate DATA even without applied calibration and rebuild the dirty reference with the output flags. Keep the recipe-specific dirty reference when calibration changes flags too. Flagging summaries do not establish that weights or sampling stayed unchanged.

Validation before use: exercise one accepted correction, one no-application/no-new-flags case, and one flagging-only case. Confirm the first uses the recorded corrected column, the second reuses before metrics, and the third is re-imaged. Missing or ambiguous bookkeeping must never silently become “unchanged”.

## Regeneration validation is opt-in

[USER] 2026-11-10. Disable regeneration hashes and reproduction checks by default. This supersedes the mandatory regeneration verification and default representative reproduction checks described above. Implemented 2026-11-10.

- Add `--verify-regeneration` (default False) to enable the regeneration audit in `casa_regen`: noiseless DATA hash, nominal-noise difference hash, paired-noise hash and comparisons with the archived case, together with the associated provenance, case-parameter and gain-consistency checks. When disabled, skip computing these audit hashes as well as comparing them. Propagate the option to noise-generation helpers so they do not compute unused audit hashes internally.
- Change the existing `--reproduce` default from `representative` to `none`. Explicit `--reproduce representative` (or a case selection) enables before-calibration re-imaging and comparison with archived eC/eD. This is independent of `--verify-regeneration`.
- To enable both checks for a pilot, supply `--verify-regeneration --reproduce representative` when creating its plan. Persist both settings in `plan.json` and pass them to the execution stages.

Keep the original saved seeds, source parameters and paired-noise generation unchanged. Disabling verification changes validation cost, not the generated experiment. Record checks as disabled or not performed, never as passed or verified. This also supersedes “verified original metrics” in the reuse rule above when validation is disabled (reuse the archived baseline under the recorded reproducibility assumption).

These switches concern regeneration/reproduction validation. Continue using pipeline calibration-application records and flagging summaries to select and evaluate the output. Keep ordinary input validation and runtime error handling. If `--verify-regeneration` is enabled and a hash check fails, invalidate every before/after comparison that depends on that regenerated case. Likewise, if an enabled reproduction check records `DISAGREE`, invalidate the affected comparisons instead of continuing to compare corrected results against the archived baseline. Record the failure reason and exclude those comparisons from recovery statistics and dataset-filtering decisions until the disagreement is resolved. Independent cases may continue. A hash mismatch and reproduction `DISAGREE` are separate failure paths, and both must enforce this rule.

## Recovery criterion and reporting

[USER] 2026-11-10. Define acceptable reconstruction as final recovery error $e_C < 0.05$ (less than 5% relative L2 reconstruction error against the known sky). This supersedes the provisional joint eC/eD threshold and the pending threshold decision above. eD remains a diagnostic, not an additional pass requirement.

- Fixed: $e_C^{before} \ge 0.05$ and $e_C^{after} < 0.05$.
- Already acceptable: both errors are below 0.05.
- Still unacceptable: both errors are at least 0.05.
- Degraded: $e_C^{before} < 0.05$ and $e_C^{after} \ge 0.05$.

Keep reporting simple. For each recipe, show the outcome counts and median eC before and after on the same valid cases. Retain per-case before/after eC values in the results. List invalid or unavailable comparisons separately. No additional percentile, spread or auxiliary-metric summaries are required.

Add a second table alongside the general per-recipe table, grouping each recipe's results by injected corruption combination: amplitude, phase, noise, amplitude+phase, amplitude+noise, phase+noise, and amplitude+phase+noise. For each group, show the number fixed out of initially unacceptable valid cases (and the corresponding percentage), plus median eC before and after on the same valid cases. If no cases were initially unacceptable, show the fixed percentage as not applicable. Keep uncorrupted controls separate.

## Dirty-only reference and metric timing

[USER] 2026-11-10. Implementation requirement (implemented 2026-11-10): in `scripts/vla_pipeline_qa.py`, `casa_image()` currently calls `image_cell(cell / "dirty_reference", ...)` (around line 780). That helper performs full recovery imaging, including CLEAN, although `scripts/dataset_v2_matched_extract.py::extract()` reads only `reference_record["products"]["dirty"]` from this reference. Replace this reference call with dirty-only imaging (zero CLEAN iterations), returning the dirty product needed by extraction. Match the evaluation dirty image's field, channels, grid, cell size, weighting, normalization and other relevant imaging settings. Use the original noiseless visibilities with the recipe output's final flags and weights. Do not CLEAN this reference. Keep CLEAN imaging of the actual recipe output to compute recovery error.

The main comparison is the original input versus the recipe's final output. No additional intermediate before/after-flagging imaging is required.

| Measurement | Data being imaged | Recovery error eC | Dirty error eD |
| :-- | :-- | :-- | :-- |
| Before either recipe | Original corrupted input, before any recipe calibration or additional flagging | Original CLEAN reconstruction against the fixed known sky at the common restoring beam | Original dirty image against its noiseless reference with the original flags and weights |
| After flagging-off recipe | Final recipe output after calibration, with no added recipe flags | Final CLEAN reconstruction against the same known sky at the same common restoring beam | Final dirty image against a noiseless reference with the final weights and unchanged flags |
| After flagging-on recipe | Final recipe output after all calibration and flagging performed by that recipe | Final CLEAN reconstruction using the final flags and weights, against the same known sky at the same common restoring beam | Final dirty image against a noiseless reference with those same final flags and weights |

For eC, retain the existing fixed evaluation grid, region, restoring beam and known-sky normalization across before/after comparisons. For eD, use the existing relative L2 definition against the corresponding noiseless dirty reference (including that reference's norm in the denominator). Reuse the original dirty reference when flags and weights are unchanged. Rebuild it with dirty-only imaging when they change.

Use final $e_C < 0.05$ to decide acceptable recovery. eD is diagnostic because its reference changes when samples are flagged. A lower eD alone does not establish better recovery or account fully for discarded information. The flagging-on and flagging-off recipes compare the effect of allowing flagging throughout the recipe (including its effect on calibration solutions).

If no calibration is applied but flags change, still image the final data and compute both metrics. If no calibration is applied and flags, weights and sampling are unchanged, reuse the original metrics under the recorded regeneration-validation policy. Missing output or invalid comparisons remain unavailable, not successful recovery.

## Skip re-imaging unchanged recipe outputs

[USER] 2026-11-10. Implementation requirement (implemented 2026-11-10): in `scripts/vla_pipeline_qa.py::unit()`, before launching `--casa-image` for a recipe, inspect the recorded pipeline application results and flagging summaries described above. If no calibration was applied and all relevant flagging stages added no flags (or were explicitly skipped), reuse the archived before metrics instead of re-imaging. This shortcut requires the recipe to preserve weights, sampling and relevant imaging metadata. Establish that from the recipe contract and pipeline bookkeeping, without comparing MS column contents.

Set the after metrics equal to the before metrics and record `evaluation_reused=True` with reason `no calibration applied and no flags changed`. Do not run evaluation CLEAN or dirty imaging, and do not rebuild the dirty reference. Reuse follows the opt-in regeneration-validation policy above (disabled validation is not a passed check).

A `not_applied` disposition alone is insufficient. If flagging changed the data, compute both final metrics. If calibration was applied, evaluate its output even if the correction might be numerically small. Missing or inconsistent pipeline bookkeeping cannot qualify an output for reuse. Explicitly failed validation still invalidates the comparison.

## Native image diagnostics are opt-in

[USER] 2026-11-10. Implementation requirement (implemented 2026-11-10): add `--native-diagnostics` (default False). Persist it in `plan.json` and pass it to `recipe_b()` in `scripts/vla_pipeline_qa.py`.

When disabled, skip `save_native()` entirely (currently called around lines 639–646). Do not collect central crops, scan internal image products or compute their whole-image minimum, maximum, sum and RMS. Record native diagnostics as disabled, not failed or missing.

When `--native-diagnostics` is enabled, retain the existing crop/statistics collection for pilot runs or selected cases. The missing recipe identifier is a concrete bug and must be fixed: `casa_recipe()` has a record containing `recipe`, but `recipe_b()` creates a separate local record without that key. Its `record['recipe']` lookup raises `KeyError`, which is caught as `native_crops_error`, preventing diagnostic capture. Pass the recipe identifier explicitly from `casa_recipe()` to `recipe_b()` and use it to construct the diagnostic output path. Verify that enabling native diagnostics saves the crops without this error.

This option controls only extra diagnostic extraction. Keep the pipeline's internal imaging needed for self-calibration, the main eC/eD evaluation, calibration application records, flagging summaries and failure reasons unchanged.

## Configurable self-calibration image size

[USER] 2026-11-10. Implementation requirement (implemented 2026-11-10): add `--selfcal-imsize N` to try different internal self-calibration image dimensions (a square N × N image). When omitted, retain the pipeline's automatic sizing. Persist the setting in `plan.json` and pass an explicit size to `hif_selfcal(hm_imsize=...)` in `recipe_b()` using the pinned task's expected format. Validate the requested size against its supported dimensions. Do not rely only on `hif_checkproductsize(maximsize=...)` to control self-calibration imaging.

For example, run separate plans with `--selfcal-imsize 512` and `--selfcal-imsize 1024`. Keep the cell-size rule unchanged across this comparison and record the actual internal image dimensions and cell size. Confirm that the requested size reaches the self-calibration images. Label each run with its chosen size, since changing the field of view can change the model, noise estimates and accepted calibration solutions. This supersedes the earlier prohibition on reducing B's internal image size when an explicit size is requested.

This argument affects only the internal images used to obtain and assess self-calibration solutions. Keep the Dataset v2 evaluation grid, field of view, dimensions, cell size, restoring beam and evaluation region unchanged. Compute final eC/eD by re-imaging the recipe output on that original evaluation grid. The noiseless dirty reference uses the same evaluation grid and the recipe's final flags and weights.

## Configurable self-calibration masking

[USER] 2026-11-10. Implementation requirement (implemented 2026-11-10): add `--selfcal-masking {auto,center}` (default `auto`). Persist the selection in `plan.json` and label the recipe results with it.

- `auto`: leave `hif_selfcal`'s user mask unset and retain its automatic masking.
- `center`: reuse `scripts.imaging.config.central_circle_mask(imsize, beam_major_arcsec, diameter_beams)`. Use the existing `DEFAULT_MASK_NBEAMS` rule (currently diameter 6 beam widths, radius 3). Generate the region for the actual internal self-calibration grid, using a fixed pre-selfcal reference beam for each source so the region is consistent across its cases and iterations. Supply it through `hif_selfcal(usermask={field_name: mask})`, using a CASA-compatible region file if required by the imaging calls. Preserve its angular size when internal image dimensions change. Do not reuse pixel coordinates from a different grid or supply the known source flux.

Version support: the installed pipeline 2025.1.0.36 already has the required code path, so center masking should work with that version. The same path was checked in VLA pipeline 2026.2.1.55. In `hif/tasks/selfcal/selfcal.py`, the field-keyed `usermask` becomes `sc_usermask`. In `hif/heuristics/auto_selfcal/auto_selfcal.py`, it is passed into initial CLEAN, the iterative self-calibration calls and final imaging. The imaging wrapper selects `usemask='user'` when a mask is supplied. The comment saying the mask is only used for dirty imaging is stale. The task documentation's “not implemented” label also does not reflect this code path. This establishes support in the checked versions, not the earliest release that introduced it.

Before production, verify on a small run that the supplied central region is used in initial, iterative and final self-calibration imaging. Check actual mask coverage and recorded imaging arguments. A requested center mask must not silently fall back to automasking. No pipeline upgrade or internal patch is expected to be necessary, subject to this runtime check.

This option changes only internal self-calibration masking. Keep the existing common evaluation imaging and eC/eD definitions unchanged. Center masking supplies approximate source-location information, so report it as a distinct recipe from automasking.

Source check: [pipeline 2026.2.1.55 self-calibration implementation](https://open-bitbucket.nrao.edu/projects/PIPE/repos/pipeline/browse/pipeline/hif/heuristics/auto_selfcal/auto_selfcal.py?at=refs/tags/2026.2.1.55). Installed 2025.1.0.36 code was inspected locally.

## Implementation notes (2026-11-10)

Code: `scripts/vla_pipeline_qa.py`, `scripts/recovery_products.py` (`image_cell.dirty_only`), `scripts/injection_bounds_pilot.py` (`add_paired_noise(audit=...)`, default unchanged for every other caller) and `scripts/vla_selfcal_mask_report.py`. Plans are now `schema: 2` and record `verify_regeneration`, `reproduce`, `options` (`native_diagnostics`, `selfcal_imsize`, `selfcal_masking`) and a `variant` label (e.g. `imsize=512, masking=center`) that is copied into every result.

- Output selection (B): `recipe_b()` unwraps the `SelfcalResults`, collects `r.applied -> calapp.calto` from `applycal_result_contline` (deduplicated by MS, field, SPW and intent), resolves the field through `ms.get_fields(task_arg=..., intent=...)` and reads the registered column with `ms.get_data_column(SELFCAL_CONTLINE_SCIENCE or SELFCAL_CONT_SCIENCE, ...)`. The list is stored as `applications` in `recipe.json`. Unresolvable fields, a field other than the target, a missing `CORRECTED_DATA` registration or applications to more than one MS give `runtime_failure` ("inconsistent calibration application records"). The MS scan and the `CORRECTED_DATA != DATA` test are removed.
- Flag bookkeeping (B): `flag_stages` holds one entry per stage (`hifv_flagtargetsdata` before/template per MS, `hif_selfcal applycal` before/applycal per applycal result, or `skipped`). Status is `unchanged`, `added` (with the count) or `unknown` (missing summary, different totals, a decrease, or template commands containing `unflag`/`restore`). `flagging_off_violated` and `no_usable_output` now come from these summaries. `contfile_present` records whether a `cont.dat` could trigger hif_selfcal's temporary line flagging (restored by the task itself).
- Evaluation decision (B), stored as `evaluation_decision`. Reuse (no imaging, after = before, `evaluation_reused=True`) only when there is no application, no `added` or `unknown` stage and no applied table with `calwt=True`. Otherwise the output is split from the recorded MS and column (DATA when nothing was applied) and imaged. The dirty reference is the source's own when calibration was applied with unchanged flags, and is rebuilt otherwise. No MS columns are hashed for B, and the old post-recipe `flag_stats`/`table_hash`/`same_rows` pass runs for A only (A is our own `applycal` and keeps column-based detection).
- Dirty-only reference: `image_cell.dirty_only()` runs one `tclean` with `niter=0`, built from the same `_direct_base_parameters`, fixed grid, restoring beam and `pblimit` as the dirty step of the evaluation imaging, and exports only `dirty.fits.gz`. `casa_image` takes a fourth argument (`auto` for A and reproduction, `reuse`/`rebuild` from B bookkeeping). The row check before rebuilding is kept.
- Validation: with `--verify-regeneration` off, `casa_regen` computes no hashes and makes no archived comparisons, and records `verification.status = "disabled"`. When on, a mismatch is recorded as `FAILED` instead of raising. A failed audit or a reproduction that is not `agree` makes `unit()` write every recipe of that case as `status: invalid` without running the recipes (saves the pipeline time), and the case is excluded from all counts.
- Center mask: the region is written as a CRTF file in world coordinates at the field phase centre, `circle[[RA, Dec], r]` with $r = 0.5 \times$ `DEFAULT_MASK_NBEAMS` $\times$ the major axis of the source's matched-v2 restoring beam (`recovery/reference.json`). This uses the `central_circle_mask` rule but not its pixel string, because with automatic sizing the internal grid is only known inside hif_selfcal, and world coordinates keep the region independent of `hm_imsize`. Runtime check: `mask_usage` counts `usemask=` values in the CASA log, and `center_mask_fallback=True` is recorded if no user mask was seen or any automasked tclean ran.
- Image size: `--selfcal-imsize N` is validated (even, factors 2, 3, 5, 7 only) at plan time and passed as `hm_imsize=[N, N]`. The internal imsize/cell from the hif_selfcal targets are kept in `selfcal_result.targets` (list values are now retained).
- Report: outcome classes use eC < 0.05 (`fixed`, `already_ok`, `still_bad`, `degraded`). The per-recipe table shows dispositions, reuse count, outcome counts and median eC before/after. The second table groups by injected corruption with fixed / initially unacceptable (%), controls separate. Invalid and unavailable comparisons are listed separately and excluded.
- `hifv_flagtargetsdata` without a template file runs only the `before` summary. Its executed command list then contains no flag command, so the stage is recorded as `unchanged` ("no flag commands executed (no template file)"), established from the commands rather than from a missing summary. Pipeline task results come back as a `ResultsList` (iterable, not a Python list), which `unwrap()` handles.

Local checks (2026-11-10, source 0449+113, pipeline 2025.1.0.36, `--selfcal-imsize 512`):

- `runs/vla-pipeline-qa-B-fixes-test-20261110` (control + phase case, center mask, audit and reproduction on, native diagnostics on): audit `passed` for both, reproduction `agree`, all 4 recipes `applied`, `mask_usage` only `user` (no fallback), crops saved without error, application records resolved to `pipe_in.ms:CORRECTED_DATA` with no problems. Phase case flagging_on: hif_selfcal applycal added 24,244 flags, the dirty reference was rebuilt dirty-only, eC 0.034 → 0.006 (fixed).
- `runs/vla-pipeline-qa-B-fixes-test2-20261110` (control, automasking): audit `disabled`, reproduction `not performed`, the template-less flagtargetsdata stage `unchanged`. At 512 px the automask found a region and the control was self-calibrated in both recipes (it was not at the pipeline's own 7200 px). So the reuse path was not exercised in a real run yet.
- Runtime: 2–4 min per B recipe at 512 px, against about 18 min per recipe at the pipeline's own size on the Mac. About 7–9 min per case in total, including regeneration and evaluation imaging.
- Flagging-off amplitude results require investigation before scientific use. The source/log review below supersedes the earlier speculative explanation involving flagged solutions or polarization normalization.


## Review of local results and amplitude-scale failure (2026-11-10)

The new plumbing runs, but these tests do not establish that the flagging-off recipe is scientifically valid. Both affected cases start below the chosen 0.05 threshold, so they are degraded controls/baselines, not failed recoveries of initially unacceptable examples. The flagging-on phase result is improved, not exactly zero. The original center-mask control does not show the amplitude anomaly because only a phase table was finally applied.

| Run and case | Flagging | Before eC | After eC | Interpretation |
| :-- | :-- | :-- | :-- | :-- |
| Test 1, center mask, control | off | 0.000189 | 0.000193 | Already acceptable, phase-only final application |
| Test 1, center mask, control | on | 0.000189 | 0.000615 | Already acceptable |
| Test 1, center mask, phase | off | 0.034413 | 0.412835 | Degraded, amplitude table applied |
| Test 1, center mask, phase | on | 0.034413 | 0.005966 | Improved and already acceptable, 24,244 flags added at final application |
| Test 2, automask, control | off | 0.000189 | 0.412824 | Degraded, amplitude table applied |
| Test 2, automask, control | on | 0.000189 | 0.000615 | Already acceptable |

Sources: `runs/vla-pipeline-qa-B-fixes-test-20261110/results/` and `runs/vla-pipeline-qa-B-fixes-test2-20261110/results/`. These are six recipe outputs from one source, not a general success-rate estimate. Test 1's target-flagging stage was recorded as unknown by the earlier wrapper-handling code. Test 2 confirms the repaired bookkeeping for the no-template case. Successful audit, reproduction and saved native diagnostics in test 1 support those individual changes. The real no-application reuse branch still needs a case the pipeline skips.

### Leading diagnosis: calonly skips writing the CLEAN model

Confirmed by inspecting the installed pipeline source and saved logs:

- In `pipeline/hif/heuristics/auto_selfcal/auto_selfcal.py::run_selfcal`, the model-prediction call is guarded by `if applymode == "calflag"` (around lines 1725–1736). It writes the current CLEAN model with `savemodel='modelcolumn'` before gain solving.
- The loop uses `np.unique([requested_applymode, 'calonly'])`. Explicitly requesting `calonly` therefore never enters that model-writing branch. Ordinary imaging calls use `savemodel='none'`.
- All saved flagging-off logs have zero model-prediction calls. Flagging-on logs contain them (seven in test 1's phase case and two in the controls). The experiment intentionally starts with no MODEL_DATA or CORRECTED_DATA column.
- The calonly amplitude solve uses `solnorm=False`. In the affected logs, the source peak rises from about 707.68 mJy/beam to about 1000 mJy/beam immediately after the first amplitude round, inside the pipeline. The final application and our evaluation therefore inherit an already shifted scale.

CASA assumes a 1 Jy point-source model when no model is defined ([NRAO self-calibration explanation](https://science.nrao.edu/science/meetings/2021/vla-data-reduction/presentations/Selfcal_NRAO_DRW_2021.pdf)). The missing model-write path is consequently the leading explanation: amplitude calibration fits the data to the default model instead of its reconstructed source. For the control's true flux 0.707660377 Jy, scaling to 1 Jy predicts fractional excess $1/0.707660377-1 \approx 0.413107$. The measured reconstructed fluxes are 0.999800–0.999807 Jy, consistent with the measured eC near 0.41283. The proximity to $\sqrt{2}-1$ is not evidence of a polarization-normalization bug.

This diagnosis is strongly supported by source and logs, but the temporary MSes and calibration tables were deleted. The actual model seen by gaincal has not been inspected and a corrected rerun has not yet been performed. Do not label the exact runtime cause experimentally confirmed. The log message about calwt=True is a generic message in `unflag_failed_antennas`, not evidence that weights were changed here. The application records specify calwt=False.

### Consequence and next verification

Treat the affected flagging-off amplitude outputs as diagnostic failures of the current recipe implementation, not evidence that Dataset v2 corruptions cannot be corrected or that flagging is necessary. Preserve the recorded values but do not use those outputs to filter the dataset or compare scientific recovery rates until this is resolved. Other tested engineering changes remain useful, and the flagging-on outputs remain pilot measurements rather than proof of general validity.

The targeted correction to test is ensuring that every attempted gain solve has the current reconstructed CLEAN model predicted into the MS, even when the requested apply mode is calonly. Use the reconstructed model, not the true injected sky. Do not compensate by dividing the output flux by sqrt(2), injecting a known flux model or disabling amplitude calibration. Merely enabling gain normalization could hide a missing-model error.

Run one isolated control with work products retained and inspect the model immediately before the amplitude solve, the resulting gains and the final recovered flux. Compare the unmodified and corrected model-write paths with identical input and masking, then repeat the phase case. No pipeline installation or recipe code was changed by this review. Also note that the internal calonly heuristic changes solver settings and can issue internal applycal calls with calflag while restoring temporary flags. The current flagging-off promise concerns the final output flags, not the absence of every internal flagging operation.

### Implemented correction: calonly model-write fix (2026-11-10)

Code (`scripts/vla_pipeline_qa.py`): `patch_calonly_model_write()` runs inside the CASA pipeline process of a B flagging_off recipe, before `hif_selfcal`. It rebuilds `SelfcalHeuristics.run_selfcal` from the installed source with the single guard `if applymode == "calflag":` (the model-write branch) replaced by `if applymode == "calflag" or applycal_mode[band][target][iteration] == "calonly":`. The installation is not modified. The rebuilt method keeps the module globals and the installed file's line numbers. A requested calflag is unchanged (its calonly fallback reuses the model written earlier in the same iteration). If the guard is not found exactly once, the recipe fails rather than running unpatched. The model written is the pipeline's own reconstructed CLEAN model, not the injected sky, and gain normalization is unchanged.

- Default on for flagging_off. `--no-calonly-model-fix` reproduces the unmodified pipeline for comparison (labelled in the variant). Each record has `calonly_model_fix` and the recipe modification label includes "calonly model-write fix".
- With `--native-diagnostics`, every gaincal call made by hif_selfcal is logged in `solve_models`: calmode, solnorm, solint, the median |MODEL_DATA| over unflagged parallel-hand samples just before the solve, and the median/min/max solved gain amplitude. The pipeline's gaincal JobRequest factory is wrapped, so the calls stay in `casa_commands.log`.
- `--keep-work` keeps the work MSs and calibration tables for inspection.

Verification (`--selfcal-imsize 512`, automasking, flagging_off, work kept, identical input and masking):

| Case | Model-write fix | Model before solves | Amplitude gain (median) | Final tables | eC before → after | Flux error after |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| control | off | no MODEL_DATA column | 0.8412 | inf_EB_0_p, inf_1_p, inf_ap_6_ap | 0.0002 → 0.4128 | +0.4128 |
| control | on | model written | no amplitude solve | inf_EB_0_p | 0.0002 → 0.0006 | 0.0000 |
| phase | off | no MODEL_DATA column | 0.8412 | inf_EB_0_p, inf_1_p, inf_ap_6_ap | 0.034 → 0.4128 | +0.4128 |
| phase | on | model written | 1.0000 | inf_EB_0_p, 15.00s_3_p, inf_ap_6_ap | 0.034 → 0.0060 | +0.0002 |

Runs: `runs/vla-pipeline-qa-B-calonly-nofix-20261110` and `runs/vla-pipeline-qa-B-calonly-fix-20261110`. The unmodified amplitude gains equal $\sqrt{0.7077} = 0.8412$, which is exactly the solution of the data against a 1 Jy default model. Applying them scales the source by $1/0.7077 = 1.413$. This confirms the diagnosis above experimentally. The kept self-calibration MS of the fixed control holds a Stokes I model in RR and LL (median 0.696 Jy against the true 0.708 Jy, cross-hands zero). The first fixed run logged 0.0 for the model because its median included the zero cross-hands. That diagnostic now uses the parallel hands only. With the fix, the flagging_off phase result (eC 0.0060) matches flagging_on (0.0060). The earlier flagging_off results in `runs/vla-pipeline-qa-B-fixes-test*-20261110` are superseded and must not be used.

Rerun with the corrected diagnostic (`runs/vla-pipeline-qa-B-calonly-fix-model-20261110`, phase case, fix on): the model before each solve grows with the CLEAN depth of each solint, 0.655 Jy (inf_EB) → 0.697 (inf) → 0.706 (48 s) → 0.7077 (15 s, int). It is 0.7077 Jy immediately before both amplitude solves (true flux 0.70766 Jy), with median amplitude gains 0.99999 and 1.00004. The result is identical to the first fixed run (eC 0.034 → 0.0060, flux error +0.0002).

## Validation checks (2026-11-11)

Two test layers. `tests/test_vla_pipeline_qa.py` (unittest, no CASA, 18 tests) covers the pure logic: flag stages (unchanged, added, unknown for missing/decreasing/different-scope/malformed summaries, named not positional), the B decision (now the pure function `decide_b()` called by `recipe_b()`), `ResultsList` unwrapping, image-size validation, eC outcome classes, mask-usage counting, variant labels and the calonly fix text. Run with `ml/.venv/bin/python -m unittest tests.test_vla_pipeline_qa`.

`scripts/vla_pipeline_qa_checks.py` runs small pipeline cases concurrently and asserts on them (exit code 1 on any FAIL, results in `ROOT/checks.json`). Run `runs/vla-pipeline-qa-checks-20261110` (`--compare-unfixed`): all three checks PASS.

- `model_fix` (flagging_off, fix on, control + phase, 512 px): the fix is active in the pipeline process, a nonzero model is present before every gain solve, the model before both amplitude solves is 0.7077 Jy (known flux 0.70766 Jy), amplitude gains 0.99999 and 1.00004, flux error after $5 \times 10^{-5}$ (control) and $2.3 \times 10^{-4}$ (phase), eC after 0.0006 and 0.0060. The unmodified pipeline on the same input reproduces the +0.413 offset in both cases.
- `calonly_preserve` (same run, work kept, then deleted): the pipeline records `hifv_flagtargetsdata` skipped and no flags added at the final applycal, no `flagging_off_violated`, calwt False on every applied table. A direct comparison of FLAG, FLAG_ROW, WEIGHT and WEIGHT_SPECTRUM between the input observed.ms and the recipe output finds identical rows and no differing column. So the internal calonly heuristics (temporary flag restores, internal calflag applycal calls) leave the final flags and weights unchanged. The flux scale matches the flagging_on reference within $4 \times 10^{-4}$.
- `reuse` (control at the pipeline's own image size, both recipes): hif_selfcal recorded no application (empty automask), the flag stages are skipped (flagging_off) or unchanged with no template (flagging_on), the decision is reuse, after equals before with `evaluation_reused=True`, and no evaluation imaging or output split happened. About 6.4 min per recipe.

Scope: one source (0449+113), two cases, 512 px internal images except the reuse runs. The checks establish that these code paths behave as intended, not general recovery rates. In the reuse runs no final applycal ran, so only the target-flagging stage contributes bookkeeping, which is correct for that path.


## Running the experiment grid (2026-11-11)

One `scripts/vla_pipeline_qa.py` run holds one experiment and one set of self-calibration options. Both flagging recipes (`flagging_off`, `flagging_on`) are always included. Image size and masking are stored once per plan, so each B combination is its own run. The full grid is 7 runs:

| Run | Experiment | `--selfcal-imsize` | `--selfcal-masking` |
|---|---|---|---|
| A | A: gaincal with the true model | – | – |
| B-512-auto | B: hif_selfcal | 512 | auto |
| B-512-center | B | 512 | center |
| B-1024-auto | B | 1024 | auto |
| B-1024-center | B | 1024 | center |
| B-orig-auto | B | not set (pipeline's own size, 7200 px for 0449+113) | auto |
| B-orig-center | B | not set | center |

Use the same `--source-run`, `--cases` and `--parallel` in every run, so the cases are paired across variants and the runtimes comparable. Each run gets its own output folder. The calonly model-write fix is on by default for flagging_off; do not pass `--no-calonly-model-fix` in these runs. The expensive optional checks stay off by default (`--verify-regeneration`, `--reproduce`, `--native-diagnostics`, `--keep-work`, `--model`).

`--cases representative` picks, for every source in the source plan, its control plus one case per corruption type (amp, phase, noise and their mixtures) of the first recipe. `representative:N` limits this to N sources evenly spread in nominal source S/N; N counts sources, not cases. The grid uses `representative:3`: 3 sources × 8 cases (control, amp, phase, noise, amp+phase, amp+noise, phase+noise, amp+phase+noise) = 24 cases per run on a source run with at least 3 sources. The smoke run has a single source (0449+113), so there it gives 8 cases (checked with `--plan`). For a fixed small set, pass comma-separated case ids instead.

The block uses Bash syntax (functions and arrays). The default macOS shell is zsh, so start Bash first.

```bash
bash
cd /Users/u1528314/Documents/radioastro-ml   # repository root
set -euo pipefail

qa() { bin/run ml scripts/vla_pipeline_qa.py "$@"; }

SRC=runs/dataset-v2-matched-smoke-20261105   # source run (read only)
CASES=representative:3                         # 3 sources spread in S/N, 8 cases each (see note above)
PAR=2                                          # cases run concurrently; keep fixed across runs
OUT=runs/vla-pipeline-qa-grid-$(date +%Y%m%d)

# Plan: A, then B for 3 image sizes x 2 maskings
qa --output "$OUT-A" --plan --source-run "$SRC" --experiment A --cases "$CASES"
for size in 512 1024 orig; do
  for mask in auto center; do
    opts=(--selfcal-masking "$mask")
    if [[ "$size" != orig ]]; then
      opts+=(--selfcal-imsize "$size")
    fi
    qa --output "$OUT-B-$size-$mask" --plan --source-run "$SRC" --experiment B --cases "$CASES" "${opts[@]}"
  done
done

# Execute and report in this order: cheapest first, the pipeline's own image size last
runs=(
  "$OUT-A"
  "$OUT-B-512-auto" "$OUT-B-512-center"
  "$OUT-B-1024-auto" "$OUT-B-1024-center"
  "$OUT-B-orig-auto" "$OUT-B-orig-center"
)
for run in "${runs[@]}"; do
  qa --output "$run" --execute --parallel "$PAR" || echo "WARNING: crashed cases in $run, see $run/logs" >&2
  qa --output "$run" --report
done

# One combined report over all runs
bin/run ml scripts/vla_pipeline_qa_compare.py --output "$OUT-compare" "${runs[@]}"
```

Runtime and resuming:

- There is no time limit on the CASA subprocesses. Disabling the optional checks does not bound self-calibration time. The `orig` variants are the expensive ones (the pipeline's own image size took about 18 min per recipe locally), which is why they run last. A stuck case blocks its worker until it is killed by hand or by the job's wall time.
- Recipe failures are recorded in the case results and do not stop `--execute`. A case process that crashes (no `units/<case>.json` written) makes `--execute` exit non-zero after the other cases finish. The loop prints a warning and continues with the report and the next runs instead of stopping the grid under `set -e`.
- `--execute` skips every case that already has `units/<case>.json`. That includes completed cases whose recipes recorded a failure. Re-running `--execute` therefore resumes cases that never finished (for example after a killed job), but does not retry failed recipes. To retry a case, delete its `units/<case>.json` and `results/<case>__*.json` first, then run `--execute` and `--report` again.

On CHPC, submit each `--execute` as a job with `bin/submit ml-cpu scripts/vla_pipeline_qa.py --output <run> --execute --parallel <N>` and match `<N>` to the job's CPUs. This submission path is UNCONFIRMED for this experiment (it starts CASA pipeline subprocesses inside the ML job). The job's wall time is then the only runtime limit.

`scripts/vla_pipeline_qa_compare.py` only reads the run folders. It writes `comparison.html` and `comparison.json`: one row per run variant × recipe with dispositions, eC outcomes (fixed, already ok, still bad, degraded), median eC before → after, the same medians on the cases measured in every row (paired), fixed counts by corruption type and median recipe/case runtime. B flagging_off rows without the calonly fix are greyed out as superseded.
