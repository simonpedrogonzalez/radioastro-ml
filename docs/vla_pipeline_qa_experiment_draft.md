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
  Optional reproduction re-images the regenerated case and compares eC/eD with the committed values.
- A (provisional, not frozen): `gaincal` p (solint int) then ap (solint inf, per scan), minsnr 3, minblperant 4,
  solnorm False, MODEL_DATA = noiseless V0, refant = most unflagged visibilities (ties lowest id),
  applycal calwt False, `calonly` (flagging_off) or `calflag` (flagging_on).
- B: pipeline 2025.1.0.36 / CASA 6.6.6.18 locally. hifv_importdata -> [hifv_flagtargetsdata if flagging_on]
  -> hif_checkproductsize -> hif_makeimlist -> hif_selfcal(apply=True). (No hif_makeimages, see decision below.)
  flagging_off = `apply_cal_mode_default="calonly"` and no flag template, labelled a modified recipe.
  Default `amplitude_selfcal` is False in this version, so `default` is run once as `phase_only`.
  The corrected MS is found by checking every MS for CORRECTED_DATA != DATA. Pipeline log decision lines are kept.
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
