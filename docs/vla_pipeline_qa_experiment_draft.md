# VLA pipeline QA experiment (draft, decisions only)

Source: vault `Notes/Week 10 RadioastroML Update 2026-11-04.md`, Jeff Feedback, TODO Idea 2.
Status: DRAFT. Decisions agreed with the user on 2026-11-09. To be rewritten into a full design.

## Goal

Find which simulated corruptions survive standard VLA processing. For each Dataset v2 case:
simulate → corrupt → CLEAN → measure, then pipeline step → CLEAN → measure again, and say whether the
pipeline step "succeeded".

## Decisions

1. **Two separate experiments** [USER]
   - **A. Pipeline-style gain calibration.** Emulate what the VLA calibration pipeline does on a gain
     calibrator: `gaincal` against the known point-source model, `applycal`, re-image with the unchanged
     matched-v2 imaging. Fast, runs on every case. Answers "would gain calibration remove this error?".
   - **B. VLA imaging pipeline with `hif_selfcal`.** Real NRAO pipeline through the existing
     `scripts/vla_pipe.py` wrapper (calibrator promoted to target), `casa-pipeline` runtime. Slow, so a
     subset only. May be skipped/fail on calibrator-as-target data: record that as an outcome, not an error.
   - The full VLA *calibration* pipeline cannot run: cases are single-field, already-calibrated data with
     corruptions injected afterwards (no flux/bandpass/phase calibrator scans).

2. **Visibilities are regenerated** [USER]
   - Dataset v2 deletes every `observed.ms` after imaging. Only image products and records are kept.
   - Regenerate each case from `plan.json` with the same code path as `create_dataset_v2_matched.execute_unit`
     (noiseless V, nominal noise seed, gain injection, paired noise).
   - Verify regeneration against the stored `v_sha256` and `nominal_noise_sha256` before using a case.
   - The "before" measurements are the already committed case values (eC, eD, features). Only "after" is new.

3. **Measurements, before and after** [USER]
   - eC (CLEAN recovery error) and eD (dirty-image change), same definitions and masks as matched v2.
   - **Model predictions before and after**: run the saved matched-v2 model (frozen reload in `ml.multitask`)
     on the before and after features. Report predicted presence and eC/d next to the measured ones.
   - Success rule (provisional): pipeline step succeeded if eC ≤ 0.05 and eD ≤ 0.05 after it.

4. **Smoke test first** [USER]
   - Local smoke on the existing smoke run `runs/dataset-v2-matched-smoke-20261105` (source 0449+113,
     85 cases) for experiment A, and a few cases for experiment B.
   - Then the full v2 train/val run on CHPC (Slurm, sharded like matched v2). Not launched yet.

5. **Report** [USER]
   - Plots/tables that show how the pipeline step performs, clear language, very little text.
   - Candidates: success rate by corruption type and strength, eC before vs after scatter,
     table of corruptions that survive (eC after > 0.05), model predictions before vs after.

## Open (to decide when writing the full design)

- Experiment A gaincal settings: `calmode` (`p` then `ap`?), `solint` (`int` / per scan), `refant`
  (never the corrupted antenna), `minsnr`, and whether to flag failed solutions like the pipeline does.
- Model for gaincal: true point-source model (known flux) vs the pipeline's own setjy-style model.
- Experiment B subset size and which cases (e.g. strongest per corruption type, survivors from A).
- Which saved model run to use for predictions, and whether its feature view needs the CHPC artifacts.
- Whether "after" should also be compared with the uncorrupted control (eC of control is not exactly 0).
- Output layout: new run folder per experiment that reads the source run read-only.
