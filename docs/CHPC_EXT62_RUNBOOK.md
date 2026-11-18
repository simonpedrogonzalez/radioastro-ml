# Add the 62 new sources without changing existing v2

Simulate only the new sources, then combine the existing and extension runs through manifests. Keep the current v2 runs, their parent data, the old thermal directory and the v1 dataset unchanged. Do not rerun generation or `--verify` inside old run folders: matched verification rewrites files.

Follow the steps in order. Handoffs are short console/chat messages with the IDs or paths the next person needs. No Git tags, mandatory commits, migration journals, separate audit reports or new audit framework.

**Coordination:** The CHPC runner pauses affected download/extraction jobs before the data move. DATA MANAGER owns physical files and stored data paths; CODE DEV owns scripts, partitions and configs; CHPC DEV / RUNNER owns execution and runtime settings. Once simulations start, leave their input paths and code alone until the jobs finish.

**Updated extension policy:** Use a valid NRAO pipeline flux first; otherwise use the original-observation CLEAN flux from a thermal run. If neither is usable, skip the source. This is not flux measured from the simulated thermal image. To support fully flagged inputs, clear inherited flags on simulation copies only. These are new extension settings, not changes to old v2 defaults. Deploy the updated code in both repositories before using them; older launchers do not support these options.

## Step 1 — Move the sources and tidy the fetcher data ([DATA MANAGER])

- Identify the completed new sources and check for existing IDs/destination folders. If there is no conflict, move and rename them from `reports/manifests/local/runs/small_remaining/extracted/<cal>_<eb>/` into `data/extracted/<cal>/<cal>/<cal>.ms`. Use real MS directories, not `.ms` symlinks. Skip incomplete/empty entries.
- Keep `extraction_metadata.json`, `flux_estimate.json`, `listobs.txt` and `extraction_result.json` beside each MS. Make sure the moved sources are visible through the existing `collect/extracted` layout.
- Move the remaining `reports/manifests/local` folder to `data/nrao-archive-fetcher`. Update stored active paths and retain `extracted` status. Reconcile any completed shard progress first, then move obsolete shard manifests out of the active directory so they cannot restore the old paths.
- Before removing raw downloads, let CODE DEV check that the saved flux sidecars contain the needed spectral model and provenance. Retain small pipeline logs needed to recover missing information; do not retain large raw products just for their logs. Then remove disposable products in `small_remaining/downloads/`, keeping anything needed by unfinished extractions. Leave existing simulation data alone.

**Handoff → CODE DEV:** “Files moved,” the actual new source IDs, the extracted/fetcher locations, and any skipped sources. No migration report needed.

## Step 2 — Make the required code changes ([CODE DEV])

- Update fetcher defaults/scripts to use the relocated folder. The manifest path controls download/extraction locations; also account for the changed incoming-request path. The runner handles deployment-specific environment settings.
- Reuse `collect/small_subset/small_selection.csv`: the candidate sources already have rows and compiled catalog metadata. Do not append duplicates or recollect metadata. Correct only observation-specific values shown to be stale; blank SPW/frequency fields do not by themselves need filling.
- Add the new IDs to `scripts/preprocessing/partitions.py` with a deterministic approximately 70/15/15 split, keeping every existing ID in its current split and the lists sorted. Update the partition tests accordingly.
- Support optional `thermal_dir`, `flux_policy: "nrao_then_clean"` and `simulation_flags: "unflag"` in both v2 configuration loaders and equivalent options in the thermal extension launcher. **Do not add them to `DEFAULT`**: absent options must preserve old behavior and hashes. Select extension sources from the selected thermal report's successful samples; keep split/parent-duplicate checks and `DefaultImagingConfig` unchanged.
- Fix the fetcher's flux extraction before relying on it: match the calibrator and applicable frequency/band, preserve all spectral-index coefficients, and reject ambiguous or invalid estimates. Existing sidecars may be recoverable from their saved `source_text`; otherwise use retained logs. Evaluate the pipeline model at the selected simulation frequency. A sidecar's existence or an `extracted` status alone does not establish a usable flux.
- Implement one flux decision in thermal preparation: valid NRAO estimate first; otherwise reuse valid original-observation CLEAN QA for this input, or image the original MS with its original flags to obtain it. If both routes fail, record the failure and skip. Freeze the chosen flux, method, frequency and provenance in the existing per-source metadata. Do not infer it from the synthetic image or require original DR.
- Allow the NRAO route to bypass original-data imaging entirely. Update thermal success checks and reports to accept that route without original-image QA. Make `original_flux_spec`, v2 preparation/execution and both previews consume the same recorded choice; remove their unconditional original-CLEAN QA/peak requirements only for this opt-in route. Reuse earlier thermal QA read-only where applicable, checking that it belongs to the same input and imaging settings.
- Implement `simulation_flags: "unflag"` consistently: clear `FLAG` and `FLAG_ROW` on simulation copies before prediction, leaving extracted originals untouched. Validate usable sampling geometry, derive image RMS from the resulting sample mask, and recompute SNR, antenna eligibility, gain norms and imaging references from those copies. Every regenerated v2 copy must use the same policy, with weights initialized consistently. This changes sampling relative to old v2 and must be recorded in existing metadata.
- Include the band-edge fix in `scripts/simulation/noise.py`: require a strict majority of selected channel centers in the requested nominal band, replacing the two-channel/two-width allowance. Do not clip channels, change the selected band or change SEFD/noise scaling to bypass the check. [NRAO permits tuning outside nominal edges](https://science.nrao.edu/facilities/vla/docs/manuals/oss2026a/performance/vla-frequency-bands-and-tunability); the existing fiducial-SEFD model remains an idealized thermal approximation, not a model of receiver-edge sensitivity.
- Test the NRAO-first route, CLEAN fallback, failure of both, spectral-frequency evaluation, absence of original QA on the NRAO route, and consistent flag handling. Retain old-config/hash and partition checks. Give the runner a small CASA smoke command before the full run; no separate audit tooling or report is needed.

**Handoff → CHPC DEV / RUNNER:** “Code ready,” the smoke/thermal commands with the new policies enabled, and the new split assignments. Tell DATA MANAGER which small logs, if any, still need retaining.

## Step 3 — Run thermal simulations for the new sources ([CHPC DEV / RUNNER])

- Use the updated code. Point the fetcher runtime at its relocated manifest/data root and set `INCOMING` explicitly to the request-manifest location. Check those stored/configured paths; no fetcher rerun is needed just to prove the move worked.
- Before launching, DATA MANAGER must resolve or exclude mismatched source identities: the local fetcher manifest maps both `1051+213` and `1407+284` to `J1125+2610`. Check the actual extracted fields on CHPC; renaming an MS does not change the observed source. Use the corrected source list for the run and partitions.
- Upgrade existing flux sidecars with the fetcher's `python -m workflows.radioastro_calibrators.flux_weblog recheck --dry-run [--downloads-root <retained-downloads>] <source-folder> ...`, then repeat without `--dry-run`. Each source folder holds `extraction_metadata.json` and `flux_estimate.json`; operate only on new extension sources. Review the printed available/unavailable/ambiguous counts; missing NRAO flux goes to CLEAN fallback, not an automatic exclusion.
- Finish or stop any thermal job using the earlier code before deploying changes. Keep its outputs for possible CLEAN fallback; do not resume that directory with different flux/flag policies.
- First smoke-test prior band-edge failures (include S and X examples), a fully flagged source with available NRAO flux, and CLEAN fallback where available. Confirm the chosen flux route, successful synthetic imaging and unchanged original MS flags. A fully flagged source without a usable external flux may still fail the fallback. Commit `07e6f04` alone does not contain the band-edge fix.
- Run the updated launcher under CASA/Slurm with only the new IDs, the new flux/flag policies and a fresh output directory. Use the established imaging settings. The code may reuse valid earlier original-CLEAN QA without rerunning that imaging.
- Check `report.json` for successful sources and usable synthetic products. Original-image QA is required only for the CLEAN fallback. Exclude sources that still fail after the new policy; do not automatically exclude sources merely because the earlier job failed on original flags. Do not reshuffle splits.
- Judge completion by `report.json`, not Slurm's `COMPLETED`: individual failures are caught and recorded. Missing Quarto affects HTML only. Load Quarto in the job if available, or render afterward with `bin/run ml quarto render <new-thermal-directory>/report.qmd`; do not rerun simulations solely to produce HTML.

**Replacing the earlier extension run:** Delete nothing initially. From the radioastro-ml checkout, use the normal CASA/Slurm wrapper with:

```bash
bin/run casa scripts/run_thermal_extension.py \
  --output <NEW-thermal-directory> \
  --ids-file <corrected-new-source-list> \
  --flux-policy nrao_then_clean \
  --simulation-flags unflag \
  --reuse-qa-from <PREVIOUS-extension-thermal-directory>
```

Use all intended new sources, including earlier successes: both flux and sampling policy have changed. Reused CLEAN QA is referenced in its original location, so keep the previous directory in place. After the replacement succeeds, DATA MANAGER may remove obsolete simulation products from the earlier **extension** only after checking that no dataset references them; retain reused original QA. Never delete the legacy thermal directory used by frozen v2.

The runner reported job `2506636` finished with 20 successes, 11 fully flagged failures and 31 band-check failures. Its previous directory is `$RADIOASTRO_SCRATCH/outputs/collect-experiments/extracted_thermal_simulation_comparison_ext62_20261117/`. Use that for `--reuse-qa-from`: sources that failed later at simulation may already have reusable original CLEAN QA, even though they are absent from the successful-samples list.

**Handoff → CODE DEV:** Thermal directory, successful/failed IDs and counts using NRAO flux versus CLEAN fallback. The existing generated `report.json` and a console message are enough.

## Step 4 — Create the extension configs ([CODE DEV])

- Create `config/dataset_v2_ext62_{train,val,test}.json` for nonempty splits with `split`, explicit successful new `source_ids`, `thermal_dir`, a fixed new `seed`, `flux_policy: "nrao_then_clean"` and `simulation_flags: "unflag"`. The policies must match thermal preparation. The seed must differ from the corresponding old run because the joiner rejects equal seeds even for different sources.
- Keep ranges, recipes and weak fraction consistent with old v2. Do not reuse the existing v2.1 recipe-extension config, which changes those settings.
- Confirm each config selects only its intended new sources and uses their recorded flux decisions. Skip empty splits. Reject policy/provenance mismatches instead of silently switching flux during resume.

**Handoff → CHPC DEV / RUNNER:** The config paths, ready to run.

## Step 5 — Generate and verify the extension runs ([CHPC DEV / RUNNER])

- Use new outputs: `runs/dataset-v2-matched-<split>-ext62`. Run `prepare → plan → execute → verify` using the supplied configs. Matched train does **not** require a preview.
- Use separate stage submissions so you can check that the plan contains only the intended new sources before execution. Choose resources, shards, caches and thread limits using the existing CHPC setup. Do not run duplicate writers for the same shard.
- Verify only the new runs. Check that all planned units/rows completed and the split NPZ exists: the current verifier can exit successfully while reporting missing units.
- Check that the extension's feature names/order match the old part before combining. If they differ, fix compatibility separately without rewriting the old run.

**Handoff:** “Extension complete,” run paths and any unresolved failures. Continue only with complete, usable extension parts.

## Step 6 — Combine and train ([CHPC DEV / RUNNER])

- For each split, use `scripts.dataset_v21 build` with the old v2 run and its new extension as parts, no filters, writing `runs/dataset-v2-expanded-<split>/manifest.json`. This joins references; it does not copy or resimulate the old data.
- Check combined row counts and that no source occurs in both parts or across splits. The joiner permits shared sources, so it does not enforce this for you. Keep both referenced run folders in place.
- Train using the expanded manifests and save models to new output folders. Compare old and new models on the same evaluation set using R² and AUC where applicable, not raw multitask `J`. Keep new test sources out of tuning. Account for the extension's different flux/sampling policy when interpreting comparisons; matching recipe ranges does not make those policies identical to old v2.

**Done:** Tell the user where the expanded manifests and models are, how many new sources were included, and which sources failed. No extra handoff documents or completion reports.
