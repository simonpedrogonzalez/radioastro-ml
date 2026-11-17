# Add the 62 new sources without changing existing v2

Simulate only the new sources, then combine the existing and extension runs through manifests. Keep the current v2 runs, their parent data, the old thermal directory and the v1 dataset unchanged. Do not rerun generation or `--verify` inside old run folders: matched verification rewrites files.

Follow the steps in order. Handoffs are short console/chat messages with the IDs or paths the next person needs. No Git tags, mandatory commits, migration journals, separate audit reports or new audit framework.

**Coordination:** The CHPC runner pauses affected download/extraction jobs before the data move. DATA MANAGER owns physical files and stored data paths; CODE DEV owns scripts, partitions and configs; CHPC DEV / RUNNER owns execution and runtime settings. Once simulations start, leave their input paths and code alone until the jobs finish.

## Step 1 — Move the sources and tidy the fetcher data ([DATA MANAGER])

- Identify the completed new sources and check for existing IDs/destination folders. If there is no conflict, move and rename them from `reports/manifests/local/runs/small_remaining/extracted/<cal>_<eb>/` into `data/extracted/<cal>/<cal>/<cal>.ms`. Use real MS directories, not `.ms` symlinks. Skip incomplete/empty entries.
- Keep `extraction_metadata.json`, `flux_estimate.json`, `listobs.txt` and `extraction_result.json` beside each MS. Make sure the moved sources are visible through the existing `collect/extracted` layout.
- Move the remaining `reports/manifests/local` folder to `data/nrao-archive-fetcher`. Update stored active paths and retain `extracted` status. Reconcile any completed shard progress first, then move obsolete shard manifests out of the active directory so they cannot restore the old paths.
- Once the extracted files are in place and stored paths are correct, remove disposable raw products in `small_remaining/downloads/`. Keep anything still needed by unfinished extractions. Leave existing simulation data alone.

**Handoff → CODE DEV:** “Files moved,” the actual new source IDs, the extracted/fetcher locations, and any skipped sources. No migration report needed.

## Step 2 — Make the required code changes ([CODE DEV])

- Update fetcher defaults/scripts to use the relocated folder. The manifest path controls download/extraction locations; also account for the changed incoming-request path. The runner handles deployment-specific environment settings.
- Reuse `collect/small_subset/small_selection.csv`: the candidate sources already have rows and compiled catalog metadata. Do not append duplicates or recollect metadata. Correct only observation-specific values shown to be stale; blank SPW/frequency fields do not by themselves need filling.
- Add the new IDs to `scripts/preprocessing/partitions.py` with a deterministic approximately 70/15/15 split, keeping every existing ID in its current split and the lists sorted. Update the partition tests accordingly.
- Add optional `thermal_dir` support to both v2 configuration loaders, **without adding it to `DEFAULT`**. When supplied, select sources from that thermal report's successful samples instead of requiring a new v1 dataset. Without it, preserve existing behavior and config hashes.
- Use the selected thermal directory throughout preparation, execution, original-flux QA lookup (`original_flux_spec`) and preview helpers. Keep existing split/parent-duplicate checks and `DefaultImagingConfig` unchanged.
- Provide a small CASA launcher that accepts an explicit new-source list and output directory for `compare_extracted_thermal_simulations.main(...)`.
- Run the relevant tests: old config hashes and source selection stay unchanged, the new thermal directory works through the relevant code paths, and partitions remain disjoint. No separate baseline archive or audit tooling is needed.

**Handoff → CHPC DEV / RUNNER:** “Code ready,” how to run the thermal launcher, and the new split assignments.

## Step 3 — Run thermal simulations for the new sources ([CHPC DEV / RUNNER])

- Use the updated code. Point the fetcher runtime at its relocated manifest/data root and set `INCOMING` explicitly to the request-manifest location. Check those stored/configured paths; no fetcher rerun is needed just to prove the move worked.
- Run the thermal launcher under CASA/Slurm with only the new IDs and a fresh directory such as `collect/experiments/extracted_thermal_simulation_comparison_ext62_<date>`. Use settings consistent with the old thermal run.
- Check the generated `report.json` for successful sources and usable simulation/QA products. Exclude failed sources from the extension; do not replace them with old sources or reshuffle their split assignments.

**Handoff → CODE DEV:** Thermal directory, successful IDs and any failed IDs. The existing generated `report.json` is enough; no extra report.

## Step 4 — Create the extension configs ([CODE DEV])

- Create `config/dataset_v2_ext62_{train,val,test}.json` for the nonempty splits, with four overrides: `split`, explicit successful new `source_ids`, `thermal_dir`, and a fixed new `seed` different from the corresponding old run. The joiner rejects equal seeds even for different sources.
- Keep ranges, recipes and weak fraction consistent with old v2. Do not reuse the existing v2.1 recipe-extension config, which changes those settings.
- Confirm each config selects only its intended new sources. Skip empty splits.

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
- Train using the expanded manifests and save models to new output folders. Compare old and new models on the same evaluation set using R² and AUC where applicable, not raw multitask `J`. Keep new test sources out of tuning.

**Done:** Tell the user where the expanded manifests and models are, how many new sources were included, and which sources failed. No extra handoff documents or completion reports.
