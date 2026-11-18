# Calibrator field mismatches in extracted data

Checked 2026-11-18. Some folders under `data/extracted/<id>/` contain a different calibrator from
the source `<id>` named in the catalog (`collect/small_subset/small_selection.csv`). The name is
wrong, not the physics: each MS holds real, calibrated data for whatever field was extracted.

## Cause

The fetcher extracts the field named by the manifest's `gain_calibrator_name`. For the affected
entries that name pointed to another phase calibrator in the same observation:

- some names were inherited from historical inputs without checking them against the requested
  source's position;
- the manifest builder used a heuristic that picked the observation's most-observed phase
  calibrator, whether or not it was the requested source.

Nothing downstream detects this. Thermal simulation, v2 generation and training model whatever is at
the MS phase centre, so a mislabeled source "succeeds" under the wrong name.

Consequences:

- **Wrong identity and flux provenance.** Catalog metadata or an NRAO flux looked up for `<id>` can be
  attached to a different object.
- **Duplicate objects.** The real calibrator may already be in the dataset under its own ID, observed in
  a different execution block.
- **Split leakage.** If the two copies land in different splits, train and test are no longer
  independent.

## How it was checked

The field name and J2000 position in each extracted MS's `listobs.txt` were compared with the catalog
`ra`/`dec` of the folder's ID. Correct fields agree within 0.02 arcsec. Every mismatch is at least
5 degrees away. Names were used only to identify which object a field actually is.

## Code status

- **nrao-archive-fetcher `1fa1fb3`: fixed** (`field_match.py`, `build_manifests.py`,
  `extract_entry.py`; fetcher tests pass).
  - The manifest builder keeps an inherited name only if it matches the catalog position. Otherwise
    it resolves the phase-calibrator field at that position from the archive's scan intents, or
    rejects the entry.
  - New archive candidates must have a phase-calibrator scan within 5 arcsec of the requested
    position. The most-observed-calibrator heuristic is gone.
  - Extraction reads the field's `PHASE_DIR` from the MS and refuses to split it unless it is within
    5 arcsec of the catalog position.
- **radioastro-ml: no identity check of its own.** It trusts the folder name and relies on the
  fetcher's extraction check. Data extracted before `1fa1fb3` was never checked by the code; this
  document is that check.

## Extension sources (ext62): 8 of 62 mismatched

The other 54 match. Archive status says whether the intended source is a phase calibrator in the
same execution block (EB), which the fixed fetcher requires.

| ID | Extracted field (wrong) | Separation | Real object already in dataset as | Intended field in the same EB | Action |
|---|---|---|---|---|---|
| 0437-187 | J0149+0555 | 48.1 deg | 0149+059 (train) | J0437-1844, phase cal | re-extracted |
| 1753+288 | J1740+5211 | 23.5 deg | 1740+521 (ext62) | J1753+2848, phase cal | re-extracted |
| 1833-210 | J1820-2528 | 5.3 deg | none | J1833-2103, phase cal | re-extracted |
| 2035+189 | J2049+1003 | 9.5 deg | none | J2035+1857, phase cal | re-extracted |
| 2206-185 | J1923-2104 | 38.2 deg | 1923-210 (train) | J2206-1835, phase cal | re-extracted |
| 1051+213 | J1125+2610 | 9.2 deg | 1125+261 (train) | J1051+2119 present, **not** a phase cal | dropped |
| 1407+284 | J1125+2610 | 35.7 deg | 1125+261 (train) | J1407+2827 present, **not** a phase cal | dropped |
| 0713+438 | J0958+3224 | 33.9 deg | none | J0713+4349 present, **not** a phase cal; that object is already train `0449+113` (see below) | dropped |

- 1051+213 and 1407+284 come from the same EB (18B-316 eb37535740), which was downloaded twice.
- The dropped sources would need a different observation, meaning a new archive search and download.
- The earlier thermal run `extracted_thermal_simulation_comparison_ext62_20261117` (job 2506636) used
  the wrong data for these 8. Three of its 20 "successes" (1051+213, 1407+284, 2206-185) were mislabeled.

### Re-extraction (done 2026-11-18)

- **Job.** Granite job 2519310, 54 s, all 5 exit 0. It ran fetcher `extract_entry.py` at `1fa1fb3`
  through `bin/run casa`, from the retained downloads, so nothing was re-downloaded. Scripts, corrected
  entry JSONs and logs are in `$RADIOASTRO_SCRATCH/ops/ext62-reextract-20261118/`.
- **Installed.** Each corrected MS and its sidecars are at `data/extracted/<id>/<id>/`, with sidecar
  paths rewritten.
- **Set aside, not deleted.** All 8 wrong extractions were moved to
  `$RADIOASTRO_SCRATCH/data/extracted_misidentified_20261118/<id>/`.
- **Fetcher manifest** (`data/nrao-archive-fetcher/runs/small_remaining/manifest.json`; backup
  `ops/ext62-reextract-20261118/manifest.before.json`):
  - The 5 fixed entries have the correct `gain_calibrator_name`, plus `replaced_gain_calibrator_name`
    and `field_check: archive_position_match`.
  - The 3 dropped entries are `request_status: extraction_error` (final; the fetcher will not
    re-download them), with the reason in `last_error`.
- **Corrected ID list.** `data/nrao-archive-fetcher/ext62_corrected_source_ids.txt` lists the 59
  remaining sources. The original 62-ID list is kept. The position check passes for all 59.
- **Downloads.** Keep the 8 raw downloads until the replacement thermal run has succeeded.

| ID | Field | SPW start | Rows | On-source | NRAO flux sidecar |
|---|---|---|---|---|---|
| 0437-187 | J0437-1844 | 7.976 GHz | 13325 | 438 s | available |
| 1753+288 | J1753+2848 | 1.988 GHz | 31200 | 508 s | available |
| 1833-210 | J1833-2103 | 1.000 GHz | 15275 | 760 s | available |
| 2035+189 | J2035+1857 | 1.988 GHz | 16575 | 366 s | ambiguous (2 models cover the SPW), so CLEAN fallback |
| 2206-185 | J2206-1835 | 1.008 GHz | 17700 | 295 s | available |

Flagging was not inspected. The replacement thermal run uses `--simulation-flags unflag`.

## Existing v2 sources: 9 mismatched (not changed)

Frozen v2 is left as is. All 9 are in train or test, and none of the duplicates crosses splits, so
there is no train/test leakage. v2 measured flux from the extracted data itself, so the physics is
consistent. The labels are wrong, and 7 train objects appear twice.

| ID | Split | Extracted field (actual object) | Separation | Same object also in dataset as |
|---|---|---|---|---|
| 0242-215 | train | J0609-1542 | 49.2 deg | 0609-157 (train) |
| 0416-209 | train | J0240-2309 | 22.3 deg | 0240-231 (train) |
| 0449+113 | train | J0713+4349 | 45.0 deg | 0713+438 (ext62 ID, dropped) |
| 0650-166 | train | J0653-0625 | 10.2 deg | 0653-064 (train) |
| 1305-105 | train | J1246-0730 | 5.7 deg | 1246-075 (train) |
| 1352-442 | train | J1432-1801 | 27.5 deg | 1432-180 (train) |
| 1411+522 | train | J1119-0302 | 66.1 deg | 1119-030 (train) |
| 1439-169 | train | J1150-0023 | 44.7 deg | none |
| 1520+202 | test | J1206+6413 | 54.4 deg | none |

Remember these when combining old and new parts (runbook Step 6) and when reading per-source
results. `scripts.dataset_v21 build` does not detect duplicates.

## Not checked: 12 existing sources without `listobs.txt`

These have an extracted MS (`data/extracted/<id>/<id>/<id>.ms`) but no `listobs.txt`, so the
text-based check could not run. Checking them requires reading each MS's `FIELD` table (`NAME`,
`PHASE_DIR`) in CASA.

| Split | IDs |
|---|---|
| train | 0129+236, 0153-331, 0616-349, 0958+474, 1033+412, 1130+382, 1147-382, 1923-210 |
| val | 0022+002, 0725-009 |
| test | 1146+399, 1310+323 |

`data/extracted/` also contains two stray non-source files, `322.html` and `tanmoy_has_seen.txt`. They
were ignored.
