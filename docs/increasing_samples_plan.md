# Increasing the Calibrator Sample Set

## Goal

Move the archive request, download, and calibrator extraction workflow into
`nrao-archive-fetcher`, then build and process four non-overlapping manifests in
this order:

1. remaining entries from the existing small selection;
2. new matches for the 181 previously missed calibrators, with no date limit;
3. the original qualifying products rejected only by the 15 GB size cut;
4. up to three additional observations per calibrator, with no date limit.

Every manifest is ordered by estimated download size, smallest first. Each
manifest has its own download and extraction directories. The implementation
should reuse the existing fetcher query, manifest, email, and download code and
the existing `radioastro-ml` extraction code. Add only the project-specific
code needed to connect those pieces.

This plan changes dataset membership, not the scientific definition of the
calibrator extraction. Except for retaining the weblog flux estimate, the
existing field and SPW extraction behavior remains the reference.

## Repositories and source material

Implementation belongs in:

```text
/Users/u1528314/Documents/nrao-archive-fetcher
```

All implementation changes, workflow scripts, tests, generated inventories,
manifests, and run configuration must be created in `nrao-archive-fetcher`.
Do not modify `radioastro-ml` while carrying out this plan. It is read-only
reference material, including its scripts, CSV files, extracted products, and
continuity files. If the fetcher needs historical metadata, read it from an
explicit source path or copy the required immutable snapshot into the fetcher
with provenance; do not rewrite or augment the source repository.

Reference inputs remain in `radioastro-ml`:

```text
collect/small_subset/small_selection.csv
collect/vla_calibrators_selected.csv
collect/vla_calibrators_one_hit.csv
collect/vla_calibrators_one_hit_missed.csv
scripts/extraction_pipeline.py
scripts/inspect_visibility_metadata.py
```

Do not edit the historical CSV files while building the new manifests. Treat
them as source snapshots.

## Git checkpoints

Work only in the `nrao-archive-fetcher` worktree. After a complete change that
accomplishes one of the goals or phases in this plan:

1. run the focused tests and checks for that change;
2. review the fetcher diff and ensure unrelated files are not included;
3. create a local Git commit in `nrao-archive-fetcher` describing the completed
   goal;
4. do not push the commit.

Do not make partial or checkpoint commits for unfinished pieces merely to save
progress. A commit marks a coherent completed goal, such as Linux email input,
calibrator extraction with flux provenance, inventory generation, one manifest
builder stage, or the manifest download/extraction runner.

## Identities and deduplication

Use these names consistently:

- `obs_id`: the full NRAO observation/product identifier from the
  `productViewer` URL. This is the global deduplication key.
- `project_code`: the proposal code such as `17B-078`. Preserve it as metadata,
  but do not use it as the deduplication key or output-folder name because one
  proposal can contain several selected observations.
- `calibrator_id`: the short catalog identifier used by this project, such as
  `0012-399`.
- `gain_calibrator_name`: the actual MS field name, such as `J0012-3954`.
- `entry_id`: a filesystem-safe identifier derived from `calibrator_id` and
  `obs_id`; it names download and extraction directories.

The phrase "same project ID" in the expansion rules means the same `obs_id`.
Using `project_code` would incorrectly collapse distinct execution blocks from
the same proposal.

Every manifest entry must contain at least:

```text
entry_id
obs_id
project_code
calibrator_id
gain_calibrator_name
viewer_url
estimated_size_gb
band_guess
usable_configs
request_status
download_command
download_root
extracted_ms
selected_spw
flux_estimate
```

Extra keys should remain ordinary manifest fields. Do not introduce a large
schema or validation framework.

## Phase 1: Linux email import

Extend the existing fetcher email module instead of creating a second matching
pipeline.

Add an IMAP gathering mode using the standard-library `imaplib` and `email`
modules. Configuration comes from CLI arguments or environment variables:

```text
host
port
username
password/app password
mailbox
sender filter
days back
```

Credentials must never be written to a manifest, log, fixture, or repository.
For Gmail, use an app password when the account permits it. OAuth support is
outside the first implementation.

Also add a `messages-json` input mode. It is the provider-independent fallback
and makes email matching testable on Linux without live account access. Both
IMAP and JSON gathering must produce the existing `MailMessage` objects and
then call the current command extraction and manifest matching functions.

Matching order:

1. exact `request_name`/`entry_id` from the email subject or body;
2. exact `obs_id`;
3. unique token overlap using the existing matcher.

Do not silently choose between tied matches. A tie is one of the few cases that
must stop that email from being imported.

Required CLI shape:

```bash
nrao-fetch import-email manifest.json --backend imap ...
nrao-fetch import-email manifest.json --messages-json messages.json
```

Keep the existing macOS Mail backend working.

## Phase 2: Calibrator and weblog-flux extraction

### Extraction behavior to preserve

Use `radioastro-ml/scripts/extraction_pipeline.py` as the behavioral reference:

1. find the downloaded MeasurementSet;
2. require `CORRECTED_DATA`;
3. identify the requested `gain_calibrator_name` field;
4. choose one SPW in `band_guess`;
5. rank matching SPWs by bandwidth and then unflagged fraction;
6. require at least 50% unflagged data and at least 10 MHz bandwidth;
7. run CASA `split` with `datacolumn="corrected"` and `keepflags=True`;
8. write `listobs.txt` and extraction metadata;
9. retain the full download unless deletion is explicitly requested.

Imaging is not part of this expansion step. Do not carry the old clean/dirty
image creation into the fetcher extraction workflow.

### Flux estimate to retain

During extraction, inspect the downloaded pipeline products for the VLA
Pipeline `hifv_fluxboot` result corresponding to `gain_calibrator_name`.
Prefer the stage-specific weblog/log material, then the top-level
`casa_commands.log`. Do not rely exclusively on `flux.csv`; the retained
`17B-078` example has an empty `flux.csv` but contains the quantitative result
in the weblog logs and `setjy(standard="manual", ...)` command.

Write a small `flux_estimate.json` beside the extracted MS containing:

```text
gain_calibrator_name
band
stokes_i_jy
reference_frequency_ghz
spectral_index
fit_snr
source_stage
source_file
source_text
```

The primary value is the `hifv_fluxboot` fitted model: Stokes-I flux density at
its reference frequency plus spectral index. Record S/N when the weblog
provides it. Preserve the exact source line or command for provenance. If no
usable estimate exists, write an unavailable status and continue the MS
extraction; do not invent a value and do not fail an otherwise usable sample.

Use `scripts/inspect_visibility_metadata.py` only as a reference for locating
fluxboot evidence. Implement one focused parser rather than porting the whole
inspection script.

### Placement and execution

Keep CASA-specific, calibrator-specific code in a small workflow area in the
fetcher repository, for example:

```text
workflows/radioastro_calibrators/
  extract_entry.py
  flux_weblog.py
  run_manifest.py
```

The generic downloader should remain CASA-independent. `run_manifest.py`
should use the existing `download_manifest` API and invoke the CASA extraction
entrypoint with the configured CASA executable. Ensure successful extraction
fields are written back to the manifest after the hook/process finishes.

## Phase 3: Snapshot the already-downloaded inventory

Add a small inventory-generation script in the fetcher workflow. Read the
historical `small_selection.csv`, and reconcile it with the actual extracted
MS directories. An entry counts as already downloaded for deduplication when a
corresponding extracted MS exists; use the CSV to recover its archive and
calibrator metadata.

Generate a committed Python data module containing immutable collections:

```python
ALREADY_DOWNLOADED_OBS_IDS = frozenset({...})
ALREADY_DOWNLOADED_CALIBRATOR_IDS = frozenset({...})
ALREADY_DOWNLOADED_GAIN_FIELDS = frozenset({...})
```

Also retain an `obs_id -> metadata` mapping containing `project_code`,
`calibrator_id`, `gain_calibrator_name`, and the historical extracted path.
The generator should make the provenance explicit and produce deterministic
sorted output. Do not hand-maintain the lists.

Ignore the invalid blank historical row whose name is `z`.

## Phase 4: Build manifests in precedence order

Build manifests sequentially. After each manifest is created, reserve all of
its `obs_id` values before building the next manifest, even if those entries
have not been downloaded yet.

The common exclusion set is:

```text
already-downloaded obs_ids
+ obs_ids in every earlier manifest
```

All output manifests must use a stable ascending sort by
`estimated_size_gb`, with `obs_id` as the tie-breaker. Entries lacking a usable
size belong at the end.

### 4.1 `small_remaining.json`

Source: `small_subset/small_selection.csv`.

- discard the invalid `z` row;
- exclude `ALREADY_DOWNLOADED_OBS_IDS`;
- retain the historical archive product and calibrator metadata;
- include prior status/error only as provenance;
- start request/download state clean unless a local download is verified;
- sort by size ascending.

Old emailed download URLs may have expired. Do not treat the presence of an
old `wget_command` as proof that it is reusable.

### 4.2 `missing_no_date.json`

Source: the 181 rows in `vla_calibrators_one_hit_missed.csv`.

For each calibrator, repeat the original archive search with only the date
constraint removed. Preserve the remaining criteria:

- 5 arcsec position radius;
- VLA/EVLA;
- visibility products;
- public data;
- original band;
- original usable configurations;
- candidate products ordered by size;
- product has calibration tables;
- product exposes a probable phase/gain calibrator using the existing scan
  intent rule.

Select the smallest qualifying non-reserved observation for each calibrator.
Do not permanently classify a transient TAP/details failure as a scientific
miss; record it separately so the build can be resumed.

### 4.3 `large_original.json`

Source: `vla_calibrators_one_hit.csv`.

- retain rows with `size_gb >= 15` GB;
- exclude all reserved `obs_id` values;
- preserve the original selected product and calibrator metadata;
- sort by size ascending.

These are the original qualifying observations rejected only by the small
selection's size cut. Do not apply a new upper-size cutoff.

### 4.4 `repeated_up_to_3.json`

Start with calibrators already represented by the downloaded inventory or one
of the three earlier manifests. Query with the same band, configuration,
position, public-visibility, calibration-table, and gain-calibrator criteria,
but with no date constraint.

For each calibrator:

```text
prior_count = appearances in inventory and earlier manifests
extra_slots = min(3, max(0, 4 - prior_count))
```

- skip calibrators with `prior_count == 0`;
- exclude every globally reserved `obs_id`;
- select at most `extra_slots` additional observations;
- never allow more than four total appearances for a calibrator;
- order candidates by size before applying the per-calibrator quota;
- globally sort the completed manifest by size.

Use the full `obs_id` for uniqueness. Distinct observations from one proposal
are allowed.

## Phase 5: Download and extract by manifest

Use one root folder per manifest:

```text
runs/small_remaining/
  manifest.json
  downloads/<entry_id>/
  extracted/<entry_id>/<entry_id>.ms
  extracted/<entry_id>/listobs.txt
  extracted/<entry_id>/flux_estimate.json

runs/missing_no_date/
runs/large_original/
runs/repeated_up_to_3/
```

Provide one thin runner that accepts a manifest and output root rather than
four copied scripts:

```bash
python workflows/radioastro_calibrators/run_manifest.py \
  runs/small_remaining/manifest.json
```

The runner should:

1. call the existing fetcher downloader;
2. resume entries whose download or extraction is incomplete;
3. validate success by locating the expected MS, not solely by the `wget`
   return code;
4. invoke the CASA extraction entrypoint;
5. parse and save the weblog flux estimate;
6. update the manifest after every entry;
7. leave failed entries resumable;
8. keep the original download by default.

The old workflow often obtained usable data even when recursive `wget`
returned code 8. A present, readable MS should therefore be checked before
marking the download as failed.

## Minimal implementation structure

Prefer these small changes over a new framework:

1. Extend the existing email gathering switch with IMAP and JSON modes.
2. Let download folder naming use `entry_id`, falling back to the existing
   `project_code` behavior for generic manifests.
3. Ensure post-download hook changes or extraction results are persisted.
4. Add one project workflow package containing inventory generation, manifest
   construction, weblog flux parsing, CASA extraction, and the runner.
5. Reuse `NRAOQuery`, detail fetching, manifest serialization,
   `download_manifest`, and existing email matching.

Avoid additional configuration classes, plugin registries, generalized
validators, wrapper layers, or compatibility paths unless an immediate step
above requires them.

## Validation

Use only checks that protect the requested behavior:

- IMAP and JSON messages produce the same `MailMessage` representation and
  update the intended manifest entry.
- Ambiguous email matches are not applied.
- Inventory generation agrees with the historical CSV and extracted folders.
- No `obs_id` appears in more than one inventory/manifest collection.
- Every manifest is sorted by size.
- Repeated entries obey the three-extra/four-total rule.
- A fixture reproduces parsing of the retained `17B-078` fluxboot result,
  including flux density, reference frequency, and spectral index.
- A mocked CASA test verifies the selected field/SPW and manifest update.
- One real downloaded sample completes download discovery, CASA split,
  `listobs`, and flux-estimate extraction.

Do not add broad runtime type validation or checks for states that these
workflows cannot produce.

## Definition of done

- Linux can import NRAO completion emails through IMAP or exported JSON.
- The already-downloaded observation and calibrator inventories are generated
  and committed in the fetcher repository.
- All four manifests are generated in the required order, globally disjoint,
  and sorted smallest first.
- Repeated observations satisfy the per-calibrator quota.
- One command can resume download and extraction for any manifest into its own
  folder.
- Every successful entry has a compact calibrator MS, `listobs.txt`, structured
  extraction metadata, and either a provenance-backed flux estimate or an
  explicit unavailable result.
- Existing generic fetcher behavior and macOS Mail support remain intact.
