# Incremental Quarto Experiment Reporting

## Goal

Add a simple reporting system that can update a Quarto HTML report while a long-running CASA experiment is still processing samples.

The reporting mechanism should:

* work alongside a sequential sample-processing loop;
* not block processing while Quarto renders;
* render only every N completed samples rather than after every sample;
* always produce one final render when processing finishes;
* work with different `.qmd` report files without knowing their internal layout;
* print the location of the updated HTML report;
* remain small and easy to understand.

Do not build a generic dashboard/reporting framework.

---

## 1. Generic asynchronous Quarto renderer

Create:

```text
scripts/reporting/
├── __init__.py
└── quarto.py
```

Expose:

```python
class QuartoReporter:
    def __init__(
        self,
        quarto_file: str | Path,
        every: int = 5,
    ):
        ...

    def sample_completed(self) -> None:
        ...

    def finish(self) -> None:
        ...
```

Example use:

```python
reporter = QuartoReporter(
    experiment_dir / "report.qmd",
    every=5,
)

try:
    for sample in samples:
        result = process_sample(sample)

        # Save/update the data consumed by report.qmd.
        record_completed_sample(result)

        # Non-blocking.
        reporter.sample_completed()
finally:
    # Render the final state even if the count is not divisible by 5.
    reporter.finish()
```

### Required behavior

`sample_completed()` increments an internal completed-sample counter.

When the count reaches the configured interval:

```text
5, 10, 15, 20, ...
```

request a Quarto render.

The render must happen asynchronously so that:

```python
reporter.sample_completed()
```

returns immediately and processing of the next CASA sample can begin.

Use the Quarto CLI:

```bash
quarto render report.qmd
```

Do not run Quarto through CASA or Jupyter APIs.

### Only one render at a time

Never launch several Quarto processes concurrently.

If another render is requested while one is already running:

* remember that another render is needed;
* allow the current render to finish;
* then render the newest state once.

It is not necessary to render every intermediate state.

For example, if samples 5, 10, and 15 finish while a previous render is still running, it is acceptable to render once afterward using the current data containing all 15 samples.

This should be implemented with one small background worker/thread and one Quarto subprocess. Avoid queues, process pools, task frameworks, or other unnecessary machinery.

### `finish()`

`finish()` must:

1. request a render regardless of the current sample count;
2. wait for the current/final render to finish;
3. stop the worker cleanly.

This ensures an experiment with 23 samples and `every=5` eventually reports all 23.

### Output message

By convention, report files use Quarto's default same-stem HTML output:

```text
report.qmd
→
report.html
```

After every successful render, print:

```text
Report updated: /absolute/path/to/report.html
```

On the final render, the same output is sufficient.

If Quarto fails, print/report the error clearly. A failed intermediate report render must not terminate the scientific sample-processing loop. A failure from `finish()` should be reported but must not destroy already completed experiment products.

### Validation

Reject:

```python
every <= 0
```

and reject a missing `.qmd` file.

No additional configuration options are needed initially.

---

# 2. First report: VLA sample imaging report

Create the first reusable Quarto template:

```text
scripts/reporting/vla_samples.qmd
```

This report is specifically for completed VLA Pipeline imaging samples.

Do not make this template generic for every future experiment.

The genericity belongs in `QuartoReporter`; other experiments can later supply different `.qmd` files.

---

## Report data

Each experiment directory should contain:

```text
experiment_dir/
├── report.qmd
├── report.json
├── sample_001/ (not this name, the ID of the ms would be better)
│   └── vla_pipeline/
│       ├── qa.json
│       ├── dirty.png
│       ├── clean.png
│       └── residual.png
├── sample_002/
│   └── ...
└── report.html
```

At experiment creation, copy:

```text
scripts/reporting/vla_samples.qmd
```

to:

```text
experiment_dir/report.qmd
```

The QMD always reads:

```text
report.json
```

from the same directory.

`report.json` should contain only the information needed to find completed samples and describe the experiment:

```json
{
  "title": "Experiment title",
  "description": "Description of what this experiment tests.",
  "samples": [
    {
      "id": "0012-399",
      "ms_path": "/path/to/sample.ms",
      "result_dir": "sample_001/vla_pipeline"
    }
  ]
}
```

Do not duplicate all QA metrics or imaging parameters into the manifest.

The report should read those from each sample's existing:

```text
result_dir/qa.json
```

This keeps `qa.json` authoritative.

### Atomic manifest updates

The processing loop may update `report.json` while Quarto is rendering.

Write it atomically:

1. write the complete new JSON to a temporary file;
2. replace `report.json` using `os.replace()`.

Quarto should therefore see either the previous complete manifest or the new complete manifest, never a half-written JSON file.

Add a sample to the manifest only after its imaging, QA JSON, and three PNG files have been successfully written.

---

# 3. Exact first-report layout

The HTML report should have:

```text
Experiment title

Experiment description
```

followed by one repeated block per completed sample.

Each sample consists of exactly two visual rows and three columns.

Conceptually:

```text
────────────────────────────────────────────────────────────────────────

ID / MS PATH              CLEANING PARAMETERS           QA METRICS
0012-399                  deconvolver: ...              scaled MAD: ...
/path/to/file.ms           nterms: ...                   peak/MAD: ...
                           weighting: ...                p99/MAD: ...
                           robust: ...                   p99.5/MAD: ...
                           niter: ...                    dynamic range: ...
                           nsigma: ...                   VLA background RMS: ...
                           mask: ...                     
                           beam
                           etc

[ DIRTY PNG ]              [ CLEAN PNG ]                 [ RESIDUAL PNG ]

────────────────────────────────────────────────────────────────────────

NEXT SAMPLE ...
```

### Separation rules

There must be:

* no visible horizontal separator between the two rows belonging to the same sample;
* no cell borders forming a traditional table;
* one clear horizontal separator between samples.

Use an HTML/CSS grid rather than a Markdown table.

Each sample can be represented by one outer `<div class="sample">` containing:

```text
sample-info grid:   3 columns
sample-images grid: 3 columns
```

and only `.sample` gets a bottom border.

---

## First row, column 1: sample identity

Display:

```text
visibility ID
absolute/resolved Measurement Set path
```

The ID should be visually prominent.

Long paths must wrap rather than overflow horizontally.

---

## First row, column 2: cleaning parameters

Read the final effective VLA Pipeline imaging parameters from `qa.json`.

Display the most useful cleaning/imaging parameters, including when available:

```text
deconvolver
nterms
weighting
robust
imsize
cell
usemask / mask
niter
nsigma
threshold
cycleniter
etc
```

Do not dump the entire parameter dictionary. Also you can group parameters together like the qa.txt is doing as to be brief and fit stuff in the cell.

Missing values should display as:

```text
—
```

rather than failing the report.

---

## First row, column 3: QA metrics

Display the existing QA metrics, including:

```text
scaled MAD / sigma
residual absolute peak
peak / scaled MAD
p99 / scaled MAD
p99.5 / scaled MAD
dynamic range
```

and specifically:

```text
VLA Pipeline background RMS
```

using:

```text
vla_background_rms_jy_per_beam
```

with a readable representation such as:

```text
1.687e-06 Jy/beam
```

This value means:

```text
VLA Pipeline non-PB-corrected noise-annulus RMS
```

The report may show that phrase in smaller explanatory text, but the main label can simply be:

```text
VLA background RMS
```

If unavailable, display:

```text
—
```

and allow the existing QA warnings to explain the missing value. You can use the same way of showing metrics as the qa.txt

---

# 4. Second row: images

The second row contains:

```text
column 1: dirty.png
column 2: clean.png
column 3: residual.png
```

Each image should:

* fill the available column width while preserving aspect ratio;
* have a short caption: `Dirty`, `Clean`, or `Residual`;
* not have an artificial border unless needed for visibility;
* use the PNGs already produced by the imaging package.

Do not regenerate these plots inside Quarto.

---

# 5. Quarto implementation

The QMD should use Python only to:

1. read `report.json`;
2. iterate through completed samples;
3. read each sample's `qa.json`;
4. emit the repeated HTML/Markdown block.

Use a Quarto Python cell with:

```text
output: asis
```

or the equivalent current Quarto syntax.

The report layout should be implemented with simple HTML and CSS embedded directly in `report.qmd`.

Do not introduce a separate CSS file in the first implementation.

The report should require no buttons, JavaScript actions, filters, interactive plots, or server.

Its purpose is static scientific reporting.

---

# 6. Incremental integration with sample processing

Refactor the current one-sample experiment logic into a function in a separate script, for the "compute_background_RMS" experiment. 

Then batch processing can look approximately like:

```python
reporter = QuartoReporter(
    experiment_dir / "report.qmd",
    every=5,
)

try:
    for ms in samples:
        
        vla_result = image_ms_VLA_pipe(
            ms,
            sample_output_dir,
        )

        reporter.sample_completed()

finally:
    reporter.finish()
```

Do not force the generic reporter to understand `ImagingResult`, VLA, CASA, QA metrics, or sample manifests.

Its only responsibilities are:

```text
count completed iterations
→ request render every N
→ render asynchronously
→ avoid concurrent renders
→ print report path
→ perform final render
```

---

# 7. Failure behavior

A failure processing one scientific sample should follow whatever experiment policy already exists; reporting must not redefine that policy.

A reporting failure must not invalidate the sample.

If Quarto fails during an intermediate asynchronous render:

```text
Report render failed: ...
```

should be printed and processing should continue.

A subsequent render request should be allowed to try again.

If a sample is incomplete, it must not be appended to `report.json`.

Therefore the report always represents only fully completed samples.

---

# 8. Initial output format

The primary output is:

```text
report.html
```

During processing, keep it as normal HTML referencing the PNG files rather than embedding all resources. This keeps repeated renders lighter.

Do not initially generate PDF or Markdown versions.

A self-contained HTML export with embedded resources can be added later as a separate final/share operation if needed.

---

# 9. Keep the implementation small

The intended implementation should be roughly:

```text
scripts/reporting/
├── __init__.py
├── quarto.py
└── vla_samples.qmd
```

plus a small manifest-update helper where it naturally belongs.

Do not add:

* generic report schemas;
* plugin systems;
* abstract renderer interfaces;
* multiple rendering backends;
* web servers;
* task queues;
* multiprocessing pools;
* file watchers;
* database-backed experiment tracking.

The first implementation should solve only:

> “While a long experiment processes samples, keep an HTML Quarto report reasonably up to date without blocking CASA, and always render the final completed state.”


## Report summary

At the top of `vla_samples.qmd`, add a compact summary section showing the current state of the experiment.

Display:

```text
Processed samples: 37 / 120

Background RMS
Mean:   2.13e-06 Jy/beam
Median: 1.89e-06 Jy/beam
Std:    8.21e-07 Jy/beam
```

The statistics must use only completed samples with a valid:

```text
vla_background_rms_jy_per_beam
```

Also generate a small histogram of those RMS values using Matplotlib.

Conceptually:

```python
rms_values = [
    sample_rms
    for sample in completed_samples
    if sample_rms is not None
]

mean = np.mean(rms_values)
median = np.median(rms_values)
std = np.std(rms_values)

fig, ax = plt.subplots(figsize=(5, 2.5))
ax.hist(rms_values)
ax.set_xlabel("VLA background RMS [Jy/beam]")
ax.set_ylabel("Samples")
```

The summary should appear before the per-sample sections and update automatically on every Quarto render.

`report.json` must therefore include the total number of samples expected:

```json
{
  "total_samples": 120,
  "samples": [...]
}
```

The number processed is the number of successfully completed samples currently present in the manifest.

If no valid RMS values exist yet, show the processed count but display the RMS statistics and histogram as unavailable rather than failing the report.

If it makes sense, instead of reading all the values from the json for this RMS noise, might be useful to keep updating a global csv with ID, VLA Background RMS values in each iteration, and the reporting can use that.