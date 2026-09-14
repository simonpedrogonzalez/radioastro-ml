
# Codex task: visibility anomaly QA module + confirmation panel

Implement a new visibility-based QA system for detecting bad antennas/baselines/time/channel anomalies. Do not automatically flag anything yet. The system should only compute metrics, infer a recommended culprit when evidence is strong, save a confirmation plot, and store the results in the existing experiment JSON/report.

## 1. Add a separate script

Create a new script:

`scripts/visibility_anomaly_qa.py`

This script must contain one main callable function:

```python
def analyze_visibility_anomalies(
    ms_path: Path,
    output_dir: Path,
    *,
    spw: str = "",
    uvrange: str = "",
    datacolumn_preference: str = "corrected",
    max_rows: int | None = None,
    config: dict | None = None,
) -> dict:
    ...
````

This function should:

1. Read visibility amplitudes from the MS.
2. Compute robust anomaly scores.
3. Aggregate anomalies by antenna, baseline, time, channel/SPW, and useful combinations.
4. Decide whether there is a likely culprit.
5. Save exactly one confirmation PNG.
6. Return a JSON-serializable dictionary with all metrics, thresholds, conclusion, recommendation, and artifact path.

No automatic flagging should be performed.

IMPORTANT: DO NOT CHANGE THE MS IN ANY WAY, the MS MUST BE PRESERVED.

---

## 2. What data to read

Read from the MS:

* `CORRECTED_DATA` if available and `datacolumn_preference="corrected"`, otherwise `DATA`.
* `FLAG`
* `FLAG_ROW`
* `UVW`
* `ANTENNA1`
* `ANTENNA2`
* `TIME`
* `DATA_DESC_ID`
* `SPECTRAL_WINDOW_ID` from `DATA_DESCRIPTION`
* channel frequencies from `SPECTRAL_WINDOW`

Use only unflagged data:

* remove rows where `FLAG_ROW=True`
* remove samples where `FLAG=True`
* ignore non-finite amplitudes
* ignore zero or negative amplitudes before taking logs

Use correlations:

* if there is one correlation, use it
* if there are multiple correlations, use the first and last correlation, matching the existing plotting logic
* keep the correlation index in the internal arrays so anomalies can be checked per correlation if needed

Use amplitude:

```python
amp = abs(vis)
log_amp = log10(amp)
```

Use `log_amp` because visibility amplitude errors are often multiplicative. A bad antenna gain, bad baseline, or bursty corruption often creates fractional amplitude changes rather than additive changes, so log-amplitude makes upward/downward deviations more comparable.

---

## 3. Local robust normalization

Do not threshold raw amplitudes directly.

Raw amplitude depends on:

* frequency / SPW
* channel
* uvdistance
* source structure
* calibration state
* correlation

Instead compute robust local z-scores.

For each visibility sample, assign bins:

* SPW
* correlation
* uvdistance bin
* channel bin

Recommended defaults:

```python
config = {
    "uvdist_nbins": 24,
    "channel_bin_size": 8,
    "min_bin_count": 50,
    "mad_floor": 1e-6,
    "strong_z": 8.0,
    "moderate_z": 5.0,
    "min_bad_samples": 20,
    "min_group_bad_fraction": 0.05,
    "min_enrichment": 5.0,
    "min_coverage": 0.25,
    "max_data_loss_for_recommendation": 0.25,
}
```

For each local bin:

```python
local_median = median(log_amp)
local_sigma = 1.4826 * MAD(log_amp)
z = (log_amp - local_median) / max(local_sigma, mad_floor)
```

Why:

* local median removes expected amplitude trends with uvdistance/frequency
* MAD is robust against outliers
* z-score measures how unusual a visibility is relative to similar visibilities
* bad data should produce unusually large `|z|`

Define anomaly masks:

```python
strong_bad = abs(z) >= strong_z
moderate_bad = abs(z) >= moderate_z
```

Use `strong_bad` for primary culprit detection. Use `moderate_bad` only as supporting evidence in plots/metrics.

If a bin has fewer than `min_bin_count` samples, fall back to a broader bin:

1. SPW + correlation + uvdistance bin
2. SPW + correlation
3. global correlation
4. global all-data robust median/MAD

Store how many samples used fallback normalization.

---

## 4. Global anomaly metrics

Return these global metrics:

```python
{
  "visibility_qa": {
    "status": "ok" | "suspect" | "insufficient_data" | "error",
    "data_column_used": "CORRECTED_DATA" | "DATA",
    "n_samples_total": int,
    "n_samples_used": int,
    "n_strong_bad": int,
    "n_moderate_bad": int,
    "strong_bad_fraction": float,
    "moderate_bad_fraction": float,
    "median_abs_z": float,
    "p95_abs_z": float,
    "p99_abs_z": float,
    "max_abs_z": float,
    "normalization": {...},
    "thresholds": {...}
  }
}
```

Interpretation:

* `strong_bad_fraction` increases when many visibilities are extreme outliers.
* `max_abs_z` catches isolated but severe spikes.
* `p99_abs_z` catches heavy-tailed corruption even when only a small fraction of data is bad.
* `median_abs_z` should stay near normal for mostly good data; if it is high, the whole dataset/model may be problematic.

---

## 5. Culprit aggregation

Aggregate the `strong_bad` mask by candidate explanations.

Compute candidates for:

### Simple groups

* antenna
* baseline
* time bin
* channel bin
* SPW

### Combination groups

* antenna × time bin
* baseline × time bin
* antenna × channel bin
* baseline × channel bin
* SPW × channel bin

Use configurable time bins:

```python
"time_nbins": 64
```

For each candidate group compute:

```python
n_total
n_bad
bad_fraction = n_bad / n_total
global_bad_fraction = total_bad / total_used
enrichment = bad_fraction / max(global_bad_fraction, eps)
coverage = n_bad / total_bad
data_loss = n_total / total_used
median_abs_z_in_group
p95_abs_z_in_group
max_abs_z_in_group
```

Why:

* `coverage` says how much of the bad data this candidate explains.
* `purity` / `bad_fraction` says how concentrated the anomaly is inside that candidate.
* `enrichment` says whether the candidate is much worse than the dataset average.
* `data_loss` says how much data would be removed if this candidate were flagged.
* A good culprit should have high coverage, high enrichment, and modest data loss.

Define:

```python
candidate_score = coverage * log1p(enrichment) * bad_fraction / sqrt(max(data_loss, eps))
```

Return the top candidates sorted by score.

---

## 6. Culprit classification logic

Choose the best recommendation, but only if evidence is strong.

A candidate can be recommended if:

```python
n_bad >= min_bad_samples
coverage >= min_coverage
enrichment >= min_enrichment
bad_fraction >= min_group_bad_fraction
data_loss <= max_data_loss_for_recommendation
```

Classification rules:

### Bad baseline

Recommend `bad_baseline` if one baseline candidate dominates and does not imply many baselines from the same antenna are also bad.

Conclusion example:
`Likely bad baseline ea03&ea17: explains 61% of strong outliers with 14x enrichment.`

### Bad antenna

Recommend `bad_antenna` if:

* many of the top bad baselines share one antenna, or
* the antenna candidate has high coverage/enrichment, and
* baselines touching that antenna have much higher bad fraction than baselines not touching it.

Compute:

```python
touching_bad_fraction
non_touching_bad_fraction
antenna_touching_enrichment = touching_bad_fraction / non_touching_bad_fraction
```

Conclusion example:
`Likely bad antenna ea12: baselines touching it have 18x higher anomaly rate.`

### Bad time range

Recommend `bad_time_range` if one or a few adjacent time bins explain the anomaly across many baselines/antennas.

Conclusion example:
`Likely bad time range: anomaly is concentrated in 2 adjacent time bins and affects many baselines.`

### Bad channel/SPW range

Recommend `bad_channel_range` if anomalies concentrate in a channel range across many baselines/antennas.

Conclusion example:
`Likely bad channel range: anomaly is concentrated in SPW 0 channels 120-152.`

### Bad antenna-time or baseline-time

Recommend a combination culprit if a simple antenna/baseline is not globally bad, but becomes highly anomalous during a short time range.

### Bad baseline-channel or antenna-channel

Recommend a combination culprit if a baseline/antenna is only anomalous in a channel range.

### No recommendation

If no candidate passes the thresholds:

```python
recommendation = "none"
status = "ok" if global anomaly rates are low else "suspect"
```

Also include a safeguard:
If anomalies correlate smoothly with uvdistance and are not concentrated by antenna/baseline/time/channel, classify as:

```python
"possible_source_structure_or_model_mismatch"
```

This avoids recommending flags for real resolved source structure or model mismatch.

---

## 7. Confirmation plot

The function should save exactly one confirmation PNG:

```python
confirmation_png = output_dir / f"{prefix}_visibility_anomaly_confirmation.png"
```

If everything looks okay, still create a simple empty/summary panel image saying:
`Visibility anomaly QA: OK / no strong culprit`

If a culprit is found, generate the most useful plot for that culprit:

### For bad antenna

Plot:

* x-axis: uvdistance
* y-axis: normalized amplitude or robust z-score
* color/group: baselines touching culprit antenna vs not touching it
* add title text with conclusion and main metrics

Preferred:

```text
amp / local median vs uvdist
or
abs(z) vs uvdist
```

### For bad baseline

Plot:

* x-axis: time or uvdistance
* y-axis: normalized amplitude or abs(z)
* highlight culprit baseline against all other baselines

### For bad time range

Plot:

* x-axis: time bin
* y-axis: strong bad fraction
* highlight recommended time range

### For bad channel range

Plot:

* x-axis: channel/frequency
* y-axis: strong bad fraction
* highlight recommended channel range

### For bad antenna-time / baseline-time

Plot:

* heatmap: time bin vs baseline or antenna
* color: bad fraction or median abs(z)
* highlight culprit row/time range

Plot requirements:

* save as PNG
* make title readable
* include conclusion text inside the plot
* include the key metrics:

  * culprit type
  * culprit id
  * coverage
  * enrichment
  * bad fraction
  * data loss
  * recommended action text, but explicitly say `No automatic flagging applied`

The plot should be useful as human confirmation, not just a metric plot.

---

## 8. Returned dictionary structure

Return JSON-serializable data like:

```python
{
  "status": "ok" | "suspect" | "insufficient_data" | "error",
  "summary": {
    "conclusion": str,
    "recommendation": "none" | "inspect" | "flag_candidate",
    "culprit_type": str | None,
    "culprit_id": str | None,
    "recommended_selection": {
      "antenna": str | None,
      "baseline": str | None,
      "timerange": str | None,
      "spw": str | None,
      "reason": str,
    },
  },
  "metrics": {
    "n_samples_used": int,
    "n_strong_bad": int,
    "strong_bad_fraction": float,
    "n_moderate_bad": int,
    "moderate_bad_fraction": float,
    "median_abs_z": float,
    "p95_abs_z": float,
    "p99_abs_z": float,
    "max_abs_z": float,
  },
  "top_candidates": [
    {
      "type": "antenna" | "baseline" | "time" | "channel" | ...,
      "id": str,
      "n_total": int,
      "n_bad": int,
      "bad_fraction": float,
      "coverage": float,
      "enrichment": float,
      "data_loss": float,
      "median_abs_z": float,
      "p95_abs_z": float,
      "max_abs_z": float,
      "score": float,
    }
  ],
  "thresholds": {...},
  "normalization": {
    "method": "local_log_amp_median_mad",
    "uvdist_nbins": int,
    "channel_bin_size": int,
    "time_nbins": int,
    "min_bin_count": int,
    "fallback_fraction": float,
  },
  "artifacts": {
    "visibility_anomaly_confirmation_png": str
  }
}
```

All values must be plain Python types, not NumPy scalars.

---

## 9. Integration into image_extracted

In the imaging pipeline, call:

```python
from scripts.visibility_anomaly_qa import analyze_visibility_anomalies
```

Call it after `ms_for_imaging` exists and after zero-visibility flagging, but before or after imaging is okay. Preferred location:

* after `flag_zero_visibilities(ms_for_imaging)`
* before first-pass dirty imaging

Reason:

* this analyzes the visibility data used for imaging
* it does not depend on image products
* it can warn about bad antennas/baselines before interpreting image artifacts

Use:

```python
visibility_qa = analyze_visibility_anomalies(
    ms_for_imaging,
    output_dir,
    spw="",
    uvrange=applied_uvrange,
    datacolumn_preference="corrected",
)
```

Add the confirmation image to the artifact map:

```python
"visibility_anomaly_confirmation_png": visibility_qa["artifacts"].get("visibility_anomaly_confirmation_png")
```

Add the returned QA block to the per-target result:

```python
"visibility_anomaly_qa": visibility_qa
```

Then include it in the report JSON under each target, for example:

```python
"visibility_anomaly_qa": row.get("visibility_anomaly_qa", {})
```

This must be saved in the existing JSON report.

---

## 10. Integration into plot_extracted

Add one final panel to the contact sheet.

Add to `PANEL_SPECS` after the spectrum panels:

```python
(
    "visibility_anomaly",
    "visibility anomaly QA",
    [
        f"{IMAGE_PREFIX}_visibility_anomaly_confirmation.png",
        "*visibility_anomaly_confirmation*.png",
    ],
    False,
),
```

Add to `PANEL_ARTIFACT_KEYS`:

```python
"visibility_anomaly": "visibility_anomaly_confirmation_png",
```

Load the visibility QA block from the image report in `load_meta_map()`:

```python
"visibility_anomaly_qa": sample.get("visibility_anomaly_qa", {}) or {}
```

Pass it through `find_valid_samples()` into each sample.

Add a title function similar to `make_residual_title()`:

```python
def make_visibility_anomaly_title(sample: dict) -> str:
    qa = sample.get("visibility_anomaly_qa", {}) or {}
    summary = qa.get("summary", {}) or {}
    metrics = qa.get("metrics", {}) or {}

    return (
        "visibility anomaly QA\n"
        f"status={qa.get('status', '?')}\n"
        f"culprit={summary.get('culprit_type', 'none')} {summary.get('culprit_id', '')}\n"
        f"bad={format_metric(metrics.get('strong_bad_fraction'))}\n"
        f"p99|z|={format_metric(metrics.get('p99_abs_z'))}\n"
        f"{summary.get('conclusion', 'No visibility anomaly analysis available')}"
    )
```

In `make_contact_sheet()`, for this panel:

* show the confirmation PNG if present
* if missing, leave panel blank or show `visibility anomaly QA not available`
* set the panel title to `make_visibility_anomaly_title(sample)`

Also include this panel and QA output in the plot report JSON.

---

## 11. Important behavior

The module must never automatically modify the MS or call `flagdata`.

It should only produce:

1. metrics
2. recommendation
3. confirmation plot
4. JSON output

Recommended wording:

* `Recommended inspection target`
* `Suggested flag candidate`
* `No automatic flagging applied`

Do not say data was flagged.

---

## 12. Minimal acceptance criteria

The implementation is complete when:

1. `scripts/visibility_anomaly_qa.py` exists.
2. `analyze_visibility_anomalies(...)` can run on one MS and returns a JSON-serializable dict.
3. A confirmation PNG is always created.
4. `image_extracted` stores the returned dict in the image report JSON.
5. `image_extracted` stores the confirmation PNG path in the target artifacts.
6. `plot_extracted` adds a final visibility anomaly panel.
7. The final panel shows the confirmation image when available.
8. The final panel title contains the status, conclusion, culprit, and key metrics.
9. No automatic flagging is performed.
