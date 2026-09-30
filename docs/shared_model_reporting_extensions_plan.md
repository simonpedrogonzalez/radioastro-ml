# Shared model-reporting extensions

## Contract

Apply the following to tabular, ResNet-18, DINOv2, and later comparison reports
through shared evaluation/report helpers.

## Strength-weighted confusion matrices

- Preserve the ordinary count matrix. Add a second matrix row where each clean
  baseline has weight `1` and each corrupted sample with target `s>0` has
  `w(s)=min(1, log1p(s)/log1p(30))`.
- Current target weights are `5=.522`, `10=.698`, `15=.807`, `20=.887`, and
  `30/40/50/100=1`. This discounts the hardest weak corruptions without
  discounting clean false positives or strong corruptions.
- Store the weighted matrix and weighting definition in every new validation
  result. The confusion figure uses columns=models, top row=raw counts, bottom
  row=weighted counts, with one shared color scale per row.
- Plot the weight curve beside the confusion section, including the special
  clean-baseline point `(0,1)` and every observed positive target.

## Increased-noise controls

- Replace the per-target table with an overall table containing exactly:
  `Model | Original FPR | Increased-noise FPR | Excess FPR`.
- Add a separate overall predicted-count table with one model per row and one
  predicted-class column per class. Counts cover all increased-noise samples.
- Retain the existing per-target FPR plot and its matched-baseline reference;
  retain per-target data in `results.json` for auditability.

## Hard samples

- Read validation predictions from every completed model/run in the report and
  require identical sample truth/metadata across models.
- A sample is hard only when every included model predicts it incorrectly.
  Select at most 10: clean false positives first, then corrupted samples by
  descending target strength, with sample ID as the deterministic tie-breaker.
- Resolve each selected sample through the main dataset manifest. Render one
  six-column table: `ID + full manifest label`, dirty, clean, residual, PSF,
  and written metrics. Metrics include measured/target strength, the four QA
  inputs, and the aggregate predicted-class vote counts.
- Cache the 40 maximum PNG panels in the run directory. Incremental NN reports
  state how many completed runs define the current hard-sample intersection.

## Code and verification

- `ml/task_evaluation.py`: strength weights and weighted confusion result.
- `ml/report.py`: two-row confusion figure, weight curve, simplified control
  tables, common prediction loading/hard selection, image/metric table.
- `ml/logreg.py`: pass dataset and predictions to the shared writer.
- `ml/run_nn_experiments.py`: collect completed-job predictions and call the
  same helpers in incremental reports.
- Tests cover exact weights/matrix values, overall control tables, hard-sample
  priority/cap/consensus, both report paths, and report rendering smoke.
