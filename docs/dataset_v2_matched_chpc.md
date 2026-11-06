# Running the matched Dataset v2 generator on CHPC (CPU, Slurm)

Generator: `scripts/create_dataset_v2_matched.py` ([plan](dataset_v2_updated_plan.md) §2, §6). CHPC setup, accounts and environments: [chpc_setup.md](chpc_setup.md).

## Prerequisites

- `config/chpc.env` exists and `bin/configure-chpc --check` passes. `make chpc-setup` has been run.
- The generator reads these inputs, which must exist in the clone (directly or through the scratch links made by setup):
  - `collect/experiments/extracted_thermal_simulation_comparison_20260909T005739/` (parent simulations and `report.json`);
  - the original Measurement Sets referenced by that report;
  - `collect/experiments/dataset_v1_20260921_all_psf/dataset.json` (source index and splits);
  - `runs/simulation-parametrization-v2-20261104/plan.json` (imaging contract).
- The collected data are only read; the generator never modifies them.

## How jobs map to Slurm

Every stage runs through the ML runtime: `bin/submit ml-cpu python -m scripts.create_dataset_v2_matched …`. The `--prepare` and `--execute` drivers launch their CASA workers with `bin/run casa`, which loads the CASA module on CHPC.

- One Slurm job runs one shard (`--shard i` of `--num-shards N`).
- Inside the job, `--parallel P` CASA processes run at once. Keep `P` at or below the job's CPUs, set by `ML_CPU_CPUS` (default 8), with `ML_CPU_MEM` and `ML_CPU_TIME` in `config/chpc.env`.
- `bin/submit` sends a single job, so submit one job per shard in a shell loop.
- Run at most one live job per shard. Rerunning a shard skips its sealed units.

## Steps

Choose an output directory `RUN`, a shard count `N` and per-job parallelism `P`. Optional overrides (for example `recipes_per_source`) go in a JSON file passed with `--config` at the prepare stage only.

```bash
RUN=runs/dataset-v2-matched-train; N=4; P=8

# 1. Prepare sources (one job per shard), then wait for all to finish.
for i in $(seq 0 $((N-1))); do
  bin/submit ml-cpu python -m scripts.create_dataset_v2_matched --output $RUN \
    --prepare --num-shards $N --shard $i --parallel $P   # add --config split.json for val
done

# 2. Plan: immutable unit list and the N-shard assignment (after all prepare jobs finish).
bin/submit ml-cpu python -m scripts.create_dataset_v2_matched --output $RUN --plan --num-shards $N

# 3. Execute (one job per shard).
for i in $(seq 0 $((N-1))); do
  bin/submit ml-cpu python -m scripts.create_dataset_v2_matched --output $RUN \
    --execute --num-shards $N --shard $i --parallel $P --scratch-root <scratch dir>
done

# 4. After every shard finishes: verify, then report.
bin/submit ml-cpu python -m scripts.create_dataset_v2_matched --output $RUN --verify
bin/submit ml-cpu python -m scripts.report_dataset_v2_matched --dataset $RUN
```

### Adding the CLEAN2 features to an existing run

New runs compute 22 features: the 20 before plus `clean.iterations` (CLEAN minor-cycle iterations) and `model.nonzero_pixels` (nonzero raw-model pixels of either sign). For a run whose cases were committed with 20 features, compute the two from the retained model FITS and the stored CLEAN record, without re-simulating, then re-verify:

```bash
bin/submit ml-cpu python -m scripts.create_dataset_v2_matched --output $RUN --backfill-features
bin/submit ml-cpu python -m scripts.create_dataset_v2_matched --output $RUN --verify
```

The backfill writes a sealed `cases/<id>/clean2_features.json` per case and leaves `case.json` and all images unchanged. It skips cases that already have them. `--verify` rewrites `<split>.npz` with 22 features; the first 20 columns and all labels are unchanged. Use `--feature-view all22` in the ML scripts to train on them.

For the validation split, use a separate `RUN` and a prepare-stage config file containing `{"split": "val"}`.

## ML experiments

Run these on CHPC too, after `--verify` has written `train.npz` and `val.npz`. Train and val must come from the same CASA environment, because local and CHPC runs draw different noise from the same seed. The models are tabular, so the CPU profile is enough.

```bash
TRAIN=runs/dataset-v2-matched-train; VAL=runs/dataset-v2-matched-val

# Feature-view comparison on the training split; outer folds hold out whole sources.
bin/submit ml-cpu python -m ml.matched_comparison --dataset $TRAIN \
  --output ml/runs/matched-v2-feature-comparison --folds 3 --jobs 8

# Fit on train, evaluate on val (one job per feature view; they are independent).
for v in combined8 combined8_dirty6 combined8_model6 all20; do
  bin/submit ml-cpu python -m ml.multitask --train $TRAIN --val $VAL \
    --output ml/runs/matched-v2-multitask-$v --feature-view $v --jobs 8
done
```

- Amplification-cap experiment: `ml/amp_cap_experiment.py` trains xgb_shared (all20) with training rows filtered to A ≤ 2, 3 and 8.5468, then evaluates all three on the same validation rows with A ≤ 2. It only filters rows; it does not re-simulate.
  ```bash
  bin/submit ml-cpu python -m ml.amp_cap_experiment --train $TRAIN --val $VAL \
    --output ml/runs/matched-v2-amp-cap --jobs 8
  ```
- Learning curve for cap 2 (`ml/amp_learning_curve.py`): xgb_shared, all20, training rows A ≤ 2 from random subsets of whole sources (25, 50, 75 with 5 draws each, then all), all evaluated on the same validation rows with A ≤ 2. Settings and target scales are chosen once on the full set and kept fixed.
  ```bash
  bin/submit ml-cpu python -m ml.amp_learning_curve --train $TRAIN --val $VAL \
    --output ml/runs/matched-v2-amp-learning-curve --jobs 8
  ```
- Error analysis of a finished multitask run (read-only, no refitting), written to `<run>/error_analysis/report.html`:
  ```bash
  bin/submit ml-cpu python -m ml.multitask_error_analysis --run ml/runs/matched-v2-signal-all20 --val $VAL
  ```
- Dirty-image task: add `--with-dirty` to `ml.multitask` to also train and report the per-type dirty change (`dirty_amp/phase/noise`) and the total dirty error `eD`, with the same tables and plots as the model-error targets. Without the flag, runs are unchanged.
  ```bash
  bin/submit ml-cpu python -m ml.multitask --train $TRAIN --val $VAL \
    --output ml/runs/matched-v2-signal-all20-dirty --feature-view all20 --with-dirty --jobs 8
  ```
- Output directories must be new; `matched_comparison` and `amp_cap_experiment` refuse an existing one.
- `multitask` needs at least 3 training sources and no source shared between train and val. With 3 folds, `matched_comparison` needs at least 4 training sources in each fold.
- Keep `--jobs` at or below `ML_CPU_CPUS`.
- `bin/sync-reports` mirrors the resulting reports to `reports/ml-runs/`.
- These were checked only locally, on relabeled copies of the smoke data. Runtimes on CHPC are not measured.

## Notes

- **Split stages:** the prepare stage uses the shard count only to divide sources; `--plan --num-shards N` sets the execution assignment. To change `N` later, stop all jobs and rerun `--plan` with the new `N`; the plan itself does not change.
- **Scratch:** `--scratch-root` holds disposable per-unit Measurement Set copies and is cleaned per unit (default `RUN/scratch`). Retained products go to `RUN/cases/`.
- **Progress and failures:** see `RUN/units/*.seal.json` (done), `RUN/failures/` (tracebacks) and `RUN/logs/` (per-unit CASA logs). `--verify` writes `RUN/status.json` and lists missing units.
- **Measured on the local smoke** (one 67 MB source, R = 12):
  - per unit: about 250 s, peak scratch 635 MB (about 3× the simulation MS) and peak memory 590 MB per CASA process;
  - retained: 7.3 MB per case.
  - Larger sources scale scratch with their simulation MS size; size `ML_CPU_MEM` and the scratch space for the largest source in the shard.
