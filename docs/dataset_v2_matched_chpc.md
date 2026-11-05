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

For the validation split, use a separate `RUN` and a prepare-stage config file containing `{"split": "val"}`.

## Notes

- **Split stages:** the prepare stage uses the shard count only to divide sources; `--plan --num-shards N` sets the execution assignment. To change `N` later, stop all jobs and rerun `--plan` with the new `N`; the plan itself does not change.
- **Scratch:** `--scratch-root` holds disposable per-unit Measurement Set copies and is cleaned per unit (default `RUN/scratch`). Retained products go to `RUN/cases/`.
- **Progress and failures:** see `RUN/units/*.seal.json` (done), `RUN/failures/` (tracebacks) and `RUN/logs/` (per-unit CASA logs). `--verify` writes `RUN/status.json` and lists missing units.
- **Measured on the local smoke** (one 67 MB source, R = 12):
  - per unit: about 250 s, peak scratch 635 MB (about 3× the simulation MS) and peak memory 590 MB per CASA process;
  - retained: 7.3 MB per case.
  - Larger sources scale scratch with their simulation MS size; size `ML_CPU_MEM` and the scratch space for the largest source in the shard.
