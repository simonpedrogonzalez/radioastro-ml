#!/usr/bin/env bash
# Submit the matched Dataset v2 chain for one split as dependent Slurm jobs:
#   prepare (N shards) -> plan -> execute (N shards) -> verify -> report
# See docs/dataset_v2_matched_chpc.md.
#
# usage: slurm/submit_dataset_v2_matched.sh SPLIT SHARDS PARALLEL
# Environment overrides (defaults from config/chpc.env CPU settings):
#   RUN            output directory           (runs/dataset-v2-matched-$SPLIT)
#   CONFIG         prepare-stage config JSON  ({"split": SPLIT} written next to RUN for val/test)
#   CLUSTER/ACCOUNT                           (CHPC_CPU_CLUSTER/CHPC_CPU_ACCOUNT)
#   PARTITION/QOS  heavy-stage partition/QoS  (CHPC_CPU_PARTITION/CHPC_CPU_QOS)
#   SMALL_PARTITION/SMALL_QOS  plan/verify/report partition/QoS (CHPC_CPU_PARTITION/CHPC_CPU_QOS)
#   EXCLUDE        nodes to avoid (sbatch --exclude)
#   PREPARE_SHARDS prepare-stage shard count  (SHARDS)
#   NAME           job-name prefix            (v2-$SPLIT)
#   SCRATCH_ROOT   disposable unit scratch    (/scratch/general/vast/$USER/v2-scratch)
#   MEM_PER_CPU    heavy-stage memory per CPU (4G)
#   PREPARE_TIME / EXECUTE_TIME               (02:00:00 / 04:00:00)
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
"$ROOT/bin/configure-chpc" --check >/dev/null
source "$ROOT/config/chpc.env"

[[ $# -eq 3 ]] || { echo "usage: $0 SPLIT SHARDS PARALLEL" >&2; exit 2; }
split=$1 shards=$2 parallel=$3
[[ "$split" =~ ^(train|val|test)$ ]] || { echo "SPLIT must be train, val or test" >&2; exit 2; }

run=${RUN:-runs/dataset-v2-matched-$split}
cluster=${CLUSTER:-$CHPC_CPU_CLUSTER}
account=${ACCOUNT:-$CHPC_CPU_ACCOUNT}
partition=${PARTITION:-$CHPC_CPU_PARTITION}
qos=${QOS-$CHPC_CPU_QOS}
small_partition=${SMALL_PARTITION:-$CHPC_CPU_PARTITION}
small_qos=${SMALL_QOS-$CHPC_CPU_QOS}
prepare_shards=${PREPARE_SHARDS:-$shards}
prefix=${NAME:-v2-$split}
scratch=${SCRATCH_ROOT:-/scratch/general/vast/$USER/v2-scratch}
mem=${MEM_PER_CPU:-4G}
mkdir -p "$ROOT/slurm/logs" "$scratch"

# Job environment (sbatch exports the submitting environment).
export RADIOASTRO_ROOT="$ROOT" RADIOASTRO_PROFILE=ml-cpu OMP_NUM_THREADS=1 RADIOASTRO_CASA_AUTO_UPDATE=0
export RADIOASTRO_ML_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"

config_args=()
if [[ -n "${CONFIG:-}" ]]; then
    config_args=(--config "$CONFIG")
elif [[ "$split" != train ]]; then
    config="$ROOT/$run.config.json"
    mkdir -p "$(dirname "$config")"
    printf '{"split": "%s"}\n' "$split" > "$config"
    config_args=(--config "$config")
fi

# submit NAME PARTITION QOS CPUS MEM TIME DEPENDENCY -- generator args...
submit() {
    local name=$1 part=$2 q=$3 cpus=$4 memory=$5 walltime=$6 dep=$7
    shift 8
    local options=(--parsable --job-name="$prefix-$name" --clusters="$cluster"
                   --partition="$part" --account="$account" --nodes=1 --ntasks=1
                   --cpus-per-task="$cpus" --mem="$memory" --time="$walltime"
                   --output="$ROOT/slurm/logs/%x-%j.out" --error="$ROOT/slurm/logs/%x-%j.err")
    [[ -z "$q" ]] || options+=(--qos="$q")
    [[ -z "${EXCLUDE:-}" ]] || options+=(--exclude="$EXCLUDE")
    [[ -z "$dep" ]] || options+=(--dependency="afterok:$dep")
    sbatch "${options[@]}" "$ROOT/slurm/run_profile.sbatch" ml-cpu \
        python -m scripts.create_dataset_v2_matched --output "$run" "$@" | cut -d';' -f1
}
# Small single-process stages stay on the non-preemptible CPU allocation.
small() { submit "$1" "$small_partition" "$small_qos" 2 16G 01:00:00 "$2" -- "${@:3}"; }
heavy_mem=$(( ${mem%G} * parallel ))G

prepare=()
for ((i = 0; i < prepare_shards; i++)); do
    prepare+=("$(submit "prepare$i" "$partition" "$qos" "$parallel" "$heavy_mem" "${PREPARE_TIME:-02:00:00}" "" -- \
        --prepare "${config_args[@]}" --num-shards "$prepare_shards" --shard "$i" --parallel "$parallel")")
done
plan=$(small plan "$(IFS=:; echo "${prepare[*]}")" --plan --num-shards "$shards")
execute=()
for ((i = 0; i < shards; i++)); do
    execute+=("$(submit "execute$i" "$partition" "$qos" "$parallel" "$heavy_mem" "${EXECUTE_TIME:-04:00:00}" "$plan" -- \
        --execute --num-shards "$shards" --shard "$i" --parallel "$parallel" --scratch-root "$scratch")")
done
verify=$(small verify "$(IFS=:; echo "${execute[*]}")" --verify)
report=$(sbatch --parsable --job-name="$prefix-report" --clusters="$cluster" \
    --partition="$small_partition" --account="$account" ${small_qos:+--qos="$small_qos"} ${EXCLUDE:+--exclude="$EXCLUDE"} \
    --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G --time=01:00:00 --dependency="afterok:$verify" \
    --output="$ROOT/slurm/logs/%x-%j.out" --error="$ROOT/slurm/logs/%x-%j.err" \
    "$ROOT/slurm/run_profile.sbatch" ml-cpu python -m scripts.report_dataset_v2_matched --dataset "$run" | cut -d';' -f1)

echo "$split -> $run  (partition $partition, $shards x $parallel CPUs)"
echo "  prepare: ${prepare[*]}"
echo "  plan:    $plan"
echo "  execute: ${execute[*]}"
echo "  verify:  $verify"
echo "  report:  $report"
