#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
[[ $# -ge 2 ]] || { echo "run_profile.sbatch requires a profile and command" >&2; exit 2; }
profile=$1
shift
export RADIOASTRO_PROFILE="$profile"

case "$profile" in
    ml-cpu|ml-gpu) runtime=ml ;;
    casa) runtime=casa ;;
    casa-pipeline) runtime=casa-pipeline ;;
    *) echo "Unknown profile: $profile" >&2; exit 2 ;;
esac

echo "Slurm job $SLURM_JOB_ID on $(hostname -f 2>/dev/null || hostname)"
exec "$ROOT/bin/run" "$runtime" "$@"
