#!/usr/bin/env bash
#SBATCH --job-name=maverick-benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx_4090:1,tmpdisk:20480
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ! -f "$repo/src/benchmark/__main__.py" ]]; then
    repo=${SLURM_SUBMIT_DIR:-$repo}
fi
if [[ ! -f "$repo/src/benchmark/__main__.py" ]]; then
    printf '%s\n' 'Submit the benchmark from the Maverick checkout root.' >&2
    exit 1
fi
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"

case "${1:-}" in
    capture|run|--help|-h) ;;
    *) set -- run "$@" ;;
esac
python=${MAVERICK_PYTHON:-}
if [[ -z $python && -n ${MAVERICK_VENV:-} ]]; then
    python="$MAVERICK_VENV/bin/python"
    if [[ ! -x $python ]]; then
        printf 'Missing Python in MAVERICK_VENV: %s\n' "$python" >&2
        exit 1
    fi
fi
exec "${python:-python3}" -m src.benchmark "$@"
