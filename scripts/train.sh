#!/usr/bin/env bash
#SBATCH --job-name=maverick-train
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx_4090:2,tmpdisk:20480
#SBATCH --time=08:00:00
#SBATCH --output=maverick-%j.log

set -euo pipefail

fail() { printf '%s\n' "$*" >&2; exit 1; }

gpus=2
smoke=0
workloads=()
extra=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) gpus=${2:?Specify 1 or 2 GPUs}; shift 2 ;;
        --workload) workloads+=("${2:?Specify a workload}"); shift 2 ;;
        --epochs) extra+=(--epochs "${2:?Specify an epoch count}"); shift 2 ;;
        --smoke) smoke=1; shift ;;
        -h|--help)
            cat <<'EOF'
Usage: bash scripts/train.sh [--gpus 1|2] [--smoke] [--epochs N]
                            [--workload NAME]...
Workloads: sage-arxiv, gcn-arxiv, gat-arxiv, sage-products
Default: all four workloads. --smoke defaults to sage-arxiv on a small graph.
When submitting with sbatch, match --gpus to the Slurm GPU allocation.
EOF
            exit 0 ;;
        *) fail "Unknown option: $1" ;;
    esac
done

[[ $gpus == 1 || $gpus == 2 ]] || fail 'Use --gpus 1 or --gpus 2.'
if [[ -z ${workloads[0]+set} ]]; then
    if (( smoke )); then
        workloads=(sage-arxiv)
    else
        workloads=(sage-arxiv gcn-arxiv gat-arxiv sage-products)
    fi
fi
seen=' '
for workload in "${workloads[@]}"; do
    case "$workload" in
        sage-arxiv|gcn-arxiv|gat-arxiv) ;;
        sage-products) (( smoke == 0 )) || fail 'Products has no smoke mode.' ;;
        *) fail "Unknown workload: $workload" ;;
    esac
    [[ $seen != *" $workload "* ]] || fail "Duplicate workload: $workload"
    seen+="$workload "
done
if (( smoke )); then
    extra+=(--smoke)
fi

repo=$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}" && pwd)
cd "$repo"
[[ -f models/train.py ]] || fail 'Submit from the root of the Maverick checkout.'

scratch="$repo/.maverick"
if [[ -n ${SLURM_JOB_ID:-} ]]; then
    scratch="${TMPDIR:-/tmp}/maverick-$SLURM_JOB_ID"
fi
data_dir=${MAVERICK_DATA_DIR:-"$scratch/data"}
runs=${MAVERICK_RUN_DIR:-"$repo/runs"}
mkdir -p "$scratch" "$data_dir" "$runs"

if [[ -n ${MAVERICK_PYTHON:-} ]]; then
    python=$MAVERICK_PYTHON
else
    venv=${MAVERICK_VENV:-"$scratch/venv"}
    python="$venv/bin/python"
    mkdir -p "$venv"
    (
        flock 9
        if [[ ! -f "$venv/.maverick-cu128" ]]; then
            export PIP_CACHE_DIR="$scratch/pip-cache"
            "${MAVERICK_BOOTSTRAP_PYTHON:-python3}" -m venv "$venv"
            "$python" -m pip install torch==2.8.0 \
                --index-url https://download.pytorch.org/whl/cu128
            "$python" -m pip install torch-geometric==2.7.0 ogb==1.3.6 \
                safetensors==0.6.2 packaging
            "$python" -m pip install --no-index --only-binary=:all: \
                --find-links https://data.pyg.org/whl/torch-2.8.0+cu128.html \
                pyg_lib==0.4.0+pt28cu128
            touch "$venv/.maverick-cu128"
        fi
    ) 9>"$venv/.install.lock"
fi

slots=$gpus
if (( slots > ${#workloads[@]} )); then
    slots=${#workloads[@]}
fi
"$python" - "$slots" <<'PY'
import sys

import torch

requested = int(sys.argv[1])
available = torch.cuda.device_count()
if available < requested:
    sys.exit(f"Requested {requested} CUDA GPUs, but only {available} are visible. "
             "Check the Slurm allocation or pass --gpus 1.")
for index in range(requested):
    print(f"cuda:{index}: {torch.cuda.get_device_name(index)}", flush=True)
PY

run=$(mktemp -d "$runs/${SLURM_JOB_ID:-local}-XXXXXXXX")
printf 'Run directory: %s\n' "$run"
export PYTHONUNBUFFERED=1
threads=$((${SLURM_CPUS_PER_TASK:-4} / slots))
export OMP_NUM_THREADS=$((threads > 0 ? threads : 1))

run_slot() {
    local slot=$1 index workload
    for (( index=slot; index<${#workloads[@]}; index+=slots )); do
        workload=${workloads[index]}
        printf 'Training %s on cuda:%s\n' "$workload" "$slot"
        "$python" -m models.train "$workload" --data-dir "$data_dir" \
            --output "$run/$workload" --device "cuda:$slot" \
            ${extra[@]+"${extra[@]}"} >"$run/$workload.log" 2>&1 || {
            tail -n 30 "$run/$workload.log" >&2
            return 1
        }
    done
}

pids=()
for (( slot=0; slot<slots; slot++ )); do
    run_slot "$slot" &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
(( failed == 0 )) || fail "Training failed. See logs in $run."
if (( smoke == 0 )); then
    touch "$run/.publishable"
fi
printf 'Training complete: %s\n' "$run"
