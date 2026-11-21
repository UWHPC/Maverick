#!/usr/bin/env bash
#SBATCH --job-name=maverick-capture
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx_4090:1,tmpdisk:20480
#SBATCH --time=02:00:00
#SBATCH --output=maverick-capture-%j.log

set -euo pipefail

workloads=(sage-arxiv gcn-arxiv gat-arxiv)
if [[ ${1:-} == --arxiv-only ]]; then
    shift
else
    workloads+=(sage-products)
fi

if [[ ${1:-} == --help || ${1:-} == -h || $# -lt 2 || $# -gt 3 ]]; then
    printf 'Usage: %s [--arxiv-only] CHECKPOINT_RUN DATA_DIR [OUTPUT_DIR]\n' "$0"
    printf '%s\n' 'Captures sage-arxiv, gcn-arxiv, gat-arxiv and sage-products by default.'
    printf '%s\n' 'Use --arxiv-only to capture just the three arXiv models.'
    if [[ ${1:-} == --help || ${1:-} == -h ]]; then
        exit 0
    fi
    exit 2
fi

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ! -f "$repo/models/cuda-env.sh" ]]; then
    repo=${SLURM_SUBMIT_DIR:-$repo}
fi
if [[ ! -f "$repo/models/cuda-env.sh" || ! -x "$repo/src/benchmark.sh" ]]; then
    printf '%s\n' 'Submit the capture job from the Maverick checkout root.' >&2
    exit 1
fi

checkpoint=$1
data_dir=$2
for workload in "${workloads[@]}"; do
    for file in model.safetensors metrics.json; do
        if [[ ! -s "$checkpoint/$workload/$file" ]]; then
            printf 'Missing checkpoint file: %s\n' "$checkpoint/$workload/$file" >&2
            exit 1
        fi
    done
done

if [[ $# -eq 3 ]]; then
    output=$3
    mkdir -p "$output"
else
    mkdir -p "$repo/captures"
    output=$(mktemp -d "$repo/captures/${SLURM_JOB_ID:-local}-XXXXXXXX")
fi

scratch="$repo/.maverick"
if [[ -n ${SLURM_JOB_ID:-} ]]; then
    scratch="${TMPDIR:-/tmp}/maverick-$SLURM_JOB_ID"
fi
mkdir -p "$scratch" "$data_dir"
source "$repo/models/cuda-env.sh"
maverick_cuda_setup "$scratch"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

printf 'Capture directory: %s\n' "$output"
failed=0
for workload in "${workloads[@]}"; do
    fixture=arxiv-full
    if [[ $workload == sage-products ]]; then
        fixture=products-250k
    fi
    printf 'Capturing %s on %s\n' "$workload" "$fixture"
    if ! "$repo/src/benchmark.sh" capture --checkpoint-dir "$checkpoint/$workload" \
        --data-dir "$data_dir" --fixture "$fixture" --out "$output/$workload.py"; then
        printf 'Capture failed for %s\n' "$workload" >&2
        failed=1
    fi
done
(( failed == 0 )) || exit 1
printf 'Captured %s workloads: %s\n' "${#workloads[@]}" "$output"
