#!/usr/bin/env bash
set -euo pipefail

fail() { printf '%s\n' "$*" >&2; exit 1; }

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
submodule=models/model_weights
case "${1:-}" in
    pull|publish) ;;
    *) printf 'Usage: bash scripts/weights.sh pull | publish RUN_DIR\n'; exit 2 ;;
esac
git xet --version >/dev/null || fail 'Install Git Xet and run git xet install first.'
if [[ $1 == pull ]]; then
    git -C "$repo" submodule update --init -- "$submodule"
    exit
fi

run=${2:?Specify the completed training run directory}
[[ -f "$run/.publishable" ]] || fail 'Use a completed non-smoke training run.'
name=$(basename "$(cd "$run" && pwd)")
[[ $name =~ ^[a-zA-Z0-9_-]+$ ]] || fail 'Invalid training run directory name.'

weights="$repo/$submodule"
if [[ ! -e "$weights/.git" ]]; then
    git -C "$repo" submodule update --init -- "$submodule"
fi
status=$(git -C "$weights" status --porcelain)
if [[ -n $status ]]; then
    printf '%s\n' "$status" >&2
    fail "The weights submodule has local changes in $submodule. Review the paths above."
fi
if ! git -C "$weights" var GIT_AUTHOR_IDENT >/dev/null 2>&1 ||
    ! git -C "$weights" var GIT_COMMITTER_IDENT >/dev/null 2>&1; then
    fail "Set Git user.name and user.email in $submodule before publishing."
fi
git -C "$weights" fetch origin main
git -C "$weights" merge-base --is-ancestor HEAD FETCH_HEAD \
    || fail 'The weights submodule contains an unpublished commit. Review it before retrying.'
git -C "$weights" switch --detach FETCH_HEAD
destination="runs/$name"
[[ ! -e "$weights/$destination" ]] || fail "Run $name is already published."

workloads=()
for workload in sage-arxiv gcn-arxiv gat-arxiv sage-products; do
    [[ -d "$run/$workload" ]] || continue
    [[ -f "$run/$workload/model.safetensors" && -f "$run/$workload/metrics.json" ]] \
        || fail "Missing checkpoint or metrics for $workload."
    workloads+=("$workload")
done
[[ -n ${workloads[0]+set} ]] || fail 'No checkpoints found.'

xet_config=$(git -C "$weights" config --get-regexp '^lfs\.' || true)
[[ $xet_config == *xet* ]] || fail 'Run git xet install to enable the Xet transfer agent.'
git -C "$weights" lfs install --local
for workload in "${workloads[@]}"; do
    filter=$(git -C "$weights" check-attr filter -- \
        "$destination/$workload/model.safetensors")
    [[ $filter == *': filter: lfs' ]] \
        || fail 'Enable LFS pointer tracking for *.safetensors on the Xet-backed repository.'
done
for workload in "${workloads[@]}"; do
    mkdir -p "$weights/$destination/$workload"
    cp "$run/$workload/model.safetensors" "$run/$workload/metrics.json" \
        "$weights/$destination/$workload/"
done

git -C "$weights" add -- "$destination"
for workload in "${workloads[@]}"; do
    git -C "$weights" show ":$destination/$workload/model.safetensors" \
        | git lfs pointer --check --stdin || fail "No weight pointer staged for $workload."
done
git -C "$weights" commit -m "Add Maverick checkpoints for $name"
git -C "$weights" push origin HEAD:refs/heads/main
printf 'Published %s. Stage %s in Maverick to pin this revision.\n' "$name" "$submodule"
