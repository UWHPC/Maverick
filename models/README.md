# Model weights

The `models/model_weights` submodule pins a revision of the [Hugging Face weights repository](https://huggingface.co/alonso130r/maverick).
A regular clone leaves it uninitialized, which is enough for development without
checkpoints or training from scratch.

## Pull checkpoints

Install Git LFS and Git Xet, then run these commands from the Maverick checkout root.

```sh
git lfs install
git xet install
bash scripts/weights.sh pull
```

Published runs are stored under `models/model_weights/runs/<run-id>/`. Each workload
directory contains a `model.safetensors` state dict and a `metrics.json` file with
the model configuration, seed, and OGB validation and test accuracy.

## Train the OGB checkpoints on WATcloud

SSH to a WATcloud login node, clone the `vijay/train_models` branch, and submit from
the checkout root. The default job requests two RTX 4090s on one node.
It trains GraphSAGE, GCN, and GAT on `ogbn-arxiv` and GraphSAGE on `ogbn-products`,
with one training process per GPU. It requires Linux x86-64, Python 3.10 through 3.13
with `venv`, and a driver compatible with CUDA 12.8.

```sh
git clone --branch vijay/train_models --single-branch https://github.com/UWHPC/Maverick.git maverick
cd maverick
sbatch scripts/train.sh
```

Use `--nodelist` to target a node with both GPUs. WATcloud's public inventory lists
one RTX 4090 on `trpro-slurm2`. For a quick GraphSAGE run there, request one GPU shard.
This trains for three epochs on a seeded 8,192-node arXiv snapshot.

```sh
sbatch --nodelist=trpro-slurm2 --gres=shard:rtx_4090:4096,tmpdisk:20480 \
  --time=00:45:00 scripts/train.sh --gpus 1 --smoke
```

Pass `--workload sage-arxiv` (repeatable) to choose checkpoints, or `--epochs N` to
override the epoch count. On an allocated Linux GPU host, the same script also runs
with `bash scripts/train.sh`. The launcher provisions pinned CUDA/PyG dependencies
and downloads the selected official OGB datasets on first use. Slurm job-local
scratch is cleared at exit. Set `MAVERICK_DATA_DIR` and optionally `MAVERICK_VENV`
to user-owned paths on a WATcloud data mount to reuse downloads and dependencies.
`MAVERICK_PYTHON` selects an existing environment and skips installation.

The Products trainer uses neighbor sampling and computes validation and test
predictions with all neighbors, one layer at a time.

[WATcloud Slurm guide](https://cloud.watonomous.ca/docs/compute-cluster/slurm) ·
[Machine inventory](https://cloud.watonomous.ca/machines)

## Publish a training run

The training launcher writes checkpoints, metrics, and per-model logs to `runs/`
by default. Set `MAVERICK_RUN_DIR` to override that location.

Authenticate Git with a Hugging Face credential that can write to the weights
repository. Set `user.name` and `user.email` for Git on the publishing machine so
Git can create the weights commit. From the checkout root, pass the completed run
directory printed by the training launcher to `bash scripts/weights.sh publish RUN_DIR`. The helper
uploads the checkpoints and metrics using Git Xet. Commit the updated
`models/model_weights` gitlink in Maverick to pin the published revision for other
clones.

[Hugging Face Git Xet setup](https://huggingface.co/docs/hub/en/xet/using-xet-storage)

## Dump all four models to Triton on WATcloud

Submit from the checkout root on a login node. Choose a completed four-model
run under `runs/` or `models/model_weights/runs/`. Set the OGB cache and CUDA venv
to paths you can write on a persistent WATcloud data mount.

```sh
export MAVERICK_VENV=/mnt/wato-driveN/your-dir/maverick-venv
sbatch --nodelist=trpro-slurm2 models/capture.sh \
  runs/REPLACE_WITH_RUN_ID /mnt/wato-driveN/your-dir/ogb-cache
```

The job uses one RTX 4090 and prepares the CUDA environment if needed. It captures
the three arXiv models on the full arXiv graph and Products GraphSAGE on a fixed,
seeded 250,000-node induced subgraph. The log `maverick-capture-<job-id>.log` prints
the output directory under `captures/`. Each model gets an editable `.py` file and
a sibling `.capture/stock/` directory with the original Inductor wrapper. Pass an
output directory as the third positional argument to choose where they go.

Set `MAVERICK_PYTHON="$MAVERICK_VENV/bin/python"` in your environment before running
or submitting an edited capture. See the [benchmark guide](../src/benchmark/README.md)
for running the experiment and collecting traces.
