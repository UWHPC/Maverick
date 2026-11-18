# Maverick

Graph topology-aware GPU execution for PyTorch Geometric within the `torch.compile` workflow.

## Train the OGB checkpoints on WATcloud

SSH to a WATcloud login node, clone the GitHub repository without `--recurse-submodules`,
and submit from the checkout root. The default job requests two RTX 4090s on one node.
It trains GraphSAGE, GCN, and GAT on `ogbn-arxiv` and GraphSAGE on `ogbn-products`,
with one training process per GPU. It requires Linux x86-64, Python 3.10 through 3.13
with `venv`, and a driver compatible with CUDA 12.8.

```sh
git clone https://github.com/UWHPC/eignn.git maverick
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

Checkpoints, metrics, and per-model logs are written to `runs/` or `MAVERICK_RUN_DIR`.
Each checkpoint has a `model.safetensors` state dict and a `metrics.json` file with
the model configuration, seed, and OGB validation and test accuracy. The Products
trainer uses neighbor sampling and computes validation and test predictions with
all neighbors, one layer at a time.

[WATcloud Slurm guide](https://cloud.watonomous.ca/docs/compute-cluster/slurm) ·
[Machine inventory](https://cloud.watonomous.ca/machines)

## Optional weights submodule

The GitHub repository pins a revision of the [Hugging Face weights repository](https://huggingface.co/alonso130r/maverick).
A regular clone leaves `weights/` uninitialized. To fetch the pinned weights, install
Git LFS and Git Xet, run `git lfs install` and `git xet install`, then run
`bash scripts/weights.sh pull`.

To publish, authenticate Git with a Hugging Face credential that can write to the
weights repository. Pass the completed run directory printed by the launcher to
`bash scripts/weights.sh publish RUN_DIR`. The publisher stores only the Safetensors
checkpoints and metrics under `runs/<run-id>` on Hugging Face, using the Git Xet
transfer agent. After the upload succeeds, commit the updated `weights` gitlink in
the GitHub repository to make that revision available to other clones.

[Hugging Face Git Xet setup](https://huggingface.co/docs/hub/en/xet/using-xet-storage)
