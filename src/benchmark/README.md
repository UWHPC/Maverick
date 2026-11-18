# Edit and benchmark Inductor kernels

Use a Linux environment with an NVIDIA CUDA GPU, PyTorch 2.8.0, its matching
Triton, PyTorch Geometric, OGB, NumPy and Safetensors. The training launcher uses
this stack. Set `MAVERICK_PYTHON` to the path to the environment's Python if
`python3` points elsewhere. Install Nsight Systems for launch traces and Nsight
Compute for optional kernel counters.

## Capture an experiment

From the checkout root, use an existing `ogbn-arxiv` checkpoint directory with
`model.safetensors` and `metrics.json`. Checkpoints can come from training or the
[optional weights submodule](../../models/README.md).

```sh
./src/benchmark.sh capture \
  --checkpoint-dir runs/REPLACE_WITH_RUN_ID/sage-arxiv \
  --data-dir /path/to/ogb-cache \
  --fixture arxiv-smoke \
  --out experiment.py
```

`arxiv-smoke` uses a seeded 8,192-node induced graph. Use `arxiv-full` for the full
graph or select an arXiv GCN or GAT checkpoint. Capture saves the original Inductor
wrapper next to the experiment in `experiment.capture/`.

Edit the function bodies in `experiment.py` and leave the functions you want to
substitute in the exported `KERNELS` list. Keep their names, signatures, and
`@triton.jit` decorators. The runner reads the file as source and uses each
function body inside the target's original Triton module. That module supplies
the imported helpers and launch configuration. The experiment stores the expected
checkpoint, graph, software versions, and original kernel fingerprints.

## Run the experiment

```sh
./src/benchmark.sh experiment.py
# Or submit the same launcher from the checkout root
sbatch src/benchmark.sh experiment.py
```

The script uses the first visible CUDA device. The Slurm job requests one GPU.
Use `--checkpoint-dir` or `--data-dir` after the experiment path if either location
changed. Paths embedded during capture are relative to the experiment file.

The runner compares stock and edited outputs to eager inference before timing.
It writes `summary.md`, raw samples in `report.json`, compiler dumps, and effective
edited sources under a fresh directory in `benchmark-results/`. Pass `--output`
to choose a different new directory. The default profiler is Nsight Systems when
available. Pass `--profile ncu`, `--profile both`, or `--profile none` to change it.
Profiles run separately from timing and retain the profiler reports and logs.

A body edit must preserve its original launch and mutation contract. If a target
emits a different kernel body or argument layout, capture a new experiment on
that target. The benchmark uses fixed, full-graph arXiv inference fixtures.
