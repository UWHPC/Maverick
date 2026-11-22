# Edit and benchmark Inductor kernels

Use a Linux environment with an NVIDIA CUDA GPU, PyTorch 2.8.0, its matching
Triton, PyTorch Geometric, OGB, NumPy and Safetensors. The training launcher uses
this stack. Set `MAVERICK_VENV` to reuse the environment created by training or
capture. `MAVERICK_PYTHON` overrides the interpreter. For a step-by-step first
run on WATcloud, see the [WATcloud walkthrough](../../docs/watcloud.md).

For profiling, use Nsight Systems **2024.6.2.225** (`nsys`) for launch traces and
Nsight Compute **2025.1.0.14** (`ncu`) for optional kernel counters. Put the tools
you use on `PATH`. Both versions ship with CUDA Toolkit 12.8.0.
[NVIDIA CUDA 12.8 release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions).

## Capture an experiment

From the checkout root, use an existing checkpoint directory with
`model.safetensors` and `metrics.json`. Checkpoints can come from training or the
[optional weights submodule](../../models/README.md).

```sh
./src/benchmark.sh capture \
  --checkpoint-dir runs/REPLACE_WITH_RUN_ID/sage-arxiv \
  --data-dir /path/to/ogb-cache \
  --fixture arxiv-smoke \
  --out experiment.py
```

The [model README](../../models/README.md#dump-all-four-models-to-triton-on-watcloud)
shows how to capture all four checkpoints in one WATcloud job.

`products-250k` is a seeded, induced subgraph of `ogbn-products`. The checked-in
capture has 250,000 nodes and 1,289,372 edge entries, about five per node. Sampling
reduces connectivity. Label results as `products-250k` so they describe the
sampled workload. For that capture, two stock logits failed the eager comparison
under the default tolerances, with a maximum absolute difference of about
`2.44e-4`. The capture records `atol=3e-4` and the default `rtol=1e-4` so reruns
use the same correctness check.

Edit the function bodies in `experiment.py` and leave the functions you want to
substitute in the exported `KERNELS` list. Keep their names, signatures, and
`@triton.jit` decorators. The runner reads the file as source and uses each
function body inside the target's original Triton module. That module supplies
the imported helpers and launch configuration. The experiment stores the expected
checkpoint, graph, software versions, and original kernel fingerprints.

Capture exports supported Triton pointwise, reduction, and persistent-reduction
kernels. It prints skipped Triton kernel names and their heuristics and records
them in `<stem>.capture/skipped_kernels.json`. Inspect the `call` function in
`<stem>.capture/stock/output_code_*.py` for argument bindings and ATen dispatches,
such as GAT's `scatter_reduce_.two`. The four [checked-in captures](../captures/)
include their stock wrappers. A run saves the current device's wrapper under
`stock/output_code_*.py` in the results directory.

## Run the experiment

```sh
./src/benchmark.sh experiment.py
# Or submit the same launcher from the checkout root
sbatch src/benchmark.sh experiment.py
```

You can pass a [capture name](../README.md#edit-and-check-the-kernel) in place of
the `.py` path. Omitting the experiment uses `src/captures/sage-arxiv.py`.
The script uses the first visible CUDA device. The Slurm job requests one GPU.
Paths embedded during capture are relative to the experiment file. For a missing
checkpoint under `runs/`, the runner also checks the corresponding path under
`models/model_weights/runs/`. Set `MAVERICK_DATA_DIR` to reuse one OGB cache across
experiments.

The runner compares stock and edited outputs to eager inference before timing.
Each measurement worker also checks the same compiled instance after warm-up and
again after timing. A failed check suppresses speedup reporting.
Prepared input graphs are cached by their fingerprint under `.maverick-fixtures/`
in the OGB data directory. Subsequent workers reuse the cached inputs after
verifying the fingerprint.

The runner writes `summary.md`, raw samples in `report.json`, compiler dumps,
and effective edited sources to the results directory. Profiling runs separately
from timing and retains the profiler reports and logs.

The report keeps samples and medians for each stock, candidate, candidate, and
stock timing block. For wall and GPU time separately, it shows the absolute gap
between each variant's two medians as a percentage of that variant's mean. The
larger absolute gap in milliseconds is the noise floor. The report compares the
mean of the candidate medians with the mean of the stock medians. It labels the
result **within noise** if the difference is at or below the noise floor or the
stock-to-candidate ratio rounds to `1.000x`.

## What an experiment can change

Body edits use the existing COO edge list, argument layout, launch configuration,
and mutation contract. The current runner cannot supply CSR row offsets, change
launch-level work partitioning, or reorder edges before a kernel runs. A variant
that needs preprocessing would need a separate input preparation and compilation
path. That path would validate both variants against the same eager graph and
report preprocessing cost separately from steady-state inference.

`arxiv-full` makes node features contiguous before fingerprinting and caching
the graph. The feature strides are `(128, 1)`. Input strides are part of the graph
fingerprint, so column-major captures need to be regenerated. See the
[full-arXiv recapture instructions](../../models/README.md#refresh-the-full-arxiv-captures).
Capture a new experiment if the target emits a different kernel body or argument
layout.

## Command-line flags

| Flag | Command | Default | What it does |
| --- | --- | --- | --- |
| `-h`, `--help` | Launcher, `capture`, `run` | Off | Shows help for the launcher or the selected command, then exits. |
| `--checkpoint-dir DIR` | `capture`, `run` | Required for `capture`. `run` uses the experiment's recorded path. | Selects the directory containing `model.safetensors` and `metrics.json`. |
| `--data-dir DIR` | `capture`, `run` | Required for `capture`. `run` uses `MAVERICK_DATA_DIR` if set, or the experiment's recorded path otherwise. | Selects the OGB dataset and prepared-input cache directory. |
| `--fixture NAME` | `capture` | `arxiv-smoke` | Chooses the input graph. `arxiv-smoke` is a seeded 8,192-node arXiv subgraph. `arxiv-full` runs full arXiv with GraphSAGE, GCN, or GAT. `products-250k` runs a seeded 250,000-node Products subgraph with GraphSAGE. |
| `--out FILE` | `capture` | Required | Writes a new `.py` experiment file. The original Inductor wrapper is saved alongside it in `<stem>.capture/`. |
| `--atol FLOAT` | `capture`, `run` | `1e-5`. `run` uses the captured value when present. | Sets the absolute tolerance when comparing compiled logits to eager inference. |
| `--rtol FLOAT` | `capture`, `run` | `1e-4`. `run` uses the captured value when present. | Sets the relative tolerance when comparing compiled logits to eager inference. |
| `--output DIR` | `run` | A fresh directory under `benchmark-results/` | Writes results to the specified directory, which must not already exist. |
| `--check-only` | `run` | Off | Compiles stock and edited variants and checks their outputs against eager inference without timing or profiling. |
| `--warmup N` | `run` | `20` | Sets the number of warm-up inferences before each timing or profiling run. Use a positive integer. |
| `--samples N` | `run` | `30` | Sets the number of samples per metric in each timing block. Use a positive integer. |
| `--profile MODE` | `run` | `auto` | `auto` uses Nsight Systems if available. `nsys` selects Nsight Systems, `ncu` selects Nsight Compute for the kernels listed in `KERNELS`, `both` selects both tools, and `none` skips profiling. |
