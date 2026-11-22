# Run Maverick on WATcloud

This walks through a first benchmark on WATcloud, from an empty home directory to
a profiled result. It assumes you have a WATcloud account and can already SSH to
a login node. For account and SSH setup, see the
[WATcloud Slurm guide](https://cloud.watonomous.ca/docs/compute-cluster/slurm).

The steps were checked on a clean checkout with one RTX 4090. For what the
benchmark measures and every command-line flag, see the
[benchmark guide](../src/benchmark/README.md). For the workflow around a kernel
change, see the [development instructions](../src/README.md).

## 1. Connect

WATcloud's login nodes are only reachable from the campus network. Off campus,
connect to the University of Waterloo VPN first. Without it, SSH times out and
gives no other error.

## 2. Get the code and the model weights

```sh
git clone https://github.com/UWHPC/Maverick.git maverick
cd maverick
bash scripts/weights.sh pull
```

The last command downloads the trained checkpoints into
`models/model_weights/runs/`. It needs Git LFS, which WATcloud provides.

## 3. Build the Python environment once

Choose a folder that persists between jobs, such as one in your home directory.
The environment is about 7 GB and takes a few minutes to build. Run this on the
login node from the checkout root.

```sh
export MAVERICK_VENV=$HOME/maverick-env/venv
source models/cuda-env.sh
maverick_cuda_setup $HOME/maverick-env
```

This installs the pinned PyTorch 2.8.0, Triton, PyTorch Geometric and OGB. It
does nothing if the environment already exists.

## 4. Set the paths for each session

Add these to your shell profile, or run them after each login. `sbatch` passes
them into the job.

```sh
export MAVERICK_VENV=$HOME/maverick-env/venv
export MAVERICK_DATA_DIR=$HOME/maverick-env/data
```

`MAVERICK_DATA_DIR` holds the downloaded datasets. The first job that needs
`ogbn-arxiv` downloads it there, about 300 MB. `sage-products` downloads
`ogbn-products`, which is several gigabytes.

## 5. Check that everything works

Submit from the checkout root.

```sh
sbatch src/benchmark.sh sage-arxiv --check-only
```

This compiles the model, runs the unedited kernels, and compares the output with
eager PyTorch. It takes about two minutes on first use, including the dataset
download. Useful commands while it runs:

```sh
squeue -u $USER          # your queued and running jobs
tail -f slurm-<jobid>.out  # the job's log, written where you ran sbatch
scancel <jobid>          # stop a job
```

A successful job ends with a `Summary:` line that names a folder under
`benchmark-results/`.

## 6. Run a benchmark

```sh
sbatch src/benchmark.sh sage-arxiv
```

The job requests one full RTX 4090. Results go to
`benchmark-results/<experiment>-<id>/`. Open `summary.md` there first. It lists
the median time for each of the four timing blocks, the comparison between stock
and candidate, and the correctness check. `report.json` holds the raw samples.

Run this once before you change anything. The experiment file is unedited, so
stock and candidate are the same code and the result should read **within noise**.

## 7. Profile

The profilers are not on `PATH` by default. WATcloud mounts the Compute Canada
software tree on its compute nodes, which includes them. Run these before
`sbatch`.

```sh
C=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/12.9.1
export PATH=$C/nsight-systems-2025.1.3/bin:$PATH
```

A benchmark then also records an Nsight Systems profile for each variant. The
file to read first is `profiles/stock-nsys.cuda_gpu_kern_sum.csv`, which lists
each GPU kernel with its share of the time.

Nsight Compute (`--profile ncu`) does not work on WATcloud at the moment. It
fails with `ERR_NVGPUCTRPERM` because GPU performance counters are restricted to
administrators.

## 8. Change a kernel

Edit the function bodies in `src/captures/sage-arxiv.py`, then repeat steps 5 and
6. The [benchmark guide](../src/benchmark/README.md#what-an-experiment-can-change)
describes what an edit may and may not change.

## Benchmark well

- **Check the baseline first.** If the unedited experiment does not come out
  within noise, something in the setup is wrong. Fix that before measuring a
  change.
- **Correctness before speed.** Use `--check-only` while you iterate. A faster
  kernel that fails the eager comparison is not a result.
- **Decide the settings before you look at timings.** Pick the model, graph and
  flags, run once, and repeat at most once to confirm. Rerunning until the number
  looks good measures your patience, not your kernel.
- **Expect noise on the shared 4090.** Other jobs run on the same node. Repeat
  blocks of identical code have differed by up to about 5% there. Treat a
  difference of that size as no difference, even when the report labels it a
  speedup or a regression.
- **Report everything.** Include regressions and results that showed nothing.
  They tell the next person what not to try.

## Do not micro-optimise

- **Find where the time goes before editing.** Profile the unedited model and
  read the kernel summary. Work on the kernels at the top of that list. Speeding
  up one that takes 3% of the run cannot change the total by more than 3%.
- **Know what you cannot change here.** The matrix multiplies are library calls.
  An experiment can edit kernel bodies only.
- **Aim above the noise.** A change worth 1 or 2% cannot be told apart from noise
  on this hardware, so it cannot be shown to work. Look for changes to how the
  work is organised, such as what is read, in what order, and how often, over
  rewording single lines.
- **Measure the whole model.** A kernel that gets faster while the full forward
  pass does not has not helped.

## When something fails

| Message or symptom | Likely cause |
| --- | --- |
| SSH times out | Not on the campus network or VPN |
| `Checkpoint files are missing in ...` | The weights were not pulled. See step 2 |
| `Missing Python in MAVERICK_VENV` | The environment in step 3 was not built, or `MAVERICK_VENV` points elsewhere |
| `Experiment and current workload differ in graph_sha256` | The experiment was captured from a different graph. Recapture it as the benchmark guide describes |
| Profiles listed as **unavailable** | The profiler is not on `PATH`. See step 7 |
| `ERR_NVGPUCTRPERM` | Nsight Compute needs counter access that WATcloud does not grant |
