# Kernel development and testing

## Start with an issue

Use the issue for the kernel idea you want to test, or open one if it does not
exist. Describe the change and what you expect it to improve. Branch from the
latest `main`. Link your PR to that issue so reviewers can see what the benchmark
was meant to evaluate.

## Edit and check the kernel

Start from the relevant experiment in [captures/](captures/). If you need a new
capture or a CUDA environment, follow the [benchmark setup guide](benchmark/README.md).

Keep the change scoped to the idea in the issue. You can optimize the
implementation of that idea. Leave unrelated kernels, model code, and benchmark
settings alone, even if changing them would improve end-to-end latency. That
keeps the result attributable to the idea being tested.

Run correctness checks while iterating. From the repo root, substitute your
experiment file in the following commands.

```sh
experiment=src/captures/sage-arxiv.py
./src/benchmark.sh "$experiment" --check-only
```

On Slurm, use `sbatch src/benchmark.sh "$experiment" --check-only`.

## Benchmark once or twice

Choose the workload and benchmark settings before looking at timings. Run one
benchmark per idea on the finished implementation, with at most one repeat to
confirm the result. Keep the scope and settings fixed across both runs. Use
correctness checks for debugging and avoid tuning against repeated measurements
on the test set.

```sh
./src/benchmark.sh "$experiment"
```

On WATcloud, prefer one full GPU for timing so other jobs cannot share it. Pick
a GPU type with `gpu:<type>:1` and request enough job-local storage with
`tmpdisk` in MiB. This example requests one RTX 4090 and 20 GiB of temporary disk.

```sh
sbatch --gres=gpu:rtx_4090:1,tmpdisk:20480 src/benchmark.sh "$experiment"
```

Make sure your persistent storage also has room for the OGB cache, compiler
artifacts, and profiling traces. See the [WATcloud machine list](https://cloud.watonomous.ca/machines)
and [Slurm guide](https://cloud.watonomous.ca/docs/compute-cluster/slurm).

The launcher writes `summary.md` and `report.json` under `benchmark-results/`.

## What the PR must include

- A link to the issue for the kernel idea.
- Benchmark results for every run, including the workload, GPU, correctness
  outcome, and stock and candidate timings. Include regressions and link the
  generated report so reviewers can check the numbers.
- A recommendation to adopt, revise, or drop the idea, with a short reason based
  on the results.
