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

Run correctness checks while iterating. From the repo root, pass `sage-arxiv`,
`gcn-arxiv`, `gat-arxiv`, or `sage-products` to use the matching file in
`src/captures/`. You can also pass the path to any experiment `.py` file. Omitting
the experiment selects `sage-arxiv`.

```sh
experiment=sage-arxiv
./src/benchmark.sh "$experiment" --check-only
```

On Slurm, use `sbatch src/benchmark.sh "$experiment" --check-only`.

## Decide settings first, then benchmark

Choose the workload and benchmark settings before looking at timings. Run one
benchmark of the finished idea, with at most one repeat to confirm the result.
Each run already includes two stock and two candidate timing blocks. Keep the
scope and settings fixed across runs. Use correctness checks for debugging and
avoid tuning against repeated measurements on the test set.

```sh
./src/benchmark.sh "$experiment"
```

On WATcloud, the launcher requests one full RTX 4090 and 20 GiB of job-local
storage by default.

```sh
sbatch src/benchmark.sh "$experiment"
```

To choose another GPU or storage size, pass `--gres=gpu:<type>:1,tmpdisk:<MiB>`
before the script path. Prefer a full GPU for timing to avoid contention.
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
