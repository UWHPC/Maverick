import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from tempfile import mkdtemp

from .driver import FIXTURES, worker, write_json
from .kernels import read_experiment, write_experiment
from .reporting import launch_profile, write_report

DEFAULT_TOLERANCES = {"atol": 1e-5, "rtol": 1e-4}
REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_tolerances(args, captured=None):
    if captured is None:
        captured = {}
    if not isinstance(captured, dict):
        raise ValueError("capture_tolerance must be a dictionary")
    tolerances = {}
    for name, default in DEFAULT_TOLERANCES.items():
        value = getattr(args, name)
        if value is None:
            value = captured.get(name, default)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{name} must be a finite, nonnegative number")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite, nonnegative number")
        tolerances[name] = value
    return tolerances


def resolve_checkpoint(experiment, workload, override):
    if override is not None:
        candidates = [override.resolve()]
    else:
        recorded = (experiment.parent / workload["checkpoint_dir"]).resolve()
        candidates = [recorded]
        training_runs = REPO_ROOT / "runs"
        published_runs = REPO_ROOT / "models" / "model_weights" / "runs"
        if recorded.is_relative_to(training_runs):
            candidates.append(published_runs / recorded.relative_to(training_runs))
        elif recorded.is_relative_to(published_runs):
            candidates.append(training_runs / recorded.relative_to(published_runs))

    for directory in candidates:
        if all((directory / name).is_file() for name in ("model.safetensors", "metrics.json")):
            return directory
    searched = ", ".join(map(str, candidates))
    raise ValueError(f"Checkpoint files are missing in {searched}. Pull the weights or use --checkpoint-dir.")


def worker_command(output, cache, request):
    output.mkdir(parents=True, exist_ok=True)
    (cache / "tmp").mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "TMPDIR": str(cache / "tmp"),
    }
    job = output / "request.json"
    result = output / "result.json"
    write_json(job, {**request, "output": str(output)})
    command = [sys.executable, "-m", "src.benchmark", "_worker", str(job), str(result)]
    return command, environment


def run_worker(output, cache, request):
    command, environment = worker_command(output, cache, request)
    print(f"{request['mode']} {request['variant']}: {output}", flush=True)
    log = output / "worker.log"
    with log.open("w") as stream:
        completed = subprocess.run(command, env=environment, stdout=stream,
                                   stderr=subprocess.STDOUT)
    if completed.returncode:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-8:])
        raise RuntimeError(f"{request['variant']} worker failed. See {log}\n{tail}")
    return json.loads((output / "result.json").read_text())


def capture(args):
    experiment = args.out.resolve()
    if experiment.suffix != ".py":
        raise ValueError("Use a .py output file for the experiment")
    if experiment.exists():
        raise ValueError(f"Experiment already exists: {experiment}")
    tolerances = resolve_tolerances(args)
    output = experiment.with_suffix(".capture")
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_dir, data_dir = args.checkpoint_dir.resolve(), args.data_dir.resolve()
    request = {
        "mode": "capture",
        "variant": "stock",
        "checkpoint_dir": str(checkpoint_dir),
        "data_dir": str(data_dir),
        "fixture": args.fixture,
        **tolerances,
    }
    stock = output / "stock"
    result = run_worker(stock, output / "cache", request)
    if not result["correctness"]["passed"]:
        raise ValueError(f"Stock logits failed the eager comparison. See {stock / 'result.json'}")
    workload = {
        **result["workload"],
        "checkpoint_dir": os.path.relpath(checkpoint_dir, experiment.parent),
        "data_dir": os.path.relpath(data_dir, experiment.parent),
        "capture_tolerance": tolerances,
    }
    wrappers = [path.read_text() for path in sorted(stock.glob("output_code_*.py"))]
    count = write_experiment(experiment, workload, wrappers)
    if not count:
        raise ValueError(f"Inductor produced no supported Triton kernels. See {stock}")
    print(f"Exported {count} kernels to {experiment}")


def compare(args):
    experiment = args.experiment.resolve()
    if experiment.suffix != ".py":
        raise ValueError("Pass a .py experiment file")
    workload, edits = read_experiment(experiment)
    if not edits:
        raise ValueError("Select at least one kernel in KERNELS")
    fixture = workload.get("fixture")
    if fixture not in FIXTURES:
        raise ValueError(f"Choose a graph fixture from {', '.join(FIXTURES)}")
    checkpoint_dir = resolve_checkpoint(experiment, workload, args.checkpoint_dir)
    data_dir = args.data_dir
    if data_dir is None:
        cache = os.environ.get("MAVERICK_DATA_DIR")
        data_dir = Path(cache) if cache else experiment.parent / workload["data_dir"]
    request = {
        "experiment": str(experiment),
        "fixture": fixture,
        "checkpoint_dir": str(checkpoint_dir),
        "data_dir": str(data_dir.resolve()),
        "expected_workload": workload,
        **resolve_tolerances(args, workload.get("capture_tolerance")),
        "warmup": args.warmup,
        "samples": args.samples,
    }

    if args.output:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        root = Path("benchmark-results").resolve()
        root.mkdir(exist_ok=True)
        output = Path(mkdtemp(prefix=f"{experiment.stem}-", dir=root))
    print(f"Benchmark results: {output}", flush=True)
    (output / "experiment.py").write_bytes(experiment.read_bytes())
    request["experiment"] = str(output / "experiment.py")
    stock_dir = output / "stock"
    request["stock_dir"] = str(stock_dir)
    variants = ("stock", "candidate")
    report = {
        "status": "failed",
        "workload": workload,
        "selected_kernels": list(edits),
        "profile": [],
        "timing_blocks": [],
        "samples": {variant: {"wall_ms": [], "gpu_ms": []} for variant in variants},
        "correctness": {"passed": False},
    }

    try:
        for variant in variants:
            result = run_worker(output / variant, output / "cache" / variant,
                                {**request, "mode": "validate", "variant": variant})
            report[variant] = result
            request["expected_gpu"] = result["gpu"]
            if not result["correctness"]["passed"]:
                raise ValueError(f"{variant} logits failed the eager comparison")
        report["correctness"]["passed"] = True

        for index, variant in enumerate((*variants, *reversed(variants)), start=1):
            result = run_worker(output / f"timing-{index}-{variant}", output / "cache" / variant,
                                {**request, "mode": "measure", "variant": variant})
            report["timing_blocks"].append({"index": index, "variant": variant})
            for timing, samples in result["samples"].items():
                report["samples"][variant][timing].extend(samples)
        report["status"] = "ok"

        profiles = {"auto": ("nsys",), "both": ("nsys", "ncu"), "none": ()}
        for mode in profiles.get(args.profile, (args.profile,)):
            for variant in variants:
                prefix = output / "profiles" / f"{variant}-{mode}"
                command, env = worker_command(prefix, output / "cache" / variant, {
                    **request, "mode": "profile", "variant": variant,
                    "samples": 1 if mode == "ncu" else 5,
                })
                profile = launch_profile(mode, command, env, prefix,
                                         report["candidate"]["kernel_names"])
                report["profile"].append({"variant": variant, **profile})
    except (OSError, RuntimeError, ValueError) as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        write_report(output, report)
    print(f"Summary: {output / 'summary.md'}")


def main():
    parser = argparse.ArgumentParser(description="Compare edited Triton with stock Inductor")
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser("capture", help="create an editable Python experiment")
    export.add_argument("--checkpoint-dir", type=Path, required=True)
    export.add_argument("--data-dir", type=Path, required=True)
    export.add_argument("--fixture", choices=FIXTURES, default="arxiv-smoke")
    export.add_argument("--out", type=Path, required=True)

    run = commands.add_parser("run", help="compare an experiment against stock Inductor")
    run.add_argument("experiment", type=Path)
    run.add_argument("--checkpoint-dir", type=Path)
    run.add_argument("--data-dir", type=Path)
    run.add_argument("--output", type=Path)
    run.add_argument("--warmup", type=int, default=20)
    run.add_argument("--samples", type=int, default=30, help="samples per timing block")
    run.add_argument("--profile", choices=("auto", "none", "nsys", "ncu", "both"), default="auto")
    for command in (export, run):
        command.add_argument("--atol", type=float)
        command.add_argument("--rtol", type=float)

    internal = commands.add_parser("_worker")
    internal.add_argument("request", type=Path)
    internal.add_argument("result", type=Path)
    args = parser.parse_args()

    if args.command == "_worker":
        write_json(args.result, worker(json.loads(args.request.read_text())))
        return
    if args.command == "run" and (args.warmup < 1 or args.samples < 1):
        parser.error("warmup and sample counts must be positive")
    try:
        (capture if args.command == "capture" else compare)(args)
    except (KeyError, OSError, RuntimeError, SyntaxError, ValueError) as error:
        parser.exit(1, f"Benchmark failed: {error}\n")


if __name__ == "__main__":
    main()
