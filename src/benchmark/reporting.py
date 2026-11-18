"""Publish benchmark results and collect separate Nsight artifacts."""

import json
import math
import re
import shutil
import statistics
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


_NVTX_RANGE = "maverick_benchmark"
_NSYS_REPORTS = (
    "cuda_gpu_kern_sum",
    "cuda_api_sum",
    "cuda_api_trace",
    "cuda_gpu_trace",
    "cuda_kern_exec_trace",
)
_VARIANTS = ("stock", "candidate")
_TIMINGS = ("wall_ms", "gpu_ms")


def _diagnostics(result: subprocess.CompletedProcess) -> str:
    return "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())


def _nsys_exports(tool: str, report: Path, env: dict[str, str], prefix: Path) -> tuple[dict, dict]:
    artifacts = {}
    failures = {}
    for name in _NSYS_REPORTS:
        csv_prefix = report.parent / name
        command = [
            tool, "stats", "--report", name, "--format", "csv",
            "--output", str(csv_prefix), str(report),
        ]
        try:
            result = subprocess.run(command, env=env, capture_output=True, text=True, errors="replace")
        except OSError as error:
            failures[name] = {"details": str(error)}
            continue
        csv_path = next(report.parent.glob(f"{name}*.csv"), None)
        if result.returncode or csv_path is None or not csv_path.stat().st_size:
            failures[name] = {"returncode": result.returncode, "details": _diagnostics(result)}
            continue
        destination = prefix.parent / f"{prefix.name}.{name}.csv"
        csv_path.replace(destination)
        artifacts[name] = str(destination)
    return artifacts, failures


def launch_profile(
    mode: str, command: list[str], env: dict[str, str], prefix: Path, kernel_names: list[str]
) -> dict:
    """Collect one profile and record missing tools, permissions, or exports."""
    if mode not in {"nsys", "ncu"}:
        raise ValueError(f"Unknown profiler {mode}")

    outcome = {"mode": mode, "artifacts": {}}
    if mode == "ncu":
        outcome["kernel_names"] = list(kernel_names)
        if not kernel_names:
            return {**outcome, "status": "unavailable", "details": "No kernel names were selected."}
    tool = shutil.which(mode, path=env.get("PATH"))
    if not tool:
        return {**outcome, "status": "unavailable", "details": f"{mode} was not found on PATH."}

    prefix.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{mode}-", dir=prefix.parent) as directory:
        capture = Path(directory).resolve() / f"profile.{mode}-rep"
        if mode == "nsys":
            arguments = [
                "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
                "--capture-range=nvtx", "--capture-range-end=stop",
                f"--nvtx-capture={_NVTX_RANGE}", "--output", str(capture),
            ]
            env = {**env, "NSYS_NVTX_PROFILER_REGISTER_ONLY": "0"}
        else:
            names = "|".join(re.escape(name) for name in kernel_names)
            arguments = [
                "--nvtx", "--nvtx-include", f"{_NVTX_RANGE}/", "--kernel-name-base", "function",
                "--kernel-name", f"regex:^({names})$", "--export", str(capture),
            ]

        try:
            result = subprocess.run(
                [tool, *arguments, *command], env=env, capture_output=True, text=True, errors="replace"
            )
        except OSError as error:
            return {**outcome, "status": "failed", "details": str(error)}

        diagnostics = _diagnostics(result)
        log = prefix.parent / f"{prefix.name}.{mode}.log"
        log.write_text(diagnostics + "\n", encoding="utf-8")
        outcome["artifacts"]["log"] = str(log)
        outcome["returncode"] = result.returncode
        if capture.is_file() and capture.stat().st_size:
            if mode == "nsys" and result.returncode == 0:
                artifacts, failures = _nsys_exports(tool, capture, env, prefix)
                outcome["artifacts"].update(artifacts)
                if failures:
                    outcome["export_failures"] = failures
            destination = prefix.parent / f"{prefix.name}.{mode}-rep"
            capture.replace(destination)
            outcome["artifacts"]["report"] = str(destination)

        permission_denied = re.search(
            r"ERR_NVGPUCTRPERM|permission denied|operation not permitted|insufficient permissions",
            diagnostics,
            re.IGNORECASE,
        )
        if result.returncode or "report" not in outcome["artifacts"] or permission_denied:
            outcome["status"] = "failed"
            outcome["reason"] = "permissions" if permission_denied else "capture_failed"
            outcome["details"] = diagnostics or "The profiler did not produce a report."
        else:
            outcome["status"] = "ok"
        return outcome


def _summarize(samples: list[float]) -> dict | None:
    if not samples or any(
        isinstance(sample, bool) or not isinstance(sample, (int, float))
        or not math.isfinite(sample) or sample <= 0
        for sample in samples
    ):
        return None
    median = statistics.median(samples)
    ordered = sorted(samples)
    rank = (len(samples) - 1) * 0.95
    low, high = math.floor(rank), math.ceil(rank)
    p95 = ordered[low] + (ordered[high] - ordered[low]) * (rank - low)
    return {
        "count": len(samples),
        "median_ms": median,
        "p95_ms": p95,
        "inferences_per_second": 1000 / median,
    }


def _json_value(value: object) -> object:
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_report(output: Path, report: dict) -> None:
    """Write raw results and a summary of valid, warmed timings."""
    output.mkdir(parents=True, exist_ok=True)
    samples = report.get("samples", {})
    summary = {}
    for variant in _VARIANTS:
        variant_samples = samples.get(variant, {})
        summary[variant] = {
            timing: _summarize(variant_samples.get(timing, []))
            for timing in _TIMINGS
        }

    results = [summary[variant][timing] for variant in _VARIANTS for timing in _TIMINGS]
    comparable = (
        report.get("status") == "ok"
        and report.get("correctness", {}).get("passed") is True
        and all(results)
    )
    speedups = {}
    if comparable and len({result["count"] for result in results}) == 1:
        for timing in _TIMINGS:
            stock = summary["stock"][timing]
            candidate = summary["candidate"][timing]
            speedups[timing] = {
                "median_speedup": stock["median_ms"] / candidate["median_ms"],
                "p95_speedup": stock["p95_ms"] / candidate["p95_ms"],
            }
    summary["candidate_vs_stock"] = speedups
    report["summary"] = summary
    (output / "report.json").write_text(
        json.dumps(_json_value(report), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Benchmark summary",
        "",
        f"Run status **{report.get('status', 'unknown')}**",
        "",
        "[Raw samples and run metadata](report.json)",
        "",
        "| Variant | Timing | Samples | Median (ms) | p95 (ms) | Inferences/s |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for variant in _VARIANTS:
        for timing in _TIMINGS:
            values = summary[variant][timing]
            if values:
                count = values["count"]
                median = values["median_ms"]
                p95 = values["p95_ms"]
                throughput = values["inferences_per_second"]
                lines.append(
                    f"| {variant} | {timing[:-3]} | {count} | {median:.4f} | {p95:.4f} "
                    f"| {throughput:.2f} |"
                )
    lines.extend(["", "## Candidate speedup", ""])
    if speedups:
        for timing, speedup in speedups.items():
            lines.append(
                f"- {timing[:-3]} time, "
                f"{speedup['median_speedup']:.3f}x median and {speedup['p95_speedup']:.3f}x p95."
            )
    else:
        lines.append("A speedup requires a passing comparison and complete paired timing samples.")

    if any(variant in report for variant in _VARIANTS):
        lines.extend([
            "", "## Correctness and first call", "",
            "| Variant | Eager comparison | Max abs. error | Setup (s) | First call (s) | Triton tuning (s) |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ])
        for variant in _VARIANTS:
            stage = report.get(variant)
            if stage is None:
                continue
            correctness = stage["correctness"]
            passed = "pass" if correctness["passed"] else "fail"
            error = correctness["max_abs_error"]
            abs_error = f"{error:.4g}" if error is not None else "unavailable"
            lines.append(
                f"| {variant} | {passed} | {abs_error} | {stage['prepare_s']:.3f} "
                f"| {stage['first_compiled_call_s']:.3f} | {stage['triton_autotune_s']:.3f} |"
            )
    if report.get("error"):
        lines.extend(["", "## Error", "", report["error"]])

    profiles = report.get("profile", [])
    if profiles:
        lines.extend(["", "## Profiles", ""])
        for profile in profiles:
            variant = profile.get("variant", "")
            label = " ".join(part for part in (variant, profile["mode"]) if part)
            artifacts = profile.get("artifacts", {})
            reason = profile.get("reason")
            suffix = f" ({reason})" if reason else ""
            lines.append(f"- {label} **{profile['status']}**{suffix}")
            for name, artifact in artifacts.items():
                path = Path(artifact)
                try:
                    target = path.resolve().relative_to(output.resolve()).as_posix()
                except ValueError:
                    target = path.as_posix()
                lines.append(f"  - [{name}](<{target}>)")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
