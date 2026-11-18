import hashlib
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

from .kernels import build_replacements, read_experiment

FIXTURES = {
    "arxiv-smoke": "ogbn-arxiv",
    "arxiv-full": "ogbn-arxiv",
    "products-250k": "ogbn-products",
}
NVTX_RANGE = "maverick_benchmark"


def write_json(path, contents):
    path.write_text(json.dumps(contents, indent=2, allow_nan=False) + "\n")


def products_snapshot(data):
    import torch
    from torch_geometric.utils import subgraph

    rng = torch.Generator().manual_seed(42)
    nodes = torch.randperm(data.num_nodes, generator=rng)[:250_000].sort().values
    edges, _ = subgraph(nodes, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
    return data.x[nodes], edges


def prepare_workload(checkpoint, data_dir, fixture):
    import torch

    if torch.__version__.split("+")[0] != "2.8.0":
        raise RuntimeError(f"The Inductor adapter requires PyTorch 2.8.0, got {torch.__version__}")
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("An NVIDIA CUDA GPU and CUDA-enabled PyTorch are required")

    import torch_geometric
    import triton
    from safetensors.torch import load_file
    from torch_geometric.utils import to_undirected

    from models.gnn import ModelConfig, NodeClassifier
    from models.train import arxiv_snapshot, load_dataset

    weights = checkpoint / "model.safetensors"
    metrics = json.loads((checkpoint / "metrics.json").read_text())
    dataset = FIXTURES.get(fixture)
    if dataset is None or metrics["dataset"] != dataset:
        raise ValueError(f"Fixture {fixture} is incompatible with {metrics['dataset']}")
    if dataset == "ogbn-arxiv" and not metrics["symmetrized"]:
        raise ValueError("Expected a checkpoint trained with symmetrized arXiv edges")
    checkpoint_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
    config = ModelConfig(**metrics["model"])
    model = NodeClassifier(config).eval()
    model.load_state_dict(load_file(weights), strict=True)
    data, splits, _ = load_dataset(dataset, data_dir)
    if fixture == "arxiv-smoke":
        data, _ = arxiv_snapshot(data, splits, metrics["seed"])
    if fixture == "products-250k":
        features, edge_index = products_snapshot(data)
    else:
        features = data.x
        edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    del data
    num_nodes = features.size(0)
    edges, edge_weight = model.prepare_edges(edge_index, num_nodes)
    cpu_inputs = (features.float(), edges, edge_weight)
    digest = hashlib.sha256()
    for tensor in cpu_inputs:
        if tensor is None:
            digest.update(b"None")
        else:
            digest.update(repr((tensor.shape, tensor.dtype, tensor.stride())).encode())
            digest.update(memoryview(tensor.contiguous().numpy()).cast("B"))

    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = model.to("cuda:0")
    inputs = tuple(
        tensor.to("cuda:0") if tensor is not None else None for tensor in cpu_inputs
    )
    torch.cuda.synchronize()
    identity = {
        "fixture": fixture,
        "architecture": config.architecture,
        "dataset": dataset,
        "nodes": num_nodes,
        "edges": edges.size(1),
        "model": metrics["model"],
        "checkpoint_sha256": checkpoint_hash,
        "graph_sha256": digest.hexdigest(),
        "versions": {
            "torch": torch.__version__,
            "pyg": torch_geometric.__version__,
            "triton": triton.__version__,
            "cuda": torch.version.cuda,
        },
    }
    return model, inputs, identity


def gpu_identity(torch):
    properties = torch.cuda.get_device_properties(0)
    gpu = {
        "hostname": socket.gethostname(),
        "name": properties.name,
        "capability": list(torch.cuda.get_device_capability(0)),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "uuid": str(getattr(properties, "uuid", "")) or None,
    }
    try:
        # Use the CUDA-to-NVML mapping for Slurm and CUDA_VISIBLE_DEVICES.
        index = torch.cuda._get_nvml_device_index(0)
        command = [
            "nvidia-smi", "--id", str(index), "--query-gpu=uuid,driver_version",
            "--format=csv,noheader",
        ]
        details = subprocess.check_output(command, text=True, timeout=5).strip()
        uuid, driver = details.split(", ")
        gpu["uuid"] = gpu["uuid"] or uuid
        gpu["driver"] = driver
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        pass
    return gpu


def compile_model(model, inputs, replacements, output):
    import torch
    from torch._dynamo.utils import compile_times
    from torch._functorch import config as functorch
    from torch._inductor import config as inductor
    from torch._inductor.async_compile import AsyncCompile
    from torch._inductor.graph import GraphLowering
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    wrappers = []
    hits = {}
    original_compile = AsyncCompile.triton
    original_autotune = CachingAutotuner.autotune_to_one_config
    autotune_s = 0.0
    options = {
        "compile_threads": 1,
        "fx_graph_cache": False,
        "fx_graph_remote_cache": False,
        "autotune_remote_cache": False,
        "bundled_autotune_remote_cache": False,
        "triton.cudagraphs": False,
        "triton.unique_kernel_names": True,
        "triton.autotune_at_compile_time": False,
    }

    def compile_kernel(compiler, kernel_name, source_code, device_str="cuda"):
        key = (kernel_name, source_code)
        if key in replacements:
            hits[key] = hits.get(key, 0) + 1
            source_code = replacements[key]
        return original_compile(compiler, kernel_name, source_code, device_str)

    def time_autotune(tuner, *args, **kwargs):
        nonlocal autotune_s
        start = time.perf_counter()
        try:
            return original_autotune(tuner, *args, **kwargs)
        finally:
            autotune_s += time.perf_counter() - start

    start = time.perf_counter()
    try:
        with (
            inductor.patch(options),
            functorch.patch(enable_autograd_cache=False, enable_remote_autograd_cache=False),
            patch.object(GraphLowering, "save_output_code", wrappers.append),
            patch.object(
                AsyncCompile, "triton", compile_kernel if replacements else original_compile
            ),
            patch.object(CachingAutotuner, "autotune_to_one_config", time_autotune),
        ):
            compiled = torch.compile(model, backend="inductor", fullgraph=True, dynamic=False)
            logits = compiled(*inputs)
            torch.cuda.synchronize()
            first_call_s = time.perf_counter() - start
    finally:
        for index, code in enumerate(wrappers):
            (output / f"output_code_{index}.py").write_text(code)
        for index, key in enumerate(hits):
            (output / f"effective_kernel_{index}.py").write_text(replacements[key])
        write_json(output / "substitutions.json", [
            {"name": name, "count": count} for (name, _), count in hits.items()
        ])

    if not wrappers:
        raise RuntimeError("Inductor did not emit a Python wrapper")
    missed = [name for (name, source) in replacements if (name, source) not in hits]
    if missed:
        raise RuntimeError(f"Candidate compilation missed kernel replacements: {', '.join(missed)}")
    phases, durations = compile_times(repr="csv", aggregate=True)
    timings = {
        "first_compiled_call_s": first_call_s,
        "triton_autotune_s": autotune_s,
        "compile_phases_s": {
            name: float(seconds) for name, seconds in zip(phases, durations, strict=True)
        },
    }
    return compiled, logits, timings, [name for name, _ in hits]


def check_correctness(torch, actual, expected, atol, rtol):
    finite = torch.isfinite(expected) & torch.isfinite(actual)
    difference = (actual - expected).abs()
    close = finite & (difference <= atol + rtol * expected.abs())
    max_error = torch.where(finite, difference, 0).amax().item()
    return {
        "passed": bool(close.all().item()),
        "max_abs_error": max_error if max_error < float("inf") else None,
        "outside_tolerance": int((~close).sum().item()),
        "nonfinite_expected": int((~torch.isfinite(expected)).sum().item()),
        "nonfinite_actual": int((~torch.isfinite(actual)).sum().item()),
        "atol": atol,
        "rtol": rtol,
    }


def measure(torch, compiled, inputs, count):
    gpu, wall = [], []
    for _ in range(count):
        start = time.perf_counter()
        compiled(*inputs)
        torch.cuda.synchronize()
        wall.append((time.perf_counter() - start) * 1000)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(count):
        start_event.record()
        compiled(*inputs)
        end_event.record()
        end_event.synchronize()
        gpu.append(start_event.elapsed_time(end_event))
    return {"gpu_ms": gpu, "wall_ms": wall}


def worker(request):
    import torch

    output = Path(request["output"])
    output.mkdir(parents=True, exist_ok=True)
    mode = request["mode"]
    start = time.perf_counter()
    model, inputs, identity = prepare_workload(
        Path(request["checkpoint_dir"]), Path(request["data_dir"]), request["fixture"]
    )
    prep_s = time.perf_counter() - start
    expected = request.get("expected_workload")
    if expected is not None:
        different = [key for key, value in identity.items() if expected.get(key) != value]
        if different:
            raise ValueError(f"Experiment and current workload differ in {', '.join(different)}")

    replacements = {}
    if request["variant"] == "candidate":
        _, edits = read_experiment(Path(request["experiment"]))
        stock_dir = Path(request["stock_dir"])
        wrappers = [path.read_text() for path in sorted(stock_dir.glob("output_code_*.py"))]
        replacements = build_replacements(wrappers, edits)

    gpu = gpu_identity(torch)
    if request.get("expected_gpu") not in (None, gpu):
        raise RuntimeError("Benchmark workers are running on different GPU devices")
    with torch.inference_mode():
        if mode in {"capture", "validate"}:
            eager = model(*inputs)
            torch.cuda.synchronize()
        compiled, logits, timings, names = compile_model(model, inputs, replacements, output)
        result = {
            "workload": identity, "gpu": gpu, "prepare_s": prep_s,
            **timings, "kernel_names": names,
        }
        if mode in {"capture", "validate"}:
            result["correctness"] = check_correctness(
                torch, logits, eager, request["atol"], request["rtol"]
            )
        else:
            for _ in range(request["warmup"]):
                compiled(*inputs)
            torch.cuda.synchronize()
            if mode == "measure":
                result["samples"] = measure(torch, compiled, inputs, request["samples"])
            else:
                with torch.cuda.nvtx.range(NVTX_RANGE):
                    for _ in range(request["samples"]):
                        compiled(*inputs)
                    torch.cuda.synchronize()
    return result
