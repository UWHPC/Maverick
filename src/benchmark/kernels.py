"""Read and patch ordinary PyTorch 2.8 Inductor Triton kernels."""

import ast
import hashlib
import keyword
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from textwrap import dedent, indent


@dataclass(frozen=True)
class Edit:
    fingerprint: str
    signature: str
    body: str


@dataclass(frozen=True)
class _Kernel:
    name: str
    compile_name: str
    source: str
    function: ast.FunctionDef
    fingerprint: str


def _bounds(source, function):
    """Return whole-line bounds for an Inductor-style multiline function body."""
    first = function.body[0]
    start = (
        min(
            [first.lineno]
            + [node.lineno for node in getattr(first, "decorator_list", [])]
        )
        - 1
    )
    if start <= function.lineno - 1:
        raise ValueError(f"Use a multiline body for {function.name}")
    lines = source.splitlines(keepends=True)
    return lines, start, function.end_lineno


def _body(source, function):
    lines, start, end = _bounds(source, function)
    return dedent("".join(lines[start:end])).rstrip("\n") + "\n"


def _signature(function):
    return repr(
        (
            ast.unparse(function.args),
            ast.unparse(function.returns) if function.returns else None,
            function.type_comment,
        )
    )


def _kernels(wrappers):
    seen = set()
    for wrapper in wrappers:
        for statement in ast.walk(ast.parse(wrapper)):
            if not (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                continue
            call = statement.value
            if not (
                isinstance(call, ast.Call)
                and ast.unparse(call.func) == "async_compile.triton"
                and len(call.args) >= 2
            ):
                continue
            name, source = call.args[:2]
            if not all(
                isinstance(x, ast.Constant) and isinstance(x.value, str)
                for x in (name, source)
            ):
                continue
            compile_name, source = name.value, source.value
            name = statement.targets[0].id
            if (name, compile_name, source) in seen:
                continue
            seen.add((name, compile_name, source))
            module = ast.parse(source)
            funcs = [n for n in ast.walk(module) if isinstance(n, ast.FunctionDef)]
            if len(funcs) != 1 or funcs[0] not in module.body:
                continue
            fn = funcs[0]
            heuristics = [
                d.func.attr
                for d in fn.decorator_list
                if isinstance(d, ast.Call)
                and isinstance(d.func, ast.Attribute)
                and ast.unparse(d.func.value) == "triton_heuristics"
            ]
            if heuristics not in (
                ["pointwise"],
                ["reduction"],
                ["persistent_reduction"],
            ) or not any(ast.unparse(d) == "triton.jit" for d in fn.decorator_list):
                continue
            # The Python function arguments omit pointer dtypes and specialized
            # argument values. Include those while allowing device metadata and
            # target launch configurations to differ.
            contract = []
            for decorator in fn.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                for arg in decorator.keywords:
                    if arg.arg == "triton_meta" and isinstance(arg.value, ast.Dict):
                        for key, value in zip(
                            arg.value.keys, arg.value.values, strict=True
                        ):
                            if isinstance(key, ast.Constant) and key.value in {
                                "signature",
                                "constants",
                            }:
                                contract.append((key.value, ast.unparse(value)))
            normalized = (
                _signature(fn),
                heuristics,
                contract,
                [ast.unparse(n) for n in fn.body],
            )
            fingerprint = hashlib.sha256(repr(normalized).encode()).hexdigest()
            yield _Kernel(name, compile_name, source, fn, fingerprint)


def write_experiment(path: Path, workload: dict, wrappers: list[str]) -> int:
    kernels = list(_kernels(wrappers))
    if not kernels:
        return 0
    originals = {}
    imports = dict.fromkeys(["import triton", "import triton.language as tl"])
    for kernel in kernels:
        referenced = {node.id for node in ast.walk(kernel.function) if isinstance(node, ast.Name)}
        for statement in ast.parse(kernel.source).body:
            if not isinstance(statement, (ast.Import, ast.ImportFrom)):
                continue
            names = [name for name in statement.names
                     if (name.asname or name.name.split(".")[0]) in referenced
                     and name.name not in {"triton_heuristics", "DeviceProperties"}]
            if names:
                statement.names = names
                imports[ast.unparse(statement)] = None
    sections = [
        '"""Edit kernel bodies here. The runner reads this file without importing it."""',
        "\n".join(imports),
        "WORKLOAD = " + pformat(workload, sort_dicts=False),
    ]
    for kernel in kernels:
        name = kernel.name
        if not name.isidentifier() or keyword.iskeyword(name) or name in originals:
            raise ValueError(f"Invalid or ambiguous generated kernel name {name!r}")
        fn = kernel.function
        originals[name] = (kernel.fingerprint, _signature(fn))
        sections.append(
            f"@triton.jit\ndef {name}({ast.unparse(fn.args)}):\n"
            + indent(_body(kernel.source, fn), "    ").rstrip()
        )
    sections.append("_ORIGINALS = " + pformat(originals, sort_dicts=False))
    sections.append("KERNELS = [" + ", ".join(originals) + "]")
    with path.open("x", encoding="utf-8") as stream:
        stream.write("\n\n".join(sections) + "\n")
    return len(kernels)


def read_experiment(path: Path) -> tuple[dict, dict[str, Edit]]:
    source = path.read_text(encoding="utf-8")
    fields = {}
    functions = {}
    for node in ast.parse(source, filename=str(path)).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {
                "WORKLOAD",
                "_ORIGINALS",
                "KERNELS",
            }:
                if target.id in fields:
                    raise ValueError(f"Duplicate {target.id}")
                fields[target.id] = node.value
        elif isinstance(node, ast.FunctionDef):
            if node.name in functions:
                raise ValueError(f"Duplicate kernel {node.name}")
            functions[node.name] = node
    if set(fields) != {"WORKLOAD", "_ORIGINALS", "KERNELS"}:
        raise ValueError("Expected WORKLOAD, _ORIGINALS and KERNELS in experiment")
    workload = ast.literal_eval(fields["WORKLOAD"])
    originals = ast.literal_eval(fields["_ORIGINALS"])
    selected = fields["KERNELS"]
    if not isinstance(workload, dict) or not isinstance(originals, dict):
        raise ValueError("Invalid workload or original kernel metadata")
    if not isinstance(selected, ast.List) or not all(
        isinstance(n, ast.Name) for n in selected.elts
    ):
        raise ValueError("KERNELS must list module-level function names")
    edits = {}
    for node in selected.elts:
        name = node.id
        if name in edits or name not in functions or name not in originals:
            raise ValueError(f"Invalid or duplicate KERNELS entry {name}")
        fn = functions[name]
        fingerprint, signature = originals[name]
        if _signature(fn) != signature:
            raise ValueError(f"Kernel signature changed for {name}")
        if [ast.unparse(d) for d in fn.decorator_list] != ["triton.jit"]:
            raise ValueError(f"Kernel decorators changed for {name}")
        edits[name] = Edit(fingerprint, signature, _body(source, fn))
    return workload, edits


def build_replacements(
    wrappers: list[str], edits: dict[str, Edit]
) -> dict[tuple[str, str], str]:
    candidates = list(_kernels(wrappers))
    replacements = {}
    for name, edit in edits.items():
        matches = [k for k in candidates if k.fingerprint == edit.fingerprint]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one compatible target for {name}, found {len(matches)}"
            )
        target = matches[0]
        key = (target.compile_name, target.source)
        if key in replacements or _signature(target.function) != edit.signature:
            raise ValueError(f"Ambiguous target or incompatible signature for {name}")
        lines, start, end = _bounds(target.source, target.function)
        prefix = lines[start][: len(lines[start]) - len(lines[start].lstrip())]
        replacements[key] = (
            "".join(lines[:start]) + indent(edit.body, prefix) + "".join(lines[end:])
        )
    return replacements
