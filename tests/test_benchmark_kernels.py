import contextlib
import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from src.benchmark import __main__ as cli
from src.benchmark.kernels import build_replacements, read_experiment, write_experiment

STOCK_SOURCE = """\
import triton
import triton.language as tl
from torch._inductor.runtime import triton_heuristics

@triton_heuristics.pointwise(triton_meta={"signature": {"in_ptr0": "*fp32"}})
@triton.jit
def copy_kernel(in_ptr0, out_ptr0, XBLOCK: tl.constexpr):
  offsets = tl.arange(0, XBLOCK)
  if XBLOCK > 0:
    tl.store(out_ptr0 + offsets, tl.load(in_ptr0 + offsets))
"""
STOCK_WRAPPER = f"copy_kernel = async_compile.triton('copy_kernel', {STOCK_SOURCE!r})\n"
WORKLOAD = {"fixture": "arxiv-full"}
CAPTURES = Path(__file__).resolve().parents[1] / "src" / "captures"


class BenchmarkKernelTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.experiment = Path(temporary.name) / "experiment.py"

    def test_unedited_experiment_round_trip(self):
        count = write_experiment(self.experiment, WORKLOAD, [STOCK_WRAPPER])
        self.assertEqual(count, 1)
        workload, edits = read_experiment(self.experiment)
        self.assertEqual(workload, WORKLOAD)
        self.assertEqual(build_replacements([STOCK_WRAPPER], edits), {
            ("copy_kernel", STOCK_SOURCE): STOCK_SOURCE,
        })

    def test_edited_body_uses_original_indentation(self):
        write_experiment(self.experiment, WORKLOAD, [STOCK_WRAPPER])
        source = self.experiment.read_text().replace(
            "      tl.store(out_ptr0 + offsets, tl.load(in_ptr0 + offsets))",
            "      value = tl.load(in_ptr0 + offsets)\n"
            "      tl.store(out_ptr0 + offsets, value * 2)",
        )
        self.experiment.write_text(source)
        _, edits = read_experiment(self.experiment)
        expected = STOCK_SOURCE.replace(
            "    tl.store(out_ptr0 + offsets, tl.load(in_ptr0 + offsets))",
            "    value = tl.load(in_ptr0 + offsets)\n"
            "    tl.store(out_ptr0 + offsets, value * 2)",
        )
        self.assertEqual(build_replacements([STOCK_WRAPPER], edits), {
            ("copy_kernel", STOCK_SOURCE): expected,
        })

    def test_changed_signature_is_rejected(self):
        write_experiment(self.experiment, WORKLOAD, [STOCK_WRAPPER])
        source = self.experiment.read_text().replace(
            "def copy_kernel(in_ptr0, out_ptr0,", "def copy_kernel(in_ptr0, out_ptr0, extra,"
        )
        self.experiment.write_text(source)
        with self.assertRaisesRegex(ValueError, "Kernel signature changed for copy_kernel"):
            read_experiment(self.experiment)

    def test_checked_in_captures(self):
        for name in ("sage-arxiv", "gcn-arxiv", "gat-arxiv", "sage-products"):
            with self.subTest(name=name):
                workload, edits = read_experiment(CAPTURES / f"{name}.py")
                self.assertIn("fixture", workload)
                self.assertTrue(edits)

    def test_capture_logs_skipped_heuristics(self):
        templated = STOCK_SOURCE.replace(".pointwise(", ".template(")
        mixed_wrapper = STOCK_WRAPPER + (
            f"template_kernel = async_compile.triton('template_kernel', {templated!r})\n"
        )

        def mock_worker(output, _cache, _request):
            output.mkdir()
            (output / "output_code_0.py").write_text(mixed_wrapper)
            return {"workload": WORKLOAD, "correctness": {"passed": True}}

        args = Namespace(
            out=self.experiment, checkpoint_dir=self.experiment.parent,
            data_dir=self.experiment.parent, atol=None, rtol=None, fixture="arxiv-full",
        )
        log = io.StringIO()
        with patch.object(cli, "run_worker", mock_worker), contextlib.redirect_stdout(log):
            cli.capture(args)

        skipped = self.experiment.with_suffix(".capture") / "skipped_kernels.json"
        self.assertEqual(json.loads(skipped.read_text()), [
            {"name": "template_kernel", "heuristics": ["template"]},
        ])
        self.assertIn("Skipped Triton kernel template_kernel (heuristics: template)", log.getvalue())
        _, edits = read_experiment(self.experiment)
        self.assertEqual(list(edits), ["copy_kernel"])
