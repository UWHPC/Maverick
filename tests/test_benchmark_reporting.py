import json
import random
import statistics
import tempfile
import unittest
from pathlib import Path

from src.benchmark.reporting import write_report


class BenchmarkReportTests(unittest.TestCase):
    def write_timing_report(self, centres, *, repeat_jitter=False):
        rng = random.Random(2)
        blocks = []
        for index, (variant, centre) in enumerate(
            zip(("stock", "candidate", "candidate", "stock"), centres, strict=True),
            start=1,
        ):
            if repeat_jitter:
                rng.seed(2)
            blocks.append({
                "index": index,
                "variant": variant,
                "samples": {
                    metric: [centre + rng.gauss(0, 0.001) for _ in range(30)]
                    for metric in ("wall_ms", "gpu_ms")
                },
            })
        report = {"status": "ok", "correctness": {"passed": True}, "timing_blocks": blocks}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_report(output, report)
            result = json.loads((output / "report.json").read_text())
            summary = (output / "summary.md").read_text()
        return result, summary

    def test_block_variation(self):
        for centres, status in (
            ((10, 10, 10, 10), "within_noise"),
            ((10, 9, 9, 10), "speedup"),
            ((10, 9, 10.6, 10), "within_noise"),
            ((10, 9.9, 9.9, 10.2), "within_noise"),
        ):
            with self.subTest(centres=centres):
                result, summary = self.write_timing_report(centres)
                self.assertIn("Stock gap | Candidate gap", summary)
                comparisons = result["summary"]["candidate_vs_stock"]
                self.assertEqual(set(comparisons), {"wall_ms", "gpu_ms"})
                for metric, comparison in comparisons.items():
                    self.assertEqual(comparison["status"], status)
                    first_stock, first_candidate, last_candidate, last_stock = (
                        statistics.median(block["samples"][metric])
                        for block in result["timing_blocks"]
                    )
                    stock_gap = abs(first_stock - last_stock)
                    candidate_gap = abs(first_candidate - last_candidate)
                    stock_mean = statistics.mean((first_stock, last_stock))
                    candidate_mean = statistics.mean((first_candidate, last_candidate))
                    stock_pct = 100 * stock_gap / stock_mean
                    candidate_pct = 100 * candidate_gap / candidate_mean
                    self.assertAlmostEqual(comparison["stock_gap_pct"], stock_pct)
                    self.assertAlmostEqual(comparison["candidate_gap_pct"], candidate_pct)
                    self.assertAlmostEqual(
                        comparison["noise_floor_pct"],
                        100 * max(stock_gap, candidate_gap) / stock_mean,
                    )
                    if status == "speedup":
                        self.assertAlmostEqual(comparison["median_speedup"], 10 / 9, places=2)
                        label = f"{comparison['median_speedup']:.3f}x (speedup)"
                    else:
                        self.assertNotIn("median_speedup", comparison)
                        label = "within noise"
                    self.assertIn(
                        f"| {metric.removesuffix('_ms')} | {stock_pct:.2f}% | "
                        f"{candidate_pct:.2f}% | {label} |",
                        summary,
                    )

    def test_ratio_that_prints_as_one_is_within_noise(self):
        for candidate in (9.998, 10.002):
            with self.subTest(candidate=candidate):
                result, summary = self.write_timing_report(
                    (10, candidate, candidate, 10), repeat_jitter=True
                )
                comparisons = result["summary"]["candidate_vs_stock"]
                self.assertEqual(len(comparisons), 2)
                for comparison in comparisons.values():
                    self.assertEqual(comparison["noise_floor_pct"], 0)
                    self.assertEqual(comparison["status"], "within_noise")
                    self.assertNotIn("median_speedup", comparison)
                self.assertNotIn("1.000x", summary)
