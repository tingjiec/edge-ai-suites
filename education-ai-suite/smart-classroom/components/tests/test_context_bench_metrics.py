# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""The benchmark's measurement contract: llm_bench units, warm-up exclusion, medians.

The tool this replaced took one sample per configuration. The same 160K profile
was measured at 247.0s and 349.5s on two such runs, which made every A/B
conclusion unfalsifiable -- there was no way to tell a configuration difference
from run-to-run noise. These tests pin the three things that fix that: the
warm-up never enters a statistic, the reported figure is a median, and the
spread is carried alongside it.
"""

import unittest
from unittest import mock

from components.llm.context_bench import benchmark, metrics, trial_runner


def _record(iteration, generation_time, ttft_ms, warmup=False, output_size=64, input_size=160000):
    return metrics.iteration_record(
        iteration=iteration,
        input_size=input_size,
        output_size=output_size,
        generation_time=generation_time,
        first_token_latency=ttft_ms,
        warmup=warmup,
    )


class TestIterationRecord(unittest.TestCase):
    def test_uses_llm_bench_field_names_and_units(self):
        record = _record(1, generation_time=247.0, ttft_ms=237_000.0)

        self.assertEqual(set(metrics.ITERATION_FIELDS), set(record))
        # Latencies in milliseconds, times in seconds -- llm_bench's convention.
        self.assertEqual(record["generation_time"], 247.0)
        self.assertEqual(record["first_token_latency"], 237_000.0)
        self.assertAlmostEqual(record["latency"], 247_000.0 / 64, places=1)

    def test_decode_latency_excludes_the_first_token(self):
        # The first token comes out of prefill and is already counted by TTFT. Dividing
        # the post-TTFT time by output_size instead of output_size - 1 understates decode
        # cost -- the bug this replaces, where decode throughput read ~6.3 instead of ~6.2.
        record = _record(1, generation_time=110.0, ttft_ms=100_000.0, output_size=11)

        self.assertAlmostEqual(record["other_tokens_avg_latency"], 10_000.0 / 10, places=3)
        self.assertAlmostEqual(record["decode_throughput"], 1.0, places=3)

    def test_runtime_tpot_wins_over_the_derived_fallback(self):
        record = metrics.iteration_record(
            iteration=1,
            input_size=160000,
            output_size=64,
            generation_time=110.0,
            first_token_latency=100_000.0,
            other_tokens_avg_latency=125.0,
        )

        self.assertEqual(record["other_tokens_avg_latency"], 125.0)
        self.assertEqual(record["decode_throughput"], 8.0)

    def test_single_token_output_has_no_decode_average(self):
        record = _record(1, generation_time=100.0, ttft_ms=99_000.0, output_size=1)

        self.assertIsNone(record["other_tokens_avg_latency"])
        self.assertEqual(record["decode_throughput"], 0.0)

    def test_prefill_throughput_is_input_tokens_over_ttft(self):
        record = _record(1, generation_time=247.0, ttft_ms=237_000.0)

        self.assertAlmostEqual(record["prefill_throughput"], 160000 / 237.0, places=1)

    def test_e2e_throughput_counts_prompt_and_output(self):
        # At 160K the prompt is 2500x the output, so an output-only rate would rank
        # configurations by the ~4% of the wall clock that decode occupies.
        record = _record(1, generation_time=247.0, ttft_ms=237_000.0)

        self.assertAlmostEqual(record["e2e_throughput"], 160064 / 247.0, places=1)

    def test_missing_ttft_still_produces_a_usable_record(self):
        record = _record(1, generation_time=247.0, ttft_ms=None)

        self.assertIsNone(record["first_token_latency"])
        self.assertIsNone(record["other_tokens_avg_latency"])
        self.assertEqual(record["prefill_throughput"], 0.0)
        self.assertGreater(record["e2e_throughput"], 0)


class TestWarmupIsExcluded(unittest.TestCase):
    def test_warmup_never_reaches_an_aggregate(self):
        records = [
            _record(0, generation_time=900.0, ttft_ms=890_000.0, warmup=True),  # cold compile
            _record(1, generation_time=250.0, ttft_ms=240_000.0),
            _record(2, generation_time=250.0, ttft_ms=240_000.0),
            _record(3, generation_time=250.0, ttft_ms=240_000.0),
        ]

        aggregate = metrics.aggregate(records)

        self.assertEqual(aggregate["iterations_measured"], 3)
        self.assertEqual(aggregate["generation_time"], 250.0)
        # The 900s warm-up must not widen the reported spread either.
        self.assertEqual(aggregate["generation_time_max"], 250.0)

    def test_measured_filters_only_the_warmup(self):
        records = [_record(0, 900.0, 890_000.0, warmup=True), _record(1, 250.0, 240_000.0)]

        self.assertEqual([r["iteration"] for r in metrics.measured(records)], [1])

    def test_no_measured_iterations_is_reported_not_inferred(self):
        # A case that only got through its warm-up before dying has no result to report;
        # silently averaging the warm-up would present a cold number as a warm one.
        self.assertEqual(
            metrics.aggregate([_record(0, 900.0, 890_000.0, warmup=True)]),
            {"iterations_measured": 0},
        )
        self.assertEqual(metrics.aggregate([]), {"iterations_measured": 0})


class TestAggregation(unittest.TestCase):
    def test_reports_the_median_not_the_mean(self):
        # One slow outlier is exactly the case a mean would smear across the result.
        records = [
            _record(1, generation_time=247.0, ttft_ms=237_000.0),
            _record(2, generation_time=250.0, ttft_ms=240_000.0),
            _record(3, generation_time=349.5, ttft_ms=338_600.0),
        ]

        aggregate = metrics.aggregate(records)

        self.assertEqual(aggregate["generation_time"], 250.0)
        self.assertNotAlmostEqual(aggregate["generation_time"], (247.0 + 250.0 + 349.5) / 3)

    def test_keeps_the_spread_visible(self):
        records = [
            _record(1, generation_time=247.0, ttft_ms=237_000.0),
            _record(2, generation_time=250.0, ttft_ms=240_000.0),
            _record(3, generation_time=349.5, ttft_ms=338_600.0),
        ]

        aggregate = metrics.aggregate(records)

        self.assertEqual(aggregate["generation_time_min"], 247.0)
        self.assertEqual(aggregate["generation_time_max"], 349.5)
        self.assertLess(aggregate["e2e_throughput_min"], aggregate["e2e_throughput_max"])

    def test_aggregates_every_advertised_metric(self):
        aggregate = metrics.aggregate([_record(1, 247.0, 237_000.0), _record(2, 250.0, 240_000.0)])

        for metric in metrics.AGGREGATED_METRICS:
            self.assertIn(metric, aggregate, f"{metric} is advertised but never aggregated")

    def test_a_metric_missing_from_every_iteration_is_omitted_not_zeroed(self):
        aggregate = metrics.aggregate([_record(1, 247.0, None), _record(2, 250.0, None)])

        self.assertNotIn("first_token_latency", aggregate)
        self.assertIn("generation_time", aggregate)


class TestProfileRanking(unittest.TestCase):
    def test_ranks_tpot_first_and_ttft_second(self):
        cases = [
            {
                "profile": "better-ttft",
                "status": "ok",
                "other_tokens_avg_latency": 170.0,
                "first_token_latency": 200_000.0,
            },
            {
                "profile": "best-tpot-slower-ttft",
                "status": "ok",
                "other_tokens_avg_latency": 160.0,
                "first_token_latency": 300_000.0,
            },
            {
                "profile": "best-tpot-faster-ttft",
                "status": "ok",
                "other_tokens_avg_latency": 160.0,
                "first_token_latency": 250_000.0,
            },
        ]

        ranked = benchmark._leaderboard(cases)

        self.assertEqual(
            [case["profile"] for case in ranked],
            ["best-tpot-faster-ttft", "best-tpot-slower-ttft", "better-ttft"],
        )

    def test_gpu_memory_limit_is_not_ranked(self):
        cases = [
            {"profile": "over-budget", "status": "gpu_memory_limit", "other_tokens_avg_latency": 1},
            {"profile": "usable", "status": "ok", "other_tokens_avg_latency": 100},
        ]

        self.assertEqual([case["profile"] for case in benchmark._leaderboard(cases)], ["usable"])


class TestMemoryStatus(unittest.TestCase):
    def test_gpu_budget_overrun_is_not_usable(self):
        case = {
            "status": "ok",
            "gpu_budget_exceeded": True,
            "system_memory_limit_exceeded": False,
        }

        self.assertEqual(benchmark._apply_memory_status(case)["status"], "gpu_memory_limit")

    def test_system_memory_overrun_is_demoted(self):
        case = {
            "status": "ok",
            "gpu_budget_exceeded": False,
            "system_memory_limit_exceeded": True,
        }

        self.assertEqual(benchmark._apply_memory_status(case)["status"], "memory_limit")

    def test_existing_failure_is_preserved(self):
        case = {
            "status": "gpu_abort",
            "gpu_budget_exceeded": True,
            "system_memory_limit_exceeded": True,
        }

        self.assertEqual(benchmark._apply_memory_status(case)["status"], "gpu_abort")


class TestSafePlatformInfo(unittest.TestCase):
    def test_does_not_import_the_wmi_platform_helper(self):
        with mock.patch.dict("sys.modules", {"utils.platform_info": None}):
            info = benchmark._safe_platform_info()

        self.assertIn("Processor", info)
        self.assertIn("Memory", info)
        self.assertEqual(info["iGPU"], "Intel Graphics")


class TestPerfMetricsAreTheStandardSource(unittest.TestCase):
    """OpenVINO GenAI's own perf_metrics is what llm_bench reads, so it wins over the
    wall clock -- but every field is optional and a partial one must not lose the case."""

    class _Pair:
        def __init__(self, mean):
            self.mean = mean

    class _Perf:
        def __init__(self, **values):
            self._values = values

        def _get(self, key):
            if key not in self._values:
                raise RuntimeError(f"{key} not available in this runtime build")
            return TestPerfMetricsAreTheStandardSource._Pair(self._values[key])

        def get_ttft(self):
            return self._get("ttft")

        def get_tpot(self):
            return self._get("tpot")

        def get_tokenization_duration(self):
            return self._get("tokenization")

        def get_detokenization_duration(self):
            return self._get("detokenization")

        def get_generate_duration(self):
            return self._get("generate")

        def get_num_generated_tokens(self):
            if "generated" not in self._values:
                raise RuntimeError("not available")
            return self._values["generated"]

    class _Result:
        def __init__(self, perf):
            self.perf_metrics = perf

    def test_reads_the_runtime_measurement(self):
        result = self._Result(
            self._Perf(ttft=237_000.0, tpot=160.0, tokenization=12.0, generate=247_000.0, generated=64)
        )

        read = trial_runner.read_perf_metrics(result)

        self.assertEqual(read["first_token_latency"], 237_000.0)
        self.assertEqual(read["other_tokens_avg_latency"], 160.0)
        self.assertEqual(read["output_size"], 64)
        # get_generate_duration() is milliseconds; the record wants seconds.
        self.assertAlmostEqual(read["generation_time"], 247.0)

    def test_a_field_the_runtime_does_not_implement_is_skipped_not_fatal(self):
        read = trial_runner.read_perf_metrics(self._Result(self._Perf(ttft=237_000.0)))

        self.assertEqual(read["first_token_latency"], 237_000.0)
        self.assertNotIn("generation_time", read)
        self.assertNotIn("output_size", read)

    def test_a_result_without_perf_metrics_falls_back_entirely(self):
        self.assertEqual(trial_runner.read_perf_metrics(object()), {})


if __name__ == "__main__":
    unittest.main()
