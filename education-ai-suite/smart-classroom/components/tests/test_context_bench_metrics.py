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
from types import SimpleNamespace
from unittest import mock

from components.llm.context_bench import benchmark, metrics, trial_runner


def _settings_args(**overrides):
    values = {
        "config": "benchmark.yaml",
        "models": None,
        "contexts": None,
        "profiles": None,
        "output_tokens": None,
        "warmup": None,
        "iterations": None,
        "device": None,
        "weight_format": None,
        "output_dir": None,
        "pipeline_config": None,
        "scheduler_config": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _settings_config(**benchmark_overrides):
    benchmark_values = {
        "models": ["Qwen/Qwen3.5-9B"],
        "context_tokens": [160000],
        "output_tokens": 64,
        "warmup": 1,
        "iterations": 3,
        "timeout_sec": 2400,
        "max_system_memory_pct": 80,
        "gpu_memory_budget_gb": 59,
        "cache_dir": None,
        "output_dir": "monitoring/executionlogs/context_bench",
    }
    benchmark_values.update(benchmark_overrides)
    return SimpleNamespace(
        model=SimpleNamespace(
            provider="openvino",
            models_base_path="models",
            device="GPU",
            weight_format="int8",
        ),
        benchmark=SimpleNamespace(**benchmark_values),
        profiles=[{"name": "default", "ov": {}, "scheduler": {}}],
    )


class TestSettingsValidation(unittest.TestCase):
    def _load(self, config, **arg_overrides):
        with mock.patch.object(benchmark, "load_config", return_value=config):
            return benchmark._load_settings(_settings_args(**arg_overrides))

    def test_output_too_short_for_tpot_is_rejected_instead_of_falling_back(self):
        with self.assertRaisesRegex(SystemExit, "output_tokens must be at least 2"):
            self._load(_settings_config(), output_tokens=0)

    def test_malformed_context_list_has_an_actionable_error(self):
        with self.assertRaisesRegex(SystemExit, "only positive integers"):
            self._load(_settings_config(context_tokens=[160000, "bad"]))

    def test_boolean_iterations_is_not_accepted_as_one(self):
        with self.assertRaisesRegex(SystemExit, "iterations must be a positive integer"):
            self._load(_settings_config(iterations=True))

    def test_negative_warmup_is_rejected_before_running_a_case(self):
        with self.assertRaisesRegex(SystemExit, "warmup must be a non-negative integer"):
            self._load(_settings_config(warmup=-1))

    def test_non_positive_timeout_is_rejected_before_running_a_case(self):
        with self.assertRaisesRegex(SystemExit, "timeout_sec must be greater than 0"):
            self._load(_settings_config(timeout_sec=0))

    def test_empty_model_list_is_rejected(self):
        with self.assertRaisesRegex(SystemExit, "at least one non-empty model name"):
            self._load(_settings_config(models=[]))

    def test_duplicate_profile_names_are_rejected(self):
        config = _settings_config()
        config.profiles.append({"name": "default", "ov": {}, "scheduler": {}})

        with self.assertRaisesRegex(SystemExit, "profile names must be unique"):
            self._load(config)

    def test_prefix_caching_is_rejected_because_it_reuses_the_warmup_prompt(self):
        config = _settings_config()
        config.profiles[0]["scheduler"] = {"enable_prefix_caching": True}

        with self.assertRaisesRegex(SystemExit, "must set enable_prefix_caching=false"):
            self._load(config)

    def test_numeric_prefix_caching_override_cannot_bypass_the_guard(self):
        with self.assertRaisesRegex(SystemExit, "must set enable_prefix_caching=false"):
            self._load(_settings_config(), scheduler_config=["enable_prefix_caching=1"])

    def test_falsey_non_mapping_profile_values_are_rejected(self):
        for key, value in (("ov", []), ("scheduler", ""), ("scheduler", 0)):
            config = _settings_config()
            config.profiles[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaisesRegex(
                SystemExit, "ov and scheduler must be mappings"
            ):
                self._load(config)

    def test_profile_level_cache_size_is_rejected_with_a_pointer_to_scheduler(self):
        # The mistake this exists for: `cache_size` written next to `ov` in the belief that it
        # caps the stateful pipeline's cache. It does not exist there, and silently ignoring it
        # would run hours of measurement on a configuration the author did not ask for.
        config = _settings_config()
        config.profiles[0]["cache_size"] = 8

        with self.assertRaisesRegex(SystemExit, "unknown key"):
            self._load(config)

    def test_scheduler_override_on_a_stateful_profile_announces_the_pipeline_change(self):
        with mock.patch("builtins.print") as printed:
            settings = self._load(_settings_config(), scheduler_config=["cache_size=8"])

        self.assertEqual(
            benchmark.pipeline_mode(settings["profiles"][0]["scheduler"]),
            benchmark.PIPELINE_PAGED,
        )
        self.assertTrue(
            any("moved profile" in str(call) for call in printed.call_args_list),
            "switching a profile off the stateful pipeline must not be silent",
        )

    def test_gpu_budget_must_be_finite_and_positive(self):
        for value in (0, -1, float("nan"), float("inf"), "not-a-number"):
            with self.subTest(value=value), self.assertRaisesRegex(
                SystemExit, "gpu_memory_budget_gb must be a finite positive number"
            ):
                self._load(_settings_config(gpu_memory_budget_gb=value))


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

    def test_inconsistent_fallback_timings_do_not_produce_negative_tpot(self):
        record = _record(1, generation_time=10.0, ttft_ms=11_000.0)

        self.assertIsNone(record["other_tokens_avg_latency"])
        self.assertEqual(record["decode_throughput"], 0.0)


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

    def test_missing_ttft_sorts_last_instead_of_comparing_none(self):
        cases = [
            {
                "profile": "missing-ttft",
                "status": "ok",
                "other_tokens_avg_latency": 160.0,
                "first_token_latency": None,
            },
            {
                "profile": "measured-ttft",
                "status": "ok",
                "other_tokens_avg_latency": 160.0,
                "first_token_latency": 250_000.0,
            },
        ]

        self.assertEqual(
            [case["profile"] for case in benchmark._leaderboard(cases)],
            ["measured-ttft", "missing-ttft"],
        )

    def test_ttft_leaderboard_ignores_tpot_so_a_prefill_win_is_visible(self):
        # The stateful-vs-paged question is decided in prefill. Ranking only by TPOT would
        # report the paged profile as the winner even when it takes 100s longer to first token.
        cases = [
            {
                "profile": "paged_min",
                "status": "ok",
                "other_tokens_avg_latency": 160.0,
                "first_token_latency": 300_000.0,
            },
            {
                "profile": "stateful",
                "status": "ok",
                "other_tokens_avg_latency": 170.0,
                "first_token_latency": 200_000.0,
            },
        ]

        self.assertEqual(
            [case["profile"] for case in benchmark._ttft_leaderboard(cases)],
            ["stateful", "paged_min"],
        )
        self.assertEqual(benchmark._leaderboard(cases)[0]["profile"], "paged_min")

    def test_ttft_leaderboard_drops_unusable_and_unmeasured_cases(self):
        cases = [
            {"profile": "over-budget", "status": "gpu_memory_limit", "first_token_latency": 1.0},
            {"profile": "no-ttft", "status": "ok", "first_token_latency": None},
            {"profile": "usable", "status": "ok", "first_token_latency": 250_000.0},
        ]

        self.assertEqual(
            [case["profile"] for case in benchmark._ttft_leaderboard(cases)], ["usable"]
        )

    def test_successful_case_without_ttft_formats_as_unavailable(self):
        case = {
            "status": "ok",
            "other_tokens_avg_latency": 160.0,
            "first_token_latency": None,
            "decode_throughput": 6.25,
            "e2e_throughput": 640.0,
            "peak_ram_gb": 40.0,
            "peak_ram_pct": 62.5,
            "peak_gpu_gb": 26.0,
            "peak_gpu_pct_of_budget": 44.1,
        }

        self.assertIn("TTFT --s", benchmark._format_case(case))


class TestMemoryStatus(unittest.TestCase):
    def test_unavailable_budget_measurement_is_not_usable(self):
        case = {
            "status": "ok",
            "memory_measurement_error": True,
            "gpu_budget_exceeded": False,
            "system_memory_limit_exceeded": False,
        }

        self.assertEqual(benchmark._apply_memory_status(case)["status"], "measurement_error")

    def test_run_case_fails_closed_when_memory_telemetry_is_unavailable(self):
        result = {
            "error": None,
            "load_ok": True,
            "stage_reached": trial_runner.STAGE_DECODED,
            "iterations": [_record(1, 250.0, 240_000.0)],
            "peak_ram_pct": None,
            "peak_gpu_gb": None,
        }
        settings = {
            "device": "GPU",
            "cache_dir": None,
            "output_tokens": 64,
            "warmup": 0,
            "iterations": 1,
            "timeout_sec": 600,
            "gpu_memory_budget_gb": 59.0,
            "max_system_memory_pct": 100,
            "weight_format": "int8",
        }
        static = {"model_config": None, "fixed_state_bytes": 0, "weight_disk_gb": 8.8}

        with mock.patch.object(
            benchmark, "_run_case_subprocess", return_value=result
        ), mock.patch("builtins.print"):
            case = benchmark._run_case(
                "Qwen/Qwen3.5-9B",
                "models/openvino/Qwen_Qwen3.5-9B_int8",
                {"name": "optimized", "ov": {}, "scheduler": {}},
                160000,
                settings,
                static,
            )

        self.assertTrue(case["memory_measurement_error"])
        self.assertEqual(case["status"], "measurement_error")

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


class TestTimeWeightedMean(unittest.TestCase):
    """Peak alone cannot distinguish a workload that touches 40 GB for a second from one that
    holds it for six minutes, and on a shared 59 GB budget that is the difference that
    matters. These pin the weighting, because an unweighted sample average would be biased by
    the sampler's own jitter -- `_read_mem()` is a PDH query whose cost varies with load."""

    def test_weights_each_reading_by_how_long_it_stood(self):
        mean = benchmark._TimeWeightedMean(10.0, 0.0)
        mean.observe(20.0, 1.0)  # 10.0 stood for 1s

        # 20.0 then stands for 9s: (10*1 + 20*9) / 10 = 19.0, not the 15.0 of a flat average.
        self.assertAlmostEqual(mean.mean(10.0), 19.0)

    def test_an_unavailable_counter_contributes_nothing_rather_than_zero(self):
        mean = benchmark._TimeWeightedMean(None, 0.0)
        mean.observe(30.0, 5.0)  # the unreadable stretch must not be averaged in as 0 GB

        self.assertAlmostEqual(mean.mean(10.0), 30.0)

    def test_a_counter_never_available_reports_no_measurement(self):
        self.assertIsNone(benchmark._TimeWeightedMean(None, 0.0).mean(10.0))

    def test_a_window_shorter_than_one_sample_reports_the_standing_reading(self):
        # No elapsed time to weight, but one sample is still a measurement.
        self.assertEqual(benchmark._TimeWeightedMean(42.0, 0.0).mean(0.0), 42.0)

    def test_reading_the_mean_twice_does_not_double_count(self):
        mean = benchmark._TimeWeightedMean(10.0, 0.0)

        self.assertAlmostEqual(mean.mean(10.0), 10.0)
        self.assertAlmostEqual(mean.mean(10.0), 10.0)
        mean.observe(20.0, 10.0)
        self.assertAlmostEqual(mean.mean(20.0), 15.0)

    def test_the_sampling_rate_does_not_change_the_answer(self):
        """The property that makes the number comparable across runs: doubling the sample
        count over the same 10s at the same levels must not move the mean."""
        coarse = benchmark._TimeWeightedMean(10.0, 0.0)
        coarse.observe(30.0, 5.0)

        fine = benchmark._TimeWeightedMean(10.0, 0.0)
        for at, value in ((2.5, 10.0), (5.0, 30.0), (7.5, 30.0)):
            fine.observe(value, at)

        self.assertAlmostEqual(coarse.mean(10.0), fine.mean(10.0))


class TestMeanOccupancyReachesTheReport(unittest.TestCase):
    def test_aggregated_as_the_median_of_the_measured_windows(self):
        rows = [
            {"warmup": True, "mean_ram_gb": 99.0, "mean_gpu_gb": 99.0},
            {"warmup": False, "mean_ram_gb": 40.0, "mean_gpu_gb": 20.0},
            {"warmup": False, "mean_ram_gb": 42.0, "mean_gpu_gb": 24.0},
            {"warmup": False, "mean_ram_gb": 41.0, "mean_gpu_gb": 22.0},
        ]

        aggregate = metrics.aggregate(rows)

        self.assertEqual(aggregate["mean_ram_gb"], 41.0)
        self.assertEqual(aggregate["mean_gpu_gb"], 22.0)
        # Median only: the case-level `peak_*` already reports the high end, so a min/max
        # of the means would be columns that answer nothing new.
        self.assertNotIn("mean_gpu_gb_max", aggregate)

    def test_a_run_without_memory_telemetry_omits_the_means(self):
        aggregate = metrics.aggregate([{"warmup": False, "generation_time": 250.0}])

        self.assertNotIn("mean_ram_gb", aggregate)
        self.assertNotIn("mean_gpu_gb", aggregate)

    def test_both_csvs_carry_the_new_columns(self):
        for field in ("mean_ram_gb", "mean_gpu_gb"):
            self.assertIn(field, benchmark.CASE_FIELDS)
            self.assertIn(field, benchmark.ITERATION_CSV_FIELDS)
        self.assertIn("mean_gpu_pct_of_budget", benchmark.CASE_FIELDS)


class TestPipelineModeIsRecorded(unittest.TestCase):
    """Which pipeline ran is a measured variable, not a footnote.

    `cache_size` only exists on SchedulerConfig and any SchedulerConfig selects continuous
    batching, so the two configurations a config ships are told apart by the presence of a
    scheduler alone. If that never reached the report, two rows with very different TTFTs
    would be indistinguishable after the run.
    """

    SETTINGS = {
        "device": "GPU",
        "cache_dir": None,
        "output_tokens": 64,
        "warmup": 0,
        "iterations": 1,
        "timeout_sec": 600,
        "gpu_memory_budget_gb": 59.0,
        "max_system_memory_pct": 100,
        "weight_format": "int8",
    }

    def test_empty_and_absent_schedulers_are_the_stateful_pipeline(self):
        self.assertEqual(benchmark.pipeline_mode({}), benchmark.PIPELINE_STATEFUL)
        self.assertEqual(benchmark.pipeline_mode(None), benchmark.PIPELINE_STATEFUL)

    def test_any_scheduler_key_at_all_is_the_paged_pipeline(self):
        self.assertEqual(
            benchmark.pipeline_mode({"cache_size": 8}), benchmark.PIPELINE_PAGED
        )
        self.assertEqual(
            benchmark.pipeline_mode({"enable_prefix_caching": False}), benchmark.PIPELINE_PAGED
        )

    def _case(self, scheduler):
        result = {
            "error": None,
            "load_ok": True,
            "stage_reached": trial_runner.STAGE_DECODED,
            "iterations": [_record(1, 250.0, 240_000.0)],
            "peak_ram_pct": 62.5,
            "peak_gpu_gb": 26.0,
        }
        static = {"model_config": None, "fixed_state_bytes": 0, "weight_disk_gb": 8.8}
        with mock.patch.object(
            benchmark, "_run_case_subprocess", return_value=result
        ), mock.patch("builtins.print"):
            return benchmark._run_case(
                "Qwen/Qwen3.5-9B",
                "models/openvino/Qwen_Qwen3.5-9B_int8",
                {"name": "profile", "ov": {}, "scheduler": scheduler},
                160000,
                dict(self.SETTINGS),
                static,
            )

    def test_run_case_records_the_mode_for_both_pipelines(self):
        self.assertEqual(self._case({})["pipeline_mode"], benchmark.PIPELINE_STATEFUL)
        self.assertEqual(
            self._case({"cache_size": 8})["pipeline_mode"], benchmark.PIPELINE_PAGED
        )

    def test_an_underivable_auto_cache_size_reports_the_pipeline_that_actually_ran(self):
        # `cache_size: auto` on an architecture the tool cannot size drops the key, and if it
        # was the only scheduler key the case runs stateful after all. Naming the mode before
        # that resolution would describe a pipeline that never ran.
        case = self._case({"cache_size": "auto"})

        self.assertEqual(case["pipeline_mode"], benchmark.PIPELINE_STATEFUL)
        self.assertIsNone(case["cache_size_gb"])

    def test_the_mode_reaches_summary_csv(self):
        self.assertIn("pipeline_mode", benchmark.CASE_FIELDS)


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

        def get_num_input_tokens(self):
            if "consumed" not in self._values:
                raise RuntimeError("not available")
            return self._values["consumed"]

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
        self.assertNotIn("input_size", read)

    def test_reads_the_prefilled_token_count_the_runtime_reports(self):
        """What the pipeline actually prefilled, which the VLMPipeline path cannot get
        from the prompt it measured -- prefill/e2e throughput divide by this."""
        read = trial_runner.read_perf_metrics(
            self._Result(self._Perf(ttft=237_000.0, consumed=160_000))
        )

        self.assertEqual(read["input_size"], 160_000)

    def test_a_zero_input_count_does_not_displace_the_measured_prompt(self):
        """A runtime that leaves the counter at 0 must fall through to the counted prompt,
        or every throughput divides by zero."""
        read = trial_runner.read_perf_metrics(
            self._Result(self._Perf(ttft=237_000.0, consumed=0))
        )

        self.assertNotIn("input_size", read)

    def test_a_result_without_perf_metrics_falls_back_entirely(self):
        self.assertEqual(trial_runner.read_perf_metrics(object()), {})


class TestGenerationConfigMatchesLlmBench(unittest.TestCase):
    class _Config:
        max_new_tokens = 0
        max_length = 0
        ignore_eos = False
        do_sample = True
        apply_chat_template = True

    class _Pipeline:
        def get_generation_config(self):
            return TestGenerationConfigMatchesLlmBench._Config()

    def test_forces_deterministic_fixed_length_generation(self):
        config = trial_runner.generation_config(self._Pipeline(), 64)

        self.assertEqual(config.max_new_tokens, 64)
        self.assertEqual(config.max_length, 2**64 - 1)
        self.assertTrue(config.ignore_eos)
        self.assertFalse(config.do_sample)
        self.assertFalse(config.apply_chat_template)


class TestGeneratedTokenCount(unittest.TestCase):
    class _Tokenizer:
        def __init__(self):
            self.encoded = None

        def encode(self, text):
            self.encoded = text
            input_ids = type("InputIds", (), {"shape": (1, 7)})()
            return type("Tokenized", (), {"input_ids": input_ids})()

    def test_uses_encoded_tokens_without_detokenizing(self):
        result = type("Result", (), {"tokens": [[1, 2, 3]]})()

        self.assertEqual(trial_runner.generated_token_count(result, self._Tokenizer()), 3)

    def test_vlm_fallback_encodes_generated_text_not_result_repr(self):
        tokenizer = self._Tokenizer()
        result = type("Result", (), {"texts": ["generated answer"]})()

        self.assertEqual(trial_runner.generated_token_count(result, tokenizer), 7)
        self.assertEqual(tokenizer.encoded, "generated answer")


if __name__ == "__main__":
    unittest.main()
