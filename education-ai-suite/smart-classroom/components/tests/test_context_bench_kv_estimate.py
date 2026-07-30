# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""The architecture-derived KV estimate and the `cache_size: auto` derived from it.

This estimate stopped being a reporting nicety when profiles gained
``cache_size: auto``: it now sizes the scheduler's KV pool, so getting it wrong
either starves a run or has the scheduler refuse to build the pipeline at all
(``cache_size=24`` at 160K was rejected outright as larger than available memory).
"""

import tempfile
import unittest
from pathlib import Path

from components.llm.context_bench import benchmark
from components.llm.context_bench.benchmark import (
    _ir_ready,
    _KV_PRECISION_BYTES,
    PIPELINE_PAGED,
    PIPELINE_STATEFUL,
    auto_cache_size_gb,
    expected_kv_gb,
    fixed_state_cache_bytes,
    kv_cache_dtype_bytes,
    kv_quantization_param_bytes,
    pipeline_mode,
    theoretical_kv_bytes_per_token,
    validate_fixed_cache_size,
)
from utils.config_loader import load_config

# Mirrors the real Qwen3.5-9B config.json shape: 32 layers repeating
# 3x linear_attention + 1x full_attention, so only 8 layers grow with context.
_QWEN35_9B = {
    "text_config": {
        "num_hidden_layers": 32,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 8,
    }
}


class TestTheoreticalKvBytesPerToken(unittest.TestCase):
    def test_dense_model_without_layer_types_counts_every_layer(self):
        config = {"num_hidden_layers": 36, "num_key_value_heads": 8, "head_dim": 128}
        self.assertEqual(theoretical_kv_bytes_per_token(config), 2 * 36 * 8 * 128 * 2)

    def test_hybrid_model_counts_only_full_attention_layers(self):
        # linear_attention layers hold an O(1) recurrent state, not one that grows
        # with token count, so charging all 32 layers would overstate KV by 4x.
        self.assertEqual(theoretical_kv_bytes_per_token(_QWEN35_9B), 2 * 8 * 4 * 256 * 2)

    def test_vlm_config_reads_nested_text_config(self):
        config = {
            "architectures": ["SomeForConditionalGeneration"],
            "vision_config": {"hidden_size": 1152},
            "text_config": {"num_hidden_layers": 28, "num_key_value_heads": 4, "head_dim": 128},
        }
        self.assertEqual(theoretical_kv_bytes_per_token(config), 2 * 28 * 4 * 128 * 2)

    def test_missing_required_keys_returns_none(self):
        self.assertIsNone(
            theoretical_kv_bytes_per_token({"text_config": {"num_hidden_layers": 32, "head_dim": 128}})
        )

    def test_layer_types_with_no_full_attention_returns_none(self):
        config = {
            "num_hidden_layers": 4,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "layer_types": ["linear_attention"] * 4,
        }
        self.assertIsNone(theoretical_kv_bytes_per_token(config))

    def test_incomplete_layer_types_returns_none_instead_of_underestimating_kv(self):
        config = {
            "num_hidden_layers": 8,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "layer_types": ["full_attention"] * 2,
        }

        self.assertIsNone(theoretical_kv_bytes_per_token(config))

    def test_compressed_cache_includes_scale_and_zero_point_per_row(self):
        config = {"num_hidden_layers": 8, "num_key_value_heads": 4, "head_dim": 256}
        self.assertEqual(
            theoretical_kv_bytes_per_token(config, kv_cache_dtype_bytes=1, quantization_param_bytes=4),
            2 * 8 * 4 * (256 + 4),
        )


class TestPrecisionDrivesTheEstimate(unittest.TestCase):
    """A u8 run compared against an fp16 estimate would report a phantom 2x saving."""

    def test_known_precisions_map_to_element_size(self):
        for precision, expected in (("f16", 2), ("u8", 1), ("u4", 0.5), ("f32", 4)):
            self.assertEqual(kv_cache_dtype_bytes({"KV_CACHE_PRECISION": precision}), expected)

    def test_unset_or_dynamic_precision_falls_back_to_fp16(self):
        # "dynamic" is the GPU plugin's default and means "the plugin decides",
        # which in practice is fp16 -- better than reporting no estimate at all.
        self.assertEqual(kv_cache_dtype_bytes({}), 2)
        self.assertEqual(kv_cache_dtype_bytes({"KV_CACHE_PRECISION": "dynamic"}), 2)

    def test_only_compressed_caches_carry_quantization_params(self):
        self.assertEqual(kv_quantization_param_bytes({"KV_CACHE_PRECISION": "u8"}), 4)
        self.assertEqual(kv_quantization_param_bytes({"KV_CACHE_PRECISION": "f16"}), 0)

    def test_qwen35_9b_at_160k_f16_matches_the_measured_footprint(self):
        # 4.93 GB is what the tool reported for this model/context/precision on the
        # 64 GB box, and is the number cache_size=8 was hand-tuned to cover.
        kv_gb = expected_kv_gb(_QWEN35_9B, 0, {"KV_CACHE_PRECISION": "f16"}, 160000)
        self.assertAlmostEqual(kv_gb, 4.88, places=1)

    def test_u8_roughly_halves_the_same_estimate(self):
        f16 = expected_kv_gb(_QWEN35_9B, 0, {"KV_CACHE_PRECISION": "f16"}, 160000)
        u8 = expected_kv_gb(_QWEN35_9B, 0, {"KV_CACHE_PRECISION": "u8"}, 160000)
        self.assertLess(u8, f16 * 0.6)

    def test_unknown_architecture_yields_no_estimate(self):
        self.assertIsNone(expected_kv_gb(None, 0, {}, 160000))
        self.assertIsNone(expected_kv_gb({"num_hidden_layers": 4}, 0, {}, 160000))


class TestLinearAttentionReservationIsPerSequence(unittest.TestCase):
    """The paged backend reserves one linear-attention state per schedulable sequence out of
    the same `cache_size` pool as the KV blocks, so `max_num_seqs` is a cache-sizing knob on a
    hybrid model. Leaving it at genai's default of 256 is what dropped a 160K request as
    out-of-memory against an 8 GiB pool -- 256 x 51 MiB is 12.75 GB of reservation alone."""

    # Qwen3.5-9B's real conv+ssm state, as fixed_state_cache_bytes() reads it from the IR.
    FIXED_BYTES = 53477376  # 51 MiB

    def _kv_gb(self, sequences):
        return expected_kv_gb(
            _QWEN35_9B, self.FIXED_BYTES, {"KV_CACHE_PRECISION": "f16"}, 160000,
            sequences=sequences,
        )

    def test_one_sequence_adds_a_single_reservation(self):
        self.assertAlmostEqual(self._kv_gb(1), 4.88 + 51 / 1024, places=2)

    def test_the_genai_default_of_256_dwarfs_the_kv_itself(self):
        # The regression this guards: an estimate blind to max_num_seqs called an 8 GiB pool
        # sufficient for a configuration needing ~17.6 GB, and the run failed after loading.
        self.assertGreater(self._kv_gb(256), 17)
        with self.assertRaises(ValueError):
            validate_fixed_cache_size(8, self._kv_gb(256))
        validate_fixed_cache_size(8, self._kv_gb(1))

    def test_a_dense_model_is_unaffected_by_the_sequence_count(self):
        # No linear-attention layers means no fixed state to reserve, so max_num_seqs stays
        # a pure batch-size knob and the estimate must not inflate with it.
        dense = {"num_hidden_layers": 8, "num_key_value_heads": 4, "head_dim": 128}
        self.assertEqual(
            expected_kv_gb(dense, 0, {}, 160000, sequences=256),
            expected_kv_gb(dense, 0, {}, 160000, sequences=1),
        )


class TestAutoCacheSize(unittest.TestCase):
    def test_lands_near_the_hand_tuned_value_for_qwen35_9b_f16_at_160k(self):
        kv_gb = expected_kv_gb(_QWEN35_9B, 0, {"KV_CACHE_PRECISION": "f16"}, 160000)

        cache_size = auto_cache_size_gb(kv_gb, weight_disk_gb=8.8, budget_gb=59)

        # cache_size=8 passed, 16 pushed peak GPU to 34.2 GB and 24 was rejected as
        # larger than available memory -- so the derived value has to sit just above
        # the ~4.9 GB the cache actually needs, not multiples of it.
        self.assertGreaterEqual(cache_size, kv_gb)
        self.assertLessEqual(cache_size, 9)

    def test_shrinks_with_a_compressed_cache(self):
        f16 = auto_cache_size_gb(
            expected_kv_gb(_QWEN35_9B, 0, {"KV_CACHE_PRECISION": "f16"}, 160000), 8.8, 59
        )
        u8 = auto_cache_size_gb(
            expected_kv_gb(_QWEN35_9B, 0, {"KV_CACHE_PRECISION": "u8"}, 160000), 8.8, 59
        )
        self.assertLess(u8, f16)

    def test_grows_with_context_length(self):
        at_160k = auto_cache_size_gb(expected_kv_gb(_QWEN35_9B, 0, {}, 160000), 8.8, 59)
        at_256k = auto_cache_size_gb(expected_kv_gb(_QWEN35_9B, 0, {}, 256000), 8.8, 59)
        self.assertGreater(at_256k, at_160k)

    def test_leaves_room_for_the_weights_it_shares_a_budget_with(self):
        # The 35B export is 33 GB of weights inside the same 59 GB shared budget, so the
        # pool cannot be sized as if it had the whole device to itself.
        huge = auto_cache_size_gb(kv_gb=40.0, weight_disk_gb=33.0, budget_gb=59)
        self.assertLessEqual(huge, 59 - 33)

    def test_never_returns_a_pool_too_small_to_build(self):
        self.assertGreaterEqual(auto_cache_size_gb(0.01, weight_disk_gb=1.0, budget_gb=59), 2)
        # Even when the weights have already eaten the entire budget, the floor holds
        # rather than emitting a zero or negative cache_size the scheduler would reject.
        self.assertGreaterEqual(auto_cache_size_gb(5.0, weight_disk_gb=58.0, budget_gb=59), 2)

    def test_unknown_architecture_leaves_the_pool_to_openvino(self):
        # None means "drop cache_size" -- letting the runtime manage the pool is honest,
        # guessing a number for an architecture we cannot size is not.
        self.assertIsNone(auto_cache_size_gb(None, weight_disk_gb=8.8, budget_gb=59))


class TestFixedCacheSizeValidation(unittest.TestCase):
    def test_accepts_a_fixed_pool_above_the_estimate(self):
        validate_fixed_cache_size(cache_size=3, kv_gb=1.61)

    def test_rejects_a_pool_smaller_than_persistent_kv(self):
        with self.assertRaisesRegex(ValueError, "below the estimated 1.61 GB"):
            validate_fixed_cache_size(cache_size=1.5, kv_gb=1.61)

    def test_auto_and_unknown_estimates_are_left_to_runtime_resolution(self):
        validate_fixed_cache_size(cache_size="auto", kv_gb=1.61)
        validate_fixed_cache_size(cache_size=1, kv_gb=None)


class TestCacheSizePreflight(unittest.TestCase):
    """The same check, hoisted ahead of the model. Enforced only per case it would abort a
    six-case run during case three, after hours already spent -- the worst of both fail-fast
    and keep-going, for a configuration rejectable by arithmetic in the first second."""

    STATIC = {
        "model_config": _QWEN35_9B,
        "fixed_state_bytes": 0,
        "weight_disk_gb": 8.8,
    }

    def _settings(self, scheduler, contexts=(160000,)):
        return {
            "context_tokens": list(contexts),
            "profiles": [{"name": "paged_min", "ov": {"KV_CACHE_PRECISION": "f16"},
                          "scheduler": scheduler}],
        }

    def test_an_undersized_fixed_pool_names_the_profile_model_and_context(self):
        settings = self._settings({"cache_size": 2, "max_num_seqs": 1})

        with self.assertRaises(SystemExit) as caught:
            benchmark._preflight_cache_sizes(settings, self.STATIC, "Qwen/Qwen3.5-9B")

        message = str(caught.exception)
        for expected in ("Qwen/Qwen3.5-9B", "paged_min", "160,000", "cache_size=2"):
            self.assertIn(expected, message)

    def test_a_context_that_only_the_longest_case_overflows_is_still_caught(self):
        # The short case would pass and the long one abort mid-run, so every context in the
        # matrix has to be checked, not just the first.
        settings = self._settings({"cache_size": 6, "max_num_seqs": 1}, contexts=(8000, 256000))

        with self.assertRaises(SystemExit) as caught:
            benchmark._preflight_cache_sizes(settings, self.STATIC, "Qwen/Qwen3.5-9B")

        self.assertIn("256,000", str(caught.exception))

    def test_the_max_num_seqs_reservation_counts_against_the_pool(self):
        # 8 GiB covers the 4.88 GiB of KV, but not 256 linear-attention reservations on top.
        static = dict(self.STATIC, fixed_state_bytes=53477376)

        benchmark._preflight_cache_sizes(
            self._settings({"cache_size": 8, "max_num_seqs": 1}), static, "Qwen/Qwen3.5-9B"
        )
        with self.assertRaises(SystemExit):
            benchmark._preflight_cache_sizes(
                self._settings({"cache_size": 8, "max_num_seqs": 256}), static, "Qwen/Qwen3.5-9B"
            )

    def test_auto_and_stateful_profiles_have_nothing_to_preflight(self):
        for scheduler in ({"cache_size": "auto"}, {}, {"max_num_batched_tokens": 32768}):
            with self.subTest(scheduler=scheduler):
                benchmark._preflight_cache_sizes(
                    self._settings(scheduler), self.STATIC, "Qwen/Qwen3.5-9B"
                )


class TestShippedConfigs(unittest.TestCase):
    """The shipped configs are the single source of truth for the run matrix, so these
    assert the values as configured rather than the values history recommends -- the point
    of the benchmark is to re-measure candidates. What they do pin is the invariants the
    measurement depends on: exact 160K, no prefix caching, and -- now that each config ships a
    stateful/paged pair -- that the pipeline really is the *only* difference between the two.
    A stray KV-precision edit to one of the pair would silently turn the comparison into a
    two-variable experiment whose TTFT gap nothing in the report could attribute."""

    CONFIGS = {
        "config_qwen3.5_9b.yaml": "Qwen/Qwen3.5-9B",
        "config_qwen3.6_35b_a3b.yaml": "Qwen/Qwen3.6-35B-A3B",
    }

    def _profiles(self, filename) -> list:
        path = Path(__file__).parents[1] / "llm" / "context_bench" / filename
        return load_config(str(path)).profiles

    def test_each_config_ships_a_measurable_160k_pipeline_pair(self):
        for filename, model in self.CONFIGS.items():
            with self.subTest(config=filename):
                path = Path(__file__).parents[1] / "llm" / "context_bench" / filename
                config = load_config(str(path))

                self.assertEqual(config.benchmark.models, [model])
                self.assertEqual(config.benchmark.context_tokens, [160000])
                # At least 2 output tokens, or there is no decode phase to average.
                self.assertGreaterEqual(config.benchmark.output_tokens, 2)
                self.assertEqual(
                    [profile["name"] for profile in config.profiles], ["stateful", "paged_min"]
                )

    def test_only_the_pipeline_differs_between_the_two_profiles(self):
        for filename in self.CONFIGS:
            with self.subTest(config=filename):
                stateful, paged = self._profiles(filename)

                self.assertEqual(stateful["ov"], paged["ov"])
                # A precision this tool can size, so `expected_kv_gb` is not silently
                # falling back to fp16 for both rows of the comparison.
                self.assertIn(stateful["ov"]["KV_CACHE_PRECISION"], _KV_PRECISION_BYTES)

    def test_the_stateful_profile_configures_no_scheduler_at_all(self):
        # Any SchedulerConfig selects continuous batching, so a single scheduler key here
        # would quietly make both profiles paged and the pair meaningless.
        for filename in self.CONFIGS:
            with self.subTest(config=filename):
                stateful = self._profiles(filename)[0]

                self.assertFalse(stateful.get("scheduler"))
                self.assertEqual(pipeline_mode(stateful.get("scheduler")), PIPELINE_STATEFUL)

    def test_the_paged_profile_caps_the_pool_and_leaves_prefix_caching_off(self):
        for filename in self.CONFIGS:
            with self.subTest(config=filename):
                scheduler = self._profiles(filename)[1]["scheduler"]

                self.assertEqual(pipeline_mode(scheduler), PIPELINE_PAGED)
                # A fixed cap, not `auto` and not omitted: bounding the KV pool is the whole
                # reason this profile is paged rather than stateful.
                cache_size = scheduler["cache_size"]
                self.assertIsInstance(cache_size, (int, float))
                self.assertGreater(cache_size, 0)
                # A positive prefill chunk, not a specific one: this is the value the
                # benchmark exists to sweep, so pinning it here would make every
                # measurement of a new candidate a test failure. What matters is that it is
                # set at all -- OpenVINO's default of 256 would prefill 160K in ~625 chunks.
                chunk = scheduler["max_num_batched_tokens"]
                self.assertIsInstance(chunk, int)
                self.assertGreater(chunk, 256)
                # Pinned, not tuning: both shipped models are hybrid, and on the paged path
                # every schedulable sequence costs one full linear-attention reservation out
                # of `cache_size`. genai's default of 256 reserves 12.75 GB on the 9B alone
                # and the 160K request is dropped after the model has loaded.
                self.assertEqual(scheduler["max_num_seqs"], 1)
                # The prompt is reused across iterations, so a warm prefix cache would
                # report a TTFT no first request ever sees.
                self.assertFalse(scheduler["enable_prefix_caching"])


class TestFixedStateCacheBytes(unittest.TestCase):
    def test_reads_fixed_conv_and_ssm_states_from_ir(self):
        # Only conv/ssm states are fixed-size; the key/value state has a dynamic
        # sequence axis and belongs to the growing estimate, not this one.
        xml = """<net><layers><layer><data
            variable_id="cache_params.past.conv.0cache_params.present.conv.0"
            variable_type="f32" variable_shape="?,8,4" /></layer><layer><data
            variable_id="cache_params.past.ssm.0cache_params.present.ssm.0"
            variable_type="f32" variable_shape="?,2,4,4" /></layer><layer><data
            variable_id="cache_params.past.key.0cache_params.present.key.0"
            variable_type="f32" variable_shape="?,4,?,256" /></layer></layers></net>"""
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, "openvino_model.xml").write_text(xml, encoding="utf-8")
            self.assertEqual(fixed_state_cache_bytes(model_dir), (8 * 4 + 2 * 4 * 4) * 4)

    def test_missing_ir_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as model_dir:
            self.assertEqual(fixed_state_cache_bytes(model_dir), 0)


class TestIrReadiness(unittest.TestCase):
    def test_requires_a_supported_root_model_and_both_tokenizer_irs(self):
        with tempfile.TemporaryDirectory() as model_dir:
            root = Path(model_dir)
            (root / "openvino_tokenizer.xml").touch()
            (root / "openvino_detokenizer.xml").touch()
            (root / "openvino_vision_model.xml").touch()
            self.assertFalse(_ir_ready(model_dir))

            (root / "openvino_language_model.xml").touch()
            self.assertTrue(_ir_ready(model_dir))


if __name__ == "__main__":
    unittest.main()
