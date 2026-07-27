# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import unittest

from components.llm.context_validation import validate_long_context
from components.llm.context_validation.trial_runner import _classify_error
from components.llm.context_validation.validate_long_context import (
    TRIAL_CSV_FIELDS,
    _MemorySampler,
    _classify_failure,
    _format_trial_line,
    _host_memory_exhausted,
    _passed,
)

# Verbatim from the 64 GB shared-memory iGPU box at 160K and 144K tokens. OpenCL reports a
# command that died on the device at the next synchronization point, so the text names the wait
# and never names memory -- which is why the old marker list did not recognise it and the step
# that establishes the whole sweep's answer came back as `generate_error`.
_OPENCL_DEVICE_ABORT = (
    "Exception from src\\inference\\src\\cpp\\infer_request.cpp:224:\n"
    "Exception from src\\plugins\\intel_gpu\\src\\runtime\\ocl\\ocl_memory.cpp:591:\n"
    "[GPU] clWaitForEvents, error code: -14 CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST\n"
)


def _successful_result(generate_time_s):
    return {
        "load_ok": True,
        "generate_ok": True,
        "generated_tokens": 10,
        "generate_time_s": generate_time_s,
        "max_generate_time_sec": 120,
        "gpu_memory_at_limit": False,
        "error": None,
    }


def _device_abort_result(min_available_ram_gb, peak_ram_pct, peak_gpu_pct=68.5):
    """A generate-stage OpenCL device abort with the headroom the sampler measured."""
    result = {
        "load_ok": True,
        "generate_ok": False,
        "generated_tokens": 0,
        "generate_time_s": None,
        "max_generate_time_sec": 600,
        "min_available_ram_gb": min_available_ram_gb,
        "peak_ram_pct": peak_ram_pct,
        "peak_gpu_pct": peak_gpu_pct,
        "gpu_memory_at_limit": False,
        "error": f"generate:{_classify_error(RuntimeError(_OPENCL_DEVICE_ABORT))}:"
        f"{_OPENCL_DEVICE_ABORT}",
    }
    result["host_memory_at_limit"] = _host_memory_exhausted(result)
    return result


class TestContextValidationPolicy(unittest.TestCase):
    def test_48k_observed_trial_meets_latency_sla(self):
        result = _successful_result(50.7)
        result["generated_tokens"] = 64

        self.assertTrue(_passed(result))

    def test_slow_trial_without_gpu_pressure_still_passes(self):
        result = _successful_result(254.4)

        self.assertTrue(_passed(result))

    def test_slow_trial_at_gpu_memory_limit_is_too_slow(self):
        result = _successful_result(254.4)
        result["gpu_memory_at_limit"] = True

        self.assertFalse(_passed(result))
        self.assertEqual(_classify_failure(result), "too_slow")

    def test_latency_sla_is_inclusive(self):
        self.assertTrue(_passed(_successful_result(120.0)))


class TestGpuFailuresAreNamed(unittest.TestCase):
    """The ceiling must not be reported as an unexplained error.

    On the 64 GB box the sweep passed 128K and then failed 160K and 144K with the OpenCL
    device abort above. Neither the child's marker list nor the parent's classifier knew the
    OpenCL vocabulary, so both rows read `FAIL (generate_error)` -- indistinguishable from a bug
    in the tool, for the one measurement the tool exists to produce.
    """

    def test_device_abort_is_named_by_the_child_not_guessed_at(self):
        self.assertEqual(_classify_error(RuntimeError(_OPENCL_DEVICE_ABORT)), "gpu_abort")

    def test_explicit_opencl_allocation_failures_are_oom_outright(self):
        for text in (
            "[GPU] CL_MEM_OBJECT_ALLOCATION_FAILURE",
            "clCreateBuffer failed: CL_OUT_OF_HOST_MEMORY",
            "[GPU] error code: -61 CL_INVALID_BUFFER_SIZE",
        ):
            with self.subTest(text=text):
                self.assertEqual(_classify_error(RuntimeError(text)), "oom")

    def test_ordinary_exceptions_stay_ordinary(self):
        self.assertEqual(_classify_error(ValueError("bad chat template")), "exception")

    def test_device_abort_with_no_headroom_left_is_oom(self):
        # The observed 144,000-token row: 1.86 GB free of 64 GB, 97.1% used.
        self.assertEqual(_classify_failure(_device_abort_result(1.86, 97.1)), "oom")

    def test_device_abort_with_headroom_left_is_not_claimed_to_be_oom(self):
        # The observed 160,000-token row: the GPU gave up while the host still had 9.3 GB, so
        # the evidence for memory exhaustion is not there and the tool must not invent it.
        self.assertEqual(_classify_failure(_device_abort_result(9.32, 85.3)), "gpu_abort")

    def test_device_abort_is_never_reported_as_generate_error(self):
        for free_ram, ram_pct in ((1.86, 97.1), (9.32, 85.3)):
            with self.subTest(free_ram=free_ram):
                self.assertNotEqual(
                    _classify_failure(_device_abort_result(free_ram, ram_pct)), "generate_error"
                )

    def test_native_abort_at_the_wall_is_oom_rather_than_an_exit_code(self):
        result = {
            "error": "crashed:exitcode=3221226505:0xC0000409 STATUS_STACK_BUFFER_OVERRUN",
            "min_available_ram_gb": 0.0,
            "peak_ram_pct": 100.0,
        }

        self.assertEqual(_classify_failure(result), "oom")

    def test_native_abort_with_headroom_left_is_still_a_crash(self):
        result = {
            "error": "crashed:exitcode=3221226505:0xC0000409 STATUS_STACK_BUFFER_OVERRUN",
            "min_available_ram_gb": 21.4,
            "peak_ram_pct": 66.0,
        }

        self.assertEqual(_classify_failure(result), "crashed")


class TestHostHeadroomIsTheMemoryPressureSignal(unittest.TestCase):
    """`gpu_memory_at_limit` cannot reach its own threshold on this hardware.

    It divides peak GPU usage by *total system RAM* and fires at 90%. On a shared-memory iGPU
    the host needs the rest of the machine, so every failing trial observed sat between 63% and
    69% -- the flag never fired, `too_slow` was unreachable, and a device abort had no memory
    evidence attached to it. The sampler's headroom low-water marks answer the same question
    directly, and are what the classifier now reads.
    """

    def test_free_ram_at_the_wall_counts_as_pressure(self):
        self.assertTrue(_host_memory_exhausted({"min_available_ram_gb": 1.86, "peak_ram_pct": 97.1}))

    def test_a_passing_step_with_real_headroom_does_not(self):
        # The 128,000-token row, which passed with 17.25 GB free.
        self.assertFalse(
            _host_memory_exhausted({"min_available_ram_gb": 17.25, "peak_ram_pct": 72.9})
        )

    def test_percentage_catches_a_box_where_3gb_free_is_still_roomy(self):
        self.assertTrue(_host_memory_exhausted({"min_available_ram_gb": 3.4, "peak_ram_pct": 97.0}))

    def test_absent_counters_are_not_evidence_of_pressure(self):
        self.assertFalse(_host_memory_exhausted({}))

    def test_latency_breach_under_host_pressure_is_too_slow(self):
        result = _successful_result(700.0)
        result.update(
            peak_gpu_pct=68.5,  # nowhere near the 90%-of-system-RAM line
            gpu_memory_at_limit=False,
            min_available_ram_gb=1.86,
            peak_ram_pct=97.1,
        )
        result["host_memory_at_limit"] = _host_memory_exhausted(result)

        self.assertFalse(_passed(result))
        self.assertEqual(_classify_failure(result), "too_slow")

    def test_host_pressure_is_reported_per_trial(self):
        self.assertIn("host_memory_at_limit", TRIAL_CSV_FIELDS)


class TestNoMemoryGuard(unittest.TestCase):
    """A predicted ceiling is not a measured one.

    An earlier revision refused to spawn a step whose peak RAM it projected past
    a reserve. On the 64 GB iGPU box that logic cancelled the 160K trial -- the
    exact number the sweep exists to produce -- by scaling the previous step's KV
    growth by the token ratio *and* a 1.25 allocator factor, turning 39.9 GB of
    genuinely linear growth into a 65.83 GB projection. Measured growth was
    0.25 GB per 1K tokens across every passing step, so 160K really peaked near
    55.8 GB with ~7.8 GB still free. These tests pin the guard's removal.
    """

    def test_no_projection_helpers_remain(self):
        for name in (
            "_memory_guard_reason",
            "_projected_memory_guard_reason",
            "_memory_guard_result",
            "_PREFLIGHT_GROWTH_SAFETY_FACTOR",
        ):
            self.assertFalse(
                hasattr(validate_long_context, name),
                f"{name} still exists; a projected ceiling must not pre-empt a real trial",
            )

    def test_memory_guard_is_not_a_failure_classification(self):
        result = _successful_result(10.0)
        result.update(generate_ok=False, generated_tokens=0, error="memory_guard:whatever")

        self.assertNotEqual(_classify_failure(result), "memory_guard")

    def test_trial_subprocess_takes_no_headroom_reserve(self):
        import inspect

        params = inspect.signature(validate_long_context._run_trial_subprocess).parameters

        self.assertNotIn("min_memory_headroom_gb", params)


class TestHeadroomIsMeasuredNotEnforced(unittest.TestCase):
    def test_sampler_tracks_low_water_marks(self):
        sampler = _MemorySampler.__new__(_MemorySampler)
        sampler.peak_ram = sampler.peak_ram_pct = sampler.peak_gpu = 0.0
        sampler.min_available_ram = sampler.min_commit_available = None

        for available, commit in ((20.0, 30.0), (7.8, 12.1), (11.2, 18.4)):
            sampler._observe(
                {
                    "ram_gb": 50.0,
                    "ram_pct": 80.0,
                    "gpu_gb": 40.0,
                    "available_ram_gb": available,
                    "commit_available_gb": commit,
                }
            )

        self.assertEqual(sampler.min_available_ram, 7.8)
        self.assertEqual(sampler.min_commit_available, 12.1)

    def test_sampler_tolerates_missing_counters(self):
        sampler = _MemorySampler.__new__(_MemorySampler)
        sampler.peak_ram = sampler.peak_ram_pct = sampler.peak_gpu = 0.0
        sampler.min_available_ram = sampler.min_commit_available = None

        sampler._observe({"ram_gb": 1.0, "ram_pct": 2.0, "gpu_gb": 3.0})

        self.assertIsNone(sampler.min_available_ram)
        self.assertIsNone(sampler.min_commit_available)

    def test_headroom_is_reported_per_trial(self):
        for field in ("min_available_ram_gb", "min_commit_available_gb"):
            self.assertIn(field, TRIAL_CSV_FIELDS)

    def test_console_line_shows_remaining_headroom(self):
        result = _successful_result(150.0)
        result.update(
            generated_tokens=64,
            peak_ram_gb=55.8,
            min_available_ram_gb=7.8,
            min_commit_available_gb=11.4,
        )

        line = _format_trial_line("Qwen/Qwen3.5-9B", 160000, result)

        self.assertIn("160,000 tok -> PASS", line)
        self.assertIn("min free RAM 7.8 GB (commit 11.4 GB)", line)


if __name__ == "__main__":
    unittest.main()