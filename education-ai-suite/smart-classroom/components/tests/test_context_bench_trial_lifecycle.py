# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the benchmark child's exit path and the parent's recovery.

These pin the fix for a crash observed on the 64 GB shared-memory iGPU box: the
160K case for Qwen/Qwen3.5-9B finished its work, then aborted the whole child
process with exit code 3221226505 (0xC0000409) while *destroying* the OpenVINO
pipeline --

    openvino_genai.dll!ov::genai::VLMPipeline::~VLMPipeline
      -> openvino.dll!ov::IAsyncInferRequest::~IAsyncInferRequest
      -> openvino_intel_gpu_plugin.dll!...
      -> openvino.dll!ov::Exception::create      (throws out of a destructor)
      -> ucrtbase.dll!terminate

-- and, because the teardown ran before the result was posted, took the
already-computed measurement with it. The run could then only report `crashed`
for the one context length the whole tool exists to measure.

The fix has two halves, one per test class below: the child reports first and
never runs the throwing teardown, and the parent recovers a result that raced
the child's exit instead of calling it a crash.
"""

import inspect
import tempfile
import unittest
from pathlib import Path
from queue import Empty
from types import SimpleNamespace
from unittest import mock

from components.llm.context_bench import benchmark, trial_runner

_FAKE_MEM = {
    "ram_gb": 10.0,
    "ram_pct": 15.6,
    "available_ram_gb": 54.0,
    "gpu_gb": 1.0,
}


def _executable_source(func) -> str:
    """Source of `func` with comments and the docstring stripped.

    The tests below assert that statements like `del pipe` are gone; the comments
    explaining *why* they're gone naturally mention them, so match on code only.
    """
    body = inspect.getsource(func)
    body = body.replace(inspect.getdoc(func) or "", "")
    return "\n".join(line for line in body.splitlines() if not line.lstrip().startswith("#"))


class _RecordingQueue:
    """Records the exact order of queue operations and the process exit."""

    def __init__(self, put_error=None):
        self.calls = []
        self._put_error = put_error

    def put(self, message):
        self.calls.append(("put", message))
        if self._put_error is not None:
            raise self._put_error

    def close(self):
        self.calls.append(("close", None))

    def join_thread(self):
        self.calls.append(("join_thread", None))


class TestChildReportsBeforeTeardown(unittest.TestCase):
    def test_result_is_flushed_then_process_exits(self):
        queue = _RecordingQueue()
        done = {"event": "done", "iterations": []}

        with mock.patch.object(
            trial_runner.os, "_exit", side_effect=lambda code: queue.calls.append(("exit", code))
        ):
            trial_runner._post_and_exit(queue, done)

        # put before close/join_thread (Queue.put is asynchronous -- os._exit skips the
        # atexit hook that normally waits for the feeder thread), and exit strictly last.
        self.assertEqual(
            queue.calls, [("put", done), ("close", None), ("join_thread", None), ("exit", 0)]
        )

    def test_exits_even_if_reporting_fails(self):
        queue = _RecordingQueue(put_error=OSError("pipe closed"))

        with mock.patch.object(trial_runner.os, "_exit") as fake_exit, mock.patch.object(
            trial_runner.traceback, "print_exc"
        ):
            trial_runner._post_and_exit(queue, {"event": "done"})

        fake_exit.assert_called_once_with(0)

    def test_run_case_never_destroys_the_pipeline(self):
        code = _executable_source(trial_runner.run_case)

        # `del pipe` is the statement whose destructor threw and aborted the process.
        self.assertNotIn("del pipe", code)
        self.assertNotIn("gc.collect", code)
        self.assertFalse(
            hasattr(trial_runner, "gc"),
            "trial_runner must not force collection; process exit is the reclamation boundary",
        )

    def test_every_exit_path_reports_before_exiting(self):
        code = _executable_source(trial_runner.run_case)

        # The load-failure path and the normal path, and no plain put() that would leave
        # the result exposed to a teardown abort afterwards.
        self.assertEqual(code.count("_post_and_exit(result_queue, done)"), 2)
        self.assertNotIn("result_queue.put(done)", code)

    def test_prompt_count_mismatch_is_checked_before_the_prompt_milestone(self):
        code = _executable_source(trial_runner.run_case)

        mismatch = code.index("if hf_tokens != context_tokens or prompt_tokens != context_tokens:")
        milestone = code.index('done["stage_reached"] = STAGE_PROMPT_BUILT')
        event = code.index('result_queue.put({"event": "prompt"')

        self.assertLess(mismatch, milestone)
        self.assertLess(mismatch, event)

    def test_prompt_is_built_once_outside_the_iteration_loop(self):
        # Rebuilding it per iteration would charge tokenization to every measurement and
        # break the like-for-like comparison llm_bench's repeated-prompt design gives.
        code = _executable_source(trial_runner.run_case)

        self.assertLess(code.index("build_benchmark_prompt"), code.index("for index in range("))


class _DeadProcess:
    """A child that is already gone by the time the parent first polls."""

    def __init__(self, exitcode=0):
        self.exitcode = exitcode
        self.terminated = False

    def start(self):
        pass

    def is_alive(self):
        return False

    def terminate(self):
        self.terminated = True

    def join(self, timeout=None):
        pass


class _RacingQueue:
    """Empty on the first poll, then yields the child's messages.

    This is the ordering the child produces on every case: it posts `done` and
    immediately calls os._exit(), so the message can land in the pipe in the window
    between the parent's poll timing out and is_alive() going False.
    """

    def __init__(self, messages):
        self._messages = list(messages)
        self._polled = False

    def get(self, timeout=None):
        if not self._polled:
            self._polled = True
            raise Empty
        if not self._messages:
            raise Empty
        return self._messages.pop(0)


class _FakeContext:
    def __init__(self, queue, process):
        self._queue = queue
        self._process = process

    def Queue(self):  # noqa: N802 - mirrors multiprocessing's context API
        return self._queue

    def Process(self, target=None, args=()):  # noqa: N802 - ditto
        return self._process


class _ParentHarness(unittest.TestCase):
    def _run(self, queue, process, **kwargs):
        ctx = _FakeContext(queue, process)
        with mock.patch.object(
            benchmark.multiprocessing, "get_context", return_value=ctx
        ), mock.patch.object(benchmark, "_read_mem", return_value=dict(_FAKE_MEM)):
            return benchmark._run_case_subprocess(
                "models/openvino/Qwen_Qwen3.5-9B_int8",
                "GPU",
                160000,
                64,
                warmup=1,
                iterations=3,
                timeout_sec=30,
                ov_config={},
                scheduler_config={},
                sample_interval=0.01,
                poll_interval=0.01,
                drain_timeout=0.5,
                **kwargs,
            )


def _iteration(index, warmup=False, generation_time=250.0):
    return {
        "event": "iteration",
        **trial_runner.metrics.iteration_record(
            iteration=index,
            input_size=160000,
            output_size=64,
            generation_time=generation_time,
            first_token_latency=generation_time * 1000 * 0.96,
            warmup=warmup,
        ),
    }


class TestParentRecoversResultRacingChildExit(_ParentHarness):
    def test_child_progress_refreshes_the_timeout(self):
        code = _executable_source(benchmark._run_case_subprocess)

        self.assertEqual(code.count("deadline = time.monotonic() + timeout_sec"), 2)
        self.assertLess(
            code.index("finished = _consume(msg, child_alive=True)"),
            code.rindex("deadline = time.monotonic() + timeout_sec"),
        )

    def test_completed_case_is_not_reported_as_a_crash(self):
        done = {
            "event": "done",
            "context_tokens": 160000,
            "load_ok": True,
            "load_time_s": 13.1,
            "prompt_tokens": 160000,
            "stage_reached": trial_runner.STAGE_DECODED,
            "iterations": [],
            "error": None,
        }
        queue = _RacingQueue(
            [
                {"event": "device", "gpu_budget_gb": 33.62},
                {"event": "loaded", "load_time_s": 13.1},
                {"event": "prompt", "prompt_tokens": 160000},
                {"event": "iteration_start", "iteration": 0, "warmup": True},
                _iteration(0, warmup=True, generation_time=900.0),
                {"event": "iteration_start", "iteration": 1, "warmup": False},
                _iteration(1),
                done,
            ]
        )

        result = self._run(queue, _DeadProcess(exitcode=0))

        self.assertIsNone(result["error"])
        self.assertEqual(benchmark._status(result), "ok")
        self.assertEqual(len(result["iterations"]), 2)
        self.assertNotIn("event", result)
        # The driver's reported budget is kept as a reference column, under a name that
        # cannot be mistaken for the configured budget the tool actually enforces.
        self.assertEqual(result["gpu_budget_driver_gb"], 33.62)
        self.assertNotIn("gpu_budget_gb", result)

    def test_only_measured_iterations_reach_the_aggregate(self):
        queue = _RacingQueue(
            [
                {"event": "loaded", "load_time_s": 13.1},
                {"event": "iteration_start", "iteration": 0, "warmup": True},
                _iteration(0, warmup=True, generation_time=900.0),
                {"event": "iteration_start", "iteration": 1, "warmup": False},
                _iteration(1, generation_time=250.0),
                {"event": "iteration_start", "iteration": 2, "warmup": False},
                _iteration(2, generation_time=250.0),
                {"event": "done", "load_ok": True, "iterations": [], "error": None},
            ]
        )

        result = self._run(queue, _DeadProcess(exitcode=0))
        aggregate = trial_runner.metrics.aggregate(result["iterations"])

        self.assertEqual(aggregate["iterations_measured"], 2)
        self.assertEqual(aggregate["generation_time"], 250.0)

    def test_each_iteration_carries_its_own_memory_window(self):
        queue = _RacingQueue(
            [
                {"event": "loaded", "load_time_s": 13.1},
                {"event": "iteration_start", "iteration": 0, "warmup": False},
                _iteration(0),
                {"event": "done", "load_ok": True, "iterations": [], "error": None},
            ]
        )

        result = self._run(queue, _DeadProcess(exitcode=0))

        self.assertEqual(result["iterations"][0]["peak_ram_gb"], _FAKE_MEM["ram_gb"])
        self.assertEqual(result["iterations"][0]["peak_gpu_gb"], _FAKE_MEM["gpu_gb"])

    def test_window_restarts_at_the_reset_while_the_case_peak_persists(self):
        """A warm-up's spike must not be charged to the measured iteration, and clearing the
        window must not clear the case-wide peak. Does not reproduce the fold/reset race
        itself -- that needs an interleaving inside `_fold`, which the lock added there
        excludes by construction -- but it pins the two accumulators' separation."""
        sampler = benchmark._MemorySampler(interval=3600)  # never ticks on its own

        spike = {"ram_gb": 60.0, "gpu_gb": 50.0, "ram_pct": 95.0, "available_ram_gb": 1.0}
        sampler._observe(spike)
        window = sampler.window()
        self.assertEqual((window["peak_ram_gb"], window["peak_gpu_gb"]), (60.0, 50.0))

        quiet = {"ram_gb": 20.0, "gpu_gb": 10.0, "ram_pct": 30.0, "available_ram_gb": 40.0}
        with mock.patch.object(benchmark, "_read_mem", return_value=quiet):
            sampler.reset_window()
        # A sample arriving after the reset folds against the *reset* baseline, so the
        # window reports this iteration's peak, not the previous one's.
        sampler._observe(quiet)

        window = sampler.window()
        self.assertEqual((window["peak_ram_gb"], window["peak_gpu_gb"]), (20.0, 10.0))
        # The mean is windowed too: the spike stood only before the reset, so this
        # iteration's mean must report the quiet level, not an average of the two.
        self.assertEqual((window["mean_ram_gb"], window["mean_gpu_gb"]), (20.0, 10.0))
        # The case-wide peak is a separate accumulator and must still remember the spike.
        self.assertEqual(sampler.peak_ram, 60.0)
        self.assertEqual(sampler.peak_gpu, 50.0)

    def test_post_mortem_loaded_event_sets_load_ok_without_a_memory_snapshot(self):
        # A post-load snapshot taken after the child is gone would measure a dead
        # process, so the drain records loading and leaves growth columns empty.
        queue = _RacingQueue(
            [
                {"event": "loaded", "load_time_s": 13.1},
                {"event": "done", "load_ok": True, "iterations": [], "error": "prefill:oom:no memory"},
            ]
        )

        result = self._run(queue, _DeadProcess(exitcode=0))

        self.assertTrue(result["load_ok"])
        self.assertIsNone(result["post_load_peak_ram_gb"])
        self.assertIsNone(result["post_load_peak_gpu_gb"])


class TestGenuineFailuresAreStillNamed(_ParentHarness):
    def test_child_that_dies_without_reporting_is_a_crash(self):
        result = self._run(_RacingQueue([]), _DeadProcess(exitcode=3221226505))

        self.assertFalse(result["load_ok"])
        self.assertEqual(benchmark._status(result), "crashed")

    def test_crash_reason_decodes_native_abort_exit_codes(self):
        reason = benchmark._crash_reason(3221226505)

        self.assertTrue(reason.startswith("crashed:exitcode=3221226505:"))
        self.assertIn("0xC0000409", reason)
        # The decoration must not disturb the status the report keys off.
        self.assertEqual(benchmark._status({"error": reason}), "crashed")

    def test_unknown_exit_code_is_left_undecorated(self):
        self.assertEqual(benchmark._crash_reason(1), "crashed:exitcode=1")

    def test_partial_iterations_survive_a_child_that_then_crashes(self):
        # A case that measured two iterations and died on the third still has two real
        # measurements; discarding them would throw away the evidence of where it broke.
        queue = _RacingQueue(
            [
                {"event": "loaded", "load_time_s": 13.1},
                {"event": "iteration_start", "iteration": 0, "warmup": False},
                _iteration(0),
                {"event": "iteration_start", "iteration": 1, "warmup": False},
                _iteration(1),
            ]
        )

        result = self._run(queue, _DeadProcess(exitcode=3221226505))

        self.assertEqual(len(result["iterations"]), 2)
        self.assertEqual(result["stage_reached"], trial_runner.STAGE_DECODED)

    def test_first_token_milestone_identifies_a_native_decode_crash(self):
        queue = _RacingQueue(
            [
                {"event": "loaded", "load_time_s": 13.1},
                {"event": "prompt", "prompt_tokens": 160000},
                {"event": "iteration_start", "iteration": 0, "warmup": False},
                {"event": "prefilled", "iteration": 0},
            ]
        )

        result = self._run(queue, _DeadProcess(exitcode=3221226505))

        self.assertEqual(result["stage_reached"], trial_runner.STAGE_PREFILLED)
        self.assertEqual(trial_runner.failing_stage(result["stage_reached"]), "decode")

    def test_status_distinguishes_the_failures_that_change_what_to_do_next(self):
        for error, expected in (
            ("timeout", "timeout"),
            ("load:oom:allocation failed", "oom"),
            ("prefill:gpu_abort:CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST", "gpu_abort"),
            ("load:unsupported:KV_CACHE_PRECISION u4 is not supported", "unsupported"),
            ("load:error:something else", "load_error"),
        ):
            self.assertEqual(benchmark._status({"error": error, "load_ok": False}), expected)

    def test_int4_kv_on_an_older_runtime_is_unsupported_not_a_hardware_limit(self):
        # Reporting this as `oom` would read as "this box cannot do int4 KV at 160K",
        # when the runtime simply does not implement the property.
        exc = ValueError("Property KV_CACHE_PRECISION with value u4 is not supported by GPU plugin")

        self.assertEqual(trial_runner.classify_error(exc), "unsupported")

    def test_scheduler_rejecting_an_oversized_cache_is_an_oom(self):
        # The literal message from cache_size=24 at 160K. It is a memory verdict, and
        # naming it anything else hides why that profile cannot run on this box.
        exc = RuntimeError("Requested cache size is larger than available memory size on the system.")

        self.assertEqual(trial_runner.classify_error(exc), "oom")

    def test_prefill_failure_names_the_stage_that_was_running(self):
        # Off-by-one on purpose: the prompt was built and no token streamed, so the
        # failure belongs to prefill, not to the prompt_built milestone it reached.
        self.assertEqual(trial_runner.failing_stage(trial_runner.STAGE_PROMPT_BUILT), "prefill")
        self.assertEqual(trial_runner.failing_stage(trial_runner.STAGE_PREFILLED), "decode")
        self.assertEqual(trial_runner.failing_stage(trial_runner.STAGE_START), "load")


class TestMemorySettlesBeforeTheNextCase(unittest.TestCase):
    """Reclamation is the OS's job (the child skips OpenVINO's teardown) and the Windows
    PDH GPU counters lag it, so the next case's baseline must not be read until usage
    comes back down -- otherwise the previous case's memory is charged to the next
    case's weights."""

    def test_returns_once_usage_is_back_near_baseline(self):
        readings = [
            {"ram_gb": 55.0, "gpu_gb": 40.0},
            {"ram_gb": 30.0, "gpu_gb": 12.0},
            {"ram_gb": 10.4, "gpu_gb": 1.2},
        ]

        with mock.patch.object(benchmark, "_read_mem", side_effect=readings), mock.patch.object(
            benchmark.time, "sleep"
        ):
            settled = benchmark._wait_for_memory_settle({"ram_gb": 10.0, "gpu_gb": 1.0}, timeout_sec=30)

        self.assertTrue(settled)

    def test_unmeasurable_memory_does_not_stall_every_case(self):
        # No psutil and no GPU counter means there is nothing to wait for. Blocking the
        # full timeout per case would only slow a run down on a box where the numbers
        # were already going to come back as `measurement_error`.
        with mock.patch.object(
            benchmark, "_read_mem", return_value={"ram_gb": None, "gpu_gb": None}
        ), mock.patch.object(benchmark.time, "sleep") as no_sleep:
            settled = benchmark._wait_for_memory_settle(
                {"ram_gb": None, "gpu_gb": None}, timeout_sec=30
            )

        self.assertTrue(settled)
        no_sleep.assert_not_called()

    def test_gives_up_instead_of_stalling_the_run(self):
        with mock.patch.object(
            benchmark, "_read_mem", return_value={"ram_gb": 40.0, "gpu_gb": 20.0}
        ), mock.patch.object(benchmark.time, "sleep"):
            settled = benchmark._wait_for_memory_settle({"ram_gb": 10.0, "gpu_gb": 1.0}, timeout_sec=0.05)

        self.assertFalse(settled)


class TestGeneratedOutputIsNotQualityJudged(unittest.TestCase):
    """A capacity/throughput probe must not turn model behaviour into a hardware verdict:
    punctuation, repetition or an immediate EOS still prove prefill completed and decode
    ran. Only producing nothing at all is a failure."""

    def test_no_tokens_generated_raises_rather_than_reporting_a_zero_rate(self):
        code = _executable_source(trial_runner.run_case)

        self.assertIn("no_output", code)
        # The token count comes from the runtime or the tokenizer -- never from
        # inspecting whether the text looks like a good answer.
        self.assertNotIn("strip()", code)


class TestMtpIsRejectedBeforeTheModelLoads(unittest.TestCase):
    """openvino_genai enforces all of these too -- inside `mtp_strategy.cpp`, after the
    pipeline has loaded. On a 14 GB int4 export that is a minute per case spent to learn
    the profile was never runnable, and the same minute again for every context in the
    matrix. These checks are filesystem and dict reads, so they cost nothing."""

    def test_a_model_without_a_draft_head_names_the_missing_file(self):
        with tempfile.TemporaryDirectory() as model_dir:
            with self.assertRaises(ValueError) as caught:
                trial_runner.validate_mtp(
                    model_dir, "GPU", {"enabled": True, "num_assistant_tokens": 3},
                    {"cache_size": 4},
                )

            self.assertIn(trial_runner.MTP_MODEL_FILE, str(caught.exception))

    def test_mtp_without_a_scheduler_says_where_the_pipeline_went_wrong(self):
        # A profile with no `scheduler` section is stateful/SDPA, and genai only runs
        # speculative decoding on paged attention off NPU. The error has to say that,
        # because "assertion failed" would send the reader to tune the wrong thing.
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, trial_runner.MTP_MODEL_FILE).touch()

            with self.assertRaises(ValueError) as caught:
                trial_runner.validate_mtp(
                    model_dir, "GPU", {"enabled": True, "num_assistant_tokens": 3}, {}
                )

            message = str(caught.exception)
            self.assertIn("scheduler", message)
            self.assertIn("PA", message)

    def test_npu_is_exempt_because_it_has_its_own_stateful_path(self):
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, trial_runner.MTP_MODEL_FILE).touch()

            trial_runner.validate_mtp(
                model_dir, "NPU", {"enabled": True, "num_assistant_tokens": 3}, {}
            )

    def test_a_profile_with_mtp_off_is_never_held_to_any_of_this(self):
        with tempfile.TemporaryDirectory() as model_dir:
            for mtp in (None, {}, {"enabled": False}):
                with self.subTest(mtp=mtp):
                    trial_runner.validate_mtp(model_dir, "GPU", mtp, {})

    def test_validation_runs_before_the_load_clock_starts(self):
        code = _executable_source(trial_runner.run_case)
        validate_at = code.index("validate_mtp(")

        self.assertLess(validate_at, code.index("_load_pipeline("))
        self.assertLess(validate_at, code.index("time.perf_counter()"))


class TestMtpDraftModelConstruction(unittest.TestCase):
    def test_uses_the_public_notebook_call_without_internal_mode_flags(self):
        fake_genai = SimpleNamespace(
            SchedulerConfig=type("SchedulerConfig", (), {"max_num_seqs": None}),
            draft_model=mock.Mock(return_value="draft"),
            LLMPipeline=mock.Mock(return_value="pipeline"),
        )
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, "openvino_model.xml").touch()
            with mock.patch.dict("sys.modules", {"openvino_genai": fake_genai}):
                trial_runner._load_pipeline(
                    model_dir, "GPU", {}, {"max_num_seqs": 1},
                    {"enabled": True, "device": None},
                )

        fake_genai.draft_model.assert_called_once_with(model_dir, "GPU")


class TestMtpGenerationConfig(unittest.TestCase):
    """The notebook's fresh config and the settings GenAI requires for MTP."""

    def test_candidate_count_and_a_zero_threshold_are_both_set(self):
        # genai's MTP strategy accepts a *static* candidate count only: a non-zero
        # confidence threshold selects the dynamic variant and is rejected outright.
        config = trial_runner.generation_config(
            64, {"enabled": True, "num_assistant_tokens": 3}
        )

        self.assertEqual(config.num_assistant_tokens, 3)
        self.assertEqual(config.assistant_confidence_threshold, 0.0)
        self.assertFalse(config.do_sample)
        self.assertEqual(config.num_return_sequences, 1)

    def test_a_non_mtp_case_uses_the_notebooks_zero_candidate_baseline(self):
        config = trial_runner.generation_config(64)

        self.assertEqual(config.num_assistant_tokens, 0)
        self.assertEqual(config.assistant_confidence_threshold, 0.0)


class TestMtpYieldIsReadFromTheRuntime(unittest.TestCase):
    def test_acceptance_rate_uses_the_public_extended_metric(self):
        result = SimpleNamespace(
            perf_metrics=SimpleNamespace(),
            extended_perf_metrics=SimpleNamespace(
                get_draft_acceptance_rate=lambda: 0.625,
                get_num_draft_tokens=lambda: 80,
                get_num_accepted_tokens=lambda: 50,
            ),
        )

        metrics = trial_runner.read_perf_metrics(result)

        self.assertEqual(metrics["mtp_acceptance_rate"], 0.625)
        self.assertEqual(metrics["mtp_draft_tokens"], 80)
        self.assertEqual(metrics["mtp_accepted_tokens"], 50)

    def test_verification_steps_come_from_the_new_token_time_series(self):
        # There is no public verification-step getter, so the count of main-model passes
        # comes from `raw_metrics.m_new_token_times`: 23 passes for 64 tokens at k=3.
        result = SimpleNamespace(
            perf_metrics=SimpleNamespace(
                raw_metrics=SimpleNamespace(m_new_token_times=[0.0] * 23),
            )
        )

        self.assertEqual(trial_runner.read_perf_metrics(result)["verification_steps"], 23)

    def test_a_runtime_without_that_series_reports_no_steps_rather_than_failing(self):
        result = SimpleNamespace(perf_metrics=SimpleNamespace())

        self.assertNotIn("verification_steps", trial_runner.read_perf_metrics(result))


if __name__ == "__main__":
    unittest.main()
