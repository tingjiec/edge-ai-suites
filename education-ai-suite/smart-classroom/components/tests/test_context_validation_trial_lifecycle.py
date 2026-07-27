# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the trial child's exit path.

These pin the fix for a crash observed on the 64 GB shared-memory iGPU box: the
160K-token trial for Qwen/Qwen3.5-9B finished its work, then aborted the whole
child process with exit code 3221226505 (0xC0000409) while *destroying* the
OpenVINO pipeline --

    openvino_genai.dll!ov::genai::VLMPipeline::~VLMPipeline
      -> openvino.dll!ov::IAsyncInferRequest::~IAsyncInferRequest
      -> openvino_intel_gpu_plugin.dll!...
      -> openvino.dll!ov::Exception::create      (throws out of a destructor)
      -> ucrtbase.dll!terminate

-- and, because the teardown ran before the result was posted, took the trial's
already-computed result with it. The sweep could then only report
`FAIL (crashed)` for the one context length the whole tool exists to measure.

The fix has two halves, one per test class below: the child reports first and
never runs the throwing teardown, and the parent recovers a result that raced
the child's exit instead of calling it a crash.
"""

import inspect
import unittest
from queue import Empty
from unittest import mock

from components.llm.context_validation import trial_runner, validate_long_context
from components.llm.context_validation.validate_long_context import (
    _classify_failure,
    _crash_reason,
    _run_trial_subprocess,
)

_FAKE_MEM = {
    "ram_gb": 10.0,
    "ram_total_gb": 64.0,
    "ram_pct": 15.6,
    "available_ram_gb": 54.0,
    "commit_available_gb": 60.0,
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
        done = {"event": "done", "generate_ok": True}

        with mock.patch.object(
            trial_runner.os, "_exit", side_effect=lambda code: queue.calls.append(("exit", code))
        ):
            trial_runner._post_result_and_exit(queue, done)

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
            trial_runner._post_result_and_exit(queue, {"event": "done"})

        fake_exit.assert_called_once_with(0)

    def test_run_trial_never_destroys_the_pipeline(self):
        code = _executable_source(trial_runner.run_trial)

        # `del pipe` is the statement whose destructor threw and aborted the process.
        self.assertNotIn("del pipe", code)
        self.assertNotIn("gc.collect", code)
        self.assertFalse(
            hasattr(trial_runner, "gc"),
            "trial_runner must not force collection; process exit is the reclamation boundary",
        )

    def test_every_exit_path_reports_before_exiting(self):
        code = _executable_source(trial_runner.run_trial)

        # Both the load-failure path and the generate path, and no plain put() that
        # would leave the result exposed to a teardown abort afterwards.
        self.assertEqual(code.count("_post_result_and_exit(result_queue, done)"), 2)
        self.assertNotIn("result_queue.put(done)", code)


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

    This is the ordering the child now produces on every trial: it posts `done`
    and immediately calls os._exit(), so the message can land in the pipe in the
    window between the parent's poll timing out and is_alive() going False.
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
    def _run(self, queue, process):
        ctx = _FakeContext(queue, process)
        with mock.patch.object(
            validate_long_context.multiprocessing, "get_context", return_value=ctx
        ), mock.patch.object(validate_long_context, "_read_mem", return_value=dict(_FAKE_MEM)):
            return _run_trial_subprocess(
                "Qwen/Qwen3.5-9B",
                "models/openvino/Qwen_Qwen3.5-9B_int8",
                "GPU",
                160000,
                64,
                timeout_sec=30,
                sample_interval=0.01,
                poll_interval=0.01,
                drain_timeout=0.5,
            )


class TestParentRecoversResultRacingChildExit(_ParentHarness):
    def test_completed_trial_is_not_reported_as_a_crash(self):
        done = {
            "event": "done",
            "tokens_requested": 160000,
            "load_ok": True,
            "load_time_s": 13.1,
            "generate_ok": True,
            "prompt_tokens": 159998,
            "generated_tokens": 64,
            "generate_time_s": 300.4,
            "error": None,
        }
        queue = _RacingQueue([{"event": "loaded", "load_time_s": 13.1}, done])

        result = self._run(queue, _DeadProcess(exitcode=0))

        self.assertIsNone(result["error"])
        self.assertTrue(result["generate_ok"])
        self.assertEqual(result["generated_tokens"], 64)
        self.assertEqual(result["prompt_tokens"], 159998)
        self.assertNotIn("event", result)

    def test_post_mortem_loaded_event_sets_load_ok_without_a_memory_snapshot(self):
        # A weight-footprint snapshot taken after the child is gone would measure a
        # dead process, so the drain records that loading happened and leaves the
        # weight columns empty rather than reporting a fabricated number.
        queue = _RacingQueue(
            [
                {"event": "loaded", "load_time_s": 13.1},
                {"event": "done", "load_ok": True, "generate_ok": False, "error": "no_output"},
            ]
        )

        result = self._run(queue, _DeadProcess(exitcode=0))

        self.assertTrue(result["load_ok"])
        self.assertIsNone(result["weight_ram_gb"])
        self.assertIsNone(result["weight_gpu_gb"])


class TestGenuineCrashStillReported(_ParentHarness):
    def test_child_that_dies_without_reporting_is_a_crash(self):
        result = self._run(_RacingQueue([]), _DeadProcess(exitcode=3221226505))

        self.assertFalse(result["load_ok"])
        self.assertFalse(result["generate_ok"])
        self.assertEqual(_classify_failure(result), "crashed")

    def test_crash_reason_decodes_native_abort_exit_codes(self):
        reason = _crash_reason(3221226505)

        self.assertTrue(reason.startswith("crashed:exitcode=3221226505:"))
        self.assertIn("0xC0000409", reason)
        # The decoration must not disturb the classification the sweep keys off.
        self.assertEqual(_classify_failure({"error": reason}), "crashed")

    def test_unknown_exit_code_is_left_undecorated(self):
        self.assertEqual(_crash_reason(1), "crashed:exitcode=1")


class TestMemorySettlesBeforeNextTrial(unittest.TestCase):
    """Reclamation is now the OS's job (the child skips OpenVINO's teardown) and the
    Windows PDH GPU counters lag it, so the next trial's baseline must not be read
    until usage comes back down -- otherwise the previous trial's memory is charged
    to the next trial's weights."""

    def test_returns_once_usage_is_back_near_baseline(self):
        baseline = {"ram_gb": 10.0, "gpu_gb": 1.0}
        readings = [
            {"ram_gb": 55.0, "gpu_gb": 40.0},
            {"ram_gb": 30.0, "gpu_gb": 12.0},
            {"ram_gb": 10.4, "gpu_gb": 1.2},
        ]

        with mock.patch.object(
            validate_long_context, "_read_mem", side_effect=readings
        ), mock.patch.object(validate_long_context.time, "sleep"):
            settled = validate_long_context._wait_for_memory_settle(baseline, timeout_sec=30)

        self.assertTrue(settled)

    def test_gives_up_instead_of_stalling_the_sweep(self):
        baseline = {"ram_gb": 10.0, "gpu_gb": 1.0}

        with mock.patch.object(
            validate_long_context, "_read_mem", return_value={"ram_gb": 40.0, "gpu_gb": 20.0}
        ), mock.patch.object(validate_long_context.time, "sleep"):
            settled = validate_long_context._wait_for_memory_settle(baseline, timeout_sec=0.05)

        self.assertFalse(settled)


if __name__ == "__main__":
    unittest.main()
