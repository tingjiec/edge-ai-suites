# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the sweep surviving its own subject matter.

The tool exists to push a box until it breaks, so it has to keep working while the box is
breaking. On the 64 GB shared-memory iGPU it did not: the sweep measured 64K/96K/128K as
passing and 160K/144K as failing, then ended during refinement without writing anything.
`summary.md` was left holding the *previous* `--dry-run`, which reported the candidate as a
**PASS at 160,000 tokens** -- the exact opposite of what the twelve minutes of real trials
just before it had measured, and with a plausible-looking timestamp on top.

Two properties keep that from recurring, one per test class: a step the orchestrator itself
cannot run becomes that step's row instead of unwinding the sweep, and the report on disk
always describes the run that is actually happening.
"""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.llm.context_validation import validate_long_context
from components.llm.context_validation.validate_long_context import (
    _classify_failure,
    _passed,
    _run_one,
)

_SETTINGS = {
    "probe_tokens": 64,
    "trial_timeout_sec": 1200,
    "max_generate_time_sec": 600,
    "gpu_memory_pressure_pct": 90,
}


class TestSweepSurvivesAStepItCannotRun(unittest.TestCase):
    def test_orchestrator_side_failure_becomes_that_steps_row(self):
        # Spawning a trial is itself work the box has to find memory for, and refinement runs
        # right after the step that emptied it -- so this side is a real failure point, not a
        # theoretical one.
        boom = OSError("Not enough memory resources are available to process this command")

        with mock.patch.object(
            validate_long_context, "_run_trial_subprocess", side_effect=boom
        ), mock.patch.object(validate_long_context.traceback, "print_exc"):
            result = _run_one(
                "Qwen/Qwen3.5-9B",
                "models/openvino/Qwen_Qwen3.5-9B_int8",
                "GPU",
                136000,
                _SETTINGS,
                dry_run=False,
                fake_ceiling=None,
            )

        self.assertFalse(_passed(result))
        self.assertEqual(_classify_failure(result), "trial_error")
        self.assertEqual(result["tokens_requested"], 136000)
        self.assertIn("Not enough memory resources", result["error"])

    def test_that_row_is_still_a_complete_csv_row(self):
        with mock.patch.object(
            validate_long_context, "_run_trial_subprocess", side_effect=RuntimeError("nope")
        ), mock.patch.object(validate_long_context.traceback, "print_exc"):
            result = _run_one(
                "Qwen/Qwen3.5-9B", "models/x", "GPU", 136000, _SETTINGS, False, None
            )

        with tempfile.TemporaryDirectory() as output_dir:
            validate_long_context._append_trial_row(
                output_dir,
                {"model": "Qwen/Qwen3.5-9B", "device": "GPU", "weight_format": "int8", **result},
            )
            rows = Path(output_dir, "trials.csv").read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(rows), 2)  # header + the failed step
        self.assertIn("trial_error", rows[1])


class TestRefinementKeepsTheCapacityAnswer(unittest.TestCase):
    """A step the orchestrator could not run must not displace a measured capacity failure.

    Refinement runs right after the step that emptied the box, so `trial_error` is most likely
    exactly where a real `oom` has already been measured one step above it.
    """

    _SWEEP = dict(
        _SETTINGS,
        refine=True,
        context_steps_tokens=[128000, 160000],
        device="GPU",
        weight_format="int8",
        provider="openvino",
        models_base_path="models",
        target_context_tokens=160000,
    )

    def _sweep(self, refinement_reason):
        # Steps 128K/160K bisect to exactly one refinement point, 144K.
        outcomes = {128000: "pass", 160000: "oom", 144000: refinement_reason}

        def run_one(_model, _dir, _device, tokens, *_args, **_kwargs):
            outcome = outcomes[tokens]
            passed = outcome == "pass"
            return {
                "tokens_requested": tokens,
                "load_ok": True,
                "generate_ok": passed,
                "generated_tokens": 64 if passed else 0,
                "generate_time_s": 100.0 if passed else None,
                "min_available_ram_gb": 17.2 if passed else 1.86,
                "peak_ram_pct": 72.9 if passed else 97.1,
                "error": None if passed else (
                    "trial_error:OSError:no memory" if outcome == "trial_error"
                    else "generate:gpu_abort:CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"
                ),
            }

        with tempfile.TemporaryDirectory() as output_dir, contextlib.redirect_stdout(io.StringIO()):
            settings = dict(self._SWEEP, output_dir=output_dir)
            with mock.patch.object(validate_long_context, "_run_one", side_effect=run_one), \
                 mock.patch.object(validate_long_context, "_ir_ready", return_value=True), \
                 mock.patch.object(validate_long_context, "_weight_disk_gb", return_value=8.78), \
                 mock.patch.object(validate_long_context, "_load_model_config", return_value=None):
                return validate_long_context._sweep_model("Qwen/Qwen3.5-9B", settings, False)

    def test_unrunnable_refinement_step_leaves_the_measured_reason_in_place(self):
        report = self._sweep("trial_error")

        self.assertEqual(report["max_stable_context"], 128000)
        self.assertEqual(report["failure_reason"], "oom")
        self.assertEqual(report["failure_tokens"], 160000)  # the step that really was measured

    def test_a_real_refinement_failure_does_tighten_the_reported_boundary(self):
        report = self._sweep("oom")

        self.assertEqual(report["max_stable_context"], 128000)
        self.assertEqual(report["failure_reason"], "oom")
        self.assertEqual(report["failure_tokens"], 144000)


def _report(model_name, _settings, _dry_run):
    return {
        "model": model_name,
        "status": "ok",
        "max_stable_context": 128000,
        "meets_target": False,
        "failure_tokens": 144000,
        "failure_reason": "oom",
        "device": "GPU",
        "weight_format": "int8",
    }


class TestReportAlwaysDescribesThisRun(unittest.TestCase):
    def _main(self, output_dir, sweep):
        argv = [
            "validate_long_context",
            "--dry-run",
            "--models",
            "Qwen/Qwen3.5-9B",
            "--steps",
            "64000",
            "--output-dir",
            output_dir,
        ]
        with mock.patch.object(validate_long_context.sys, "argv", argv), mock.patch.object(
            validate_long_context, "_safe_platform_info", return_value={}
        ), mock.patch.object(
            validate_long_context, "_sweep_model", side_effect=sweep
        ), contextlib.redirect_stdout(io.StringIO()):
            validate_long_context.main()

    def _read(self, output_dir):
        return (
            Path(output_dir, "summary.md").read_text(encoding="utf-8"),
            json.loads(Path(output_dir, "summary.json").read_text(encoding="utf-8")),
        )

    def test_a_sweep_that_dies_replaces_the_previous_runs_report(self):
        with tempfile.TemporaryDirectory() as output_dir:
            stale = os.path.join(output_dir, "summary.md")
            Path(stale).write_text(
                "| Qwen/Qwen3.5-9B | GPU | int8 | 160,000 | PASS | 0.0 GB |\n", encoding="utf-8"
            )

            with self.assertRaises(RuntimeError):
                self._main(output_dir, RuntimeError("the box gave up mid-sweep"))

            report, summary = self._read(output_dir)

        self.assertNotIn("160,000 | PASS", report)
        self.assertIn("ended early", report)
        self.assertIn("not run", report)
        self.assertFalse(summary["completed"])
        self.assertEqual([m["status"] for m in summary["models"]], ["not_run"])

    def test_keyboard_interrupt_still_leaves_a_report(self):
        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaises(KeyboardInterrupt):
                self._main(output_dir, KeyboardInterrupt())

            _report_md, summary = self._read(output_dir)

        self.assertFalse(summary["completed"])

    def test_a_finished_sweep_is_marked_complete(self):
        with tempfile.TemporaryDirectory() as output_dir:
            self._main(output_dir, _report)
            report, summary = self._read(output_dir)

        self.assertTrue(summary["completed"])
        self.assertNotIn("ended early", report)
        # Where it broke, not just what capped it -- after refinement the two differ.
        self.assertIn("capped by oom at 144,000 tokens", report)


if __name__ == "__main__":
    unittest.main()
