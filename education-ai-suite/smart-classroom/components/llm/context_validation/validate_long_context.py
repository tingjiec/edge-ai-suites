# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Sweep candidate summarizer models across context lengths to find the maximum
context each one can reliably prefill and decode on this hardware, versus the
customer's target (default 160K tokens).

What it measures is deliberately narrow (mirroring refer/long_context): whether
*this machine* can load the model and prefill + decode a prompt of a given token
size without running out of GPU/host memory (or hanging). It does NOT judge
answer quality -- content is irrelevant to a capacity check, only whether the
box survives the token volume. Each trial reports where its memory went: the
weight footprint (measured just after load) versus the KV-cache footprint (the
extra memory prefill+decode adds on top). See
docs/dev-guide/validate_long_context.md.

This is a standalone diagnostic tool: it reads its own bundled config.yaml
(next to this script), never smart-classroom/config.yaml. It is independent of
the production summarizer -- running it, or editing its config, never affects
the running application.

Simplest way to run it (handles venv creation/activation, see setup_env.ps1 /
run_validate_long_context.ps1 -- mirrors setup-smart-classroom.ps1 /
start-smart-classroom.ps1's own venv convention):

    .\\components\\llm\\context_validation\\run_validate_long_context.ps1
    .\\components\\llm\\context_validation\\run_validate_long_context.ps1 --dry-run
    .\\components\\llm\\context_validation\\run_validate_long_context.ps1 --models Qwen/Qwen3-8B

Equivalent, if you already have the right interpreter active (run from the
smart-classroom/ directory so relative model paths resolve):

    python -m components.llm.context_validation.validate_long_context

See docs/dev-guide/validate_long_context.md for the full design and usage guide.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import gc
import importlib.util
import json
import multiprocessing
import os
import re
import sys
import threading
import time
import traceback
from datetime import datetime
from queue import Empty

from components.llm.context_validation import trial_runner
from utils.config_loader import load_config
from utils.storage_manager import StorageManager

_TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG_PATH = os.path.join(_TOOL_DIR, "config.yaml")

# smart-classroom/ is 3 levels up from this file's directory
# (context_validation -> llm -> components -> smart-classroom).
_SC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_TOOL_DIR)))
# setup-smart-classroom.ps1 creates the backend venv as a sibling of smart-classroom/,
# named "smartclassroom" (no hyphen) -- see $venvBackend in that script.
_BACKEND_VENV_PYTHON = os.path.join(os.path.dirname(_SC_ROOT), "smartclassroom", "Scripts", "python.exe")
_SETUP_SCRIPT = os.path.join(_TOOL_DIR, "setup_env.ps1")
_LAUNCHER_SCRIPT = os.path.join(_TOOL_DIR, "run_validate_long_context.ps1")

_REQUIRED_MODULES = ("openvino_genai", "transformers")

# Mirrors content_search/providers/utils/model_utils.py::is_model_ready. A converted
# candidate can be a plain causal LM (openvino_model.xml) or a multimodal/VLM export
# (openvino_language_model.xml plus vision-embedding components), so this matches
# either layout instead of assuming the plain-LLM one.
_MODEL_IR_RE = re.compile(r"(.*)?openvino(.*)?_model(.*)?\.xml$")
_TOKENIZER_IR_NAME = "openvino_tokenizer.xml"
_DETOKENIZER_IR_NAME = "openvino_detokenizer.xml"

# Windows exit codes a trial child can die with that mean "a native runtime aborted", not "the
# program chose to exit". Decoded onto the reported error so `crashed:exitcode=3221226505` isn't
# an opaque number -- the tool's whole failure story at the capacity ceiling is told through it.
_NATIVE_ABORT_EXIT_CODES = {
    3221226505: "0xC0000409 STATUS_STACK_BUFFER_OVERRUN - how the CRT reports abort()/std::terminate",
    3221225477: "0xC0000005 STATUS_ACCESS_VIOLATION",
    3221225725: "0xC00000FD STATUS_STACK_OVERFLOW",
}

# After a trial child exits, its RAM/GPU allocations are reclaimed by the OS rather than by
# OpenVINO's own teardown (see trial_runner._post_result_and_exit), and the Windows PDH GPU
# counters this tool reads are themselves sampled, so both lag the process a little. The next
# trial's baseline is taken the moment this one returns, so wait -- briefly and with a cap -- for
# usage to come back down, or the previous trial's memory gets charged to the next trial's weights.
_MEMORY_SETTLE_TIMEOUT_SEC = 30.0
_MEMORY_SETTLE_TOLERANCE_GB = 1.0

# What "the box had nothing left to give" looks like in the low-water marks the sampler already
# records. This is read *after* a trial, from what that trial measured -- it never cancels a trial
# and never predicts one (§3.2.1); it only names the failure a trial actually produced. Both forms
# are needed: the absolute figure catches a large machine where a comfortable-looking percentage
# still hides a wall (63.2% of a 64 GB box was 1.9 GB free when the GPU aborted), and the
# percentage catches a small one where 3 GB free is plenty of room.
_MEMORY_EXHAUSTED_FREE_RAM_GB = 3.0
_MEMORY_EXHAUSTED_RAM_PCT = 95.0

# Measured kv_gpu_gb / expected_kv_gpu_gb at or above this is called out in the summary notes as a
# scope mismatch between the two numbers (persistent-cache-only estimate vs. all post-load growth)
# rather than a plain capacity limit -- see _theoretical_kv_bytes_per_token()'s docstring and the
# note built in _write_summary() for what this can and cannot be blamed on.
_KV_OVERHEAD_RATIO_NOTE_THRESHOLD = 3.0

TRIAL_CSV_FIELDS = [
    "model",
    "tokens_requested",
    "device",
    "weight_format",
    "load_ok",
    "load_time_s",
    "generate_ok",
    "prompt_tokens",
    "generated_tokens",
    "generate_time_s",
    "tokens_per_second",
    "max_generate_time_sec",
    "latency_limit_exceeded",
    "gpu_memory_pressure_pct",
    "peak_gpu_pct",
    "gpu_memory_at_limit",
    "host_memory_at_limit",
    "weight_disk_gb",
    "weight_ram_gb",
    "kv_ram_gb",
    "peak_ram_gb",
    "peak_ram_pct",
    "min_available_ram_gb",
    "min_commit_available_gb",
    "weight_gpu_gb",
    "kv_gpu_gb",
    "expected_kv_gpu_gb",
    "kv_overhead_ratio",
    "peak_gpu_gb",
    "status",
    "error",
]


# ---------------------------------------------------------------------------
# Memory sampling (parent side)
#
# System RAM / GPU counters are process-wide, so sampling from the orchestrator
# captures the trial subprocess's footprint -- and, unlike sampling inside the
# child, these readings survive even when the child is killed on a timeout
# (exactly the case where memory matters most: the box was thrashing, not idle).
# ---------------------------------------------------------------------------
def _read_mem() -> dict:
    ram_used = ram_total = ram_pct = 0.0
    available_ram = None
    try:
        import psutil

        vm = psutil.virtual_memory()
        ram_used, ram_total, ram_pct = vm.used / (1024 ** 3), vm.total / (1024 ** 3), vm.percent
        available_ram = vm.available / (1024 ** 3)
    except Exception:
        pass
    commit_available = None
    if sys.platform == "win32":
        try:
            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                commit_available = status.ullAvailPageFile / (1024 ** 3)
        except (AttributeError, OSError):
            pass
    gpu_gb = 0.0
    try:
        from monitoring.scripts.windows.collect_gpu import get_gpu_memory_total

        used_mb, _dedicated, _shared = get_gpu_memory_total()
        if used_mb is not None:
            gpu_gb = used_mb / 1024
    except Exception:
        pass
    return {
        "ram_gb": ram_used,
        "ram_total_gb": ram_total,
        "ram_pct": ram_pct,
        "available_ram_gb": available_ram,
        "commit_available_gb": commit_available,
        "gpu_gb": gpu_gb,
    }


class _MemorySampler(threading.Thread):
    """Tracks peak (and latest) system RAM / GPU usage while a trial runs, plus the
    low-water mark of free physical RAM and Windows commit capacity.

    The low-water marks are pure instrumentation: nothing acts on them mid-trial.
    An earlier revision aborted a trial when they crossed a configured reserve,
    which turned "how much headroom was left" into an unanswerable question --
    the trial that would have told you never ran. Reporting the minimum instead
    answers it directly: a step that passes with 7 GB still free is a step with
    real headroom above it, and one that passes with 0.3 GB free is at the wall.
    """

    def __init__(self, interval: float = 0.5):
        super().__init__(daemon=True)
        self._stop_event = threading.Event()
        self.interval = interval
        self.peak_ram = 0.0
        self.peak_ram_pct = 0.0
        self.peak_gpu = 0.0
        self.min_available_ram = None
        self.min_commit_available = None
        self.latest = _read_mem()
        self._observe(self.latest)

    def _observe(self, m: dict) -> None:
        self.peak_ram = max(self.peak_ram, m["ram_gb"])
        self.peak_ram_pct = max(self.peak_ram_pct, m["ram_pct"])
        self.peak_gpu = max(self.peak_gpu, m["gpu_gb"])
        for key, attr in (
            ("available_ram_gb", "min_available_ram"),
            ("commit_available_gb", "min_commit_available"),
        ):
            value = m.get(key)
            if value is None:
                continue
            current = getattr(self, attr)
            setattr(self, attr, value if current is None else min(current, value))

    def run(self):
        while not self._stop_event.is_set():
            m = _read_mem()
            self.latest = m
            self._observe(m)
            self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()


def _delta(higher: float, lower: float) -> float:
    return round(max(0.0, higher - lower), 2)


def _crash_reason(exitcode) -> str:
    """Human-readable `crashed:...` error for a child that died without reporting.

    Keeps the `crashed:` prefix `_classify_failure()` matches on, and appends what
    the exit code means when it's a known native-abort status, so a reader doesn't
    have to look up 3221226505 to learn the trial was killed by a C++ abort rather
    than by an ordinary non-zero return.
    """
    detail = _NATIVE_ABORT_EXIT_CODES.get(exitcode)
    return f"crashed:exitcode={exitcode}" + (f":{detail}" if detail else "")


def _wait_for_memory_settle(
    baseline: dict,
    timeout_sec: float = _MEMORY_SETTLE_TIMEOUT_SEC,
    tolerance_gb: float = _MEMORY_SETTLE_TOLERANCE_GB,
    interval: float = 0.5,
) -> bool:
    """Block until the finished trial's memory is actually back to the OS.

    Returns True once RAM *and* GPU usage are within `tolerance_gb` of the
    pre-trial baseline, or False if `timeout_sec` elapses first -- the timeout is
    not an error, it just means something else on the box is holding memory, and
    the sweep continues either way rather than stalling on an unmet condition.
    """
    deadline = time.monotonic() + timeout_sec
    while True:
        current = _read_mem()
        if (
            current["ram_gb"] <= baseline["ram_gb"] + tolerance_gb
            and current["gpu_gb"] <= baseline["gpu_gb"] + tolerance_gb
        ):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval)


def _weight_disk_gb(model_dir: str) -> float:
    """Approximate weight footprint from the on-disk IR weights (.bin). For an
    int8/int4 export this closely tracks the resident weight memory, and it's a
    deterministic reference that's available even if a trial OOMs before we can
    measure the loaded footprint."""
    total = 0
    for root, _dirs, files in os.walk(model_dir):
        for f in files:
            if f.endswith(".bin"):
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    return round(total / (1024 ** 3), 2)


def _load_model_config(model_dir: str) -> dict | None:
    """Best-effort read of the exported IR's config.json (present for every
    optimum-cli export). Returns None if missing/unreadable so callers can
    degrade to "no theoretical estimate" instead of failing the trial."""
    path = os.path.join(model_dir, "config.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _theoretical_kv_bytes_per_token(config: dict, kv_cache_dtype_bytes: int = 2) -> float | None:
    """Expected KV-cache growth per token from the model's own declared
    architecture, treating hybrid linear-attention layers correctly.

    VLM exports nest the causal-LM config under `text_config` (confirmed on
    the Qwen3.5-9B / Qwen3.6-35B-A3B IR exports this tool targets); plain LLM
    configs have these keys top-level, so this reads from
    ``config.get("text_config", config)`` to handle both.

    Only layers whose `layer_types` entry is "full_attention" hold a KV cache
    that grows with sequence length; entries like "linear_attention" are
    Mamba/GatedDeltaNet-style recurrent-state layers with an O(1) state that
    does NOT grow with context length. This split is not just a config.json
    label -- it is confirmed directly against the exported IR
    (openvino_language_model.xml / openvino_model.xml): only the
    full_attention layers' `cache_params.past.{key,value}.N` state variables
    carry a dynamic sequence-length axis; the linear_attention layers'
    `cache_params.past.{conv,ssm}.N` variables have a fixed shape regardless
    of context length.

    This function returns a *persistent-cache-only* estimate, which is
    deliberately narrower than what trial_runner's kv_gpu_gb/kv_ram_gb
    measure (see the `kv_overhead_ratio` note in `_write_summary()`): those
    also include prefill/decode working memory (Q/K/V projections, MLP
    activations, ...) for *every* layer, including the linear-attention ones
    -- a hybrid model still runs all its layers on every prompt token during
    prefill even though only the full_attention layers keep a cache
    afterwards. A large kv_overhead_ratio for a hybrid model is therefore
    expected from this scope mismatch alone; it is not, by itself, evidence
    of a specific OpenVINO defect (an earlier version of this comment cited
    a known OpenVINO Model Server issue where continuous-batching prefix
    caching over-allocates memory for linear-attention models -- that issue
    is real, but its precondition isn't met here: trial_runner's
    `_load_pipeline()` never sets `scheduler_config`/`ATTENTION_BACKEND=PA`,
    so OpenVINO GenAI's own backend-selection logic keeps this tool on the
    plain stateful single-sequence backend, not continuous batching, so that
    specific issue cannot be what's being observed in this tool's trials).
    If `layer_types` is absent, every layer is assumed to be a standard
    growing-KV-cache attention layer (ordinary dense transformer).

    Returns None if the config doesn't expose enough info to compute this
    (num_hidden_layers / num_key_value_heads / head_dim), so callers can
    degrade gracefully for architectures/exports this doesn't understand yet.

    kv_cache_dtype_bytes defaults to 2 (fp16), OpenVINO GenAI's default KV
    cache precision when not otherwise configured -- this is a stated
    assumption, not something measured, since the tool has no way to read
    back the runtime's actual KV precision.
    """
    text_cfg = config.get("text_config", config)
    num_layers = text_cfg.get("num_hidden_layers")
    num_kv_heads = text_cfg.get("num_key_value_heads")
    head_dim = text_cfg.get("head_dim")
    if not num_layers or not num_kv_heads or not head_dim:
        return None
    layer_types = text_cfg.get("layer_types")
    growing_layers = (
        sum(1 for t in layer_types if t == "full_attention") if layer_types else num_layers
    )
    if not growing_layers:
        return None
    return 2 * growing_layers * num_kv_heads * head_dim * kv_cache_dtype_bytes


@contextlib.contextmanager
def _suppress_native_stderr():
    """Silence C-level stderr for the duration of the block.

    The WMI/COM hardware probe in utils/platform_info.py emits benign
    "Win32 exception occurred releasing IUnknown" teardown noise from the native
    COM layer (not via Python's logging/warnings), so only an fd-level redirect
    can hide it. Used solely around the best-effort platform-info call, which
    already falls back to defaults on any failure.
    """
    try:
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        yield
        return
    saved = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stderr.flush()
        os.dup2(devnull, stderr_fd)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, stderr_fd)
        os.close(saved)
        os.close(devnull)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Find the max context length each candidate summarizer model can "
            "prefill and decode on this hardware, against the configured target (default 160K)."
        ),
    )
    parser.add_argument(
        "--config",
        default=_DEFAULT_CONFIG_PATH,
        help="Path to this tool's own config.yaml (default: the file bundled next to this "
        "script -- independent of smart-classroom/config.yaml)",
    )
    parser.add_argument("--models", nargs="+", help="Override models.summarizer.long_context_validation.candidate_models")
    parser.add_argument("--target-tokens", type=int, help="Override target_context_tokens")
    parser.add_argument("--steps", type=int, nargs="+", help="Override context_steps_tokens")
    parser.add_argument("--device", help="Override models.summarizer.device (e.g. GPU, CPU)")
    parser.add_argument("--weight-format", help="Override models.summarizer.weight_format")
    parser.add_argument("--probe-tokens", type=int, help="Override probe_tokens (decode length per trial)")
    parser.add_argument(
        "--max-generate-time-sec",
        type=float,
        help="Override the maximum acceptable prefill + probe generation time",
    )
    parser.add_argument(
        "--gpu-memory-pressure-pct",
        type=float,
        help="Override the GPU-used/system-RAM percentage treated as the practical iGPU limit",
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Skip the bisection between the last pass and the first failure. Refinement is on by "
        "default because the point of the sweep is the actual ceiling, not the nearest step below it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Exercise the sweep/reporting pipeline with a synthetic fake trial -- no OpenVINO/GPU required",
    )
    parser.add_argument("--output-dir", help="Override output_dir")
    return parser.parse_args()


def _load_settings(args) -> dict:
    cfg = load_config(args.config)
    summarizer = getattr(cfg, "summarizer", None)
    lcv = getattr(summarizer, "long_context_validation", None) if summarizer else None
    if lcv is None:
        raise SystemExit(
            f"summarizer.long_context_validation is missing from {args.config}. "
            "See docs/dev-guide/validate_long_context.md for the expected schema."
        )
    return {
        "provider": summarizer.provider,
        "models_base_path": summarizer.models_base_path,
        "device": args.device or summarizer.device,
        "weight_format": args.weight_format or summarizer.weight_format,
        "candidate_models": args.models or lcv.candidate_models,
        "target_context_tokens": (
            args.target_tokens if args.target_tokens is not None else lcv.target_context_tokens
        ),
        "context_steps_tokens": sorted(args.steps or lcv.context_steps_tokens),
        "probe_tokens": (
            args.probe_tokens if args.probe_tokens is not None else getattr(lcv, "probe_tokens", 64)
        ),
        "max_generate_time_sec": (
            args.max_generate_time_sec
            if args.max_generate_time_sec is not None
            else getattr(lcv, "max_generate_time_sec", 600)
        ),
        "gpu_memory_pressure_pct": (
            args.gpu_memory_pressure_pct
            if args.gpu_memory_pressure_pct is not None
            else getattr(lcv, "gpu_memory_pressure_pct", 90)
        ),
        "trial_timeout_sec": lcv.trial_timeout_sec,
        "output_dir": args.output_dir or lcv.output_dir,
        "refine": not args.no_refine,
    }


def _model_ir_dir(models_base_path: str, provider: str, model_name: str, weight_format: str) -> str:
    # Mirrors utils/ensure_model.py::get_model_path's path convention, parameterized
    # per candidate model instead of hardcoded to config.models.summarizer.name.
    return os.path.join(models_base_path, provider, f"{model_name.replace('/', '_')}_{weight_format}")


def _ir_ready(model_dir: str) -> bool:
    if not os.path.isdir(model_dir):
        return False
    xml_names = []
    for _root, _dirs, files in os.walk(model_dir):
        xml_names.extend(f for f in files if f.endswith(".xml"))
    has_model = any(_MODEL_IR_RE.search(name) for name in xml_names)
    has_tokenizer = _TOKENIZER_IR_NAME in xml_names
    has_detokenizer = _DETOKENIZER_IR_NAME in xml_names
    return has_model and has_tokenizer and has_detokenizer


def _prep_command(model_name: str, model_dir: str, weight_format: str) -> str:
    return (
        f'optimum-cli export openvino --model "{model_name}" --trust-remote-code '
        f'--weight-format {weight_format} "{model_dir}"'
    )


def _safe_platform_info() -> dict:
    try:
        from utils.platform_info import get_platform_and_model_info

        with _suppress_native_stderr():
            info = get_platform_and_model_info()
            gc.collect()  # force COM release inside the suppressed window
        return info
    except Exception:
        return {}


def _passed(result: dict) -> bool:
    return (
        bool(result.get("load_ok"))
        and bool(result.get("generate_ok"))
        and result.get("generated_tokens", 0) > 0
        and not _resource_limit_reached(result)
    )


def _host_memory_exhausted(result: dict) -> bool:
    """Did this trial drive the *box* out of memory, per the sampler's low-water marks?

    `gpu_memory_at_limit` alone cannot answer this. It compares peak GPU usage against total
    system RAM, and on a shared-memory iGPU that ratio has no route to its 90% default -- the
    host needs the rest of the machine. Every failing trial observed on the 64 GB box sat between
    63% and 69%, so the flag never fired, and with it `too_slow` was unreachable and a device
    abort had no memory evidence attached. The headroom the sampler measures does answer it:
    1.9 GB of free RAM at 97.1% used is the wall, whatever the GPU-versus-RAM ratio says.
    """
    free_ram = result.get("min_available_ram_gb")
    peak_ram_pct = result.get("peak_ram_pct")
    return bool(
        (free_ram is not None and free_ram <= _MEMORY_EXHAUSTED_FREE_RAM_GB)
        or (peak_ram_pct is not None and peak_ram_pct >= _MEMORY_EXHAUSTED_RAM_PCT)
    )


def _memory_at_limit(result: dict) -> bool:
    """True when either memory ceiling -- shared-GPU or host -- was reached during the trial.

    Host exhaustion is recomputed from the trial's own numbers rather than read back from the
    `host_memory_at_limit` column, so classification never depends on that column having been
    filled in first.
    """
    return bool(result.get("gpu_memory_at_limit")) or _host_memory_exhausted(result)


def _resource_limit_reached(result: dict) -> bool:
    """A soft latency breach is a capacity failure only under memory pressure."""
    generate_time = result.get("generate_time_s")
    max_generate_time = result.get("max_generate_time_sec")
    return bool(
        max_generate_time is not None
        and generate_time is not None
        and generate_time > max_generate_time
        and _memory_at_limit(result)
    )


def _error_classification(error: str) -> str:
    """Middle field of a run_trial "stage:classification:detail" error string (e.g.
    "oom" or "exception"), or "" if the error isn't in that format -- an output-
    validation reason like "no_output" has no colons at all, and the raw exception
    text in the detail field can itself contain arbitrary colons, so this must read
    only the classification field rather than search the whole string."""
    parts = error.split(":", 2)
    return parts[1] if len(parts) >= 2 else ""


def _classify_failure(result: dict) -> str:
    """Name the failure, using the memory the parent measured to resolve what the child couldn't.

    A GPU device abort (`gpu_abort`, e.g. OpenCL -14) is the ambiguous case: the child only knows
    the device killed the command, so this promotes it to `oom` when -- and only when -- the
    trial's own measurements show the memory was gone. Same for a native abort that killed the
    child outright (`crashed`). Without that promotion, the single most important row the sweep
    produces -- the step that establishes the ceiling -- was labelled `generate_error`, which
    reads as "the tool hit an unexplained error" rather than "this box ran out of memory here".
    """
    error = str(result.get("error") or "")
    if error.startswith("trial_error"):
        return "trial_error"
    if error == "timeout":
        return "timeout"
    if error.startswith("crashed"):
        return "oom" if _memory_at_limit(result) else "crashed"
    classification = _error_classification(error)
    is_gpu_abort = classification == "gpu_abort"
    is_oom = classification == "oom" or (is_gpu_abort and _memory_at_limit(result))
    if not result.get("load_ok"):
        if is_oom:
            return "oom"
        return "gpu_abort" if is_gpu_abort else "load_error"
    if not result.get("generate_ok"):
        if is_oom:
            return "oom"
        if is_gpu_abort:
            return "gpu_abort"
        if error == "no_output":
            return "no_output"
        return "generate_error"
    if not result.get("generated_tokens", 0):
        return "no_output"
    if _resource_limit_reached(result):
        return "too_slow"
    return "unknown"


def _status_label(result: dict) -> str:
    return "PASS" if _passed(result) else _classify_failure(result)


def _append_trial_row(output_dir: str, row: dict) -> None:
    path = os.path.join(output_dir, "trials.csv")
    flat = {field: row.get(field) for field in TRIAL_CSV_FIELDS}
    flat["model"] = row["model"]
    flat["device"] = row["device"]
    flat["weight_format"] = row["weight_format"]
    flat["status"] = _status_label(row)
    StorageManager.save_csv(path, flat, headers=TRIAL_CSV_FIELDS, append=True)


def _fmt_gb(value) -> str:
    return f"{value:.1f} GB" if isinstance(value, (int, float)) else "--"


def _format_trial_line(model_name: str, tokens: int, result: dict) -> str:
    passed = _passed(result)
    status = "PASS" if passed else f"FAIL ({_classify_failure(result)})"
    parts = [f"[{model_name}] {tokens:>9,} tok -> {status}"]

    timing = []
    if result.get("load_time_s") is not None:
        timing.append(f"load {result['load_time_s']:.1f}s")
    if result.get("generate_time_s") is not None:
        timing.append(
            f"gen {result['generate_time_s']:.1f}s ({result.get('generated_tokens', 0)} tok, "
            f"{result.get('tokens_per_second', 0):.2f} tok/s)"
        )
    if timing:
        parts.append(", ".join(timing))

    if result.get("peak_ram_gb") is not None:
        seg = f"peak RAM {_fmt_gb(result.get('peak_ram_gb'))}"
        if result.get("weight_ram_gb") is not None and result.get("kv_ram_gb") is not None:
            seg += f" (weights +{result['weight_ram_gb']:.1f}, kv +{result['kv_ram_gb']:.1f})"
        parts.append(seg)
    if result.get("peak_gpu_gb") is not None:
        seg = f"peak GPU {_fmt_gb(result.get('peak_gpu_gb'))}"
        if result.get("peak_gpu_pct") is not None:
            seg += f" ({result['peak_gpu_pct']:.1f}% of system RAM)"
        if result.get("weight_gpu_gb") is not None and result.get("kv_gpu_gb") is not None:
            seg += f" (weights +{result['weight_gpu_gb']:.1f}, kv +{result['kv_gpu_gb']:.1f}"
            if result.get("kv_overhead_ratio") is not None:
                seg += f", expected {result['expected_kv_gpu_gb']:.2f}, {result['kv_overhead_ratio']:.1f}x"
            seg += ")"
        parts.append(seg)

    # How close the box actually came to the wall -- the number the removed
    # memory guard used to spend a trial guessing at instead of measuring.
    if result.get("min_available_ram_gb") is not None:
        seg = f"min free RAM {result['min_available_ram_gb']:.1f} GB"
        if result.get("min_commit_available_gb") is not None:
            seg += f" (commit {result['min_commit_available_gb']:.1f} GB)"
        parts.append(seg)

    if result.get("latency_limit_exceeded") and not _memory_at_limit(result):
        parts.append("latency budget exceeded without memory saturation")

    if not passed and result.get("error"):
        parts.append(f"error={result.get('error')}")
    return "  |  ".join(parts)


def _print_trial_start(model_name: str, tokens: int, settings: dict, indent: str = "") -> None:
    """Announce a trial before it runs, not only after.

    A single step at 128K+ takes minutes, and the step that finds the ceiling is the slowest of
    all -- it thrashes first and then dies. Printing only on completion makes that indis-
    tinguishable from a hung sweep for as long as `trial_timeout_sec` (20 minutes by default),
    which is how the run this fixes was read as "it crashed". `flush` because stdout is
    block-buffered whenever the sweep is piped or redirected to a log.
    """
    print(
        f"{indent}[{model_name}] {tokens:>9,} tok -> running "
        f"(load + prefill + {settings['probe_tokens']} tok decode, timeout "
        f"{settings['trial_timeout_sec']:g}s) ...",
        flush=True,
    )


def _fake_ceiling_tokens(model_name: str, steps: list) -> int:
    """Deterministic per-model fake ceiling so --dry-run exercises a mix of
    pass/fail across the configured steps without touching any hardware."""
    digest = sum(ord(c) for c in model_name)
    return steps[digest % len(steps)]


def _run_trial_dry_run(model_name: str, tokens: int, probe_tokens: int, fake_ceiling: int) -> dict:
    ok = tokens <= fake_ceiling
    weight_ram = 4.0  # pretend a small fixed weight footprint
    weight_gpu = 3.5
    kv_ram = round(tokens / 8000.0, 2)  # KV grows with context
    kv_gpu = round(tokens / 10000.0, 2)
    peak_gpu = round(1.0 + weight_gpu + kv_gpu, 2)
    return {
        "tokens_requested": tokens,
        "load_ok": True,
        "load_time_s": 0.01,
        "generate_ok": ok,
        "prompt_tokens": tokens,
        "generated_tokens": probe_tokens if ok else 0,
        "generate_time_s": 0.01,
        "weight_ram_gb": weight_ram,
        "weight_gpu_gb": weight_gpu,
        "kv_ram_gb": kv_ram,
        "kv_gpu_gb": kv_gpu,
        "peak_ram_gb": round(2.0 + weight_ram + kv_ram, 2),
        "peak_ram_pct": 50.0,
        "min_available_ram_gb": round(max(0.0, 64.0 - (2.0 + weight_ram + kv_ram)), 2),
        "min_commit_available_gb": round(max(0.0, 72.0 - (2.0 + weight_ram + kv_ram)), 2),
        "peak_gpu_gb": peak_gpu,
        "ram_total_gb": 64.0,
        "error": None if ok else "generate:oom:allocation failed (dry-run)",
    }


def _run_trial_subprocess(
    model_name: str,
    model_dir: str,
    device: str,
    tokens: int,
    probe_tokens: int,
    timeout_sec: int,
    sample_interval: float = 0.5,
    poll_interval: float = 0.25,
    drain_timeout: float = 5.0,
) -> dict:
    """Run one trial to completion, OOM, native abort, or the hard timeout.

    Nothing here stops the child early on a memory threshold: the whole point of
    the sweep is to find where this box actually breaks, so the trial is allowed
    to run until the hardware (or `timeout_sec`) decides the outcome. Subprocess
    isolation is what makes that safe -- a native GPU allocation abort near
    shared-memory exhaustion kills only this child, and the parent's sampler has
    already recorded the memory high-water mark that explains why.
    """
    baseline = _read_mem()
    sampler = _MemorySampler(interval=sample_interval)
    sampler.start()

    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=trial_runner.run_trial,
        args=(model_dir, model_name, device, tokens, probe_tokens, result_queue),
    )
    process.start()

    loaded_mem = None
    load_ok = False
    result = None

    def _consume(msg: dict, child_alive: bool) -> bool:
        """Apply one child message; True once the terminal "done" has arrived."""
        nonlocal loaded_mem, load_ok, result
        event = msg.get("event")
        if event == "loaded":
            load_ok = True
            # Snapshot the weight footprint before prefill grows it. Only meaningful
            # while the child still holds the weights, so a "loaded" recovered from
            # the post-mortem drain below records load_ok without a memory reading
            # rather than attributing a dead child's footprint to its weights.
            if child_alive and loaded_mem is None:
                loaded_mem = _read_mem()
        elif event == "done":
            result = msg
            load_ok = load_ok or bool(msg.get("load_ok"))
            return True
        return False

    deadline = time.monotonic() + timeout_sec
    while result is None and time.monotonic() < deadline:
        try:
            msg = result_queue.get(timeout=poll_interval)
        except Empty:
            if not process.is_alive():
                break  # child exited; anything it queued is recovered by the drain below
        else:
            if _consume(msg, child_alive=True):
                break

    timed_out = result is None and time.monotonic() >= deadline
    if process.is_alive():
        process.terminate()
    process.join(10)
    sampler.stop()
    sampler.join(2 * sample_interval + 1)

    if result is None:
        # The child posts its result and then exits immediately, without running
        # OpenVINO's teardown (trial_runner._post_result_and_exit), so a finished
        # trial's message can land in the pipe in the window between the poll above
        # timing out and is_alive() going False. Drain what's left instead of
        # reporting a completed measurement as a crash.
        drain_deadline = time.monotonic() + drain_timeout
        while time.monotonic() < drain_deadline:
            try:
                msg = result_queue.get(timeout=poll_interval)
            except Empty:
                break
            if _consume(msg, child_alive=False):
                timed_out = False
                break

    # Reclamation now happens on process exit rather than in OpenVINO's destructors,
    # so let it land before the next trial reads its baseline (see the constants). Giving up
    # is not fatal, but it does mean the next trial's baseline is polluted and its weight/KV
    # split will be wrong, so say so rather than letting a bad number look like a measurement.
    if not _wait_for_memory_settle(baseline):
        current = _read_mem()
        print(
            f"  [warn] memory has not returned to the pre-trial baseline after "
            f"{_MEMORY_SETTLE_TIMEOUT_SEC:g}s (RAM {current['ram_gb']:.1f} GB vs "
            f"{baseline['ram_gb']:.1f} GB, GPU {current['gpu_gb']:.1f} GB vs "
            f"{baseline['gpu_gb']:.1f} GB); the next trial's weights/kv split may be charged "
            "with what is left over",
            flush=True,
        )

    mem = {
        "peak_ram_gb": round(sampler.peak_ram, 2),
        "peak_ram_pct": round(sampler.peak_ram_pct, 1),
        "min_available_ram_gb": (
            round(sampler.min_available_ram, 2) if sampler.min_available_ram is not None else None
        ),
        "min_commit_available_gb": (
            round(sampler.min_commit_available, 2)
            if sampler.min_commit_available is not None
            else None
        ),
        "peak_gpu_gb": round(sampler.peak_gpu, 2),
        "weight_ram_gb": _delta(loaded_mem["ram_gb"], baseline["ram_gb"]) if loaded_mem else None,
        "weight_gpu_gb": _delta(loaded_mem["gpu_gb"], baseline["gpu_gb"]) if loaded_mem else None,
        "kv_ram_gb": _delta(sampler.peak_ram, loaded_mem["ram_gb"]) if loaded_mem else None,
        "kv_gpu_gb": _delta(sampler.peak_gpu, loaded_mem["gpu_gb"]) if loaded_mem else None,
        "ram_total_gb": round(baseline["ram_total_gb"], 2),
    }

    if result is not None:
        result.pop("event", None)
        result.update(mem)
        return result

    # A native GPU allocation abort near shared-memory exhaustion kills the child
    # without a Python exception, so it surfaces here as `crashed` rather than
    # `oom`. Reaching this point now means the abort happened *before* the child
    # could report -- i.e. during load, prefill or decode, which is a real capacity
    # ceiling -- because a post-result teardown abort can no longer occur (the child
    # skips the teardown) and a result racing the child's exit is recovered by the
    # drain above. The memory columns still explain it either way: the sampler
    # recorded the peak and the low-water headroom from the parent, which survives
    # the child.
    reason = "timeout" if timed_out else _crash_reason(process.exitcode)
    return {
        "tokens_requested": tokens,
        "load_ok": load_ok,  # crashed/hung during generate, not load
        "load_time_s": None,
        "generate_ok": False,
        "prompt_tokens": 0,
        "generated_tokens": 0,
        "generate_time_s": None,
        "error": reason,
        **mem,
    }


def _failed_to_run_result(tokens: int, exc: Exception) -> dict:
    """A step the orchestrator itself could not carry out, recorded as that step's failure.

    Running the sweep is not free: each trial spawns a fresh interpreter that re-imports the
    OpenVINO stack, and the sweep deliberately keeps going until the box breaks, so the spawn
    can be what breaks. Letting that exception unwind would discard every measurement already
    taken along with the report -- which is exactly what happened on the 64 GB box, where the
    sweep died during refinement and left `summary.md` still describing an earlier `--dry-run`.
    """
    return {
        "tokens_requested": tokens,
        "load_ok": False,
        "load_time_s": None,
        "generate_ok": False,
        "prompt_tokens": 0,
        "generated_tokens": 0,
        "generate_time_s": None,
        "error": f"trial_error:{type(exc).__name__}:{exc}",
    }


def _run_one(model_name, model_dir, device, tokens, settings, dry_run, fake_ceiling, kv_bytes_per_token=None):
    if dry_run:
        result = _run_trial_dry_run(model_name, tokens, settings["probe_tokens"], fake_ceiling)
    else:
        try:
            result = _run_trial_subprocess(
                model_name,
                model_dir,
                device,
                tokens,
                settings["probe_tokens"],
                settings["trial_timeout_sec"],
            )
        except Exception as exc:  # noqa: BLE001 - recorded as this step's failure, see helper
            traceback.print_exc()
            result = _failed_to_run_result(tokens, exc)
    generate_time = result.get("generate_time_s")
    generated_tokens = result.get("generated_tokens", 0)
    result["tokens_per_second"] = (
        round(generated_tokens / generate_time, 3) if generate_time and generated_tokens else 0.0
    )
    result["max_generate_time_sec"] = settings["max_generate_time_sec"]
    result["latency_limit_exceeded"] = bool(
        generate_time is not None and generate_time > settings["max_generate_time_sec"]
    )
    ram_total_gb = result.pop("ram_total_gb", 0.0)
    peak_gpu_gb = result.get("peak_gpu_gb", 0.0)
    result["gpu_memory_pressure_pct"] = settings["gpu_memory_pressure_pct"]
    result["peak_gpu_pct"] = (
        round(peak_gpu_gb / ram_total_gb * 100, 1) if peak_gpu_gb and ram_total_gb else None
    )
    result["gpu_memory_at_limit"] = bool(
        result["peak_gpu_pct"] is not None
        and result["peak_gpu_pct"] >= settings["gpu_memory_pressure_pct"]
    )
    result["host_memory_at_limit"] = _host_memory_exhausted(result)
    if kv_bytes_per_token and result.get("prompt_tokens"):
        expected_kv_gb = kv_bytes_per_token * result["prompt_tokens"] / (1024 ** 3)
        result["expected_kv_gpu_gb"] = round(expected_kv_gb, 2)
        kv_gpu = result.get("kv_gpu_gb")
        result["kv_overhead_ratio"] = round(kv_gpu / expected_kv_gb, 2) if kv_gpu else None
    else:
        result["expected_kv_gpu_gb"] = None
        result["kv_overhead_ratio"] = None
    return result


def _refine_boundary(
    model_name, model_dir, device, settings, low, high, dry_run, fake_ceiling, weight_disk,
    kv_bytes_per_token=None, max_extra=3,
):
    """Bisect between the last pass (`low`) and the first failure (`high`).

    Returns ``(highest_pass, best_result, lowest_failure)``, where `lowest_failure` is
    ``{"tokens", "reason"}`` for the smallest context refinement saw fail, or None if every
    refinement trial passed. The summary quotes it, because after refinement the context this
    box actually broke at is no longer the configured step that first failed -- and the reason
    can differ too (`oom` at 160K, but the run this fixes could equally have produced a
    `trial_error` at 144K once the box was that close to the wall).
    """
    lo, hi = low, high
    smallest_step = settings["context_steps_tokens"][0]
    best_result = None
    lowest_failure = None
    for _ in range(max_extra):
        if hi - lo <= max(1, smallest_step // 8):
            break
        mid = (lo + hi) // 2
        _print_trial_start(model_name, mid, settings, indent="  ")
        result = _run_one(
            model_name, model_dir, device, mid, settings, dry_run, fake_ceiling,
            kv_bytes_per_token,
        )
        _append_trial_row(
            settings["output_dir"],
            {
                "model": model_name,
                "device": device,
                "weight_format": settings["weight_format"],
                "weight_disk_gb": weight_disk,
                **result,
            },
        )
        passed = _passed(result)
        print("  " + _format_trial_line(model_name, mid, result), flush=True)
        if passed:
            best_result = result
        else:
            lowest_failure = {"tokens": mid, "reason": _classify_failure(result)}
        lo, hi = (mid, hi) if passed else (lo, mid)
    return lo, best_result, lowest_failure


def _sweep_model(model_name: str, settings: dict, dry_run: bool) -> dict:
    device = settings["device"]
    weight_format = settings["weight_format"]
    model_dir = _model_ir_dir(settings["models_base_path"], settings["provider"], model_name, weight_format)

    if not dry_run and not _ir_ready(model_dir):
        prep_command = _prep_command(model_name, model_dir, weight_format)
        print(f"[{model_name}] IR not found at {model_dir}\n  Run first: {prep_command}", flush=True)
        return {
            "model": model_name,
            "status": "missing_ir",
            "max_stable_context": None,
            "meets_target": False,
            "device": device,
            "weight_format": weight_format,
            "prep_command": prep_command,
        }

    weight_disk = 0.0 if dry_run else _weight_disk_gb(model_dir)
    print(
        f"\n=== {model_name} ===  weights on disk: {_fmt_gb(weight_disk)} ({weight_format})",
        flush=True,
    )

    model_config = None if dry_run else _load_model_config(model_dir)
    kv_bytes_per_token = _theoretical_kv_bytes_per_token(model_config) if model_config else None

    fake_ceiling = _fake_ceiling_tokens(model_name, settings["context_steps_tokens"]) if dry_run else None

    max_stable = 0
    max_stable_result = None
    fail_reason = None
    completed_all_steps = True
    last_tokens = settings["context_steps_tokens"][0]

    for tokens in settings["context_steps_tokens"]:
        last_tokens = tokens
        _print_trial_start(model_name, tokens, settings)
        result = _run_one(
            model_name, model_dir, device, tokens, settings, dry_run, fake_ceiling,
            kv_bytes_per_token,
        )
        _append_trial_row(
            settings["output_dir"],
            {
                "model": model_name,
                "device": device,
                "weight_format": weight_format,
                "weight_disk_gb": weight_disk,
                **result,
            },
        )

        passed = _passed(result)
        print(_format_trial_line(model_name, tokens, result), flush=True)
        if not passed:
            fail_reason = _classify_failure(result)
            completed_all_steps = False
            break
        max_stable = tokens
        max_stable_result = result

    fail_tokens = None
    if completed_all_steps:
        fail_reason = None
    else:
        fail_tokens = last_tokens
        if settings["refine"] and max_stable:
            max_stable, refined_result, refined_failure = _refine_boundary(
                model_name, model_dir, device, settings, max_stable, last_tokens, dry_run,
                fake_ceiling, weight_disk, kv_bytes_per_token=kv_bytes_per_token,
            )
            if refined_result is not None:
                max_stable_result = refined_result
            # A refinement step the orchestrator could not run is not a capacity measurement, so
            # it must not displace one: "capped by oom at 144,000" is the sweep's answer, and
            # "capped by trial_error at 136,000" would bury it. The un-runnable step is still on
            # the console and in trials.csv, and `max_stable_context` is the same either way.
            if refined_failure is not None and refined_failure["reason"] != "trial_error":
                fail_tokens = refined_failure["tokens"]
                fail_reason = refined_failure["reason"]

    peak = max_stable_result or {}
    return {
        "model": model_name,
        "status": "ok",
        "max_stable_context": max_stable,
        "meets_target": max_stable >= settings["target_context_tokens"],
        "failure_tokens": fail_tokens,
        "device": device,
        "weight_format": weight_format,
        "weight_disk_gb": weight_disk,
        "weight_ram_gb": peak.get("weight_ram_gb"),
        "weight_gpu_gb": peak.get("weight_gpu_gb"),
        "kv_ram_gb": peak.get("kv_ram_gb"),
        "kv_gpu_gb": peak.get("kv_gpu_gb"),
        "expected_kv_gpu_gb": peak.get("expected_kv_gpu_gb"),
        "kv_overhead_ratio": peak.get("kv_overhead_ratio"),
        "peak_ram_gb": peak.get("peak_ram_gb"),
        "peak_gpu_gb": peak.get("peak_gpu_gb"),
        "min_available_ram_gb": peak.get("min_available_ram_gb"),
        "min_commit_available_gb": peak.get("min_commit_available_gb"),
        "failure_reason": fail_reason,
    }


def _write_summary(
    output_dir: str,
    settings: dict,
    model_reports: list,
    platform_info: dict,
    completed: bool = True,
) -> None:
    """(Re)write summary.json and summary.md for the run so far.

    Called after every model rather than once at the end, and with `completed=False` until the
    last one lands, because the sweep's whole job is to push the box until it breaks and it can
    break hard enough to take the orchestrator with it. When that happened on the 64 GB box, no
    summary was written at all and `summary.md` still held the previous `--dry-run` -- which
    reported the candidate as a **PASS at 160,000 tokens**, the exact opposite of what twelve
    minutes of real trials had just measured. A stale report that looks current is worse than
    no report, so the file on disk always describes the run that is actually happening.
    """
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "completed": completed,
        "target_context_tokens": settings["target_context_tokens"],
        "probe_tokens": settings["probe_tokens"],
        "max_generate_time_sec": settings["max_generate_time_sec"],
        "hardware": platform_info,
        "models": model_reports,
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    incomplete_banner = (
        []
        if completed
        else [
            "> **Run in progress or ended early.** Rows below cover only the models the sweep "
            "reached; anything marked `not run` has no measurement yet. `trials.csv` has every "
            "step that did run.",
            "",
        ]
    )
    lines = [
        "# Long-Context Capacity Validation Summary",
        "",
        *incomplete_banner,
        f"Generated: {summary['generated_at']}",
        f"Target context: {settings['target_context_tokens']:,} tokens | "
        f"Probe decode: {settings['probe_tokens']} tokens/trial | "
        f"Max prefill + generation: {settings['max_generate_time_sec']:g}s | "
        f"GPU memory pressure: {settings['gpu_memory_pressure_pct']:g}% of system RAM",
        "",
        f"Hardware: {platform_info.get('Processor', '--')}, {platform_info.get('Memory', '--')} RAM, "
        f"{platform_info.get('iGPU', '--')}",
        "",
        "Memory columns are measured at the max stable context: weights = footprint just after "
        "load; KV = additional memory prefill+decode added on top; peak = total high-water mark; "
        "Min free RAM = low-water mark of available physical RAM, i.e. the real headroom left at "
        "that context.",
        "",
        "| Model | Device | Weight | Max stable context | Meets target | Weights (disk) | "
        "Peak RAM | KV RAM | Min free RAM | Peak GPU | KV GPU | Expected KV | KV Ratio | Notes |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in model_reports:
        if r["status"] == "not_run":
            lines.append(
                f"| {r['model']} | {r['device']} | {r['weight_format']} | - | - | - | - | - | - | "
                "- | - | - | - | not run |"
            )
            continue
        if r["status"] == "missing_ir":
            lines.append(
                f"| {r['model']} | {r['device']} | {r['weight_format']} | - | - | - | - | - | - | - | - | - | - | "
                f"IR not found; run: `{r['prep_command']}` |"
            )
            continue
        max_ctx = f"{r['max_stable_context']:,}" if r["max_stable_context"] else "0"
        meets = "PASS" if r["meets_target"] else "FAIL"
        ratio = r.get("kv_overhead_ratio")
        notes = []
        if r.get("failure_reason"):
            capped = f"capped by {r['failure_reason']}"
            if r.get("failure_tokens"):
                capped += f" at {r['failure_tokens']:,} tokens"
            notes.append(capped)
        if ratio is not None and ratio >= _KV_OVERHEAD_RATIO_NOTE_THRESHOLD:
            notes.append(
                f"kv {ratio:g}x the persistent-cache-only estimate -- expected_kv_gpu_gb counts "
                "only growing-KV-cache layers, kv_gpu_gb also includes prefill/decode working "
                "memory across all layers, so a large ratio here is not a capacity problem with "
                "this box by itself, and not proof of a specific OpenVINO defect either"
            )
        if not notes:
            notes.append("reached top configured step without failing")
        note = "; ".join(notes)
        lines.append(
            f"| {r['model']} | {r['device']} | {r['weight_format']} | {max_ctx} | {meets} | "
            f"{_fmt_gb(r.get('weight_disk_gb'))} | {_fmt_gb(r.get('peak_ram_gb'))} | "
            f"{_fmt_gb(r.get('kv_ram_gb'))} | {_fmt_gb(r.get('min_available_ram_gb'))} | "
            f"{_fmt_gb(r.get('peak_gpu_gb'))} | "
            f"{_fmt_gb(r.get('kv_gpu_gb'))} | {_fmt_gb(r.get('expected_kv_gpu_gb'))} | "
            f"{f'{ratio:g}x' if ratio is not None else '--'} | {note} |"
        )

    with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _preflight_environment_check() -> None:
    """Fail fast with an actionable message if this interpreter can't import what
    trial_runner.py needs, instead of letting every trial in the sweep repeat the
    same import failure. Missing openvino_genai/transformers here almost always
    means "wrong Python interpreter", not a bug in this tool: those packages live
    in the project's backend venv, not the system/base Python.
    """
    missing = [name for name in _REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if not missing:
        return

    lines = [
        f"Missing required package(s) in this interpreter ({sys.executable}): {', '.join(missing)}.",
        "This is almost always the wrong Python environment, not a code bug -- the OpenVINO "
        "stack (openvino-genai, transformers, optimum-intel, torch) lives in the project's "
        "backend venv, not the interpreter picked up from PATH.",
        "Simplest fix -- use the launcher script, which creates the venv if needed (via "
        "setup_env.ps1), activates it, and re-runs this tool with the same arguments:",
        "  " + " ".join([_LAUNCHER_SCRIPT, *sys.argv[1:]]),
    ]
    if os.path.exists(_BACKEND_VENV_PYTHON):
        lines.append(f"Or run directly with the venv's interpreter, which already exists at {_BACKEND_VENV_PYTHON}:")
        lines.append(f'  "{_BACKEND_VENV_PYTHON}" -m components.llm.context_validation.validate_long_context')
    else:
        lines.append(
            f"Or prepare the venv yourself first (no venv found yet at {_BACKEND_VENV_PYTHON}):"
        )
        lines.append(f"  {_SETUP_SCRIPT}")
        lines.append(
            "(If PowerShell blocks either script: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)"
        )
    lines.append("(Use --dry-run to exercise the sweep/report pipeline without any of these packages.)")
    raise SystemExit("\n".join(lines))


def _pending_model_report(model_name: str, settings: dict) -> dict:
    return {
        "model": model_name,
        "status": "not_run",
        "device": settings["device"],
        "weight_format": settings["weight_format"],
    }


def main() -> None:
    args = _parse_args()
    if not args.dry_run:
        _preflight_environment_check()
    settings = _load_settings(args)
    os.makedirs(settings["output_dir"], exist_ok=True)
    platform_info = _safe_platform_info()

    print(f"Long-context capacity validation sweep starting (dry_run={args.dry_run})", flush=True)
    print(f"Candidates: {settings['candidate_models']}", flush=True)
    print(f"Steps: {settings['context_steps_tokens']}", flush=True)
    print(
        f"Target: {settings['target_context_tokens']:,} tokens | "
        f"Probe decode: {settings['probe_tokens']} tok | "
        f"Max generation: {settings['max_generate_time_sec']:g}s | Output: {settings['output_dir']}",
        flush=True,
    )

    # Every candidate starts as an explicit "not run" placeholder and is replaced as the sweep
    # reaches it, so the report on disk describes *this* run from the first moment -- never a
    # previous one. See _write_summary() for what the stale-report failure looked like.
    model_reports = [_pending_model_report(name, settings) for name in settings["candidate_models"]]
    completed = False
    try:
        _write_summary(settings["output_dir"], settings, model_reports, platform_info, completed=False)
        for index, model_name in enumerate(settings["candidate_models"]):
            model_reports[index] = _sweep_model(model_name, settings, args.dry_run)
            _write_summary(
                settings["output_dir"], settings, model_reports, platform_info, completed=False
            )
        completed = True
    finally:
        # Reached on Ctrl-C and on an unhandled failure too: the measurements already in
        # trials.csv are worth a report either way, and the banner says the run ended early.
        try:
            _write_summary(
                settings["output_dir"], settings, model_reports, platform_info, completed=completed
            )
        except Exception:  # noqa: BLE001 - must not mask whatever is already unwinding
            traceback.print_exc()
        else:
            print(
                f"\nReports written to {settings['output_dir']} "
                f"(trials.csv, summary.json, summary.md)"
                + ("" if completed else " -- sweep ended early, report marked incomplete"),
                flush=True,
            )


if __name__ == "__main__":
    main()
