# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Long-context benchmark for candidate summarizer models on this hardware.

Answers two questions per (model, context length):

  * Can this box prefill and decode it at all, within its memory budget?
  * Which OpenVINO configuration does it fastest?

The second question is why this exists in its current form. It benchmarks a
matrix of named `profiles` -- KV precision, prefill chunk size, continuous
batching vs the stateful pipeline -- and ranks them by measured end-to-end
throughput, following llm_bench's methodology: one warm-up iteration that is
excluded from every statistic, N measured iterations, and the median reported
with min/max so run-to-run spread is visible. The same 160K configuration was
previously measured at 247.0s and 349.5s on two single-shot runs; a single
sample cannot tell a configuration difference from noise, which is the whole
reason iterations are not optional here.

It does NOT judge answer quality -- content is irrelevant to a capacity and
throughput measurement, only the token volume and the clock matter.

Standalone diagnostic: reads its own bundled model config, never
smart-classroom/config.yaml, and running it never affects the application.

    .\\components\\llm\\context_bench\\run_benchmark.ps1
    .\\components\\llm\\context_bench\\run_benchmark.ps1 --profiles optimized-f16-32k --iterations 1
    .\\components\\llm\\context_bench\\run_benchmark.ps1 --list-profiles

Equivalent with the right interpreter already active, run from smart-classroom/
so relative model paths resolve:

    python -m components.llm.context_bench.benchmark

See docs/dev-guide/context-bench/context_bench_guide.md.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import math
import multiprocessing
import os
import platform
import re
import shutil
import sys
import threading
import time
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime
from queue import Empty
from types import SimpleNamespace

from components.llm.context_bench import metrics, trial_runner
from utils.config_loader import load_config
from utils.storage_manager import StorageManager

_TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG_PATH = os.path.join(_TOOL_DIR, "config_qwen3.5_9b.yaml")

# smart-classroom/ is 3 levels up (context_bench -> llm -> components -> smart-classroom).
_SC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_TOOL_DIR)))
# setup-smart-classroom.ps1 creates the backend venv as a sibling of smart-classroom/.
_BACKEND_VENV_PYTHON = os.path.join(
    os.path.dirname(_SC_ROOT), "smartclassroom", "Scripts", "python.exe"
)
_LAUNCHER_SCRIPT = os.path.join(_TOOL_DIR, "run_benchmark.ps1")
_SETUP_SCRIPT = os.path.join(_TOOL_DIR, "setup_env.ps1")

_REQUIRED_MODULES = ("openvino_genai", "transformers", "psutil")

# A converted candidate can be a plain causal LM (openvino_model.xml) or a multimodal
# export (openvino_language_model.xml), so match either layout.
_MODEL_IR_RE = re.compile(r"(.*)?openvino(.*)?_model(.*)?\.xml$")

# Windows exit codes meaning "a native runtime aborted", decoded so that
# `crashed:exitcode=3221226505` is not an opaque number.
_NATIVE_ABORT_EXIT_CODES = {
    3221226505: "0xC0000409 STATUS_STACK_BUFFER_OVERRUN - how the CRT reports abort()/std::terminate",
    3221225477: "0xC0000005 STATUS_ACCESS_VIOLATION",
    3221225725: "0xC00000FD STATUS_STACK_OVERFLOW",
}

# A finished child's allocations are reclaimed by the OS, and the Windows PDH GPU counters
# are themselves sampled, so both lag the process. Wait -- briefly, with a cap -- before the
# next case takes its baseline, or the previous case's memory is charged to the next one.
_MEMORY_SETTLE_TIMEOUT_SEC = 30.0
_MEMORY_SETTLE_TOLERANCE_GB = 1.0

# Bytes per KV element per precision, so `expected_kv_gb` stays honest when the cache is
# compressed: comparing a u8 run against an fp16 estimate would report a phantom saving.
# Compressed caches also store a scale and zero-point per row, added separately. "dynamic"
# (the GPU default) means "the plugin decides", in practice fp16, so unknowns fall back to 2.
_KV_PRECISION_BYTES = {
    "f32": 4, "f16": 2, "bf16": 2, "u8": 1, "i8": 1, "u4": 0.5, "i4": 0.5,
    "f8e4m3": 1, "f8e5m2": 1,
}
_DEFAULT_KV_BYTES = 2
_COMPRESSED_KV_PRECISIONS = {"u8", "i8", "u4", "i4"}
_KV_QUANT_PARAMS_BYTES = 4
_IR_TYPE_BYTES = {"f32": 4, "fp32": 4, "f16": 2, "fp16": 2, "bf16": 2, "i8": 1, "u8": 1}

# Headroom left outside the KV pool when sizing `cache_size: auto`: prefill workspace,
# activations and the runtime's own allocations all come out of the same shared budget.
_AUTO_CACHE_SAFETY = 1.3
_AUTO_CACHE_RESERVE_GB = 8.0
_AUTO_CACHE_MIN_GB = 2.0

CASE_FIELDS = [
    "model", "profile", "context_tokens", "device", "weight_format",
    "status", "error", "stage_reached",
    "kv_cache_precision", "cache_size_gb", "ov_config", "scheduler_config",
    "load_ok", "load_time_s", "prompt_tokens", "iterations_measured",
    "generation_time", "generation_time_min", "generation_time_max",
    "latency", "first_token_latency", "first_token_latency_min", "first_token_latency_max",
    "other_tokens_avg_latency",
    "prefill_throughput", "prefill_throughput_min", "prefill_throughput_max",
    "decode_throughput", "e2e_throughput", "e2e_throughput_min", "e2e_throughput_max",
    "weight_disk_gb", "expected_kv_gb",
    "peak_ram_gb", "peak_ram_pct", "min_available_ram_gb", "post_load_peak_ram_gb",
    "peak_gpu_gb", "post_load_peak_gpu_gb",
    "gpu_budget_gb", "gpu_budget_driver_gb", "peak_gpu_pct_of_budget",
    "system_memory_limit_exceeded", "gpu_budget_exceeded",
]

ITERATION_CSV_FIELDS = [
    "model", "profile", "context_tokens", *metrics.ITERATION_FIELDS,
    "peak_ram_gb", "peak_gpu_gb",
]


# ---------------------------------------------------------------------------
# Memory sampling (parent side)
#
# RAM / GPU counters are process-wide, so sampling from the orchestrator captures the
# child's footprint -- and unlike sampling inside the child, these readings survive a
# child that is killed on a timeout, exactly the case where memory matters most.
# ---------------------------------------------------------------------------
def _read_mem() -> dict:
    ram_used = ram_total = 0.0
    ram_pct = available_ram = None
    try:
        import psutil

        vm = psutil.virtual_memory()
        ram_used, ram_total, ram_pct = vm.used / (1024 ** 3), vm.total / (1024 ** 3), vm.percent
        available_ram = vm.available / (1024 ** 3)
    except Exception:  # noqa: BLE001
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
    except Exception:  # noqa: BLE001
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
    """Peak RAM/GPU and the low-water mark of free RAM, for the case and per iteration.

    The low-water mark is instrumentation, not a trigger: nothing cancels a case on it.
    A case that passes with 7 GB still free has real headroom above it; one that passes
    with 0.3 GB free is at the wall. Reporting the minimum answers that; aborting on a
    threshold would make it unanswerable.

    `reset_window()` / `window()` carve the same stream into per-iteration slices so a
    warm-up's allocation spike is not charged to the measured iterations.
    """

    def __init__(self, interval: float = 0.5):
        super().__init__(daemon=True)
        self._stop_event = threading.Event()
        self.interval = interval
        self.peak_ram = 0.0
        self.peak_ram_pct = None
        self.peak_gpu = 0.0
        self.min_available_ram = None
        self.min_commit_available = None
        self._window_ram = 0.0
        self._window_gpu = 0.0
        self.latest = _read_mem()
        self._observe(self.latest)

    def _observe(self, m: dict) -> None:
        self.peak_ram = max(self.peak_ram, m["ram_gb"])
        self.peak_gpu = max(self.peak_gpu, m["gpu_gb"])
        self._window_ram = max(self._window_ram, m["ram_gb"])
        self._window_gpu = max(self._window_gpu, m["gpu_gb"])
        if m.get("ram_pct") is not None:
            self.peak_ram_pct = (
                m["ram_pct"] if self.peak_ram_pct is None
                else max(self.peak_ram_pct, m["ram_pct"])
            )
        for key, attr in (
            ("available_ram_gb", "min_available_ram"),
            ("commit_available_gb", "min_commit_available"),
        ):
            value = m.get(key)
            if value is None:
                continue
            current = getattr(self, attr)
            setattr(self, attr, value if current is None else min(current, value))

    def reset_window(self) -> None:
        current = _read_mem()
        self._window_ram = current["ram_gb"]
        self._window_gpu = current["gpu_gb"]

    def window(self) -> tuple:
        return round(self._window_ram, 2), round(self._window_gpu, 2)

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


def _wait_for_memory_settle(baseline: dict, timeout_sec: float = _MEMORY_SETTLE_TIMEOUT_SEC) -> bool:
    """Block until the finished case's memory is actually back with the OS.

    True once RAM and GPU are within tolerance of the pre-case baseline, False on timeout.
    A timeout is not an error -- something else on the box may hold memory -- so the run
    continues either way rather than stalling on an unmet condition.
    """
    deadline = time.monotonic() + timeout_sec
    while True:
        current = _read_mem()
        if (
            current["ram_gb"] <= baseline["ram_gb"] + _MEMORY_SETTLE_TOLERANCE_GB
            and current["gpu_gb"] <= baseline["gpu_gb"] + _MEMORY_SETTLE_TOLERANCE_GB
        ):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# Model introspection
# ---------------------------------------------------------------------------
def _weight_disk_gb(model_dir: str) -> float:
    """Weight footprint from the on-disk IR .bin files. For an int8/int4 export this
    closely tracks resident weight memory and is available even if a case OOMs first."""
    total = 0
    for root, _dirs, files in os.walk(model_dir):
        for name in files:
            if name.endswith(".bin"):
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
    return round(total / (1024 ** 3), 2)


def _load_model_config(model_dir: str) -> dict | None:
    """Best-effort read of the exported IR's config.json. None if missing, so callers
    degrade to "no estimate" rather than failing the case."""
    try:
        with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def theoretical_kv_bytes_per_token(
    config: dict,
    kv_cache_dtype_bytes: float = 2,
    quantization_param_bytes: int = 0,
) -> float | None:
    """Persistent KV growth per token from the model's declared architecture.

    Only layers whose `layer_types` entry is "full_attention" hold a cache that grows with
    sequence length; "linear_attention" entries are Mamba/GatedDeltaNet-style recurrent
    states with an O(1) size. This is confirmed against the exported IR, not just the
    config label: only full_attention layers' `cache_params.past.{key,value}.N` state
    variables carry a dynamic sequence-length axis, while the linear layers'
    `cache_params.past.{conv,ssm}.N` variables have a fixed shape at any context length.

    VLM exports nest the causal-LM config under `text_config`; plain LLM configs have
    these keys top-level, so both layouts are read.

    Persistent cache only -- temporary prefill activations are deliberately excluded.
    Returns None when the config lacks num_hidden_layers / num_key_value_heads / head_dim.
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
    bytes_per_row = head_dim * kv_cache_dtype_bytes + quantization_param_bytes
    return 2 * growing_layers * num_kv_heads * bytes_per_row


def fixed_state_cache_bytes(model_dir: str) -> int:
    """Fixed linear-attention cache state size, read from the exported IR."""
    model_xml = next(
        (
            os.path.join(model_dir, name)
            for name in ("openvino_language_model.xml", "openvino_model.xml")
            if os.path.isfile(os.path.join(model_dir, name))
        ),
        None,
    )
    if model_xml is None:
        return 0

    total = 0
    seen = set()
    try:
        for _event, elem in ET.iterparse(model_xml, events=("start",)):
            variable_id = elem.attrib.get("variable_id", "")
            if not re.search(r"cache_params\.past\.(?:conv|ssm)\.", variable_id):
                continue
            if variable_id in seen:
                continue
            seen.add(variable_id)
            item_bytes = _IR_TYPE_BYTES.get(elem.attrib.get("variable_type", "").lower())
            shape = elem.attrib.get("variable_shape", "")
            if not item_bytes or not shape:
                continue
            elements = 1
            for value in shape.split(","):
                elements *= 1 if value == "?" else int(value)
            total += elements * item_bytes
    except (ET.ParseError, OSError, ValueError):
        return 0
    return total


def kv_cache_dtype_bytes(ov_config: dict) -> float:
    precision = str((ov_config or {}).get("KV_CACHE_PRECISION", "")).strip().lower()
    return _KV_PRECISION_BYTES.get(precision, _DEFAULT_KV_BYTES)


def kv_quantization_param_bytes(ov_config: dict) -> int:
    precision = str((ov_config or {}).get("KV_CACHE_PRECISION", "")).strip().lower()
    return _KV_QUANT_PARAMS_BYTES if precision in _COMPRESSED_KV_PRECISIONS else 0


def expected_kv_gb(model_config: dict | None, fixed_bytes: int, ov_config: dict,
                   context_tokens: int) -> float | None:
    """Persistent KV cache this model needs at `context_tokens` under `ov_config`'s precision.

    Recomputed per profile rather than once per model: KV_CACHE_PRECISION changes
    bytes-per-token, and the `cache_size: auto` derived from this has to follow it.
    """
    if not model_config:
        return None
    bytes_per_token = theoretical_kv_bytes_per_token(
        model_config, kv_cache_dtype_bytes(ov_config), kv_quantization_param_bytes(ov_config)
    )
    if bytes_per_token is None:
        return None
    return round((bytes_per_token * context_tokens + fixed_bytes) / (1024 ** 3), 2)


def auto_cache_size_gb(
    kv_gb: float | None, weight_disk_gb: float, budget_gb: float
) -> float | None:
    """Size the scheduler's KV pool from the model's architecture instead of a magic number.

    The pool has to cover the persistent cache with room for block-allocation slack, but
    oversizing it is not free: at 160K on the 64 GB box, cache_size=8 passed, 16 pushed peak
    GPU to 34.2 GB, and 24 was rejected outright by the scheduler as larger than available
    memory. Deriving it from `expected_kv_gb` lands near the hand-tuned 8 for Qwen3.5-9B at
    f16 while automatically shrinking for a u8/int4 cache and growing for a longer context --
    which is what lets a 9B and a 35B share one profile definition.

    Returns None when the architecture is unknown, so the caller leaves cache_size unset
    and lets OpenVINO manage the pool rather than guessing.
    """
    if not kv_gb:
        return None
    needed = kv_gb * _AUTO_CACHE_SAFETY
    ceiling = max(_AUTO_CACHE_MIN_GB, budget_gb - weight_disk_gb - _AUTO_CACHE_RESERVE_GB)
    return round(max(_AUTO_CACHE_MIN_GB, min(math.ceil(needed), ceiling)), 2)


def validate_fixed_cache_size(cache_size, kv_gb: float | None) -> None:
    """Reject a fixed scheduler pool that cannot hold the estimated persistent KV."""
    if cache_size is None or str(cache_size).lower() == "auto" or kv_gb is None:
        return
    try:
        fixed_gb = float(cache_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cache_size must be a number or 'auto', got {cache_size!r}") from exc
    if fixed_gb < kv_gb:
        raise ValueError(
            f"cache_size={fixed_gb:g} GB is below the estimated {kv_gb:g} GB persistent KV; "
            "increase cache_size or use 'auto'"
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def _namespace_to_dict(value):
    """Undo config_loader's dict -> SimpleNamespace conversion.

    Convenient for a fixed schema, wrong for this one: `ov` is an open-ended property bag
    whose keys are OpenVINO's, and it has to reach the pipeline constructor as a plain dict.
    """
    if isinstance(value, SimpleNamespace):
        return {key: _namespace_to_dict(val) for key, val in vars(value).items()}
    if isinstance(value, dict):
        return {key: _namespace_to_dict(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_namespace_to_dict(item) for item in value]
    return value


def _coerce(value: str):
    lowered = str(value).lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _parse_overrides(items) -> dict:
    """`["KV_CACHE_PRECISION=f16", "CACHE_DIR="]` -> a merge map; empty value drops a key.

    Without the drop form, the only way to measure the box without one of the config's
    properties would be to edit the config file -- the kind of uncommitted local change
    that makes a run unreproducible.
    """
    parsed = {}
    for item in items or []:
        key, sep, value = str(item).partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"expected KEY=VALUE (or KEY= to remove a key), got: {item!r}")
        parsed[key.strip()] = value.strip()
    return parsed


def _apply_overrides(base: dict, overrides: dict, coerce: bool) -> dict:
    merged = dict(base)
    for key, value in (overrides or {}).items():
        if value == "":
            merged.pop(key, None)
        else:
            merged[key] = _coerce(value) if coerce else value
    return merged


def format_config(config: dict) -> str:
    """One-line rendering for the console, the CSV column and the summary."""
    if not config:
        return "(none)"
    return " ".join(f"{key}={value}" for key, value in sorted(config.items()))


def _resolve_profiles(raw_profiles, names_filter, ov_overrides, sched_overrides) -> list:
    profiles = _namespace_to_dict(raw_profiles) or []
    if not isinstance(profiles, list) or not profiles:
        raise SystemExit("`profiles` must be a non-empty list of {name, ov, scheduler} entries")

    resolved = []
    for entry in profiles:
        name = str(entry.get("name") or "").strip()
        if not name:
            raise SystemExit("every profile needs a `name`")
        resolved.append({
            "name": name,
            "ov": _apply_overrides(entry.get("ov") or {}, ov_overrides, coerce=False),
            # `scheduler: {}` is meaningful -- it selects the stateful pipeline -- so an
            # empty mapping is preserved rather than treated as "unset".
            "scheduler": _apply_overrides(entry.get("scheduler") or {}, sched_overrides, coerce=True),
        })

    if names_filter:
        wanted = {n.strip() for n in names_filter}
        unknown = wanted - {p["name"] for p in resolved}
        if unknown:
            raise SystemExit(
                f"unknown profile(s): {', '.join(sorted(unknown))}. "
                f"Available: {', '.join(p['name'] for p in resolved)}"
            )
        resolved = [p for p in resolved if p["name"] in wanted]
    return resolved


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark candidate summarizer models at long context across "
        "OpenVINO configuration profiles, and rank the profiles by measured throughput.",
    )
    parser.add_argument(
        "--config",
        default=_DEFAULT_CONFIG_PATH,
        help="model-specific benchmark config (default: config_qwen3.5_9b.yaml)",
    )
    parser.add_argument("--models", nargs="+", help="override benchmark.models")
    parser.add_argument("--contexts", type=int, nargs="+", help="override benchmark.context_tokens")
    parser.add_argument("--profiles", nargs="+", help="run only these profiles, by name")
    parser.add_argument("--output-tokens", type=int, help="override benchmark.output_tokens")
    parser.add_argument("--warmup", type=int, help="override benchmark.warmup")
    parser.add_argument("--iterations", type=int, help="override benchmark.iterations")
    parser.add_argument("--device", help="override model.device (e.g. GPU, CPU)")
    parser.add_argument("--weight-format", help="override model.weight_format")
    parser.add_argument("--output-dir", help="override benchmark.output_dir")
    parser.add_argument(
        "--pipeline-config", nargs="+", metavar="KEY=VALUE",
        help="add to or override every profile's OpenVINO properties; KEY= removes one",
    )
    parser.add_argument(
        "--scheduler-config", nargs="+", metavar="KEY=VALUE",
        help="add to or override every profile's SchedulerConfig values; KEY= removes one",
    )
    parser.add_argument(
        "--list-profiles", action="store_true",
        help="print the resolved run matrix and exit -- no model, GPU or OpenVINO needed",
    )
    return parser.parse_args()


def _load_settings(args) -> dict:
    cfg = load_config(args.config)
    model = getattr(cfg, "model", None)
    bench = getattr(cfg, "benchmark", None)
    if model is None or bench is None:
        raise SystemExit(
            f"{args.config} must define both `model` and `benchmark`. "
            "See docs/dev-guide/context-bench/context_bench_guide.md."
        )

    contexts = sorted(set(args.contexts or bench.context_tokens))
    if not contexts or any(not isinstance(c, int) or isinstance(c, bool) or c <= 0 for c in contexts):
        raise SystemExit("benchmark.context_tokens must contain only positive integers")

    max_ram_pct = getattr(bench, "max_system_memory_pct", 80)
    if not 0 < max_ram_pct <= 100:
        raise SystemExit("benchmark.max_system_memory_pct must be in (0, 100]")

    iterations = args.iterations if args.iterations is not None else bench.iterations
    if iterations < 1:
        raise SystemExit("benchmark.iterations must be at least 1")

    return {
        "provider": model.provider,
        "models_base_path": model.models_base_path,
        "device": args.device or model.device,
        "weight_format": args.weight_format or model.weight_format,
        "models": args.models or bench.models,
        "context_tokens": contexts,
        "output_tokens": args.output_tokens or bench.output_tokens,
        "warmup": args.warmup if args.warmup is not None else bench.warmup,
        "iterations": iterations,
        "timeout_sec": bench.timeout_sec,
        "max_system_memory_pct": max_ram_pct,
        "gpu_memory_budget_gb": float(getattr(bench, "gpu_memory_budget_gb", 0) or 0),
        "cache_dir": getattr(bench, "cache_dir", None),
        "output_dir": args.output_dir or bench.output_dir,
        "profiles": _resolve_profiles(
            getattr(cfg, "profiles", None),
            args.profiles,
            _parse_overrides(args.pipeline_config),
            _parse_overrides(args.scheduler_config),
        ),
    }


def _model_ir_dir(base: str, provider: str, model_name: str, weight_format: str) -> str:
    # Mirrors utils/ensure_model.py::get_model_path, parameterized per candidate.
    return os.path.join(base, provider, f"{model_name.replace('/', '_')}_{weight_format}")


def _ir_ready(model_dir: str) -> bool:
    if not os.path.isdir(model_dir):
        return False
    names = []
    for _root, _dirs, files in os.walk(model_dir):
        names.extend(files)
    return (
        any(_MODEL_IR_RE.search(n) for n in names)
        and "openvino_tokenizer.xml" in names
        and "openvino_detokenizer.xml" in names
    )


def _prep_command(model_name: str, model_dir: str, weight_format: str) -> str:
    return (
        f'optimum-cli export openvino --model "{model_name}" --trust-remote-code '
        f'--weight-format {weight_format} "{model_dir}"'
    )


def _preflight_environment_check() -> None:
    """Fail fast with an actionable message if this interpreter cannot import what
    trial_runner needs, rather than letting every case repeat the same import failure.
    Missing openvino_genai/transformers almost always means "wrong Python interpreter":
    they live in the project's backend venv, not the one on PATH."""
    missing = [name for name in _REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if not missing:
        return

    lines = [
        f"Missing required package(s) in this interpreter ({sys.executable}): {', '.join(missing)}.",
        "This is almost always the wrong Python environment, not a code bug -- the OpenVINO "
        "stack lives in the project's backend venv, not the interpreter picked up from PATH.",
        "Simplest fix -- use the launcher, which creates the venv if needed and re-runs this "
        "tool with the same arguments:",
        "  " + " ".join([_LAUNCHER_SCRIPT, *sys.argv[1:]]),
    ]
    if os.path.exists(_BACKEND_VENV_PYTHON):
        lines.append(f"Or run directly with the venv interpreter at {_BACKEND_VENV_PYTHON}:")
        lines.append(f'  "{_BACKEND_VENV_PYTHON}" -m components.llm.context_bench.benchmark')
    else:
        lines.append(f"Or prepare the venv first (none found at {_BACKEND_VENV_PYTHON}):")
        lines.append(f"  {_SETUP_SCRIPT}")
        lines.append(
            "(If PowerShell blocks it: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)"
        )
    lines.append("(Use --list-profiles to check the run matrix without any of these packages.)")
    raise SystemExit("\n".join(lines))


# ---------------------------------------------------------------------------
# Case execution
# ---------------------------------------------------------------------------
def _crash_reason(exitcode) -> str:
    detail = _NATIVE_ABORT_EXIT_CODES.get(exitcode)
    return f"crashed:exitcode={exitcode}" + (f":{detail}" if detail else "")


def _run_case_subprocess(
    model_dir: str,
    device: str,
    context_tokens: int,
    output_tokens: int,
    warmup: int,
    iterations: int,
    timeout_sec: float,
    ov_config: dict,
    scheduler_config: dict,
    on_iteration=None,
    sample_interval: float = 0.5,
    poll_interval: float = 0.25,
    drain_timeout: float = 5.0,
) -> dict:
    """Run one case to completion, OOM, native abort, or the hard timeout.

    Nothing stops the child on a memory threshold: the point is to find where this box
    actually breaks. Subprocess isolation is what makes that safe -- a native GPU abort
    near shared-memory exhaustion kills only the child, and the parent's sampler has
    already recorded the high-water mark that explains it.

    Milestones are tracked as they arrive rather than read off the final result, because
    a child killed by the timeout or by a native abort never posts one: "hung in prefill"
    and "hung in decode" are different findings about the same context length.
    """
    baseline = _read_mem()
    sampler = _MemorySampler(interval=sample_interval)
    sampler.start()

    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=trial_runner.run_case,
        args=(
            model_dir, device, context_tokens, output_tokens, warmup, iterations,
            result_queue, ov_config, scheduler_config,
        ),
    )
    process.start()

    loaded_mem = None
    result = None
    state = {
        "load_ok": False,
        "stage_reached": trial_runner.STAGE_START,
        "prompt_tokens": 0,
        "gpu_budget_driver_gb": None,
        "iterations": [],
    }

    def _consume(msg: dict, child_alive: bool) -> bool:
        """Apply one child message; True once the terminal `done` has arrived."""
        nonlocal loaded_mem, result
        event = msg.get("event")
        if event == "device":
            state["gpu_budget_driver_gb"] = msg.get("gpu_budget_gb")
        elif event == "loaded":
            state["load_ok"] = True
            state["stage_reached"] = trial_runner.STAGE_LOADED
            # Snapshot construction memory before inference. Only meaningful while the
            # child lives; a post-mortem "loaded" must not fabricate a delta.
            if child_alive and loaded_mem is None:
                loaded_mem = _read_mem()
        elif event == "prompt":
            state["stage_reached"] = trial_runner.STAGE_PROMPT_BUILT
            state["prompt_tokens"] = msg.get("prompt_tokens", 0)
        elif event == "iteration_start":
            sampler.reset_window()
        elif event == "iteration":
            state["stage_reached"] = trial_runner.STAGE_DECODED
            record = {k: v for k, v in msg.items() if k != "event"}
            peak_ram, peak_gpu = sampler.window()
            record["peak_ram_gb"], record["peak_gpu_gb"] = peak_ram, peak_gpu
            state["iterations"].append(record)
            if on_iteration:
                on_iteration(record)
        elif event == "done":
            result = msg
            return True
        return False

    deadline = time.monotonic() + timeout_sec
    while result is None and time.monotonic() < deadline:
        try:
            msg = result_queue.get(timeout=poll_interval)
        except Empty:
            if not process.is_alive():
                break  # anything it queued is recovered by the drain below
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
        # The child posts its result and exits immediately without OpenVINO's teardown, so a
        # finished case's message can land in the pipe between the poll timing out and
        # is_alive() going False. Drain rather than report a completed run as a crash.
        drain_deadline = time.monotonic() + drain_timeout
        while time.monotonic() < drain_deadline:
            try:
                msg = result_queue.get(timeout=poll_interval)
            except Empty:
                break
            if _consume(msg, child_alive=False):
                timed_out = False
                break

    if not _wait_for_memory_settle(baseline):
        current = _read_mem()
        print(
            f"  [warn] memory has not returned to baseline after {_MEMORY_SETTLE_TIMEOUT_SEC:g}s "
            f"(RAM {current['ram_gb']:.1f} vs {baseline['ram_gb']:.1f} GB, GPU "
            f"{current['gpu_gb']:.1f} vs {baseline['gpu_gb']:.1f} GB); the next case's "
            "weights/KV split may be charged with the leftover",
            flush=True,
        )

    mem = {
        "peak_ram_gb": round(sampler.peak_ram, 2),
        "peak_ram_pct": round(sampler.peak_ram_pct, 1) if sampler.peak_ram_pct is not None else None,
        "min_available_ram_gb": (
            round(sampler.min_available_ram, 2) if sampler.min_available_ram is not None else None
        ),
        "peak_gpu_gb": round(sampler.peak_gpu, 2),
        "post_load_peak_ram_gb": _delta(sampler.peak_ram, loaded_mem["ram_gb"]) if loaded_mem else None,
        "post_load_peak_gpu_gb": _delta(sampler.peak_gpu, loaded_mem["gpu_gb"]) if loaded_mem else None,
        "ram_total_gb": round(baseline["ram_total_gb"], 2),
    }

    if result is not None:
        result.pop("event", None)
        # The parent's per-iteration records carry the memory windows the child cannot see.
        result["iterations"] = state["iterations"] or result.get("iterations") or []
        result["gpu_budget_driver_gb"] = (
            state["gpu_budget_driver_gb"] or result.pop("gpu_budget_gb", None)
        )
        result.pop("gpu_budget_gb", None)
        result.update(mem)
        return result

    # A native GPU abort near shared-memory exhaustion kills the child without a Python
    # exception, so it surfaces as `crashed`. Reaching here means the abort happened before
    # the child could report -- during load, prefill or decode -- because the teardown abort
    # can no longer occur and a result racing the exit is recovered by the drain above.
    return {
        "context_tokens": context_tokens,
        "load_ok": state["load_ok"],
        "load_time_s": None,
        "prompt_tokens": state["prompt_tokens"],
        "stage_reached": state["stage_reached"],
        "iterations": state["iterations"],
        "gpu_budget_driver_gb": state["gpu_budget_driver_gb"],
        "error": "timeout" if timed_out else _crash_reason(process.exitcode),
        **mem,
    }


def _status(result: dict) -> str:
    """Name the outcome. Deliberately coarse: a benchmark needs to know whether a number
    is usable and, if not, roughly why -- not to adjudicate between six shades of failure.
    `oom` / `unsupported` / `gpu_abort` come from the child's own error text."""
    error = str(result.get("error") or "")
    if not error:
        return "ok" if metrics.measured(result.get("iterations")) else "no_output"
    if error == "timeout":
        return "timeout"
    if error.startswith("crashed"):
        return "crashed"
    parts = error.split(":", 2)
    classification = parts[1] if len(parts) >= 2 else "error"
    if classification == "unsupported":
        return "unsupported"
    if classification in ("oom", "gpu_abort"):
        return classification
    return "load_error" if not result.get("load_ok") else "error"


def _apply_memory_status(case: dict) -> dict:
    """Demote a completed measurement that exceeded a configured memory limit."""
    if case.get("status") != "ok":
        return case
    if case.get("gpu_budget_exceeded"):
        case["status"] = "gpu_memory_limit"
    elif case.get("system_memory_limit_exceeded"):
        case["status"] = "memory_limit"
    return case


def _run_case(model_name, model_dir, profile, context_tokens, settings, static) -> dict:
    """One (model, profile, context) point: resolve its config, run it, score it."""
    device = settings["device"]
    ov_config = dict(profile["ov"])
    if settings.get("cache_dir"):
        ov_config.setdefault("CACHE_DIR", str(settings["cache_dir"]))

    kv_gb = expected_kv_gb(
        static["model_config"], static["fixed_state_bytes"], ov_config, context_tokens
    )

    scheduler_config = dict(profile["scheduler"])
    validate_fixed_cache_size(scheduler_config.get("cache_size"), kv_gb)
    if str(scheduler_config.get("cache_size")).lower() == "auto":
        auto = auto_cache_size_gb(kv_gb, static["weight_disk_gb"], settings["gpu_memory_budget_gb"])
        if auto is None:
            scheduler_config.pop("cache_size")
        else:
            scheduler_config["cache_size"] = auto
            if kv_gb and auto < kv_gb:
                print(
                    f"  [warn] auto cache_size {auto} GB is below the estimated {kv_gb} GB KV "
                    f"cache at {context_tokens:,} tokens -- the budget leaves no more room",
                    flush=True,
                )

    print(
        f"\n[{model_name} | {profile['name']} | {context_tokens:,} tok] "
        f"{settings['warmup']} warmup + {settings['iterations']} iterations, timeout "
        f"{settings['timeout_sec']:g}s",
        flush=True,
    )
    print(f"  ov: {format_config(ov_config)}", flush=True)
    print(f"  scheduler: {format_config(scheduler_config) if scheduler_config else '(stateful)'}",
          flush=True)

    def _echo(record):
        print("  " + metrics.format_iteration(record), flush=True)

    try:
        result = _run_case_subprocess(
            model_dir, device, context_tokens, settings["output_tokens"],
            settings["warmup"], settings["iterations"], settings["timeout_sec"],
            ov_config, scheduler_config, on_iteration=_echo,
        )
    except Exception as exc:  # noqa: BLE001 - a case the orchestrator could not carry out
        # Spawning a fresh interpreter that re-imports the OpenVINO stack is itself work the
        # box can fail at, and letting that unwind would discard every measurement taken so far.
        traceback.print_exc()
        result = {"error": f"orchestrator:error:{exc}", "iterations": [], "load_ok": False}

    peak_gpu = result.get("peak_gpu_gb") or 0.0
    budget = settings["gpu_memory_budget_gb"]
    ram_pct = result.get("peak_ram_pct")

    case = {
        "model": model_name,
        "profile": profile["name"],
        "context_tokens": context_tokens,
        "device": device,
        "weight_format": settings["weight_format"],
        "status": _status(result),
        "error": result.get("error"),
        "stage_reached": result.get("stage_reached"),
        "kv_cache_precision": ov_config.get("KV_CACHE_PRECISION", "(plugin default)"),
        "cache_size_gb": scheduler_config.get("cache_size"),
        "ov_config": format_config(ov_config),
        "scheduler_config": format_config(scheduler_config) if scheduler_config else "(stateful)",
        "load_ok": result.get("load_ok"),
        "load_time_s": result.get("load_time_s"),
        "prompt_tokens": result.get("prompt_tokens"),
        "weight_disk_gb": static["weight_disk_gb"],
        "expected_kv_gb": kv_gb,
        "gpu_budget_gb": budget or None,
        "gpu_budget_driver_gb": result.get("gpu_budget_driver_gb"),
        "peak_gpu_pct_of_budget": round(peak_gpu / budget * 100, 1) if peak_gpu and budget else None,
        "system_memory_limit_exceeded": bool(
            ram_pct is not None and ram_pct > settings["max_system_memory_pct"]
        ),
        "gpu_budget_exceeded": bool(budget and peak_gpu > budget),
        **{k: result.get(k) for k in (
            "peak_ram_gb", "peak_ram_pct", "min_available_ram_gb",
            "post_load_peak_ram_gb", "post_load_peak_gpu_gb", "peak_gpu_gb",
        )},
        **metrics.aggregate(result.get("iterations")),
    }
    case["_iterations"] = result.get("iterations") or []

    # A measurement taken past the memory ceiling is a real measurement of an unusable
    # configuration, so it keeps its numbers and is demoted rather than deleted.
    return _apply_memory_status(case)


def _format_case(case: dict) -> str:
    if case["status"] not in ("ok", "memory_limit", "gpu_memory_limit"):
        detail = f" ({case['error']})" if case.get("error") else ""
        stage = case.get("stage_reached")
        where = (
            f" [failed in {trial_runner.failing_stage(stage)}]"
            if stage and stage != trial_runner.STAGE_DECODED else ""
        )
        return f"  => {case['status'].upper()}{where}{detail}"
    return (
        f"  => {case['status'].upper()} | TPOT {case.get('other_tokens_avg_latency', 0):.1f} ms/token "
        f"| TTFT {case.get('first_token_latency', 0) / 1000:.1f}s "
        f"| decode {case.get('decode_throughput', 0):.2f} tok/s "
        f"| e2e {case.get('e2e_throughput', 0):.1f} tok/s "
        f"| peak RAM {case.get('peak_ram_gb', 0):.1f} GB ({case.get('peak_ram_pct') or 0:.1f}%) "
        f"| peak GPU {case.get('peak_gpu_gb', 0):.1f} GB "
        f"({case.get('peak_gpu_pct_of_budget') or 0:.1f}% of budget)"
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt(value, spec="{:.1f}") -> str:
    return spec.format(value) if isinstance(value, (int, float)) else "--"


def _leaderboard(cases: list) -> list:
    """Usable profiles ranked by TPOT first, then TTFT as the tie-breaker."""
    return sorted(
        (
            c for c in cases
            if c["status"] == "ok" and c.get("other_tokens_avg_latency") is not None
        ),
        key=lambda c: (
            c["other_tokens_avg_latency"],
            c.get("first_token_latency", float("inf")),
        ),
    )


def write_reports(output_dir: str, settings: dict, cases: list, platform_info: dict,
                  completed: bool) -> None:
    """(Re)write summary.json / summary.md / summary.csv for the run so far.

    Rewritten after every case, not once at the end, because this tool's job is to push the
    box until it breaks and it can break hard enough to take the orchestrator with it. When
    that happened previously no summary was written at all and the file on disk still
    described an earlier run -- reporting a PASS at 160K that the real trials had just
    disproved. A stale report that looks current is worse than none.
    """
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "completed": completed,
        "device": settings["device"],
        "weight_format": settings["weight_format"],
        "context_tokens": settings["context_tokens"],
        "output_tokens": settings["output_tokens"],
        "warmup": settings["warmup"],
        "iterations": settings["iterations"],
        "max_system_memory_pct": settings["max_system_memory_pct"],
        "gpu_memory_budget_gb": settings["gpu_memory_budget_gb"],
        "hardware": platform_info,
        "cases": [{k: v for k, v in c.items() if not k.startswith("_")} for c in cases],
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    for case in cases:
        StorageManager.save_csv(
            os.path.join(output_dir, "summary.csv"),
            {field: case.get(field) for field in CASE_FIELDS},
            headers=CASE_FIELDS,
            append=False if case is cases[0] else True,
        )

    banner = [] if completed else [
        "> **Run in progress or ended early.** Only the cases below have been measured; "
        "`iterations.csv` has every iteration that ran.",
        "",
    ]
    lines = [
        "# Long-Context Benchmark Summary",
        "",
        *banner,
        f"Generated: {summary['generated_at']}",
        "",
        f"Hardware: {platform_info.get('Processor', '--')}, {platform_info.get('Memory', '--')} RAM, "
        f"{platform_info.get('iGPU', '--')}",
        f"Device: {settings['device']} | Weights: {settings['weight_format']} | "
        f"Output: {settings['output_tokens']} tokens | "
        f"{settings['warmup']} warmup + {settings['iterations']} measured iterations",
        f"Budgets: system RAM <= {settings['max_system_memory_pct']:g}%, "
        f"GPU <= {settings['gpu_memory_budget_gb']:g} GB",
        "",
        "Latencies are milliseconds and throughputs tokens/second, in llm_bench's units. "
        "Every figure is the **median** of the measured iterations; the warm-up is excluded. "
        "Profiles are ranked by TPOT (lower is better), with TTFT as the tie-breaker.",
        "",
    ]

    for context in settings["context_tokens"]:
        at_context = [c for c in cases if c["context_tokens"] == context]
        if not at_context:
            continue
        lines += [
            f"## {context:,} tokens",
            "",
            "| Model | Profile | Status | TPOT ms/token | TTFT s | Decode tok/s | "
            "E2E tok/s | Prefill tok/s | Peak RAM | Peak GPU (% of 59 GB) | Expected KV | cache_size |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        ranked = _leaderboard(at_context)
        # Identity, not equality: two profiles can produce byte-identical rows, and `in`
        # on dicts would drop the duplicate from the report entirely.
        ranked_ids = {id(c) for c in ranked}
        rest = [c for c in at_context if id(c) not in ranked_ids]
        for case in ranked + rest:
            usable = case["status"] in ("ok", "memory_limit", "gpu_memory_limit")
            ttft = case.get("first_token_latency")
            gpu = _fmt(case.get('peak_gpu_gb'))
            gpu_pct = _fmt(case.get('peak_gpu_pct_of_budget'))
            lines.append(
                f"| {case['model']} | {case['profile']} | {case['status']} | "
                f"{_fmt(case.get('other_tokens_avg_latency')) if usable else '--'} | "
                f"{_fmt(ttft / 1000) if usable and ttft else '--'} | "
                f"{_fmt(case.get('decode_throughput'), '{:.2f}') if usable else '--'} | "
                f"{_fmt(case.get('e2e_throughput')) if usable else '--'} | "
                f"{_fmt(case.get('prefill_throughput')) if usable else '--'} | "
                f"{_fmt(case.get('peak_ram_gb'))} GB | {gpu} GB ({gpu_pct}%) | "
                f"{_fmt(case.get('expected_kv_gb'), '{:.2f}')} GB | "
                f"{_fmt(case.get('cache_size_gb'), '{:g}')} |"
            )
        lines.append("")

        best = ranked[0] if ranked else None
        if best:
            lines += [
                f"**Best TPOT at {context:,} tokens: `{best['profile']}`** on {best['model']} -- "
                f"TPOT {best['other_tokens_avg_latency']:.1f} ms/token "
                f"({best['decode_throughput']:.2f} decode tok/s), TTFT "
                f"{best['first_token_latency'] / 1000:.1f}s, e2e "
                f"{best['e2e_throughput']:.1f} tok/s, peak GPU "
                f"{best.get('peak_gpu_gb', 0):.1f} GB "
                f"({best.get('peak_gpu_pct_of_budget', 0):.1f}% of configured budget).",
                "",
                f"    ov:        {best['ov_config']}",
                f"    scheduler: {best['scheduler_config']}",
                "",
            ]

    failures = [
        c for c in cases
        if c["status"] not in ("ok", "memory_limit", "gpu_memory_limit")
    ]
    if failures:
        lines += ["## Cases that did not produce a measurement", "",
                  "| Model | Profile | Context | Status | Error |", "|---|---|---|---|---|"]
        for case in failures:
            error = str(case.get("error") or "").replace("|", "/").replace("\n", " ")[:180]
            lines.append(
                f"| {case['model']} | {case['profile']} | {case['context_tokens']:,} | "
                f"{case['status']} | {error} |"
            )
        lines.append("")

    with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _append_iterations(output_dir: str, case: dict) -> None:
    path = os.path.join(output_dir, "iterations.csv")
    for record in case.get("_iterations", []):
        row = {
            "model": case["model"],
            "profile": case["profile"],
            "context_tokens": case["context_tokens"],
            **{field: record.get(field) for field in metrics.ITERATION_FIELDS},
            "peak_ram_gb": record.get("peak_ram_gb"),
            "peak_gpu_gb": record.get("peak_gpu_gb"),
        }
        StorageManager.save_csv(path, row, headers=ITERATION_CSV_FIELDS, append=True)


def _safe_platform_info() -> dict:
    """Collect report metadata without WMI/COM calls.

    ``utils.platform_info`` queries GPU and NPU through WMI. On the PTL test
    system that can raise a native access violation, which Python cannot catch
    and which used to terminate a long benchmark before the model case started.
    Benchmark correctness does not depend on those descriptive fields, so use
    process-safe standard-library and psutil sources here.
    """
    info = {
        "Processor": platform.processor() or platform.machine() or "--",
        "Memory": "--",
        "Storage": "--",
        "iGPU": "Intel Graphics",
    }
    try:
        import psutil

        info["Memory"] = f"{math.ceil(psutil.virtual_memory().total / (1024 ** 3))} GB"
    except Exception:  # noqa: BLE001
        pass
    try:
        total = shutil.disk_usage(_SC_ROOT).total / (1024 ** 4)
        info["Storage"] = f"{total:.1f} TB"
    except Exception:  # noqa: BLE001
        pass
    return info


# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()
    settings = _load_settings(args)

    if args.list_profiles:
        print(f"Device: {settings['device']} | Weights: {settings['weight_format']}")
        print(f"Models: {', '.join(settings['models'])}")
        print(f"Contexts: {', '.join(f'{c:,}' for c in settings['context_tokens'])}")
        print(
            f"Iterations: {settings['warmup']} warmup + {settings['iterations']} measured, "
            f"{settings['output_tokens']} output tokens, timeout {settings['timeout_sec']:g}s"
        )
        total = len(settings["models"]) * len(settings["context_tokens"]) * len(settings["profiles"])
        print(f"\n{len(settings['profiles'])} profile(s), {total} case(s):\n")
        for profile in settings["profiles"]:
            print(f"  {profile['name']}")
            print(f"    ov:        {format_config(profile['ov'])}")
            print(
                "    scheduler: "
                + (format_config(profile["scheduler"]) if profile["scheduler"] else "(stateful)")
            )
        return

    _preflight_environment_check()

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(settings["output_dir"], run_id)
    os.makedirs(output_dir, exist_ok=True)
    platform_info = _safe_platform_info()

    print(f"Long-context benchmark run {run_id} -> {output_dir}", flush=True)
    print(
        f"Models: {settings['models']} | Contexts: {settings['context_tokens']} | "
        f"Profiles: {[p['name'] for p in settings['profiles']]}",
        flush=True,
    )

    cases = []
    completed = False
    try:
        for model_name in settings["models"]:
            model_dir = _model_ir_dir(
                settings["models_base_path"], settings["provider"], model_name,
                settings["weight_format"],
            )
            if not _ir_ready(model_dir):
                print(
                    f"\n[{model_name}] IR not found at {model_dir}\n"
                    f"  Run first: {_prep_command(model_name, model_dir, settings['weight_format'])}",
                    flush=True,
                )
                cases.append({
                    "model": model_name, "profile": "-", "context_tokens": 0,
                    "device": settings["device"], "weight_format": settings["weight_format"],
                    "status": "missing_ir",
                    "error": _prep_command(model_name, model_dir, settings["weight_format"]),
                })
                continue

            static = {
                "weight_disk_gb": _weight_disk_gb(model_dir),
                "model_config": _load_model_config(model_dir),
                "fixed_state_bytes": fixed_state_cache_bytes(model_dir),
            }
            print(
                f"\n=== {model_name} === weights on disk: {static['weight_disk_gb']:.1f} GB "
                f"({settings['weight_format']})",
                flush=True,
            )

            for context in settings["context_tokens"]:
                for profile in settings["profiles"]:
                    case = _run_case(model_name, model_dir, profile, context, settings, static)
                    print(_format_case(case), flush=True)
                    cases.append(case)
                    _append_iterations(output_dir, case)
                    write_reports(output_dir, settings, cases, platform_info, completed=False)
        completed = True
    finally:
        # Reached on Ctrl-C and on an unhandled failure too: the iterations already recorded
        # are worth a report, and the banner says the run ended early.
        try:
            write_reports(output_dir, settings, cases, platform_info, completed=completed)
        except Exception:  # noqa: BLE001 - must not mask whatever is already unwinding
            traceback.print_exc()
        else:
            best = _leaderboard(cases)
            if best:
                top = best[0]
                print(
                    f"\nBest TPOT: {top['profile']} on {top['model']} at "
                    f"{top['context_tokens']:,} tokens -- "
                    f"{top['other_tokens_avg_latency']:.1f} ms/token, "
                    f"{top['decode_throughput']:.2f} decode tok/s, "
                    f"TTFT {top['first_token_latency'] / 1000:.1f}s",
                    flush=True,
                )
            print(
                f"Reports written to {output_dir} (iterations.csv, summary.csv, summary.md, "
                "summary.json)" + ("" if completed else " -- run ended early, marked incomplete"),
                flush=True,
            )


if __name__ == "__main__":
    main()
