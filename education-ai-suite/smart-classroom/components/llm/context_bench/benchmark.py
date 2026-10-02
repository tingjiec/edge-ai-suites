# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Long-context benchmark for candidate summarizer models on this hardware.

Answers two questions per (model, context length):

  * Can this box prefill and decode it at all, within its memory budget?
  * Which OpenVINO configuration does it fastest?

The second question is why this exists in its current form. It benchmarks a
matrix of named `profiles` -- KV precision, prefill chunk size, continuous
batching vs the stateful pipeline, multi-token prediction and its candidate
count -- and ranks them by TPOT, then TTFT,
following llm_bench's methodology: one warm-up iteration that is
excluded from every statistic, N measured iterations, and the median reported
with min/max so run-to-run spread is visible. The same 160K configuration was
previously measured at 247.0s and 349.5s on two single-shot runs; a single
sample cannot tell a configuration difference from noise, which is the whole
reason iterations are not optional here.

The default (throughput) path does NOT judge answer quality -- content is irrelevant to a
capacity and throughput measurement, only the token volume and the clock matter. `--accuracy`
adds two suites that do: RULER's synthetic probes (retrieval, multi-hop tracing, aggregation)
scored against known ground truth, and a who_what_benchmark-style fidelity comparison of the
real task's output against the baseline profile's. So precision loss from quantization or
speculative decoding is visible next to the speed it bought, whether it shows up as a missed
fact or as worse prose. See `accuracy.py`.

An accuracy run reports speed and resources too: its probes carry the same timing records the
throughput path produces, so one run gives TTFT, TPOT, memory and every accuracy metric. That
is the shape of the whole tool -- ONE config per model (throughput matrix + accuracy suites)
and ONE `report.txt`, rewritten after every case and printed to the console at the end.

Standalone diagnostic: reads its own bundled model config, never
smart-classroom/config.yaml, and running it never affects the application.

    .\\components\\llm\\context_bench\\run_benchmark.ps1
    .\\components\\llm\\context_bench\\run_benchmark.ps1 --accuracy
    .\\components\\llm\\context_bench\\run_benchmark.ps1 --accuracy --list-profiles

Equivalent with the right interpreter already active, run from smart-classroom/
so relative model paths resolve:

    python -m components.llm.context_bench.benchmark

See docs/dev-guide/context-bench/context_bench_guide.md.
"""

from __future__ import annotations

import argparse
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

from components.llm.context_bench import (
    accuracy,
    metrics,
    scoring,
    tasks,
    trial_runner,
    wwb_adapter,
)
from utils.config_loader import load_config

_TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG_PATH = os.path.join(_TOOL_DIR, "config_qwen3.6_35b.yaml")

# smart-classroom/ is 3 levels up (context_bench -> llm -> components -> smart-classroom).
_SC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_TOOL_DIR)))
# setup-smart-classroom.ps1 creates the backend venv as a sibling of smart-classroom/.
_BACKEND_VENV_PYTHON = os.path.join(
    os.path.dirname(_SC_ROOT), "smartclassroom", "Scripts", "python.exe"
)
_LAUNCHER_SCRIPT = os.path.join(_TOOL_DIR, "run_benchmark.ps1")
_SETUP_SCRIPT = os.path.join(_TOOL_DIR, "setup_env.ps1")

_REQUIRED_MODULES = ("openvino_genai", "transformers", "psutil")

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

# Extra pool an MTP profile needs beyond the cache its two models actually hold.
#
# Not the same thing as `mtp_head_layers`: that accounts for the draft head's own KV, which
# on Qwen3.8-27B is one full-attention layer against the target's 16, about +6%. The pool
# requirement grows far more than that. Measured at 32K on this box, where the target's
# persistent cache estimates at 2.1 GB: `cache_size=3` ran the no-MTP baseline comfortably
# and had every MTP profile *dropped before prefill* --
#
#   Request 0 was dropped by the scheduler because it did not fit in the available
#   cache budget (out of memory).
#
# -- while `cache_size=4` ran the same profile. Speculative decoding builds a second
# continuous-batching pipeline for the draft model and genai sizes both from the single
# `cache_size` this profile sets, so the pool has to cover the pair. The exact split genai
# applies is not documented and this factor is not derived from it: 2.0 is a margin over the
# smallest value measured to work, chosen because the failure mode is a case that loads a
# 14 GB model and then produces nothing.
_MTP_CACHE_POOL_FACTOR = 2.0

# Below this point most drafted candidates are discarded. This is diagnostic rather than a
# validity threshold: a low-acceptance profile can still win on TPOT, but a single-profile
# run needs to say that lowering k is the next measurement rather than implying that an
# OpenVINO property can make the same six candidates agree more often.
_LOW_MTP_ACCEPTANCE_RATE = 0.6

# Above this the draft head is agreeing so often that the profile is probably leaving yield on
# the table: acceptance this high means a *larger* candidate batch would likely commit more
# tokens per verification pass. The objective is tokens/step (= 1 + k * acceptance), not
# acceptance itself -- measured on this box k=4 (50% accepted, 2.90 tok/step) beat k=2 (74%,
# 2.44) on TPOT, so a high-acceptance k=2 run should be told to try k=3/k=4, not celebrated.
_HIGH_MTP_ACCEPTANCE_RATE = 0.7

# Stop suggesting ever-larger k past here. Acceptance falls with k, so in practice the
# high-acceptance branch stops firing on its own well before this; the ceiling is a backstop so
# a pathologically agreeable draft head cannot produce an unbounded "try k+1" chain.
_MTP_TOKEN_SUGGESTION_CEILING = 8

# dFlash's ceiling is the draft's own: the shipped Qwen3.6 export has block_size=16, seed
# token included, so it cannot propose more than 15 candidates per pass.
_DFLASH_TOKEN_SUGGESTION_CEILING = 15

# `SchedulerConfig.max_num_seqs` when a profile leaves it out. On a hybrid model this is not
# just a batch-size default: the paged backend reserves one full linear-attention state per
# schedulable sequence out of the same `cache_size` pool the KV blocks come from -- see
# `expected_kv_gb`.
_GENAI_DEFAULT_MAX_NUM_SEQS = 256

# Statuses whose numbers are real measurements and are printed as such. The two
# memory-limit ones are measurements of a configuration this box cannot use, so they keep
# every figure and are only excluded from the ranking -- see `_leaderboard`.
_USABLE_STATUSES = ("ok", "memory_limit", "gpu_memory_limit")

# Measured by the orchestrator's sampler, not the child, and named once so the three places
# that move them -- the sampler's result, the case row and the CSV header -- cannot drift.
_CASE_MEMORY_FIELDS = (
    "peak_ram_gb", "peak_ram_pct", "min_available_ram_gb", "post_load_peak_ram_gb",
    "peak_gpu_gb", "post_load_peak_gpu_gb",
)

REPORT_NAME = "report.txt"

# The prompts the throughput iterations can decode; the first is the default. See
# `throughput_task_prompt`. `_APP_CONFIG_PATH` is the application config, relative to the
# smart-classroom/ working directory the benchmark runs from.
THROUGHPUT_TASKS = ("summary_2s", "classroom_summary")
_APP_CONFIG_PATH = "config.yaml"


# ---------------------------------------------------------------------------
# Memory sampling (parent side)
#
# RAM / GPU counters are system-wide, so sampling from the orchestrator captures the
# child's footprint -- and unlike sampling inside the child, these readings survive a
# child that is killed on a timeout, exactly the case where memory matters most.
# ---------------------------------------------------------------------------
def _read_mem() -> dict:
    ram_used = ram_pct = available_ram = None
    try:
        import psutil

        vm = psutil.virtual_memory()
        ram_used, ram_pct = vm.used / (1024 ** 3), vm.percent
        available_ram = vm.available / (1024 ** 3)
    except Exception:  # noqa: BLE001
        pass

    gpu_gb = None
    try:
        from monitoring.scripts.windows.collect_gpu import get_gpu_memory_total

        used_mb = get_gpu_memory_total()[0]
        if used_mb is not None:
            gpu_gb = used_mb / 1024
    except Exception:  # noqa: BLE001
        pass

    return {
        "ram_gb": ram_used,
        "ram_pct": ram_pct,
        "available_ram_gb": available_ram,
        "gpu_gb": gpu_gb,
    }


class _TimeWeightedMean:
    """Mean of one counter over a window, weighted by how long each reading stood.

    A plain average of the samples would be biased by the sampler's own jitter: `_read_mem()`
    is a PDH query costing tens of milliseconds and that cost varies with load, so samples are
    not evenly spaced and the readings that happened to come back fast would carry the same
    weight as ones that stood for twice as long. Crediting each reading with the time until
    the next one also makes the result independent of `interval`, so changing the sampling
    rate does not change the number.

    An unavailable counter contributes nothing rather than folding in as 0, matching
    `_MemorySampler._fold`.
    """

    __slots__ = ("_weighted", "_seconds", "_value", "_since")

    def __init__(self, value: float | None, now: float):
        self._weighted = 0.0
        self._seconds = 0.0
        self._value = value
        self._since = now

    def _settle(self, now: float) -> None:
        """Credit the standing reading with the time it stood, then open a new interval.
        Idempotent: once it has run there is no outstanding interval left to credit."""
        elapsed = now - self._since
        if self._value is not None and elapsed > 0:
            self._weighted += self._value * elapsed
            self._seconds += elapsed
        self._since = now

    def observe(self, value: float | None, now: float) -> None:
        self._settle(now)
        self._value = value

    def mean(self, now: float) -> float | None:
        """None only if no reading was ever available. A window shorter than one sample
        interval has no elapsed time to weight, so it reports the standing reading rather
        than nothing -- one sample is still a measurement."""
        self._settle(now)
        return self._weighted / self._seconds if self._seconds > 0 else self._value


class _MemorySampler(threading.Thread):
    """Peak and mean RAM/GPU and the low-water mark of free RAM, per case and per iteration.

    Peak and mean answer different questions and the report needs both. Peak is one 0.5 s
    sample and decides whether the box can run the configuration at all -- it is set by the
    transient prefill workspace. The time-weighted mean is what the configuration actually
    costs while it runs, which is the number that matters when the 59 GB GPU budget is shared
    with the rest of the application: at 160K a case spends minutes in decode holding far less
    than its peak, and ranking configurations by peak alone would call two very differently
    sized workloads equivalent.

    The low-water mark is instrumentation, not a trigger: nothing cancels a case on it.
    A case that passes with 7 GB still free has real headroom above it; one that passes
    with 0.3 GB free is at the wall. Reporting the minimum answers that; aborting on a
    threshold would make it unanswerable.

    `reset_window()` / `window()` carve the same stream into per-iteration slices so a
    warm-up's allocation spike is not charged to the measured iterations. The mean is
    per-window only, deliberately: a whole-case mean would average the load phase in with
    inference and describe neither.

    The fold and the reset are locked against each other. Without that they race: this
    thread reads `_window_ram`, the orchestrator resets it to the current reading, and then
    this thread writes back `max(pre-reset peak, sample)` -- restoring the peak the reset
    just cleared and charging the previous iteration's spike to the next one, which is the
    one thing `reset_window()` exists to prevent. `_read_mem()` is a PDH query taking tens
    of milliseconds, so the window in which that interleaving can happen is wide.
    """

    def __init__(self, interval: float = 0.5, baseline: dict | None = None):
        super().__init__(daemon=True)
        self._stop_event = threading.Event()
        self.interval = interval
        self.peak_ram = None
        self.peak_ram_pct = None
        self.peak_gpu = None
        self.min_available_ram = None
        self._window_ram = None
        self._window_gpu = None
        self._window_mean = {}
        self._lock = threading.Lock()
        # The caller's pre-case baseline is reused when it has one: `_read_mem()` is the
        # expensive part of a sample and querying it twice in a row at case start buys nothing.
        first = baseline if baseline is not None else _read_mem()
        self._open_window(first)
        self._observe(first)

    def _fold(self, m: dict, pick, key: str, attr: str) -> None:
        """Fold one reading into a running max/min. An unavailable counter stays absent
        rather than folding in as 0, which would read as "measured, and it was zero"."""
        value = m.get(key)
        if value is None:
            return
        current = getattr(self, attr)
        setattr(self, attr, value if current is None else pick(current, value))

    def _open_window(self, m: dict) -> None:
        """Discard the previous per-iteration window and start a new one at reading `m`.

        Window state only -- the case-wide peaks are never rewound, which is the whole point
        of keeping the two separate.
        """
        now = time.monotonic()
        with self._lock:
            self._window_ram = m.get("ram_gb")
            self._window_gpu = m.get("gpu_gb")
            self._window_mean = {
                key: _TimeWeightedMean(m.get(key), now) for key in ("ram_gb", "gpu_gb")
            }

    def _observe(self, m: dict) -> None:
        now = time.monotonic()
        with self._lock:
            for key, attr in (
                ("ram_gb", "peak_ram"),
                ("gpu_gb", "peak_gpu"),
                ("ram_pct", "peak_ram_pct"),
                ("ram_gb", "_window_ram"),
                ("gpu_gb", "_window_gpu"),
            ):
                self._fold(m, max, key, attr)
            self._fold(m, min, "available_ram_gb", "min_available_ram")
            for key, accumulator in self._window_mean.items():
                accumulator.observe(m.get(key), now)

    def reset_window(self) -> None:
        # Sampled before taking the lock: holding it across a PDH query would stall the
        # sampler thread for as long as the query takes.
        self._open_window(_read_mem())

    def window(self) -> dict:
        """This iteration's memory window, keyed as the iteration record's own fields."""
        now = time.monotonic()
        with self._lock:
            return {
                "peak_ram_gb": _round_optional(self._window_ram),
                "peak_gpu_gb": _round_optional(self._window_gpu),
                "mean_ram_gb": _round_optional(self._window_mean["ram_gb"].mean(now)),
                "mean_gpu_gb": _round_optional(self._window_mean["gpu_gb"].mean(now)),
            }

    def run(self):
        while not self._stop_event.is_set():
            self._observe(_read_mem())
            self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()


def _round_optional(value: float | None, digits: int = 2) -> float | None:
    return round(value, digits) if value is not None else None


def _delta(higher: float | None, lower: float | None) -> float | None:
    if higher is None or lower is None:
        return None
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
        # Vacuously true when neither counter is readable: with nothing to compare there is
        # nothing to wait for, and blocking the full timeout on every case would only slow
        # a run down on a box where memory is already unmeasurable.
        if all(
            current[key] <= baseline[key] + _MEMORY_SETTLE_TOLERANCE_GB
            for key in ("ram_gb", "gpu_gb")
            if current.get(key) is not None and baseline.get(key) is not None
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
    for root, _, files in os.walk(model_dir):
        for name in files:
            if name.endswith(".bin"):
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
    return round(total / (1024 ** 3), 2)


# The IR that carries a directory's language-model weights, in the order optimum-cli names it:
# VLM exports split the language model out, plain causal-LM exports (and dFlash drafts) do not.
_LANGUAGE_MODEL_IRS = ("openvino_language_model.xml", "openvino_model.xml")

# The model-level rt_info, NNCF's record included, is the last element of an IR; this much of
# the file's tail holds it on every export seen here, even the 8 MB Qwen3.6 graph.
_IR_TAIL_BYTES = 64 * 1024


def ir_weight_precision(model_dir: str | None, names: tuple = _LANGUAGE_MODEL_IRS) -> str | None:
    """The weight quantization an IR actually carries, read from its NNCF rt_info.

    `model.weight_format` is only the label a directory was named with; the IR records what
    NNCF did -- mode, group size, and the backup mode for layers kept out of the primary one.
    Those decide both speed and how closely a dFlash draft can track its target, so they are
    printed per case rather than inferred from a folder name. Only the XML's tail is read,
    never the weights. "uncompressed" when the IR has no NNCF block (an fp16/bf16 export);
    None when there is no IR to read.
    """
    path = next(
        (os.path.join(model_dir, name) for name in names
         if model_dir and os.path.isfile(os.path.join(model_dir, name))),
        None,
    )
    if path is None:
        return None
    try:
        with open(path, "rb") as ir_file:
            ir_file.seek(0, os.SEEK_END)
            ir_file.seek(max(0, ir_file.tell() - _IR_TAIL_BYTES))
            tail = ir_file.read().decode("utf-8", "replace")
    except OSError:
        return None
    block = re.search(r"<weight_compression>(.*?)</weight_compression>", tail, re.S)
    if block is None:
        return "uncompressed"
    fields = dict(re.findall(r'<(\w+) value="([^"]*)"', block.group(1)))
    mode = fields.get("mode", "?")
    group = fields.get("group_size")
    parts = [mode, "per-channel" if group in (None, "-1") else f"g{group}"]
    try:
        ratio = float(fields.get("ratio", "1"))
    except ValueError:
        ratio = 1.0
    if ratio < 1.0:
        parts.append(f"{ratio:.0%} of layers")
    parts += [name.upper() for name in ("awq", "gptq") if fields.get(name) == "True"]
    if fields.get("scale_estimation") == "True":
        parts.append("SE")
    if fields.get("all_layers") == "True":
        parts.append("all layers")
    elif fields.get("backup_mode") and fields["backup_mode"] != mode:
        parts.append(f"backup {fields['backup_mode']}")
    return " ".join(parts)


def _precision_matches_label(precision: str | None, label: str | None) -> bool:
    """Whether a detected precision agrees with the config's `weight_format` label.

    Only integer labels are checked: "int4" must name the IR's mode (`int4_asym`). A float
    label or an unreadable IR is not evidence of a mismatch.
    """
    if not precision or not label or not label.lower().startswith("int"):
        return True
    return precision.split()[0].lower().startswith(label.lower())


def _draft_weight_precision(model_dir: str, mtp: dict) -> str | None:
    """Precision of the speculative draft: the external dFlash IR, or the export's MTP head."""
    if not (mtp or {}).get("enabled"):
        return None
    if mtp.get("strategy") == "dflash":
        return ir_weight_precision(mtp.get("model"))
    return ir_weight_precision(model_dir, (trial_runner.MTP_MODEL_FILE,))


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
    if layer_types is not None and (
        not isinstance(layer_types, (list, tuple)) or len(layer_types) != num_layers
    ):
        return None
    growing_layers = (
        sum(1 for t in layer_types if t == "full_attention") if layer_types else num_layers
    )
    if not growing_layers:
        return None
    bytes_per_row = head_dim * kv_cache_dtype_bytes + quantization_param_bytes
    return 2 * growing_layers * num_kv_heads * bytes_per_row


def mtp_head_layers(model_dir: str) -> int:
    """Full-attention layers in the MTP draft head, read from its exported IR.

    The draft head runs alongside the target model and keeps its own KV cache over the same
    context, so it is not free at long context: on Qwen3.8-27B it is one full-attention
    layer against the target's 16, a 6% surcharge that `cache_size: auto` has to cover or
    the pool it derives is short of what the pair actually allocates.

    Counted from the IR rather than assumed to be 1: `mtp_num_hidden_layers` is a config
    value that other MTP models set higher. Returns 0 when the head is absent or
    unreadable, which leaves the estimate at the target model's own requirement.
    """
    model_xml = os.path.join(model_dir, trial_runner.MTP_MODEL_FILE)
    if not os.path.isfile(model_xml):
        return 0
    keys = set()
    try:
        for _, elem in ET.iterparse(model_xml, events=("start",)):
            match = re.search(r"past_key_values\.(\d+)\.key", elem.attrib.get("variable_id", ""))
            if match:
                keys.add(match.group(1))
    except (ET.ParseError, OSError):
        return 0
    return len(keys)


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
        for _, elem in ET.iterparse(model_xml, events=("start",)):
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
                   context_tokens: int, sequences: int = 1,
                   mtp_layers: int = 0) -> float | None:
    """Cache this model needs at `context_tokens` under `ov_config`'s precision.

    Recomputed per profile rather than once per model: KV_CACHE_PRECISION changes
    bytes-per-token, and the `cache_size: auto` derived from this has to follow it.

    `fixed_bytes` is multiplied by `sequences`, which is `max_num_seqs` on the paged path and
    1 on the stateful one. The linear-attention state of a hybrid model does not grow with the
    context (§5.2), but the paged backend reserves one *per schedulable sequence* out of the
    same `cache_size` pool that holds the KV blocks -- so `max_num_seqs` is a cache-sizing
    knob here, not only a batch-size one. Measured on this box: Qwen3.5-9B carries 51 MiB of
    linear-attention state, so genai's default `max_num_seqs=256` reserves 12.75 GB and leaves
    a `cache_size=7` pool with no room for a single KV block -- the 160K request is then
    dropped as `GenerationStatus::IGNORED` after the model has already loaded. Counting the
    reservation here is what lets `auto_cache_size_gb` and `validate_fixed_cache_size` reject
    that arrangement up front instead of after a full load.

    `mtp_layers` adds the multi-token-prediction draft head's own full-attention cache,
    which is allocated over the same context and out of the same pool. It is 0 on a profile
    that does not run MTP, so the two configurations are sized against what each of them
    actually allocates rather than against one shared guess. The head reuses the target's
    attention shape -- same `num_key_value_heads` and `head_dim` -- so its per-token cost is
    the target's divided by the target's growing-layer count, times `mtp_layers`.
    """
    if not model_config:
        return None
    bytes_per_token = theoretical_kv_bytes_per_token(
        model_config, kv_cache_dtype_bytes(ov_config), kv_quantization_param_bytes(ov_config)
    )
    if bytes_per_token is None:
        return None
    if mtp_layers:
        per_layer = theoretical_kv_bytes_per_token(
            {**model_config.get("text_config", model_config), "num_hidden_layers": 1,
             "layer_types": None},
            kv_cache_dtype_bytes(ov_config), kv_quantization_param_bytes(ov_config),
        )
        bytes_per_token += (per_layer or 0) * mtp_layers
    reserved = fixed_bytes * max(1, sequences)
    return round((bytes_per_token * context_tokens + reserved) / (1024 ** 3), 2)


def auto_cache_size_gb(
    kv_gb: float | None, weight_disk_gb: float, budget_gb: float,
    pool_factor: float = 1.0,
) -> int | None:
    """Size the scheduler's KV pool from the model's architecture instead of a magic number.

    The pool has to cover the persistent cache with room for block-allocation slack, but
    oversizing it is not free: at 160K on the 64 GB box, cache_size=8 passed, 16 pushed peak
    GPU to 34.2 GB, and 24 was rejected outright by the scheduler as larger than available
    memory. Deriving it from `expected_kv_gb` lands near the hand-tuned 8 for Qwen3.5-9B at
    f16 while automatically shrinking for a u8/int4 cache and growing for a longer context --
    which is what lets a 9B and a 35B share one profile definition.

    Returns None when the architecture is unknown, so the caller leaves cache_size unset
    and lets OpenVINO manage the pool rather than guessing.

    Whole GiB, as an int: `SchedulerConfig.cache_size` takes an integer on genai 2026.5 and
    rejects a float outright, so a derived 2.0 would fail the case at pipeline construction
    with a pybind type error that says nothing about pool sizing.
    """
    if not kv_gb:
        return None
    needed = max(_AUTO_CACHE_MIN_GB, kv_gb * _AUTO_CACHE_SAFETY) * pool_factor
    ceiling = max(_AUTO_CACHE_MIN_GB, budget_gb - weight_disk_gb - _AUTO_CACHE_RESERVE_GB)
    return int(max(_AUTO_CACHE_MIN_GB, min(math.ceil(needed), math.floor(ceiling))))


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
            f"cache_size={fixed_gb:g} GB is below the estimated {kv_gb:g} GB persistent cache; "
            "increase cache_size, use 'auto', or -- on a hybrid model, where the estimate "
            "includes one linear-attention reservation per schedulable sequence -- lower "
            "max_num_seqs"
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
    """One-line rendering for the console and the report."""
    if not config:
        return "(none)"
    return " ".join(f"{key}={value}" for key, value in sorted(config.items()))


def format_mtp(mtp: dict, device: str | None = None) -> str:
    """One-line rendering of a profile's speculative-decoding setting."""
    if not (mtp or {}).get("enabled"):
        return "off"
    draft_device = mtp.get("device") or device
    where = f" on {draft_device}" if draft_device else ""
    strategy = mtp.get("strategy", "mtp")
    model = f" model={mtp['model']}" if mtp.get("model") else ""
    return f"{strategy} num_assistant_tokens={mtp['num_assistant_tokens']}{model}{where}"


# The two pipelines OpenVINO GenAI can run a case on, and the reason they are two profiles
# rather than one. Passing a SchedulerConfig at all selects continuous batching / paged
# attention -- there is no "stateful pipeline with a SchedulerConfig". And `cache_size` exists
# only on SchedulerConfig: on genai 2026.4 the GPU plugin advertises 51 properties and not one
# of them bounds the KV cache (`KV_CACHE_PRECISION` changes bytes per token, not the total).
# So "stateful, with the cache capped" is not a configuration that exists: capping the pool
# and running stateful are alternatives, and the shipped configs measure both.
PIPELINE_STATEFUL = "stateful"
PIPELINE_PAGED = "paged"

_PROFILE_KEYS = ("name", "ov", "scheduler", "mtp", "dflash")
_MTP_KEYS = ("enabled", "num_assistant_tokens", "device")
_DFLASH_KEYS = ("enabled", "model", "num_assistant_tokens", "device")

# What a profile's `mtp` section resolves to when it is absent. Normalized rather than left
# as None so every consumer -- the child, the CSV row, the report -- reads one shape.
_MTP_OFF = {
    "enabled": False,
    "strategy": None,
    "model": None,
    "num_assistant_tokens": None,
    "device": None,
}


def pipeline_mode(scheduler_config: dict | None) -> str:
    """Which pipeline a resolved profile selects, named for the report.

    Recorded per case because it is now a measured variable, not a footnote: reading a TTFT
    without knowing whether prefill ran as one SDPA pass or as paged chunks explains nothing.
    """
    return PIPELINE_PAGED if scheduler_config else PIPELINE_STATEFUL


def _resolve_mtp(name: str, section, scheduler: dict, override, kind: str = "mtp") -> dict:
    """Normalize and validate one profile's `mtp` section.

    Multi-token prediction is self-speculative decoding: the model's own draft head
    proposes `num_assistant_tokens` candidates per step and the main model verifies them
    in a single pass. Turning it on is a change of *what is being measured*, not a plugin
    property, which is why it is a profile section of its own rather than a key under `ov`.

    The rules rejected here are openvino_genai's, enforced early. `num_assistant_tokens`
    must be a positive integer -- genai asserts `> 0` -- and `max_num_batched_tokens` has
    to leave room for the whole candidate batch plus the token being verified, or the
    scheduler cannot admit a step. Both are cheap to check and expensive to discover after
    a 14 GB load.
    """
    if section is None:
        section = {}
    if isinstance(section, bool):  # `mtp: true` -- accepted, k comes from the override
        section = {"enabled": section}
    if not isinstance(section, dict):
        raise SystemExit(
            f"profile {name!r} mtp must be a mapping of {{{', '.join(_MTP_KEYS)}}} or a boolean"
        )
    unknown = [key for key in section if key not in _MTP_KEYS]
    if unknown:
        raise SystemExit(
            f"profile {name!r} mtp has unknown key(s): {', '.join(sorted(unknown))}. "
            f"An mtp section is {{{', '.join(_MTP_KEYS)}}}."
        )

    # `enabled` defaults to "yes, if this profile says anything about MTP at all": a
    # section written out with a candidate count and no `enabled: true` is a profile whose
    # author meant to run MTP, and silently ignoring it would report a baseline under an
    # `mtp_k3` name.
    if not bool(section.get("enabled", bool(section))):
        return dict(_MTP_OFF)

    # The CLI override sweeps `k` across the profiles that already run MTP and deliberately
    # does NOT switch it on elsewhere: a config's no-MTP baseline is the row every speedup
    # is measured against, and converting it would leave the run with nothing to compare to.
    tokens = override if override is not None else section.get("num_assistant_tokens")

    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 1:
        raise SystemExit(
            f"profile {name!r} {kind}.num_assistant_tokens must be an integer >= 1 "
            f"(got {tokens!r}); it is how many candidates the draft offers per "
            "verification pass, and openvino_genai rejects 0"
        )
    chunk = scheduler.get("max_num_batched_tokens")
    if isinstance(chunk, int) and not isinstance(chunk, bool) and chunk < tokens + 1:
        raise SystemExit(
            f"profile {name!r} has max_num_batched_tokens={chunk}, below the "
            f"{tokens + 1} tokens one speculative step submits (num_assistant_tokens={tokens} "
            "candidates plus the token being verified)"
        )
    device = section.get("device")
    if device is not None and (not isinstance(device, str) or not device.strip()):
        raise SystemExit(f"profile {name!r} {kind}.device must be a non-empty string")
    return {
        "enabled": True,
        "strategy": "mtp",
        "model": None,
        "num_assistant_tokens": tokens,
        "device": device.strip() if device else None,
    }


def _resolve_dflash(name: str, section, scheduler: dict, override) -> dict:
    """Normalize an external dFlash draft model into the speculative runtime shape.

    Validation is shared with `_resolve_mtp` (k >= 1, room in `max_num_batched_tokens`);
    on top of that the hybrid target's linear-attention pool must hold one committed row
    per sequence plus the 1 + k rows a verification window borrows.
    """
    if not isinstance(section, dict):
        raise SystemExit(
            f"profile {name!r} dflash must be a mapping of {{{', '.join(_DFLASH_KEYS)}}}"
        )
    unknown = [key for key in section if key not in _DFLASH_KEYS]
    if unknown:
        raise SystemExit(
            f"profile {name!r} dflash has unknown key(s): {', '.join(sorted(unknown))}. "
            f"A dflash section is {{{', '.join(_DFLASH_KEYS)}}}."
        )
    if not bool(section.get("enabled", bool(section))):
        return dict(_MTP_OFF)

    model = section.get("model")
    if not isinstance(model, str) or not model.strip():
        raise SystemExit(f"profile {name!r} dflash.model must be a non-empty path")
    resolved = _resolve_mtp(
        name,
        {
            "enabled": True,
            "num_assistant_tokens": section.get("num_assistant_tokens"),
            "device": section.get("device"),
        },
        scheduler,
        override,
        kind="dflash",
    )
    linear_blocks = scheduler.get("num_linear_attention_blocks")
    sequences = scheduler.get("max_num_seqs", _GENAI_DEFAULT_MAX_NUM_SEQS)
    required_blocks = sequences + resolved["num_assistant_tokens"] + 1
    if (
        isinstance(linear_blocks, int)
        and not isinstance(linear_blocks, bool)
        and linear_blocks < required_blocks
    ):
        raise SystemExit(
            f"profile {name!r} has num_linear_attention_blocks={linear_blocks}, below the "
            f"{required_blocks} rows required by max_num_seqs={sequences} and dflash "
            f"num_assistant_tokens={resolved['num_assistant_tokens']}"
        )
    resolved.update({"strategy": "dflash", "model": model.strip()})
    return resolved


def _resolve_profiles(raw_profiles, names_filter, ov_overrides, sched_overrides,
                      mtp_tokens=None) -> list:
    profiles = _namespace_to_dict(raw_profiles) or []
    if not isinstance(profiles, list) or not profiles:
        raise SystemExit("`profiles` must be a non-empty list of {name, ov, scheduler} entries")

    resolved = []
    for entry in profiles:
        if not isinstance(entry, dict):
            raise SystemExit("every profile must be a {name, ov, scheduler} mapping")
        name = str(entry.get("name") or "").strip()
        if not name:
            raise SystemExit("every profile needs a `name`")
        # Rejected rather than ignored: a misplaced key is a profile that does not measure what
        # its author meant, and the whole run is hours long. `cache_size` at profile level is
        # the specific mistake worth naming -- see `pipeline_mode`.
        unknown = [key for key in entry if key not in _PROFILE_KEYS]
        if unknown:
            hint = (
                " `cache_size` goes under `scheduler`, and putting it there selects the "
                "continuous-batching pipeline: it is a SchedulerConfig property, and the "
                "stateful pipeline has no KV pool to cap."
                if "cache_size" in unknown else ""
            )
            raise SystemExit(
                f"profile {name!r} has unknown key(s): {', '.join(sorted(unknown))}. "
                f"A profile is {{{', '.join(_PROFILE_KEYS)}}}.{hint}"
            )
        sections = {}
        for key in ("ov", "scheduler"):
            value = entry.get(key)
            if value is not None and not isinstance(value, dict):
                raise SystemExit(f"profile {name!r} ov and scheduler must be mappings")
            sections[key] = value or {}
        resolved_scheduler = _apply_overrides(sections["scheduler"], sched_overrides, coerce=True)
        if (
            "enable_prefix_caching" in resolved_scheduler
            and resolved_scheduler["enable_prefix_caching"] is not False
        ):
            raise SystemExit(
                f"profile {name!r} enables prefix caching; repeated-prompt benchmark "
                "profiles must set enable_prefix_caching=false"
            )
        # Warned about after the rejections above, so a profile that is about to fail
        # validation does not also get advice. Said out loud because it is not a tweak of the
        # same measurement: the override just moved this profile onto the other backend.
        if not sections["scheduler"] and resolved_scheduler:
            print(
                f"[warn] --scheduler-config moved profile {name!r} from the {PIPELINE_STATEFUL} "
                f"pipeline to {PIPELINE_PAGED} attention -- any SchedulerConfig selects "
                "continuous batching",
                flush=True,
            )
        if entry.get("mtp") is not None and entry.get("dflash") is not None:
            raise SystemExit(f"profile {name!r} cannot enable both mtp and dflash")
        speculative = (
            _resolve_dflash(name, entry.get("dflash"), resolved_scheduler, mtp_tokens)
            if entry.get("dflash") is not None
            else _resolve_mtp(name, entry.get("mtp"), resolved_scheduler, mtp_tokens)
        )
        resolved.append({
            "name": name,
            "ov": _apply_overrides(sections["ov"], ov_overrides, coerce=False),
            # `scheduler: {}` is meaningful -- it selects the stateful pipeline -- so an
            # empty mapping is preserved rather than treated as "unset".
            "scheduler": resolved_scheduler,
            "mtp": speculative,
        })

    names = [profile["name"] for profile in resolved]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise SystemExit(f"profile names must be unique; duplicated: {', '.join(duplicates)}")

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
        help="model-specific benchmark config (default: config_qwen3.6_35b.yaml)",
    )
    parser.add_argument("--models", nargs="+", help="override benchmark.models")
    parser.add_argument("--contexts", type=int, nargs="+", help="override benchmark.context_tokens")
    parser.add_argument("--profiles", nargs="+", help="run only these profiles, by name")
    parser.add_argument("--output-tokens", type=int, help="override benchmark.output_tokens")
    parser.add_argument("--warmup", type=int, help="override benchmark.warmup")
    parser.add_argument("--iterations", type=int, help="override benchmark.iterations")
    parser.add_argument(
        "--iteration-gap", type=float, metavar="SECONDS",
        help="override benchmark.iteration_gap_sec (untimed idle before each measured run)",
    )
    parser.add_argument(
        "--throughput-task", choices=THROUGHPUT_TASKS,
        help="override benchmark.throughput_task (the prompt the speed iterations decode)",
    )
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
        "--mtp-tokens", "--assistant-tokens", dest="mtp_tokens", type=int, metavar="K",
        help="override num_assistant_tokens on every profile that already enables MTP or "
             "dFlash; non-speculative baseline profiles stay the baseline",
    )
    parser.add_argument(
        "--accuracy", action="store_true",
        help="measure long-context accuracy (RULER retrieval + WWB-style generation fidelity) "
             "instead of throughput; requires an `accuracy` section in the config",
    )
    parser.add_argument(
        "--accuracy-suites", nargs="+", metavar="S",
        choices=(accuracy.SUITE_RETRIEVAL, accuracy.SUITE_GENERATION),
        help="run only these accuracy suites (overrides accuracy.suites)",
    )
    parser.add_argument(
        "--accuracy-tasks", nargs="+", metavar="T",
        help="run only these accuracy tasks, by name, across every suite being run. "
             f"Known tasks: {', '.join(tasks.KNOWN_TASKS)}",
    )
    parser.add_argument(
        "--accuracy-depths", type=float, nargs="+", metavar="D",
        help="override the depths of the depth-swept tasks "
             f"({', '.join(tasks.DEPTH_SWEPT_TASKS)}); fractions in [0,1]",
    )
    parser.add_argument(
        "--accuracy-samples", type=int, metavar="N",
        help="override accuracy.<suite>.samples (distinct probes measured per task/depth)",
    )
    parser.add_argument(
        "--list-profiles", action="store_true",
        help="print the resolved run matrix and exit -- no model, GPU or OpenVINO needed",
    )
    return parser.parse_args()


def _whole_number(value, message: str, minimum: int = 1) -> int:
    """A config integer, rejecting bool: a stray `true` is not 1 iteration."""
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise SystemExit(message)
    return value


_ACCURACY_KEYS = (
    "suites", "baseline_profile", "seed", "embedding_model",
    accuracy.SUITE_RETRIEVAL, accuracy.SUITE_GENERATION,
)
_SUITE_KEYS = ("tasks", "depths", "samples", "output_tokens", "options")

# Defaults per suite, applied when the section leaves a key out. Retrieval answers are short
# codes and word lists; generation answers are prose, and scoring prose at 24 tokens would
# measure truncation rather than fidelity.
#
# Deliberately small. A probe at 32K costs about a minute of prefill on this box, so the
# defaults are sized to be a run someone will actually wait for: 2 samples over 3 depths is
# 22 retrieval probes across the full eight tasks, not the 60 a five-depth three-sample sweep
# would be. Raise `samples` when a rate rather than an indication is needed -- a task scored
# over 2 probes can only report 0%, 50% or 100%.
_SUITE_DEFAULTS = {
    accuracy.SUITE_RETRIEVAL: {
        "tasks": list(tasks.RETRIEVAL_TASKS), "depths": [0.0, 0.5, 1.0],
        "samples": 2, "output_tokens": 64,
    },
    accuracy.SUITE_GENERATION: {
        "tasks": list(tasks.GENERATION_TASKS), "depths": [0.5],
        "samples": 2, "output_tokens": 256,
    },
}


def _resolve_accuracy(cfg, args, profiles) -> dict | None:
    """Resolve and validate the `accuracy` config section, or None when --accuracy is off.

    Off by default: an accuracy section may sit unused in a config, and only --accuracy turns
    it on. Every rule is checked before a model loads -- an unknown task name, a depth outside
    [0,1] or a `baseline_profile` naming a profile that is not in the run is a configuration
    error, not a thing to discover after an hour of probes, exactly like `_resolve_profiles`.

    The shape is one nested section per suite::

        accuracy:
          suites: [retrieval, generation]
          baseline_profile: paged_min
          retrieval: {tasks: [...], depths: [...], samples: 2, output_tokens: 64}
          generation: {tasks: [...], samples: 3, output_tokens: 256}

    Both suites are optional and each defaults to its full task list, so `suites: [retrieval]`
    with nothing else is a complete RULER run.
    """
    if not getattr(args, "accuracy", False):
        return None
    section = _namespace_to_dict(getattr(cfg, "accuracy", None))
    if not isinstance(section, dict) or not section:
        raise SystemExit(
            "--accuracy needs an `accuracy` section in the config. The minimum is "
            "`accuracy: {suites: [retrieval]}`; see "
            "docs/dev-guide/context-bench/context_bench_guide.md."
        )
    unknown = [k for k in section if k not in _ACCURACY_KEYS]
    if unknown:
        raise SystemExit(
            f"accuracy has unknown key(s): {', '.join(sorted(unknown))}. "
            f"An accuracy section is {{{', '.join(_ACCURACY_KEYS)}}}."
        )

    wanted = args.accuracy_suites if args.accuracy_suites else section.get(
        "suites", [accuracy.SUITE_RETRIEVAL, accuracy.SUITE_GENERATION]
    )
    if not isinstance(wanted, list) or not wanted or any(
        suite not in _SUITE_DEFAULTS for suite in wanted
    ):
        raise SystemExit(
            f"accuracy.suites must be a non-empty subset of "
            f"{{{', '.join(_SUITE_DEFAULTS)}}}"
        )

    resolved = {
        "suites": [s for s in _SUITE_DEFAULTS if s in wanted],  # canonical order
        "baseline_profile": _accuracy_baseline(section, profiles),
        "seed": _accuracy_seed(section),
        "embedding_model": section.get("embedding_model", wwb_adapter.DEFAULT_EMBEDDING_MODEL),
        accuracy.SUITE_RETRIEVAL: None,
        accuracy.SUITE_GENERATION: None,
    }
    for suite in list(resolved["suites"]):
        resolved[suite] = _resolve_suite(suite, _namespace_to_dict(section.get(suite)), args)
        if resolved[suite] is None:
            # `--accuracy-tasks` spans every suite, so narrowing to retrieval tasks leaves the
            # generation suite with nothing to run. Dropping it is what the user meant;
            # refusing the whole run and telling them to also pass --accuracy-suites is not.
            resolved["suites"].remove(suite)
            print(
                f"[info] --accuracy-tasks selects no {suite} task, so the {suite} suite is "
                "not being run",
                flush=True,
            )
    if not resolved["suites"]:
        raise SystemExit(
            f"--accuracy-tasks {' '.join(args.accuracy_tasks)} selects no task in any suite "
            "being run"
        )
    return resolved


def _accuracy_baseline(section: dict, profiles: list) -> str | None:
    baseline = section.get("baseline_profile")
    names = [p["name"] for p in profiles]
    if baseline is not None and baseline not in names:
        raise SystemExit(
            f"accuracy.baseline_profile {baseline!r} is not one of the profiles being run: "
            f"{', '.join(names)}"
        )
    return baseline


def _accuracy_seed(section: dict) -> int:
    seed = section.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise SystemExit("accuracy.seed must be an integer")
    return seed


def _resolve_suite(suite: str, section, args) -> dict:
    """One suite's settings, defaulted, validated and with the CLI overrides applied.

    CLI overrides (`--accuracy-tasks`, `--accuracy-depths`, `--accuracy-samples`) apply to
    every suite being run: they exist to cut a long sweep down to one task or one depth while
    debugging, and having to say which suite as well would make the common case wordy.
    """
    defaults = _SUITE_DEFAULTS[suite]
    section = section if isinstance(section, dict) else {}
    unknown = [k for k in section if k not in _SUITE_KEYS]
    if unknown:
        raise SystemExit(
            f"accuracy.{suite} has unknown key(s): {', '.join(sorted(unknown))}. "
            f"A suite section is {{{', '.join(_SUITE_KEYS)}}}."
        )

    names = args.accuracy_tasks if args.accuracy_tasks else section.get(
        "tasks", defaults["tasks"]
    )
    if not isinstance(names, list) or not names:
        raise SystemExit(f"accuracy.{suite}.tasks must be a non-empty list of task names")
    # A CLI --accuracy-tasks list spans both suites, so keep only the ones that belong here;
    # a name belonging to no suite at all is still an error.
    for task in names:
        if not isinstance(task, str) or not tasks.is_known(task):
            raise SystemExit(
                f"accuracy.{suite}.tasks has unknown task {task!r}. Known tasks are "
                f"retrieval: {', '.join(tasks.RETRIEVAL_TASKS)}; "
                f"generation: {', '.join(tasks.GENERATION_TASKS)}."
            )
    selected = [t for t in names if tasks.suite_of(t) == suite]
    if not selected:
        if args.accuracy_tasks:
            # A CLI filter that selects nothing here means "not this suite this time"; the
            # caller prunes it. A *config* that does the same is a mistake worth refusing.
            return None
        raise SystemExit(
            f"accuracy.{suite}.tasks selects no {suite} task; it lists only "
            f"{', '.join(names)}. Drop `{suite}` from accuracy.suites instead."
        )

    depths = args.accuracy_depths if args.accuracy_depths is not None else section.get(
        "depths", defaults["depths"]
    )
    if not isinstance(depths, list) or not depths or any(
        not isinstance(d, (int, float)) or isinstance(d, bool) or not 0.0 <= d <= 1.0
        for d in depths
    ):
        raise SystemExit(
            f"accuracy.{suite}.depths must be a non-empty list of fractions in [0.0, 1.0]"
        )

    return {
        "tasks": selected,
        "depths": sorted({float(d) for d in depths}),
        "samples": _whole_number(
            args.accuracy_samples if args.accuracy_samples is not None
            else section.get("samples", defaults["samples"]),
            f"accuracy.{suite}.samples must be a positive integer",
        ),
        "output_tokens": _whole_number(
            section.get("output_tokens", defaults["output_tokens"]),
            f"accuracy.{suite}.output_tokens must be a positive integer",
        ),
        "options": _resolve_task_options(suite, section.get("options")),
    }


def _resolve_task_options(suite: str, options) -> dict:
    """Per-task knobs (`num_distractors`, `chain_length`, ...), keyed by task name.

    Left as a free-form dict on purpose -- each task reads the keys it knows and ignores the
    rest -- but the *task names* are validated, because `options: {niah_multkey: ...}` with a
    typo would otherwise be silently dropped and the run would report the defaults as if they
    had been configured.
    """
    options = _namespace_to_dict(options)
    if options in (None, {}):
        return {}
    if not isinstance(options, dict):
        raise SystemExit(f"accuracy.{suite}.options must be a mapping of task name to settings")
    resolved = {}
    for task, values in options.items():
        if not tasks.is_known(task) or tasks.suite_of(task) != suite:
            raise SystemExit(
                f"accuracy.{suite}.options names {task!r}, which is not a {suite} task"
            )
        values = _namespace_to_dict(values)
        if not isinstance(values, dict):
            raise SystemExit(f"accuracy.{suite}.options.{task} must be a mapping")
        resolved[task] = values
    return resolved


def _load_settings(args) -> dict:
    cfg = load_config(args.config)
    model = getattr(cfg, "model", None)
    bench = getattr(cfg, "benchmark", None)
    if model is None or bench is None:
        raise SystemExit(
            f"{args.config} must define both `model` and `benchmark`. "
            "See docs/dev-guide/context-bench/context_bench_guide.md."
        )

    configured_contexts = args.contexts if args.contexts is not None else bench.context_tokens
    if not isinstance(configured_contexts, list) or not configured_contexts or any(
        not isinstance(context, int) or isinstance(context, bool) or context <= 0
        for context in configured_contexts
    ):
        raise SystemExit("benchmark.context_tokens must contain only positive integers")
    contexts = sorted(set(configured_contexts))

    max_ram_pct = getattr(bench, "max_system_memory_pct", 80)
    if (
        not isinstance(max_ram_pct, (int, float))
        or isinstance(max_ram_pct, bool)
        or not 0 < max_ram_pct <= 100
    ):
        raise SystemExit("benchmark.max_system_memory_pct must be in (0, 100]")

    iterations = _whole_number(
        args.iterations if args.iterations is not None else bench.iterations,
        "benchmark.iterations must be a positive integer",
    )
    # At least 2 output tokens: TPOT divides by output_size - 1, so a 1-token generation
    # has no decode phase to measure at all.
    output_tokens = _whole_number(
        args.output_tokens if args.output_tokens is not None else bench.output_tokens,
        "benchmark.output_tokens must be at least 2 to measure TPOT", minimum=2,
    )
    warmup = _whole_number(
        args.warmup if args.warmup is not None else bench.warmup,
        "benchmark.warmup must be a non-negative integer", minimum=0,
    )

    # Untimed idle before each measured generation. A shared-power iGPU boosts for the first
    # few seconds of load and then settles lower: on Qwen3.6-35B-A3B at 8K, back-to-back
    # generations decode at ~28-31 ms/token while every generation after 30 s idle runs at
    # 24.6 ms with a 3.0 s instead of ~4 s TTFT. 0 measures sustained throughput; a gap
    # measures what an occasional request (the classroom case) sees.
    iteration_gap_sec = (
        args.iteration_gap if args.iteration_gap is not None
        else getattr(bench, "iteration_gap_sec", 0)
    )
    if (
        not isinstance(iteration_gap_sec, (int, float)) or isinstance(iteration_gap_sec, bool)
        or not math.isfinite(iteration_gap_sec) or iteration_gap_sec < 0
    ):
        raise SystemExit("benchmark.iteration_gap_sec must be a non-negative number of seconds")

    throughput_task = (
        args.throughput_task if getattr(args, "throughput_task", None) is not None
        else getattr(bench, "throughput_task", THROUGHPUT_TASKS[0])
    )
    if throughput_task not in THROUGHPUT_TASKS:
        raise SystemExit(
            f"benchmark.throughput_task must be one of {', '.join(THROUGHPUT_TASKS)} "
            f"(got {throughput_task!r})"
        )

    models = args.models if args.models is not None else bench.models
    if not isinstance(models, list) or not models or any(
        not isinstance(name, str) or not name.strip() for name in models
    ):
        raise SystemExit("benchmark.models must contain at least one non-empty model name")

    timeout_sec = bench.timeout_sec
    if not isinstance(timeout_sec, (int, float)) or isinstance(timeout_sec, bool) or timeout_sec <= 0:
        raise SystemExit("benchmark.timeout_sec must be greater than 0")

    try:
        gpu_memory_budget_gb = float(bench.gpu_memory_budget_gb)
    except (AttributeError, TypeError, ValueError) as exc:
        raise SystemExit("benchmark.gpu_memory_budget_gb must be a finite positive number") from exc
    if not math.isfinite(gpu_memory_budget_gb) or gpu_memory_budget_gb <= 0:
        raise SystemExit("benchmark.gpu_memory_budget_gb must be a finite positive number")

    resolved_profiles = _resolve_profiles(
        getattr(cfg, "profiles", None),
        args.profiles,
        _parse_overrides(args.pipeline_config),
        _parse_overrides(args.scheduler_config),
        args.mtp_tokens,
    )

    return {
        "provider": model.provider,
        "models_base_path": model.models_base_path,
        # Explicit per-model IR directories, for exports whose folder name does not follow
        # `<name>_<weight_format>` -- the published OpenVINO IRs are named e.g.
        # `Qwen3.8-27B-int4-ov`. Naming the path beats guessing at a second convention.
        "model_dirs": _namespace_to_dict(getattr(model, "model_dirs", None)) or {},
        "device": args.device or model.device,
        "weight_format": args.weight_format or model.weight_format,
        "models": models,
        "context_tokens": contexts,
        "output_tokens": output_tokens,
        "warmup": warmup,
        "iterations": iterations,
        "iteration_gap_sec": float(iteration_gap_sec),
        "throughput_task": throughput_task,
        "timeout_sec": timeout_sec,
        "max_system_memory_pct": max_ram_pct,
        "gpu_memory_budget_gb": gpu_memory_budget_gb,
        "cache_dir": getattr(bench, "cache_dir", None),
        "output_dir": args.output_dir or bench.output_dir,
        "profiles": resolved_profiles,
        # None unless --accuracy is passed; selects the Needle-in-a-Haystack path per case.
        "accuracy": _resolve_accuracy(cfg, args, resolved_profiles),
    }


def _model_ir_dir(base: str, provider: str, model_name: str, weight_format: str,
                  model_dirs: dict | None = None) -> str:
    """Where this candidate's IR lives.

    Derived from the model name and weight format, mirroring
    utils/ensure_model.py::get_model_path -- unless the config names the directory
    explicitly under `model.model_dirs`, which is what a downloaded IR with its own naming
    needs. An explicit path is taken relative to the working directory (smart-classroom/),
    like `models_base_path` itself.
    """
    override = (model_dirs or {}).get(model_name)
    if override:
        return os.path.normpath(str(override))
    return os.path.join(base, provider, f"{model_name.replace('/', '_')}_{weight_format}")


def _ir_ready(model_dir: str) -> bool:
    if not os.path.isdir(model_dir):
        return False
    return (
        any(
            os.path.isfile(os.path.join(model_dir, name))
            for name in ("openvino_model.xml", "openvino_language_model.xml")
        )
        and os.path.isfile(os.path.join(model_dir, "openvino_tokenizer.xml"))
        and os.path.isfile(os.path.join(model_dir, "openvino_detokenizer.xml"))
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
    mtp: dict | None = None,
    on_iteration=None,
    probes: list | None = None,
    on_probe=None,
    iteration_gap_sec: float = 0.0,
    throughput_task: dict | None = None,
    sample_interval: float = 0.5,
    poll_interval: float = 0.25,
    drain_timeout: float = 5.0,
) -> dict:
    """Run one case to completion, OOM, native abort, or a progress timeout.

    Nothing stops the child on a memory threshold: the point is to find where this box
    actually breaks. Subprocess isolation is what makes that safe -- a native GPU abort
    near shared-memory exhaustion kills only the child, and the parent's sampler has
    already recorded the high-water mark that explains it.

    Milestones are tracked as they arrive rather than read off the final result, because
    a child killed by the timeout or by a native abort never posts one: "hung in prefill"
    and "hung in decode" are different findings about the same context length.
    """
    baseline = _read_mem()
    sampler = _MemorySampler(interval=sample_interval, baseline=baseline)
    sampler.start()

    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=trial_runner.run_case,
        args=(
            model_dir, device, context_tokens, output_tokens, warmup, iterations,
            result_queue, ov_config, scheduler_config, mtp, probes, iteration_gap_sec,
            throughput_task,
        ),
    )
    process.start()

    loaded_mem = None
    result = None
    state = {
        "load_ok": False,
        "stage_reached": trial_runner.STAGE_START,
        # Unknown until a prompt milestone arrives from the child.
        "prompt_tokens": None,
        "gpu_budget_driver_gb": None,
        "iterations": [],
        # Only the accuracy path fills this; kept present so both modes read one shape.
        "probes": [],
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
        elif event == "prefilled":
            state["stage_reached"] = trial_runner.STAGE_PREFILLED
        elif event == "iteration":
            state["stage_reached"] = trial_runner.STAGE_DECODED
            record = {k: v for k, v in msg.items() if k != "event"}
            record.update(sampler.window())
            state["iterations"].append(record)
            if on_iteration:
                on_iteration(record)
        elif event == "probe":
            # An accuracy probe: same handling as `iteration` (memory window merged in) plus
            # the planted depth/sample and the answer text, which the parent scores.
            state["stage_reached"] = trial_runner.STAGE_DECODED
            record = {k: v for k, v in msg.items() if k != "event"}
            record.update(sampler.window())
            state["probes"].append(record)
            if on_probe:
                on_probe(record)
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
            finished = _consume(msg, child_alive=True)
            deadline = time.monotonic() + timeout_sec
            if finished:
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
        def _mem_text(value):
            return f"{value:.1f}" if value is not None else "unavailable"

        print(
            f"  [warn] memory has not returned to baseline after {_MEMORY_SETTLE_TIMEOUT_SEC:g}s "
            f"(RAM {_mem_text(current['ram_gb'])} vs {_mem_text(baseline['ram_gb'])} GB, GPU "
            f"{_mem_text(current['gpu_gb'])} vs {_mem_text(baseline['gpu_gb'])} GB); the next case's "
            "weights/KV split may be charged with the leftover",
            flush=True,
        )

    mem = {
        "peak_ram_gb": _round_optional(sampler.peak_ram),
        "peak_ram_pct": _round_optional(sampler.peak_ram_pct, 1),
        "min_available_ram_gb": _round_optional(sampler.min_available_ram),
        "peak_gpu_gb": _round_optional(sampler.peak_gpu),
        "post_load_peak_ram_gb": _delta(sampler.peak_ram, loaded_mem["ram_gb"]) if loaded_mem else None,
        "post_load_peak_gpu_gb": _delta(sampler.peak_gpu, loaded_mem["gpu_gb"]) if loaded_mem else None,
    }

    if result is not None:
        result.pop("event", None)
        # The parent's per-iteration records carry the memory windows the child cannot see.
        result["iterations"] = state["iterations"] or result.get("iterations") or []
        result["probes"] = state["probes"] or result.get("probes") or []
        # Renamed on the way out: the driver's own figure must not sit under a name that
        # could be mistaken for the configured budget the tool actually enforces.
        from_done = result.pop("gpu_budget_gb", None)
        result["gpu_budget_driver_gb"] = state["gpu_budget_driver_gb"] or from_done
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
        "probes": state["probes"],
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
        measured = metrics.measured(result.get("iterations")) or result.get("probes")
        return "ok" if measured else "no_output"
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


def _preflight_cache_sizes(settings: dict, static: dict, model_name: str) -> None:
    """Reject a fixed `cache_size` too small for the cache it has to hold, before loading.

    `validate_fixed_cache_size` is enforced per case inside `_run_case`, which on its own means
    a profile that could never have run aborts the whole run *after* the earlier cases have
    already spent hours -- the worst of both fail-fast and keep-going. Checking the model's
    whole (profile x context) matrix here costs nothing: the estimate is pure arithmetic over
    `config.json`, and the first case of this model has not started yet.

    Only fixed values are checked; `cache_size: auto` derives a value that cannot be too small
    by construction and warns for itself when the budget leaves no room.
    """
    for profile in settings["profiles"]:
        scheduler = profile["scheduler"]
        cache_size = scheduler.get("cache_size")
        if cache_size is None or str(cache_size).lower() == "auto":
            continue
        for context_tokens in settings["context_tokens"]:
            kv_gb = expected_kv_gb(
                static["model_config"], static["fixed_state_bytes"], profile["ov"],
                context_tokens,
                sequences=scheduler.get("max_num_seqs", _GENAI_DEFAULT_MAX_NUM_SEQS),
                mtp_layers=(
                    static.get("mtp_layers", 0)
                    if profile.get("mtp", _MTP_OFF)["enabled"] else 0
                ),
            )
            try:
                validate_fixed_cache_size(cache_size, kv_gb)
            except ValueError as exc:
                raise SystemExit(
                    f"{model_name}, profile {profile['name']!r} at {context_tokens:,} "
                    f"tokens: {exc}"
                ) from exc


def _apply_memory_status(case: dict) -> dict:
    """Demote a completed measurement that exceeded a configured memory limit."""
    if case.get("status") != "ok":
        return case
    if case.get("memory_measurement_error"):
        case["status"] = "measurement_error"
    elif case.get("gpu_budget_exceeded"):
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

    # Scheduler first, because the estimate depends on it: on the paged path `max_num_seqs`
    # decides how many linear-attention reservations come out of the pool being sized.
    scheduler_config = dict(profile["scheduler"])
    mtp = dict(profile.get("mtp") or _MTP_OFF)
    kv_gb = expected_kv_gb(
        static["model_config"], static["fixed_state_bytes"], ov_config, context_tokens,
        sequences=(
            scheduler_config.get("max_num_seqs", _GENAI_DEFAULT_MAX_NUM_SEQS)
            if scheduler_config else 1
        ),
        mtp_layers=static.get("mtp_layers", 0) if mtp["enabled"] else 0,
    )

    validate_fixed_cache_size(scheduler_config.get("cache_size"), kv_gb)
    if str(scheduler_config.get("cache_size")).lower() == "auto":
        auto = auto_cache_size_gb(
            kv_gb,
            static["weight_disk_gb"],
            settings["gpu_memory_budget_gb"],
            pool_factor=_MTP_CACHE_POOL_FACTOR if mtp["enabled"] else 1.0,
        )
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

    # After the `auto` resolution, not before: dropping an underivable `cache_size` can empty
    # a one-key scheduler, and that really does hand the case to the stateful pipeline. The
    # reported mode has to be the one the child will run.
    mode = pipeline_mode(scheduler_config)
    draft_precision = _draft_weight_precision(model_dir, mtp)

    accuracy_cfg = settings.get("accuracy")
    # Built before the banner so the probe count printed is the number that will actually run.
    specs = accuracy.build_probe_specs(accuracy_cfg, context_tokens) if accuracy_cfg else []
    if accuracy_cfg:
        work = (
            f"{settings['warmup']} warmup + {len(specs)} accuracy probe(s) across "
            f"{', '.join(accuracy_cfg['suites'])}"
        )
    else:
        work = f"{settings['warmup']} warmup + {settings['iterations']} iterations"
    print(
        f"\n[{model_name} | {profile['name']} | {context_tokens:,} tok] {work}, timeout "
        f"{settings['timeout_sec']:g}s",
        flush=True,
    )
    print(f"  ov: {format_config(ov_config)}", flush=True)
    print(f"  pipeline: {mode}", flush=True)
    print(f"  weights: {static.get('weight_precision') or '--'} ({model_dir})", flush=True)
    print(f"  speculative: {format_mtp(mtp, settings['device'])}", flush=True)
    if mtp["enabled"]:
        print(f"  draft weights: {draft_precision or '--'}", flush=True)
    if mode == PIPELINE_PAGED:
        print(f"  scheduler: {format_config(scheduler_config)}", flush=True)
        if "cache_size" not in scheduler_config:
            print(
                "  note: cache_size is unset; OpenVINO runtime manages the KV pool. Peak GPU "
                "memory can look flatter across contexts when the pool is pre-allocated, and a "
                "160K run with an unset pool once went past 372s with no first token.",
                flush=True,
            )
    else:
        print(
            "  note: no scheduler_config, so prefill is one SDPA pass over the whole context "
            "and the KV cache is model state that grows with it. `cache_size` does not exist on "
            "this path -- KV_CACHE_PRECISION is the only lever on cache bytes.",
            flush=True,
        )

    # Accuracy mode: ship the built probe specs to the child, which only renders and generates
    # them. `probes=None` (the default) leaves the throughput path untouched.
    probes = [
        {
            "suite": s.suite, "task": s.task, "sample": s.sample, "depth": s.depth,
            "inserts": s.inserts, "question": s.question, "filler": s.filler,
            "output_tokens": s.output_tokens,
        }
        for s in specs
    ] or None
    # The child's fallback decode length; each probe carries its own, since a retrieval answer
    # is a six-character code and a generation answer is a paragraph.
    output_tokens = max((s.output_tokens for s in specs), default=settings["output_tokens"])

    def _echo(record):
        print("  " + metrics.format_iteration(record), flush=True)

    def _echo_probe(record):
        answer = (record.get("prediction_text") or "").replace("\n", " ").strip()
        if len(answer) > 60:
            answer = answer[:57] + "..."
        depth = record.get("depth")
        where = f" d={depth:.2f}" if isinstance(depth, (int, float)) else ""
        print(
            f"  probe {record.get('task')}{where} #{record.get('sample')}: "
            f"{metrics.format_iteration(record)} | answer: {answer!r}",
            flush=True,
        )

    try:
        result = _run_case_subprocess(
            model_dir, device, context_tokens, output_tokens,
            settings["warmup"], settings["iterations"], settings["timeout_sec"],
            ov_config, scheduler_config, mtp, on_iteration=_echo,
            probes=probes, on_probe=_echo_probe,
            iteration_gap_sec=settings.get("iteration_gap_sec", 0.0),
            throughput_task=throughput_task_prompt(
                settings.get("throughput_task"), model_name
            ),
        )
    except Exception as exc:  # noqa: BLE001 - a case the orchestrator could not carry out
        # Spawning a fresh interpreter that re-imports the OpenVINO stack is itself work the
        # box can fail at, and letting that unwind would discard every measurement taken so far.
        traceback.print_exc()
        result = {"error": f"orchestrator:error:{exc}", "iterations": [], "probes": [],
                  "load_ok": False}

    peak_gpu = result.get("peak_gpu_gb")
    budget = settings["gpu_memory_budget_gb"]
    ram_pct = result.get("peak_ram_pct")
    # Aggregated first because `mean_gpu_pct_of_budget` is derived from a median the
    # aggregation produces, and the case row below is assembled in one expression. In accuracy
    # mode there are no timing iterations; the probes carry the same per-generate record shape,
    # so latency and memory aggregate from them instead.
    aggregated = metrics.aggregate(
        result.get("probes") if accuracy_cfg else result.get("iterations")
    )
    if accuracy_cfg:
        # `output_consistent` asks whether repeated greedy generations of the SAME prompt
        # agreed -- the determinism check the MTP comparison rests on. Every accuracy probe
        # is a different prompt, so their hashes differ by construction and the check reads
        # False for every run, warning that a perfectly deterministic box is nondeterministic.
        # Nothing measured it here, so report nothing rather than a false negative.
        aggregated["output_consistent"] = None
        aggregated["output_sha256"] = None
    mean_gpu = aggregated.get("mean_gpu_gb")

    case = {
        "model": model_name,
        "profile": profile["name"],
        "context_tokens": context_tokens,
        "device": device,
        "weight_format": settings["weight_format"],
        # Detected from the IR, not the config label: see ir_weight_precision.
        "weight_precision": static.get("weight_precision"),
        "draft_weight_precision": draft_precision,
        "throughput_task": None if accuracy_cfg else settings.get("throughput_task"),
        "pipeline_mode": mode,
        "status": _status(result),
        "error": result.get("error"),
        "stage_reached": result.get("stage_reached"),
        "kv_cache_precision": ov_config.get("KV_CACHE_PRECISION", "(plugin default)"),
        "cache_size_gb": scheduler_config.get("cache_size"),
        # From the profile, not from a measurement: a case that failed before generating
        # still has to say which configuration failed.
        "mtp": mtp["enabled"],
        "speculative_strategy": mtp["strategy"],
        "draft_model": mtp["model"],
        "num_assistant_tokens": mtp["num_assistant_tokens"],
        "mtp_device": (mtp["device"] or device) if mtp["enabled"] else None,
        "ov_config": format_config(ov_config),
        # `format_config({})` already renders "(none)". Spelling it "(stateful)" here made the
        # CSV disagree with --list-profiles about the same empty scheduler, and the
        # `pipeline_mode` column now carries that meaning anyway.
        "scheduler_config": format_config(scheduler_config),
        "load_ok": result.get("load_ok"),
        "load_time_s": result.get("load_time_s"),
        "prompt_tokens": result.get("prompt_tokens"),
        "weight_disk_gb": static["weight_disk_gb"],
        "expected_kv_gb": kv_gb,
        "gpu_budget_gb": budget,
        "gpu_budget_driver_gb": result.get("gpu_budget_driver_gb"),
        "peak_gpu_pct_of_budget": (
            round(peak_gpu / budget * 100, 1) if peak_gpu is not None else None
        ),
        "mean_gpu_pct_of_budget": (
            round(mean_gpu / budget * 100, 1) if mean_gpu is not None else None
        ),
        "memory_measurement_error": ram_pct is None or peak_gpu is None,
        "system_memory_limit_exceeded": bool(
            ram_pct is not None and ram_pct > settings["max_system_memory_pct"]
        ),
        # The peak is what decides usability: a configuration whose transient prefill
        # workspace exceeds the budget is unusable even if its mean sits comfortably below.
        "gpu_budget_exceeded": bool(peak_gpu is not None and peak_gpu > budget),
        **{k: result.get(k) for k in _CASE_MEMORY_FIELDS},
        **aggregated,
    }
    case["_iterations"] = result.get("iterations") or []
    if accuracy_cfg:
        # Keyed by suite and nothing else -- the baseline profile's name belongs to the run,
        # not to a case, and lives in settings["accuracy"]. Mixing a
        # scalar in here made the delta pass walk a string as though it were a suite.
        scored, summary = _score_case_accuracy(result.get("probes") or [], specs)
        case["accuracy"] = summary
        case["_probes"] = scored
        case["_model_dir"] = model_dir

    # A measurement taken past the memory ceiling is a real measurement of an unusable
    # configuration, so it keeps its numbers and is demoted rather than deleted.
    return _apply_memory_status(case)


def _depth_label(depth) -> str:
    """A stable string key for a depth fraction, shared by the score map, the per-depth
    aggregate and the report matrix so they cannot drift. Tasks that place their
    needles by construction have no single depth and key as "-"; see `tasks.ProbeSpec`."""
    return "-" if depth is None else f"{float(depth):.2f}"


def _score_case_accuracy(probes: list, specs: list) -> tuple:
    """Score each probe's answer against what its spec planted, and aggregate per suite.

    Returns ``(scored, summary)``. `scored` are the probe records with the ground truth, the
    prediction and every score merged in -- the rows the probe-detail section writes. `summary` is the
    nested ``case["accuracy"]``: per-suite overall rates, the per-task and per-depth
    breakdowns, and the probe count.

    Only what a case can score *alone* is done here. The retrieval suite has its own ground
    truth, so it is fully scored; the generation suite gets its grounded `fact_coverage` now
    and its fidelity columns later, in `_fill_generation_fidelity`, because those compare
    against the baseline profile's answers and that case may not have run yet.

    Scored in the parent because it holds the ground truth; the child only returned text.
    """
    by_coord = {s.coordinate: s for s in specs}
    scored, by_suite = [], {}
    for probe in probes:
        coordinate = (probe.get("task"), _depth_label(probe.get("depth")), probe.get("sample"))
        spec = by_coord.get(coordinate)
        if spec is None:
            continue
        prediction = probe.get("prediction_text", "")
        score = (
            accuracy.score_grounding(prediction, spec)
            if spec.suite == accuracy.SUITE_GENERATION
            else accuracy.score_retrieval(prediction, spec)
        )
        row = {
            **probe,
            "suite": spec.suite,
            "task": spec.task,
            "truth": " | ".join(spec.truths),
            "distractors": " | ".join(spec.distractors),
            "prediction": prediction,
            **score,
        }
        scored.append(row)
        by_suite.setdefault(spec.suite, []).append(row)

    return scored, {suite: _summarize_suite(suite, rows) for suite, rows in by_suite.items()}


def _summarize_suite(suite: str, rows: list) -> dict:
    """One suite's overall / per-task / per-depth aggregates.

    The per-depth breakdown covers only the depth-swept tasks -- it is the NIAH matrix, and
    folding in tasks that spread their needles by construction would put a column header on a
    number that has no depth. The per-task breakdown covers everything, and is the table that
    shows a profile holding `niah_single` while it loses `vt`.
    """
    fields = accuracy.metrics_for(suite)
    summary = accuracy.aggregate(rows, fields)
    summary["per_task"] = accuracy.group_aggregate(rows, fields, lambda r: r["task"])
    swept = [r for r in rows if tasks.is_depth_swept(r["task"])]
    summary["per_depth"] = accuracy.group_aggregate(
        swept, fields, lambda r: _depth_label(r.get("depth"))
    )
    return summary


def _format_case(case: dict) -> str:
    if case["status"] not in _USABLE_STATUSES:
        detail = f" ({case['error']})" if case.get("error") else ""
        stage = case.get("stage_reached")
        where = (
            f" [failed in {trial_runner.failing_stage(stage)}]"
            if stage and stage != trial_runner.STAGE_DECODED else ""
        )
        return f"  => {case['status'].upper()}{where}{detail}"
    if case.get("accuracy"):
        parts = []
        retrieval = case["accuracy"].get(accuracy.SUITE_RETRIEVAL)
        if retrieval:
            parts.append(
                f"retrieval recall {_acc_pct(retrieval.get('recall_rate'))} "
                f"(EM {_acc_pct(retrieval.get('exact_match_rate'))}, "
                f"F1 {_acc_pct(retrieval.get('token_f1'))}) over "
                f"{retrieval.get('probe_count')} probes"
            )
        generation = case["accuracy"].get(accuracy.SUITE_GENERATION)
        if generation:
            # Fidelity columns are filled at report time, so this line shows what the case
            # could score on its own plus whatever is already in hand.
            parts.append(
                f"generation ROUGE-L {_acc_pct(generation.get('rouge_l'))}, "
                f"coverage {_acc_pct(generation.get('fact_coverage'))} over "
                f"{generation.get('probe_count')} probes"
            )
        return (
            f"  => {case['status'].upper()} | " + " | ".join(parts or ["no probes scored"])
            + f" | TPOT {_fmt(case.get('other_tokens_avg_latency'))} ms/token "
            f"| TTFT {_ttft_seconds(case)}s"
        )
    line = (
        f"  => {case['status'].upper()} | TPOT {_fmt(case.get('other_tokens_avg_latency'))} ms/token "
        f"{_mtp_cell(case, prefix='| Speculative ')}"
        f"| TTFT {_ttft_seconds(case)}s "
        f"| decode {_fmt(case.get('decode_throughput'), '{:.2f}')} tok/s "
        f"| e2e {_fmt(case.get('e2e_throughput'))} tok/s "
        f"| RAM peak {_fmt(case.get('peak_ram_gb'))} GB ({_fmt(case.get('peak_ram_pct'))}%) "
        f"/ mean {_fmt(case.get('mean_ram_gb'))} GB "
        f"| GPU peak {_fmt(case.get('peak_gpu_gb'))} GB "
        f"({_fmt(case.get('peak_gpu_pct_of_budget'))}% of budget) "
        f"/ mean {_fmt(case.get('mean_gpu_gb'))} GB "
        f"({_fmt(case.get('mean_gpu_pct_of_budget'))}%)"
    )
    hint = _mtp_tuning_hint(case)
    return line + (f"\n{hint}" if hint else "")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt(value, spec="{:.1f}") -> str:
    return spec.format(value) if isinstance(value, (int, float)) else "--"


def _ttft_seconds(case: dict) -> str:
    """TTFT is recorded in milliseconds (llm_bench's unit) but read in seconds."""
    ttft = case.get("first_token_latency")
    return _fmt(ttft / 1000) if isinstance(ttft, (int, float)) else "--"


def _output_tokens_cell(case: dict) -> str:
    smallest = case.get("output_size_min", case.get("output_size"))
    largest = case.get("output_size_max", case.get("output_size"))
    if smallest != largest:
        return f"{_fmt(smallest, '{:g}')}-{_fmt(largest, '{:g}')}"
    return _fmt(largest, "{:g}")


def _mtp_cell(case: dict, prefix: str = "", empty: str = "") -> str:
    """The speculative-decoding column: what was configured and what it yielded.

    `k` alone would not say whether it worked -- a candidate count is a request, and the
    finding is how many of those candidates survived verification. A case with MTP off
    still reports its measured yield when there is one, because reading exactly 1.00
    tok/step there is what confirms the draft head is not quietly running.
    """
    if not case.get("mtp"):
        return empty
    strategy = metrics.strategy_label(case)
    parts = [strategy, f"k={case.get('num_assistant_tokens')}"]
    # Mode and group only; the full record is in the case banner. A draft quantized more
    # coarsely than its target agrees with it less often, so it belongs next to acceptance.
    if case.get("draft_weight_precision"):
        parts.append("draft " + " ".join(case["draft_weight_precision"].split()[:2]))
    tokens_per_step = case.get("tokens_per_step")
    if isinstance(tokens_per_step, (int, float)):
        parts.append(f"{tokens_per_step:.2f} tok/step")
    acceptance = case.get("mtp_acceptance_rate")
    if isinstance(acceptance, (int, float)):
        qualifier = " estimated" if case.get("mtp_acceptance_estimated") else ""
        parts.append(f"{acceptance * 100:.0f}% accepted{qualifier}")
    # What proposing those candidates cost, so the cell that says "62% accepted" also says
    # whether the draft head is cheap enough for that acceptance to be a net decode win.
    ratio = case.get("mtp_draft_to_main_ratio")
    if isinstance(ratio, (int, float)):
        parts.append(f"draft/main {ratio:.2f}x")
    return prefix + ", ".join(parts) + " "


def _mtp_tuning_hint(case: dict) -> str | None:
    """Point at the next k to measure -- in *either* direction.

    The mistake this guards against is treating acceptance as the thing to maximize. It is
    not: the objective is tokens committed per verification pass, ``1 + k * acceptance(k)``,
    and acceptance(k) falls as k rises, so the two pull against each other. A profile whose
    candidates are mostly rejected should lower k; a profile whose candidates almost all land
    is leaving yield unclaimed and should raise k. Only the flat middle -- accepting well but
    not lavishly -- needs no next measurement.
    """
    acceptance = case.get("mtp_acceptance_rate")
    tokens = case.get("num_assistant_tokens")
    if (
        not case.get("mtp")
        or not isinstance(acceptance, (int, float))
        or not isinstance(tokens, int)
    ):
        return None
    dflash = case.get("speculative_strategy") == "dflash"
    if acceptance < _LOW_MTP_ACCEPTANCE_RATE and tokens > 1:
        next_tokens = max(1, math.ceil(tokens / 2))
        if dflash:
            return (
                f"  hint: dFlash accepted {acceptance * 100:.0f}% of its {tokens} candidates; "
                f"compare k={next_tokens}. The draft proposes its block in one parallel pass, "
                "so a larger k costs little to draft, but the target still verifies all k + 1 "
                "tokens every pass -- rank by TPOT against the matched baseline, not by "
                "acceptance."
            )
        return (
            f"  hint: only {acceptance * 100:.0f}% of MTP candidates were accepted; compare "
            f"k={next_tokens} -- fewer candidates per pass usually raises acceptance (for "
            "Qwen3.8-27B at 8K, k=2 is the higher-acceptance balanced profile and k=3 is the "
            "minimum-TPOT profile)."
        )
    ceiling = _DFLASH_TOKEN_SUGGESTION_CEILING if dflash else _MTP_TOKEN_SUGGESTION_CEILING
    if acceptance >= _HIGH_MTP_ACCEPTANCE_RATE and tokens < ceiling:
        tokens_per_step = case.get("tokens_per_step")
        yield_now = (
            f" for {tokens_per_step:.2f} tok/step"
            if isinstance(tokens_per_step, (int, float)) else ""
        )
        if dflash:
            return (
                f"  hint: dFlash accepted {acceptance * 100:.0f}% of candidates{yield_now}; "
                f"compare k={tokens + 1} and rank by TPOT against the matched baseline."
            )
        return (
            f"  hint: {acceptance * 100:.0f}% of MTP candidates were accepted{yield_now}; the "
            f"draft head is agreeing often, so compare k={tokens + 1} -- a larger candidate "
            "batch can commit more tokens per pass even as acceptance eases. Rank by tokens/"
            "step (yield), not acceptance: on this box k=4 (50% accepted, 2.90 tok/step) beat "
            "k=2 (74%, 2.44) on TPOT."
        )
    return None


def _best_summary(case: dict) -> str:
    """The winning profile's one-liner."""
    return (
        f"TPOT {_fmt(case.get('other_tokens_avg_latency'))} ms/token "
        f"({_fmt(case.get('decode_throughput'), '{:.2f}')} decode tok/s), "
        f"TTFT {_ttft_seconds(case)}s"
    )


def _leaderboard(cases: list) -> list:
    """Usable profiles ranked by TPOT first, then TTFT as the tie-breaker."""
    return sorted(
        (
            c for c in cases
            if c["status"] == "ok" and c.get("other_tokens_avg_latency") is not None
        ),
        key=lambda c: (
            c["other_tokens_avg_latency"],
            c.get("first_token_latency")
            if isinstance(c.get("first_token_latency"), (int, float))
            else float("inf"),
        ),
    )


def _ttft_leaderboard(cases: list) -> list:
    """The same usable cases, ranked by TTFT alone.

    Reported next to the TPOT ranking rather than replacing it. TPOT stays the primary
    service metric, but at 160K TTFT is ~96% of the wall clock, and the stateful-vs-paged
    question is decided almost entirely in prefill: a leaderboard that only ranks decode
    would hide the difference the run exists to measure.
    """
    return sorted(
        (
            c for c in cases
            if c["status"] == "ok"
            and isinstance(c.get("first_token_latency"), (int, float))
        ),
        key=lambda c: c["first_token_latency"],
    )


def _suite_summary(case: dict, suite: str) -> dict:
    return (case.get("accuracy") or {}).get(suite) or {}


def _accuracy_leaderboard(cases: list, suite: str = accuracy.SUITE_RETRIEVAL) -> list:
    """Usable accuracy cases for one suite, best first.

    The accuracy analog of `_leaderboard`: where the throughput ranking answers "which config
    is fastest", this answers "which config is still right" -- the question the whole mode
    exists for, and the one a quantized or MTP profile can lose while winning on TPOT.

    Retrieval ranks by recall, then exact match, then token-F1. Recall is primary because it
    is RULER's own scorer and is robust to a model that answers "the code is X" rather than
    "X". Generation ranks by embedding similarity to the baseline, then ROUGE-L, then chrF --
    semantic first, because a rewording that preserves the content is not the regression this
    suite is hunting, and the lexical scores are there to catch the case where similarity is
    the metric being fooled.
    """
    keys = (
        ("similarity", "rouge_l", "chrf") if suite == accuracy.SUITE_GENERATION
        else ("recall_rate", "exact_match_rate", "token_f1")
    )
    # Ranked on *any* of the keys, not just the first: the generation suite's primary key is
    # the embedding similarity, and that column is empty whenever who_what_benchmark is not
    # installed. Requiring it would leave the whole leaderboard -- and the "largest drift"
    # finding the suite exists to surface -- silently empty on the common install.
    ranked = [
        c for c in cases
        if c.get("status") == "ok"
        and any(_suite_summary(c, suite).get(k) is not None for k in keys)
    ]
    return sorted(
        ranked,
        key=lambda c: tuple(-(_suite_summary(c, suite).get(k) or 0.0) for k in keys),
    )


def _baseline_case(cases: list, settings: dict):
    """The case the deltas and the fidelity references are taken from, or None.

    `cases` is one context group: a baseline is per (model, context), because a profile's
    answers are only comparable with another profile's on the identical prompt.
    """
    name = (settings.get("accuracy") or {}).get("baseline_profile")
    if not name:
        return None
    return next(
        (c for c in cases if c["profile"] == name and c.get("status") == "ok"), None
    )


def _context_groups(cases: list) -> dict:
    groups = {}
    for case in cases:
        if case.get("accuracy"):
            groups.setdefault((case["model"], case["context_tokens"]), []).append(case)
    return groups


def _fill_accuracy_deltas(cases: list, settings: dict) -> None:
    """Attach each suite's per-metric delta against the baseline profile.

    Computed here rather than in `_run_case` because the baseline profile's own case may not
    have run yet when a given case finishes; by report time the whole context group is in hand.
    A negative delta on recall is a profile retrieving less often than the baseline precision.
    """
    for group in _context_groups(cases).values():
        baseline = _baseline_case(group, settings)
        if baseline is None:
            continue
        for case in group:
            # Over the known suite names, not over whatever keys the dict happens to hold:
            # `case["accuracy"]` is suite-keyed by contract, and walking it blindly is how a
            # stray scalar in there becomes an AttributeError at the end of a long run.
            for suite in (accuracy.SUITE_RETRIEVAL, accuracy.SUITE_GENERATION):
                summary = _suite_summary(case, suite)
                base = _suite_summary(baseline, suite)
                if summary and base:
                    summary["deltas"] = accuracy.delta(
                        summary, base, accuracy.metrics_for(suite)
                    )


def _fill_generation_fidelity(cases: list, settings: dict) -> None:
    """Score every generation probe against the baseline profile's answer to the same prompt.

    This is the WWB measurement and it is inherently cross-case, which is why it cannot happen
    in `_run_case`: the reference is another profile's output, and that profile may run after
    this one. By report time the whole (model, context) group is in hand, so each probe is
    paired with the baseline's probe of the same `(task, depth, sample)` -- the identical
    prompt, by construction, because the specs are seeded from those coordinates.

    Idempotent: `write_reports` runs after every case, so this re-scores the group each time
    rather than accumulating. That is also what makes the report correct mid-run -- a case
    measured before the baseline gets its fidelity columns as soon as the baseline lands.
    """
    if not any(
        accuracy.SUITE_GENERATION in (c.get("accuracy") or {}) for c in cases
    ):
        return
    embedding_model = (settings.get("accuracy") or {}).get("embedding_model")

    for (model, _context), group in _context_groups(cases).items():
        baseline = _baseline_case(group, settings)
        if baseline is None:
            continue
        references = {
            _probe_key(p): p.get("prediction", "")
            for p in baseline.get("_probes", [])
            if p.get("suite") == accuracy.SUITE_GENERATION
        }
        if not references:
            continue
        encode = _answer_encoder(baseline.get("_model_dir"))
        reference_ids = {key: encode(text) for key, text in references.items()}

        for case in group:
            paired = [
                probe for probe in case.get("_probes", [])
                if probe.get("suite") == accuracy.SUITE_GENERATION
                and _probe_key(probe) in references
            ]
            for probe in paired:
                key = _probe_key(probe)
                probe.update(accuracy.score_fidelity(
                    probe.get("prediction", ""), references[key],
                    prediction_ids=encode(probe.get("prediction", "")),
                    reference_ids=reference_ids[key],
                ))
            if not paired:
                pass
            elif embedding_model:
                result = accuracy.fill_similarity(
                    paired, [references[_probe_key(p)] for p in paired], embedding_model
                )
                if not result.available:
                    _note_similarity_unavailable(settings, result.reason)
            else:
                _note_similarity_unavailable(
                    settings,
                    "accuracy.embedding_model is null, so the semantic column was switched "
                    "off deliberately. The lexical (ROUGE/chrF) and token-exact (FDT/SDT) "
                    "fidelity columns are unaffected.",
                )
            _resummarize_generation(case)
        _note_fidelity_tokenizer(settings, model, encode)


def _probe_key(probe: dict) -> tuple:
    return (probe.get("task"), _depth_label(probe.get("depth")), probe.get("sample"))


def _resummarize_generation(case: dict) -> None:
    """Rebuild the generation suite's aggregates now that the fidelity columns exist."""
    rows = [
        p for p in case.get("_probes", [])
        if p.get("suite") == accuracy.SUITE_GENERATION
    ]
    if rows:
        case["accuracy"][accuracy.SUITE_GENERATION] = _summarize_suite(
            accuracy.SUITE_GENERATION, rows
        )


# One loaded tokenizer per model directory: FDT/SDT are defined over model tokens, and
# re-reading a tokenizer for every probe in the sweep would dominate the scoring pass.
_ANSWER_ENCODERS: dict = {}


def _answer_encoder(model_dir):
    """A `text -> token ids` function for FDT/SDT, falling back to normalized words.

    FDT/SDT are token metrics, so the model's own tokenizer is the right unit and is what
    who_what_benchmark uses. But the accuracy path is otherwise runnable with nothing
    installed, and a report that refuses to compute a lexical metric because transformers is
    missing would be worse than one that computes it over words and says so -- the *shape* of
    the finding ("these two answers diverged after 3 units of 200") survives the change of
    unit. `_note_fidelity_tokenizer` records which was used.
    """
    if model_dir in _ANSWER_ENCODERS:
        return _ANSWER_ENCODERS[model_dir]

    encoder = None
    if model_dir:
        try:
            tokenizer = trial_runner._load_tokenizer(model_dir)
            encoder = lambda text: list(  # noqa: E731 - a named def buys nothing here
                tokenizer.encode(text or "", add_special_tokens=False)
            )
            encoder.unit = "model tokens"
        except Exception as exc:  # noqa: BLE001 - an optional unit must not end the run
            print(
                f"  [warn] could not load the tokenizer at {model_dir} for FDT/SDT "
                f"({type(exc).__name__}: {exc}); falling back to word-level divergence",
                flush=True,
            )
            encoder = None
    if encoder is None:
        # `lexical_tokens`, not the SQuAD `tokens`: the latter drops articles, which would
        # make "on a mat" and "on the mat" identical and report a divergence as agreement.
        encoder = lambda text: scoring.lexical_tokens(text)  # noqa: E731
        encoder.unit = "words"
    _ANSWER_ENCODERS[model_dir] = encoder
    return encoder


def _note_fidelity_tokenizer(settings: dict, model: str, encode) -> None:
    settings.setdefault("_fidelity_units", {})[model] = encode.unit


def _note_similarity_unavailable(settings: dict, reason) -> None:
    if reason:
        settings.setdefault("_similarity_notes", set()).add(reason)


def _acc_pct(value) -> str:
    return f"{value * 100:.0f}%" if isinstance(value, (int, float)) else "--"


def _acc_delta_pp(value) -> str:
    """A delta as signed percentage points, e.g. -12pp; '--' when it was not computed."""
    if not isinstance(value, (int, float)):
        return "--"
    return f"{value * 100:+.0f}pp"


def _acc_num(value, spec="{:.2f}") -> str:
    return spec.format(value) if isinstance(value, (int, float)) else "--"


def _ordered_cases(cases: list, suite: str) -> list:
    """Every case carrying `suite`, best first, with the unrankable ones kept at the end.

    Identity, not equality: two profiles can produce byte-identical rows, and `in` on dicts
    would drop the duplicate from the report entirely -- the same reason the throughput table
    ranks by `id`.
    """
    ranked = _accuracy_leaderboard(cases, suite)
    ranked_ids = {id(c) for c in ranked}
    return ranked + [
        c for c in cases if id(c) not in ranked_ids and _suite_summary(c, suite)
    ]


def _accuracy_section(cases: list, context: int, settings: dict) -> list:
    """Every accuracy table for one context, one section per suite that ran."""
    lines = []
    for suite in settings["accuracy"]["suites"]:
        if not any(_suite_summary(c, suite) for c in cases):
            continue
        lines += (
            _generation_section(cases, context, settings)
            if suite == accuracy.SUITE_GENERATION
            else _retrieval_section(cases, context, settings)
        )
    return lines


def _retrieval_section(cases: list, context: int, settings: dict) -> list:
    """RULER accuracy for one context: a per-task table, then the NIAH depth matrix.

    Two tables rather than one because they answer different questions. The per-task table is
    the headline -- a profile can hold `niah_single` at 100% and collapse on `vt` or `cwe`,
    and that is precisely the regression a single-needle number used to hide. The depth matrix
    is the classic NIAH presentation and covers only the depth-swept tasks, because a probe
    that spreads eight needles by construction has no depth to put in a column.
    """
    section = settings["accuracy"][accuracy.SUITE_RETRIEVAL]
    baseline_name = settings["accuracy"].get("baseline_profile")
    ordered = _ordered_cases(cases, accuracy.SUITE_RETRIEVAL)
    task_names = section["tasks"]

    lines = _heading(f"{context:,} tokens -- RETRIEVAL ACCURACY (RULER)")
    lines += [
        "Cells are each task's own RULER score: share of planted items found, or "
        "intersection-over-union for the aggregation tasks (cwe/fwe) which must punish "
        "over-answering. Decoys is the share of planted decoys that wrongly appeared -- lower "
        "is better, and it is the precision signal recall cannot give."
        + (f" Delta is the overall recall gap to `{baseline_name}`." if baseline_name else ""),
        "",
        f"| Profile | Pipeline | Speculative | {' | '.join(task_names)} | Overall | EM | F1 | "
        f"Decoys | Delta vs {baseline_name or '--'} |",
        "|---|---|---|" + "|".join(["---"] * len(task_names)) + "|---|---|---|---|---|",
    ]
    for case in ordered:
        summary = _suite_summary(case, accuracy.SUITE_RETRIEVAL)
        per_task = summary.get("per_task", {})
        cells = [
            _acc_pct((per_task.get(task) or {}).get("recall_rate")) for task in task_names
        ]
        lines.append(
            f"| {case['profile']} | {case.get('pipeline_mode') or '--'} "
            f"| {_mtp_cell(case, empty='off').strip() or 'off'} | "
            + " | ".join(cells)
            + f" | {_acc_pct(summary.get('recall_rate'))} "
            f"| {_acc_pct(summary.get('exact_match_rate'))} "
            f"| {_acc_pct(summary.get('token_f1'))} "
            f"| {_acc_pct(summary.get('distractor_rate'))} "
            f"| {_acc_delta_pp((summary.get('deltas') or {}).get('recall_rate_delta'))} |"
        )
    lines.append("")
    lines += _depth_matrix(ordered, section)

    ranked = _accuracy_leaderboard(cases, accuracy.SUITE_RETRIEVAL)
    if ranked:
        best = _suite_summary(ranked[0], accuracy.SUITE_RETRIEVAL)
        lines += [
            f"Best retrieval: {ranked[0]['profile']} -- recall "
            f"{_acc_pct(best.get('recall_rate'))}, EM {_acc_pct(best.get('exact_match_rate'))}, "
            f"F1 {_acc_pct(best.get('token_f1'))} over {best.get('probe_count')} probes",
            "",
        ]
    return lines


def _depth_matrix(ordered: list, section: dict) -> list:
    """The classic NIAH depth x profile matrix, over the depth-swept tasks only.

    A profile that holds the bottom of the context and loses the top has a working context
    shorter than the one it was given, and that is only visible split out by depth.
    """
    depths = section["depths"]
    swept = [t for t in section["tasks"] if tasks.is_depth_swept(t)]
    if not swept or not depths:
        return []
    lines = [
        f"Depth sweep -- recall by where the fact was planted ({', '.join(swept)}); "
        "0.00 is the very top of the transcript and furthest from the question.",
        "",
        f"| Profile | {' | '.join(f'd={_depth_label(d)}' for d in depths)} |",
        "|---|" + "|".join(["---"] * len(depths)) + "|",
    ]
    for case in ordered:
        per_depth = _suite_summary(case, accuracy.SUITE_RETRIEVAL).get("per_depth", {})
        cells = [
            _acc_pct((per_depth.get(_depth_label(d)) or {}).get("recall_rate")) for d in depths
        ]
        lines.append(f"| {case['profile']} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _generation_section(cases: list, context: int, settings: dict) -> list:
    """WWB-style fidelity for one context: each profile's real-task output against the
    baseline profile's output on the identical prompt.

    Three families of number side by side, because each is blind to something the next one
    catches. `Similarity` is who_what_benchmark's embedding cosine -- semantic, so a
    reworded but equivalent answer scores near 1.0. `ROUGE`/`chrF` are lexical, so they fall
    when the wording changes even if the meaning did not. `FDT`/`SDT` are token-exact, and
    are the only ones that can say greedy decoding stopped being reproducible at token 3.
    Reporting only the first would call a visibly reworded answer "lossless"; reporting only
    the last would call a legitimate tie-break "a regression".
    """
    baseline_name = settings["accuracy"].get("baseline_profile")
    ordered = _ordered_cases(cases, accuracy.SUITE_GENERATION)
    unit = (settings.get("_fidelity_units") or {}).get(
        ordered[0]["model"] if ordered else None, "tokens"
    )

    lines = _heading(f"{context:,} tokens -- GENERATION FIDELITY (who_what_benchmark)")
    lines += [
        "The model does the real classroom task at full answer length; each answer is scored "
        f"against the baseline profile `{baseline_name or '--'}`'s answer to the identical "
        "prompt. This is the measurement retrieval cannot make -- a profile can return every "
        "planted code and still write worse prose.",
        "",
        f"| Profile | Pipeline | Speculative | Similarity | ROUGE-1 | ROUGE-2 | ROUGE-L | chrF "
        f"| FDT ({unit}) | SDT norm | Identical | Coverage |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for case in ordered:
        summary = _suite_summary(case, accuracy.SUITE_GENERATION)
        lines.append(
            f"| {case['profile']} | {case.get('pipeline_mode') or '--'} "
            f"| {_mtp_cell(case, empty='off').strip() or 'off'} "
            f"| {_acc_num(summary.get('similarity'), '{:.4f}')} "
            f"| {_acc_num(summary.get('rouge1'))} | {_acc_num(summary.get('rouge2'))} "
            f"| {_acc_num(summary.get('rouge_l'))} | {_acc_num(summary.get('chrf'))} "
            f"| {_acc_num(summary.get('fdt'), '{:.1f}')} "
            f"| {_acc_num(summary.get('sdt_norm'))} "
            f"| {_acc_pct(summary.get('identical_rate'))} "
            f"| {_acc_pct(summary.get('fact_coverage'))} |"
        )
    lines += [
        "",
        "Similarity is an embedding cosine (semantic); ROUGE/chrF are lexical overlap; FDT is "
        f"how many {unit} the two answers agreed on before first differing and SDT norm the "
        "share that differs (lower better); Identical is a byte-for-byte match. Each family "
        "is blind to something the next catches -- a reworded but equivalent answer scores "
        "~1.0 on similarity and well below it on ROUGE. Coverage is the only column scored "
        "against real ground truth rather than another profile's output.",
        f"The `{baseline_name or '--'}` row compares against itself, so it reads 1.00 "
        "throughout by construction: a sanity check, not a result.",
        "",
    ]
    for note in sorted(settings.get("_similarity_notes") or ()):
        lines += [f"NOTE: similarity not measured -- {note}", ""]

    ranked = _accuracy_leaderboard(cases, accuracy.SUITE_GENERATION)
    # Only profiles that actually drifted: calling a byte-identical profile the "largest
    # drift" reads as a finding when the finding is that there was none.
    drifted = [
        c for c in ranked
        if c["profile"] != baseline_name
        and (_suite_summary(c, accuracy.SUITE_GENERATION).get("identical_rate") or 0.0) < 1.0
    ]
    if not drifted:
        others = [c for c in ranked if c["profile"] != baseline_name]
        if others:
            lines += [
                f"No drift: every one of {len(others)} profile(s) reproduced the baseline's "
                "answers byte for byte.",
                "",
            ]
    else:
        worst = _suite_summary(drifted[-1], accuracy.SUITE_GENERATION)
        # The similarity clause is dropped rather than printed as "--": a sentence reading
        # "similarity --, ROUGE-L 0.37" invites the reader to treat the dash as a low score.
        semantic = (
            f"similarity {_acc_num(worst.get('similarity'), '{:.4f}')}, "
            if worst.get("similarity") is not None else ""
        )
        lines += [
            f"Largest drift: {drifted[-1]['profile']} -- {semantic}ROUGE-L "
            f"{_acc_num(worst.get('rouge_l'))}, chrF {_acc_num(worst.get('chrf'))}, first "
            f"divergence after {_acc_num(worst.get('fdt'), '{:.1f}')} {unit} over "
            f"{worst.get('probe_count')} probes",
            "",
        ]
    return lines


def _matched_baseline(candidate: dict, cases: list) -> dict | None:
    """Find a non-speculative case with the same model and execution settings."""
    fields = (
        "model", "context_tokens", "device", "weight_format", "weight_precision", "pipeline_mode",
        "throughput_task",
        "ov_config", "scheduler_config",
    )
    return next((case for case in cases if not case.get("mtp") and all(
        case.get(field) == candidate.get(field) for field in fields
    )), None)


def _verification_cost(case: dict, baseline: dict) -> str:
    """How expensive one speculative pass was, in units of one baseline decode step.

    The speedup is ``tokens_per_step / pass_cost``: yield alone cannot predict it. On the
    Qwen3.6 MoE, verifying k + 1 tokens activates the union of every token's experts, so a
    pass costs several single-token steps -- which is why 3.4 tok/step can net ~1.2x, and
    why the sentence that reports the speedup has to say what each pass cost.

    The runtime's steady-state per-pass inference time is used when both sides have it (see
    metrics.pass_profile); otherwise the pass is approximated as TPOT x tokens per pass,
    which also folds in the first decode pass's one-time cost.
    """
    pass_ms, step_ms = case.get("steady_pass_ms"), baseline.get("steady_pass_ms")
    if not pass_ms or not step_ms:
        tokens_per_step = case.get("tokens_per_step")
        tpot, step_ms = case.get("other_tokens_avg_latency"), baseline.get("other_tokens_avg_latency")
        if not isinstance(tokens_per_step, (int, float)) or not tpot or not step_ms:
            return ""
        pass_ms = tpot * tokens_per_step
    return (
        f"; each verification pass (draft + target) took {_fmt(pass_ms, '{:.1f}')} ms, "
        f"{pass_ms / step_ms:.2f}x a baseline decode step"
    )


def _steady_state(case: dict, baseline: dict) -> str:
    """The decode rate once the first pass is paid, which a longer answer converges to."""
    tpot, base = case.get("steady_tpot"), baseline.get("steady_tpot")
    if not tpot or not base:
        return ""
    return (
        f" Steady state, excluding the first decode pass ({_fmt(case.get('first_decode_ms'), '{:.0f}')} "
        f"ms vs {_fmt(baseline.get('first_decode_ms'), '{:.0f}')} ms): {_fmt(tpot)} vs "
        f"{_fmt(base)} ms/token of inference, **{base / tpot:.2f}x**."
    )


def _mtp_speedup(cases: list) -> str | None:
    """Compare the fastest speculative case that has a matched usable baseline."""
    ranked = _leaderboard(cases)
    pair = next(((case, baseline) for case in ranked if case.get("mtp")
                 if (baseline := _matched_baseline(case, ranked)) is not None), None)
    if pair is None:
        return None
    with_mtp, without = pair
    baseline_tpot = without["other_tokens_avg_latency"]
    mtp_tpot = with_mtp["other_tokens_avg_latency"]
    if not mtp_tpot or not baseline_tpot:
        return None
    strategy = metrics.strategy_label(with_mtp)
    return (
        f"**{strategy} speedup: {baseline_tpot / mtp_tpot:.2f}x on decode** -- "
        f"`{with_mtp['profile']}` (k={with_mtp.get('num_assistant_tokens')}) at "
        f"{_fmt(mtp_tpot)} ms/token against `{without['profile']}` at "
        f"{_fmt(baseline_tpot)} ms/token"
        + (
            f", {'estimated ' if with_mtp.get('mtp_acceptance_estimated') else ''}"
            f"acceptance {with_mtp['mtp_acceptance_rate'] * 100:.0f}% of drafted candidates "
            f"for {_fmt(with_mtp.get('tokens_per_step'), '{:.2f}')} tokens per verification pass"
            if isinstance(with_mtp.get("mtp_acceptance_rate"), (int, float)) else ""
        )
        + _verification_cost(with_mtp, without)
        + ". Measured TTFT: "
        f"{_ttft_seconds(with_mtp)}s vs {_ttft_seconds(without)}s."
        + _steady_state(with_mtp, without)
    )


def _mtp_output_check(cases: list) -> str | None:
    """Compare greedy outputs only between matched speculative and baseline cases."""
    internally_inconsistent = [
        case["profile"] for case in cases
        if case.get("status") == "ok" and case.get("output_consistent") is False
    ]
    if internally_inconsistent:
        return (
            "**WARNING: repeated greedy outputs differ within profile(s)** "
            + ", ".join(f"`{name}`" for name in internally_inconsistent)
            + ". The run is nondeterministic, so its speculative output comparison is inconclusive."
        )
    usable = [case for case in cases if case.get("status") == "ok" and case.get("output_sha256")]
    pairs = [(case, baseline) for case in usable if case.get("mtp")
             if (baseline := _matched_baseline(case, usable)) is not None]
    if not pairs:
        return None
    mismatches = [
        case["profile"] for case, baseline in pairs
        if case["output_sha256"] != baseline["output_sha256"]
    ]
    if mismatches:
        return (
            "**WARNING: greedy speculative output differs from the baseline** for "
            + ", ".join(f"`{name}`" for name in mismatches)
            + ". Do not treat their speedup as valid until the divergence is explained."
        )
    return "**Greedy output check: all matched speculative profiles match their baseline exactly.**"


def write_report(output_dir: str, settings: dict, cases: list, platform_info: dict,
                 completed: bool) -> list:
    """(Re)write the run's single `report.txt` and return its lines.

    Rewritten after every case, not once at the end, because this tool's job is to push the
    box until it breaks and it can break hard enough to take the orchestrator with it. When
    that happened previously the file on disk still described an earlier run -- reporting a
    PASS at 160K the real trials had just disproved. A stale report that looks current is
    worse than none.

    One file, not five. Everything the run measured -- speed, resources and (with
    --accuracy) every accuracy metric and every individual probe -- is in here, and the
    caller prints these same lines to the console, so the two can never disagree.
    """
    # Generation fidelity and every delta are cross-case: the reference is the baseline
    # profile's own output, which may not have run when a given case finished. By report
    # time the whole group is in hand, so fill them before anything is rendered.
    if any(c.get("accuracy") for c in cases):
        _fill_generation_fidelity(cases, settings)
        _fill_accuracy_deltas(cases, settings)

    lines = _report_lines(settings, cases, platform_info, completed)
    with open(os.path.join(output_dir, REPORT_NAME), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return lines


def throughput_task_prompt(name: str | None, model_name: str,
                           app_config_path: str | None = None) -> dict | None:
    """The framing a throughput task puts around the transcript, for build_benchmark_prompt.

    `summary_2s` (None) is the built-in two-sentence summary. `classroom_summary` is the
    application's own summarizer request, rebuilt the way summarizer_component._get_message
    builds it: the configured language and mode's system prompt from the app config, and
    `/no_think` ahead of the transcript for Qwen3 models. Its answer is a long, templated
    Markdown summary rather than two free sentences -- the output dFlash actually has to
    draft in production, and the one its speedup should be judged on.
    """
    if name in (None, THROUGHPUT_TASKS[0]):
        return None
    path = app_config_path or _APP_CONFIG_PATH
    try:
        app = load_config(path)
        summarizer = app.models.summarizer
        prompts = vars(summarizer.system_prompt)[app.app.language]
        mode = str(getattr(summarizer, "mode", "dialog")).strip().lower()
        system_prompt = {"teacher": prompts.Teacher, "hybrid": prompts.Hybrid}.get(
            mode, prompts.Dialog
        )
    except (OSError, AttributeError, KeyError, TypeError) as exc:
        raise SystemExit(
            f"throughput_task {name!r} needs models.summarizer.system_prompt in {path}: {exc}"
        ) from exc
    return {
        "name": name,
        "system_prompt": system_prompt,
        "user_prefix": "/no_think\n" if "qwen3" in model_name.lower() else "",
        "suffix": "",
    }


def _gap_note(settings: dict) -> str:
    """Header suffix naming the idle gap, so a burst and a sustained figure never pass as one."""
    gap = settings.get("iteration_gap_sec") or 0
    return f", {gap:g}s idle before each measured run" if gap else ", back-to-back (sustained)"


def _report_lines(settings: dict, cases: list, platform_info: dict,
                  completed: bool) -> list:
    """The whole report: header, then per context speed + accuracy, then probes and
    failures. Speed and resources are reported for every run, accuracy only when it ran --
    an accuracy run measures both, because its probes carry the same timing records the
    throughput path produces."""
    acc = settings.get("accuracy")
    workload = (
        "accuracy probes (per-suite decode ceilings; EOS respected)" if acc else
        f"{settings['iterations']} iteration(s) of "
        f"`{settings.get('throughput_task') or THROUGHPUT_TASKS[0]}` "
        f"@ up to {settings['output_tokens']} output tokens (EOS respected)"
    ) + _gap_note(settings)
    lines = [
        "=" * 78,
        f" Long-Context Benchmark -- {', '.join(settings['models'])}",
        "=" * 78,
        f"Generated : {datetime.now().isoformat(timespec='seconds')}"
        f"   [{'complete' if completed else 'IN PROGRESS / ENDED EARLY'}]",
        f"Hardware  : {platform_info.get('Processor', '--')}, "
        f"{platform_info.get('Memory', '--')} RAM, {platform_info.get('iGPU', '--')}",
        f"Device    : {settings['device']} | weights {settings['weight_format']} | "
        f"budgets RAM <= {settings['max_system_memory_pct']:g}%, "
        f"GPU <= {settings['gpu_memory_budget_gb']:g} GB",
        f"Contexts  : {', '.join(f'{c:,}' for c in settings['context_tokens'])} tokens | "
        f"{settings['warmup']} warmup + {workload}",
    ]
    if acc:
        lines.append(
            f"Accuracy  : {', '.join(acc['suites'])} | baseline profile "
            f"{acc.get('baseline_profile') or '(none)'} | seed {acc['seed']}"
        )
    lines.append("")

    for context in settings["context_tokens"]:
        at_context = [c for c in cases if c["context_tokens"] == context]
        if at_context:
            lines += _speed_section(at_context, context, settings)
            if acc:
                lines += _accuracy_section(at_context, context, settings)

    lines += _probe_detail(cases)
    lines += _failure_section(cases)
    return lines


def _heading(text: str) -> list:
    return ["-" * 78, f" {text}", "-" * 78, ""]


def _speed_section(at_context: list, context: int, settings: dict) -> list:
    """TTFT, TPOT, throughput and RAM/GPU for one context, ranked by TPOT.

    Latencies are milliseconds and throughputs tokens/second, in llm_bench's units; every
    figure is the median of the measured generations with the warm-up excluded. Memory is
    peak / mean and the two are not interchangeable: peak includes model load and decides
    whether the box can run the configuration at all, mean excludes it and is what the
    configuration costs for the minutes it runs.
    """
    lines = _heading(f"{context:,} tokens -- SPEED AND RESOURCES")
    lines += [
        "| Model | Profile | Weights | Pipeline | KV | Speculative | Status | Output tok "
        "| TPOT ms/tok | TTFT s "
        f"| Decode tok/s | E2E tok/s | Prefill tok/s | RAM peak/mean GB | GPU peak/mean GB "
        f"(% of {settings['gpu_memory_budget_gb']:g}) | KV est GB | cache |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    ranked = _leaderboard(at_context)
    # Identity, not equality: two profiles can produce byte-identical rows, and `in` on
    # dicts would drop the duplicate from the report entirely.
    ranked_ids = {id(c) for c in ranked}
    for case in ranked + [c for c in at_context if id(c) not in ranked_ids]:
        # A case that never produced a measurement still gets a row -- its status is the
        # finding -- but its timing columns stay blank rather than reading as zeros.
        timings = [
            _fmt(case.get("other_tokens_avg_latency")),
            _ttft_seconds(case),
            _fmt(case.get("decode_throughput"), "{:.2f}"),
            _fmt(case.get("e2e_throughput")),
            _fmt(case.get("prefill_throughput")),
        ]
        if case["status"] not in _USABLE_STATUSES:
            timings = ["--"] * len(timings)
        lines.append(
            f"| {case['model']} | {case['profile']} "
            f"| {case.get('weight_precision') or case.get('weight_format') or '--'} "
            f"| {case.get('pipeline_mode') or '--'} "
            # KV precision is a column and not a footnote because it is the variable most
            # often changed by accident: two profiles meant to differ only in the pipeline,
            # one edited to f16, and the TTFT gap between them is no longer attributable.
            f"| {case.get('kv_cache_precision') or '--'} "
            f"| {_mtp_cell(case, empty='off').strip() or 'off'} | {case['status']} | "
            f"{_output_tokens_cell(case)} | "
            + " | ".join(timings)
            + f" | {_fmt(case.get('peak_ram_gb'))}/{_fmt(case.get('mean_ram_gb'))} "
            f"| {_fmt(case.get('peak_gpu_gb'))}/{_fmt(case.get('mean_gpu_gb'))} "
            f"({_fmt(case.get('peak_gpu_pct_of_budget'))}%/"
            f"{_fmt(case.get('mean_gpu_pct_of_budget'))}%) "
            f"| {_fmt(case.get('expected_kv_gb'), '{:.2f}')} "
            f"| {_fmt(case.get('cache_size_gb'), '{:g}')} |"
        )
    lines.append("")

    if settings.get("accuracy"):
        lines += [
            "Timing medians pool accuracy probes with different answer lengths. "
            "Use the default throughput run for a repeated-prompt speed comparison; "
            "Output tok shows the measured range. Decode tok/s is 1000 / median TPOT.",
            "",
        ]
    if ranked:
        best = ranked[0]
        lines += [
            f"Best TPOT  : {best['profile']} ({best.get('pipeline_mode') or '--'}) -- "
            f"{_best_summary(best)}, peak GPU {_fmt(best.get('peak_gpu_gb'))} GB "
            f"({_fmt(best.get('peak_gpu_pct_of_budget'))}% of budget)",
            f"             ov {best['ov_config']}",
            f"             scheduler {best['scheduler_config']}",
        ]
    by_ttft = _ttft_leaderboard(at_context)
    if by_ttft:
        fastest = by_ttft[0]
        lines.append(
            f"Fastest TTFT: {fastest['profile']} "
            f"({fastest.get('pipeline_mode') or '--'}) -- {_ttft_seconds(fastest)}s, prefill "
            f"{_fmt(fastest.get('prefill_throughput'))} tok/s"
        )
    for note in (_mtp_speedup(at_context), _mtp_output_check(at_context)):
        if note:
            lines.append(note.replace("**", ""))
    lines.append("")
    return lines


def _probe_detail(cases: list) -> list:
    """Every scored probe, one line each -- what was planted, what came back, what it scored.

    This is the only place the model's
    actual answers survive, and reading the worst-scoring ones is how a suspicious rate gets
    explained, so it stays even though it is the longest section.
    """
    rows = [(c, p) for c in cases for p in c.get("_probes", [])]
    if not rows:
        return []
    lines = _heading(f"PROBE DETAIL ({len(rows)} probe(s))")
    for case, probe in rows:
        depth = probe.get("depth")
        where = f"d={depth:.2f}" if isinstance(depth, (int, float)) else "  -  "
        if probe.get("suite") == accuracy.SUITE_GENERATION:
            scores = (
                f"rougeL {_acc_num(probe.get('rouge_l'))} chrf {_acc_num(probe.get('chrf'))} "
                f"fdt {_acc_num(probe.get('fdt'), '{:g}')} "
                f"sim {_acc_num(probe.get('similarity'), '{:.4f}')} "
                f"cover {_acc_num(probe.get('fact_coverage'))}"
            )
        else:
            scores = (
                f"recall {_acc_num(probe.get('recall'))} "
                f"em {1 if probe.get('exact_match') else 0} "
                f"f1 {_acc_num(probe.get('token_f1'))} "
                f"decoy {_acc_num(probe.get('distractor_rate'))}"
            )
        lines += [
            f"{case['profile']:<12} {probe.get('task', '?'):<16} {where} #{probe.get('sample')}"
            f"  {probe.get('input_size', 0):,} tok  {scores}",
            f"    truth  : {_one_line(probe.get('truth'), 110)}",
            f"    answer : {_one_line(probe.get('prediction'), 110)}",
        ]
    lines.append("")
    return lines


def _one_line(text, limit: int) -> str:
    """Collapse an answer to a single readable line -- the report is a text file, and a
    generated paragraph with newlines in it would break every column after it."""
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def _failure_section(cases: list) -> list:
    failures = [c for c in cases if c["status"] not in _USABLE_STATUSES]
    if not failures:
        return []
    lines = _heading("CASES THAT DID NOT PRODUCE A MEASUREMENT")
    for case in failures:
        lines.append(
            f"{case['model']} / {case['profile']} @ {case['context_tokens']:,} tok -- "
            f"{case['status']}: {_one_line(case.get('error'), 200)}"
        )
    lines.append("")
    return lines



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


def _print_accuracy_plan(settings: dict) -> None:
    """What `--accuracy` will actually run, before anything loads.

    The probe count is the number that decides whether this is a ten-minute run or an
    overnight one -- `--list-profiles` exists so that is knowable in advance, and the full
    RULER suite across a depth sweep multiplies faster than it looks.
    """
    acc = settings["accuracy"]
    total = 0
    print(f"\nAccuracy mode -- suites: {', '.join(acc['suites'])}")
    for suite in acc["suites"]:
        section = acc[suite]
        count = sum(
            (len(section["depths"]) if tasks.is_depth_swept(task) else 1) * section["samples"]
            for task in section["tasks"]
        )
        total += count
        print(
            f"  {suite}: {count} probe(s) per case -- {', '.join(section['tasks'])}; "
            f"{section['samples']} sample(s), depths {section['depths']} "
            f"(depth-swept tasks only), {section['output_tokens']} output tokens"
        )
    print(
        f"  {total} probe(s) per case x "
        f"{len(settings['models']) * len(settings['context_tokens']) * len(settings['profiles'])}"
        f" case(s); baseline {acc.get('baseline_profile') or '(none)'}"
    )
    if acc.get("embedding_model"):
        print(
            f"  similarity: {acc['embedding_model']} via who_what_benchmark "
            "(optional -- the column reports '--' if it is not installed)"
        )


# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()
    settings = _load_settings(args)

    if args.list_profiles:
        print(f"Device: {settings['device']} | Weights: {settings['weight_format']}")
        print(f"Models: {', '.join(settings['models'])}")
        for model_name in settings["models"]:
            model_dir = _model_ir_dir(
                settings["models_base_path"], settings["provider"], model_name,
                settings["weight_format"], settings["model_dirs"],
            )
            print(f"  {model_name}: {ir_weight_precision(model_dir) or 'IR not found'} -- {model_dir}")
        print(f"Contexts: {', '.join(f'{c:,}' for c in settings['context_tokens'])}")
        print(
            f"Iterations: {settings['warmup']} warmup + {settings['iterations']} measured, "
            f"{settings['output_tokens']} output tokens, timeout {settings['timeout_sec']:g}s"
            + _gap_note(settings)
        )
        total = len(settings["models"]) * len(settings["context_tokens"]) * len(settings["profiles"])
        print(f"\n{len(settings['profiles'])} profile(s), {total} case(s):\n")
        for profile in settings["profiles"]:
            print(f"  {profile['name']}")
            print(f"    pipeline:  {pipeline_mode(profile['scheduler'])}")
            print(f"    ov:        {format_config(profile['ov'])}")
            print(
                "    scheduler: "
                + (format_config(profile["scheduler"]) if profile["scheduler"] else "(none)")
            )
            print(f"    speculative: {format_mtp(profile['mtp'], settings['device'])}")
            if profile["mtp"].get("strategy") == "dflash":
                print(
                    "    draft weights: "
                    f"{ir_weight_precision(profile['mtp']['model']) or 'IR not found'}"
                )
        if settings.get("accuracy"):
            _print_accuracy_plan(settings)
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
                settings["weight_format"], settings["model_dirs"],
            )
            if not _ir_ready(model_dir):
                export = _prep_command(model_name, model_dir, settings["weight_format"])
                print(
                    f"\n[{model_name}] IR not found at {model_dir}\n  Run first: {export}",
                    flush=True,
                )
                cases.append({
                    "model": model_name, "profile": "-", "context_tokens": 0,
                    "device": settings["device"], "weight_format": settings["weight_format"],
                    "status": "missing_ir", "error": export,
                })
                continue

            static = {
                "weight_disk_gb": _weight_disk_gb(model_dir),
                "model_config": _load_model_config(model_dir),
                "fixed_state_bytes": fixed_state_cache_bytes(model_dir),
                "mtp_layers": mtp_head_layers(model_dir),
                "weight_precision": ir_weight_precision(model_dir),
            }
            print(
                f"\n=== {model_name} === weights on disk: {static['weight_disk_gb']:.1f} GB "
                f"({static['weight_precision'] or settings['weight_format']}) -- {model_dir}",
                flush=True,
            )
            if not _precision_matches_label(static["weight_precision"], settings["weight_format"]):
                print(
                    f"  [warn] model.weight_format is {settings['weight_format']!r} but the IR "
                    f"is {static['weight_precision']}; rows report the IR's precision",
                    flush=True,
                )
            _preflight_cache_sizes(settings, static, model_name)

            for context in settings["context_tokens"]:
                for profile in settings["profiles"]:
                    case = _run_case(model_name, model_dir, profile, context, settings, static)
                    cases.append(case)
                    # Before the per-case line, not after: write_report is what fills the
                    # generation suite's fidelity columns (rouge_l, chrf, fdt, similarity),
                    # which are cross-case and cannot be scored until the baseline profile's
                    # answers are in `cases`. Printing first showed every case's ROUGE-L as
                    # "--" even on a single profile comparing against itself.
                    write_report(output_dir, settings, cases, platform_info, completed=False)
                    print(_format_case(case), flush=True)
        completed = True
    finally:
        # Reached on Ctrl-C and on an unhandled failure too: the cases already measured are
        # worth a report, and its header says the run ended early.
        try:
            lines = write_report(output_dir, settings, cases, platform_info, completed)
        except Exception:  # noqa: BLE001 - must not mask whatever is already unwinding
            traceback.print_exc()
        else:
            # The console gets exactly what the file got -- one builder, two sinks, so the
            # two can never disagree about what was measured.
            print("\n" + "\n".join(lines), flush=True)
            print(
                f"Report written to {os.path.join(output_dir, REPORT_NAME)}"
                + ("" if completed else " -- run ended early, marked incomplete"),
                flush=True,
            )


if __name__ == "__main__":
    main()
