# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Per-iteration benchmark records and their aggregation, in llm_bench's units.

Field names and units follow openvino.genai/tools/llm_bench so a run here can be
read next to an llm_bench report without a translation table: latencies are
milliseconds, throughputs are tokens/second, and iteration 0 is the warm-up and
is excluded from every aggregate.

Two throughput figures matter at 160K and they are not interchangeable.
`prefill_throughput` (input tokens / TTFT) dominates the wall clock -- at these
context lengths TTFT is ~96% of total generation time -- while
`decode_throughput` is llm_bench's "2nd token" rate. `e2e_throughput` is the one
number that ranks configurations end to end.
"""

from __future__ import annotations

import statistics

# Written per iteration, in this order. Names/units mirror llm_bench's report
# columns; prefill/decode/e2e throughput are this tool's long-context additions.
ITERATION_FIELDS = [
    "iteration",
    "warmup",
    "input_size",
    "output_size",
    "generation_time",
    "latency",
    "first_token_latency",
    "other_tokens_avg_latency",
    "tokenization_time",
    "detokenization_time",
    "prefill_throughput",
    "decode_throughput",
    "e2e_throughput",
]

# Aggregated as median/min/max across the measured iterations.
AGGREGATED_METRICS = [
    "generation_time",
    "latency",
    "first_token_latency",
    "other_tokens_avg_latency",
    "prefill_throughput",
    "decode_throughput",
    "e2e_throughput",
]


def _rate(numerator: float, seconds: float) -> float:
    return round(numerator / seconds, 3) if numerator and seconds and seconds > 0 else 0.0


def iteration_record(
    iteration: int,
    input_size: int,
    output_size: int,
    generation_time: float,
    first_token_latency: float | None,
    other_tokens_avg_latency: float | None = None,
    tokenization_time: float = 0.0,
    detokenization_time: float = 0.0,
    warmup: bool = False,
) -> dict:
    """One measured generation, in llm_bench units (ms for latency, s for time).

    `other_tokens_avg_latency` divides the post-TTFT time by ``output_size - 1``,
    not by ``output_size``: the first token is produced by prefill and is already
    accounted for by `first_token_latency`, so including it in the decode average
    understates per-token decode cost.
    """
    total_ms = generation_time * 1000.0
    ttft = first_token_latency
    tpot = other_tokens_avg_latency
    if tpot is None and ttft is not None and output_size > 1:
        tpot = round((total_ms - ttft) / (output_size - 1), 3)

    return {
        "iteration": iteration,
        "warmup": warmup,
        "input_size": input_size,
        "output_size": output_size,
        "generation_time": round(generation_time, 3),
        "latency": round(total_ms / output_size, 3) if output_size else None,
        "first_token_latency": round(ttft, 3) if ttft is not None else None,
        "other_tokens_avg_latency": tpot,
        "tokenization_time": round(tokenization_time, 3),
        "detokenization_time": round(detokenization_time, 3),
        "prefill_throughput": _rate(input_size, ttft / 1000.0) if ttft else 0.0,
        "decode_throughput": round(1000.0 / tpot, 3) if tpot else 0.0,
        "e2e_throughput": _rate(input_size + output_size, generation_time),
    }


def measured(records: list) -> list:
    """The iterations that count: everything but the warm-up."""
    return [r for r in records or [] if not r.get("warmup")]


def aggregate(records: list) -> dict:
    """Median/min/max of each metric over the measured iterations.

    Median rather than mean because the failure mode this exists to expose is a
    single slow outlier: the same 160K configuration was measured at 247.0s and
    349.5s on separate single-shot runs, and a two-sample mean would have hidden
    which of those is representative. `*_min`/`*_max` keep the spread visible.
    """
    rows = measured(records)
    if not rows:
        return {"iterations_measured": 0}

    out = {"iterations_measured": len(rows)}
    for metric in AGGREGATED_METRICS:
        values = [r[metric] for r in rows if r.get(metric) is not None]
        if not values:
            continue
        out[metric] = round(statistics.median(values), 3)
        out[f"{metric}_min"] = round(min(values), 3)
        out[f"{metric}_max"] = round(max(values), 3)
    for field in ("input_size", "output_size"):
        out[field] = rows[-1].get(field)
    return out


def format_iteration(record: dict) -> str:
    """llm_bench-style one-liner, so a live log reads the same as one of its runs."""
    tag = "warmup" if record.get("warmup") else f"iter {record['iteration']}"
    line = (
        f"[{tag}] input {record['input_size']:,} tok, output {record['output_size']} tok, "
        f"generation {record['generation_time']:.2f}s"
    )
    ttft = record.get("first_token_latency")
    tpot = record.get("other_tokens_avg_latency")
    if ttft is None or tpot is None:
        return line
    return (
        f"{line}, latency {record['latency']:.1f} ms/token | "
        f"1st token {ttft:.1f} ms, other tokens {tpot:.1f} ms/token | "
        f"prefill {record['prefill_throughput']:.1f} tok/s, "
        f"decode {record['decode_throughput']:.2f} tok/s, "
        f"e2e {record['e2e_throughput']:.1f} tok/s"
    )
