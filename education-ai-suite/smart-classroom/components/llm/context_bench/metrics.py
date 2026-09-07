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
number that summarizes end-to-end work; profile ranking uses TPOT, then TTFT.

The `mean_*_gb` memory fields are the exception to "one record, one generate": they
are measured by the orchestrator, not the child, and merged into each record before
aggregation. See `benchmark._MemorySampler` for why peak and mean are both reported.

`tokens_per_step` / `mtp_acceptance_rate` are this tool's multi-token-prediction
additions and have no llm_bench counterpart. They exist because TPOT alone cannot
say *why* an MTP profile is faster: on the shipped 8K workload k=3 accepts about
51% of candidates while k=6 accepts about 30%, and only that comparison explains
why raising k further has stopped buying anything.
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
    # Multi-token prediction. Present on every record so iterations.csv keeps one
    # shape; None on a profile that does not run MTP.
    "num_assistant_tokens",
    "mtp_draft_tokens",
    "mtp_accepted_tokens",
    "mtp_rejected_tokens",
    "verification_steps",
    "tokens_per_step",
    "mtp_acceptance_rate",
    # Draft-model inference time as a fraction of the main model's, from
    # SDPerModelsPerfMetrics. The signal a k-sweep actually turns on: acceptance says how
    # many candidates stuck, this says what proposing them cost -- a draft head whose ratio
    # climbs toward 1 is eating the decode saving even while acceptance still looks healthy.
    "mtp_draft_to_main_ratio",
    "output_sha256",
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
    "mtp_draft_tokens",
    "mtp_accepted_tokens",
    "mtp_rejected_tokens",
    "tokens_per_step",
    "mtp_acceptance_rate",
    "mtp_draft_to_main_ratio",
]

# Median only, no _min/_max. These are the orchestrator's per-iteration memory windows,
# merged into each record before aggregation (see benchmark._MemorySampler): the case-level
# `peak_*` is already the whole-case high-water mark, so a min/max *of the means* would add
# columns that answer nothing the peak does not.
MEDIAN_ONLY_METRICS = [
    "mean_ram_gb",
    "mean_gpu_gb",
]


def _rate(numerator: float, seconds: float) -> float:
    return round(numerator / seconds, 3) if numerator and seconds and seconds > 0 else 0.0


def mtp_yield(
    output_size: int, verification_steps: int | None, num_assistant_tokens: int | None
) -> tuple[float | None, float | None]:
    """How much work each main-model pass produced, and how much of the draft stuck.

    Under multi-token prediction the main model no longer runs once per output token:
    it verifies a batch of drafted candidates and keeps the accepted prefix plus one
    bonus token. `tokens_per_step` is that yield -- ``output_size / steps`` -- and it
    is what TPOT is actually divided by. A profile with MTP off yields exactly 1.00,
    which doubles as a self-check that the draft head is really engaged.

    `mtp_acceptance_rate` converts the yield into the fraction of the ``k`` drafted
    candidates that survived verification, ``(tokens_per_step - 1) / k``: one token
    per step is the bonus the main model produces on its own and was never drafted,
    so counting it as an acceptance would report a non-zero rate for a run that
    accepted nothing. It is the number that says whether raising ``k`` still buys
    anything -- measured on Qwen3.8-27B at 8K, 63% at k=1 down to 30% at k=6.

    Both are None when the runtime did not report step counts, rather than 0: "not
    measured" and "nothing accepted" are different findings.
    """
    if not verification_steps or verification_steps <= 0 or not output_size:
        return None, None
    tokens_per_step = output_size / verification_steps
    if not num_assistant_tokens or num_assistant_tokens <= 0:
        return round(tokens_per_step, 3), None
    # Clamped because the ratio is a measurement, not an identity: a runtime that
    # counts the prefill pass as a step (or omits one) must not produce a rate
    # outside [0, 1] and make the column unreadable.
    rate = (tokens_per_step - 1.0) / num_assistant_tokens
    return round(tokens_per_step, 3), round(min(1.0, max(0.0, rate)), 4)


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
    num_assistant_tokens: int | None = None,
    verification_steps: int | None = None,
    mtp_acceptance_rate: float | None = None,
    mtp_draft_tokens: int | None = None,
    mtp_accepted_tokens: int | None = None,
    mtp_rejected_tokens: int | None = None,
    mtp_draft_to_main_ratio: float | None = None,
    output_sha256: str | None = None,
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
        post_ttft_ms = total_ms - ttft
        tpot = round(post_ttft_ms / (output_size - 1), 3) if post_ttft_ms > 0 else None

    tokens_per_step, _ = mtp_yield(
        output_size, verification_steps, num_assistant_tokens
    )
    acceptance = (
        round(min(1.0, max(0.0, mtp_acceptance_rate)), 4)
        if mtp_acceptance_rate is not None
        else None
    )

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
        "num_assistant_tokens": num_assistant_tokens,
        "mtp_draft_tokens": mtp_draft_tokens,
        "mtp_accepted_tokens": mtp_accepted_tokens,
        "mtp_rejected_tokens": mtp_rejected_tokens,
        "verification_steps": verification_steps,
        "tokens_per_step": tokens_per_step,
        "mtp_acceptance_rate": acceptance,
        "mtp_draft_to_main_ratio": mtp_draft_to_main_ratio,
        "output_sha256": output_sha256,
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
    for metric in MEDIAN_ONLY_METRICS:
        values = [r[metric] for r in rows if r.get(metric) is not None]
        if values:
            out[metric] = round(statistics.median(values), 2)
    for field in ("input_size", "output_size"):
        out[field] = rows[-1].get(field)
    output_hashes = {row.get("output_sha256") for row in rows if row.get("output_sha256")}
    out["output_consistent"] = len(output_hashes) <= 1 if output_hashes else None
    out["output_sha256"] = next(iter(output_hashes)) if len(output_hashes) == 1 else None
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
        return line + format_mtp(record)
    return (
        f"{line}, latency {record['latency']:.1f} ms/token | "
        f"1st token {ttft:.1f} ms, other tokens {tpot:.1f} ms/token | "
        f"prefill {record['prefill_throughput']:.1f} tok/s, "
        f"decode {record['decode_throughput']:.2f} tok/s, "
        f"e2e {record['e2e_throughput']:.1f} tok/s"
    ) + format_mtp(record)


def format_mtp(record: dict) -> str:
    """The MTP tail of an iteration line; empty string when the profile ran without it."""
    tokens_per_step = record.get("tokens_per_step")
    if tokens_per_step is None:
        return ""
    k = record.get("num_assistant_tokens")
    text = f" | MTP k={k} {tokens_per_step:.2f} tok/step" if k else (
        f" | {tokens_per_step:.2f} tok/step"
    )
    acceptance = record.get("mtp_acceptance_rate")
    if acceptance is not None:
        accepted = record.get("mtp_accepted_tokens")
        drafted = record.get("mtp_draft_tokens")
        counts = (
            f", {accepted}/{drafted} candidates"
            if isinstance(accepted, int) and isinstance(drafted, int) else ""
        )
        text += f" ({acceptance * 100:.0f}% accepted{counts})"
    # The draft head's cost, next to what it bought: a high acceptance is only a win while
    # proposing the candidates stays cheap relative to the main-model pass verifying them.
    ratio = record.get("mtp_draft_to_main_ratio")
    if isinstance(ratio, (int, float)):
        text += f", draft/main {ratio:.2f}x"
    return text
