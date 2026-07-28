<!--
Copyright (C) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Long-Context Benchmark

`components/llm/context_bench/` is a standalone diagnostic tool that answers two
questions for a candidate summarizer model at long context:

1. Can this machine prefill and decode it at all, inside its memory budget?
2. **Which OpenVINO configuration does it fastest?**

It benchmarks a matrix of named configuration *profiles* — KV cache precision, prefill
chunk size and scheduler cache size — and ranks them by measured TPOT, using TTFT as the
tie-breaker. Measurement follows the methodology of
[llm_bench](https://github.com/openvinotoolkit/openvino.genai/tree/master/tools/llm_bench):
one warm-up iteration excluded from every statistic, N measured iterations, and the
**median** reported with min/max so run-to-run spread stays visible.

It is **not** part of the runtime pipeline. Nothing in the app imports it, and it reads its
own model-specific configs —
[`config_qwen3.5_9b.yaml`](../../../components/llm/context_bench/config_qwen3.5_9b.yaml) and
[`config_qwen3.6_35b_a3b.yaml`](../../../components/llm/context_bench/config_qwen3.6_35b_a3b.yaml)
— never `smart-classroom/config.yaml`. Editing either has no effect on the application.

```
components/llm/context_bench/
  config_qwen3.5_9b.yaml       Qwen3.5-9B single 160K profile
  config_qwen3.6_35b_a3b.yaml  Qwen3.6-35B-A3B single 160K profile
  context_builder.py     synthetic transcript sized to an exact token count
  metrics.py             llm_bench-unit iteration records and their aggregation
  trial_runner.py        runs ONE (model, profile, context) case, in a subprocess
  benchmark.py           CLI orchestrator: run matrix, memory sampling, reporting
  setup_env.ps1          prepares the backend venv (one-time)
  run_benchmark.ps1      one-command launcher
```

> **Scope — capacity and speed, not answer quality.** Prompt content is irrelevant here;
> only token volume and the clock matter. This tool does not score whether the model
> *understood* the long context. Validate recall separately against real transcripts.

## Quick start

```powershell
# Prepares the venv on first use, then runs the full matrix
.\components\llm\context_bench\run_benchmark.ps1

# Check the run matrix without a model, a GPU, or OpenVINO installed
.\components\llm\context_bench\run_benchmark.ps1 --list-profiles

# One profile, one short context -- the fast way to confirm the plumbing works
.\components\llm\context_bench\run_benchmark.ps1 --profiles optimized-f16-32k --contexts 8000 --iterations 1

# Select the 35B model matrix
.\components\llm\context_bench\run_benchmark.ps1 --config components/llm/context_bench/config_qwen3.6_35b_a3b.yaml
```

Budget the time before starting a full run: at 160K a single Qwen3.5-9B iteration takes
roughly 245–300 s and a Qwen3.6-35B-A3B iteration roughly 400–500 s. Each model config
intentionally contains one selected profile. Use `--warmup 0 --iterations 1` for a quick
capacity check, then use the configured one warm-up and three measured iterations for the
final median.

Equivalent invocation if you already have the backend venv active — run from
`smart-classroom/` so relative model paths resolve:

```powershell
python -m components.llm.context_bench.benchmark
```

If a model's OpenVINO IR is missing, the tool prints the exact `optimum-cli export openvino`
command to produce it and moves on to the next model.

## What gets measured

Per iteration, in llm_bench's field names and units (milliseconds for latency, seconds for
time, tokens/second for throughput):

| Field | Meaning |
|---|---|
| `first_token_latency` | TTFT — one forward pass over the whole context |
| `other_tokens_avg_latency` | TPOT — post-TTFT time divided by `output_size - 1` |
| `latency` | `generation_time / output_size`, ms/token |
| `generation_time` | end-to-end seconds for the call |
| `tokenization_time`, `detokenization_time` | as reported by the runtime |
| `prefill_throughput` | `input_size / TTFT` |
| `decode_throughput` | llm_bench's 2nd-token rate, `1000 / TPOT` |
| `e2e_throughput` | `(input_size + output_size) / generation_time` |

Timings come from OpenVINO GenAI's own `perf_metrics` (the same source llm_bench reads)
wherever the runtime provides them, falling back per field to wall-clock timing around the
streamer callback.

**Ranking policy.** TPOT is primary because steady decode latency is the requested service
metric; lower is better. TTFT is the secondary key. Prefill and e2e throughput remain in the
report because at 160K TTFT dominates wall time and explains the user-visible wait.

**Why the median.** Before iterations existed, the same 160K configuration was measured at
247.0 s and 349.5 s on two separate single-shot runs. One sample cannot distinguish a
configuration difference from noise, which made every A/B conclusion unfalsifiable.

## Configuration

Each model-specific config has three sections.

**`model`** — provider, `device` (GPU/CPU), `weight_format`, and `models_base_path`.

**`benchmark`** — what to run and the acceptance budgets:

| Key | Notes |
|---|---|
| `models`, `context_tokens` | the run matrix, together with `profiles` |
| `output_tokens` | decode length per iteration; 64 is enough to measure TPOT |
| `warmup`, `iterations` | iteration 0 is the warm-up and never enters a statistic |
| `timeout_sec` | hard per-case wall clock; the only time limit |
| `max_system_memory_pct` | set to 100 on this PTL run so successful 160K trials are not rejected on host-RAM percentage |
| `gpu_memory_budget_gb` | **set this per machine** (see below) |
| `cache_dir` | set to a path to cache compiled blobs; cuts repeated load time for the 33 GB 35B export |
| `output_dir` | each run writes to a timestamped subdirectory |

**`profiles`** — the configurations benchmarked in order. `ov` are OpenVINO plugin
properties; `scheduler` are OpenVINO GenAI `SchedulerConfig` values.

The shipped cache pools are model-specific. At 160K, model-derived persistent KV is
2.53/4.93 GiB (u8/f16) for 9B and 1.61 GiB (u8) for 35B. The measured-stable 35B config
reserves 4 GiB, leaving the rest of the 59 GB shared-GPU budget for its 32.8 GB weights and
temporary prefill allocations. A fixed pool below the estimated persistent KV fails
before model loading. A completed case above 59 GB keeps its measurements but is marked
`gpu_memory_limit` and excluded from ranking.

`enable_prefix_caching` must stay `false`. The same prompt is reused across iterations (as
llm_bench does), so a warm prefix cache would make every iteration after the warm-up report a
TTFT that no first request will ever see.

### `gpu_memory_budget_gb` is a per-machine value

On Panther Lake the platform shares 59 GB of its 64 GB with the iGPU, but the driver's
`GPU_DEVICE_TOTAL_MEM_SIZE` reports only 33.62 GB. The driver value is recorded as
`gpu_budget_driver_gb` for reference; the configured value is what sizes `cache_size: auto`
and flags an overrun. Trusting the driver number previously caused a 160K case that really did
prefill and decode successfully (34.2 GB peak) to be reported as a failure with a max stable
context of 0.

### CLI overrides

`--models` `--contexts` `--profiles` `--iterations` `--warmup` `--output-tokens` `--device`
`--weight-format` `--output-dir` `--config` override config values for one run.

`--pipeline-config KEY=VALUE` and `--scheduler-config KEY=VALUE` add to or override every
profile's properties; `KEY=` with an empty value removes one. This exists so a one-off
measurement never requires an uncommitted config edit.

```powershell
# Try a property across the whole matrix without editing the config
.\components\llm\context_bench\run_benchmark.ps1 --pipeline-config CACHE_DIR=models/.ov_cache

# Drop a property the config sets
.\components\llm\context_bench\run_benchmark.ps1 --pipeline-config GPU_ENABLE_LARGE_ALLOCATIONS=
```

## Output

Each run writes to `<output_dir>/<YYYYMMDD-HHMMSS>/`, so runs never mix:

| File | Contents |
|---|---|
| `iterations.csv` | one row per iteration, warm-up included and flagged, with its own memory window |
| `summary.csv` | one row per (model, profile, context) with aggregated metrics |
| `summary.md` | ranked leaderboard per context, plus the exact config of the winner |
| `summary.json` | the same data structured, including hardware info |

The reports are rewritten after every case, not once at the end: this tool's job is to push
the box until something breaks, and it can break hard enough to take the orchestrator with it.
An incomplete run is marked as such rather than leaving a stale report that looks current.

## Statuses

| Status | Meaning |
|---|---|
| `ok` | measured successfully within budget |
| `memory_limit` | measured, but peak system RAM exceeded `max_system_memory_pct` — numbers kept, ranked out |
| `oom` | an explicit allocation failure, including the scheduler refusing an oversized `cache_size` |
| `gpu_abort` | the device aborted a queued command (OpenCL −14 and friends); at the ceiling this is usually memory, but a driver reset looks identical from inside the process |
| `unsupported` | this runtime build does not implement a requested property (e.g. int4 KV on an older OpenVINO) — skipped, not a hardware verdict |
| `timeout` | exceeded `timeout_sec` |
| `crashed` | the child died without reporting; native abort exit codes are decoded in the error text |
| `no_output` | the case ran without error but produced no measured iteration |
| `load_error`, `error` | anything else, with `stage_reached` naming the phase that was running |
| `missing_ir` | the model has not been exported yet; the error column holds the command to run |

`stage_reached` distinguishes failures that need different answers: `oom in prefill` is about
the size of one forward pass over the context, `oom in decode` is about the cache that pass
left behind.

## How it runs

Each case runs in a fresh spawned subprocess. The child loads the pipeline **once** and then
generates `warmup + iterations` times over the same prompt — reloading 33 GB of weights per
iteration would dominate the measurement, and the warm-up is what absorbs lazy weight paging
and first-run kernel compilation.

Memory is sampled by the **parent**: RAM and GPU counters are process-wide, so the parent sees
the child's footprint and its readings survive a child killed on a timeout — exactly the case
where memory matters most. The parent opens a fresh sampling window per iteration so a
warm-up's allocation spike is not charged to the measured iterations.

The child posts its result and then calls `os._exit(0)` **without** running OpenVINO's
teardown. Destroying a GPU pipeline that has just prefilled a very long context can throw an
`ov::Exception` out of a destructor, which is `std::terminate` — no Python `try/except` can
contain it, and it previously destroyed measurements that had already completed. Process exit
is the memory-reclamation boundary; the orchestrator waits for reclamation to land before the
next case takes its baseline.

Before anything is sent to the device, the prompt's token count is verified three ways —
configured, HuggingFace tokenizer, and the pipeline's own OpenVINO tokenizer IR must agree
exactly. A prompt that is 160,001 tokens where 160,000 was requested is a different
measurement.

## Tests

```powershell
cd smart-classroom
python -m unittest discover -s components/tests -p "test_context_bench_*.py"
```

| File | Covers |
|---|---|
| `test_context_bench_context_builder.py` | exact-token prompt construction |
| `test_context_bench_kv_estimate.py` | architecture-derived KV size and `cache_size: auto` |
| `test_context_bench_metrics.py` | llm_bench units, TPOT-first ranking, medians, `perf_metrics` fallback |
| `test_context_bench_trial_lifecycle.py` | child exit path, parent recovery, failure classification |

All four run without a GPU, a model, or the OpenVINO stack.

## Environment

`run_benchmark.ps1` creates and activates the backend venv (`../smartclassroom`, sibling of
`smart-classroom/`) via `setup_env.ps1` on first use. Running `benchmark.py` with the wrong
interpreter fails fast with the exact command to fix it rather than repeating the same import
error for every case.

If PowerShell blocks either script:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

## See also

- [`context_bench_design.md`](context_bench_design.md) — internal design, control
  flow, and the recorded tuning evidence behind the shipped profiles.
