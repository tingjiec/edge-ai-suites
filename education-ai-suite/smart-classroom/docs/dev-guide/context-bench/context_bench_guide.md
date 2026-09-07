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
chunk size, scheduler cache size, continuous batching vs the stateful pipeline, multi-token
prediction and its candidate count — and ranks
them by measured TPOT, using TTFT as the tie-breaker; the fastest TTFT is also called out on
its own, because at long context TTFT is what the user waits for. The 9B and 35B configs pin two
profiles that differ **only** in the pipeline — `stateful` and `paged_min` — so one run answers
which pipeline reaches the first token sooner; the Qwen3.8 config instead sweeps
[multi-token prediction](#multi-token-prediction-mtp). Add entries to `profiles` to A/B more in one
run. Measurement follows the methodology of
[llm_bench](https://github.com/openvinotoolkit/openvino.genai/tree/master/tools/llm_bench):
one warm-up iteration excluded from every statistic, N measured iterations, and the
**median** reported with min/max so run-to-run spread stays visible.

It is **not** part of the runtime pipeline. Nothing in the app imports it, and it reads its
own model-specific configs —
[`config_qwen3.5_9b.yaml`](../../../components/llm/context_bench/config_qwen3.5_9b.yaml),
[`config_qwen3.6_35b_a3b.yaml`](../../../components/llm/context_bench/config_qwen3.6_35b_a3b.yaml), and
[`config_qwen3.8_27b.yaml`](../../../components/llm/context_bench/config_qwen3.8_27b.yaml)
— never `smart-classroom/config.yaml`. Editing either has no effect on the application.

```
components/llm/context_bench/
  config_qwen3.5_9b.yaml       Qwen3.5-9B 160K: stateful vs paged_min
  config_qwen3.6_35b_a3b.yaml  Qwen3.6-35B-A3B 160K: stateful vs paged_min
  config_qwen3.8_27b.yaml      Qwen3.8-27B 8K: no-MTP baseline vs a num_assistant_tokens sweep
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
.\components\llm\context_bench\run_benchmark.ps1 --profiles stateful --contexts 8000 --iterations 1

# Select the 35B model matrix
.\components\llm\context_bench\run_benchmark.ps1 --config components/llm/context_bench/config_qwen3.6_35b_a3b.yaml
```

Historical 160K runs took roughly 245–300 s per Qwen3.5-9B iteration and 400–500 s per
Qwen3.6-35B-A3B iteration, all of them on paged attention with a fixed pool; the `stateful`
profile has never been measured on this box, so use those figures only for rough scheduling.
The 9B and 35B configs ship the same two profiles:

| Profile | Pipeline | `cache_size` | `max_num_batched_tokens` | KV precision |
|---|---|---:|---:|---|
| `stateful` | stateful (SDPA) — no `scheduler` section at all | n/a | n/a | 9B f16, 35B u8 |
| `paged_min` | continuous batching | 9B 8 GiB, 35B 4 GiB | 32,768 | same as `stateful` |

Both profiles in a config carry an identical `ov` block, so the pipeline is the only variable
and the two TTFTs are comparable — a test enforces this. Neither name is a performance claim:
`stateful` is unmeasured here, and `paged_min` re-uses the values with the strongest historical
support (8 GiB covers the 9B's ~4.93 GiB f16 cache; 32,768 is where the prefill-chunk sweep
saturated) rather than the wider chunk sizes an earlier config guessed at. Use
`--warmup 0 --iterations 1` (the shipped setting) for a first capacity and TTFT check, then
raise `iterations` for a median once both profiles are known to complete.

### `cache_size` and the stateful pipeline are alternatives, not a combination

`cache_size` is a property of OpenVINO GenAI's `SchedulerConfig`, and **passing any
`SchedulerConfig` selects continuous batching / paged attention** — there is no "stateful
pipeline with a bounded KV pool". On the stateful path the KV cache is model state that grows
with the context, and on genai 2026.4 none of the GPU plugin's 51 advertised properties bound
it; `KV_CACHE_PRECISION` changes bytes per token, not the total. That is why the pool cap and
the stateful pipeline are shipped as two profiles instead of one, and why a `cache_size`
written directly under a profile (next to `ov` / `scheduler`) is rejected with a message saying
where it belongs rather than silently ignored.

## Multi-Token Prediction (MTP)

Some models ship a small **draft head** alongside the main network — for Qwen3.8-27B it is
`openvino_mtp_model.xml` inside the same IR directory. OpenVINO GenAI can run it as
*self-speculative decoding*: the head proposes `k` candidate tokens, the main model verifies
all of them in one forward pass, and every candidate that matches is kept. Fewer main-model
passes for the same output means lower TPOT at identical output.

It is a **decode-side** lever only. Prefill still processes the whole context exactly once, so
expect TPOT to move and TTFT not to.

Turn it on with an `mtp` section on a profile:

```yaml
- name: mtp_k3
  ov:
    ATTENTION_BACKEND: PA
    KV_CACHE_PRECISION: f16
  scheduler:
    max_num_batched_tokens: 32768
    max_num_seqs: 1
    enable_prefix_caching: false
    cache_size: auto
  mtp:
    num_assistant_tokens: 3      # candidates offered per verification pass
    # enabled: true              # implied by the section existing
    # device: CPU                # defaults to the main model's device
```

Four constraints are OpenVINO GenAI's, not this tool's, and all four are checked **before**
the model loads rather than after a minute of loading a 14 GB export:

| Constraint | Why |
|---|---|
| the profile needs a `scheduler` section, and `ATTENTION_BACKEND: PA` | off NPU, genai only runs speculative decoding on paged attention. A profile with no `scheduler` is stateful/SDPA and cannot carry MTP |
| `num_assistant_tokens` ≥ 1 | genai asserts `> 0` |
| greedy decoding | set for every profile already (`do_sample = False`) — genai's MTP strategy rejects sampling |
| `assistant_confidence_threshold == 0` | genai's MTP path accepts a *static* candidate count only; a non-zero threshold selects the dynamic variant it refuses. This tool always sets 0 |

`max_num_batched_tokens` also has to be at least `num_assistant_tokens + 1`, the size of one
verification step. That is checked too.

Each request uses a fresh `openvino_genai.GenerationConfig`, matching the notebook rather than
mutating the model's sampling-enabled defaults. Warmup keeps the same prompt and MTP settings
but generates only four tokens; measured iterations use the configured output length. Since the
benchmark pre-renders the native chat template to hit the exact context size, it disables only
the pipeline's second template application.

### Reading the result

Two columns exist only for this, and TPOT alone cannot replace them:

| Field | Meaning |
|---|---|
| `tokens_per_step` | output tokens produced per main-model verification pass. **Exactly 1.00 when MTP is off** — that is the self-check that a baseline row really is a baseline |
| `mtp_acceptance_rate` | share of drafted candidates accepted, read only from GenAI's public `extended_perf_metrics`; unavailable on older runtimes rather than approximated |
| `mtp_draft_to_main_ratio` | draft-model inference time as a fraction of the main model's, from `SDPerModelsPerfMetrics.get_draft_to_main_inference_duration_ratio()`. This is the cost side that acceptance cannot show: a head accepting 60% of candidates is only a decode win while proposing them stays cheap relative to the pass that verifies them, and a ratio drifting toward 1 is where raising `k` stops paying even though `tok/step` still rises |
| `mtp_rejected_tokens` | drafted candidates the main model rejected, completing the accepted/draft/rejected picture; `None` unless MTP is on and the runtime reports it |

Acceptance is what says whether raising `k` still buys anything. Measured on Qwen3.8-27B (GPU,
the shipped 8K prompt, 64 output tokens) while reusing one loaded pipeline:

| Profile | TPOT ms | accepted |
|---|---:|---:|
| `mtp_k1` | 197.2 | 63.2% |
| `mtp_k2` | 156.8 | 61.8% |
| `mtp_k3` | **150.9** | **50.7%** |
| `mtp_k4` | 152.0 | 41.3% |
| `mtp_k6` | 161.3 | 30.2% |

**Acceptance is not the number to maximize — yield is.** What decode speed actually tracks is
`tokens_per_step`, the tokens committed per verification pass, which is `1 + k · acceptance(k)`.
Because acceptance falls as k rises, the two pull against each other, and the fastest profile is
often *not* the highest-acceptance one. Measured on this box at 8K: k=2 accepted **74%** for
2.44 tok/step at 85.5 ms TPOT, while k=4 accepted only **50%** but reached **2.90 tok/step** and
a **faster 81.4 ms** TPOT. Raising acceptance by lowering k would have made it slower.

So the console hint is **bidirectional**. Below 60% acceptance it suggests the *next smaller* k
(candidates are mostly wasted). At 70% or above it suggests the *next larger* k (the draft head
is agreeing often enough that a bigger candidate batch would likely commit more per pass) — this
is what points a healthy k=2 run toward the faster k=4. The flat middle needs no next
measurement. The MTP path requires a zero confidence threshold, so **k is the only lever on
acceptance** — dynamic thresholds are rejected — but the target of tuning it is yield, not
acceptance. Acceptance alone can mislead, though: read it next to
`mtp_draft_to_main_ratio`, the draft head's share of each verification pass. A profile can
hold acceptance steady while that ratio climbs, and once proposing the candidates costs a
large fraction of the pass that verifies them, a higher `k` stops paying even while
`tok/step` still rises. `summary.md` prints an **MTP speedup** line per
context comparing the best MTP profile against the best non-MTP one, so the ranking does not
have to be read as a subtraction.

Like the official notebook, a complete baseline + MTP sweep checks that deterministic greedy
outputs match exactly. The report warns when an MTP output hash differs from the baseline, since
a speedup that changes output is not a valid like-for-like result.

To sweep `k` without editing the config, `--mtp-tokens K` overrides it on every profile that
already enables MTP — and deliberately leaves the no-MTP baseline alone, since that row is the
denominator of the speedup.

```powershell
.\components\llm\context_bench\run_benchmark.ps1 `
  --config components/llm/context_bench/config_qwen3.8_27b.yaml --mtp-tokens 8
```

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
| `tokens_per_step` | output tokens per main-model verification pass — 1.00 unless MTP is on |
| `mtp_acceptance_rate` | share of drafted candidates accepted; `None` unless MTP is on |
| `mtp_draft_to_main_ratio` | draft-model inference time / main-model inference time; the cost signal for whether a higher `k` still pays |
| `mtp_rejected_tokens` | drafted candidates rejected by the main model; `None` unless MTP is on |

Timings come from OpenVINO GenAI's own `perf_metrics` (the same source llm_bench reads)
wherever the runtime provides them, falling back per field to wall-clock timing around the
streamer callback. Acceptance comes from `extended_perf_metrics.get_draft_acceptance_rate()`.
The verification-step count behind `tokens_per_step` has no public getter, so its compatibility
path reads `perf_metrics.raw_metrics.m_new_token_times`, which holds one entry per main-model
pass rather than per token.

**Ranking policy.** TPOT is primary because steady decode latency is the requested service
metric; lower is better. TTFT is the secondary key. Prefill and e2e throughput remain in the
report because at 160K TTFT dominates wall time and explains the user-visible wait — and since
the stateful-vs-paged difference is decided almost entirely in prefill, the report also names
the fastest TTFT separately from the TPOT winner rather than burying it in a tie-breaker.

**Why the median.** Before iterations existed, the same 160K configuration was measured at
247.0 s and 349.5 s on two separate single-shot runs. One sample cannot distinguish a
configuration difference from noise, which made every A/B conclusion unfalsifiable.

## Configuration

Each model-specific config has three sections.

**`model`** — provider, `device` (GPU/CPU), `weight_format`, and `models_base_path`.

`model_dirs` is optional and maps a model name to an explicit IR directory. Without it the
path is derived as `<models_base_path>/<provider>/<name>_<weight_format>`, which is what
`utils/ensure_model.py` produces — but a downloaded IR carries the publisher's naming
(`models/openvino/Qwen3.8-27B-int4-ov`), so `config_qwen3.8_27b.yaml` names it instead of
guessing at a second convention:

```yaml
model:
  model_dirs:
    Qwen3.8-27B: models/openvino/Qwen3.8-27B-int4-ov
```

**`benchmark`** — what to run and the acceptance budgets:

| Key | Notes |
|---|---|
| `models`, `context_tokens` | the run matrix, together with `profiles` |
| `output_tokens` | decode length per iteration; must be at least 2 to measure TPOT; 64 is the shipped value |
| `warmup`, `iterations` | iteration 0 is the warm-up and never enters a statistic |
| `timeout_sec` | maximum seconds **without a child progress event**, not a cap on the case; refreshed by every milestone. 1200 s for the 9B, 1800 s for the 35B — see the note below |
| `max_system_memory_pct` | set to 100 on this PTL run so successful 160K trials are not rejected on host-RAM percentage |
| `gpu_memory_budget_gb` | finite positive GiB budget; **set this per machine** (see below) |
| `cache_dir` | set to a path to cache compiled blobs; cuts repeated load time for the 33 GB 35B export |
| `output_dir` | each run writes to a timestamped subdirectory |

**`profiles`** — the configurations benchmarked in order. A profile is exactly
`{name, ov, scheduler, mtp}`; any other key is rejected. `ov` are OpenVINO plugin properties;
`scheduler` are OpenVINO GenAI `SchedulerConfig` values, and **omitting the `scheduler`
section is what selects the stateful pipeline** (reported as `pipeline_mode: stateful`).
`mtp` is `{enabled, num_assistant_tokens, device}` and is absent on a profile that does not
run speculative decoding — see [Multi-Token Prediction](#multi-token-prediction-mtp).

Inside a `scheduler`, three `cache_size` states are distinct. Omitted leaves pool sizing
entirely to OpenVINO — measurably a third configuration, not a neutral default: a 160K run
with a runtime-managed pool once went past 372 s with no first token. `auto` asks this tool to
derive a pool from the model architecture and configured GPU budget. A numeric value uses a
fixed GiB pool and is rejected before loading when it is below the estimated persistent KV
requirement. A completed case above the configured GPU budget keeps its measurements but is
marked `gpu_memory_limit` and excluded from ranking.

### `timeout_sec` is a TTFT ceiling in practice

Because the timer resets on every child event, the longest silent interval in a case is one
prefill — so `timeout_sec` effectively bounds TTFT, not the whole case. Historical 160K TTFTs
were 237–350 s for the 9B and 432 s for the 35B on paged attention, a **warm-up iteration is
slower than that** because it also pays first-run kernel compilation, and the `stateful`
profile's single-pass prefill has no measurement at all here. The configs therefore ship a
deliberately loose ceiling — 1200 s for the 9B, 1800 s for the 35B — because a `timeout`
verdict says nothing about TTFT except that it is above whatever ceiling was chosen. If a run
reports `timeout` with `stage_reached: prompt_built`, the box was still prefilling: raise
`timeout_sec` rather than concluding the context length failed.

`enable_prefix_caching` must stay `false`. The same prompt is reused across iterations (as
llm_bench does), so a warm prefix cache would make every iteration after the warm-up report a
TTFT that no first request will ever see. Configuration loading rejects profiles or CLI
overrides that enable it.

Configuration is validated before model loading: contexts and iteration counts must be
positive integers, warm-up must be non-negative, profile names must be unique, model names
must be non-empty, timeout must be positive, profiles must carry no key outside
`{name, ov, scheduler, mtp}`, and profile `ov` / `scheduler` values must be mappings.
`enable_prefix_caching`, when present, must be the boolean `false`. An `mtp` section must use
only `{enabled, num_assistant_tokens, device}`, `num_assistant_tokens` must be an integer ≥ 1,
and `max_num_batched_tokens` must leave room for one whole verification step. Invalid values
fail once with an actionable message instead of failing every case.

A fixed `cache_size` is checked against the estimated cache for **every** context in the
matrix as soon as the model's `config.json` can be read, before the first case of that model
loads. On a hybrid model that estimate includes one linear-attention reservation per
`max_num_seqs`, so an 8 GiB pool at the genai default of 256 is rejected in the first second
rather than after the earlier cases have already spent hours.

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
`--mtp-tokens K` sweeps the candidate count on the profiles that already enable MTP.

`--pipeline-config KEY=VALUE` and `--scheduler-config KEY=VALUE` add to or override every
profile's properties; `KEY=` with an empty value removes one. This exists so a one-off
measurement never requires an uncommitted config edit. Note that `--scheduler-config` applies
to *every* profile, so it moves the `stateful` profile onto paged attention — a different
pipeline, not a tweak of the same one. The tool prints a warning when that happens; to sweep a
scheduler value, add `--profiles paged_min`.

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
| `iterations.csv` | one row per iteration, warm-up included and flagged, with its own peak and mean memory window |
| `summary.csv` | one row per (model, profile, context) with aggregated metrics, including `pipeline_mode`, `cache_size_gb`, `mtp` and `num_assistant_tokens` |
| `summary.md` | ranked leaderboard per context with `Pipeline` and `MTP` columns, the fastest-TTFT call-out, the MTP speedup line, and the exact config of the TPOT winner |
| `summary.json` | the same data structured, including hardware info |

The TPOT ranking and the fastest-TTFT line can name different profiles — that is the point of
printing both. At 160K, TTFT is ~96% of the wall clock, so a profile that wins on TPOT while
losing 100 s on first token is not the one to ship.

### Memory is reported as peak / mean

Both are needed and they answer different questions:

| Column | Covers | Answers |
|---|---|---|
| `peak_ram_gb` / `peak_gpu_gb` | the whole case, model load included | whether the box can run this configuration **at all**. Set by the transient prefill workspace, and it is what `gpu_budget_exceeded` is judged on |
| `mean_ram_gb` / `mean_gpu_gb` | the measured iterations only — no load, no warm-up | what the configuration **costs while it runs**, which is the figure to use when the same budget also has to hold the rest of the application |

The mean is time-weighted, not a flat average of samples, so it does not move when the
sampling interval changes and is not skewed by the sampler's own jitter. Case-level means are
the median across the measured iterations, like every other aggregate here.

A large peak/mean gap means a spiky workload, not a cheap one: a case peaking at 26 GB but
averaging 19 GB still needs 26 GB of headroom to start.

The reports are rewritten after every case, not once at the end: this tool's job is to push
the box until something breaks, and it can break hard enough to take the orchestrator with it.
An incomplete run is marked as such rather than leaving a stale report that looks current.

## Statuses

| Status | Meaning |
|---|---|
| `ok` | measured successfully within budget |
| `memory_limit` | measured, but peak system RAM exceeded `max_system_memory_pct` — numbers kept, ranked out |
| `gpu_memory_limit` | measured, but peak GPU memory exceeded `gpu_memory_budget_gb` — numbers kept, ranked out |
| `measurement_error` | generation completed, but RAM or GPU budget telemetry was unavailable — timing kept, ranked out |
| `oom` | an explicit allocation failure, including the scheduler refusing an oversized `cache_size` |
| `gpu_abort` | the device aborted a queued command (OpenCL −14 and friends); at the ceiling this is usually memory, but a driver reset looks identical from inside the process |
| `unsupported` | this runtime build does not implement a requested property (e.g. int4 KV on an older OpenVINO) — skipped, not a hardware verdict |
| `timeout` | no child progress event arrived within `timeout_sec` |
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

Memory is sampled by the **parent**: RAM and GPU counters are system-wide, so the parent sees
the child's footprint and its readings survive a child killed on a timeout — exactly the case
where memory matters most. The parent opens a fresh sampling window per iteration so a
warm-up's allocation spike is not charged to the measured iterations.
Unavailable counters remain unavailable rather than becoming zero. A completed generation
without both peak RAM percentage and peak GPU usage is marked `measurement_error`; otherwise
the tool could claim that a profile passed a budget it never measured.

The child emits a `prefilled` event on the first streamed token. This refreshes the progress
timeout and lets the parent report a later native abort as a decode failure even when the
child dies before it can send its final `done` record.

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
| `test_context_bench_kv_estimate.py` | architecture-derived KV size, the per-sequence linear-attention reservation `max_num_seqs` controls, `cache_size: auto`, the MTP draft head's own KV surcharge, and the shipped configs' invariants (the 9B/35B pair differ only in the pipeline; the Qwen3.8 matrix differs only in MTP) |
| `test_context_bench_metrics.py` | llm_bench units, TPOT-first ranking, the TTFT ranking, `pipeline_mode` recording, `mtp` profile resolution, the MTP yield/acceptance arithmetic, medians, the time-weighted mean occupancy, `perf_metrics` fallback |
| `test_context_bench_trial_lifecycle.py` | child exit path, parent recovery, failure classification, MTP rejected before the model loads |

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
  flow, current profile contract, and historical tuning evidence.
