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
its own, because at long context TTFT is what the user waits for. The 35B config compares a
`paged_min` baseline with external dFlash drafts at k=3, 5, 7, and 15 on math, code, and
summary workloads; the Qwen3.8 config sweeps
[multi-token prediction](#multi-token-prediction-mtp). Add entries to `profiles` to A/B more in one
run. Measurement follows the methodology of
[llm_bench](https://github.com/openvinotoolkit/openvino.genai/tree/master/tools/llm_bench):
one warm-up iteration excluded from every statistic, N measured iterations, and the
**median** reported with min/max so run-to-run spread stays visible.

It is **not** part of the runtime pipeline. Nothing in the app imports it, and it reads its
own model-specific configs —
[`config_qwen3.6_35b.yaml`](../../../components/llm/context_bench/config_qwen3.6_35b.yaml) and
[`config_qwen3.8_27b.yaml`](../../../components/llm/context_bench/config_qwen3.8_27b.yaml)
— never `smart-classroom/config.yaml`. Editing either has no effect on the application.

**One config per model, one report per run.** Each config carries the throughput matrix *and*
the accuracy suites, and every run writes a single `report.txt` — TTFT, TPOT, throughput,
RAM/GPU, and (with `--accuracy`) every accuracy metric and every individual probe — which is
also printed to the console when the run ends.

```
components/llm/context_bench/
  config_qwen3.6_35b.yaml  Qwen3.6-35B-A3B: paged baseline vs dFlash sweep, + accuracy suites
  config_qwen3.8_27b.yaml  Qwen3.8-27B: no-MTP baseline vs a k sweep, + accuracy suites
  context_builder.py     synthetic transcript sized to an exact token count (+ probe placement)
  tasks.py               the RULER + generation task registry: what is planted and what is asked
  scoring.py             pure-Python metrics: SQuAD, RULER, ROUGE/chrF, FDT/SDT
  accuracy.py            probe sets, score dispatch and aggregation (opt-in --accuracy)
  wwb_adapter.py         optional bridge to who_what_benchmark for embedding similarity
  metrics.py             llm_bench-unit iteration records and their aggregation
  trial_runner.py        runs ONE (model, profile, context) case, in a subprocess
  benchmark.py           CLI orchestrator: run matrix, memory sampling, reporting
  setup_env.ps1          prepares the backend venv (one-time)
  run_benchmark.ps1      one-command launcher
```

> **Scope — the default run measures capacity and speed, not answer quality.** Prompt content
> is not scored on the throughput path, but it affects draft acceptance and output length. To score
> whether the model *understood* the long context, use the opt-in
> [`--accuracy` mode](#long-context-accuracy-ruler--wwb-fidelity), which runs RULER's
> retrieval probes and a who_what_benchmark-style fidelity comparison. The two paths never mix:
> with `--accuracy` off, everything else here is unchanged.

## Quick start

```powershell
# Prepares the venv on first use, then runs the full matrix
.\components\llm\context_bench\run_benchmark.ps1

# Check the run matrix without a model, a GPU, or OpenVINO installed
.\components\llm\context_bench\run_benchmark.ps1 --list-profiles

# One profile, one short context -- the fast way to confirm the plumbing works
.\components\llm\context_bench\run_benchmark.ps1 --profiles paged_min --contexts 8000 --iterations 1

# Select the 35B model matrix
.\components\llm\context_bench\run_benchmark.ps1 --config components/llm/context_bench/config_qwen3.6_35b.yaml
```

The Qwen3.6 profiles share f16 KV precision, a 65,536-token scheduler batch ceiling,
one live sequence, 17 linear-attention rows, and disabled prefix caching. The runtime
manages `cache_size`. Defaults are one full-budget warmup and three measured iterations;
`--warmup 0 --iterations 1` is a smoke test, not a steady-state performance comparison.

`benchmark.iteration_gap_sec` (CLI `--iteration-gap`) idles before each measured
generation, outside the timer and the memory window. It exists because the iGPU boosts only
for the first few seconds of load: at 8K, back-to-back `paged_min` generations decode at
~28-31 ms/token with ~4 s TTFT, while every generation after 30 s idle runs at 24.6 ms/token
(~40 tok/s) with 3.0 s TTFT. The Qwen3.6 config sets 30 s, which reports what an occasional
classroom request sees; `0` (the default elsewhere) measures sustained back-to-back load.
The report header states which one a run used, so do not compare figures across the two.

## dFlash Speculative Decoding

The [Qwen3.6 dFlash draft](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash)
is an external block-diffusion model, not the target's MTP head. The
[upstream implementation](https://github.com/z-lab/dflash/blob/main/dflash/model.py)
selects post-decoder hidden states using `hidden_states[layer_id + 1]`, predicts a block,
and lets the target accept a prefix and produce a correction or bonus token.
The local export's block size is 16 including the seed: the corresponding maximum
proposal is `num_assistant_tokens: 15`. Smaller k values can be faster on this GPU.

The benchmark calls `draft_model(dflash.model, device)` and passes it to the paged
pipeline. `num_assistant_tokens` belongs to `GenerationConfig`, not GPU plugin properties.
For the installed hybrid-model runtime, one live sequence plus a k-token proposal needs
at least `1 + (k + 1)` linear-attention rows. The shared value 17 covers the entire sweep.
Do not use a stateful or differently quantized profile as the dFlash baseline.

```powershell
# Repeated-prompt speed comparison, including the no-draft baseline, on every configured task
.\components\llm\context_bench\run_benchmark.ps1 --profiles paged_min dflash_k3 dflash_k5 dflash_k7 dflash_k15

# One domain only
.\components\llm\context_bench\run_benchmark.ps1 --profiles paged_min dflash_k5 dflash_k7 --throughput-task math

# Quick check against the published DFlash numbers (~15 min): no transcript, k=7
.\components\llm\context_bench\run_benchmark.ps1 --contexts 0 --profiles paged_min dflash_k7 --throughput-task code math chat

# Natural-answer accuracy and timings; reduce probe count while investigating
.\components\llm\context_bench\run_benchmark.ps1 --profiles paged_min dflash_k3 dflash_k7 --accuracy --accuracy-tasks niah_single fact_sheet --accuracy-depths 0.5 --accuracy-samples 1
```

Both throughput and accuracy respect EOS. `output_tokens` is a ceiling, not a guaranteed
answer length. Forcing tokens after EOS changed the measured acceptance and produced
an artificial slowdown in testing. Warmup uses the full decode budget instead of four
tokens, which may not exercise a complete speculative window. A two-token smoke test
does not establish steady-state speculative performance.

Interpret the report as follows:

- Weights is the precision recorded in the loaded IR's NNCF rt_info (for example
  `int4_asym g64 backup int8_sym`), not the `model.weight_format` label; a label that
  disagrees with the IR is warned about at startup. The Speculative cell adds the draft's
  precision (`draft int4_asym g128`), and the case banner prints both records in full.
- Decode tok/s is `1000 / median TPOT`; TTFT and total generation time are separate.
- E2E tok/s includes input tokens as well as generated tokens; it is not decode throughput.
- Output tok shows actual lengths, or their range for mixed accuracy probes.
- `AL` is the **acceptance length**: decode tokens committed per target verification pass,
  `(output - 1) / (steps - 1)`, meaning the accepted draft tokens plus the target's own bonus
  token. The first recorded step is the prefill pass, which emits one token and verifies
  nothing. AL is the number to read: the decode speedup is `AL / pass cost`, so AL tracks
  speedup in a way the accepted % cannot.
- dFlash acceptance is the runtime's own `accepted / drafted`. The dFlash path leaves
  `get_num_draft_tokens()` at 0 and the acceptance rate NaN, so the drafted count is read
  from `get_num_draft_processed_tokens()` (k per pass). Acceptance marked `estimated` is
  used only when no counter is available and comes from `(AL - 1) / k`. The rate falls with
  k by construction, so it is printed for context only; never rank windows by it.
- The speedup line also prints what each verification pass cost, as a multiple of a baseline
  decode step: the speedup is `AL / pass cost`. On the Qwen3.6 MoE a pass costs
  ~2.2x (k=3) to ~3.7x (k=15) a single-token step because every verified token routes to
  its own experts, so a summary's AL of ~3.4 nets only a modest gain and k=15 is a loss.
- The **SPECULATIVE WINDOW SWEEP** table, one per context, lists every speculative case
  against its matched baseline: k, AL, accepted %, steady pass ms, pass cost (which is also
  the break-even AL), and the end-to-end and steady-state decode speedups. It names the best
  window per task and, when several tasks ran, the domain spread between them.
- `draft/main` divides total draft time by total target time *including prefill*, so on a
  long prompt it understates the draft's share of each decode pass (~0.05x reported versus
  ~20% of a k=7 pass measured).
- Speedup and greedy-output checks require matching model, context, device, weight format,
  OpenVINO properties, and scheduler. Accuracy timing medians pool different tasks;
  use repeated-prompt runs for throughput comparisons.

On the 2026-10-03 local 8K run (45-token natural answers, three measured iterations),
baseline decoded at 29.67 tok/s, k=3 at 35.81, k=7 at 32.04, and k=15 at 22.85.
All natural answers matched exactly. The best decode gain was 1.21x, but total latency
did not improve because prefill dominated. One retrieval and one fact-sheet probe per
baseline/k3/k7 profile also matched exactly, with full recall and fact coverage. This is
a small regression check, not broad model certification or a guarantee for other workloads.
Embedding similarity was not measured because `whowhatbench` was unavailable.

### Domain, window size, and what raises throughput

Speculative decoding on a MoE target is tricky. The draft proposes its whole block in one
parallel pass, so a wider window is nearly free to draft. The target, however, verifies all
k + 1 tokens every pass, and each one reads its own experts. A wider window pays only while
its AL grows faster than its pass cost. Where that happens depends on the *domain*: code and
math answers are far more predictable for the draft than open-ended chat or summaries. That is
why the Qwen3.6 config runs `code`, `math`, `chat`, and `summary_2s` and reports a best window
per task. Judge the integration on code and math against the published reference (see
[Comparing with the published DFlash results](#comparing-with-the-published-dflash-results)).
Judge the classroom gain on `summary_2s` or `classroom_summary`.

Acceptance is set by the workload and by this draft, not by the benchmark's settings. Same
pipeline and draft, runtime-reported accepted/drafted counts under greedy decoding:

| Workload | k=3 | k=7 | k=15 | Best AL |
|---|---:|---:|---:|---:|
| 8K two-sentence summary (`summary_2s`, the default) | 71% | 34% | 16% | 3.4 |
| 8K app summary (`classroom_summary`, 479 tokens) | 59% | 32% | 15% | 3.25 |
| 8K summary, thinking enabled | 72% | 41% | 21% | 4.05 |
| Math, no context | 87% | 70% | 38% | 6.4 |
| The same math question after an 8K transcript | 85% | 61% | 34% | 5.9 |

The integration works: on math, k=7 decodes at ~10 ms/token against the ~22.6 ms baseline
(2.1x). Summaries of this transcript are simply hard for the draft to predict. Because
acceptance is `(AL - 1) / k`, a saturated AL makes every larger k read as a lower rate, so
rank windows by AL against pass cost (the sweep table), not by acceptance.

The console hint follows the same rule. A dFlash window that verifies more than twice its AL
(for example, k=15 reaching AL 3.4) is pointed at `k = ceil(AL)`. A window whose draft slots
fill at least 80% of the time is pointed at `2k`. Anything in between needs no further
measurement.

Measured 2026-10-08 with the shipped config: int4-head draft, 512-token ceiling, 3 iterations
after 30 s idle, greedy. Each cell is AL and decode speedup over `paged_min`. Baseline decode
was 43.7-45.2 tok/s without a transcript and 40.8-41.8 tok/s at 8K.

| Task | Context | k=3 | k=5 | k=7 | k=15 | Best window |
|---|---|---|---|---|---|---|
| `code` | none | 3.65, 1.85x | 4.96, 2.27x | 5.94, **2.40x** | 7.20, 2.03x | k=7 (105 t/s) |
| `code` | 8K | 3.70, 1.69x | 5.11, 2.08x | 6.08, **2.22x** | 7.74, 2.07x | k=7 |
| `math` | none | 3.52, 1.78x | 4.82, 2.14x | 5.68, **2.26x** | 6.91, 1.96x | k=7 (99 t/s) |
| `math` | 8K | 3.62, 1.66x | 4.87, 1.97x | 5.74, **2.07x** | 7.20, 1.92x | k=7 |
| `chat` | none | 2.59, 1.34x | 2.85, 1.32x | 3.30, **1.38x** | 3.17, 0.94x | k=7 (61 t/s) |
| `chat` | 8K | 2.65, 1.23x | 3.01, **1.25x** | 3.02, 1.14x | 3.01, 0.77x | k=5 |
| `summary_2s` | 8K | 3.14, **1.32x** | 3.14, 1.19x | 3.38, 1.19x | 3.38, 0.90x | k=3 |

Pass costs were 2.0-2.2x (k=3), 2.2-2.4x (k=5), 2.5-2.7x (k=7), and 3.5-3.7x (k=15) a
baseline step; a pass is cheaper without a transcript. AL divided by pass cost predicts the
steady speedup to within about 0.01 in every row. k=15 has the highest AL on math and code,
yet loses to k=7 because its pass costs more. Chat and summaries stop gaining AL beyond
k=5-7 and k=3, so narrower windows win there. The best k is a property of the domain, not
of the draft alone.

### Comparing with the published DFlash results

The reference is the Hugging Face blog *Accelerating Qwen3.6 on Intel Core Ultra Series 3 with
DFlash* (2026-07-30). Its setup matches this config: Qwen3.6-35B-A3B with the z-lab DFlash
draft, both int4 W4A16, an Arc B390 iGPU (Core Ultra X7 368H, 64 GB), greedy decoding, and
k=7. The differences are the runtime (OpenVINO 2026.3) and the prompts (whole datasets with no
long-context prefix). The config's `speculative_reference` section holds its figures, and every
window sweep prints them beside the measured AL and speedup at the same k:

| Domain | Blog dataset | Blog AL / speedup (t/s) | No transcript, k=7 | 8K transcript, k=7 |
|---|---|---|---|---|
| code | HumanEval | 6.4 / 2.2x (89.8) | 5.94 / 2.40x (105.4) | 6.08 / 2.22x |
| math | GSM8K | 5.0 / 1.6x (68.5) | 5.68 / 2.26x (99.0) | 5.74 / 2.07x |
| chat | MT-Bench | 4.0 / 1.3x (54.7) | 3.30 / 1.38x (60.6) | 3.02 / 1.14x |

The blog's baseline is 41 t/s; this box measured 43.9 t/s without a transcript. Every domain
reaches or beats the published speedup at k=7, and the domain ordering is the same: code,
then math, then chat. The per-domain gaps in AL are prompt effects. This benchmark uses one
prompt per domain, not a dataset. The `math` prompt is easier to draft than GSM8K. The
`chat` prompt has one item per MT-Bench category, but MT-Bench answers run longer and include
second turns. An earlier `chat` prompt with only open-ended items (email, advice, roleplay)
reached just AL 2.4, which shows how much the mix matters. Treat the reference as a sanity
band, not a pass/fail target. The blog's dense-model results (Qwen3.6-27B, Qwen3.5-9B) are not
reproduced here because no DFlash drafts for them are available locally.

**Greedy forks.** Without a transcript, several speculative outputs differ from the baseline's
greedy output. Each profile repeats its own output exactly, and the fork position moves with k.
The report's output check prints every fork with the text on either side. Every fork measured
so far is a near-equivalent token: "our Science Fair" vs "our school science fair", "T-Rex" vs
"Tyrannosaurus Rex", and "Solve for velocity" vs "Solve for $v$" at token 506 of 512. The
target scores k + 1 tokens in one verification pass, on a different kernel path than a
one-token step, so near-ties can flip. Greedy speculative decoding is lossless only up to that
floating-point difference. A late fork leaves the speedup comparable, an early fork means it
compares two different answers, and a garbled fork would be a pipeline bug. At 8K, the `code`,
`math`, and `summary_2s` outputs matched the baseline exactly.

Each of these was A/B-tested on an Arc B390 iGPU and ruled out as a cause:
- the hidden-state layer mapping (shifting it one layer earlier lowers acceptance);
- context length (the same math question keeps its acceptance after an 8K transcript);
- the draft's 4096-token sliding window (GenAI honours it: a 64-token window collapses
  acceptance);
- the draft's KV cache precision and activation precision (identical counts);
- an int8 target (no acceptance gain, 40% slower passes).

On the throughput side, `DYNAMIC_QUANTIZATION_GROUP_SIZE` and the host-priority properties are
within run-to-run noise, and the GPU plugin's MoE and PagedAttention route options
(`GPU_MOE_*`, `GPU_PA_MIXED_ROUTE_MODE`) exist only in debug builds; this release rejects them.

The cost side is mostly physical. A steady k=3 pass costs ~52 ms against a 23.5 ms baseline
step, because verifying 4 tokens on the MoE reads the union of their experts. One part is
removable: GenAI grafts the target's 509 MB int8 lm_head onto the draft and streams it every
pass. `dflash_draft_head.py` writes a copy of the draft that carries its own int4 head
instead, which GenAI then uses without the graft:

```powershell
python -m components.llm.context_bench.dflash_draft_head `
    --target models/openvino/Qwen3.6-35B-A3B_int4 `
    --draft models/openvino/qwen3.6-35b-a3b-dflash-int4-ov `
    --output models/openvino/qwen3.6-35b-a3b-dflash-int4-ov-int4head
```

Point the dflash profiles' `model` at the output. Results on `classroom_summary` at 8K:

| Draft | k=3 TPOT | k=7 TPOT | Acceptance |
|---|---|---|---|
| Stock | 20.4 ms (1.17x) | 20.9 ms (1.14x) | 59% / 32% |
| Int4 head | 18.6 ms (1.28x) | 19.6 ms (1.22x) | 59% / 32% |

The baseline is 23.9 ms/token. `--bits 8` writes an exact copy of the grafted head; it gives
identical results to the stock draft and serves as the control. The derived draft is tied to
the target export it was built from, so rebuild it after re-exporting the target. The
remaining untested lever for acceptance is an int8 or fp16 re-export of the bf16 draft
checkpoint; the local draft is int4 on all layers, and the export needs network access.

### Throughput task and steady-state reporting

`benchmark.throughput_task` (or `--throughput-task`) picks the prompt the speed iterations
decode. It takes one name or a list. Each profile then runs once per task, and each task gets
its own speed table, speedup line, and best window:
- `summary_2s`, the default, is the built-in two-sentence summary, about 45 tokens.
- `code` asks for a typed Python module of five lab functions with unittest cases
  (HumanEval-like).
- `math` asks for six worked physics, arithmetic, and algebra problems, each ending with
  `Answer: <number>` (GSM8K-like).
- `chat` asks four open-ended assistant requests: an email, advice, an explanation for a
  child, and a roleplay (MT-Bench-like).
- The three domain tasks run after the transcript at a positive context, changing only the
  task after it. At `context_tokens: 0` they run alone, as a dataset prompt would; the summary
  tasks and the accuracy suites skip context 0. Give the domain tasks `output_tokens: 512` or
  more so the first decode pass does not dominate.
- `classroom_summary` is the app's own summarizer request. It uses the configured language and
  mode's system prompt from `config.yaml` and puts `/no_think` before the transcript for Qwen3,
  exactly as `summarizer_component` does. Raise `output_tokens` (for example to 1024) so the
  Markdown summary can finish.

Every record also carries `first_decode_ms`, `steady_pass_ms` and `steady_tpot`, taken from
the runtime's per-pass inference durations. The first decode pass carries one-time
per-request work: about 160 ms for dFlash and 64 ms for the baseline at 8K. On a 45-token
answer that pass is a large share of decode, so the speedup line reports the steady-state
rate next to the end-to-end TPOT.

Older local target exports may lack `hidden_states_decoder_layers` RT info. The compatibility
helper backfills this annotation in the target XML using final residual-node locators;
it does not change weight files. This is an artifact modification, not a GPU tuning setting.
Use compatible target/draft exports and verify accuracy after export or runtime changes.

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

It primarily accelerates decoding. Prefill still processes the whole context, and extra
speculative setup can affect TTFT; compare the measured values rather than assuming equality.

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
mutating the model's sampling-enabled defaults. Warmup keeps the same prompt, speculative settings,
and decode budget as measured throughput iterations. Both honor EOS. Since the
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
`tok/step` still rises. The report prints an **MTP speedup** line per
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

## Long-Context Accuracy (RULER + WWB fidelity)

The throughput path proves the box can *run* a context length; it says nothing about whether the
model still *used* it correctly. `--accuracy` measures that. This is the axis where
**quantization and speculative decoding silently regress**: a u8/int4 KV cache or an MTP draft
head is a decode-side speedup that is *supposed* to preserve the output, and accuracy mode is
what checks it did.

```powershell
.\components\llm\context_bench\run_benchmark.ps1 `
  --config components/llm/context_bench/config_qwen3.8_27b.yaml --accuracy
```

The shipped accuracy config runs **one** paged profile, so what it reports is the model's own
accuracy on this box. Take that measurement first: a retrieval rate for an MTP or u8-KV profile
means nothing without knowing what the unaccelerated baseline scored on the same probes. To turn
it into a precision-loss sweep, add profiles to that config and keep `baseline_profile` pointing
at the plain one — every added profile is then reported as a signed delta against it. With a
single profile the delta and fidelity columns are self-comparisons and read 0 / 1.00 by
construction: correct, and not a result.

Two suites, because "still correct" has two halves that fail independently. A profile can return
every planted code and write visibly worse prose; it can also summarize fluently while having
lost the middle of the context.

### Suite 1 — retrieval (RULER)

Synthetic probes with known ground truth, following
[RULER](https://github.com/NVIDIA/RULER) (Hsieh et al., COLM 2024), which generalizes
[Needle-in-a-Haystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) into three
behaviours that break separately:

| Task | Behaviour | What it plants |
|---|---|---|
| `niah_single` | retrieval | one high-entropy code at one depth — the classic NIAH probe |
| `niah_multikey` | retrieval under distraction | the target code plus decoy codes under other keys |
| `niah_multivalue` | exhaustive retrieval | several values of **one** key, spread through the context |
| `niah_multiquery` | exhaustive retrieval | one value for **each** of several keys |
| `vt` | multi-hop tracing | chains of variable assignments that must be followed to the end |
| `cwe` | aggregation | a word list whose 10 most frequent words must be **counted** out |
| `fwe` | aggregation | the same, with Zeta-distributed frequencies (no clean split) |
| `qa` | comprehension | a planted paragraph answered in natural language |

Scored with RULER's own rules, all pure Python — no reference model, no embedding dependency,
unit-testable without a GPU:

| Metric | Meaning |
|---|---|
| `recall` | the **primary** score, and the task's own RULER scorer: the share of ground-truth items found, or intersection-over-union for `cwe`/`fwe` (which must punish over-answering), or any-acceptable-answer for `qa` |
| `exact_match` | SQuAD exact match — the answer is the ground truth and nothing else |
| `token_f1` | SQuAD token-level F1, the graded fallback |
| `distractor_rate` | share of planted **decoys** that wrongly appeared — lower is better. A profile losing precision does not stop answering, it starts answering with the wrong key's value, and recall alone reads that as a plain miss |

`niah_single`, `niah_multikey` and `qa` are swept across `depths` (0.0 = very top of the
transcript, the hardest); the rest place their needles by construction and run `samples` probes
each. Every `(task, depth, sample)` gets **distinct** planted values seeded from the config, so a
profile cannot score a hit by memorizing one answer.

> **Probe prompts are sized to within a few tokens of the configured context, not exactly to
> it** — the throughput path's `configured == HF == OpenVINO` check still demands exactness,
> and the two differ on purpose. Each planted insert adds two tokenizer seams (a `vt` probe
> has 34), and the size correction moves one number, the filler budget, against the sum of all
> that drift; it can orbit the target by a token forever without landing on it. Throughput
> *divides* by the token count, so a token matters there. A retrieval verdict does not: "did
> the model find the code planted at depth 0.25 of an 8,000-token context" is the same
> question at 8,001. Measured with the Qwen3.8 tokenizer, every probe lands within 2 tokens.
> The count actually used is what the report records and what any per-token figure divides
> by, so nothing is reported against a number no forward pass saw.
>
> A probe whose prompt cannot be built at all is **skipped with a note** rather than failing
> the case — the model is loaded and working, and a partial accuracy result is worth far more
> than losing the probes that already ran.

> RULER draws its QA tasks from SQuAD and HotpotQA. Those datasets are not shipped here, so `qa`
> plants a generated paragraph instead: the task shape is the same, but a score from it is not
> comparable with a published RULER QA number.

### Suite 2 — generation fidelity (who_what_benchmark)

The model does the **real** classroom task (`summary`: summarize, explain, list, write quiz
questions; `fact_sheet`: a long-form report over planted facts) at a realistic answer length, and
its answer is compared with the answer the **baseline profile** gave to the identical prompt.
That is [WWB](https://github.com/openvinotoolkit/openvino.genai/blob/master/tools/who_what_benchmark/README.md)'s
framing — how far did the optimized model drift from the reference — and it replaces the old
binary `output_sha256` check with a graded one.

Three families of number, kept side by side because each is blind to something:

| Metric | Family | Meaning |
|---|---|---|
| `similarity` | semantic | WWB's sentence-embedding cosine against the baseline answer. A paraphrase scores ~1.0 |
| `rouge1` / `rouge2` / `rouge_l` | lexical | n-gram and LCS F-measure. Falls when the wording changes even if the meaning did not |
| `chrf` | lexical | character n-gram F-score, so a near-miss still scores |
| `fdt` / `sdt_norm` | token-exact | [divergent token metrics](https://arxiv.org/abs/2311.01544): how many tokens the two agreed on before the first difference, and the share of the answer that differs |
| `identical` | token-exact | byte-for-byte match — the old `output_sha256` answer |
| `fact_coverage` | ground truth | share of the facts planted for `fact_sheet` that the answer named. The one column here scored against real ground truth rather than another model's output |

Reporting only `similarity` would call a visibly reworded answer "lossless"; reporting only
`identical` would call a legitimate tie-break "a regression".

> **`similarity` needs who_what_benchmark, which is optional and not required to run.** Every
> other metric is computed natively. When WWB is absent the column reports `--` and the report
> says why; nothing else changes and no run fails. To get the column:
>
> ```powershell
> pip install "whowhatbench @ git+https://github.com/openvinotoolkit/openvino.genai.git#subdirectory=tools/who_what_benchmark"
> ```
>
> Set `accuracy.embedding_model: null` to switch it off deliberately.

### Configuring an accuracy run

Add an `accuracy` section to a config; it is read only when `--accuracy` is passed. Both suites
are optional and each defaults to its full task list, so `accuracy: {suites: [retrieval]}` is a
complete RULER run.

```yaml
accuracy:
  suites: [retrieval, generation]
  baseline_profile: paged_min     # every profile is reported against this one
  seed: 20260917                  # fixes the planted values, so a re-run reproduces them
  embedding_model: sentence-transformers/all-mpnet-base-v2   # null to skip `similarity`

  retrieval:
    tasks: [niah_single, niah_multikey, niah_multivalue, niah_multiquery, vt, cwe, fwe, qa]
    depths: [0.0, 0.5, 1.0]               # depth-swept tasks only; 0.0 = very top (hardest)
    samples: 2                            # distinct probes per task (and per depth)
    output_tokens: 64                     # codes and short word lists
    options:                              # per-task knobs; unknown task names are rejected
      niah_multikey: {num_distractors: 3}
      vt: {chain_length: 3, num_chains: 3}
      cwe: {num_target_words: 10}

  generation:
    tasks: [summary, fact_sheet]
    samples: 2                            # distinct prompts per task -- WWB's "dataset"
    output_tokens: 256                    # prose; 64 would measure truncation, not fidelity
    options:
      fact_sheet: {num_facts: 6}
```

**Keep it small by default.** A probe at 32K costs about a minute of prefill on this box, so
the shipped settings above are 32 probes per case — top/middle/bottom depths, 2 samples —
rather than the 68 a five-depth three-sample sweep would be. 2 samples is an *indication*, not
a rate: a task scored over 2 probes can only report 0%, 50% or 100%. Raise `samples` once a
profile looks suspect and the question becomes "how often"; add the quarter-point depths when
you want a publication-shaped NIAH curve rather than a pass/fail.

The section is validated **before any model loads**: unknown suite, task or option names are
rejected, depths must be in `[0, 1]`, and `baseline_profile` must name a profile in the run — a
typo is a configuration error, not something to discover after an hour of probes.

`--accuracy-suites`, `--accuracy-tasks`, `--accuracy-depths` and `--accuracy-samples` override
the config for one run, across every suite being run. Check the probe count before committing to
a long sweep — the full suite across a depth sweep multiplies faster than it looks:

```powershell
... --config <accuracy config> --accuracy --list-profiles
... --accuracy --accuracy-tasks niah_single summary --accuracy-samples 1   # a quick pass
```

### Reading the result

`report.txt` adds, per context:

* a **per-task table** — the headline, because it is where a profile holding `niah_single` at
  100% while collapsing on `vt` or `cwe` becomes visible — with `Overall` / `EM` / `F1` /
  `Decoys` columns and a `Delta vs <baseline>` column. A negative delta is accuracy that
  profile's speed cost.
* the classic **depth sweep** matrix over the depth-swept tasks. A profile that holds the bottom
  and loses the top has a context window shorter than the one it was given.
* the **generation fidelity** table, one row per profile, with all three metric families. The
  baseline row compares against itself, so it reads 1.00 throughout — that row is the sanity
  check, not a result.

A **probe detail** section follows, one entry per probe: what was planted, what the model
answered, and what it scored. It is the only place the answers themselves survive, and
reading the worst-scoring ones is how a suspicious rate gets explained.

An accuracy run also reports **speed and resources** in the same file — its probes carry the
same timing records the throughput path produces, so one run gives TTFT, TPOT, throughput,
memory and accuracy together.

> **The generation suite's fidelity scores are cross-case**, so they are filled at report time
> rather than when a case finishes: the reference is the baseline profile's output, and that
> profile may run after the profile being scored. That is also why `report.txt` is rewritten
> in full after every case rather than appended to.

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
`--accuracy` switches the run to the
[accuracy mode](#long-context-accuracy-ruler--wwb-fidelity) (the config needs an `accuracy`
section); `--accuracy-suites`, `--accuracy-tasks`, `--accuracy-depths` and
`--accuracy-samples` narrow it for one run, across every suite being run.

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

Each run writes **one file**, `<output_dir>/<YYYYMMDD-HHMMSS>/report.txt`, so runs never mix.
The same lines are printed to the console when the run ends — one builder, two sinks, so the
file and the terminal can never disagree about what was measured. Its sections:

| Section | Contents |
|---|---|
| header | hardware, device, weights, budgets, the context and iteration plan, and (with `--accuracy`) the suites, baseline profile and seed |
| `SPEED AND RESOURCES` | one row per profile, ranked by TPOT: pipeline, KV precision, MTP yield, TPOT, TTFT, decode / e2e / prefill throughput, RAM and GPU peak/mean, the KV estimate and `cache_size`. Then the best-TPOT and fastest-TTFT call-outs, the MTP speedup line and the greedy-output check |
| `RETRIEVAL ACCURACY (RULER)` | per-task recall with `Overall` / `EM` / `F1` / `Decoys` and a `Delta vs <baseline>` column, then the depth × profile sweep |
| `GENERATION FIDELITY` | similarity, ROUGE-1/2/L, chrF, FDT, SDT norm, Identical and Coverage per profile, then the largest drift (or a no-drift line) |
| `PROBE DETAIL` | every probe: task, depth, sample, prompt size, its scores, the planted truth and the model's answer |
| `CASES THAT DID NOT PRODUCE A MEASUREMENT` | status and error for anything that failed, so a partial run still says what broke |

It is rewritten in full after **every** case, not once at the end: this tool's job is to push
the box until it breaks, and it can break hard enough to take the orchestrator with it. A
stale report that looks current is worse than none.

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
| `test_context_bench_kv_estimate.py` | architecture-derived KV size, the per-sequence linear-attention reservation `max_num_seqs` controls, `cache_size: auto`, the MTP draft head's own KV surcharge, and the shipped configs' invariants (loadable, unique profile names, every paged profile valid, and `accuracy.baseline_profile` naming a profile that actually runs) |
| `test_context_bench_metrics.py` | llm_bench units, TPOT-first ranking, the TTFT ranking, `pipeline_mode` recording, `mtp` profile resolution, the MTP yield/acceptance arithmetic, medians, the time-weighted mean occupancy, `perf_metrics` fallback |
| `test_context_bench_trial_lifecycle.py` | child exit path, parent recovery, failure classification, MTP rejected before the model loads |
| `test_context_bench_accuracy.py` | every scorer (SQuAD, RULER item-recall/IoU, ROUGE, chrF, FDT/SDT) including the trap that the lexical metrics must not use the article-stripping SQuAD tokenizer; each RULER task's shape, value determinism and corpus-disjointness, and that `cwe`/`fwe` ground truth survives the filler being sliced to an exact token count; exact-token prompt construction for up to sixteen inserts at their depths; probe-set sizing, per-suite scoring and aggregation over partially-applicable metrics; the cross-case generation-fidelity pairing; the who_what_benchmark adapter against a stubbed package; `accuracy` config validation; and the invariant that accuracy mode leaves `CASE_FIELDS`/`ITERATION_CSV_FIELDS` untouched |

All five run without a GPU, a model, or the OpenVINO stack.

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
