<!--
Copyright (C) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3.6-35B-A3B OpenVINO DFlash Evaluation Report

## Summary

This report verifies DFlash speculative decoding against the no-draft baseline for
`Qwen3.6-35B-A3B` int4 on the Arc B390 iGPU. Both modes use the shipped
[config_qwen3.6_35b.yaml](config_qwen3.6_35b.yaml). The sweep covers four task types,
two context settings, and `num_assistant_tokens` (k) = 3, 5, 7, and 15.

- **Code and math reach the >= 2x target at k=7.** Without a transcript, decode speeds up
  **2.32x on code** (45.2 -> 104.9 tok/s) and **2.23x on math** (43.6 -> 97.1 tok/s).
  After an 8K transcript the speedups are 2.22x and 2.12x.
- **Chat and summarization gain less, as expected.** Without a transcript, chat speeds up
  1.34x at k=7. At 8K the best results are 1.26x for chat (k=5) and 1.31x for the classroom
  summary (k=3).
- **Every domain meets or beats the published reference at k=7.** The DFlash PTL blog
  reports HumanEval 2.2x, GSM8K 1.6x, and MT-Bench 1.3x. Measured here: code 2.32x, math
  2.23x, chat 1.34x.
- **The best k depends on the domain.** k=7 is best for code and math, k=5 to k=7 for
  chat, and k=3 for summaries. **k=15 is never best**: its acceptance length is the highest
  on code and math, but verifying 16 tokens on this MoE costs 3.5-3.7 baseline decode steps.
- **Accuracy is unchanged.** RULER retrieval is identical for all five profiles (98% recall,
  same per-task scores, same answers byte for byte). Generation fidelity matches on 3 of 4
  probes; the fourth rewords after 37 tokens and still covers every planted fact.
- **TTFT is unchanged.** DFlash costs about +0.4 GB GPU memory without a transcript and
  +1.6 to +2.2 GB at 8K.

## Scope and Source Data

| Run | Directory (`monitoring/executionlogs/context_bench/qwen3.6_35b/`) | Content |
|---|---|---|
| Performance | `20261008-081216` | 35 cases: 5 profiles x (3 tasks without a transcript + 4 tasks at 8K) |
| Accuracy | `20261008-094129` | 5 profiles x 32 probes at 8K (RULER retrieval + generation fidelity) |

Both runs used the shipped config unchanged, with one exception: the accuracy run passed
`--iteration-gap 0`. The 30 s idle only conditions timing, and accuracy timings are not used
for speed conclusions. All performance values are medians of 3 measured iterations after
1 warm-up, which is excluded from every statistic.

## Test Environment

| Item | Value |
|---|---|
| Platform | Intel Core Ultra X7 358H (Panther Lake), 64 GB RAM |
| GPU | Intel Arc B390 iGPU, driver 32.0.101.8826 |
| Runtime | OpenVINO 2026.5.0.dev20260901, OpenVINO GenAI 2026.5.0.0.dev20260901, NNCF 2.19.0 |
| Target model | `models/openvino/Qwen3.6-35B-A3B_int4`: int4_asym g64, int8_sym backup (NNCF rt_info) |
| Draft model | `models/openvino/qwen3.6-35b-a3b-dflash-int4-ov-int4head`: z-lab Qwen3.6-35B-A3B-DFlash, int4_asym g128, block_size 16, plus its own int4 lm_head (see [dflash_draft_head.py](dflash_draft_head.py)) |
| Device | Target and draft both on GPU |
| GPU budget | 59 GB of shared memory |
| Decoding | Greedy (`do_sample=False`), EOS respected, `assistant_confidence_threshold=0`, fixed k |

## Common Runtime Configuration

All profiles share identical OpenVINO properties and scheduler settings, so DFlash and k are
the only variables:

```yaml
ov:
  GPU_ENABLE_LARGE_ALLOCATIONS: "YES"
  KV_CACHE_PRECISION: f16
scheduler:
  max_num_batched_tokens: 65536   # whole prompt in one prefill pass
  max_num_seqs: 1                 # DFlash serves one request at a time
  num_linear_attention_blocks: 17 # 1 committed + (1 + max k = 15) verification rows
  enable_prefix_caching: false    # required by DFlash; every TTFT is a full prefill
```

| Profile | DFlash | k |
|---|---|---|
| `paged_min` | off (baseline) | -- |
| `dflash_k3` | on | 3 |
| `dflash_k5` | on | 5 |
| `dflash_k7` | on | 7 (the blog's setting) |
| `dflash_k15` | on | 15 (the draft's maximum: block_size 16 minus the seed token) |

## Workloads

| Task | Domain (reference dataset) | Prompt | Output ceiling |
|---|---|---|---|
| `code` | Coding (HumanEval) | Typed Python module of five lab functions plus unittest cases | 512 (all runs hit it) |
| `math` | Math (GSM8K) | Six worked physics, arithmetic, and algebra problems, each ending `Answer: <n>` | 512 (all runs hit it) |
| `chat` | Assistant (MT-Bench) | Eight short requests, one per MT-Bench category: writing, roleplay, reasoning, math, coding, extraction, STEM, humanities | 512 (493-512 produced) |
| `summary_2s` | Classroom summarization | Two-sentence summary of the transcript | 512 (EOS at 45) |

The two context settings are:
- **No transcript** (`context_tokens: 0`): the task prompt alone, 228-285 tokens. This is
  the setting the published DFlash numbers were measured in.
- **8K**: an 8,000-token synthetic classroom transcript, followed by the task. The summary
  task runs at 8K only, because it summarizes the transcript.

**Metrics.** *Acceptance length (AL)* is the tokens committed per target verification pass:
accepted draft tokens plus the target's bonus token, `(output - 1) / (passes - 1)`.
*Accepted %* is `(AL - 1) / k`. It falls with k by construction, so it is reported for
context only. *Pass cost* is the steady-state time of one DFlash pass (draft + verify)
divided by one baseline decode step. It is also the break-even AL, and the speedup is
approximately AL / pass cost. *Decode speedup* is the ratio of baseline TPOT to DFlash TPOT
over the whole answer. *Steady speedup* excludes the first decode pass, which carries
one-time per-request work.

## Performance Results

### No transcript (task prompt only)

| Task | Profile | TPOT ms | Decode tok/s | AL | Accepted | Pass ms | Pass cost | Decode speedup | Steady speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| code | `paged_min` | 22.1 | 45.2 | 1.00 | -- | -- | 1.00x | 1.00x | 1.00x |
| code | `dflash_k3` | 12.0 | 83.3 | 3.65 | 89% | 43.2 | 1.98x | 1.84x | 1.86x |
| code | `dflash_k5` | 10.1 | 99.2 | 4.96 | 79% | 49.0 | 2.25x | 2.20x | 2.21x |
| code | **`dflash_k7`** | **9.5** | **104.9** | 5.94 | 71% | 55.0 | 2.52x | **2.32x** | **2.35x** |
| code | `dflash_k15` | 11.1 | 90.2 | 7.20 | 42% | 79.7 | 3.65x | 2.00x | 2.01x |
| math | `paged_min` | 22.9 | 43.6 | 1.00 | -- | -- | 1.00x | 1.00x | 1.00x |
| math | `dflash_k3` | 13.1 | 76.3 | 3.52 | 85% | 45.1 | 2.06x | 1.75x | 1.76x |
| math | `dflash_k5` | 10.7 | 93.5 | 4.82 | 77% | 50.6 | 2.31x | 2.14x | 2.15x |
| math | **`dflash_k7`** | **10.3** | **97.1** | 5.68 | 67% | 56.5 | 2.58x | **2.23x** | **2.24x** |
| math | `dflash_k15` | 11.6 | 86.0 | 6.91 | 40% | 79.7 | 3.64x | 1.97x | 1.97x |
| chat | `paged_min` | 22.1 | 45.2 | 1.00 | -- | -- | 1.00x | 1.00x | 1.00x |
| chat | `dflash_k3` | 17.3 | 57.9 | 2.59 | 53% | 44.2 | 2.02x | 1.28x | 1.28x |
| chat | `dflash_k5` | 17.2 | 58.1 | 2.85 | 38% | 48.1 | 2.20x | 1.29x | 1.29x |
| chat | **`dflash_k7`** | **16.5** | **60.7** | 3.30 | 33% | 53.8 | 2.45x | **1.34x** | **1.35x** |
| chat | `dflash_k15` | 24.7 | 40.5 | 3.17 | 15% | 77.2 | 3.52x | 0.90x | 0.90x |

TTFT was 0.3-0.4 s for every profile. Prefill throughput was 640-740 tok/s on these short
prompts.

### 8K transcript

| Task | Profile | TPOT ms | Decode tok/s | AL | Accepted | Pass ms | Pass cost | Decode speedup | Steady speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| code | `paged_min` | 24.0 | 41.7 | 1.00 | -- | -- | 1.00x | 1.00x | 1.00x |
| code | `dflash_k3` | 14.3 | 70.1 | 3.70 | 90% | 51.8 | 2.18x | 1.68x | 1.70x |
| code | `dflash_k5` | 11.5 | 87.1 | 5.11 | 83% | 57.0 | 2.40x | 2.09x | 2.13x |
| code | **`dflash_k7`** | **10.8** | **92.5** | 6.08 | 74% | 63.4 | 2.67x | **2.22x** | **2.27x** |
| code | `dflash_k15` | 11.5 | 86.6 | 7.74 | 47% | 87.9 | 3.70x | 2.08x | 2.11x |
| math | `paged_min` | 24.1 | 41.4 | 1.00 | -- | -- | 1.00x | 1.00x | 1.00x |
| math | `dflash_k3` | 14.5 | 69.0 | 3.62 | 88% | 51.4 | 2.15x | 1.67x | 1.69x |
| math | `dflash_k5` | 12.1 | 82.5 | 4.87 | 78% | 57.3 | 2.40x | 1.99x | 2.03x |
| math | **`dflash_k7`** | **11.4** | **87.8** | 5.74 | 69% | 63.8 | 2.67x | **2.12x** | **2.15x** |
| math | `dflash_k15` | 12.4 | 80.5 | 7.20 | 43% | 87.9 | 3.68x | 1.94x | 1.97x |
| chat | `paged_min` | 24.0 | 41.7 | 1.00 | -- | -- | 1.00x | 1.00x | 1.00x |
| chat | `dflash_k3` | 19.4 | 51.5 | 2.65 | 55% | 50.5 | 2.13x | 1.23x | 1.25x |
| chat | **`dflash_k5`** | **19.0** | **52.5** | 3.01 | 40% | 56.1 | 2.36x | **1.26x** | **1.28x** |
| chat | `dflash_k7` | 20.8 | 48.1 | 3.02 | 29% | 61.6 | 2.60x | 1.15x | 1.16x |
| chat | `dflash_k15` | 29.6 | 33.7 | 3.01 | 14% | 84.8 | 3.57x | 0.81x | 0.81x |
| summary_2s | `paged_min` | 24.6 | 40.7 | 1.00 | -- | -- | 1.00x | 1.00x | 1.00x |
| summary_2s | **`dflash_k3`** | **18.7** | **53.4** | 3.14 | 71% | 50.3 | 2.13x | **1.31x** | **1.44x** |
| summary_2s | `dflash_k5` | 20.1 | 49.9 | 3.14 | 43% | 55.7 | 2.36x | 1.23x | 1.30x |
| summary_2s | `dflash_k7` | 20.7 | 48.3 | 3.38 | 34% | 62.3 | 2.64x | 1.19x | 1.27x |
| summary_2s | `dflash_k15` | 26.7 | 37.5 | 3.38 | 16% | 84.3 | 3.57x | 0.92x | 0.94x |

TTFT was 3.0 s for every profile; the one exception, `code` k=3, measured 3.2 s. Prefill
throughput was 2,470-2,670 tok/s. The summary answer is only 45 tokens, so the first decode
pass weighs heavily on it. This is why its end-to-end speedup (1.31x) sits well below the
steady-state figure (1.44x).

### Best window per task

| Context | Task | Best k | Decode speedup | Runner-up | k=15 |
|---|---|---:|---:|---|---:|
| None | code | 7 | 2.32x | k=5 (2.20x) | 2.00x |
| None | math | 7 | 2.23x | k=5 (2.14x) | 1.97x |
| None | chat | 7 | 1.34x | k=5 (1.29x) | 0.90x |
| 8K | code | 7 | 2.22x | k=5 (2.09x) | 2.08x |
| 8K | math | 7 | 2.12x | k=5 (1.99x) | 1.94x |
| 8K | chat | 5 | 1.26x | k=3 (1.23x) | 0.81x |
| 8K | summary_2s | 3 | 1.31x | k=5 (1.23x) | 0.92x |

### Comparison with the published reference (k=7)

The reference is the Hugging Face blog *Accelerating Qwen3.6 on Intel Core Ultra Series 3
with DFlash* (2026-07-30). It used the same model and draft in int4 W4A16, an Arc B390
(Core Ultra X7 368H, driver 32.0.101.8860), OpenVINO 2026.3, greedy decoding, and k=7, on
whole datasets with no long-context prefix.

| Domain | Blog dataset | Blog AL | Blog tok/s (speedup) | Measured AL, no transcript | Measured tok/s (speedup), no transcript | Measured, 8K |
|---|---|---:|---:|---:|---:|---:|
| Coding | HumanEval | 6.4 | 89.8 (2.2x) | 5.94 | 104.9 (2.32x) | AL 6.08, 2.22x |
| Math | GSM8K | 5.0 | 68.5 (1.6x) | 5.68 | 97.1 (2.23x) | AL 5.74, 2.12x |
| Assistant | MT-Bench | 4.0 | 54.7 (1.3x) | 3.30 | 60.7 (1.34x) | AL 3.02, 1.15x |
| Baseline | -- | -- | 41 | -- | 43.6-45.2 | 41.4-41.7 |

Speedups meet or exceed the reference in every domain, and the domain ordering is the same.
AL differs in both directions because each domain here is one prompt rather than a dataset.
The `math` prompt is easier to draft than GSM8K. The `chat` prompt is harder than the
MT-Bench average: an earlier chat prompt containing only open-ended requests measured just
AL 2.4. This box decodes faster at the same AL because its baseline is 6-10% faster (newer
runtime and driver) and its verification passes are cheap (2.45-2.58x a decode step at k=7).

### Resource cost

| Context | Profile | RAM peak / mean GB | GPU peak / mean GB | TTFT s |
|---|---|---|---|---:|
| None | `paged_min` | 32.8-33.8 / 31.2-32.0 | 21.2-21.3 / 19.5-19.6 | 0.4 |
| None | DFlash (any k) | 33.3-34.3 / 32.0-33.0 | 21.6-21.7 / 20.4-20.6 | 0.3-0.4 |
| 8K | `paged_min` | 33.5-34.1 / 31.8-32.4 | 22.4-22.5 / 20.7 | 3.0 |
| 8K | DFlash (any k) | 35.1-36.3 / 33.6-34.5 | 24.1-24.6 / 22.5-23.1 | 3.0-3.2 |

The draft adds about 0.4 GB of GPU memory without a transcript and 1.6-2.2 GB at 8K. The
draft's weights are 0.46 GB; the rest is its own KV cache and workspace. Memory does not
depend on k. Every case stayed under 42% of the 59 GB GPU budget.

### Measurement stability

Across the 35 cases, per-iteration TPOT spread (max - min over min) averaged 1.5%. Eight
cases exceeded 2%, the largest being 6.0% (`summary_2s` k=5, whose answer is only 45
tokens). Across runs, the same profiles measured earlier the same day reproduced AL exactly,
because greedy decoding is deterministic. Decode speedups moved by up to 0.08x (for example,
code k=7 without a transcript: 2.40x earlier, 2.32x here), because this run's baseline was
3% faster. Treat differences under ~0.1x as noise.

## Accuracy Results (8K)

The accuracy run covered 32 probes per profile with seed 20260917: 28 RULER retrieval probes
(8 tasks, 2 samples, depths 0/0.5/1.0 for the depth-swept tasks) and 4 generation probes
(`summary` and `fact_sheet`, 2 samples each). Speculative decoding is meant to be lossless, so
each DFlash profile is scored against the baseline.

### RULER retrieval

| Profile | niah_single | niah_multikey | niah_multivalue | niah_multiquery | vt | cwe | fwe | qa | Overall recall | EM | F1 | Decoys | Delta vs baseline |
|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| `paged_min` | 100% | 100% | 100% | 100% | 67% | 100% | 100% | 100% | 98% | 71% | 85% | 12% | -- |
| `dflash_k3` | 100% | 100% | 100% | 100% | 67% | 100% | 100% | 100% | 98% | 71% | 85% | 12% | +0 pp |
| `dflash_k5` | 100% | 100% | 100% | 100% | 67% | 100% | 100% | 100% | 98% | 71% | 85% | 12% | +0 pp |
| `dflash_k7` | 100% | 100% | 100% | 100% | 67% | 100% | 100% | 100% | 98% | 71% | 85% | 12% | +0 pp |
| `dflash_k15` | 100% | 100% | 100% | 100% | 67% | 100% | 100% | 100% | 98% | 71% | 85% | 12% | +0 pp |

The depth sweep (niah_single, niah_multikey, qa) was 100% at every depth for every profile.
All 28 retrieval answers are byte-identical between the baseline and every DFlash profile.
The `vt` (variable tracing) misses and the EM/decoy figures are the target model's own
behavior. It over-answers chains and wraps codes in sentences. DFlash does not change any of
it.

### Generation fidelity (scored against `paged_min`'s answer to the same prompt)

| Profile | ROUGE-1 | ROUGE-2 | ROUGE-L | chrF | First divergence (model tokens, mean) | Identical answers | Fact coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dflash_k3` | 0.96 | 0.92 | 0.96 | 0.94 | 63.8 | 3 / 4 | 100% |
| `dflash_k5` | 0.96 | 0.92 | 0.96 | 0.94 | 63.8 | 3 / 4 | 100% |
| `dflash_k7` | 0.96 | 0.92 | 0.96 | 0.94 | 63.8 | 3 / 4 | 100% |
| `dflash_k15` | 0.96 | 0.92 | 0.96 | 0.94 | 63.8 | 3 / 4 | 100% |

Three probes (both `summary` probes and `fact_sheet` #0) are identical to the baseline. In
the fourth, `fact_sheet` #1, every k diverges from the baseline at the same token (37) and
produces the same text (ROUGE-L 0.85, chrF 0.77). It still reports all six planted codes
(coverage 1.00). Embedding similarity was not measured, because `whowhatbench` is not
installed; every other metric is computed natively.

## Greedy Output Consistency (throughput prompts)

Each profile reproduced its own output exactly across iterations. Compared with the
baseline's greedy output:

| Context | Task | Identical to baseline | Differs (fork position in characters) |
|---|---|---|---|
| None | code | k=5, k=7 | k=3 (537 of 1,894), k=15 (932) |
| None | math | k=3, k=5, k=15 | k=7 (1,328 of 1,344) |
| None | chat | -- | k=3 and k=5 (21 of 2,064), k=7 (353), k=15 (618) |
| 8K | code | all | -- |
| 8K | math | all | -- |
| 8K | chat | -- | k=3, k=5, and k=7 (616 of 2,079), k=15 (431) |
| 8K | summary_2s | all | -- |

Every fork is a near-equivalent token, not corrupted output:

| Fork | Baseline continues | DFlash continues |
|---|---|---|
| code k=3: `"""Calculate gravitational potential energy` | ` as m * g * h."""` | `: m * g * h."""` |
| code k=15: `"""Calculate braking distance using ` | `d = v^2 / (2 * a).` | `v^2 / (2 * a).` |
| math k=7: `Solve for ` | `velocity ($v$):` | `$v$:` |
| chat k=3/5: `1. Dear Parents, our ` | `Science Fair is next month!` | `school science fair is next month!` |
| chat k=15: `Since Ana finished before Ben, ` | `Ben is behind Ana.` | `Ana is ahead of Ben.` |
| chat 8K k=15: `towering skeletons of T` | `-Rex and gentle Brachiosaurus` | `yrannosaurus Rex and gentle Brachiosaurus` |

The target scores the k + 1 verification tokens in one pass, on a different GPU kernel path
than a single-token decode step. The fp16 logits therefore differ slightly, and where two
tokens nearly tie, the argmax can flip. This is why the fork position moves with k and why
chat, whose next-token distribution is flattest, forks most often. Greedy speculative
decoding here is lossless up to that floating-point difference. The code and math forks occur
after 28-99% of the answer has been produced. The chat forks are early, so for chat the
speedup compares two equally valid but different answers.

## Findings for Optimization Teams

1. **The MoE verification cost is the ceiling.** A k=7 pass costs 2.45-2.67x a baseline
   decode step, and k=15 costs 3.5-3.7x. On a dense model the same pass would cost close to
   1x. The draft is cheap (its share of a pass is small), so the remaining lever is the
   target's batched MoE verification: expert loading for k + 1 tokens on the GPU MoE path.
2. **AL saturates by domain, not by k.** Code and math keep gaining AL up to k=15 (7.2-7.7),
   but the pass cost grows faster beyond k=7. Chat plateaus at AL ~3.0-3.3 from k=5, and the
   summary plateaus at 3.1-3.4 from k=3. A fixed k is a compromise across domains. Choosing k
   per request type, or adapting it to a running AL, would be worth about +0.1x on chat and
   summary (versus a fixed k=7) and avoid k=15's 0.8-0.9x losses.
3. **The first decode pass is expensive.** A DFlash request's first decode pass costs 91-99 ms
   without a transcript and 171-201 ms at 8K, against 61-71 ms for the baseline. On a 45-token
   answer, that pass lowers the speedup from 1.44x (steady state) to 1.31x.
4. **The draft lm_head graft is avoidable.** Giving the draft its own int4 lm_head (509 MB of
   int8 not streamed per pass) is already applied in this config. GenAI could do the same
   natively.
5. **Accuracy needs no tolerance work.** Retrieval is bit-identical and generation stays at
   ROUGE-L 0.96 with full fact coverage. The remaining differences are fp16 near-tie flips
   between verification and decode kernels.

## Recommendations

- For code- and math-heavy use, run k=7: 2.1-2.3x faster decode, unchanged TTFT, +0.4-2.2 GB
  GPU memory.
- For the classroom summarizer, use k=3 (1.31x end to end, 1.44x steady state). Validate it
  on `classroom_summary` with the app's real prompt before changing the app default.
- Do not use k=15. It is never best and loses to the baseline on chat and summary.

## Limitations

- Each domain is one representative prompt per context, not a full dataset. AL varies by
  prompt, so compare domains by their ordering and size, not to the second decimal.
- The accuracy suites run at 8K only; they do not run without a transcript.
- Embedding similarity (`whowhatbench`) was not installed.
- Only Qwen3.6-35B-A3B was tested. The blog's dense models (Qwen3.6-27B, Qwen3.5-9B) have no
  local DFlash drafts.
- The runtime is a 2026.5 development build. Re-verify on the release build.

## Reproduction

From `smart-classroom/`, with the backend venv active:

```powershell
# One-time: build the draft with its own int4 lm_head (the config points at it)
python -m components.llm.context_bench.dflash_draft_head `
    --target models/openvino/Qwen3.6-35B-A3B_int4 `
    --draft models/openvino/qwen3.6-35b-a3b-dflash-int4-ov `
    --output models/openvino/qwen3.6-35b-a3b-dflash-int4-ov-int4head

# Performance: 35 cases, ~1.6 h
.\components\llm\context_bench\run_benchmark.ps1

# Accuracy: 5 cases x 32 probes at 8K, ~15 min with no idle gap
.\components\llm\context_bench\run_benchmark.ps1 --accuracy --iteration-gap 0

# Quick check against the blog: no transcript, k=7, ~15 min
.\components\llm\context_bench\run_benchmark.ps1 --contexts 0 --profiles paged_min dflash_k7 --throughput-task code math chat
```

Both runs write a `report.txt` that contains every table above. Each report also has a
SPECULATIVE WINDOW SWEEP per context, with the blog reference, and per-probe accuracy
detail. Methodology is described in the
[context bench guide](../../../docs/dev-guide/context-bench/context_bench_guide.md).
