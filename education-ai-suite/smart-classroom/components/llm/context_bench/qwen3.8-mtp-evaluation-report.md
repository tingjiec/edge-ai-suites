# Qwen3.8-27B OpenVINO MTP Evaluation Report

## Scope

This report compares the 8K-context paged-attention baseline against OpenVINO
GenAI self-speculative multi-token prediction (MTP) runs for
`Qwen3.8-27B_int4` on GPU. It is intended for model, OpenVINO GenAI, and GPU
plugin optimization work.

The benchmark configuration is [config_qwen3.8_27b.yaml](config_qwen3.8_27b.yaml).
Source measurements are:

- `monitoring/executionlogs/context_bench/qwen3.8_27b/20260903-174224` (`paged_min`)
- `monitoring/executionlogs/context_bench/qwen3.8_27b/20260903-173859` (`mtp_k2`)
- `monitoring/executionlogs/context_bench/qwen3.8_27b/20260903-174032` (`mtp_k4`)

All summary values are medians of three measured iterations. One 4-token warm-up
is excluded from every result.

## Test Environment

| Item | Value |
|---|---|
| Model | `Qwen3.8-27B` OpenVINO IR, INT4 weights |
| Model directory | `models/openvino/Qwen3.8-27B_int4` |
| Model weights on disk | 14.8 GB |
| Device | Intel Graphics GPU |
| Host memory | 64 GB RAM |
| GPU memory budget | 59 GB shared-memory budget |
| Context length | 8,000 tokens |
| Requested output | 64 tokens; each measured run produced 61 tokens |
| Repetitions | 1 warm-up + 3 measured iterations |
| Timeout | 1,800 s per case |

## Common Runtime Configuration

Every profile used the same target pipeline and scheduler settings. This keeps
MTP lookahead as the controlled variable.

```yaml
device: GPU
weight_format: int4
ov:
  ATTENTION_BACKEND: PA
  GPU_ENABLE_LARGE_ALLOCATIONS: "YES"
  KV_CACHE_PRECISION: f16
scheduler:
  max_num_batched_tokens: 32768
  max_num_seqs: 1
  enable_prefix_caching: false
  # cache_size is unset: OpenVINO runtime manages the KV pool.
```

`ATTENTION_BACKEND: PA` and a `SchedulerConfig` are required by the Qwen3.8 MTP
path. Prefix caching is disabled so measured TTFT always includes the complete
8K prefill. The current configuration deliberately leaves `cache_size` unset;
the OpenVINO runtime therefore owns KV-pool sizing.

## MTP Implementation

When `mtp` is enabled, context_bench follows the OpenVINO Qwen3.8 MTP notebook:

1. It requires both `openvino_mtp_model.xml` and `openvino_mtp_model.bin` in the
   target model directory.
2. It creates `ov_genai.draft_model(model_dir, device)` on the same GPU as the
   target pipeline.
3. It creates a paged `VLMPipeline` or `LLMPipeline` with that draft model and
   the shared scheduler configuration.
4. It creates a fresh greedy `GenerationConfig`: `do_sample=False`,
   `num_return_sequences=1`, `assistant_confidence_threshold=0.0`, and a fixed
   `num_assistant_tokens=k`.
5. It reads acceptance and draft/accepted token counts from
   `extended_perf_metrics`.

NPU and cross-device draft configurations are rejected for this experimental
Qwen3.8 MTP workflow. The implementation is in `trial_runner.py`.

## Profile Configuration

| Profile | MTP state | Draft model | `num_assistant_tokens` | Expected behavior |
|---|---|---|---:|---|
| `paged_min` | Off | Not loaded | 0 | One target-model decode pass per token |
| `mtp_k2` | On | Built-in Qwen3.8 MTP head on GPU | 2 | Verify up to two draft candidates per target-model pass |
| `mtp_k4` | On | Built-in Qwen3.8 MTP head on GPU | 4 | Verify up to four draft candidates per target-model pass |

## Median Results

| Profile | MTP metrics | TTFT | TPOT | Decode throughput | Prefill throughput | End-to-end throughput | Generation time | RAM peak / mean | GPU peak / mean |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `paged_min` | Off, 1.00 tok/step | 12.5 s | 135.8 ms/token | 7.37 tok/s | 639.2 tok/s | 389.5 tok/s | 20.70 s | 30.8 / 29.4 GB | 19.1 / 17.8 GB |
| `mtp_k2` | k=2, 2.44 tok/step, 74% accepted (34/46) | 12.6 s | 81.3 ms/token | 12.30 tok/s | 634.4 tok/s | 460.1 tok/s | 17.52 s | 34.3 / 32.9 GB | 21.9 / 20.8 GB |
| `mtp_k4` | k=4, 2.90 tok/step, 50% accepted (38/76) | 13.0 s | 77.4 ms/token | 12.93 tok/s | 615.6 tok/s | 455.1 tok/s | 17.71 s | 33.6 / 32.5 GB | 21.9 / 20.9 GB |

## Change Relative to MTP-Off Baseline

| Profile | TTFT change | TPOT change | Decode throughput change | Prefill throughput change | End-to-end throughput change | GPU peak change |
|---|---:|---:|---:|---:|---:|---:|
| `mtp_k2` | +0.1 s (+0.8%) | -54.5 ms/token (-40.1%) | +4.93 tok/s (+66.9%) | -4.8 tok/s (-0.8%) | +70.6 tok/s (+18.1%) | +2.8 GB (+14.7%) |
| `mtp_k4` | +0.5 s (+4.0%) | -58.4 ms/token (-43.0%) | +5.56 tok/s (+75.4%) | -23.6 tok/s (-3.7%) | +65.6 tok/s (+16.8%) | +2.8 GB (+14.7%) |

MTP improves the decode phase as expected. TTFT and prefill remain close to the
baseline, while draft-model allocation increases GPU peak usage by 2.8 GB.
`mtp_k4` is the fastest measured decode profile; `mtp_k2` has the highest
candidate acceptance and the best end-to-end throughput in this set.

## Per-Iteration Results

### MTP Off: `paged_min`

| Iteration | Output | TTFT | TPOT | Decode throughput | Prefill throughput | End-to-end throughput | Generation time |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 61 | 11.164 s | 136.799 ms/token | 7.310 tok/s | 716.563 tok/s | 415.123 tok/s | 19.418 s |
| 2 | 61 | 12.581 s | 135.775 ms/token | 7.365 tok/s | 635.856 tok/s | 388.019 tok/s | 20.775 s |
| 3 | 61 | 12.515 s | 135.621 ms/token | 7.373 tok/s | 639.247 tok/s | 389.454 tok/s | 20.698 s |

### MTP On: `mtp_k2`

| Iteration | Output | TTFT | TPOT | Decode throughput | MTP yield | Accepted / draft | End-to-end throughput | Generation time |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | 61 | 11.653 s | 82.564 ms/token | 12.112 tok/s | 2.44 tok/step | 34 / 46 (73.91%) | 482.651 tok/s | 16.702 s |
| 2 | 61 | 12.611 s | 80.321 ms/token | 12.450 tok/s | 2.44 tok/step | 34 / 46 (73.91%) | 460.079 tok/s | 17.521 s |
| 3 | 61 | 12.850 s | 81.328 ms/token | 12.296 tok/s | 2.44 tok/step | 34 / 46 (73.91%) | 452.204 tok/s | 17.826 s |

### MTP On: `mtp_k4`

| Iteration | Output | TTFT | TPOT | Decode throughput | MTP yield | Accepted / draft | End-to-end throughput | Generation time |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | 61 | 15.747 s | 80.773 ms/token | 12.380 tok/s | 2.905 tok/step | 38 / 76 (50.00%) | 389.536 tok/s | 20.694 s |
| 2 | 61 | 12.995 s | 77.266 ms/token | 12.942 tok/s | 2.905 tok/step | 38 / 76 (50.00%) | 455.057 tok/s | 17.714 s |
| 3 | 61 | 12.514 s | 77.361 ms/token | 12.926 tok/s | 2.905 tok/step | 38 / 76 (50.00%) | 467.508 tok/s | 17.242 s |

## Correctness Signal

All three measured runs generated 61 output tokens and share this output SHA-256:

```
783098df224db950379963959b9d471e29cbc34d9e98f2286fed53897ee4ed3f
```

For this prompt, MTP=on preserves the baseline text while reducing decode time.
This is a useful correctness signal, but it does not establish broad output
equivalence across representative classroom requests.

## Findings for Optimization Teams

1. The MTP draft head is active. `mtp_k2` commits 2.44 tokens per verification
   pass and `mtp_k4` commits 2.90, both above the 1.00-token baseline.
2. MTP substantially improves decode. The measured peak is `mtp_k4` at
   12.93 tok/s, a 75.4% improvement over the 7.37 tok/s baseline.
3. Candidate quality falls as lookahead grows: acceptance decreases from 73.91%
   at k=2 to 50.00% at k=4. Four additional draft candidates yield four more
   accepted tokens (38 versus 34).
4. `mtp_k4` has the best TPOT but not the best end-to-end result. Its 13.0 s
   median TTFT and lower prefill rate produce 455.1 tok/s E2E, below `mtp_k2`
   at 460.1 tok/s.
5. `mtp_k4` has visible TTFT variance (12.514-15.747 s), making its first
   measured iteration an end-to-end outlier at 20.694 s. Its post-prefill TPOT
   remains faster than the baseline in every measured iteration.
6. MTP costs approximately 2.8 GB peak GPU memory and 3.1-3.5 GB peak RAM
   relative to MTP off. This is within the configured 59 GB GPU budget, but
   should be included in capacity and concurrency planning.
7. `cache_size` is unset in the active configuration. The runtime-managed pool
   makes this comparison valid because all profiles use it, but explicit pool
   sizing should be evaluated for long-context predictability and concurrency.

## Recommended Follow-Up Experiments

1. Run `mtp_k1`, `mtp_k2`, `mtp_k3`, `mtp_k4`, and `mtp_k6` in one 8K matrix so
   they share machine state and driver conditions.
2. Repeat with 128 and 256 output tokens. TTFT currently dominates the 61-token
   response and masks some decode-only benefit.
3. Use representative classroom prompts and compare token sequences with MTP
   off. Record divergence position, throughput, and acceptance.
4. Profile draft-model and target-model execution separately to identify whether
   optimization belongs in draft proposal, target verification, or GPU scheduling.
5. Sweep explicit `cache_size` values using the same MTP matrix, recording the
   largest stable context and GPU allocation behavior before tuning concurrency.