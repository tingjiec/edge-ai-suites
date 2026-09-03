# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Runs one (model, profile, context) benchmark case in an isolated subprocess.

The case loads the pipeline once and then generates ``warmup + iterations``
times over the same prompt, as llm_bench does. Loading once is not only faster
-- reloading 33 GB of Qwen3.6-35B-A3B weights per iteration would dominate the
measurement -- it is also what makes the warm-up meaningful: iteration 0 absorbs
lazy weight paging and first-run kernel compilation so the measured iterations
see a warm pipeline.

Reusing one prompt across iterations requires ``enable_prefix_caching: false``.
With prefix caching on, iteration 1 would hit a cache populated by the warm-up
and report a TTFT that no first request will ever see.

Each case runs in a fresh subprocess (see benchmark.py) so GPU/host memory from
an OOM'd or aborted attempt cannot bleed into the next one -- the same rationale
as components/vlm/vlm_openvino_serving/utils/utils.py::_convert_model_worker.
Memory is sampled by the parent: RAM/GPU counters are system-wide, so the
parent sees this child's footprint and, crucially, its readings survive a child
that gets killed on a timeout.

Timing comes from OpenVINO GenAI's own ``perf_metrics`` where the runtime
provides it (the same source llm_bench uses), falling back to wall-clock timing
around the streamer callback.

A profile may also enable multi-token prediction, which is self-speculative
decoding: the model's own ``openvino_mtp_model.xml`` draft head proposes ``k``
candidates and the main model verifies them in one pass. It is a decode-side
lever only -- measured on Qwen3.8-27B it cuts TPOT by 1.6-1.9x and leaves TTFT
untouched -- and it only runs on the paged backend. See ``validate_mtp``.

OpenVINO / transformers are imported lazily inside run_case() so this module can
be imported on a machine without the OpenVINO stack.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
import time
import traceback

from components.llm.context_bench import metrics
from components.llm.context_bench.context_builder import build_benchmark_prompt

# Text that says outright that memory was never handed over. Includes OpenCL's own
# vocabulary, because at the capacity ceiling the exception is the GPU plugin's, not
# Python's, and it reaches us as the plain text of an ov::Exception.
_OOM_MARKERS = (
    "out of gpu resources",
    "out of memory",
    "allocation failed",
    "bad_alloc",
    "cannot allocate",
    "can't allocate",
    "insufficient memory",
    "memoryerror",
    "larger than available memory",  # scheduler refusing an oversized cache_size
    "cl_mem_object_allocation_failure",  # -4: the device could not back the buffer
    "cl_out_of_host_memory",  # -6: the runtime could not back it on the host either
    "cl_invalid_buffer_size",  # -61: one buffer larger than the device permits
)

# A GPU *command* that was accepted and then died on the device. OpenCL surfaces this at
# the next synchronization point, so the message names the wait (`clWaitForEvents`) and
# never names what ran out. On a shared-memory iGPU at its ceiling this is usually memory
# exhaustion, but a driver reset (TDR) produces the same code and this process cannot tell
# them apart -- it reports the narrower fact and lets the parent's memory samples speak.
_GPU_ABORT_MARKERS = (
    "cl_exec_status_error_for_events_in_wait_list",  # -14
    "cl_out_of_resources",  # -5
    "cl_invalid_command_queue",  # -36, typical after a device reset
    "clwaitforevents",
    "clfinish",
)

# A property or value this runtime build does not implement. Distinguished from a real
# failure so an int4-KV profile on a runtime without int4 KV is skipped, not reported as
# a hardware limit.
_UNSUPPORTED_MARKERS = (
    "unsupported property",
    "unsupported configuration",
    "is not supported",
    "not supported by",
    "unknown property",
    "unsupported schedulerconfig property",
)

# How far the case got. Survives a child that dies without reporting, which is where it
# earns its keep: "timed out at 2400s" says nothing, "timed out having reached prefill"
# says the box never finished one forward pass over the context.
STAGE_START = "start"  # nothing done yet; a failure here is the plugin refusing to build
STAGE_LOADED = "loaded"  # pipeline constructed
STAGE_PROMPT_BUILT = "prompt_built"  # context tokenized, nothing sent to the device yet
STAGE_PREFILLED = "prefilled"  # first token streamed: the whole context is through the model
STAGE_DECODED = "decoded"  # at least one generate() returned

# The multi-token-prediction draft head, as `optimum-cli`/the published OpenVINO IR names
# it. Both files are required: the XML is the graph and the BIN holds its weights.
# The same directory serves as both the target and the draft model.
MTP_MODEL_FILES = ("openvino_mtp_model.xml", "openvino_mtp_model.bin")
MTP_MODEL_FILE = MTP_MODEL_FILES[0]

# Properties openvino_genai consumes itself rather than forwarding to the plugin. The GPU
# plugin does not advertise them, so without this list `_device_diagnostics` reports every
# MTP profile as passing an unsupported property -- advice that would break the profile if
# followed.
_GENAI_LEVEL_PROPERTIES = frozenset({"ATTENTION_BACKEND", "scheduler_config", "draft_model"})

# The stage a failure belongs to, given the last milestone reached. Off-by-one on purpose:
# an exception is thrown by the step *after* the last completed milestone, so a crash with
# the prompt built but no token streamed is a prefill failure.
_FAILING_STAGE = {
    STAGE_START: "load",
    STAGE_LOADED: "prompt",
    STAGE_PROMPT_BUILT: "prefill",
    STAGE_PREFILLED: "decode",
    STAGE_DECODED: "decode",
}


def failing_stage(stage_reached: str) -> str:
    """Name the stage that was running when a case at `stage_reached` failed."""
    return _FAILING_STAGE.get(stage_reached, "generate")


def has_mtp_head(model_dir: str) -> bool:
    """Whether this export ships the multi-token-prediction draft head.

    Not every OpenVINO export of an MTP-capable model includes it, and genai's own
    failure for a missing one is a deep assertion inside the speculative-decoding
    strategy. Checking the file is what lets a profile be rejected up front.
    """
    return all(os.path.isfile(os.path.join(model_dir, filename)) for filename in MTP_MODEL_FILES)


def validate_mtp(model_dir: str, device: str, mtp: dict | None,
                 scheduler_config: dict | None) -> None:
    """Reject an MTP profile this model/device/pipeline combination cannot run.

    Every rule here is enforced by openvino_genai as well -- the point is *where*. The
    runtime's checks fire inside `mtp_strategy.cpp` after the pipeline has loaded, which
    on a 14 GB int4 export costs a minute per case to learn that the profile was never
    runnable. These are pure filesystem and dict checks, so they cost nothing.
    """
    if not (mtp or {}).get("enabled"):
        return
    if not has_mtp_head(model_dir):
        missing = [
            filename for filename in MTP_MODEL_FILES
            if not os.path.isfile(os.path.join(model_dir, filename))
        ]
        raise ValueError(
            f"mtp is enabled but {model_dir} is missing {missing}; this export "
            "carries no draft head, so there is nothing to speculate with"
        )
    if device.upper().startswith("NPU"):
        raise ValueError(
            "mtp is enabled on NPU, but the Qwen3.8 experimental MTP path supports "
            "CPU and GPU only"
        )
    draft_device = mtp.get("device")
    if draft_device and draft_device.upper() != device.upper():
        raise ValueError(
            "mtp draft_model must use the same device as the target pipeline; "
            "the Qwen3.8 MTP workflow does not support cross-device speculation"
        )
    # PA is not a tuning preference here: genai refuses speculative decoding on the SDPA
    # backend for anything but a Gemma4 MTP pair, and a SchedulerConfig is what selects PA.
    if not scheduler_config:
        raise ValueError(
            "mtp is enabled without a `scheduler` section, which leaves the profile on "
            "the stateful (SDPA) pipeline. OpenVINO GenAI only runs this speculative "
            "decoding path on paged attention -- give the profile a `scheduler` "
            'section and set ATTENTION_BACKEND: PA under `ov`'
        )


def classify_error(exc: Exception) -> str:
    text = str(exc).lower()
    if any(marker in text for marker in _UNSUPPORTED_MARKERS):
        return "unsupported"
    if any(marker in text for marker in _OOM_MARKERS):
        return "oom"
    if any(marker in text for marker in _GPU_ABORT_MARKERS):
        return "gpu_abort"
    return "error"


def _device_diagnostics(device: str, ov_config: dict) -> dict:
    """What the plugin says about `device` and the properties it is handed.

    Read here rather than in the parent because initializing a GPU context in the
    orchestrator would pollute the very memory baseline it exists to measure.
    `gpu_budget_gb` is GPU_DEVICE_TOTAL_MEM_SIZE; on a shared-memory iGPU the driver
    under-reports what the platform actually makes available, so the parent treats it
    as a reference column and enforces its own configured budget instead.

    Best-effort: any failure yields empty diagnostics, never a failed case.
    """
    info = {"gpu_budget_gb": None, "unsupported_properties": []}
    try:
        import openvino as ov

        core = ov.Core()
        if ov_config:
            supported = set(core.get_property(device, "SUPPORTED_PROPERTIES"))
            info["unsupported_properties"] = [
                k for k in ov_config
                if k not in supported and k not in _GENAI_LEVEL_PROPERTIES
            ]
        if device.upper().startswith("GPU"):
            total_bytes = core.get_property(device, "GPU_DEVICE_TOTAL_MEM_SIZE")
            info["gpu_budget_gb"] = round(float(total_bytes) / (1024 ** 3), 2)
    except Exception:  # noqa: BLE001 - diagnostics must never fail a case
        pass
    return info


def _load_tokenizer(model_dir: str, trust_remote_code: bool = True):
    """Load the tokenizer, working around two `optimum-cli export openvino` quirks:
    `extra_special_tokens` written as a list where transformers wants a dict, and a
    declared `tokenizer_class` AutoTokenizer cannot resolve even though tokenizer.json
    is a valid fast-tokenizer file. Try progressively more permissive loaders rather
    than assuming which quirk, if any, is present.

    Used only to size the prompt; the model's own openvino_tokenizer.xml handles
    inference. transformers' mismatched-class warning during the PreTrainedTokenizerFast
    fallback is therefore expected and is silenced to keep the log readable.
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()

    attempts = [
        (AutoTokenizer, {}),
        (AutoTokenizer, {"extra_special_tokens": {}}),
        (PreTrainedTokenizerFast, {}),
        (PreTrainedTokenizerFast, {"extra_special_tokens": {}}),
    ]
    last_exc = None
    for tokenizer_cls, extra_kwargs in attempts:
        try:
            return tokenizer_cls.from_pretrained(
                model_dir, trust_remote_code=trust_remote_code, **extra_kwargs
            )
        except (ValueError, AttributeError) as exc:
            last_exc = exc
    raise last_exc


def _load_pipeline(model_dir: str, device: str, ov_config: dict,
                   scheduler_config: dict | None, mtp: dict | None = None):
    """Pick LLMPipeline or VLMPipeline from the IR layout on disk.

    optimum-cli decides whether a candidate exports as a plain causal LM
    (openvino_model.xml) or a multimodal/VLM layout (openvino_language_model.xml plus
    vision components). The returned flag records whether generate() accepts the
    TokenizedInputs used by llm_bench; VLMPipeline only accepts text.

    An empty `scheduler_config` is meaningful, not a default: it leaves continuous
    batching off entirely and runs the stateful pipeline, which skips paged-attention
    block management. That is a distinct configuration to benchmark, not an omission.
    The choice is all-or-nothing -- passing a SchedulerConfig with a single key in it is
    still the paged backend -- which is why a bounded KV pool (`cache_size`, a
    SchedulerConfig-only property) and the stateful pipeline are separate profiles rather
    than one configuration. See benchmark.pipeline_mode.

    With `mtp` enabled the *same* directory is handed back as the draft model. That is not
    a shortcut: multi-token prediction is self-speculative decoding, and GenAI recognizes
    `openvino_mtp_model.xml` in that export. Keep this call aligned with the official
    Qwen3.8 MTP notebook; passing the internal `mtp_mode=True` kwarg crashes this runtime.
    """
    import openvino_genai as ov_genai

    pipeline_args = dict(ov_config)
    if scheduler_config:
        scheduler = ov_genai.SchedulerConfig()
        for key, value in scheduler_config.items():
            if not hasattr(scheduler, key):
                raise ValueError(f"Unsupported SchedulerConfig property: {key}")
            try:
                setattr(scheduler, key, value)
            except TypeError as exc:
                # pybind's own message names neither the property nor the config that set
                # it, which turns a one-character config mistake into a stack trace.
                raise ValueError(
                    f"SchedulerConfig.{key} rejected {value!r} ({type(value).__name__}): {exc}"
                ) from exc
        pipeline_args["scheduler_config"] = scheduler

    if (mtp or {}).get("enabled"):
        pipeline_args["draft_model"] = ov_genai.draft_model(
            model_dir, device
        )

    if os.path.exists(os.path.join(model_dir, "openvino_language_model.xml")):
        return ov_genai.VLMPipeline(model_dir, device=device, **pipeline_args), False
    if os.path.exists(os.path.join(model_dir, "openvino_model.xml")):
        return ov_genai.LLMPipeline(model_dir, device=device, **pipeline_args), True
    raise RuntimeError(
        f"Unrecognized OpenVINO IR layout in {model_dir}: expected openvino_model.xml "
        "(plain LLM) or openvino_language_model.xml (multimodal/VLM export)"
    )


def prepare_pipeline_input(pipe, prompt: str, accepts_tokenized_input: bool):
    """Tokenize once with the pipeline tokenizer and return the exact inference input.

    LLMPipeline accepts TokenizedInputs, matching llm_bench and guaranteeing that the
    token sequence counted here is the sequence sent to the model. VLMPipeline only
    accepts text, so its validated string is returned and the pipeline tokenizes it.
    """
    tokenized = pipe.get_tokenizer().encode(prompt)
    prompt_tokens = int(tokenized.input_ids.shape[-1])
    return (tokenized if accepts_tokenized_input else prompt), prompt_tokens


def generation_config(output_tokens: int, mtp: dict | None = None):
    """Build the deterministic request config used by the Qwen3.8 MTP notebook.

    A fresh config is intentional: the model's generation_config.json enables sampling,
    while MTP requires greedy decoding. Reusing the pipeline-owned config risks retaining
    model defaults or fields changed by an earlier request. `apply_chat_template` is the
    one benchmark-specific difference from the notebook because our prompt is already
    rendered to an exact token count before it reaches VLMPipeline.
    """
    import openvino_genai as ov_genai

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = output_tokens
    config.do_sample = False
    config.num_return_sequences = 1
    config.assistant_confidence_threshold = 0.0
    config.num_assistant_tokens = (
        mtp["num_assistant_tokens"] if (mtp or {}).get("enabled") else 0
    )
    if hasattr(config, "apply_chat_template"):
        config.apply_chat_template = False
    return config


def generated_token_count(result, tokenizer) -> int:
    """Count encoded output tokens, falling back to pipeline-tokenizer text encoding."""
    tokens = getattr(result, "tokens", None)
    if tokens is not None:
        try:
            return len(tokens[0])
        except (IndexError, TypeError):
            pass
    output_text = generated_text(result)
    return int(tokenizer.encode(output_text).input_ids.shape[-1])


def generated_text(result) -> str:
    """Return the first generated text, matching the notebook's comparison target."""
    texts = getattr(result, "texts", None)
    return texts[0] if texts else str(result)


def _mean_ms(pair) -> float | None:
    """Read a MeanStdPair from perf_metrics; None if the runtime did not fill it in."""
    value = getattr(pair, "mean", None)
    return float(value) if isinstance(value, (int, float)) else None


def read_perf_metrics(result) -> dict:
    """OpenVINO GenAI's own measurement of a generate() call, in llm_bench's units.

    This is the standardized source -- the same one llm_bench reads -- so it wins over
    wall-clock timing wherever the runtime provides it. Every field is optional: older
    or partial runtime builds leave some unset, and the caller falls back per field
    rather than discarding the whole record.

    `input_size` is the runtime's own count of the tokens it prefilled, which is not
    redundant with the prompt measured before the call: on the VLMPipeline path the
    pipeline tokenizes the string itself (see prepare_pipeline_input), so the count taken
    here is what was requested, not necessarily what ran -- and prefill/e2e throughput
    divide by it.

    Newer GenAI builds expose MTP's draft acceptance directly on
    `result.extended_perf_metrics`. The verification-step count has no public getter, so it
    remains a best-effort compatibility metric read from `raw_metrics.m_new_token_times`.
    """
    out = {}
    perf = getattr(result, "perf_metrics", None)
    if perf is None:
        return out
    for key, getter in (
        ("first_token_latency", "get_ttft"),
        ("other_tokens_avg_latency", "get_tpot"),
        ("tokenization_time", "get_tokenization_duration"),
        ("detokenization_time", "get_detokenization_duration"),
    ):
        try:
            value = _mean_ms(getattr(perf, getter)())
        except Exception:  # noqa: BLE001 - a missing metric is not a failed case
            value = None
        if value is not None:
            out[key] = value
    try:
        generated = perf.get_num_generated_tokens()
        if isinstance(generated, int) and generated > 0:
            out["output_size"] = generated
    except Exception:  # noqa: BLE001
        pass
    try:
        consumed = perf.get_num_input_tokens()
        if isinstance(consumed, int) and consumed > 0:
            out["input_size"] = consumed
    except Exception:  # noqa: BLE001
        pass
    try:
        duration = _mean_ms(perf.get_generate_duration())
        if duration is not None:
            out["generation_time"] = duration / 1000.0
    except Exception:  # noqa: BLE001
        pass
    try:
        extended = result.extended_perf_metrics
        acceptance = extended.get_draft_acceptance_rate()
        if isinstance(acceptance, (int, float)) and math.isfinite(acceptance):
            out["mtp_acceptance_rate"] = float(acceptance)
        draft_tokens = extended.get_num_draft_tokens()
        accepted_tokens = extended.get_num_accepted_tokens()
        if isinstance(draft_tokens, int) and draft_tokens >= 0:
            out["mtp_draft_tokens"] = draft_tokens
        if isinstance(accepted_tokens, int) and accepted_tokens >= 0:
            out["mtp_accepted_tokens"] = accepted_tokens
    except Exception:  # noqa: BLE001
        pass
    try:
        steps = len(perf.raw_metrics.m_new_token_times)
        if steps > 0:
            out["verification_steps"] = steps
    except Exception:  # noqa: BLE001
        pass
    return out


def _post_and_exit(result_queue, message: dict) -> None:
    """Hand the result to the parent, then leave *without* OpenVINO's teardown. Never returns.

    Destroying a GPU pipeline that has just prefilled a very long context can throw an
    ov::Exception out of a destructor. Observed on the 64 GB shared-memory iGPU box at
    160K, immediately after a clean 128K pass:

        openvino_genai.dll!ov::genai::VLMPipeline::~VLMPipeline
          -> openvino.dll!ov::IAsyncInferRequest::~IAsyncInferRequest
          -> openvino_intel_gpu_plugin.dll!...
          -> openvino.dll!ov::Exception::create        <- throws out of a destructor
          -> ucrtbase.dll!terminate                    <- nothing above can catch it
        exit code 3221226505 (0xC0000409)

    An escaping exception in a destructor is std::terminate, not a Python exception, so
    no try/except could contain it -- and because the teardown ran before the result was
    posted, it also destroyed a measurement that had already completed. Posting first and
    skipping the teardown fixes both. Nothing leaks: the child exists to run one case and
    the parent relies on process exit -- not Python cleanup -- to reclaim memory.
    """
    try:
        result_queue.put(message)
        # Queue.put() is asynchronous: a feeder thread copies into the pipe, and os._exit()
        # skips the atexit hook that waits for it. Flush explicitly or the result races the exit.
        result_queue.close()
        result_queue.join_thread()
    except Exception:  # noqa: BLE001 - the parent's crash path is the fallback
        traceback.print_exc(file=sys.stderr)
    finally:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:  # noqa: BLE001
                pass
        os._exit(0)


def run_case(
    model_dir: str,
    device: str,
    context_tokens: int,
    output_tokens: int,
    warmup: int,
    iterations: int,
    result_queue,
    ov_config: dict | None = None,
    scheduler_config: dict | None = None,
    mtp: dict | None = None,
) -> None:
    """Load once, generate ``warmup + iterations`` times, report each one.

    Messages posted to `result_queue`:
      ``device``          plugin diagnostics, before anything is loaded
      ``loaded``          pipeline constructed (parent snapshots post-load memory here)
      ``prompt``          context tokenized to exactly `context_tokens`
      ``iteration_start`` about to generate (parent opens a fresh memory window)
    ``prefilled``       first token produced (parent can distinguish decode failures)
      ``iteration``       one completed generation, as a metrics.iteration_record
      ``done``            terminal: every record, plus any error

    Does not return: every path exits via `_post_and_exit()`, which posts `done` and ends
    the process before OpenVINO's destructors can run. `pipe` is deliberately left alive.
    """
    ov_config = dict(ov_config or {})
    done = {
        "event": "done",
        "context_tokens": context_tokens,
        "load_ok": False,
        "load_time_s": None,
        # Unknown until the prompt has been built and tokenized.
        "prompt_tokens": None,
        "iterations": [],
        "stage_reached": STAGE_START,
        "error": None,
    }

    diagnostics = _device_diagnostics(device, ov_config)
    if diagnostics["unsupported_properties"]:
        print(
            f"[trial_runner] {device} does not advertise: "
            f"{', '.join(diagnostics['unsupported_properties'])} -- passing them anyway; "
            "if the pipeline refuses to build, they are the first thing to remove",
            file=sys.stderr,
        )
    result_queue.put({"event": "device", **diagnostics})
    done["gpu_budget_gb"] = diagnostics["gpu_budget_gb"]

    try:
        # Before the clock starts: a profile this model cannot run is a configuration
        # error, not a load time worth reporting.
        validate_mtp(model_dir, device, mtp, scheduler_config)
        t0 = time.perf_counter()
        tokenizer = _load_tokenizer(model_dir)
        pipe, accepts_tokenized_input = _load_pipeline(
            model_dir, device, ov_config, scheduler_config, mtp
        )
        done["load_ok"] = True
        done["load_time_s"] = round(time.perf_counter() - t0, 3)
        done["stage_reached"] = STAGE_LOADED
        result_queue.put({"event": "loaded", "load_time_s": done["load_time_s"]})
    except Exception as exc:  # noqa: BLE001 - reported to the parent, not re-raised
        print(f"[trial_runner] load failed: {traceback.format_exc()}", file=sys.stderr)
        done["error"] = f"load:{classify_error(exc)}:{exc}"
        _post_and_exit(result_queue, done)

    import openvino_genai as ov_genai

    try:
        # Built once and reused across iterations, as llm_bench does. Requires prefix
        # caching to stay off, or iteration 1 inherits the warm-up's cached prefix.
        prompt, hf_tokens = build_benchmark_prompt(tokenizer, context_tokens)
        pipeline_input, prompt_tokens = prepare_pipeline_input(
            pipe, prompt, accepts_tokenized_input
        )
        done["prompt_tokens"] = prompt_tokens
        if hf_tokens != context_tokens or prompt_tokens != context_tokens:
            raise ValueError(
                "Prompt token count mismatch: "
                f"requested={context_tokens}, huggingface={hf_tokens}, openvino={prompt_tokens}"
            )
        done["stage_reached"] = STAGE_PROMPT_BUILT
        result_queue.put({"event": "prompt", "prompt_tokens": prompt_tokens})

        # Plain greedy decoding. Proving and timing prefill + decode is the whole goal, and
        # grammar-constrained decoding was observed to collapse into "!!!!" on some models,
        # which would score a false failure for a context the box handled.
        gen_config = generation_config(output_tokens, mtp)
        warmup_config = generation_config(min(4, output_tokens), mtp)
        assistant_tokens = (
            mtp["num_assistant_tokens"] if (mtp or {}).get("enabled") else None
        )

        # Warned about once per case, not once per iteration: every iteration reuses the
        # one prompt, so a divergence is a property of the case.
        input_size_reported = False

        for index in range(warmup + iterations):
            is_warmup = index < warmup
            result_queue.put({"event": "iteration_start", "iteration": index, "warmup": is_warmup})

            ttft_ms = None
            t1 = time.perf_counter()

            def _on_token(_: str):
                """First call = prefill is over; the whole context made it through the model.

                Records only *when*, never asks generation to stop -- stopping early would
                cut short the decode phase being measured.
                """
                nonlocal ttft_ms
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t1) * 1000.0
                    done["stage_reached"] = STAGE_PREFILLED
                    result_queue.put({"event": "prefilled", "iteration": index})
                return ov_genai.StreamingStatus.RUNNING

            result = pipe.generate(
                pipeline_input,
                generation_config=warmup_config if is_warmup else gen_config,
                streamer=_on_token,
            )
            wall_seconds = time.perf_counter() - t1
            done["stage_reached"] = STAGE_DECODED

            perf = read_perf_metrics(result)
            output_size = perf.get("output_size") or generated_token_count(
                result, pipe.get_tokenizer()
            )
            if not output_size:
                raise RuntimeError("no_output: generate() produced no tokens")

            # Prefer what the runtime says it prefilled over what was asked for: on the
            # VLMPipeline path the pipeline re-tokenizes the prompt string, so the two can
            # differ, and reporting a throughput per *requested* token would divide by a
            # number no forward pass ever saw. Not fatal -- the measurement is real either
            # way, and iterations.csv's input_size now says which context length it is of.
            input_size = perf.get("input_size") or prompt_tokens
            if input_size != prompt_tokens and not input_size_reported:
                input_size_reported = True
                print(
                    f"[trial_runner] runtime prefilled {input_size:,} tokens where the "
                    f"prompt measured {prompt_tokens:,}; throughputs use the runtime's "
                    "count. Expected on the VLMPipeline path, which tokenizes the string "
                    "itself rather than accepting TokenizedInputs.",
                    file=sys.stderr,
                )

            record = metrics.iteration_record(
                iteration=index,
                input_size=input_size,
                output_size=output_size,
                generation_time=perf.get("generation_time", wall_seconds),
                first_token_latency=perf.get("first_token_latency", ttft_ms),
                other_tokens_avg_latency=perf.get("other_tokens_avg_latency"),
                tokenization_time=perf.get("tokenization_time", 0.0),
                detokenization_time=perf.get("detokenization_time", 0.0),
                warmup=is_warmup,
                num_assistant_tokens=assistant_tokens,
                verification_steps=perf.get("verification_steps"),
                mtp_acceptance_rate=perf.get("mtp_acceptance_rate"),
                mtp_draft_tokens=perf.get("mtp_draft_tokens"),
                mtp_accepted_tokens=perf.get("mtp_accepted_tokens"),
                output_sha256=hashlib.sha256(
                    generated_text(result).encode("utf-8")
                ).hexdigest(),
            )
            done["iterations"].append(record)
            result_queue.put({"event": "iteration", **record})
    except Exception as exc:  # noqa: BLE001
        stage = failing_stage(done["stage_reached"])
        print(f"[trial_runner] {stage} failed: {traceback.format_exc()}", file=sys.stderr)
        done["error"] = f"{stage}:{classify_error(exc)}:{exc}"

    # No `del pipe` / gc.collect() here on purpose -- see _post_and_exit().
    _post_and_exit(result_queue, done)
