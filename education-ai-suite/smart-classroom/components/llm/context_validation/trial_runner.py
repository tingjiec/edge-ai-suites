# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Runs ONE (model, context length) capacity trial for the long-context validator.

The question a trial answers is deliberately narrow, mirroring
refer/long_context/validate_long_context.py: *can this hardware load the model,
prefill a prompt of the requested token size, and decode a few tokens without
running out of memory (or hanging)?* It does NOT judge answer quality -- content
is irrelevant, only whether the box survives the token volume.

Executed as a fresh subprocess per trial (see validate_long_context.py) so that
GPU/host memory from a crashed or OOM'd attempt can never bleed into the next
trial -- the same "convert in a subprocess so memory is fully reclaimed on exit"
rationale already used for model conversion in
components/vlm/vlm_openvino_serving/utils/utils.py::_convert_model_worker.

Memory is sampled by the *orchestrator* (parent), not here: system RAM / GPU
counters are process-wide, so the parent sees this child's footprint just as
well, and -- crucially -- its readings survive even when this child is killed on
a timeout. This child signals milestones over the queue so the parent can time
its snapshots and know how far the trial got even when it never reports:
"device" (what the plugin says about the device and the requested properties),
"loaded" once the pipeline is constructed (so the parent can measure subsequent
peak growth), "prompt" once the context is tokenized, "prefilled" the
moment the first token is streamed out, and finally "done" with the outcome.

The prefill/decode split matters because those two phases fail for different
reasons and are told apart by nothing else: prefill processes the whole context
in one forward pass -- the peak-memory moment of the trial -- while decode adds
one token at a time to an already-allocated cache. A pipeline that dies inside
`generate()` used to be reported as a single opaque "generate" failure, so an
abort during the first forward pass over 144K tokens looked exactly like one on
decode step 60. The streamer callback below draws the line: no token streamed
yet means the failure happened in prefill.

Which OpenVINO plugin properties the pipeline is built with is the parent's
decision (`ov_config`), not a constant here -- KV cache precision in particular
is one of the knobs a capacity sweep exists to compare.

Because process exit is the resource-reclamation boundary, this module never
tears the pipeline down itself: it reports its result and then calls os._exit()
(see _post_result_and_exit, which documents the native abort that made running
the teardown actively harmful).

OpenVINO / transformers are imported lazily inside run_trial() rather than at
module scope, so this module -- and therefore validate_long_context.py, which
imports it -- can be imported (e.g. for --dry-run) on a machine that doesn't
have the OpenVINO stack installed.
"""

from __future__ import annotations

import os
import sys
import time
import traceback

from components.llm.context_validation.context_builder import build_context_prompt

# Text that says outright that the memory was never handed over. Includes OpenCL's own
# vocabulary, because at the capacity ceiling the exception the pipeline raises is the GPU
# plugin's, not Python's, and it reaches us as the plain text of an ov::Exception.
_OOM_MARKERS = (
    "out of gpu resources",
    "out of memory",
    "allocation failed",
    "bad_alloc",
    "cannot allocate",
    "can't allocate",
    "insufficient memory",
    "memoryerror",
    "cl_mem_object_allocation_failure",  # -4: the device could not back the buffer
    "cl_out_of_host_memory",  # -6: the runtime could not back it on the host either
    "cl_invalid_buffer_size",  # -61: one buffer larger than the device permits
)

# A GPU *command* that was accepted and then died on the device. OpenCL surfaces this on the
# next synchronization point rather than at enqueue, which is why the message names the wait
# (`clWaitForEvents`) and never names what actually ran out:
#
#     Exception from src\\plugins\\intel_gpu\\src\\runtime\\ocl\\ocl_memory.cpp:591:
#     [GPU] clWaitForEvents, error code: -14 CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST
#
# On a shared-memory iGPU at its ceiling this is the usual face of memory exhaustion, but a
# driver reset (TDR) produces the same code, and this process cannot tell the two apart: it
# sees neither the host's free-RAM low-water mark nor the GPU counters, both of which are
# sampled by the parent. So it reports the honest, narrower fact -- the device aborted the
# command -- while the orchestrator reports memory pressure independently rather than rewriting
# correlation as an OOM diagnosis (see validate_long_context._classify_failure).
_GPU_ABORT_MARKERS = (
    "cl_exec_status_error_for_events_in_wait_list",  # -14
    "cl_out_of_resources",  # -5
    "cl_invalid_command_queue",  # -36, typical after a device reset
    "clwaitforevents",
    "clfinish",
)


# How far the trial got. Reported as `stage_reached` on every result -- including the ones the
# parent has to synthesize for a child that died or hung without reporting, which is where it
# earns its keep: "timed out at 1200s" says nothing, "timed out at 1200s having reached prefill"
# says the box never finished one forward pass over the context.
STAGE_START = "start"  # nothing done yet; a failure here is the plugin refusing to build
STAGE_LOADED = "loaded"  # pipeline constructed
STAGE_PROMPT_BUILT = "prompt_built"  # context tokenized, nothing sent to the device yet
STAGE_PREFILLED = "prefilled"  # first token streamed out: the whole context is through the model
STAGE_DECODED = "decoded"  # generate() returned

# The stage a failure belongs to, given the last milestone reached. Off-by-one on purpose: an
# exception is thrown by the step *after* the last completed milestone, so a crash with the prompt
# built but no token streamed is a prefill failure, not a "prompt_built" one.
_FAILING_STAGE = {
    STAGE_START: "load",
    STAGE_LOADED: "prompt",
    STAGE_PROMPT_BUILT: "prefill",
    STAGE_PREFILLED: "decode",
    STAGE_DECODED: "decode",
}


def failing_stage(stage_reached: str) -> str:
    """Name the stage that was running when a trial at `stage_reached` failed."""
    return _FAILING_STAGE.get(stage_reached, "generate")


def _classify_error(exc: Exception) -> str:
    text = str(exc).lower()
    if any(marker in text for marker in _OOM_MARKERS):
        return "oom"
    if any(marker in text for marker in _GPU_ABORT_MARKERS):
        return "gpu_abort"
    return "exception"


def _device_diagnostics(device: str, ov_config: dict) -> dict:
    """What the plugin says about `device` and about the properties it's being handed.

    Two things the parent cannot find out for itself without initializing a GPU context in
    the orchestrator process -- which would pollute the very memory baseline it exists to
    measure -- so they're read here, in the process that is about to build a pipeline anyway:

    * `gpu_budget_gb` (GPU_DEVICE_TOTAL_MEM_SIZE): the memory the driver will hand this device.
      On a shared-memory iGPU this is a fraction of system RAM, not all of it, and a long-context
      run reaches it long before the host runs out -- so it is the ceiling that explains a device
      abort whose message never mentions memory.
    * `unsupported_properties`: keys of `ov_config` the device does not advertise. A warning
      rather than an error, because pipeline-level keys handled by OpenVINO GenAI itself
      (ATTENTION_BACKEND, scheduler settings) legitimately don't appear in a plugin's
      SUPPORTED_PROPERTIES; if the key really is wrong, the constructor throws next and this
      list is what explains the throw.

    Best-effort throughout: any failure here yields empty diagnostics, never a failed trial.
    """
    info = {"gpu_budget_gb": None, "unsupported_properties": []}
    try:
        import openvino as ov

        core = ov.Core()
        if ov_config:
            supported = set(core.get_property(device, "SUPPORTED_PROPERTIES"))
            info["unsupported_properties"] = [k for k in ov_config if k not in supported]
        if device.upper().startswith("GPU"):
            total_bytes = core.get_property(device, "GPU_DEVICE_TOTAL_MEM_SIZE")
            info["gpu_budget_gb"] = round(float(total_bytes) / (1024 ** 3), 2)
    except Exception:  # noqa: BLE001 - diagnostics must never be the reason a trial fails
        pass
    return info


def _validate_generated_output(output: str, tokenizer) -> tuple[bool, int, str | None]:
    """Confirm that decode produced at least one token.

    This is a capacity probe, not an answer-quality evaluation. Punctuation, repetition, an
    immediate EOS, or otherwise low-information text still proves that prefill completed and
    decode ran; rejecting it would turn model behavior into a false hardware-capacity failure.
    """
    if not output or not output.strip():
        return False, 0, "no_output"

    token_ids = tokenizer.encode(output, add_special_tokens=False)
    if not token_ids:
        return False, 0, "no_output_tokens"
    return True, len(token_ids), None


def _load_tokenizer(model_dir: str, trust_remote_code: bool = True):
    """Load the tokenizer, working around two OpenVINO-export quirks seen on
    real `optimum-cli export openvino` output (neither is specific to one
    candidate model, so try progressively more permissive loaders rather than
    assuming which quirk, if any, is present):

    1. `extra_special_tokens` written as a list where transformers expects a
       dict -- same workaround as components/vlm/text_gen_vlm.py::VLMTextGen._load.
    2. `tokenizer_config.json`'s declared `tokenizer_class` (e.g.
       "TokenizersBackend") isn't a class AutoTokenizer can resolve, even
       though `tokenizer.json` is a perfectly valid fast-tokenizer file --
       load it directly via PreTrainedTokenizerFast, which doesn't need to
       resolve a class name.

    The tokenizer is used only to size the prompt and count tokens; the model's
    own openvino_tokenizer.xml handles real inference. transformers' "tokenizer
    class you load ... is not the same type as the class this function is called
    from" warning during the PreTrainedTokenizerFast fallback is therefore
    expected and harmless, so it's silenced here to keep the sweep log readable
    (it would otherwise repeat for every trial's fresh subprocess).
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


def _load_pipeline(model_dir: str, device: str, ov_config: dict, scheduler_config: dict | None = None):
    """Pick LLMPipeline or VLMPipeline based on which IR layout is on disk.

    A candidate model can convert to a plain causal-LM export
    (openvino_model.xml) or a multimodal/VLM export (openvino_language_model.xml
    plus vision-embedding components) depending on the model's own architecture
    -- optimum-cli decides this, not us. Both pipeline classes expose the same
    .generate(prompt, generation_config=...) surface used below, so the rest of
    run_trial doesn't need to know which one it got.
    """
    import openvino_genai as ov_genai

    pipeline_args = dict(ov_config)
    if scheduler_config:
        scheduler = ov_genai.SchedulerConfig()
        for key, value in scheduler_config.items():
            if not hasattr(scheduler, key):
                raise ValueError(f"Unsupported SchedulerConfig property: {key}")
            setattr(scheduler, key, value)
        pipeline_args["scheduler_config"] = scheduler

    if os.path.exists(os.path.join(model_dir, "openvino_language_model.xml")):
        return ov_genai.VLMPipeline(model_dir, device=device, **pipeline_args)
    if os.path.exists(os.path.join(model_dir, "openvino_model.xml")):
        return ov_genai.LLMPipeline(model_dir, device=device, **pipeline_args)
    raise RuntimeError(
        f"Unrecognized OpenVINO IR layout in {model_dir}: expected openvino_model.xml "
        "(plain LLM) or openvino_language_model.xml (multimodal/VLM export)"
    )


def _post_result_and_exit(result_queue, message: dict) -> None:
    """Hand the trial result to the parent, then leave the process *without*
    running OpenVINO's teardown. Never returns.

    Destroying an OpenVINO GPU pipeline that has just prefilled a very long
    context can throw an `ov::Exception` from inside a destructor. Observed on
    the 64 GB shared-memory iGPU box at 160K tokens, immediately after a clean
    128K pass:

        openvino_genai.dll!ov::genai::VLMPipeline::~VLMPipeline
          -> openvino.dll!ov::IAsyncInferRequest::~IAsyncInferRequest
          -> openvino.dll!ov::ISyncInferRequest::~ISyncInferRequest
          -> openvino_intel_gpu_plugin.dll!...
          -> openvino.dll!ov::Exception::create        <- throws out of a destructor
          -> ucrtbase.dll!terminate                    <- nothing above can catch it
        exit code 3221226505 (0xC0000409)

    An escaping exception in a destructor is `std::terminate`, not a Python
    exception, so the old `finally: try: del pipe ... except Exception: pass`
    could not contain it -- the process was gone before the next bytecode ran.
    And because that teardown ran *before* the `done` message was posted, the
    abort also destroyed a result the trial had already finished computing: the
    orchestrator saw only `crashed`, with no way to tell whether 160K had in
    fact prefilled and decoded successfully.

    Both halves are fixed by this function. The result is posted first, so an
    abort can no longer erase a completed measurement, and the teardown that
    throws is not run at all. Skipping it does not leak anything: it is the
    design this module already documents. The child exists to run exactly
    one trial, and the orchestrator relies on process exit -- not on Python-level
    cleanup -- to reclaim GPU/host memory between trials. `os._exit()` hands that
    reclamation to the OS, which cannot throw. (The orchestrator waits for the
    reclamation to land before starting the next trial; see
    `validate_long_context._wait_for_memory_settle`.)
    """
    try:
        result_queue.put(message)
        # Queue.put() is asynchronous: a feeder thread copies the message into the
        # pipe. os._exit() skips the atexit hook that normally waits for that thread,
        # so flush it explicitly here or the result races the process exit.
        result_queue.close()
        result_queue.join_thread()
    except Exception:  # noqa: BLE001 - the parent's crash path is the fallback
        traceback.print_exc(file=sys.stderr)
    finally:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:  # noqa: BLE001 - flushing must not block the exit
                pass
        os._exit(0)


def run_trial(
    model_dir: str,
    model_name: str,
    device: str,
    tokens: int,
    probe_tokens: int,
    result_queue,
    ov_config: dict | None = None,
    scheduler_config: dict | None = None,
) -> None:
    """Load the model, prefill a ~`tokens`-token prompt, decode up to
    `probe_tokens` tokens, and report the outcome over `result_queue`.

    `ov_config` is the OpenVINO plugin property map the pipeline is built with
    (KV_CACHE_PRECISION, GPU_ENABLE_LARGE_ALLOCATIONS, ...), resolved by the
    orchestrator from its config file so the same trial can be re-measured under
    a different memory configuration without touching this module.

    Posts up to five messages: `device` (plugin diagnostics, before anything is
    loaded), `loaded` once the weights are resident, `prompt` once the context is
    tokenized, `prefilled` the moment the first token is streamed -- i.e. the whole
    context made it through the model -- and finally `done` with the result. All
    but `done` are milestones: they let the parent time its memory snapshots and,
    when this process dies without reporting, still say how far it got.

    Finding the memory ceiling only needs a few decode steps, not a full summary,
    so `probe_tokens` is small on purpose -- generating thousands of tokens at
    128K+ context would add many minutes per step for no extra signal.

    Does not return: every exit path goes through `_post_result_and_exit()`,
    which posts `done` and then ends the process before OpenVINO's destructors
    can run. `pipe` is deliberately left alive.
    """
    ov_config = dict(ov_config or {})
    done = {
        "event": "done",
        "tokens_requested": tokens,
        "load_ok": False,
        "load_time_s": None,
        "generate_ok": False,
        "prompt_tokens": 0,
        "generated_tokens": 0,
        "generate_time_s": None,
        "time_to_first_token_s": None,
        "decode_time_s": None,
        "stage_reached": STAGE_START,
        "error": None,
    }

    diagnostics = _device_diagnostics(device, ov_config)
    if diagnostics["unsupported_properties"]:
        print(
            f"[trial_runner] {device} does not advertise these requested properties: "
            f"{', '.join(diagnostics['unsupported_properties'])} -- passing them anyway; if the "
            "pipeline refuses to build, they are the first thing to remove",
            file=sys.stderr,
        )
    result_queue.put({"event": "device", **diagnostics})
    done["gpu_budget_gb"] = diagnostics["gpu_budget_gb"]

    try:
        t0 = time.perf_counter()
        tokenizer = _load_tokenizer(model_dir)
        pipe = _load_pipeline(model_dir, device, ov_config, scheduler_config)
        done["load_ok"] = True
        done["load_time_s"] = round(time.perf_counter() - t0, 3)
        done["stage_reached"] = STAGE_LOADED
        # Tell the parent the pipeline is constructed so it can measure all
        # subsequent growth through prefill and decode.
        result_queue.put({"event": "loaded", "load_time_s": done["load_time_s"]})
    except Exception as exc:  # noqa: BLE001 - reported to orchestrator, not re-raised
        print(f"[trial_runner] load failed: {traceback.format_exc()}", file=sys.stderr)
        done["error"] = f"load:{_classify_error(exc)}:{exc}"
        _post_result_and_exit(result_queue, done)

    import openvino_genai as ov_genai

    try:
        prompt, prompt_tokens = build_context_prompt(tokenizer, tokens)
        done["prompt_tokens"] = prompt_tokens
        done["stage_reached"] = STAGE_PROMPT_BUILT
        result_queue.put({"event": "prompt", "prompt_tokens": prompt_tokens})

        # Plain greedy decoding, no structured output: proving the hardware can
        # prefill + decode this context is the only goal, and grammar-constrained
        # decoding was observed to collapse into garbage output ("!!!!") on some
        # models, which would score a false FAIL for a context the box handled.
        gen_config = ov_genai.GenerationConfig(max_new_tokens=probe_tokens, do_sample=False)
        t1 = time.perf_counter()

        def _on_token(_chunk: str):
            """First call = prefill is over; the whole context made it through the model.

            Everything this records is about *when*, not what -- the decoded text comes from
            generate()'s return value -- and it never asks generation to stop, which would cut
            the decode phase this trial is measuring short.
            """
            if done["time_to_first_token_s"] is None:
                done["stage_reached"] = STAGE_PREFILLED
                done["time_to_first_token_s"] = round(time.perf_counter() - t1, 3)
                result_queue.put(
                    {"event": "prefilled", "time_to_first_token_s": done["time_to_first_token_s"]}
                )
            return ov_genai.StreamingStatus.RUNNING

        output = str(pipe.generate(prompt, generation_config=gen_config, streamer=_on_token))
        done["generate_time_s"] = round(time.perf_counter() - t1, 3)
        done["stage_reached"] = STAGE_DECODED
        if done["time_to_first_token_s"] is not None:
            done["decode_time_s"] = round(
                done["generate_time_s"] - done["time_to_first_token_s"], 3
            )
        output_ok, generated_tokens, output_error = _validate_generated_output(output, tokenizer)
        done["generated_tokens"] = generated_tokens
        done["generate_ok"] = output_ok
        done["error"] = output_error
    except Exception as exc:  # noqa: BLE001
        stage = failing_stage(done["stage_reached"])
        print(f"[trial_runner] {stage} failed: {traceback.format_exc()}", file=sys.stderr)
        done["error"] = f"{stage}:{_classify_error(exc)}:{exc}"
        done["generate_ok"] = False

    # No `del pipe` / gc.collect() here on purpose -- destroying the pipeline at
    # this point is what aborted the process and threw away the result this line
    # is about to report. See _post_result_and_exit().
    _post_result_and_exit(result_queue, done)
