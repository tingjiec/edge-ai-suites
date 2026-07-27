# Long-Context Capacity Validator

`components/llm/context_validation/` is a standalone diagnostic tool that answers one
question: **for a given summarizer model, how many tokens of context can this machine's
hardware (primarily the Intel iGPU) prefill and decode within an explicit latency SLA,
and is that enough to meet the customer's 160K-token requirement?**

It is **not** part of the runtime pipeline. `pipeline.py` / `SummarizerComponent` never import
it, and it reads its own bundled
[`config.yaml`](../../components/llm/context_validation/config.yaml) next to the script —
never `smart-classroom/config.yaml`. Editing either file has no effect on the other: this is a
capacity-planning script you run manually before committing to a model choice, or whenever you
change device / weight format / hardware, and it can't drift the production config or be
affected by changes to it.

```
components/llm/context_validation/
  config.yaml                      # this tool's own config -- see §3
  context_builder.py               # synthetic transcript construction (exact token size)
  trial_runner.py                  # runs ONE (model, context length) trial, in a subprocess
  validate_long_context.py         # CLI orchestrator: stepping loop, reporting
  setup_env.ps1                    # prepares the backend venv (one-time)
  run_validate_long_context.ps1    # one-command launcher -- see §5
```

## 1. Purpose

`smart-classroom/config.yaml`'s `models.summarizer.name` lets you swap between candidate
OpenVINO models (`Qwen/Qwen3-8B`, `Qwen/Qwen3.6-35B-A3B`, `Qwen/Qwen3.5-9B`, ...), but swapping
the name alone doesn't tell you whether that model will actually survive a 160K-token classroom
transcript on the target machine. The model advertises a long context (160K); whether *this box*
can load it and prefill + decode a prompt that large **without exhausting the memory the iGPU can
allocate** is a property of the hardware, not the model — and that is exactly what this tool
measures.

This mirrors the reference harness in `refer/long_context`: the transcript content is irrelevant
to the check, only the token *volume* matters. The tool sweeps context length per model and
reports the largest size the hardware can sustain (load + prefill + decode a few tokens) before
it runs out of resources, hangs, or becomes too slow to be operationally useful. This matters on
shared-memory iGPUs: memory pressure can cause severe paging/thrashing without producing a clean
OOM, so "eventually returned one token" is not a meaningful capacity result.

> **Scope — capacity, not answer quality.** This tool deliberately does **not** score whether the
> model *understood* the long context (e.g. a needle-in-a-haystack recall test). Earlier revisions
> tried that with grammar-constrained decoding, but on some models the constrained decoder
> collapsed into garbage output (`!!!!`) and produced a *false* FAIL for a context the hardware
> had actually handled fine. A capacity check is simple, robust, and answers the question that
> actually blocks deployment ("does it fit on this box?"). Validate answer quality separately
> against a couple of real long transcripts before shipping (§8).

## 2. Quick start

```powershell
# From anywhere -- prepares the venv on first use, then runs the sweep
.\components\llm\context_validation\run_validate_long_context.ps1
```

That's the whole thing. See §5 for more commands (custom steps, `--dry-run`, etc.) and §4 if
you'd rather manage the Python environment yourself.

## 3. Design & Methodology

### 3.1 What one trial does

At each context length being tested, the tool builds a synthetic classroom transcript
(`TEACHER:` / `STUDENT_NN:` dialog lines) sized to the target token count using the *model's own
tokenizer*, wraps it in the chat template (system prompt + transcript as the user turn), then:

1. **Loads** the model fresh (records load time, or a load failure / OOM).
2. **Prefills** the full prompt and **decodes** up to `probe_tokens` tokens (default 64) with
  plain greedy decoding. The transcript and final task are explicitly delimited so truncated
  synthetic input still gives the model a meaningful instruction.
3. **Validates** that decoded output is non-empty natural-language content, rejecting output
  made only of special tokens, control characters, punctuation, or one repeated character.
4. **Passes** unless generation fails or both capacity-pressure signals occur together: the
  complete `generate()` call exceeds `max_generate_time_sec` (default 600s) **and** the box was
  at a memory ceiling while it ran — either sampled GPU memory reaching
  `gpu_memory_pressure_pct` (default 90% of system RAM) or measured host headroom being gone
  (§3.2.3). OOM, crash, hard trial timeout, and invalid output remain unconditional failures.
  No memory threshold can produce a failure on its own — see §3.2.1.

The console and CSV also report `tokens_per_second = generated_tokens / generate_time_s`. This is
an **effective end-to-end output rate that includes prefill**, not pure decode throughput. Models
may stop before `probe_tokens` when they emit EOS; the wall-clock SLA is therefore the stable
policy signal, while tok/s is diagnostic context.

`probe_tokens` is small on purpose: proving the box can hold a context size only needs a few
decode steps, not a full summary — generating thousands of tokens at 128K+ context would add
many minutes per step for no extra capacity signal. See
[`context_builder.py`](../../components/llm/context_validation/context_builder.py) and
[`trial_runner.py`](../../components/llm/context_validation/trial_runner.py).

### 3.1.1 Memory breakdown: weights vs. KV-cache

The core signal on hardware where the iGPU shares system RAM is *where the memory went*, so each
trial reports it split three ways:

- **Weights** — the RAM/GPU footprint measured the instant the model finishes loading, before any
  prefill. This is roughly constant across context sizes for a given model/weight-format, and is
  cross-checked against the on-disk IR weight size (`weights on disk`, the summed `.bin` bytes).
- **KV-cache** — the *additional* memory the peak reaches during prefill+decode, on top of the
  loaded weights. This is what grows with context length and is what eventually exhausts the box.
- **Peak** — the total high-water mark (weights + KV + everything else), i.e. how close the trial
  came to the hardware limit.

The child subprocess signals two milestones over its result queue — `loaded` (weights resident)
and `done` (trial finished) — and the orchestrator snapshots system memory at the `loaded`
milestone and tracks the running peak throughout, so weights (post-load delta from a pre-spawn
baseline) and KV-cache (peak minus post-load) fall straight out of those two snapshots. Sampling
in the *parent* rather than the child is deliberate: system RAM/GPU counters are process-wide, so
the parent sees the child's footprint just as well, and — crucially — its readings **survive even
when the child is killed on a timeout**, which is exactly the case where memory matters most (the
box was thrashing on a context it couldn't hold, not sitting idle). RAM comes from `psutil`; GPU
is best-effort and Windows-only via the repo's perf-counter collector (reads `0.0` elsewhere).

**Is a measured KV-cache number too big?** The measured `kv_ram_gb`/`kv_gpu_gb` is a system-level
delta, not a pure KV-cache tensor size — it also picks up prefill scratch memory and any other
post-load growth (see §9 caveats below). To make it possible to tell "this looks architecturally
expected" apart from "this looks inflated" without manually reading the model's `config.json`
every time, each trial also reports `expected_kv_gpu_gb` and `kv_overhead_ratio` when the model's
own `config.json` (already present next to every `optimum-cli`-exported IR) has enough
information: `2 × num_full_attention_layers × num_key_value_heads × head_dim × 2 bytes/token`
(fp16 KV-cache assumed — the tool has no way to read back the runtime's actual KV precision, this
is a stated assumption). Plain dense transformers count every layer. **Hybrid linear-attention
models** — `Qwen/Qwen3.5-9B` and `Qwen/Qwen3.6-35B-A3B`'s exported `config.json` (`text_config.
layer_types`) declare a repeating 3:1 `linear_attention`:`full_attention` pattern, i.e. only 1-in-4
layers is a genuine growing-KV-cache attention layer; the rest are Mamba/GatedDeltaNet-style
recurrent-state layers that should only need a small, constant-size state — so
`_theoretical_kv_bytes_per_token()` counts only the `full_attention` layers (confirmed against the
exported IR itself: only the `full_attention` layers' `cache_params.past.{key,value}.N` state
variables have a sequence-length axis; the `linear_attention` layers' `cache_params.past.{conv,ssm}.N`
variables are fixed-shape). A `kv_overhead_ratio` well above 1 for these two candidates is
**expected by construction, not evidence of one specific upstream bug**: `expected_kv_gpu_gb` only
counts the persistent, growing-KV-cache layers, while `kv_gpu_gb` is (per §3.1's definition above)
*all* post-load memory growth — which for a hybrid model also includes prefill/decode working
memory (Q/K/V projections, MLP activations) for the other 3-in-4 layers too, since every layer still
runs on every prompt token during prefill even though only `full_attention` layers keep a cache
afterwards. (An earlier version of this note instead blamed a specific known OpenVINO Model Server
issue — continuous-batching prefix caching over-allocating memory for linear-attention models. That
issue is real, but doesn't apply here: this tool's `_load_pipeline()` never sets `scheduler_config`
or `ATTENTION_BACKEND=PA`, and OpenVINO GenAI only enables continuous batching/prefix caching when
one of those is set — otherwise it uses the plain stateful single-sequence backend, which is what
this tool exercises, so that issue's precondition isn't met and it isn't a valid explanation for the
ratio observed here.) This is diagnostic only; it never changes PASS/FAIL (§3.1's policy is
unchanged).

### 3.2 One subprocess per trial

Each (model, context length) trial loads the model fresh in its own `multiprocessing` child
process, runs its probe, reports a result dict over a queue, and exits. This mirrors the existing
"run conversion in a subprocess so memory is fully reclaimed on exit" pattern already used for
model conversion in
[`components/vlm/vlm_openvino_serving/utils/utils.py::_convert_model_worker`](../../components/vlm/vlm_openvino_serving/utils/utils.py).
It matters here for two reasons:

- **Clean memory state.** iGPU memory on this hardware is shared with system RAM; a leak or
  fragmentation carried over from one trial could make the *next* trial fail for reasons
  unrelated to that model/size.
- **Crash containment.** A hard OOM/driver crash at, say, 224K tokens kills only that trial's
  process — the orchestrator (`validate_long_context.py`) detects the dead process, records the
  failure, and moves on instead of taking down the whole sweep.

The orchestrator polls the result queue every 0.25 seconds up to `trial_timeout_sec` (default
1200s). If the process dies without reporting a result it's classified `crashed`, and if it's
still alive at the deadline it's terminated and classified `timeout`. This hard timeout protects
the sweep from a hung process. It is intentionally separate from `max_generate_time_sec`: a
trial that completes after the operational SLA is recorded as `too_slow`, with its real timing
and memory high-water marks intact.

### 3.2.2 The child reports its result, then exits without tearing anything down

The trial child posts `done` and immediately calls `os._exit(0)`. It never destroys the OpenVINO
pipeline, and that omission is deliberate. On the 64 GB shared-memory iGPU box, destroying a
pipeline that had just prefilled 160K tokens threw an `ov::Exception` from *inside* a destructor:

```
openvino_genai.dll!ov::genai::VLMPipeline::~VLMPipeline
  -> openvino.dll!ov::IAsyncInferRequest::~IAsyncInferRequest
  -> openvino_intel_gpu_plugin.dll!...
  -> openvino.dll!ov::Exception::create      <- throws out of a destructor
  -> ucrtbase.dll!terminate                  <- exit code 3221226505 / 0xC0000409
```

An escaping exception in a destructor is `std::terminate`, not a Python exception, so the old
`try: del pipe ... except Exception: pass` never got a chance to run. Worse, that teardown ran
*before* the result was posted, so the abort also erased a measurement the trial had already
finished — the 160K step, the one number the tool exists to produce, came back as
`FAIL (crashed)` with nothing else in the row. Reporting first and skipping the teardown fixes
both halves, and skipping it is safe because §3.2's process exit was always the real reclamation
boundary. (The stack you may see printed with it comes from PyTorch's `std::terminate` handler,
which `transformers` installs on Windows; torch is not involved in the failure itself.)

Two consequences worth knowing:

- The orchestrator **drains the queue** after the child exits. Because `put` is immediately
  followed by `os._exit`, a finished result can land in the pipe in the window between a poll
  timing out and `is_alive()` going False; without the drain a passing trial would be reported as
  `crashed:exitcode=0`.
- The orchestrator **waits for memory to settle** (up to 30s) before returning, since reclamation
  is now the OS's job and the Windows PDH GPU counters lag it. Otherwise the previous trial's
  memory would be charged to the next trial's `weight_gpu_gb`.

Reaching `crashed` now means the abort happened *before* the child could report — during load,
prefill or decode — which is a genuine hardware ceiling. Known native-abort exit codes are decoded
in the error column (`crashed:exitcode=3221226505:0xC0000409 STATUS_STACK_BUFFER_OVERRUN - ...`).

### 3.2.1 No memory guard: every configured step is actually attempted

A trial ends only on success, OOM, a native abort, or `trial_timeout_sec` — never because the
orchestrator predicted it would be too big. This is a deliberate reversal of an earlier design,
and the reason is worth recording, because the earlier design looked prudent and was not.

That version did two things: it aborted a running trial when available RAM or Windows commit
dropped below a `min_memory_headroom_gb` reserve, and — more damagingly — it *pre-emptively*
rejected a larger step by projecting its peak from the previous passing step, scaling the
measured post-load growth by the token ratio **and** a 1.25 allocator safety factor. On the
64 GB iGPU box this is aimed at, that projection cancelled the 160K trial:

```
[Qwen/Qwen3.5-9B] 160,000 tok -> FAIL (memory_guard) | error=memory_guard:projected_peak=65.83GB>=safe_limit=59.56GB(from=128000,to=160000,factor=1.25)
```

160K is the target the whole tool exists to answer, and it was reported as a failure by a trial
that never ran. The projection was also simply wrong. Measured growth across the three passing
steps was **linear to three digits** — 0.2500, 0.2500, 0.2492 GB of peak RAM per 1K tokens — so
128K's 31.9 GB of post-load growth extrapolates to 39.9 GB at 160K, for a peak near **55.8 GB
with ~7.8 GB still free**. Applying the token ratio and then a further 1.25× on top inflated
that to 65.83 GB and manufactured a ceiling below the target. A guard that turns a passing
configuration into a `FAIL` is worse than no guard, because it is indistinguishable in the
report from a real hardware limit.

What replaces it is measurement, not a looser threshold. The sampler tracks the **low-water
mark** of available physical RAM and Windows commit capacity and reports both per trial
(`min_available_ram_gb` / `min_commit_available_gb`, and a `Min free RAM` column in
`summary.md`). That answers the question the guard was reaching for — *how much room was left?*
— without ever cancelling the trial that produces the answer. A step that passes with 7 GB free
has real headroom above it; one that passes with 0.3 GB free is at the wall.

Nothing about this makes the sweep unsafe to run, because §3.2's subprocess isolation was always
the actual safety mechanism. The guard's stated justification — that OpenVINO GPU allocation
failure near shared-memory exhaustion can terminate the native process (`0xC0000409`) without
raising a catchable Python exception — is true, but that abort kills only the trial child. The
parent survives it, has already recorded the peak and headroom that explain it, writes the row,
and moves on. Hitting the ceiling is the tool working, not the tool failing.

One caveat that the original version of this section got wrong, and §3.2.2 fixes: a `crashed` row
is only a *measurement* of the ceiling if the abort happened while the box was actually doing the
work. The 0xC0000409 abort observed at 160K was thrown by the pipeline **destructor**, after the
trial had finished and computed its result, and it destroyed that result on the way out. That row
was not a measurement of anything — it was the tool losing the answer. Since the child now reports
before it cleans up (and no longer cleans up at all), a `crashed` row again means what this
paragraph claims it means.

### 3.2.3 Naming the ceiling: GPU aborts and the headroom that explains them

The step that establishes the ceiling is the single most valuable row the sweep produces, and it
used to be the least legible one. On the 64 GB box, 128K passed cleanly and then:

```
[Qwen/Qwen3.5-9B]   160,000 tok -> FAIL (generate_error)  | ... peak GPU 43.6 GB (68.5% of system RAM) | min free RAM 9.3 GB
  [Qwen/Qwen3.5-9B] 144,000 tok -> FAIL (generate_error)  | ... peak GPU 40.1 GB (63.2% of system RAM) | min free RAM 1.9 GB
    error=generate:exception:Exception from src\plugins\intel_gpu\src\runtime\ocl\ocl_memory.cpp:591:
    [GPU] clWaitForEvents, error code: -14 CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST
```

`generate_error` is the tool's label for *an error it does not understand*, so a real, measured
hardware ceiling was indistinguishable in the report from a bug in the tool. Two things caused it.

**The classifier did not speak OpenCL.** At the ceiling the exception is the GPU plugin's, not
Python's, and it reaches the tool as the plain text of an `ov::Exception`. OpenCL also reports a
command that died *on the device* at the next synchronization point, so the message names the
wait (`clWaitForEvents`) and never names memory. Both marker lists now cover that vocabulary,
split by how much each message actually proves:

- **`oom`** — the allocation demonstrably did not happen: `CL_MEM_OBJECT_ALLOCATION_FAILURE`,
  `CL_OUT_OF_HOST_MEMORY`, `CL_INVALID_BUFFER_SIZE`, alongside the existing `bad_alloc` /
  `out of GPU resources` / etc.
- **`gpu_abort`** — the device accepted a command and then killed it:
  `CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST` (-14), `CL_OUT_OF_RESOURCES` (-5),
  `CL_INVALID_COMMAND_QUEUE` (-36). On a shared-memory iGPU at its ceiling this is usually
  memory, but a driver reset (TDR) produces the same code, and the trial child cannot tell them
  apart — it sees neither the host's free-RAM low-water mark nor the GPU counters, both of which
  the parent samples. So the child reports the narrower fact and the orchestrator decides.

**The orchestrator had no usable memory-pressure signal to decide with.** `gpu_memory_at_limit`
divides peak GPU usage by *total system RAM* and fires at 90%. On a shared-memory iGPU the host
needs the rest of the machine, so that ratio has no route to 90%: every failing trial observed
sat between 63% and 69%. The flag never fired, `too_slow` was unreachable, and a device abort had
no evidence attached to it. The headroom the sampler already records answers the same question
directly, so a new `host_memory_at_limit` column (and `_host_memory_exhausted()`) reads it:
`min_available_ram_gb ≤ 3 GB` **or** `peak_ram_pct ≥ 95%`. Both forms are needed — the absolute
figure catches a large box where a comfortable-looking percentage still hides a wall, the
percentage catches a small box where 3 GB free is plenty of room.

With both in place, `gpu_abort` (and a `crashed` native abort) is promoted to **`oom`** when, and
only when, the trial's own measurements show the memory was gone. The two rows above now read:

| Context | Peak GPU | Min free RAM | Peak RAM | Reported as |
|---:|---:|---:|---:|---|
| 160,000 | 43.6 GB (68.5%) | 9.3 GB | 85.3% | `gpu_abort` — the GPU gave up while the host still had room |
| 144,000 | 40.1 GB (63.2%) | 1.9 GB | 97.1% | `oom` — the box had nothing left |

Note that this is still a *post-hoc* read of what a trial measured. It never cancels a trial and
never predicts one; §3.2.1's rule is unchanged. It only names the failure a trial produced.

### 3.2.4 The report on disk always describes the run that is happening

The sweep's job is to push a box until it breaks, which means it can break hard enough to take
the orchestrator with it. When that happened during the refinement bisection above — after the
144K trial left the machine with 1.9 GB free — nothing was written at all, and `summary.md` was
left holding the **previous `--dry-run`**, which reported the candidate as a *PASS at 160,000
tokens*: the exact opposite of what the twelve minutes of real trials just before it had
measured, with a plausible-looking timestamp on top. A stale report that looks current is worse
than no report.

Three changes, none of which make the sweep stop earlier:

- **The summary is rewritten continuously** — once before any model runs (every candidate as an
  explicit `not run` row), again after each model, and again from a `finally` block that also
  covers Ctrl-C and an unhandled failure. `summary.json` carries `completed: true|false` and
  `summary.md` gets a *"Run in progress or ended early"* banner until the sweep finishes.
- **A step the orchestrator itself cannot run becomes that step's row**, classified
  `trial_error`, instead of unwinding the sweep. Spawning a trial is work the box has to find
  memory for too, and refinement runs immediately after the step that emptied it. A
  `trial_error` during *refinement* never displaces an already-measured capacity reason in the
  summary — `capped by oom at 144,000 tokens` is the sweep's answer, and `capped by trial_error
  at 136,000 tokens` would bury it. `max_stable_context` is the same either way, and the
  un-runnable step is still on the console and in `trials.csv`.
- **Each trial is announced before it runs**, not only after. A step at 128K+ takes minutes and
  the one that finds the ceiling is the slowest of all, so printing on completion alone made a
  running sweep indistinguishable from a hung one for up to `trial_timeout_sec` (20 minutes by
  default). All output is flushed, since stdout is block-buffered when piped to a log.

The orchestrator also now warns when `_wait_for_memory_settle()` gives up: that is not fatal, but
it does mean the next trial's baseline is polluted and its weights/KV split will be wrong, which
should not pass silently as a measurement.

### 3.3 Stepping strategy

For each candidate model, `context_steps_tokens` (default `[64000, 96000, 128000, 160000, 176000,
192000, 224000, 256000]`) is walked in ascending order.
The **first step that fails** (OOM, crash, timeout, or no output) stops the sweep for that model —
the max stable context is the last step that passed. This assumes capacity degrades monotonically
with size, which holds in practice for memory exhaustion.

The tool then bisects up to 3 extra points between the last pass and the first failure,
tightening the reported ceiling instead of only reporting one of the configured step values.
This is **on by default**, because the point of the sweep is the actual ceiling rather than the
nearest configured step below it; pass `--no-refine` to skip it when you only care whether a
specific step passes.

### 3.4 Model preparation is explicit, not automatic

The tool requires each candidate model to already be converted to OpenVINO IR on disk. If the
IR is missing it fails fast for that model and prints the exact `optimum-cli` command to
prepare it, rather than auto-downloading/converting mid-sweep. Given `Qwen/Qwen3.6-35B-A3B`-class
models can mean tens of GB and a long export time, silently kicking that off in the middle of a
context sweep would make run time unpredictable. Prepare all candidates up front instead.

### 3.5 Plain LLM vs. multimodal (VLM) export auto-detection

`optimum-cli export openvino` picks the export layout from the model's own architecture, not
from anything this tool tells it: a plain causal LM exports as a single
`openvino_model.xml`/`.bin` pair, while a multimodal/VLM model (as all three default candidates
are) exports as several components — `openvino_language_model.xml`, `openvino_text_embeddings_model.xml`,
`openvino_vision_embeddings_model.xml`, etc. — the same layout
[`components/vlm/text_gen_vlm.py`](../../components/vlm/text_gen_vlm.py) loads for the production
warm VLM. `trial_runner.py` checks which layout is on disk and loads it with the matching
`ov_genai.LLMPipeline` or `ov_genai.VLMPipeline` accordingly (both expose the same
`.generate(prompt, generation_config=...)` call used for probing here); `_ir_ready()` in
`validate_long_context.py` recognizes either layout too (mirroring
[`content_search/providers/utils/model_utils.py::is_model_ready`](../../content_search/providers/utils/model_utils.py)),
so a converted multimodal model is never misreported as a missing IR.

### 3.6 Tokenizer-loading quirks in raw `optimum-cli` exports

The tokenizer is used only to size the prompt and count tokens; the model's own
`openvino_tokenizer.xml` handles real inference. A tokenizer converted by `optimum-cli export
openvino` directly (as this tool's Prerequisites recommend) can differ from one converted by the
project's own `convert_model()` helper in two ways that trip up a plain
`AutoTokenizer.from_pretrained(model_dir)` call:

- `tokenizer_config.json`'s `extra_special_tokens` is written as a list where transformers
  expects a dict (`AttributeError: 'list' object has no attribute 'keys'`) -- the same issue
  [`components/vlm/text_gen_vlm.py::VLMTextGen._load`](../../components/vlm/text_gen_vlm.py) already
  works around for production.
- `tokenizer_config.json`'s declared `tokenizer_class` can name something
  `AutoTokenizer` doesn't recognize (e.g. `TokenizersBackend`, seen on a real int8 VLM export),
  raising `ValueError: Tokenizer class ... does not exist or is not currently imported.` even
  though `tokenizer.json` is a perfectly valid fast-tokenizer file.

`trial_runner.py::_load_tokenizer` tries `AutoTokenizer` and, on either failure, falls back to
loading `PreTrainedTokenizerFast` directly (which doesn't need to resolve a class name) — trying
both with and without the `extra_special_tokens` override, so whichever quirk (if any) is present
in a given export is handled without needing to know in advance which one it is. This fallback
makes transformers log a "tokenizer class you load ... is not the same type as the class this
function is called from" warning; it is harmless here (the tokenizer is only used for prompt
sizing, not inference), so `_load_tokenizer` sets transformers' log level to error to silence it —
otherwise it would repeat three lines for **every** trial's fresh subprocess and bury the actual
results.

### 3.7 Keeping the console log clean

Two sources of benign native noise are suppressed so the sweep log shows just the per-trial
results:

- **The tokenizer fallback warning** (above) — silenced via transformers' log level in
  `_load_tokenizer`, once per subprocess.
- `Win32 exception occurred releasing IUnknown at 0x...` — COM-teardown noise emitted by the
  Windows WMI/`pythoncom` layer in [`utils/platform_info.py`](../../utils/platform_info.py) while
  it collects the hardware fingerprint. It comes from the native COM layer, not Python's
  logging/warnings, so `validate_long_context.py` wraps only that best-effort call in an fd-level
  stderr redirect (`_suppress_native_stderr`) to hide it. It never affected the sweep; this just
  removes the distraction.

## 4. Configuration reference

The tool has its own config file,
[`components/llm/context_validation/config.yaml`](../../components/llm/context_validation/config.yaml),
loaded by default (resolved relative to the script's own location, not the current directory).
It is a standalone copy — `provider` / `device` / `weight_format` / `models_base_path` start out
matching `smart-classroom/config.yaml`'s `models.summarizer` section, but the two are not linked:
changing one does not change the other. If you update the production summarizer's device or
weight format, update this file's copies too if you want the sweep to stay representative of
what's actually deployed. Any key can also be overridden per-run via CLI flags without editing
either file.

```yaml
summarizer:
  provider: openvino
  device: GPU
  weight_format: int8
  models_base_path: "models"
  long_context_validation:
    candidate_models:
      - Qwen/Qwen3-8B
      - Qwen/Qwen3.6-35B-A3B
      - Qwen/Qwen3.5-9B
    target_context_tokens: 160000
    context_steps_tokens: [8000, 16000, 32000, 48000, 64000, 96000, 128000, 144000, 160000, 176000, 192000, 224000, 256000]
    probe_tokens: 64
    max_generate_time_sec: 600
    gpu_memory_pressure_pct: 90
    trial_timeout_sec: 1200
    output_dir: monitoring/executionlogs/long_context_validation
```

| Key | Meaning |
|---|---|
| `provider` / `device` / `weight_format` | How to load each candidate model — mirrors `models.summarizer` in the main config, but is an independent copy. |
| `models_base_path` | Where converted model IRs live, resolved relative to the `smart-classroom/` working directory (same convention as production). |
| `candidate_models` | Model names to sweep (HuggingFace repo id form). |
| `target_context_tokens` | The customer requirement to check the ceiling against (160K). |
| `context_steps_tokens` | Ascending token sizes to probe. |
| `probe_tokens` | Tokens to decode per trial. Small on purpose — a capacity check only needs a few decode steps (default 64). |
| `max_generate_time_sec` | Soft time limit for the full prefill + probe `generate()` call (default 600s). It produces `too_slow` only together with memory pressure. |
| `gpu_memory_pressure_pct` | Practical shared-iGPU memory pressure line, measured as peak GPU usage divided by total system RAM (default 90%). It is combined with the soft time limit rather than treated as an independent failure. On a shared-memory iGPU this ratio is hard to reach — the measured host headroom is the second, and in practice the effective, pressure signal (§3.2.3). |
| `trial_timeout_sec` | Hard per-trial wall-clock budget before the subprocess is killed. Keep this above `max_generate_time_sec` so slow trials can return diagnostics. There is deliberately no memory-headroom setting alongside it — see §3.2.1. |
| `output_dir` | Where `trials.csv` / `summary.json` / `summary.md` are written (relative to `smart-classroom/`). |

## 5. Prerequisites & setup

Everything below is handled automatically by
[`run_validate_long_context.ps1`](../../components/llm/context_validation/run_validate_long_context.ps1)
(§6) — read this section if you want to understand what it's doing, run the tool without the
launcher, or troubleshoot.

1. **A Python environment with the project's `requirements.txt` installed** (OpenVINO GenAI,
   optimum-intel, transformers, torch, psutil) — **not** whatever `python` resolves to on `PATH`,
   which is the single most common way to hit "it doesn't run". This tool reuses the exact same
   backend venv `setup-smart-classroom.ps1` / `start-smart-classroom.ps1` use: created at
   `../smartclassroom` (sibling of `smart-classroom/`, no hyphen), activated with
   `Scripts\Activate.ps1` before launching `python`.
   - **Already ran `setup-smart-classroom.ps1`?** That venv already exists; nothing more to do.
   - **Haven't, or just want this tool working on its own?** Run this tool's own
     [`setup_env.ps1`](../../components/llm/context_validation/setup_env.ps1) once — it
     creates/reuses that exact same venv and `pip install`s `requirements.txt` into it, without
     running the full interactive `setup-smart-classroom.ps1` (which also sets up the frontend
     and content_search, unrelated to this tool) and without touching
     `smart-classroom/config.yaml`. Safe to re-run any time — it detects an existing venv and
     just verifies/updates packages.
     ```powershell
     .\components\llm\context_validation\setup_env.ps1
     ```
   - Either way, the tool also checks for this itself before starting a real sweep (not under
     `--dry-run`) and prints the exact venv path / setup command to fix it with if
     `openvino_genai` / `transformers` aren't importable in whatever interpreter you used,
     rather than letting every trial in the sweep fail on the same missing import.
2. **Each candidate model converted to OpenVINO IR** under
   `<models_base_path>/<provider>/<model_name_with_slashes_replaced>_<weight_format>/` — the
   same layout `utils/ensure_model.py::get_model_path()` uses for the production summarizer
   model. For example:
   ```bash
   optimum-cli export openvino --model "Qwen/Qwen3-8B" --trust-remote-code --weight-format int4 "models/openvino/Qwen_Qwen3-8B_int4"
   ```
   Repeat per candidate model / weight format you want to test. The tool prints this exact
   command (with the right path filled in) whenever it detects a missing IR, so you don't have
   to compute the path by hand.

If PowerShell blocks either `.ps1` script with an UnauthorizedAccess/SecurityError:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

## 6. Usage

**Simplest path** — run from anywhere; it finds `smart-classroom/` and the backend venv
relative to its own location, preparing the venv on first use if needed:

```powershell
.\components\llm\context_validation\run_validate_long_context.ps1
```

Any extra arguments are forwarded to `validate_long_context.py` as-is:

```powershell
# Exercise the sweep/report pipeline itself with no OpenVINO/GPU required
.\components\llm\context_validation\run_validate_long_context.ps1 --dry-run

# Skip the default bisection near the pass/fail boundary (faster, coarser ceiling)
.\components\llm\context_validation\run_validate_long_context.ps1 --no-refine

# Test one model only, with a custom step list
.\components\llm\context_validation\run_validate_long_context.ps1 --models Qwen/Qwen3-8B --steps 32000 64000 128000 160000 192000

# Test the same model at a different device/weight_format without editing any config file
.\components\llm\context_validation\run_validate_long_context.ps1 --models Qwen/Qwen3-8B --weight-format int8 --device GPU

# Adjust either half of the combined time + GPU-memory pressure policy
.\components\llm\context_validation\run_validate_long_context.ps1 --max-generate-time-sec 900 --gpu-memory-pressure-pct 92

# Point at a different config file entirely (e.g. a scratch copy for one-off experiments)
.\components\llm\context_validation\run_validate_long_context.ps1 --config C:\path\to\other.yaml
```

**If you already have the right interpreter active** (see §5), the launcher is just a
convenience wrapper around, run from `smart-classroom/`:

```bash
python -m components.llm.context_validation.validate_long_context [same arguments as above]
```

Console output streams one line per trial as it happens, with the memory split into weights vs.
KV-cache (§3.1.1):

```
=== Qwen/Qwen3-8B ===  weights on disk: 8.5 GB (int8)
[Qwen/Qwen3-8B]     8,000 tok -> PASS  |  load 12.3s, gen 3.1s (64 tok, 20.65 tok/s)  |  peak RAM 12.4 GB (weights +8.6, kv +1.8)  |  peak GPU 10.1 GB (weights +8.5, kv +1.6)  |  min free RAM 50.8 GB (commit 58.2 GB)
...
[Qwen/Qwen3-8B]   160,000 tok -> PASS  |  load 12.1s, gen 9.4s (64 tok, 6.81 tok/s)  |  peak RAM 58.1 GB (weights +8.6, kv +47.5)  |  peak GPU 21.4 GB (weights +8.5, kv +12.9)  |  min free RAM 5.2 GB (commit 9.7 GB)
[Qwen/Qwen3-8B]   176,000 tok -> FAIL (timeout)  |  load 12.4s  |  peak RAM 63.9 GB (weights +8.6, kv +53.3)  |  peak GPU 22.1 GB  |  min free RAM 0.3 GB (commit 1.1 GB)  |  error=timeout
```

Because memory is sampled by the orchestrator, the failing row still reports the peak the box
reached (here ~64 GB RAM — the machine was thrashing, which is why it timed out rather than
raising a clean OOM), instead of dropping to zeros when the trial subprocess is killed. Reading
the `min free RAM` column down the sweep is the quickest way to see how much room is left above
the last passing step: 5.2 GB at 160K here means the ceiling is close but real, and the 0.3 GB
on the failing row confirms the box genuinely ran out rather than being stopped by policy.

## 7. Interpreting the report

Three files land in `output_dir`:

- **`trials.csv`** — one row per trial, written immediately after each trial completes (so a
  crash mid-sweep doesn't lose earlier results): model, tokens requested, device, weight_format,
  load/generate success, prompt/generated tokens, load & generate time, effective
  `tokens_per_second`, the configured `max_generate_time_sec`, and the memory breakdown
  (`weight_disk_gb`, `weight_ram_gb`/`weight_gpu_gb`, `kv_ram_gb`/`kv_gpu_gb`,
  `expected_kv_gpu_gb`/`kv_overhead_ratio` (§3.1.1, `None` when the model's `config.json` doesn't
  expose enough architecture info), `peak_ram_gb`/`peak_ram_pct`/`peak_gpu_gb`, and the headroom
  low-water marks `min_available_ram_gb`/`min_commit_available_gb` (§3.2.1)), the two
  memory-pressure decision flags `gpu_memory_at_limit`/`host_memory_at_limit` (§3.2.3), a
  `status` (`PASS` or the failure reason), and the raw `error` string.
- **`summary.json`** — machine-readable rollup per model: `max_stable_context`,
  `meets_target` (bool, compared against `target_context_tokens`), the memory breakdown at that
  max stable size (`weight_disk_gb`, `weight_ram_gb`/`weight_gpu_gb`, `kv_ram_gb`/`kv_gpu_gb`,
  `expected_kv_gpu_gb`/`kv_overhead_ratio`, `peak_ram_gb`/`peak_gpu_gb`,
  `min_available_ram_gb`/`min_commit_available_gb`), `failure_reason` and the
  `failure_tokens` it was measured at, a top-level `completed` flag (§3.2.4), and the
  hardware fingerprint the sweep ran on (from `utils/platform_info.py`).
- **`summary.md`** — the same rollup as a table (memory measured at the max stable context;
  weights = footprint just after load, KV = extra memory prefill+decode added on top, Min free
  RAM = headroom low-water mark (§3.2.1), Expected KV/KV Ratio = architecture-derived reference
  and measured/expected ratio, §3.1.1), e.g.:

  | Model | Device | Weight | Max stable context | Meets target | Weights (disk) | Peak RAM | KV RAM | Min free RAM | Peak GPU | KV GPU | Expected KV | KV Ratio | Notes |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | Qwen/Qwen3-8B | GPU | int4 | 160,000 | PASS | 4.6 GB | 58.1 GB | 47.5 GB | 5.2 GB | 21.4 GB | 12.9 GB | 11.0 GB | 1.2x | reached top configured step without failing |
  | Qwen/Qwen3.6-35B-A3B | GPU | int4 | 64,000 | FAIL | 18.2 GB | 63.6 GB | 20.1 GB | 0.4 GB | 58.4 GB | 21.4 GB | 1.7 GB | 12.6x | kv 12.6x theoretical -- known OpenVINO linear-attention cache issue (see release notes), not a capacity problem with this box |

  A `KV Ratio` at or above `_KV_OVERHEAD_RATIO_NOTE_THRESHOLD` (default 3x) replaces the generic
  `failure_reason` note with a call-out that the gap looks like a known cache-efficiency issue
  rather than a plain capacity limit — see §3.1.1.

`failure_reason` values: `oom`, `gpu_abort`, `timeout`, `crashed`, `load_error`,
`generate_error`, `no_output`, `too_slow`, `trial_error`. `gpu_abort` means the GPU killed a
command it had accepted and the box still had headroom, so it was not called `oom` (§3.2.3);
`trial_error` means the orchestrator could not carry the step out at all (§3.2.4).
A model whose max stable context still meets the target can show a failure reason
too — it just means the sweep found the *next* configured step above the target failed for that
reason, which is still useful context for headroom planning.

### 7.1 Conclusion for the supplied int8 / iGPU / 64 GB run

The earlier 120-second-only policy established **48,000 tokens as the maximum supported context
among the completed steps** for `Qwen/Qwen3.6-35B-A3B` int8 on one measured platform:

| Context | Generate time | Output | Result under 120s SLA |
|---:|---:|---:|---|
| 32,000 | 31.7s | 64 tokens | PASS |
| 48,000 | 50.7s | 64 tokens | PASS |
| 64,000 | 254.4s | 10 tokens | FAIL (`too_slow`) |
| 68,000 | 288.7s | 10 tokens | FAIL (`too_slow`) |
| 72,000 | 328.3s | 10 tokens | FAIL (`too_slow`) |
| 80,000 | 405.2s | 10 tokens | FAIL (`too_slow`) |

Under the current combined policy these historical timings must be re-run: a trial is capped by
`too_slow` only when its measured GPU peak also crosses the configured pressure line. The sharp
timing jump remains useful evidence of paging/thrashing, but old rows do not contain the new
`peak_gpu_pct` / `gpu_memory_at_limit` decision fields. Conclusions remain specific to the model,
weight format, device, driver, system memory, time budget, and pressure threshold used in a run.

## 8. Hardware caveats

- **Windows iGPU shares system memory.** Unlike a discrete GPU with dedicated VRAM, the Intel
  iGPU's usable memory is bounded by how much the OS/driver lets it allocate. If a model that
  should plausibly fit still hits an OOM-classified failure, first try increasing the dedicated
  GPU memory allocation in **Intel® Graphics Software → Graphics tab**, per the existing
  troubleshooting note for `CL_OUT_OF_RESOURCES` in
  [`advance-setup-guide.md`](../user-guide/advance-setup-guide.md#troubleshooting), before
  concluding the model can't reach the target.
- **`weight_format` trades memory for capacity, but changing it changes what you measured.**
  `int4` leaves more headroom for a large KV-cache (longer max context) than `int8` or `fp16` at
  the same context length. That makes it a tempting way to make a target "pass" — but the result
  then describes a *different model* than the one being deployed, and on this hardware the weights
  are the small term anyway: at 128K the `Qwen/Qwen3.5-9B` int8 weights account for 9.8 GB of a
  47.9 GB peak, while KV growth accounts for 31.9 GB. Dropping to int4 would buy roughly 5 GB, or
  about 20K tokens of context. Only re-run at a lower precision if you would actually ship that
  precision; otherwise keep `weight_format` pinned to the deployed value and report the ceiling
  it really has.
- **`Qwen/Qwen3.5-9B` and `Qwen/Qwen3.6-35B-A3B` are hybrid linear-attention models, and a large
  `kv_overhead_ratio` for them is currently expected, not a sign this box is under-provisioned.**
  Their exported `config.json` declares only 1-in-4 layers as `full_attention` (the rest are
  Mamba/GatedDeltaNet-style `linear_attention` layers, which should hold an O(1) state rather than
  one that grows with context length). The ratio being large is a scope mismatch, not a single
  identifiable upstream bug: `kv_gpu_gb` measures *all* post-load memory growth, which for a hybrid
  model includes prefill/decode working memory across all layers (not just the ones that keep a
  growing cache). See §3.1.1 for the full theoretical-vs-measured comparison, including why an
  earlier version of this doc's more specific "known OpenVINO prefix-caching bug" explanation
  doesn't actually apply to how this tool invokes the pipeline.
- **Large candidates cost real disk/RAM even to attempt.** `Qwen/Qwen3.6-35B-A3B`-class models
  need substantial disk space for the IR and host RAM just to load, independent of how far the
  context sweep gets.
- **Each step reloads the model from scratch, on purpose (§3.2).** For a 30B+-class model,
  loading is roughly a minute per step on real hardware. Prefill at 128K+ tokens is itself slow,
  so a full 13-step sweep is a real commitment of time for large candidates, not a five-minute
  check; pass a shorter custom `--steps` list while iterating.
- **Capacity is not answer quality.** A PASS means the hardware can hold and decode that context
  size within the configured latency SLA — not that the model produces a good summary of it. Validate the winning model/size
  combination against a couple of real long transcripts before shipping.

## 9. Limitations

- This is a hardware-capacity probe, not a comprehension test: it confirms the box can prefill and
  decode a context of size N, not that the model still *uses* facts stated far back in it. If you
  need to check long-range recall quality, do it separately against real transcripts.
- The stepping strategy assumes monotonic degradation (a fail at N is assumed to persist for all
  sizes above N). If a model's behavior is non-monotonic, re-run with a denser custom `--steps`
  list around the suspect region.
- Boundary refinement adds at most 3 extra trials per model — it narrows the reported boundary,
  it doesn't binary-search to token-level precision.
- `min_available_ram_gb` / `min_commit_available_gb` are sampled every 0.5s from the parent, so a
  very short allocation spike between samples can be missed. They bound the headroom that was
  actually observed, which is the right basis for a capacity judgement, but they are not a
  guarantee that no instant was tighter.
- Peak GPU memory sampling is best-effort and Windows-only (via the repo's perf-counter
  collector); on other platforms, or if the counter is unavailable, the GPU column reads `0.0`
  and only RAM is reported.
```
