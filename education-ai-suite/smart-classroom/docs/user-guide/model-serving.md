# VLM/LLM Model Serving

Every text-generation feature (summary, mind map, topic segmentation, report,
board-OCR summary, Content Search Q&A and video summarization) uses one shared
OpenVINO™ model, selected under `models.text_gen` in `smart-classroom/config.yaml`.
The same model is exposed as an OpenAI-compatible API on
`http://127.0.0.1:8000/v1/chat/completions`, which also accepts tool calls for
agent integrations.

## Choosing a model

```yaml
models:
  text_gen:
    vlm_name: Qwen/Qwen3.6-35B-A3B   # or Qwen/Qwen3-VL-8B-Instruct, Qwen/Qwen3.8-27B
    weight_format: int4              # int4 | int8
    device: GPU
```

The IR is read from `models/openvino/<model>/<weight_format>/`. If it is missing,
the pre-converted OpenVINO IR is downloaded on first start, or the model is
exported locally when no pre-converted IR is published. `vlm_name` can also be an
absolute path to an IR directory, such as a custom or fine-tuned export.

| Model | Type | INT4 IR on disk | Pre-converted IR | OpenVINO | Verified on Panther Lake (Arc B390) |
| --- | --- | --- | --- | --- | --- |
| Qwen3-VL-8B-Instruct (default) | dense VLM | ~5.4 GB | INT4 and INT8 | 2026.3+ | INT4: chat, images, tools, streaming |
| Qwen3.6-35B-A3B | MoE VLM, 3B active | ~19.6 GB | INT4 only (INT8 is exported locally) | 2026.2+ | INT4: ~45 tok/s, images, tools, thinking, DFlash |
| Qwen3.8-27B | dense VLM | ~15.9 GB | INT4 and INT8 | **2026.4+** | INT4: ~7.6 tok/s, images, tools, thinking |

INT8 variants and Arrow Lake have not been measured yet. INT8 needs roughly twice
the INT4 weight memory, which is about 39 GB for Qwen3.6-35B-A3B and 32 GB for
Qwen3.8-27B. Plan for 64 GB of system RAM for the larger models; the iGPU shares it.

If the installed runtime is older than a model's model card requires, the model
is refused at load with a message that names the required version. Without that
check, Qwen3.8 on OpenVINO 2026.3 crashes the process. Re-run the setup script to
install the versions pinned in `requirements.txt`.

Thinking-capable models (Qwen3.6 and Qwen3.8) think by default.
`models.text_gen.enable_thinking: false` keeps the Smart Classroom workflows
and existing API clients on direct answers. A request can still opt in with
`"enable_thinking": true`, or with
`"chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "low"}`.
The reasoning is then returned separately in `message.reasoning_content`.

## Where the model runs

`models.text_gen.serving.mode` decides which process hosts the model. Port 8000,
the UI, Content Search and all request payloads are the same in every mode.

| Mode | What happens | Use it when |
| --- | --- | --- |
| `inprocess` (default) | The model loads inside the Smart Classroom app, as before. | Single-user desktop installs. |
| `managed` | The app starts `python -m model_serving` on `serving.port` (8010), restarts it if it crashes or stalls, and proxies `/v1/*` on :8000 to it. | You want a GPU or driver fault to leave the rest of the app running. |
| `external` | The app connects to an already running model server and never starts or stops it. | One model server is shared by several apps or workflows, or runs on another host. |

In `managed` and `external` mode the app starts without waiting for the model.
`GET /health` reports `hub.text_gen.state` as `loading` until the model server is
ready, and Content Search waits for that state just as it does today.

### Running only the model server

The model server reads the same `config.yaml` and loads only the model. ASR, OCR,
video analytics, ChromaDB and the UI are not loaded. From `smart-classroom/`:

```powershell
..\smartclassroom\Scripts\python.exe -m model_serving --port 8000
```

Content Search, grading or any OpenAI-compatible client can then use
`http://127.0.0.1:8000` with no further changes. Other options are `--host`
(default `127.0.0.1`) and `--config <file>`. Set `MODEL_SERVING_API_KEY`
(or `serving.api_key`) to require `Authorization: Bearer <key>` on `/v1/*`;
do this before binding to anything other than loopback.

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/chat/completions` | Chat, images, tools, streaming |
| `GET /v1/models` | The served model |
| `GET /live` | The process is responsive (503 once generation has stalled) |
| `GET /ready` | The model is loaded and warmed up |
| `GET /health` | Same shape as the main app's `/health` |

## Chat and tool-call API

The API supports the following subset of the OpenAI Chat Completions API:

- The full ordered history of `system`, `developer`, `user`, `assistant` and `tool`
  messages. A tool result must follow the assistant message that made the call.
- Text parts and `image_url` parts (data URLs or local paths; remote URLs are not
  fetched) in user messages.
- `tools` (functions with a JSON-schema `parameters` object), `tool_choice`
  (`auto`, `none`, `required` or a named function) and `parallel_tool_calls`.
- `response_format` (`json_object` or `json_schema`), `max_completion_tokens` /
  `max_tokens`, `temperature`, `top_p`, `top_k`, `seed`, `stop`, the penalties,
  `stream` and `stream_options.include_usage`.

The server proposes tool calls; it never executes them. Your client or agent
controller runs each call and sends the result back as a `tool` message:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
tools = [{"type": "function", "function": {
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]
messages = [{"role": "user", "content": "Weather in Paris?"}]

reply = client.chat.completions.create(model="local", messages=messages, tools=tools)
call = reply.choices[0].message.tool_calls[0]           # finish_reason == "tool_calls"
messages += [reply.choices[0].message,
             {"role": "tool", "tool_call_id": call.id, "content": '{"forecast": "sunny"}'}]
answer = client.chat.completions.create(model="local", messages=messages, tools=tools)
```

The model's own chat template renders the tools. Both Qwen tool-call formats
are parsed: Hermes JSON from Qwen3-VL, and qwen3coder XML from Qwen3.5 and later.
Each call is checked against the offered tools and their required arguments
before it is returned. A malformed call is returned as plain text, never as a
`tool_calls` entry. A forced `tool_choice` (`required` or a named function)
always produces a call, because the assistant turn is started inside the call.

Errors use the OpenAI error shape. Status codes:

- `400`: invalid input.
- `503` with `Retry-After`: the request queue is full (`queue_max`), memory is
  short, or the model is still loading.
- During a stream, an `error` event without `[DONE]`.

## Reliability

- **Bounded admission.** `concurrency` requests run at a time and up to
  `queue_max` wait. Beyond that the server answers 503 immediately instead of
  queueing without limit.
- **Cancellation.** A client that disconnects stops its generation. The slot is
  freed only once the device is idle, so two requests never share the pipeline.
- **Warm-up and readiness.** A one-token generation runs at load, so compile and
  allocation errors appear at startup. `/ready` stays 503 until then.
- **Stall watchdog.** If no token is produced for `serving.stall_timeout_s`
  (default 300 s), the model server exits with code 4 so it can be restarted;
  a hung native call cannot be interrupted from a thread.
- **Supervisor (`managed` mode).** Restarts use exponential backoff capped at
  60 s, with a budget of `serving.max_restarts` restarts per 10 minutes. A load
  or configuration error (exit code 3) is not retried. If a model server is
  already running on the port, it is adopted rather than duplicated.
- **Orphan protection.** A managed model server exits when its parent app dies,
  so it does not keep holding GPU memory.
- **Retries.** The app retries a request only before any output has been
  produced, and only while the server is starting or overloaded.

## DFlash speculative decoding

[DFlash](https://huggingface.co/blog/ofirzaf/intel-dflash-ptl) drafts a block of
tokens with a small drafter model conditioned on the target model's hidden
states, and the target verifies the whole block in one pass. To use it you need:

1. `openvino-genai` 2026.4 or later. Older runtimes crash on a DFlash drafter,
   so it is refused at load.
2. A target IR exported with hidden-state metadata (`hidden_states_decoder_layers`).
   The `OpenVINO/Qwen3.6-35B-A3B-int4-ov` IR has it from the 2026-09-25 upload
   onward. Older local copies must be downloaded again.
3. The drafter as OpenVINO IR, for example `z-lab/Qwen3.6-35B-A3B-DFlash`
   exported with `optimum-cli export openvino --task text-generation-with-past
   --weight-format int4 --trust-remote-code`, placed in
   `models/openvino/Qwen3.6-35B-A3B-DFlash/int4/`.

```yaml
models:
  text_gen:
    speculative:
      mode: auto                     # "off" | auto | on
      draft_model: z-lab/Qwen3.6-35B-A3B-DFlash
      num_assistant_tokens: 5
```

`auto` uses DFlash when the runtime, target and drafter support it, and
otherwise logs why and decodes normally. `on` refuses to start without DFlash.
`/health` reports `hub.text_gen.speculative` as `active`, `off` or
`unavailable: <reason>`.

Measured end-to-end through the API on an Intel® Core™ Ultra X7 358H
(Arc B390 iGPU, 64 GB, driver 32.0.101.8826, OpenVINO 2026.4.1) with
Qwen3.6-35B-A3B INT4. Each figure is the best of two runs of about 170 tokens:

| Workload | Plain | DFlash, 5 tokens | Speedup |
| --- | --- | --- | --- |
| Code | 45.1 tok/s | 99.3 tok/s | 2.2x |
| JSON lesson report | 45.1 tok/s | 63.7 tok/s | 1.4x |
| Prose summary | 37.8 tok/s | 38.8 tok/s | ~1.0x |

Raising `num_assistant_tokens` to 7 gave the same speed on code and was slower
on prose, so 5 is the default. Outputs differ slightly from plain greedy
decoding, because block verification uses different GPU kernels, but quality is
equivalent.

Current limitations of the OpenVINO GenAI DFlash pipeline:

- It decodes greedily only. While DFlash is active, `temperature` and `top_p`
  are ignored, and this is logged once.
- It drafts text only. Requests with images decode normally on the same pipeline.
- Prefix caching is turned off.
