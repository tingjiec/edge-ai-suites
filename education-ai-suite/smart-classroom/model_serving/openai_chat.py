# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible ``/v1/chat/completions`` on top of a ``text_gen`` engine.

One implementation backs both the in-process route of the Smart Classroom app
and the standalone model server (``python -m model_serving``), so the two never
drift. The engine is anything with ``TextGenHandler.generate``'s keywords.

Supported subset (see docs/user-guide/model-serving.md):

* full ordered history: ``system`` / ``developer`` / ``user`` / ``assistant`` /
  ``tool`` messages, text and ``image_url`` (data URL or local path) parts;
* ``tools`` + ``tool_choice`` (``auto`` / ``none`` / ``required`` / named) and
  ``parallel_tool_calls``; calls come back as ``message.tool_calls``;
* reasoning split into ``reasoning_content`` (``enable_thinking`` or
  ``chat_template_kwargs``), ``response_format`` JSON schema, sampling fields,
  ``stream`` (+ ``stream_options.include_usage``) and ``usage``.

The server proposes tool calls but never runs them.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from model_serving.tool_parser import (
    OutputSplitter,
    ToolCallError,
    forced_call_prefix,
    parse_tool_calls,
)

logger = logging.getLogger(__name__)

_ROLES = {"system", "developer", "user", "assistant", "tool"}
_MAX_IMAGE_BYTES = 32 * 1024 * 1024


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Optional[Union[str, List[Any]]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    reasoning_content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    messages: List[ChatMessage] = Field(..., min_length=1)
    model: Optional[str] = None
    stream: bool = False
    stream_options: Optional[Dict[str, Any]] = None
    max_completion_tokens: Optional[int] = Field(None, ge=1)
    max_tokens: Optional[int] = Field(None, ge=1)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, gt=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=0)
    repetition_penalty: Optional[float] = Field(None, gt=0.0)
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    enable_thinking: Optional[bool] = None
    chat_template_kwargs: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None
    response_format: Optional[Dict[str, Any]] = None


class ChatError(Exception):
    """A request the server will not run; becomes an OpenAI-style error body."""

    def __init__(self, message: str, status: int = 400, kind: str = "invalid_request_error"):
        super().__init__(message)
        self.status = status
        self.kind = kind


def error_response(message: str, status: int = 400,
                   kind: str = "invalid_request_error", headers=None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        headers=headers,
        content={"error": {"message": message, "type": kind, "code": status}},
    )


# ---------------------------------------------------------------------------
# Liveness accounting, read by the standalone server's stall watchdog.
# ---------------------------------------------------------------------------
class _Activity:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.last_progress = time.monotonic()

    def begin(self) -> None:
        with self._lock:
            self.active += 1
            self.last_progress = time.monotonic()

    def touch(self) -> None:
        self.last_progress = time.monotonic()

    def end(self) -> None:
        with self._lock:
            self.active = max(0, self.active - 1)

    def stalled_for(self) -> float:
        """Seconds without a generated token while a request is running."""
        return time.monotonic() - self.last_progress if self.active else 0.0


ACTIVITY = _Activity()


# ---------------------------------------------------------------------------
# Request -> engine call
# ---------------------------------------------------------------------------
class _Job:
    def __init__(self) -> None:
        self.messages: List[dict] = []
        self.image_urls: List[str] = []
        self.images: Optional[list] = None
        self.tools: Optional[list] = None
        self.prefill: Optional[str] = None
        self.parallel_tool_calls = True
        self.enable_thinking: Optional[bool] = None
        self.template_kwargs: Dict[str, Any] = {}
        self.json_schema: Optional[str] = None
        self.max_new_tokens: Optional[int] = None
        self.temperature: Optional[float] = None
        self.sampling: Dict[str, Any] = {}
        self.stats: Dict[str, Any] = {}
        self.cancel = threading.Event()

    def engine_kwargs(self) -> dict:
        kwargs = dict(
            messages=self.messages,
            images=self.images,
            stream=True,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            enable_thinking=self.enable_thinking,
            json_schema=self.json_schema,
            stats=self.stats,
            cancel_event=self.cancel,
        )
        # Only pass what is in use, so engines without these options still work.
        for key in ("tools", "prefill", "sampling", "template_kwargs"):
            value = getattr(self, key)
            if value:
                kwargs[key] = value
        return kwargs


def build_job(req: ChatCompletionRequest, tool_call_format: str,
              default_thinking: Optional[bool]) -> _Job:
    job = _Job()
    job.messages, job.image_urls = _convert_messages(req.messages)

    tools = _validate_tools(req.tools)
    choice = req.tool_choice
    forced: Optional[str] = None
    if choice is not None and not tools and choice != "none":
        raise ChatError("tool_choice requires tools")
    if choice in (None, "auto"):
        pass
    elif choice == "none":
        tools = None
    elif choice == "required":
        forced = ""
    elif isinstance(choice, dict) and choice.get("type") == "function":
        name = (choice.get("function") or {}).get("name")
        offered = [t for t in tools if t["function"]["name"] == name]
        if not offered:
            raise ChatError(f"tool_choice names unknown function {name!r}")
        tools, forced = offered, name
    else:
        raise ChatError(f"unsupported tool_choice {choice!r}")
    job.tools = tools
    job.parallel_tool_calls = req.parallel_tool_calls is not False

    template_kwargs = dict(req.chat_template_kwargs or {})
    thinking = req.enable_thinking
    template_thinking = template_kwargs.pop("enable_thinking", None)
    if thinking is None and template_thinking is not None:
        thinking = bool(template_thinking)
    if thinking is None:
        thinking = default_thinking
    if forced is not None:
        # The forced call opens the assistant turn, so the think block is closed.
        thinking = False
        job.prefill = forced_call_prefix(tool_call_format, forced or None)
    job.enable_thinking = thinking
    job.template_kwargs = template_kwargs

    job.json_schema = _json_schema(req.response_format)
    if job.json_schema and tools:
        raise ChatError("response_format cannot be combined with tools")

    job.max_new_tokens = req.max_completion_tokens or req.max_tokens
    job.temperature = req.temperature
    stop = [req.stop] if isinstance(req.stop, str) else req.stop
    job.sampling = {
        key: value for key, value in {
            "top_p": req.top_p,
            "top_k": req.top_k,
            "repetition_penalty": req.repetition_penalty,
            "presence_penalty": req.presence_penalty,
            "frequency_penalty": req.frequency_penalty,
            "rng_seed": req.seed,
            "stop_strings": stop,
        }.items() if value is not None
    }
    return job


def _convert_messages(messages: List[ChatMessage]):
    """Validate the history and turn it into chat-template input."""
    converted: List[dict] = []
    image_urls: List[str] = []
    open_calls: Optional[set] = None  # unanswered ids of the last assistant turn's calls
    call_ids: set = set()  # every id that assistant turn issued
    for index, message in enumerate(messages):
        role = message.role
        if role not in _ROLES:
            raise ChatError(f"messages[{index}]: unsupported role {role!r}")
        text, urls = _flatten(message.content, index, allow_images=(role == "user"))
        image_urls.extend(urls)
        entry: Dict[str, Any] = {
            "role": "system" if role == "developer" else role,
            "content": text,
        }
        if role == "assistant":
            if message.reasoning_content:
                entry["reasoning_content"] = message.reasoning_content
            if message.tool_calls:
                entry["tool_calls"] = [
                    _template_tool_call(call, index) for call in message.tool_calls
                ]
                call_ids = {c.get("id") for c in message.tool_calls if c.get("id")}
                open_calls = set(call_ids)
            else:
                open_calls = None
        elif role == "tool":
            if open_calls is None:
                raise ChatError(
                    f"messages[{index}]: a tool result must follow an assistant "
                    "message with tool_calls"
                )
            if call_ids and message.tool_call_id is not None:
                if message.tool_call_id not in open_calls:
                    raise ChatError(
                        f"messages[{index}]: unknown or duplicate tool_call_id "
                        f"{message.tool_call_id!r}"
                    )
                open_calls.discard(message.tool_call_id)
            if message.name:
                entry["name"] = message.name
        else:
            open_calls = None
        converted.append(entry)
    if not any(m["role"] == "user" for m in converted):
        raise ChatError("messages must contain a user message")
    return converted, image_urls


def _flatten(content, index: int, allow_images: bool):
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    texts: List[str] = []
    urls: List[str] = []
    for part in content:
        if isinstance(part, str):
            texts.append(part)
            continue
        kind = part.get("type") if isinstance(part, dict) else None
        if kind == "text":
            texts.append(str(part.get("text", "")))
        elif kind == "image_url":
            if not allow_images:
                raise ChatError(f"messages[{index}]: images are only accepted from the user")
            url = part.get("image_url")
            url = url.get("url") if isinstance(url, dict) else url
            if not url:
                raise ChatError(f"messages[{index}]: image_url without a url")
            urls.append(str(url))
        else:
            raise ChatError(f"messages[{index}]: unsupported content part {kind!r}")
    return "\n".join(t for t in texts if t), urls


def _template_tool_call(call: dict, index: int) -> dict:
    """Chat templates iterate ``arguments`` as a mapping; OpenAI sends JSON text."""
    function = dict(call.get("function") or {})
    if not function.get("name"):
        raise ChatError(f"messages[{index}]: tool_calls entry without a function name")
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except ValueError as exc:
            raise ChatError(f"messages[{index}]: tool call arguments are not JSON: {exc}")
    function["arguments"] = arguments or {}
    return {"id": call.get("id"), "type": "function", "function": function}


def _validate_tools(tools: Optional[list]) -> Optional[list]:
    if not tools:
        return None
    for index, tool in enumerate(tools):
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict) or not function.get("name") \
                or tool.get("type", "function") != "function":
            raise ChatError(f"tools[{index}]: expected a function with a name")
        parameters = function.get("parameters")
        if parameters is not None and not isinstance(parameters, dict):
            raise ChatError(f"tools[{index}]: parameters must be a JSON schema object")
    return tools


def _json_schema(response_format: Optional[dict]) -> Optional[str]:
    if not response_format:
        return None
    kind = response_format.get("type")
    if kind in (None, "text"):
        return None
    if kind == "json_object":
        return json.dumps({"type": "object"})
    if kind == "json_schema":
        schema = (response_format.get("json_schema") or {}).get("schema")
        if not isinstance(schema, dict):
            raise ChatError("response_format.json_schema.schema must be an object")
        return json.dumps(schema)
    raise ChatError(f"unsupported response_format type {kind!r}")


def decode_images(urls: List[str]) -> list:
    """Decode data URLs / local paths into ``ov.Tensor`` frames (no network fetch)."""
    import numpy as np
    import openvino as ov
    from PIL import Image

    tensors = []
    for url in urls:
        if url.startswith("data:"):
            try:
                payload = url.split(",", 1)[1]
            except IndexError:
                raise ValueError("malformed data URL")
            if len(payload) > _MAX_IMAGE_BYTES * 4 // 3:
                raise ValueError("image exceeds the 32 MB limit")
            source = io.BytesIO(base64.b64decode(payload))
        elif url.startswith(("http://", "https://")):
            raise ValueError("remote image URLs are not fetched; send a data: URL")
        else:
            source = Path(url[len("file://"):] if url.startswith("file://") else url)
        try:
            image = Image.open(source).convert("RGB")
        except Exception as exc:  # noqa: BLE001 - any decode failure is a bad input
            raise ValueError(f"cannot decode image: {exc}") from exc
        tensors.append(ov.Tensor(np.asarray(image, dtype=np.uint8)[None]))
    return tensors


# ---------------------------------------------------------------------------
# Engine call -> response
# ---------------------------------------------------------------------------
async def _aiter_in_thread(iterator: Iterator[str], cancel: threading.Event) -> AsyncIterator[str]:
    """Consume a blocking token iterator on its own thread.

    The thread always drains the iterator to the end, so the engine's runner
    slot is released even if the client is gone; ``cancel`` makes that end
    come quickly by stopping the native generation.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def put(item) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, item)
        except RuntimeError:  # event loop closed; nobody is listening
            pass

    def pump() -> None:
        try:
            for token in iterator:
                ACTIVITY.touch()
                put(("token", token))
        except BaseException as exc:  # noqa: BLE001 - forwarded to the consumer
            put(("error", exc))
        else:
            put(("done", None))
        finally:
            ACTIVITY.end()

    threading.Thread(target=pump, name="chat-stream", daemon=True).start()
    try:
        while True:
            kind, value = await queue.get()
            if kind == "token":
                yield value
            elif kind == "error":
                raise value
            else:
                return
    finally:
        cancel.set()


async def _events(job: _Job, tokens: Iterator[str]) -> AsyncIterator[tuple]:
    """Yield ``(channel, text)`` pairs, then a single ``("end", splitter)``."""
    splitter = OutputSplitter(
        reasoning=bool(job.stats.get("thinking_open")), tools=bool(job.tools)
    )
    if job.prefill:
        splitter.feed(job.prefill)
    async for chunk in _aiter_in_thread(tokens, job.cancel):
        for event in splitter.feed(chunk):
            yield event
    for event in splitter.finish():
        yield event
    yield ("end", splitter)


def _finish(job: _Job, splitter: OutputSplitter):
    """Resolve tool calls and finish reason once generation is over."""
    calls: List[dict] = []
    leftover = ""
    if splitter.tool_text:
        try:
            calls = parse_tool_calls(splitter.tool_text, job.tools)
        except ToolCallError as exc:
            # Never publish an invalid call; hand the raw text back instead.
            logger.warning("Discarding invalid tool call: %s", exc)
            leftover = splitter.tool_text
        if calls and not job.parallel_tool_calls:
            calls = calls[:1]
    completion = int(job.stats.get("completion_tokens") or 0)
    limit = int(job.stats.get("max_new_tokens") or 0)
    if calls:
        reason = "tool_calls"
    elif limit and completion >= limit:
        reason = "length"
    else:
        reason = "stop"
    return calls, leftover, reason


def _usage(job: _Job) -> Optional[dict]:
    if "completion_tokens" not in job.stats:
        return None
    prompt = int(job.stats.get("prompt_tokens") or 0)
    completion = int(job.stats["completion_tokens"])
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion}


async def chat_completion(body: Any, engine, *, model_name: str,
                          default_thinking: Optional[bool] = None):
    """Serve one ``/v1/chat/completions`` request against ``engine``.

    Overload (``QueueFullError``) and memory errors (``OomError``) propagate to
    the app's exception handlers, which answer 503 with ``Retry-After``.
    """
    if hasattr(engine, "load"):
        # Lazy engines load here, so the tool-call format below is the model's.
        await run_in_threadpool(engine.load)
    try:
        req = ChatCompletionRequest.model_validate(body)
        job = build_job(req, getattr(engine, "tool_call_format", "json"), default_thinking)
        if job.image_urls:
            job.images = await run_in_threadpool(decode_images, job.image_urls)
    except ValidationError as exc:
        return error_response(f"Invalid request: {exc.errors(include_url=False)}")
    except (ChatError, ValueError) as exc:
        return error_response(str(exc), getattr(exc, "status", 400))

    ACTIVITY.begin()
    try:
        tokens = await run_in_threadpool(lambda: engine.generate(**job.engine_kwargs()))
    except ValueError as exc:
        ACTIVITY.end()
        return error_response(str(exc))
    except BaseException:
        ACTIVITY.end()
        raise

    events = _events(job, tokens)
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    if req.stream:
        # Wait for the first event so a failure before any output still gets a
        # proper HTTP status instead of a broken stream.
        first = await events.__anext__()
        include_usage = bool((req.stream_options or {}).get("include_usage"))
        return StreamingResponse(
            _sse(job, events, first, completion_id, created, model_name, include_usage),
            media_type="text/event-stream",
        )

    reasoning: List[str] = []
    content: List[str] = []
    splitter = None
    async for channel, value in events:
        if channel == "end":
            splitter = value
        elif channel == "reasoning":
            reasoning.append(value)
        else:
            content.append(value)
    calls, leftover, reason = _finish(job, splitter)
    text = "".join(content) + leftover
    message: Dict[str, Any] = {"role": "assistant", "content": text if text or not calls else None}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning).strip()
    if calls:
        message["tool_calls"] = calls
    return JSONResponse(content={
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "message": message, "finish_reason": reason}],
        "usage": _usage(job),
    })


async def _sse(job: _Job, events, first, completion_id: str, created: int,
               model_name: str, include_usage: bool):
    def chunk(delta: dict, finish_reason: Optional[str] = None, **extra) -> str:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            **extra,
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    yield chunk({"role": "assistant"})
    try:
        pending = [first]
        while True:
            event = pending.pop() if pending else await events.__anext__()
            channel, value = event
            if channel == "end":
                calls, leftover, reason = _finish(job, value)
                if leftover:
                    yield chunk({"content": leftover})
                for index, call in enumerate(calls):
                    yield chunk({"tool_calls": [{"index": index, **call}]})
                yield chunk({}, reason)
                if include_usage:
                    payload = {"id": completion_id, "object": "chat.completion.chunk",
                               "created": created, "model": model_name,
                               "choices": [], "usage": _usage(job)}
                    yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"
                return
            key = "reasoning_content" if channel == "reasoning" else "content"
            yield chunk({key: value})
    except Exception as exc:  # noqa: BLE001 - headers are sent; report in-band
        logger.error("Streaming generation failed: %s", exc)
        # No [DONE]: a stream that ends on an error event is not a valid answer.
        error = {"error": {"message": str(exc), "type": "server_error", "code": 500}}
        yield f"data: {json.dumps(error)}\n\n"


def models_payload(model_name: str, **details) -> dict:
    """``/v1/models`` in the standard list envelope."""
    entry = {"id": model_name, "object": "model", "created": 0, "owned_by": "edu-ai-suite"}
    entry.update({k: v for k, v in details.items() if v is not None})
    return {"object": "list", "data": [entry]}
