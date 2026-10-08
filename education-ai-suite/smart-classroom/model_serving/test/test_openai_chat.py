# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The /v1/chat/completions contract, against a fake engine."""

import asyncio
import json
import threading

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from model_manager.capability.runner import QueueFullError
from model_serving import openai_chat

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                       "required": ["city"]},
    },
}]
XML_CALL = ["<tool", "_call>\n<function=get_weather>\n<parameter=city>\nParis\n",
            "</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>"]


def _client(engine, default_thinking=False):
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await openai_chat.chat_completion(
            await request.json(), engine, model_name="Qwen/Fake",
            default_thinking=default_thinking,
        )

    @app.exception_handler(QueueFullError)
    async def _full(request, exc):
        return JSONResponse(status_code=503, headers={"Retry-After": "2"}, content={})

    return TestClient(app)


def _post(client, **body):
    body.setdefault("messages", [{"role": "user", "content": "hi"}])
    return client.post("/v1/chat/completions", json=body)


def _sse(resp):
    chunks, done = [], False
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            data = line[len("data: "):]
            if data == "[DONE]":
                done = True
            else:
                chunks.append(json.loads(data))
    return chunks, done


# ---------------------------------------------------------------- plain text

def test_non_streaming_shape_is_what_content_search_reads(fake_engine):
    resp = _post(_client(fake_engine(["Hello", " world"])))
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion" and data["model"] == "Qwen/Fake"
    choice = data["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"] == {"role": "assistant", "content": "Hello world"}
    assert data["usage"] == {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}


def test_streaming_shape_matches_the_legacy_sse(fake_engine):
    resp = _post(_client(fake_engine(["ok", "!"])), stream=True)
    chunks, done = _sse(resp)
    assert done
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "ok!"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)


def test_hitting_max_tokens_reports_length(fake_engine):
    resp = _post(_client(fake_engine(["a"], completion_tokens=8)), max_tokens=8)
    assert resp.json()["choices"][0]["finish_reason"] == "length"


def test_full_history_reaches_the_engine(fake_engine):
    engine = fake_engine(["ok"])
    _post(_client(engine), messages=[
        {"role": "developer", "content": "be brief"},
        {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
        {"role": "assistant", "content": "c"},
        {"role": "user", "content": "d"},
    ])
    assert engine.calls[0]["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "a\nb"},
        {"role": "assistant", "content": "c"},
        {"role": "user", "content": "d"},
    ]


# ---------------------------------------------------------------- reasoning

def test_reasoning_is_split_from_the_answer(fake_engine):
    engine = fake_engine(["\nthink", "ing</think>\n\n", "42"], thinking_open=True)
    message = _post(_client(engine), enable_thinking=True).json()["choices"][0]["message"]
    assert message["reasoning_content"] == "thinking"
    assert message["content"] == "42"


def test_thinking_defaults_to_the_server_setting(fake_engine):
    engine = fake_engine(["x"])
    client = _client(engine, default_thinking=False)
    _post(client)
    _post(client, enable_thinking=True)
    _post(client, chat_template_kwargs={"enable_thinking": True, "reasoning_effort": "low"})
    assert [c["enable_thinking"] for c in engine.calls] == [False, True, True]
    assert engine.calls[2]["template_kwargs"] == {"reasoning_effort": "low"}


# ---------------------------------------------------------------- tools

def test_tool_call_comes_back_as_tool_calls(fake_engine):
    engine = fake_engine(XML_CALL)
    data = _post(_client(engine), tools=TOOLS).json()
    choice = data["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    [call] = choice["message"]["tool_calls"]
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris", "days": 2}
    assert engine.calls[0]["tools"] == TOOLS


def test_streamed_tool_call_is_one_complete_delta(fake_engine):
    chunks, done = _sse(_post(_client(fake_engine(["Checking. "] + XML_CALL)),
                              tools=TOOLS, stream=True))
    assert done
    deltas = [c["choices"][0]["delta"] for c in chunks]
    assert "".join(d.get("content", "") for d in deltas) == "Checking. "
    [calls] = [d["tool_calls"] for d in deltas if "tool_calls" in d]
    assert calls[0]["index"] == 0 and calls[0]["function"]["name"] == "get_weather"
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_an_invalid_call_is_returned_as_text_not_as_a_call(fake_engine):
    bad = [c.replace("get_weather", "format_disk") for c in XML_CALL]
    choice = _post(_client(fake_engine(bad)), tools=TOOLS).json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert "tool_calls" not in choice["message"]
    assert "format_disk" in choice["message"]["content"]


def test_named_tool_choice_forces_the_call(fake_engine):
    # The prefix is in the prompt, so the model only writes the parameters.
    engine = fake_engine(["<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>"])
    choice = _post(_client(engine), tools=TOOLS, tool_choice={
        "type": "function", "function": {"name": "get_weather"}}).json()["choices"][0]
    assert engine.calls[0]["prefill"] == "<tool_call>\n<function=get_weather>\n"
    assert engine.calls[0]["enable_thinking"] is False
    assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {"city": "Oslo"}


def test_tool_choice_none_hides_the_tools(fake_engine):
    engine = fake_engine(["plain"])
    _post(_client(engine), tools=TOOLS, tool_choice="none")
    assert "tools" not in engine.calls[0]


def test_a_tool_round_trip_renders_arguments_as_a_mapping(fake_engine):
    engine = fake_engine(["It is sunny."])
    resp = _post(_client(engine), tools=TOOLS, messages=[
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ])
    assert resp.status_code == 200
    history = engine.calls[0]["messages"]
    assert history[1]["tool_calls"][0]["function"]["arguments"] == {"city": "Paris"}
    assert history[2] == {"role": "tool", "content": "sunny"}


@pytest.mark.parametrize("messages", [
    [{"role": "user", "content": "x"}, {"role": "tool", "content": "orphan"}],
    [{"role": "user", "content": "x"},
     {"role": "assistant", "tool_calls": [{"id": "a", "function": {"name": "get_weather",
                                                                    "arguments": "{}"}}]},
     {"role": "tool", "tool_call_id": "a", "content": "1"},
     {"role": "tool", "tool_call_id": "a", "content": "duplicate"}],
    [{"role": "narrator", "content": "x"}],
    [{"role": "assistant", "content": [{"type": "image_url", "image_url": {"url": "data:,"}}]}],
    [{"role": "user", "content": [{"type": "image_url",
                                   "image_url": {"url": "http://169.254.169.254/x.png"}}]}],
])
def test_malformed_conversations_are_rejected_before_generation(fake_engine, messages):
    engine = fake_engine(["never"])
    resp = _post(_client(engine), tools=TOOLS, messages=messages)
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"
    assert engine.calls == []


def test_response_format_with_tools_is_rejected(fake_engine):
    resp = _post(_client(fake_engine([])), tools=TOOLS,
                 response_format={"type": "json_object"})
    assert resp.status_code == 400


def test_response_format_schema_reaches_the_engine(fake_engine):
    engine = fake_engine(["{}"])
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    _post(_client(engine), response_format={"type": "json_schema",
                                            "json_schema": {"name": "x", "schema": schema}})
    assert json.loads(engine.calls[0]["json_schema"]) == schema


# ---------------------------------------------------------------- failures

def test_overload_is_a_503_before_any_stream_starts(fake_engine):
    class Full(fake_engine):
        def generate(self, *a, **k):
            raise QueueFullError("Queue full (8)")

    resp = _post(_client(Full([])), stream=True)
    assert resp.status_code == 503 and resp.headers["Retry-After"] == "2"


def test_a_mid_stream_failure_ends_on_an_error_event(fake_engine):
    engine = fake_engine(["partial"], error=RuntimeError("device lost"))
    chunks, done = _sse(_post(_client(engine), stream=True))
    assert not done
    assert chunks[-1]["error"]["message"] == "device lost"


def test_an_abandoned_stream_cancels_generation_and_drains():
    """The token iterator is always exhausted so the runner slot is released."""
    drained = threading.Event()

    def tokens():
        try:
            yield "a"
            yield "b"
        finally:
            drained.set()

    cancel = threading.Event()

    async def take_one():
        stream = openai_chat._aiter_in_thread(tokens(), cancel)
        await stream.__anext__()
        await stream.aclose()

    asyncio.run(take_one())
    assert cancel.is_set()
    assert drained.wait(5)
