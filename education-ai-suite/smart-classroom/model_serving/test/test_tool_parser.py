# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Both Qwen tool-call dialects parse, and nothing invalid looks executable."""

import json

import pytest

from model_serving.tool_parser import (
    OutputSplitter,
    ToolCallError,
    forced_call_prefix,
    parse_tool_calls,
)

WEATHER = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"},
                           "metric": {"type": "boolean"}},
            "required": ["city"],
        },
    },
}]

XML_CALL = ("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
            "<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>")


def _args(call):
    return json.loads(call["function"]["arguments"])


def test_qwen3coder_xml_call_is_typed_by_the_schema():
    """Qwen3.6/3.8 output, verbatim from a PTL run."""
    [call] = parse_tool_calls(XML_CALL, WEATHER)
    assert call["type"] == "function" and call["id"].startswith("call_")
    assert call["function"]["name"] == "get_weather"
    assert _args(call) == {"city": "Paris", "days": 2}


def test_a_string_parameter_stays_a_string():
    text = XML_CALL.replace("Paris", "1984")
    assert _args(parse_tool_calls(text, WEATHER)[0])["city"] == "1984"


def test_python_booleans_from_the_qwen36_template_are_read():
    text = XML_CALL.replace("<parameter=days>\n2", "<parameter=metric>\nTrue")
    assert _args(parse_tool_calls(text, WEATHER)[0])["metric"] is True


def test_hermes_json_call_from_qwen3_vl():
    text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Oslo"}}\n</tool_call>'
    assert _args(parse_tool_calls(text, WEATHER)[0]) == {"city": "Oslo"}


def test_parallel_calls_keep_their_order():
    text = XML_CALL + "\n" + XML_CALL.replace("Paris", "Rome")
    calls = parse_tool_calls(text, WEATHER)
    assert [_args(c)["city"] for c in calls] == ["Paris", "Rome"]
    assert calls[0]["id"] != calls[1]["id"]


@pytest.mark.parametrize("text", [
    XML_CALL[:-len("</tool_call>")],                       # cut off by max_tokens
    XML_CALL.replace("get_weather", "rm_rf"),              # tool never offered
    XML_CALL.replace("<parameter=city>\nParis\n</parameter>\n", ""),  # required arg
    "<tool_call>\n{not json}\n</tool_call>",
])
def test_invalid_calls_are_rejected(text):
    with pytest.raises(ToolCallError):
        parse_tool_calls(text, WEATHER)


def test_splitter_separates_reasoning_answer_and_call_across_chunk_edges():
    splitter = OutputSplitter(reasoning=True, tools=True)
    events = []
    for chunk in ["\nweigh", "ing it</th", "ink>\n\nIt is", " sunny.\n<tool", "_call>\n<fun"]:
        events += splitter.feed(chunk)
    events += splitter.finish()
    reasoning = "".join(t for c, t in events if c == "reasoning")
    content = "".join(t for c, t in events if c == "content")
    assert reasoning == "weighing it"
    assert content == "It is sunny.\n"
    assert splitter.tool_text == "<tool_call>\n<fun"


def test_splitter_without_tools_passes_tool_tags_through():
    splitter = OutputSplitter(tools=False)
    events = splitter.feed("say <tool_call> literally") + splitter.finish()
    assert "".join(t for _, t in events) == "say <tool_call> literally"


def test_forced_prefix_matches_the_dialect():
    assert forced_call_prefix("xml", "get_weather") == "<tool_call>\n<function=get_weather>\n"
    assert forced_call_prefix("json", "f") == '<tool_call>\n{"name": "f", "arguments": '
    assert forced_call_prefix("xml", None) == "<tool_call>\n"
