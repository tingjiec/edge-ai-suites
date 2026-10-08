# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Turn raw model output into OpenAI ``reasoning_content``, ``content`` and ``tool_calls``.

Qwen families write tool calls in one of two dialects, both inside ``<tool_call>``:

* Hermes JSON (Qwen2.5, Qwen3, Qwen3-VL)::

    <tool_call>
    {"name": "get_weather", "arguments": {"city": "Paris"}}
    </tool_call>

* qwen3coder XML (Qwen3.5, Qwen3.6, Qwen3.8)::

    <tool_call>
    <function=get_weather>
    <parameter=city>
    Paris
    </parameter>
    </function>
    </tool_call>

Both are accepted regardless of the model, so an upgrade that switches dialect
needs no code change. The parser only proposes calls; executing them is the
client's (agent controller's) job.
"""

from __future__ import annotations

import ast
import json
import re
import uuid
from typing import Dict, List, Optional, Tuple

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"

_BLOCK_RE = re.compile(re.escape(TOOL_OPEN) + r"(.*?)" + re.escape(TOOL_CLOSE), re.S)
_FUNCTION_RE = re.compile(r"<function=([^>\n]+)>(.*?)(?:</function>|\Z)", re.S)
_PARAMETER_RE = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.S)
_PY_LITERALS = {"True": True, "False": False, "None": None}


class ToolCallError(ValueError):
    """The model wrote something that is not a valid call to an offered tool."""


class OutputSplitter:
    """Route streamed text to the ``reasoning`` and ``content`` channels.

    Everything from the first ``<tool_call>`` on is held in :attr:`tool_text`
    for :func:`parse_tool_calls`, so a partial call never reaches the client.
    A tag split across chunks is held back until it can be told from text.
    """

    def __init__(self, *, reasoning: bool = False, tools: bool = False) -> None:
        self._channel = "reasoning" if reasoning else "content"
        self._tools = tools
        self._pending = ""
        self._fresh = True
        self.tool_text = ""

    def feed(self, text: str) -> List[Tuple[str, str]]:
        self._pending += text
        out: List[Tuple[str, str]] = []
        while self._pending:
            if self._channel == "tools":
                self.tool_text += self._pending
                self._pending = ""
                break
            tags = self._tags()
            pos, tag = _first_tag(self._pending, tags)
            if tag is None:
                keep = _partial_tag_suffix(self._pending, tags)
                cut = len(self._pending) - keep
                self._emit(out, self._pending[:cut])
                self._pending = self._pending[cut:]
                break
            self._emit(out, self._pending[:pos])
            self._pending = self._pending[pos + len(tag):]
            self._switch(tag)
        return out

    def finish(self) -> List[Tuple[str, str]]:
        """Flush whatever was held back once generation has ended."""
        out: List[Tuple[str, str]] = []
        if self._channel == "tools":
            self.tool_text += self._pending
        else:
            self._emit(out, self._pending)
        self._pending = ""
        return out

    def _tags(self) -> Tuple[str, ...]:
        if self._channel == "reasoning":
            return (THINK_CLOSE,)
        # A stray </think> in the answer is dropped rather than shown.
        tags = (THINK_OPEN, THINK_CLOSE)
        return tags + (TOOL_OPEN,) if self._tools else tags

    def _switch(self, tag: str) -> None:
        if tag == TOOL_OPEN:
            self._channel = "tools"
            self.tool_text = TOOL_OPEN
        elif tag == THINK_OPEN:
            self._channel = "reasoning"
        else:
            self._channel = "content"
        self._fresh = True

    def _emit(self, out: list, text: str) -> None:
        if self._fresh:
            # Templates put newlines around the think block; they are not output.
            text = text.lstrip()
            if not text:
                return
            self._fresh = False
        if text:
            out.append((self._channel, text))


def _first_tag(text: str, tags) -> Tuple[int, Optional[str]]:
    best, best_tag = -1, None
    for tag in tags:
        pos = text.find(tag)
        if pos != -1 and (best == -1 or pos < best):
            best, best_tag = pos, tag
    return best, best_tag


def _partial_tag_suffix(text: str, tags) -> int:
    """Length of the longest suffix of ``text`` that could start a tag."""
    longest = 0
    for tag in tags:
        for size in range(min(len(tag) - 1, len(text)), 0, -1):
            if text.endswith(tag[:size]):
                longest = max(longest, size)
                break
    return longest


# ---------------------------------------------------------------------------
# Tool-call parsing
# ---------------------------------------------------------------------------
def tool_schemas(tools: Optional[list]) -> Dict[str, dict]:
    """Map each offered function name to its JSON-schema ``parameters``."""
    schemas: Dict[str, dict] = {}
    for tool in tools or []:
        function = tool.get("function") or {}
        name = function.get("name")
        if name:
            schemas[str(name)] = function.get("parameters") or {}
    return schemas


def parse_tool_calls(text: str, tools: Optional[list]) -> List[dict]:
    """Parse every ``<tool_call>`` block in ``text`` into OpenAI ``tool_calls``.

    Raises :class:`ToolCallError` when a block is unterminated (e.g. cut off
    by ``max_tokens``), malformed, names a tool that was not offered, or omits
    a required argument -- an invalid call must never look executable.
    """
    schemas = tool_schemas(tools)
    blocks = _BLOCK_RE.findall(text)
    if not blocks or len(blocks) != text.count(TOOL_OPEN):
        raise ToolCallError("unterminated <tool_call> block")

    calls = []
    for block in blocks:
        body = block.strip()
        if body.startswith("<function="):
            name, arguments = _parse_xml_call(body, schemas)
        else:
            name, arguments = _parse_json_call(body)
        _validate(name, arguments, schemas)
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        })
    return calls


def _parse_xml_call(body: str, schemas: Dict[str, dict]) -> Tuple[str, dict]:
    match = _FUNCTION_RE.match(body)
    if match is None:
        raise ToolCallError("malformed <function=...> block")
    name = match.group(1).strip()
    properties = (schemas.get(name) or {}).get("properties") or {}
    arguments = {}
    for key, raw in _PARAMETER_RE.findall(match.group(2)):
        key = key.strip()
        arguments[key] = _coerce(raw, properties.get(key))
    return name, arguments


def _parse_json_call(body: str) -> Tuple[str, dict]:
    try:
        payload = json.loads(body)
    except ValueError as exc:
        raise ToolCallError(f"tool call is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or not payload.get("name"):
        raise ToolCallError("tool call JSON has no 'name'")
    arguments = payload.get("arguments", payload.get("parameters", {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except ValueError as exc:
            raise ToolCallError(f"tool call arguments are not JSON: {exc}") from exc
    return str(payload["name"]), arguments


def _coerce(raw: str, schema: Optional[dict]):
    """Type an XML parameter value using the tool's schema where it has one."""
    value = raw.strip("\n")
    declared = (schema or {}).get("type")
    if declared == "string" or (isinstance(declared, list) and declared == ["string"]):
        return value
    text = value.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    if text in _PY_LITERALS:  # the Qwen3.6 template writes Python booleans
        return _PY_LITERALS[text]
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return value


def _validate(name: str, arguments, schemas: Dict[str, dict]) -> None:
    if schemas and name not in schemas:
        raise ToolCallError(f"model called unknown tool {name!r}")
    if not isinstance(arguments, dict):
        raise ToolCallError(f"arguments for {name!r} are not an object")
    required = (schemas.get(name) or {}).get("required") or []
    missing = [key for key in required if key not in arguments]
    if missing:
        raise ToolCallError(f"call to {name!r} misses required argument(s) {missing}")


def forced_call_prefix(tool_call_format: str, name: Optional[str]) -> str:
    """Text that opens a tool call, appended to the prompt for ``tool_choice``.

    Starting the assistant turn inside ``<tool_call>`` makes ``required`` (and
    a named function) a guarantee instead of a hint.
    """
    if tool_call_format == "xml":
        return TOOL_OPEN + "\n" + (f"<function={name}>\n" if name else "")
    return TOOL_OPEN + "\n" + (f'{{"name": {json.dumps(name)}, "arguments": ' if name else "")
