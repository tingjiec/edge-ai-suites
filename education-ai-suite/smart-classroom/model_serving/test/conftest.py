# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The repo on the import path, and a fake engine so no model is needed."""

import os
import sys
from typing import Iterator, List, Optional

import pytest

_SC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SC_ROOT not in sys.path:
    sys.path.insert(0, _SC_ROOT)


class FakeEngine:
    """Replays canned token chunks through ``TextGenHandler.generate``'s keywords."""

    model_name = "Qwen/Fake"

    def __init__(self, chunks: List[str], *, tool_call_format: str = "xml",
                 thinking_open: bool = False, completion_tokens: int = 5,
                 max_new_tokens: int = 100, error: Optional[Exception] = None):
        self.chunks = chunks
        self.tool_call_format = tool_call_format
        self.thinking_open = thinking_open
        self.completion_tokens = completion_tokens
        self.max_new_tokens = max_new_tokens
        self.error = error
        self.calls: List[dict] = []

    def generate(self, prompt=None, *, stats=None, cancel_event=None, **kwargs) -> Iterator[str]:
        self.calls.append(kwargs)
        if stats is not None:
            stats.update(thinking_open=self.thinking_open, prompt_tokens=11,
                         completion_tokens=self.completion_tokens,
                         max_new_tokens=kwargs.get("max_new_tokens") or self.max_new_tokens)
        chunks, error = list(self.chunks), self.error

        def stream():
            for chunk in chunks:
                yield chunk
            if error is not None:
                raise error

        return stream()


@pytest.fixture
def fake_engine():
    return FakeEngine
