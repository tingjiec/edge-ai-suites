# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Where the ``text_gen`` model runs, read from ``models.text_gen.serving``.

* ``inprocess`` (default) -- inside the Smart Classroom app, as before.
* ``managed`` -- in a separate ``python -m model_serving`` process the app
  starts, watches and restarts; the app proxies ``/v1/*`` on :8000 to it.
* ``external`` -- an already running model server the app only connects to
  (never starts or stops), e.g. one shared by several workflows.

Port 8000, the UI, Content Search and every request payload stay the same in
all three modes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

MODES = ("inprocess", "managed", "external")

#: Model server exit codes the supervisor acts on.
EXIT_LOAD_ERROR = 3  # model/config cannot load; restarting will not help
EXIT_STALLED = 4  # generation stopped making progress; restart

_MISSING = object()


@dataclass(frozen=True)
class ServingSettings:
    mode: str = "inprocess"
    host: str = "127.0.0.1"
    port: int = 8010
    endpoint: str = "http://127.0.0.1:8010"
    api_key: Optional[str] = None
    stall_timeout_s: float = 300.0
    max_restarts: int = 5
    ready_wait_s: float = 900.0
    # Applied when a chat request leaves enable_thinking unset.
    default_thinking: Optional[bool] = False

    @property
    def remote(self) -> bool:
        return self.mode != "inprocess"


def serving_settings(text_gen=None) -> ServingSettings:
    if text_gen is None:
        from utils.config_loader import config

        text_gen = getattr(config.models, "text_gen", None)
    serving = getattr(text_gen, "serving", None)

    mode = str(getattr(serving, "mode", None) or "inprocess").strip().lower()
    if mode not in MODES:
        raise ValueError(
            f"models.text_gen.serving.mode must be one of {', '.join(MODES)}; got {mode!r}"
        )
    host = str(getattr(serving, "host", None) or "127.0.0.1")
    port = int(getattr(serving, "port", None) or 8010)
    endpoint = str(getattr(serving, "endpoint", None) or f"http://{host}:{port}")
    api_key = os.environ.get("MODEL_SERVING_API_KEY") or getattr(serving, "api_key", None)

    thinking = getattr(text_gen, "enable_thinking", _MISSING)
    default_thinking = False if thinking is _MISSING else thinking

    return ServingSettings(
        mode=mode,
        host=host,
        port=port,
        endpoint=endpoint.rstrip("/"),
        api_key=str(api_key) if api_key else None,
        stall_timeout_s=float(getattr(serving, "stall_timeout_s", None) or 300),
        max_restarts=int(getattr(serving, "max_restarts", None) or 5),
        ready_wait_s=float(getattr(serving, "ready_wait_s", None) or 900),
        default_thinking=None if default_thinking is None else bool(default_thinking),
    )
