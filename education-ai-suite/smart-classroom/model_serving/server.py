# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Standalone model server: the ``text_gen`` engine behind an OpenAI API.

Endpoints:

* ``POST /v1/chat/completions`` / ``GET /v1/models`` -- same contract as :8000;
* ``GET /live`` -- the process answers (503 once generation has stalled);
* ``GET /ready`` -- the model is loaded and warmed up;
* ``GET /health`` -- the main app's ``{"status", "hub": {"text_gen"}}`` shape,
  so Content Search's readiness gate can point here directly.

The model loads on a background thread so ``/live`` answers at once. A load
failure exits with :data:`EXIT_LOAD_ERROR`; a generation that makes no
progress for ``stall_timeout_s`` exits with :data:`EXIT_STALLED`, so any
supervisor (the app's, a Windows service, systemd) can recycle the process --
a thread timeout cannot stop a hung native inference.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from model_manager.capability.runner import OomError, QueueFullError
from model_serving import openai_chat
from model_serving.settings import EXIT_LOAD_ERROR, EXIT_STALLED, serving_settings

logger = logging.getLogger(__name__)

_WATCHDOG_INTERVAL_S = 5


class _Service:
    def __init__(self) -> None:
        from components.vlm.text_gen_handle import TextGenHandler

        # Always the in-process engine: this process *is* the model service.
        from utils.config_loader import config

        self.handler = TextGenHandler(mode="inprocess")
        self.settings = serving_settings()
        text_gen = getattr(config.models, "text_gen", None)
        self.model_name = str(getattr(text_gen, "vlm_name", "") or "text_gen")
        self.phase = "loading"
        self.error: Optional[str] = None
        self.loaded_at: Optional[float] = None

    def load(self) -> None:
        started = time.monotonic()
        try:
            self.handler.load()
        except BaseException as exc:  # noqa: BLE001 - reported, then the process exits
            self.phase, self.error = "failed", str(exc)
            logger.exception("Model load failed; exiting with code %d.", EXIT_LOAD_ERROR)
            logging.shutdown()
            os._exit(EXIT_LOAD_ERROR)
        self.phase, self.loaded_at = "ready", time.monotonic()
        logger.info("Model service ready in %.0fs.", self.loaded_at - started)

    def text_gen_health(self) -> dict:
        h = self.handler
        return {
            "state": h.state.value,
            "loaded": h.loaded,
            "provider": h.provider,
            "device": h.device,
            "max_concurrency": h.max_concurrency,
            **h.describe(),
        }


def _watchdog(service: _Service, parent_pid: Optional[int]) -> None:
    stall_limit = service.settings.stall_timeout_s
    while True:
        time.sleep(_WATCHDOG_INTERVAL_S)
        if parent_pid and not _pid_alive(parent_pid):
            logger.warning("Owning process %d is gone; model service exiting.", parent_pid)
            os._exit(0)
        stalled = openai_chat.ACTIVITY.stalled_for()
        if stall_limit and stalled > stall_limit:
            logger.critical(
                "No token generated for %.0fs (limit %.0fs); exiting so the "
                "supervisor restarts a clean worker.", stalled, stall_limit,
            )
            logging.shutdown()
            os._exit(EXIT_STALLED)


def _pid_alive(pid: int) -> bool:
    try:
        import psutil

        return psutil.pid_exists(pid)
    except ImportError:
        return True


def create_app(parent_pid: Optional[int] = None) -> FastAPI:
    service = _Service()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        threading.Thread(target=service.load, name="model-load", daemon=True).start()
        threading.Thread(
            target=_watchdog, args=(service, parent_pid), name="model-watchdog", daemon=True
        ).start()
        yield
        service.handler.shutdown()

    app = FastAPI(title="edu-ai-suite model serving", lifespan=lifespan)
    app.state.service = service

    @app.middleware("http")
    async def _auth(request: Request, call_next):
        key = service.settings.api_key
        if key and request.url.path.startswith("/v1/"):
            if request.headers.get("authorization") != f"Bearer {key}":
                return openai_chat.error_response(
                    "invalid or missing API key", 401, "authentication_error"
                )
        return await call_next(request)

    @app.exception_handler(QueueFullError)
    async def _queue_full(request: Request, exc: QueueFullError):
        return openai_chat.error_response(
            str(exc), 503, "overloaded", headers={"Retry-After": "2"}
        )

    @app.exception_handler(OomError)
    async def _oom(request: Request, exc: OomError):
        return openai_chat.error_response(
            str(exc), 503, "memory_pressure", headers={"Retry-After": "5"}
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        if service.phase != "ready":
            return openai_chat.error_response(
                f"model is {service.phase}", 503, "unavailable",
                headers={"Retry-After": "10"},
            )
        try:
            body = await request.json()
        except Exception as exc:  # noqa: BLE001 - malformed JSON body
            return openai_chat.error_response(f"Invalid request: {exc}")
        return await openai_chat.chat_completion(
            body,
            service.handler,
            model_name=service.model_name,
            default_thinking=service.settings.default_thinking,
        )

    @app.get("/v1/models")
    async def models():
        return openai_chat.models_payload(service.model_name)

    @app.get("/live")
    async def live():
        stalled = openai_chat.ACTIVITY.stalled_for()
        limit = service.settings.stall_timeout_s
        if limit and stalled > limit:
            return JSONResponse(status_code=503, content={"status": "stalled",
                                                          "stalled_s": round(stalled)})
        return {"status": "alive"}

    @app.get("/ready")
    async def ready():
        body = {"status": service.phase, "model": service.model_name}
        if service.error:
            body["error"] = service.error
        return JSONResponse(status_code=200 if service.phase == "ready" else 503, content=body)

    @app.get("/health")
    async def health():
        return {"status": "ok", "hub": {"text_gen": service.text_gen_health()}}

    return app
