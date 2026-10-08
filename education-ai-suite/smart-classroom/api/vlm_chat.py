# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""``/v1/chat/completions`` and ``/v1/models`` on the main app (:8000).

With ``models.text_gen.serving.mode: inprocess`` (the default) requests run on
the warm in-process VLM. With ``managed`` / ``external`` they are proxied
byte-for-byte (SSE included) to the separate model server, so every existing
client keeps using :8000 unchanged.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx
from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from model_manager import ModelManager
from model_serving import openai_chat
from model_serving.settings import ServingSettings, serving_settings

logger = logging.getLogger(__name__)

router = APIRouter()

_FORWARDED_HEADERS = ("content-type", "retry-after", "cache-control")
_client: Optional[httpx.AsyncClient] = None


def _model_name() -> str:
    try:
        from utils.config_loader import config

        name = getattr(getattr(config.models, "text_gen", None), "vlm_name", None)
        if name:
            return str(name)
    except Exception:
        pass
    return "text_gen"


def _http(settings: ServingSettings) -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            trust_env=False,  # loopback must bypass corporate proxies
            # Read timeout spans the gap between streamed bytes, which covers a
            # long prefill; it is what turns a dead server into a fast 503.
            timeout=httpx.Timeout(connect=5.0, read=settings.stall_timeout_s,
                                  write=60.0, pool=None),
        )
    return _client


async def _proxy(request: Request, settings: ServingSettings, path: str):
    # Starts the managed server on first use if no feature warmed it up.
    await run_in_threadpool(ModelManager.instance().text_gen().load)
    headers = {"content-type": request.headers.get("content-type", "application/json")}
    if settings.api_key:
        headers["authorization"] = f"Bearer {settings.api_key}"
    client = _http(settings)
    upstream_request = client.build_request(
        request.method, settings.endpoint + path,
        content=await request.body() if request.method == "POST" else None,
        headers=headers,
    )
    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        logger.warning("Model service at %s unreachable: %s", settings.endpoint, exc)
        return openai_chat.error_response(
            f"model service unavailable: {exc.__class__.__name__}", 503,
            "unavailable", headers={"Retry-After": "5"},
        )
    return StreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        headers={k: upstream.headers[k] for k in _FORWARDED_HEADERS if k in upstream.headers},
        background=BackgroundTask(upstream.aclose),
    )


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completion (text, images, tools, streaming)."""
    settings = serving_settings()
    if settings.remote:
        return await _proxy(request, settings, "/v1/chat/completions")
    try:
        body = await request.json()
    except Exception as exc:  # noqa: BLE001 - malformed JSON body
        return openai_chat.error_response(f"Invalid request: {exc}")
    return await openai_chat.chat_completion(
        body,
        ModelManager.instance().text_gen(),
        model_name=_model_name(),
        default_thinking=settings.default_thinking,
    )


@router.get("/v1/models")
async def list_models(request: Request):
    settings = serving_settings()
    if settings.remote:
        return await _proxy(request, settings, "/v1/models")
    return JSONResponse(openai_chat.models_payload(_model_name()))
