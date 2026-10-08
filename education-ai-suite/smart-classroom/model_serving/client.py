# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The app side of a separate model server (``serving.mode`` managed/external).

:class:`RemoteTextGen` is a drop-in for ``VLMTextGen``: the summarizer, mind
map, segmentation, report and board-OCR features keep calling
``ModelManager.text_gen().generate(...)`` and get the same strings and token
streams, now produced over HTTP. :class:`ServingSupervisor` owns the server
process in ``managed`` mode.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import random
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Iterator, Optional, Union

import requests

from model_manager.capability.runner import QueueFullError
from model_serving.settings import EXIT_LOAD_ERROR, ServingSettings

logger = logging.getLogger(__name__)

_SC_ROOT = Path(__file__).resolve().parents[1]
_READY_CACHE_S = 2.0
_RESTART_WINDOW_S = 600
_HEALTHY_RUN_S = 600
_LIVE_FAILURES_BEFORE_RECYCLE = 6  # x 5 s poll = 30 s unresponsive
_UNSET = object()


class ModelServiceUnavailable(QueueFullError):
    """The model server is down, still loading, or overloaded (HTTP 503)."""


def _probe(url: str, timeout: float = 2.0) -> bool:
    try:
        return requests.get(url, timeout=timeout, proxies={"http": None, "https": None}).ok
    except requests.RequestException:
        return False


class ServingSupervisor:
    """Start ``python -m model_serving`` and keep it running.

    A crashed or stalled server is restarted with capped exponential backoff
    and jitter, at most ``max_restarts`` times per 10 minutes. Exit code
    :data:`EXIT_LOAD_ERROR` is final: a missing model or an old runtime will
    not load on the next attempt either. If a server already answers on the
    port, it is used and left alone.
    """

    def __init__(self, settings: ServingSettings) -> None:
        self._settings = settings
        self._base = settings.endpoint
        self._stop = threading.Event()
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self.failed = False
        self.last_error: Optional[str] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        if _probe(f"{self._base}/live"):
            logger.info("Model service already running at %s; using it as is.", self._base)
            return
        self._thread = threading.Thread(target=self._run, name="model-supervisor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._terminate()
        if self._thread is not None:
            self._thread.join(timeout=15)

    def _spawn(self) -> subprocess.Popen:
        cmd = [
            sys.executable, "-m", "model_serving",
            "--host", self._settings.host,
            "--port", str(self._settings.port),
            "--parent-pid", str(os.getpid()),
        ]
        logger.info("Starting model service: %s", " ".join(cmd))
        # Inherits SC_CONFIG_PATH and the console, so it reads the same config
        # and its load progress shows up next to the app's own log.
        return subprocess.Popen(cmd, cwd=str(_SC_ROOT), env=os.environ.copy())

    def _run(self) -> None:
        restarts: deque = deque()
        attempt = 0
        while not self._stop.is_set():
            started = time.monotonic()
            self._proc = self._spawn()
            code = self._watch(self._proc)
            if self._stop.is_set():
                break
            if code == EXIT_LOAD_ERROR:
                self._fail(f"model service could not load the model (exit {code}); "
                           "see its log above")
                break
            now = time.monotonic()
            if now - started > _HEALTHY_RUN_S:
                attempt = 0
            restarts.append(now)
            while restarts and now - restarts[0] > _RESTART_WINDOW_S:
                restarts.popleft()
            if len(restarts) > self._settings.max_restarts:
                self._fail(f"model service exited {len(restarts)} times in "
                           f"{_RESTART_WINDOW_S // 60} min (last exit {code}); giving up")
                break
            delay = min(60.0, 2.0 ** attempt) * random.uniform(0.8, 1.2)
            attempt += 1
            logger.warning("Model service exited (code %s); restarting in %.0fs.", code, delay)
            self._stop.wait(delay)

    def _watch(self, proc: subprocess.Popen) -> int:
        """Wait for ``proc`` to exit; recycle it if it stops answering /live."""
        failures, seen_live = 0, False
        while True:
            try:
                return proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            if self._stop.is_set():
                self._terminate()
                return proc.wait()
            if _probe(f"{self._base}/live"):
                seen_live, failures = True, 0
            elif seen_live:
                failures += 1
                if failures >= _LIVE_FAILURES_BEFORE_RECYCLE:
                    logger.error("Model service stopped answering /live; recycling it.")
                    self._terminate()
                    return proc.wait()

    def _terminate(self) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    def _fail(self, message: str) -> None:
        self.failed, self.last_error = True, message
        logger.error(message)


class RemoteTextGen:
    """``VLMTextGen``'s interface, served by a model server over HTTP."""

    def __init__(self, settings: ServingSettings, text_gen_cfg=None,
                 supervisor: Optional[ServingSupervisor] = None) -> None:
        self._settings = settings
        self._base = settings.endpoint
        self._supervisor = supervisor
        self._session = requests.Session()
        self._session.trust_env = False  # loopback must bypass corporate proxies
        self._headers = {"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {}
        self._model_name = str(getattr(text_gen_cfg, "vlm_name", "") or "") or None
        self._device = str(getattr(text_gen_cfg, "device", "") or "").upper() or None
        self._weight_format = str(getattr(text_gen_cfg, "weight_format", "") or "").lower() or None
        self._tokenizer = _UNSET
        self._ready_checked = 0.0
        self._ready = False
        self.speculative_status = "remote"
        self.tool_call_format = "json"
        self.chat_template = ""

    @property
    def device(self) -> Optional[str]:
        return self._device

    @property
    def model_name(self) -> Optional[str]:
        return self._model_name

    @property
    def endpoint(self) -> str:
        return self._base

    @property
    def error(self) -> Optional[str]:
        return self._supervisor.last_error if self._supervisor else None

    @property
    def ready(self) -> bool:
        now = time.monotonic()
        if now - self._ready_checked > _READY_CACHE_S:
            self._ready = _probe(f"{self._base}/ready")
            self._ready_checked = now
        return self._ready

    @property
    def tokenizer(self):
        """Tokenizer files from the local IR, for token budgeting only."""
        if self._tokenizer is _UNSET:
            self._tokenizer = None
            if self._model_name and self._weight_format:
                from utils.model_paths import openvino_model_dir

                model_dir = openvino_model_dir(self._model_name, self._weight_format)
                if (model_dir / "tokenizer_config.json").exists():
                    try:
                        from transformers import AutoTokenizer

                        self._tokenizer = AutoTokenizer.from_pretrained(
                            str(model_dir), extra_special_tokens={}
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Local tokenizer for %s unavailable: %s", model_dir, exc)
        if self._tokenizer is None:
            raise RuntimeError("no local tokenizer for the remote model")
        return self._tokenizer

    def release(self) -> None:
        self._session.close()
        if self._supervisor is not None:
            self._supervisor.stop()

    def generate(
        self,
        prompt: Optional[str] = None,
        *,
        messages: Optional[list] = None,
        images: Optional[list] = None,
        stream: bool = True,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        enable_thinking: Optional[bool] = None,
        json_schema: Optional[str] = None,
        stats: Optional[dict] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Union[Iterator[str], str]:
        if (messages is None) == (prompt is None):
            raise ValueError("Provide exactly one of prompt or messages.")
        messages = [dict(m) for m in messages] if messages is not None else [
            {"role": "user", "content": prompt}
        ]
        if not messages:
            raise ValueError("Invalid messages provided.")
        if images:
            _attach_images(messages, images)

        # Always streamed on the wire: a long non-streamed answer would sit
        # silent past the read timeout that detects a dead server.
        payload = {"messages": messages, "stream": True,
                   "stream_options": {"include_usage": True}}
        if self._model_name:
            payload["model"] = self._model_name
        if max_new_tokens is not None:
            payload["max_completion_tokens"] = int(max_new_tokens)
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if enable_thinking is not None:
            payload["enable_thinking"] = bool(enable_thinking)
        if json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": json.loads(json_schema)},
            }

        tokens = self._stream(self._post(payload), stats, cancel_event)
        return tokens if stream else "".join(tokens)

    def _post(self, payload: dict) -> requests.Response:
        """POST, retrying only while nothing has been generated yet."""
        url = f"{self._base}/v1/chat/completions"
        deadline = time.monotonic() + self._settings.ready_wait_s
        delay = 1.0
        while True:
            if self._supervisor is not None and self._supervisor.failed:
                raise ModelServiceUnavailable(self._supervisor.last_error or "model service failed")
            try:
                resp = self._session.post(
                    url, json=payload, headers=self._headers, stream=True,
                    timeout=(5, self._settings.stall_timeout_s),
                )
            except requests.RequestException as exc:
                reason = f"unreachable ({exc.__class__.__name__})"
            else:
                if resp.status_code == 200:
                    return resp
                body = resp.text[:500]
                resp.close()
                if resp.status_code not in (502, 503, 504):
                    if resp.status_code < 500:
                        raise ValueError(f"model service rejected the request "
                                         f"({resp.status_code}): {body}")
                    raise RuntimeError(f"model service error ({resp.status_code}): {body}")
                reason = f"HTTP {resp.status_code}: {body}"
                try:
                    delay = max(delay, float(resp.headers.get("Retry-After", 0)))
                except ValueError:
                    pass
            if time.monotonic() + delay > deadline:
                raise ModelServiceUnavailable(f"model service at {self._base} {reason}")
            logger.info("Model service %s; retrying in %.0fs.", reason, delay)
            time.sleep(delay)
            delay = min(delay * 2, 10.0)

    @staticmethod
    def _stream(resp: requests.Response, stats: Optional[dict],
                cancel_event: Optional[threading.Event]) -> Iterator[str]:
        resp.encoding = "utf-8"  # SSE carries no charset; requests would guess latin-1
        try:
            for line in resp.iter_lines(decode_unicode=True):
                if cancel_event is not None and cancel_event.is_set():
                    return
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    return
                chunk = json.loads(data)
                if "error" in chunk:
                    raise RuntimeError(f"model service error: {chunk['error']}")
                if chunk.get("usage") and stats is not None:
                    stats.update(chunk["usage"])
                for choice in chunk.get("choices") or []:
                    text = (choice.get("delta") or {}).get("content")
                    if text:
                        yield text
            raise RuntimeError("model service stream ended without [DONE]")
        except requests.RequestException as exc:
            raise RuntimeError(f"model service stream interrupted: {exc}") from exc
        finally:
            resp.close()


def _attach_images(messages: list, images: list) -> None:
    """Put the frames on the first user turn, where the in-process path tags them."""
    target = next((m for m in messages if m.get("role") == "user"), None)
    if target is None:
        raise ValueError("images need a user message")
    content = target.get("content")
    parts = [{"type": "text", "text": content}] if isinstance(content, str) else list(content or [])
    parts.extend({"type": "image_url", "image_url": {"url": _tensor_to_data_url(t)}}
                 for t in images)
    target["content"] = parts


def _tensor_to_data_url(tensor) -> str:
    import numpy as np
    from PIL import Image

    array = np.asarray(getattr(tensor, "data", tensor), dtype=np.uint8)
    while array.ndim > 3:
        array = array[0]
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")  # lossless: board OCR reads it
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_remote_text_gen(settings: ServingSettings, text_gen_cfg=None) -> RemoteTextGen:
    supervisor = None
    if settings.mode == "managed":
        supervisor = ServingSupervisor(settings)
        supervisor.start()
    logger.info("text_gen served by %s model service at %s.", settings.mode, settings.endpoint)
    return RemoteTextGen(settings, text_gen_cfg, supervisor)
