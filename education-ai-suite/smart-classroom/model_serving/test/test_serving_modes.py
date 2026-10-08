# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Serving settings, the remote client, the supervisor and the engine guards."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from model_serving import client as client_mod
from model_serving.client import ModelServiceUnavailable, RemoteTextGen, ServingSupervisor
from model_serving.settings import EXIT_LOAD_ERROR, ServingSettings, serving_settings


# ---------------------------------------------------------------- settings

def test_defaults_keep_the_model_in_process():
    s = serving_settings(SimpleNamespace(vlm_name="x"))
    assert s.mode == "inprocess" and not s.remote
    assert s.endpoint == "http://127.0.0.1:8010"
    assert s.default_thinking is False


def test_external_endpoint_and_null_thinking():
    cfg = SimpleNamespace(enable_thinking=None,
                          serving=SimpleNamespace(mode="EXTERNAL", endpoint="http://h:9/"))
    s = serving_settings(cfg)
    assert s.remote and s.endpoint == "http://h:9" and s.default_thinking is None


def test_an_unknown_mode_is_a_config_error():
    with pytest.raises(ValueError):
        serving_settings(SimpleNamespace(serving=SimpleNamespace(mode="cloud")))


# ---------------------------------------------------------------- remote client

class _Resp:
    def __init__(self, status, lines=(), headers=None, text=""):
        self.status_code, self._lines = status, list(lines)
        self.headers, self.text, self.encoding = headers or {}, text, None

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)

    def close(self):
        pass


def _sse(*texts, usage=None):
    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': t}}]})}" for t in texts]
    if usage:
        lines.append(f"data: {json.dumps({'choices': [], 'usage': usage})}")
    return lines + ["data: [DONE]"]


def _remote(responses, **settings):
    remote = RemoteTextGen(ServingSettings(mode="external", **settings),
                           SimpleNamespace(vlm_name="Qwen/X", device="gpu", weight_format="int4"))
    sent = []

    def post(url, json=None, **kwargs):
        sent.append(json)
        return responses.pop(0)

    remote._session.post = post
    return remote, sent


def test_remote_streams_content_and_usage(monkeypatch):
    remote, sent = _remote([_Resp(200, _sse("Hel", "lo", usage={"completion_tokens": 2}))])
    stats = {}
    assert "".join(remote.generate("hi", enable_thinking=False, stats=stats)) == "Hello"
    assert stats["completion_tokens"] == 2
    assert sent[0]["enable_thinking"] is False and sent[0]["stream"] is True


def test_remote_non_streaming_returns_a_string():
    remote, sent = _remote([_Resp(200, _sse("ok"))])
    assert remote.generate(messages=[{"role": "user", "content": "x"}], stream=False,
                           json_schema='{"type": "object"}') == "ok"
    assert sent[0]["response_format"]["json_schema"]["schema"] == {"type": "object"}


def test_remote_waits_out_a_loading_server(monkeypatch):
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: None)
    remote, _ = _remote([_Resp(503, headers={"Retry-After": "1"}), _Resp(200, _sse("up"))])
    assert remote.generate("x", stream=False) == "up"


def test_remote_gives_up_with_a_503_class_error(monkeypatch):
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: None)
    remote, _ = _remote([_Resp(503)] * 3, ready_wait_s=0.5)
    with pytest.raises(ModelServiceUnavailable):
        remote.generate("x", stream=False)


def test_remote_passes_client_errors_through_as_value_errors():
    remote, _ = _remote([_Resp(400, text="bad")])
    with pytest.raises(ValueError):
        remote.generate("x", stream=False)


def test_remote_stream_error_event_raises():
    lines = ["data: " + json.dumps({"error": {"message": "boom"}})]
    remote, _ = _remote([_Resp(200, lines)])
    with pytest.raises(RuntimeError):
        list(remote.generate("x"))


def test_images_ride_on_the_first_user_turn_as_png_data_urls():
    import numpy as np

    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "describe"}]
    client_mod._attach_images(messages, [np.zeros((1, 4, 4, 3), dtype=np.uint8)])
    parts = messages[1]["content"]
    assert parts[0] == {"type": "text", "text": "describe"}
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


# ---------------------------------------------------------------- supervisor

def _supervisor(monkeypatch, exit_codes, max_restarts=2):
    monkeypatch.setattr(client_mod, "_probe", lambda *a, **k: False)
    monkeypatch.setattr(client_mod.random, "uniform", lambda a, b: 0.0)
    sup = ServingSupervisor(ServingSettings(mode="managed", max_restarts=max_restarts))
    codes = list(exit_codes)
    spawned = []

    def spawn():
        code = codes.pop(0) if codes else 1
        spawned.append(code)
        return subprocess.Popen([sys.executable, "-c", f"raise SystemExit({code})"])

    sup._spawn = spawn
    return sup, spawned


def test_supervisor_restarts_a_crash_then_respects_the_budget(monkeypatch):
    sup, spawned = _supervisor(monkeypatch, [139, 139, 139, 139])
    sup._run()
    assert len(spawned) == 3  # first run + max_restarts
    assert sup.failed and "giving up" in sup.last_error


def test_supervisor_does_not_retry_a_load_error(monkeypatch):
    sup, spawned = _supervisor(monkeypatch, [EXIT_LOAD_ERROR])
    sup._run()
    assert spawned == [EXIT_LOAD_ERROR] and sup.failed


def test_supervisor_adopts_a_server_that_is_already_up(monkeypatch):
    monkeypatch.setattr(client_mod, "_probe", lambda *a, **k: True)
    sup = ServingSupervisor(ServingSettings(mode="managed"))
    sup.start()
    assert sup._thread is None  # nothing spawned, nothing to stop later


# ---------------------------------------------------------------- engine guards

def test_runtime_guard_reads_the_model_card(tmp_path, monkeypatch):
    from components.vlm import text_gen_vlm
    import openvino

    (tmp_path / "README.md").write_text(
        "## Compatibility\n- OpenVINO 2026.4.0 (nightly) and higher\n", encoding="utf-8")
    monkeypatch.setattr(openvino, "__version__", "2026.3.0-1-abc")
    assert "2026.4.0" in text_gen_vlm.runtime_incompatibility(tmp_path)
    monkeypatch.setattr(openvino, "__version__", "2026.4.1-22982-e213")
    assert text_gen_vlm.runtime_incompatibility(tmp_path) is None
    assert text_gen_vlm.runtime_incompatibility(tmp_path / "missing") is None


@pytest.mark.parametrize("raw, mode", [
    (None, "off"), (False, "off"), (True, "on"), ("auto", "auto"), ("ON", "on"),
])
def test_speculative_mode_survives_yaml_booleans(raw, mode):
    from components.vlm.text_gen_vlm import speculative_mode

    assert speculative_mode(SimpleNamespace(mode=raw)) == mode


def test_speculative_mode_rejects_typos():
    from components.vlm.text_gen_vlm import speculative_mode

    with pytest.raises(ValueError):
        speculative_mode(SimpleNamespace(mode="fast"))


def test_hybrid_attention_models_size_the_kv_cache_from_full_attention_layers(tmp_path):
    from utils import text_chunker

    config = {"text_config": {"num_hidden_layers": 8, "num_attention_heads": 4,
                              "num_key_value_heads": 2, "head_dim": 64,
                              "layer_types": ["linear_attention"] * 3 + ["full_attention"]
                              + ["linear_attention"] * 3 + ["full_attention"]}}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    assert text_chunker._kv_geometry(Path(tmp_path)) == (2, 2, 64)


# ---------------------------------------------------------------- DFlash gating

def test_a_dflash_drafter_is_recognised_by_its_config(tmp_path):
    from components.vlm.text_gen_vlm import is_dflash_drafter

    (tmp_path / "config.json").write_text(json.dumps(
        {"architectures": ["DFlashDraftModel"], "dflash_config": {"block_size": 16}}))
    assert is_dflash_drafter(tmp_path)
    assert not is_dflash_drafter(tmp_path / "missing")


def test_dflash_needs_genai_2026_4(monkeypatch):
    from components.vlm import text_gen_vlm

    monkeypatch.setattr(text_gen_vlm.ov_genai, "__version__", "2026.3.0.0-3277", raising=False)
    assert "2026.4.0" in text_gen_vlm.dflash_incompatibility()
    monkeypatch.setattr(text_gen_vlm.ov_genai, "__version__", "2026.4.1.0-3408", raising=False)
    assert text_gen_vlm.dflash_incompatibility() is None


@pytest.mark.parametrize("do_sample, images, expected", [
    (False, False, 5), (True, False, 5), (False, True, 0),
])
def test_dflash_requests_are_greedy_and_images_are_not_drafted(do_sample, images, expected):
    """The DFlash pipeline rejects sampling and drafts text only (genai 2026.4)."""
    from components.vlm.text_gen_vlm import VLMTextGen

    engine = VLMTextGen.__new__(VLMTextGen)
    engine.speculative_status, engine._num_assistant_tokens = "active", 5
    config = SimpleNamespace(do_sample=do_sample, num_assistant_tokens=0)
    engine._apply_speculative(config, has_images=images)
    assert config.num_assistant_tokens == expected
    assert config.do_sample is False


def test_plain_pipelines_keep_sampling():
    from components.vlm.text_gen_vlm import VLMTextGen

    engine = VLMTextGen.__new__(VLMTextGen)
    engine.speculative_status = "off"
    config = SimpleNamespace(do_sample=True, num_assistant_tokens=0)
    engine._apply_speculative(config, has_images=False)
    assert config.do_sample is True and config.num_assistant_tokens == 0
