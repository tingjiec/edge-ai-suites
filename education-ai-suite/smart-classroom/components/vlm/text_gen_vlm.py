# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import gc
import json
import logging
import re
import sys
import threading
import time
from pathlib import Path
from typing import Iterator, Optional, Union

import openvino_genai as ov_genai
from transformers import AutoTokenizer

from utils.chat_prompt import render_chat_prompt
from utils.markdown_cleaner import StreamThinkFilter, strip_think_tokens
from utils.model_paths import openvino_model_dir
from utils.ov_genai_util import YieldingTextStreamer

logger = logging.getLogger(__name__)

_SC_ROOT = Path(__file__).resolve().parents[2]
_CONTENT_SEARCH_DIR = _SC_ROOT / "content_search"

_DEFAULT_MAX_NEW_TOKENS = 5120

# Block size the drafter proposes per verification step. On a PTL Arc B390 with
# Qwen3.6-35B-A3B INT4, 5 matched 7 on code (~2.3x) and cost less on prose.
_DEFAULT_NUM_ASSISTANT_TOKENS = 5

# How long an abandoned stream waits for the native generation to honour CANCEL
# before the runner slot is handed on regardless.
_CANCEL_JOIN_TIMEOUT_S = 30

# OpenVINO model cards state the oldest runtime that can load the IR, e.g.
# "OpenVINO version 2026.2.0 and higher" / "OpenVINO 2026.4.0 (nightly) and higher".
_MIN_RUNTIME_RE = re.compile(
    r"OpenVINO\S*\s+(?:version\s+)?(\d{4})\.(\d+)(?:\.(\d+))?[^\n]*?\band higher",
    re.IGNORECASE,
)
_VERSION_RE = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")

# First openvino-genai with the DFlash strategy. Older runtimes do not reject a
# DFlash drafter cleanly -- they crash the process -- so this is checked first.
_DFLASH_MIN_GENAI = (2026, 4, 0)


def _filtered(tokens: Iterator[str]) -> Iterator[str]:
    """Drop reasoning from a token stream, suppressing now-empty chunks."""
    think_filter = StreamThinkFilter()
    for token in tokens:
        clean = think_filter.filter(token)
        if clean:
            yield clean


def _version_tuple(match) -> tuple:
    return tuple(int(part or 0) for part in match.groups())


def runtime_incompatibility(model_dir: Path) -> Optional[str]:
    """Return why the installed OpenVINO cannot load ``model_dir``, or None.

    A too-old runtime does not fail cleanly on a newer architecture -- it can
    crash the process inside ``VLMPipeline``. The model card's compatibility
    line is checked first so the operator gets a message instead of a segfault.
    """
    try:
        card = (Path(model_dir) / "README.md").read_text(encoding="utf-8")
    except OSError:
        return None
    need = _MIN_RUNTIME_RE.search(card)
    if need is None:
        return None
    import openvino

    have = _VERSION_RE.match(str(openvino.__version__))
    if have is None or _version_tuple(have) >= _version_tuple(need):
        return None
    return (
        f"{model_dir} requires OpenVINO >= {'.'.join(map(str, _version_tuple(need)))} "
        f"(per its model card); installed {openvino.__version__}"
    )


def is_dflash_drafter(draft_dir: Path) -> bool:
    try:
        cfg = json.loads((Path(draft_dir) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return "dflash_config" in cfg or "DFlashDraftModel" in (cfg.get("architectures") or [])


def dflash_incompatibility() -> Optional[str]:
    """Return why this runtime cannot run a DFlash drafter, or None."""
    have = _VERSION_RE.match(str(getattr(ov_genai, "__version__", "")))
    if have is not None and _version_tuple(have) >= _DFLASH_MIN_GENAI:
        return None
    need = ".".join(map(str, _DFLASH_MIN_GENAI))
    return (f"DFlash drafters need openvino-genai >= {need}; installed "
            f"{getattr(ov_genai, '__version__', 'unknown')}")


def _has_linear_attention(model_dir: Path) -> bool:
    try:
        cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    cfg = cfg.get("text_config", cfg)
    return "linear_attention" in (cfg.get("layer_types") or [])


def speculative_mode(spec) -> str:
    """Normalise ``models.text_gen.speculative.mode`` to off / auto / on."""
    mode = getattr(spec, "mode", "off") if spec is not None else "off"
    # YAML 1.1 reads a bare off/on as a boolean.
    if mode is True:
        return "on"
    if mode is False or mode is None:
        return "off"
    mode = str(mode).strip().lower()
    if mode not in ("off", "auto", "on"):
        raise ValueError(
            f"models.text_gen.speculative.mode must be off, auto or on; got {mode!r}"
        )
    return mode


def _record_usage(stats: Optional[dict], result) -> None:
    if stats is None or result is None:
        return
    try:
        metrics = result.perf_metrics
        stats["prompt_tokens"] = int(metrics.get_num_input_tokens())
        stats["completion_tokens"] = int(metrics.get_num_generated_tokens())
    except Exception:  # noqa: BLE001 - usage is best-effort metadata
        logger.debug("perf_metrics unavailable", exc_info=True)


def _import_convert_helpers():
    if str(_CONTENT_SEARCH_DIR) not in sys.path:
        sys.path.append(str(_CONTENT_SEARCH_DIR))
    from components.vlm.vlm_openvino_serving.utils.utils import (  # noqa: E402
        convert_model,
        is_model_ready,
    )

    return convert_model, is_model_ready


class VLMTextGen:
    """Warm ``ov_genai.VLMPipeline`` fronting the ``text_gen`` capability."""

    def __init__(self) -> None:
        self._pipe = None
        self.tokenizer = None
        self._model_name: Optional[str] = None
        self._device: Optional[str] = None
        self._weight_format: Optional[str] = None
        self._max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS
        self._speculative = None
        self._num_assistant_tokens: int = 0
        # "off", "active" or "unavailable: <reason>"; reported by /health.
        self.speculative_status: str = "off"
        self.chat_template: str = ""
        self._load_config()
        self._load()

    @property
    def device(self) -> Optional[str]:
        return self._device

    @property
    def model_name(self) -> Optional[str]:
        return self._model_name

    @property
    def weight_format(self) -> Optional[str]:
        return self._weight_format

    @property
    def tool_call_format(self) -> str:
        """``xml`` for the qwen3coder ``<function=...>`` form, else Hermes ``json``."""
        return "xml" if "<function=" in self.chat_template else "json"

    def _load_config(self) -> None:
        from utils.config_loader import config

        text_gen = getattr(config.models, "text_gen", None)
        if text_gen is None:
            raise ValueError(
                "models.text_gen is not configured; the warm VLM cannot start"
            )
        self._model_name = str(text_gen.vlm_name)
        self._device = str(text_gen.device).upper()
        self._weight_format = str(text_gen.weight_format).lower()
        self._max_new_tokens = int(
            getattr(text_gen, "max_new_tokens", _DEFAULT_MAX_NEW_TOKENS)
        )
        self._speculative = getattr(text_gen, "speculative", None)

    def _model_dir(self) -> Path:
        """Return the shared IR directory ``models/openvino/<name>/<weight>``."""
        return openvino_model_dir(self._model_name, self._weight_format)

    @staticmethod
    def _ov_config(device: str) -> dict:
        """Runtime config for the pipeline; large allocations help on GPU."""
        if device.startswith("GPU"):
            return {"GPU_ENABLE_LARGE_ALLOCATIONS": "YES"}
        return {}

    def _load(self) -> None:
        model_dir = self._model_dir()
        model_dir.mkdir(parents=True, exist_ok=True)

        convert_model, is_model_ready = _import_convert_helpers()
        if not is_model_ready(model_dir, require_detokenizer=True):
            logger.info(
                "Converting VLM %s -> OpenVINO IR (%s) at %s",
                self._model_name,
                self._weight_format,
                model_dir,
            )
            convert_model(
                self._model_name,
                str(model_dir),
                model_type="vlm",
                weight_format=self._weight_format,
            )

        incompatible = runtime_incompatibility(model_dir)
        if incompatible:
            raise RuntimeError(
                f"{incompatible}. Upgrade openvino / openvino-genai / "
                "openvino-tokenizers, or pick another models.text_gen.vlm_name."
            )

        logger.info(
            "Loading warm VLMPipeline: model=%s device=%s weight=%s",
            self._model_name,
            self._device,
            self._weight_format,
        )
        self._pipe = self._build_pipeline(model_dir)
        try:
            # Think tags are ordinary vocabulary entries in every Qwen3 family,
            # so the streamer decodes them intact. Marking them special would
            # make skip_special_tokens drop the tags while keeping the reasoning
            # text between them, leaving nothing for StreamThinkFilter to match.
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir), extra_special_tokens={}
            )
        except Exception as exc:
            logger.warning(
                "transformers AutoTokenizer failed (%s); falling back to the "
                "OpenVINO pipeline tokenizer.", exc
            )
            self.tokenizer = self._pipe.get_tokenizer()
        self.chat_template = str(getattr(self.tokenizer, "chat_template", "") or "")
        self._warm_up()
        logger.info("Warm VLM ready.")

    def _build_pipeline(self, model_dir: Path):
        """Build the pipeline, attaching the speculative drafter when configured.

        ``auto`` falls back to plain decoding when the drafter or the runtime
        cannot do it; ``on`` refuses to start rather than run slower than asked.
        """
        props = self._ov_config(self._device)
        mode = speculative_mode(self._speculative)
        if mode != "off":
            try:
                draft = self._draft_model()
                self._num_assistant_tokens = int(
                    getattr(self._speculative, "num_assistant_tokens", None)
                    or _DEFAULT_NUM_ASSISTANT_TOKENS
                )
                pipe = ov_genai.VLMPipeline(
                    str(model_dir), device=self._device, draft_model=draft,
                    scheduler_config=self._speculative_scheduler(model_dir), **props
                )
                self.speculative_status = "active"
                logger.info(
                    "Speculative decoding active: draft=%s, %d tokens per step.",
                    getattr(self._speculative, "draft_model", None),
                    self._num_assistant_tokens,
                )
                return pipe
            except Exception as exc:  # noqa: BLE001 - runtime/drafter capability probe
                if "hidden-state" in str(exc):
                    exc = RuntimeError(
                        f"{exc} -- the target IR was exported without the hidden-"
                        "state outputs DFlash conditions on; re-export it with a "
                        "current optimum-intel or re-download the pre-converted IR"
                    )
                if mode == "on":
                    raise RuntimeError(
                        f"speculative.mode is 'on' but speculative decoding could "
                        f"not be enabled: {exc}"
                    ) from exc
                self.speculative_status = f"unavailable: {exc}"
                logger.warning(
                    "Speculative decoding unavailable (%s); using plain decoding.", exc
                )
        return ov_genai.VLMPipeline(str(model_dir), device=self._device, **props)

    def _draft_model(self):
        """Resolve the drafter IR: a directory, or ``models/openvino/<name>/<weight>``."""
        spec = self._speculative
        name = getattr(spec, "draft_model", None)
        if not name:
            raise ValueError("models.text_gen.speculative.draft_model is not set")
        path = Path(str(name))
        if not path.is_dir():
            weight = getattr(spec, "weight_format", None) or self._weight_format
            path = openvino_model_dir(str(name), str(weight))
        if not any(path.glob("*.xml")):
            raise FileNotFoundError(
                f"draft model IR not found in {path}; export the drafter to "
                "OpenVINO IR there first"
            )
        if is_dflash_drafter(path):
            incompatible = dflash_incompatibility()
            if incompatible:
                raise RuntimeError(incompatible)
        device = str(getattr(spec, "device", None) or self._device).upper()
        return ov_genai.draft_model(str(path), device, **self._ov_config(device))

    def _speculative_scheduler(self, model_dir: Path):
        """Scheduler the speculative (continuous-batching) pipeline needs."""
        scheduler = ov_genai.SchedulerConfig()
        # DFlash does not support prefix caching yet.
        scheduler.enable_prefix_caching = False
        if _has_linear_attention(model_dir):
            # Hybrid models (Qwen3.5+) verify a draft block in borrowed
            # recurrent-state rows: one per live sequence plus 1 + block size.
            from utils.config_loader import config

            live = int(getattr(config.models.text_gen, "concurrency", 1) or 1)
            scheduler.num_linear_attention_blocks = live + 1 + self._num_assistant_tokens
        return scheduler

    def _apply_speculative(self, config, has_images: bool) -> None:
        if self.speculative_status != "active":
            return
        if config.do_sample:
            # The DFlash pipeline decodes greedily only (openvino-genai 2026.4);
            # a sampled request is served greedy rather than refused.
            if not getattr(self, "_warned_sampling", False):
                logger.warning("Speculative decoding is greedy-only; temperature/top_p "
                               "requests are decoded greedily while it is active.")
                self._warned_sampling = True
            config.do_sample = False
        # Text-only: image requests decode plainly on the same pipeline.
        if not has_images:
            config.num_assistant_tokens = self._num_assistant_tokens

    def _warm_up(self) -> None:
        """Generate one token so compile/allocation failures surface at load
        time rather than on the first user request."""
        config = ov_genai.GenerationConfig(max_new_tokens=1, do_sample=False)
        config.apply_chat_template = False
        self._apply_speculative(config, has_images=False)
        self._pipe.generate("Hi", generation_config=config)

    def release(self) -> None:
        """Release the resident pipeline and reclaim device/host memory."""
        try:
            self._pipe = None
            self.tokenizer = None
            gc.collect()
            logger.info("Warm VLM released and memory reclaimed.")
        except Exception:  # noqa: BLE001 - shutdown best-effort
            logger.warning("Failed to fully release warm VLM", exc_info=True)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
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
        tools: Optional[list] = None,
        template_kwargs: Optional[dict] = None,
        prefill: Optional[str] = None,
        sampling: Optional[dict] = None,
        stats: Optional[dict] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Union[Iterator[str], str]:
        """Generate from a chat history using the warm pipeline.

        Mirrors ``TextGen.generate``: streaming yields decoded token chunks,
        non-streaming returns the full string.

        Pass ``messages`` (a chat history) or ``prompt`` (a single user turn),
        never both. Either way this method owns the templating, so callers must
        not run ``apply_chat_template`` themselves -- doing so would wrap the
        rendered string in a second user turn, burying the ``<think></think>``
        prefill and bringing reasoning back.

        ``images`` (a list of ``ov.Tensor`` frames, already decoded by the
        caller) enables the multimodal path used by content-search video
        summarization; when omitted the call is text-only.
        ``enable_thinking=False`` suppresses Qwen3 thinking and strips any
        reasoning that slips through; ``None`` keeps the model default.
        ``json_schema`` (a JSON-schema string) constrains decoding to output
        matching that schema.

        The remaining options serve the OpenAI-compatible API
        (``model_serving.openai_chat``): ``tools`` and ``template_kwargs`` go to
        the chat template, ``prefill`` is appended after the generation prompt
        (to force a tool call), ``sampling`` sets extra ``GenerationConfig``
        fields, ``stats`` receives token usage, and setting ``cancel_event``
        stops a running stream.
        """
        if self._pipe is None:
            raise RuntimeError("VLM pipeline is not loaded")
        if (messages is None) == (prompt is None):
            raise ValueError("Provide exactly one of prompt or messages.")
        if messages is None:
            if not prompt.strip():
                raise ValueError("Invalid prompt provided.")
            messages = [{"role": "user", "content": prompt}]
        elif not messages:
            raise ValueError("Invalid messages provided.")

        config = self._generation_config(
            max_new_tokens, temperature, json_schema, sampling
        )
        self._apply_speculative(config, bool(images))
        prompt = self._render(
            messages, config, enable_thinking, bool(images), tools, template_kwargs
        )
        if prefill:
            prompt += prefill
        if stats is not None:
            # Lets the API tell reasoning from answer when the template itself
            # opened the <think> block.
            stats["thinking_open"] = prompt.rstrip().endswith("<think>")
            stats["max_new_tokens"] = int(config.max_new_tokens)

        if stream:
            tokens = self._generate_stream(prompt, config, images, stats, cancel_event)
            return _filtered(tokens) if enable_thinking is False else tokens
        if images:
            result = self._pipe.generate(prompt, images=images, generation_config=config)
        else:
            result = self._pipe.generate(prompt, generation_config=config)
        _record_usage(stats, result)
        raw = str(result)
        return strip_think_tokens(raw) if enable_thinking is False else raw

    def _render(
        self,
        messages: list,
        config: "ov_genai.GenerationConfig",
        enable_thinking: Optional[bool],
        has_images: bool,
        tools: Optional[list] = None,
        template_kwargs: Optional[dict] = None,
    ) -> str:
        """Render ``messages`` here rather than inside the pipeline.

        Keeping templating on this side means one engine and one set of rules
        for every caller. The pipeline only takes over for a model that has no
        chat template at all, where rendering is impossible.
        """
        try:
            prompt = render_chat_prompt(
                self.tokenizer,
                messages,
                self._model_name,
                enable_thinking,
                has_images,
                tools,
                template_kwargs,
            )
            config.apply_chat_template = False
            return prompt
        except Exception as exc:  # noqa: BLE001 - model without a chat template
            if tools or len(messages) > 1:
                # The pipeline fallback only sees the last turn; dropping tools
                # or history silently would answer a different question.
                raise ValueError(f"Chat template rendering failed: {exc}") from exc
            logger.warning(
                "Chat template rendering failed (%s); letting the pipeline "
                "template the last user turn.", exc
            )
            config.apply_chat_template = True
            return str(messages[-1].get("content", ""))

    def _generation_config(
        self,
        max_new_tokens: Optional[int],
        temperature: Optional[float],
        json_schema: Optional[str] = None,
        sampling: Optional[dict] = None,
    ) -> "ov_genai.GenerationConfig":
        max_tokens = (
            int(max_new_tokens) if max_new_tokens is not None else self._max_new_tokens
        )
        kwargs = {"max_new_tokens": max_tokens, "do_sample": False}
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
            kwargs["do_sample"] = float(temperature) > 0.0
        config = ov_genai.GenerationConfig(**kwargs)
        for key, value in (sampling or {}).items():
            if value is not None:
                setattr(config, key, set(value) if key == "stop_strings" else value)
        if json_schema:
            try:
                config.structured_output_config = ov_genai.StructuredOutputConfig(
                    json_schema=json_schema
                )
            except Exception as exc:  # noqa: BLE001 - runtime without a grammar backend
                logger.warning(
                    "Structured output unavailable (%s); generating unconstrained.", exc
                )
        return config

    def _generate_stream(
        self,
        prompt: str,
        config: "ov_genai.GenerationConfig",
        images: Optional[list] = None,
        stats: Optional[dict] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Iterator[str]:
        """Run generation on a worker thread, yielding tokens as they arrive.

        A memory/runtime error raised during generation is re-raised in the
        consuming thread once the queue drains, so the ``CapabilityRunner`` can
        surface it as an ``OomError`` while keeping the capability resident.

        The pipeline calls ``streamer.end()`` itself before ``generate`` returns,
        so the iterator also joins the worker before finishing: only then is the
        pipeline idle for the next request the runner admits, and usage stats
        recorded. A consumer that stops early cancels the generation first.
        """
        streamer = YieldingTextStreamer(self.tokenizer, cancel_event=cancel_event)
        error: list[Exception] = []

        def run_generation() -> None:
            try:
                streamer.generation_start_time = time.perf_counter()
                if images:
                    result = self._pipe.generate(
                        prompt,
                        images=images,
                        generation_config=config,
                        streamer=streamer,
                    )
                else:
                    result = self._pipe.generate(
                        prompt, generation_config=config, streamer=streamer
                    )
                _record_usage(stats, result)
            except Exception as exc:  # noqa: BLE001 - re-raised in the consumer
                logger.error("VLM text_gen streaming failed: %s", exc)
                error.append(exc)
            finally:
                streamer.end()

        worker = threading.Thread(target=run_generation, daemon=True)
        worker.start()

        def _iterator() -> Iterator[str]:
            finished = False
            try:
                for token in streamer:
                    yield token
                finished = True
            finally:
                if not finished:
                    streamer.cancel()
                worker.join(timeout=_CANCEL_JOIN_TIMEOUT_S)
                if worker.is_alive():
                    logger.error("VLM generation did not stop within %ss.",
                                 _CANCEL_JOIN_TIMEOUT_S)
            if error:
                raise error[0]

        return _iterator()
