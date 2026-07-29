# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Synthetic classroom-transcript builder for the long-context benchmark.

Builds a chat prompt whose token count exactly matches a target size, or fails
explicitly when the tokenizer cannot converge, so trial_runner.py can push the real OpenVINO
pipeline up to a given context length and measure whether *this hardware* can
prefill and decode it without running out of memory.

Everything here is model-agnostic: it only needs a HuggingFace-style tokenizer
(``encode`` / ``decode`` / ``apply_chat_template``), so it can be unit-tested
with a stub tokenizer and no model download. This mirrors the approach in
refer/long_context/long_context_probe.py -- the content is irrelevant to the
validation, only the token *volume* matters, but realistic, varied dialog keeps
the tokenizer's merge behaviour representative of a real transcript instead of a
single repeated token.
"""

from __future__ import annotations

from typing import Protocol


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = True) -> list:
        ...

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        ...

    def apply_chat_template(self, messages, tokenize: bool = False, **kwargs) -> str:
        ...


SYSTEM_PROMPT = (
    "You are a teaching assistant validating a classroom transcript. Follow the "
    "task after the transcript and answer with useful natural language."
)

_USER_PREFIX = "CLASSROOM TRANSCRIPT\n---\n"
_USER_SUFFIX = (
    "\n---\nTASK\nSummarize the lesson's main topic and one key relationship "
    "explained by the teacher in two concise sentences."
)

# A small pool of generic classroom-dialog lines. Content is irrelevant to the
# validation -- only the token *volume* matters -- but realistic, varied text
# keeps the tokenizer's merge behaviour representative of a real transcript
# rather than a single repeated token.
_CORPUS_LINES = [
    "TEACHER: Let's begin today's lesson by reviewing what we covered last week about energy transfer.",
    "STUDENT_01: Could you explain again why kinetic energy depends on the square of the velocity?",
    "TEACHER: Good question. When we double the speed, the energy increases by a factor of four.",
    "STUDENT_02: So a car moving twice as fast needs four times the braking distance?",
    "TEACHER: Precisely, and that is why speed limits matter so much for road safety.",
    "STUDENT_03: What happens to that energy when the car finally stops?",
    "TEACHER: Most of it is converted into heat through friction in the brakes and the tyres.",
    "STUDENT_01: Does that mean energy is never actually lost, only transformed?",
    "TEACHER: Correct, that is the principle of conservation of energy in a closed system.",
    "STUDENT_04: Can you give an everyday example where potential energy becomes kinetic energy?",
    "TEACHER: Think of a roller coaster climbing to the top of a hill and then racing down.",
    "STUDENT_02: At the very top it has the most potential energy and almost no motion.",
    "TEACHER: Exactly, and at the bottom that potential energy has become kinetic energy.",
    "STUDENT_03: How do engineers account for friction and air resistance in real designs?",
    "TEACHER: They add safety margins and measure the losses experimentally in testing.",
    "STUDENT_04: Is there a formula that ties all of these ideas together for the exam?",
    "TEACHER: Yes, we will derive the work-energy theorem step by step on the board now.",
    "STUDENT_01: Should we memorise the derivation or just the final expression?",
    "TEACHER: Understand the derivation; the final expression will follow naturally from it.",
    "STUDENT_02: Thank you, that makes the relationship between force and distance much clearer.",
]


def _corpus_text() -> str:
    return "\n".join(_CORPUS_LINES) + "\n"


def build_text_of_token_length(tokenizer: Tokenizer, target_tokens: int) -> str:
    """Synthetic transcript text of ``target_tokens`` tokens, as close as the
    tokenizer's merge rules allow.

    Only ever "as close as": decoding a sliced id list can re-merge at the seams, so
    the exact figure is whatever the *rendered prompt* measures. `build_benchmark_prompt`
    owns that measurement and corrects against it, which is why nothing here re-encodes
    the result -- a second pass over 160K tokens per correction round buys no accuracy.
    """
    if target_tokens <= 0:
        return ""

    corpus = _corpus_text()
    corpus_ids = tokenizer.encode(corpus, add_special_tokens=False)
    if not corpus_ids:
        raise ValueError("Tokenizer produced no tokens for the built-in corpus.")

    # Repeat the corpus until we have at least the requested number of tokens,
    # then slice the id list to the exact target and decode back to text.
    reps = (target_tokens // len(corpus_ids)) + 1
    ids = tokenizer.encode(corpus * reps, add_special_tokens=False)[:target_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def render_prompt(tokenizer: Tokenizer, transcript: str, **template_kwargs) -> tuple:
    """Render the benchmark chat prompt around ``transcript`` and measure it.

    Returns ``(prompt_text, prompt_tokens)``, where the count includes the chat template
    and special tokens -- the real prefill size the pipeline will see. Passing an empty
    transcript measures the scaffolding overhead alone.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _USER_PREFIX + transcript + _USER_SUFFIX},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, **template_kwargs)
    return prompt, len(tokenizer.encode(prompt))


def build_benchmark_prompt(tokenizer: Tokenizer, target_tokens: int) -> tuple:
    """Build a rendered chat prompt whose token length is exactly ``target_tokens``.

    Sizes the transcript content so that, once the system prompt and chat-template
    scaffolding are added back, the whole rendered prompt lands on the target.
    Returns ``(prompt_text, prompt_tokens)``.
    """
    template_kwargs = dict(add_generation_prompt=True, enable_thinking=False)

    # An empty transcript prices the template + system prompt + task instructions, which
    # is the room the content cannot have.
    _, overhead = render_prompt(tokenizer, "", **template_kwargs)
    content_target = max(0, target_tokens - overhead)

    # Token merges at the transcript/template boundaries can make the first
    # estimate differ by a token or two. Rebuild against the measured delta so
    # the value sent to the pipeline is the configured context size, not merely
    # a nearby value. Four corrections are ample for deterministic tokenizers;
    # fail explicitly if a tokenizer cannot converge instead of misreporting it.
    for _ in range(5):
        prompt, prompt_tokens = render_prompt(
            tokenizer, build_text_of_token_length(tokenizer, content_target), **template_kwargs
        )
        if prompt_tokens == target_tokens:
            return prompt, prompt_tokens
        content_target = max(0, content_target + target_tokens - prompt_tokens)

    raise ValueError(
        f"Could not build an exact {target_tokens}-token prompt; last count was {prompt_tokens}"
    )
