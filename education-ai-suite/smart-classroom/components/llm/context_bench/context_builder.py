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


def corpus_text() -> str:
    """The built-in classroom-dialog haystack, as one text block.

    Public so the accuracy path can check a planted needle value is disjoint from it
    (tasks.draw_value) -- a value that happened to be a substring here would score a
    false retrieval hit.
    """
    return "\n".join(_CORPUS_LINES) + "\n"


# Back-compat alias for the original private name used within this module.
_corpus_text = corpus_text


def build_text_of_token_length(tokenizer: Tokenizer, target_tokens: int,
                               corpus: str | None = None) -> str:
    """Synthetic transcript text of ``target_tokens`` tokens, as close as the
    tokenizer's merge rules allow.

    Only ever "as close as": decoding a sliced id list can re-merge at the seams, so
    the exact figure is whatever the *rendered prompt* measures. `build_benchmark_prompt`
    owns that measurement and corrects against it, which is why nothing here re-encodes
    the result -- a second pass over 160K tokens per correction round buys no accuracy.

    `corpus` overrides the classroom dialog for the accuracy suite's aggregation tasks
    (`cwe`/`fwe`), whose context *is* the data being counted rather than a haystack to hide a
    fact in. Those word lists are built so that one repetition already carries the intended
    frequency ratio, which is what lets this function slice the repeated text to an arbitrary
    token count without reordering the answer.
    """
    if target_tokens <= 0:
        return ""

    corpus = corpus or _corpus_text()
    corpus_ids = tokenizer.encode(corpus, add_special_tokens=False)
    if not corpus_ids:
        raise ValueError("Tokenizer produced no tokens for the built-in corpus.")

    # Repeat the corpus until we have at least the requested number of tokens,
    # then slice the id list to the exact target and decode back to text.
    reps = (target_tokens // len(corpus_ids)) + 1
    ids = tokenizer.encode(corpus * reps, add_special_tokens=False)[:target_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def render_prompt(tokenizer: Tokenizer, transcript: str, suffix: str = _USER_SUFFIX,
                  system_prompt: str = SYSTEM_PROMPT, user_prefix: str = _USER_PREFIX,
                  **template_kwargs) -> tuple:
    """Render the benchmark chat prompt around ``transcript`` and measure it.

    Returns ``(prompt_text, prompt_tokens)``, where the count includes the chat template
    and special tokens -- the real prefill size the pipeline will see. Passing an empty
    transcript measures the scaffolding overhead alone.

    `suffix` is the task text after the transcript. It defaults to the summarization
    ``_USER_SUFFIX`` so the throughput path is byte-for-byte unchanged; the accuracy path
    passes the probe's own question instead (see `build_probe_prompt`).
    `system_prompt` / `user_prefix` replace the benchmark's own framing when a throughput
    task reproduces an application prompt instead (see `build_benchmark_prompt`).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prefix + transcript + suffix},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, **template_kwargs)
    return prompt, len(tokenizer.encode(prompt))


def build_benchmark_prompt(tokenizer: Tokenizer, target_tokens: int,
                           task: dict | None = None) -> tuple:
    """Build a rendered chat prompt whose token length is exactly ``target_tokens``.

    Sizes the transcript content so that, once the system prompt and chat-template
    scaffolding are added back, the whole rendered prompt lands on the target.
    Returns ``(prompt_text, prompt_tokens)``.

    `task` overrides the framing around the transcript -- a mapping with any of
    `system_prompt`, `user_prefix` and `suffix` (see `render_prompt`). None keeps the
    built-in two-sentence summary, byte for byte.

    ``target_tokens == 0`` means *no transcript*: the task's `standalone` framing (a
    `system_prompt` and a `user` message) is rendered as-is, like a dataset prompt, and its
    own length is returned. Only tasks that carry a `standalone` framing can do this; a
    summary of nothing is not a task.
    """
    if target_tokens == 0:
        standalone = (task or {}).get("standalone")
        if not standalone:
            raise ValueError(
                "context_tokens 0 (no transcript) needs a task with a standalone prompt; "
                f"{(task or {}).get('name', 'summary_2s')!r} summarizes the transcript"
            )
        messages = [
            {"role": "system", "content": standalone["system_prompt"]},
            {"role": "user", "content": standalone["user"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return prompt, len(tokenizer.encode(prompt))

    template_kwargs = dict(add_generation_prompt=True, enable_thinking=False)
    template_kwargs.update({
        key: value for key, value in (task or {}).items()
        if key in ("system_prompt", "user_prefix", "suffix")
    })

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


# The accuracy path's task framing, mirroring `_USER_SUFFIX`'s "---/TASK" shape so the only
# difference the model sees between a throughput prompt and a probe prompt is the task text.
_PROBE_SUFFIX_TEMPLATE = "\n---\nTASK\n{question}"

_PROBE_CORRECTION_TRIES = 4


def build_probe_prompt(tokenizer: Tokenizer, target_tokens: int, inserts: list,
                       question: str, filler: str | None = None,
                       tolerance: int = 0) -> tuple:
    """Build a rendered prompt of about ``target_tokens`` with each ``(depth, sentence)`` in
    ``inserts`` planted at its depth (0.0 = very top of the transcript, 1.0 = very bottom) and
    ``question`` as the task.

    Generalizes the single-needle case to RULER's multi-needle tasks: the inserts are sorted by
    depth and the filler between consecutive depths is sized from the gap, so a probe with
    eight needles spreads them across the context in the proportions the task asked for. The
    insert bytes are held fixed and all size correction goes into the filler, so the planted
    facts stay intact and at their depths.

    `tolerance` is how many tokens off `target_tokens` the result may land, and is the reason
    this builder does not share the throughput path's insistence on an exact count. Each insert
    adds two seams where the tokenizer can re-merge, so a `vt` probe has thirty-four of them;
    the correction below moves one number (the filler budget) against the sum of all that
    drift, and it can orbit the target by a token forever without landing on it. Demanding
    exactness there threw away a whole case -- a model load and forty minutes of completed
    probes -- over a prompt of 8,001 tokens where 8,000 was asked for.

    That trade is right for throughput and wrong here. Throughput *divides* by the token count,
    so a token matters; a retrieval verdict does not -- "did the model find the code planted at
    depth 0.25 of an 8,000-token context" is the same question at 8,001. The count actually
    used is returned, recorded per probe in the report, and used for any per-token figure, so
    nothing is reported against a number no forward pass saw. Callers wanting the old contract
    pass `tolerance=0` (the default) and still get an exact prompt or an exception.

    `filler` replaces the classroom-dialog haystack for the aggregation tasks, whose context is
    the word list being counted. `inserts` may be empty, which is the open-ended generation
    probe: a plain transcript with a real task at the end.

    Returns ``(prompt_text, prompt_tokens)``.
    """
    for depth, _sentence in inserts:
        if not 0.0 <= depth <= 1.0:
            raise ValueError(f"insert depth must be in [0.0, 1.0], got {depth}")

    ordered = sorted(inserts, key=lambda pair: pair[0])
    template_kwargs = dict(add_generation_prompt=True, enable_thinking=False)
    suffix = _PROBE_SUFFIX_TEMPLATE.format(question=question)

    # Scaffolding priced with this probe's own question, not the summarization task, so the
    # room left for the transcript accounts for this prompt's overhead.
    _, overhead = render_prompt(tokenizer, "", suffix=suffix, **template_kwargs)
    insert_tokens = sum(
        len(tokenizer.encode("\n" + sentence + "\n", add_special_tokens=False))
        for _depth, sentence in ordered
    )
    content_target = max(0, target_tokens - overhead)
    best = None

    for _ in range(_PROBE_CORRECTION_TRIES):
        # Split re-derived every try (never cached): a correction to content_target has to move
        # the filler, and every insert must stay put at its depth between the segments.
        filler_target = max(0, content_target - insert_tokens)
        transcript = _weave(tokenizer, ordered, filler_target, filler)
        prompt, prompt_tokens = render_prompt(
            tokenizer, transcript, suffix=suffix, **template_kwargs
        )
        # Inside tolerance is good enough to stop on, not just to fall back to: every extra try
        # re-encodes the whole prompt, which at 32K is most of what building a probe costs.
        if abs(prompt_tokens - target_tokens) <= tolerance:
            return prompt, prompt_tokens
        if best is None or abs(prompt_tokens - target_tokens) < abs(best[1] - target_tokens):
            best = (prompt, prompt_tokens)
        content_target = max(insert_tokens, content_target + target_tokens - prompt_tokens)

    raise ValueError(
        f"Could not build a {target_tokens}-token probe prompt with {len(ordered)} insert(s) "
        f"within {tolerance} token(s); closest was {best[1]}. If target is close to the "
        f"{overhead}-token overhead plus the {insert_tokens}-token insert(s), the context is "
        "too small to place them."
    )


def _weave(tokenizer: Tokenizer, ordered: list, filler_target: int,
           filler: str | None) -> str:
    """Filler and inserts interleaved, with ``filler_target`` tokens of filler in total.

    Segment sizes come from the *cumulative* split rather than from rounding each gap
    independently: rounding per gap loses up to half a token per insert, which on an
    eight-needle probe is enough drift to cost the correction loop a round for nothing. Taking
    differences of rounded cumulative positions makes the segments sum to `filler_target`
    exactly, and keeps them non-negative because the boundaries are sorted.
    """
    boundaries = [0.0] + [depth for depth, _ in ordered] + [1.0]
    cumulative = [round(filler_target * boundary) for boundary in boundaries]
    parts = []
    for index, (_depth, sentence) in enumerate(ordered):
        parts.append(
            build_text_of_token_length(
                tokenizer, cumulative[index + 1] - cumulative[index], filler
            )
        )
        parts.append("\n" + sentence + "\n")
    parts.append(
        build_text_of_token_length(tokenizer, cumulative[-1] - cumulative[-2], filler)
    )
    return "".join(parts)
