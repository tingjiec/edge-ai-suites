# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""The accuracy task suite: what gets planted in the long context, what is asked about it,
and what counts as the right answer.

Two families, because "is this configuration still accurate" is two different questions.

**Retrieval** (`SUITE_RETRIEVAL`) follows RULER (Hsieh et al., COLM 2024), which generalizes
Needle-in-a-Haystack into three behaviours a long context can fail at independently:

  * *retrieval* -- `niah_single` (one fact), `niah_multikey` (one fact among decoys),
    `niah_multivalue` (every value of one key), `niah_multiquery` (one value of every key)
  * *multi-hop tracing* -- `vt`, chains of variable assignments that must be followed to the
    end to know which names hold a value
  * *aggregation* -- `cwe` / `fwe`, which need the whole context counted rather than one span
    found, and are RULER's proxies for summarization

plus `qa`, a natural-language question over a planted paragraph, where the answer has to be
understood rather than copied. A profile can hold `niah_single` at 100% and collapse on `vt`
or `cwe`, which is exactly why the single-needle result on its own was not enough.

**Generation** (`SUITE_GENERATION`) is the fidelity family, and it is a different measurement:
the model does the *real* task (summarize the transcript, explain a relationship, list the
recorded facts) at a realistic output length, and its answer is compared with the answer the
**baseline profile** gave to the same prompt -- WWB's framing, and the one that catches a
quantization or speculative-decoding change that degrades prose without ever touching a
retrievable fact. `fact_sheet` is the one generation task with its own ground truth, so the
long-form path is not left scored only against another model's output.

Everything is synthetic and seeded. RULER draws its QA tasks from SQuAD and HotpotQA; those
are not shipped here, so `qa` plants a generated paragraph instead -- the task shape is the
same (question as query, planted paragraph as the needle, dialog as the distractor haystack)
but a score is not comparable with a published RULER QA number, and the guide says so.

Pure data and strings: this module builds specs, `context_builder` renders them into a prompt
of an exact token length, and `scoring` grades the answers. Nothing here needs a tokenizer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

SUITE_RETRIEVAL = "retrieval"
SUITE_GENERATION = "generation"

# The planted values are drawn from unambiguous upper-case letters and digits (no O/0 or
# I/1/l), so a scorer never has to decide whether the model "meant" the right character. High
# entropy is deliberate: the value has to be something no run of classroom dialog would
# produce on its own, or a match would fire on text the needle never planted.
VALUE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
DEFAULT_VALUE_LENGTH = 6

# Distinct nouns for the needle keys, so `niah_multikey`'s decoys are told apart by the key
# and not by where they sit.
_KEYS = (
    "access", "session", "registry", "laboratory", "archive",
    "roster", "ledger", "equipment", "library", "workshop",
)

# Vocabulary for the aggregation tasks. Real words rather than generated tokens so the
# tokenizer's merge behaviour over the word list stays representative; ~150 is plenty, because
# `cwe` scoring depends on the target:distractor frequency *ratio* within one repetition of
# the list, not on how large the vocabulary is.
_VOCABULARY = (
    "lesson quiz teacher student pencil notebook diagram formula energy motion velocity "
    "friction gravity magnet circuit voltage battery lantern compass triangle circle square "
    "fraction decimal integer equation average median pattern sequence measure balance ruler "
    "thermometer beaker funnel crystal mineral fossil planet comet orbit telescope satellite "
    "atmosphere climate rainfall glacier desert forest meadow habitat species mammal reptile "
    "insect blossom pollen harvest granary compost windmill turbine pulley lever wedge screw "
    "hammer anvil forge timber quarry marble granite pottery weaving dyeing pigment canvas "
    "sculpture chorus rhythm melody harmony octave flute drumbeat ballad theatre costume "
    "puppet lantern parchment scroll ledger abacus calendar sundial almanac voyage harbour "
    "anchor cargo lighthouse shoreline estuary current tidepool coral plankton whale dolphin "
    "penguin falcon sparrow beetle spider lizard bison antler burrow thicket canyon plateau "
    "volcano geyser aquifer boulder pebble sediment magnet lantern trellis orchard beehive "
    "silo barn tractor furrow seedling sapling"
).split()

# `cwe`'s two frequencies. Targets appear ten times as often as distractors inside every
# repetition of the generated word list, so slicing that list to an exact token count -- which
# is how the filler is sized -- cannot reorder the ranking: a partial final repetition can move
# a count by at most one occurrence.
_CWE_TARGET_FREQ = 30
_CWE_DISTRACTOR_FREQ = 3

# The planted-paragraph QA pool. Each entry is (paragraph template, question, answer field).
_QA_ITEMS = (
    (
        "Administrative note: the field trip to the {place} observatory is scheduled for "
        "{month}, and it will be led by Doctor {person}.",
        "Which month is the field trip to the observatory scheduled for? "
        "Answer with only the month.",
        "month",
    ),
    (
        "Administrative note: the practical examination will be held in room {place}, and "
        "Doctor {person} is the examiner responsible for it.",
        "Who is the examiner responsible for the practical examination? "
        "Answer with only the name.",
        "person",
    ),
    (
        "Administrative note: the {month} laboratory session on energy transfer takes place "
        "in room {place} under the supervision of Doctor {person}.",
        "Where does the laboratory session on energy transfer take place? "
        "Answer with only the room.",
        "place",
    ),
)
_QA_MONTHS = ("January", "March", "April", "June", "September", "November")
_QA_PLACES = ("Halverson", "Pine Ridge", "Castellane", "Northbrook", "Eastgate")
_QA_PEOPLE = ("Ferreira", "Okonkwo", "Lindqvist", "Nakamura", "Abrahams")

# The generation suite's prompt pool -- the real classroom work, at a realistic answer length.
# Several distinct instructions rather than one repeated: WWB scores a *set* of prompts, and a
# fidelity number from a single summarization request says less than one taken across a
# summarize / explain / list / quiz spread, which exercise different output shapes.
_GENERATION_PROMPTS = (
    "Summarize the lesson's main topic and one key relationship explained by the teacher in "
    "two concise sentences.",
    "Explain, in your own words and in about three sentences, the relationship between speed "
    "and energy as the teacher described it.",
    "List the three most important points a student should revise from this lesson.",
    "Write two short quiz questions a teacher could ask to check understanding of this "
    "lesson, with their answers.",
    "Describe how the teacher moved from the opening review to the final derivation, in two "
    "or three sentences.",
)

# How the answer is graded. `match_all` is RULER's share-of-items-found, `match_part` its
# any-acceptable-answer rule for paraphrasable questions, `iou` its over-answering-aware
# metric for the aggregation tasks. See scoring.py.
SCORER_MATCH_ALL = "match_all"
SCORER_MATCH_PART = "match_part"
SCORER_IOU = "iou"


@dataclass(frozen=True)
class ProbeSpec:
    """One prompt to run and everything needed to grade its answer.

    `inserts` are `(depth, sentence)` pairs planted into the synthetic transcript, where depth
    0.0 is the very top and 1.0 the very bottom; `context_builder.build_probe_prompt` places
    them and sizes the filler around them so the whole prompt still lands on an exact token
    count. `filler` overrides the classroom-dialog haystack for the aggregation tasks, whose
    context *is* the data. `truths` are the acceptable answers and `distractors` the planted
    decoys that must NOT appear -- the precision signal item recall cannot give.

    `depth` is the one coordinate the report's depth × profile matrix is built on, and is None
    for tasks whose needles are spread by construction (multi-value, variable tracking,
    aggregation): those get their own per-task row instead of a depth sweep, because "the
    depth" of a probe with eight needles is not a thing that exists.
    """

    suite: str
    task: str
    sample: int
    question: str
    truths: list = field(default_factory=list)
    distractors: list = field(default_factory=list)
    inserts: list = field(default_factory=list)
    depth: float | None = None
    filler: str | None = None
    scorer: str = SCORER_MATCH_ALL
    output_tokens: int = 32

    @property
    def coordinate(self) -> tuple:
        """The key that pairs a returned answer back to this spec, and pairs a probe with the
        baseline profile's probe for the same work. Stable across processes and runs."""
        return (self.task, _depth_key(self.depth), self.sample)


def _depth_key(depth) -> str:
    return "-" if depth is None else f"{float(depth):.2f}"


# ---------------------------------------------------------------------------
# Deterministic draws
# ---------------------------------------------------------------------------
def probe_rng(seed: int, context_tokens: int, task: str, depth, sample: int) -> random.Random:
    """A generator seeded from the probe's coordinates, salt-free across processes.

    Seeding from the full coordinate is what makes every (task, depth, sample) carry different
    planted values while the same config still reproduces the same run: a profile cannot score
    a hit by memorizing one answer across the sweep, and a re-run is comparable with the last.
    Python's `hash()` is salted per process, so the tuple is rendered to a string and fed
    through Random rather than hashed.
    """
    rng = random.Random()
    rng.seed(f"{seed}|{context_tokens}|{task}|{_depth_key(depth)}|{sample}")
    return rng


def draw_value(rng: random.Random, haystack: str, length: int = DEFAULT_VALUE_LENGTH) -> str:
    """A high-entropy code that does not occur in `haystack`.

    Checked rather than assumed disjoint: a value that happened to be a substring of the
    planted dialog would score a retrieval hit the model never earned. Collisions are
    astronomically unlikely at this alphabet and length, but the check costs nothing and the
    failure it prevents is a silently inflated accuracy.
    """
    upper = haystack.upper()
    for _ in range(1000):
        value = "".join(rng.choice(VALUE_ALPHABET) for _ in range(length))
        if value not in upper:
            return value
    raise ValueError(
        "Could not draw a value disjoint from the haystack; widen VALUE_ALPHABET or raise "
        "the length"
    )


def _spread(count: int, rng: random.Random) -> list:
    """`count` depths spread over the transcript, one per band, jittered inside its band.

    Even spread rather than random placement: the point of a multi-needle task is that the
    needles are far apart, and an unlucky uniform draw that clusters all eight in the last
    tenth would quietly turn a long-context task into a short-context one.
    """
    if count <= 1:
        return [0.5]
    band = 1.0 / count
    return [
        round(min(1.0, max(0.0, index * band + rng.uniform(0.15, 0.85) * band)), 4)
        for index in range(count)
    ]


def _needle(key: str, value: str) -> str:
    return f"Note for the record: the {key} code for this session is {value}."


# ---------------------------------------------------------------------------
# Retrieval tasks (RULER)
# ---------------------------------------------------------------------------
# Tasks swept across `depths`: they plant one needle whose position is the measured variable.
# The rest place their needles by construction and run `samples` probes at depth None.
DEPTH_SWEPT_TASKS = ("niah_single", "niah_multikey", "qa")


def _build_niah_single(rng, haystack, depth, options) -> dict:
    key = _KEYS[0]
    value = draw_value(rng, haystack, options.get("value_length", DEFAULT_VALUE_LENGTH))
    return {
        "question": (
            f"What is the {key} code mentioned in the transcript? Answer with only the code."
        ),
        "truths": [value],
        "inserts": [(depth, _needle(key, value))],
    }


def _build_niah_multikey(rng, haystack, depth, options) -> dict:
    """The target needle at `depth`, decoys with different keys spread around it.

    The decoys are the measurement: a profile losing precision under quantization does not
    stop answering, it starts answering with the wrong key's value, and `distractor_rate`
    catches that where recall alone reads it as a plain miss.
    """
    count = options.get("num_distractors", 4)
    length = options.get("value_length", DEFAULT_VALUE_LENGTH)
    keys = _KEYS[:count + 1]
    values = [draw_value(rng, haystack, length) for _ in keys]
    inserts = [(depth, _needle(keys[0], values[0]))]
    inserts += [
        (decoy_depth, _needle(key, value))
        for decoy_depth, key, value in zip(_spread(count, rng), keys[1:], values[1:])
    ]
    return {
        "question": (
            f"What is the {keys[0]} code mentioned in the transcript? Several different codes "
            "are recorded; answer with only the one for "
            f"{keys[0]}."
        ),
        "truths": [values[0]],
        "distractors": values[1:],
        "inserts": inserts,
    }


def _build_niah_multivalue(rng, haystack, _depth, options) -> dict:
    """One key, several values, spread through the transcript: every one has to come back."""
    count = options.get("num_values", 4)
    length = options.get("value_length", DEFAULT_VALUE_LENGTH)
    key = _KEYS[0]
    values = [draw_value(rng, haystack, length) for _ in range(count)]
    return {
        "question": (
            f"The transcript records several {key} codes. List every {key} code it mentions, "
            "separated by commas, and nothing else."
        ),
        "truths": values,
        "inserts": list(zip(_spread(count, rng), (_needle(key, v) for v in values))),
    }


def _build_niah_multiquery(rng, haystack, _depth, options) -> dict:
    """Several keys, one value each: every needle has to be found, not just the easiest one."""
    count = options.get("num_queries", 4)
    length = options.get("value_length", DEFAULT_VALUE_LENGTH)
    keys = _KEYS[:count]
    values = [draw_value(rng, haystack, length) for _ in keys]
    listed = ", ".join(keys[:-1]) + f" and {keys[-1]}" if len(keys) > 1 else keys[0]
    return {
        "question": (
            f"The transcript records a code for each of these: {listed}. List all of them, "
            "separated by commas, and nothing else."
        ),
        "truths": values,
        "inserts": list(zip(
            _spread(count, rng), (_needle(k, v) for k, v in zip(keys, values))
        )),
    }


def _build_vt(rng, haystack, _depth, options) -> dict:
    """Variable tracking: chains of assignments that only resolve if every hop is followed.

    The target chain seeds a value and then re-binds it name to name; the decoy chains carry
    different values. Answering needs the model to hold a coreference chain across the whole
    context, which is a different failure mode from finding a span -- and the one that breaks
    first when a KV cache is quantized.
    """
    length = options.get("chain_length", 4)
    chains = options.get("num_chains", 4)
    value = draw_value(rng, haystack, options.get("value_length", DEFAULT_VALUE_LENGTH))

    statements, target_names, decoy_names = [], [], []
    for chain in range(chains):
        names = [f"VAR_{chain}_{step}" for step in range(length)]
        chain_value = value if chain == 0 else draw_value(rng, haystack)
        statements.append(f"Bookkeeping: {names[0]} is set to {chain_value}.")
        statements += [
            f"Bookkeeping: {names[step]} is set to the value of {names[step - 1]}."
            for step in range(1, length)
        ]
        (target_names if chain == 0 else decoy_names).extend(names)

    # Interleaved before spreading, so the chains are not each contiguous -- following one
    # requires carrying it across the others rather than reading a single block.
    rng.shuffle(statements)
    return {
        "question": (
            f"The transcript assigns values to variables, sometimes by copying another "
            f"variable. List every variable whose final value is {value}, separated by "
            "commas, and nothing else."
        ),
        "truths": target_names,
        "distractors": decoy_names,
        "inserts": list(zip(_spread(len(statements), rng), statements)),
    }


def _build_cwe(rng, haystack, _depth, options) -> dict:
    """Common word extraction: the context is a word list, and the answer is its top words.

    Aggregation, not retrieval -- there is no span to find, and getting it right means having
    counted the whole context. The word list replaces the classroom dialog as the filler, so
    `truths` are the words made frequent by construction.
    """
    count = options.get("num_target_words", 10)
    vocabulary = list(dict.fromkeys(_VOCABULARY))
    rng.shuffle(vocabulary)
    targets = vocabulary[:count]
    distractors = vocabulary[count:]

    words = [w for w in targets for _ in range(_CWE_TARGET_FREQ)]
    words += [w for w in distractors for _ in range(_CWE_DISTRACTOR_FREQ)]
    rng.shuffle(words)
    return {
        "question": (
            f"The word list above contains {count} words that occur far more often than any "
            f"other. List those {count} words, separated by commas, and nothing else."
        ),
        "truths": targets,
        "inserts": [],
        "filler": "WORD LIST\n" + " ".join(words) + "\n",
        "scorer": SCORER_IOU,
    }


def _build_fwe(rng, haystack, _depth, options) -> dict:
    """Frequent word extraction: the same aggregation, but with Zeta-distributed frequencies.

    Harder than `cwe` because the counts tail off smoothly instead of splitting into two
    groups, so a model that has only approximately counted gets the ordering wrong.
    """
    top_k = options.get("top_k", 3)
    alpha = options.get("alpha", 2.0)
    vocabulary = list(dict.fromkeys(_VOCABULARY))
    rng.shuffle(vocabulary)
    # Zeta weights: rank r appears proportionally to r**-alpha, scaled so the top word is
    # frequent enough to survive the filler being sliced to an exact token count.
    counts = [max(1, int(200 * (rank + 1) ** -alpha)) for rank in range(len(vocabulary))]
    words = [w for w, n in zip(vocabulary, counts) for _ in range(n)]
    rng.shuffle(words)
    return {
        "question": (
            f"List the {top_k} words that occur most often in the word list above, most "
            "frequent first, separated by commas, and nothing else."
        ),
        "truths": vocabulary[:top_k],
        "inserts": [],
        "filler": "WORD LIST\n" + " ".join(words) + "\n",
        "scorer": SCORER_IOU,
    }


def _build_qa(rng, haystack, depth, _options) -> dict:
    """A natural-language question over a planted paragraph.

    The answer is a normal word, not a high-entropy code, so the model has to understand the
    paragraph rather than copy the one string in the context that looks unlike the rest.
    Scored with `match_part`: a correct answer wrapped in a sentence is still correct.
    """
    paragraph, question, answer_field = _QA_ITEMS[rng.randrange(len(_QA_ITEMS))]
    fields = {
        "month": _QA_MONTHS[rng.randrange(len(_QA_MONTHS))],
        "place": _QA_PLACES[rng.randrange(len(_QA_PLACES))],
        "person": _QA_PEOPLE[rng.randrange(len(_QA_PEOPLE))],
    }
    return {
        "question": question,
        "truths": [fields[answer_field]],
        "inserts": [(depth, paragraph.format(**fields))],
        "scorer": SCORER_MATCH_PART,
    }


# ---------------------------------------------------------------------------
# Generation tasks (WWB fidelity / lexical task scoring)
# ---------------------------------------------------------------------------
def _build_summary(rng, _haystack, _depth, _options) -> dict:
    """The real workload: a classroom task over the transcript, at a realistic answer length.

    No ground truth, by design -- this is the WWB measurement, where the reference is the
    baseline profile's own answer to the same prompt and the question is how far the optimized
    profile drifted from it. `sample` selects the prompt, so N samples are N different tasks.
    """
    return {"question": _GENERATION_PROMPTS[rng.randrange(len(_GENERATION_PROMPTS))],
            "truths": [], "inserts": []}


def _build_fact_sheet(rng, haystack, _depth, options) -> dict:
    """Long-form answering with its own ground truth: facts planted, then asked for together.

    The generation suite's other task is scored only against the baseline, which measures drift
    but cannot tell a *pair* of equally wrong answers from a pair of right ones. This one can:
    the planted codes are known, so `recall` is real accuracy and `distractor_rate` is real
    hallucination, and they sit next to the same ROUGE/chrF/similarity columns.
    """
    count = options.get("num_facts", 6)
    length = options.get("value_length", DEFAULT_VALUE_LENGTH)
    keys = _KEYS[:count]
    values = [draw_value(rng, haystack, length) for _ in keys]
    return {
        "question": (
            "The transcript records an administrative code for several items. Write a short "
            "report listing each item together with its code, one per line."
        ),
        "truths": values,
        "inserts": list(zip(
            _spread(count, rng), (_needle(k, v) for k, v in zip(keys, values))
        )),
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_BUILDERS = {
    "niah_single": (SUITE_RETRIEVAL, _build_niah_single),
    "niah_multikey": (SUITE_RETRIEVAL, _build_niah_multikey),
    "niah_multivalue": (SUITE_RETRIEVAL, _build_niah_multivalue),
    "niah_multiquery": (SUITE_RETRIEVAL, _build_niah_multiquery),
    "vt": (SUITE_RETRIEVAL, _build_vt),
    "cwe": (SUITE_RETRIEVAL, _build_cwe),
    "fwe": (SUITE_RETRIEVAL, _build_fwe),
    "qa": (SUITE_RETRIEVAL, _build_qa),
    "summary": (SUITE_GENERATION, _build_summary),
    "fact_sheet": (SUITE_GENERATION, _build_fact_sheet),
}

RETRIEVAL_TASKS = tuple(n for n, (s, _) in _BUILDERS.items() if s == SUITE_RETRIEVAL)
GENERATION_TASKS = tuple(n for n, (s, _) in _BUILDERS.items() if s == SUITE_GENERATION)
KNOWN_TASKS = RETRIEVAL_TASKS + GENERATION_TASKS


def is_known(task) -> bool:
    return isinstance(task, str) and task in _BUILDERS


def suite_of(task: str) -> str:
    """The suite a task belongs to; raises for an unknown name so a typo in `tasks:` is a
    configuration error rather than a silently skipped measurement."""
    if task not in _BUILDERS:
        raise KeyError(
            f"unknown accuracy task {task!r}; known tasks are "
            f"{', '.join(sorted(_BUILDERS))}"
        )
    return _BUILDERS[task][0]


def is_depth_swept(task: str) -> bool:
    """Whether the task is measured across `depths`. See `ProbeSpec.depth`."""
    return task in DEPTH_SWEPT_TASKS


def build_spec(task: str, sample: int, depth, haystack: str, seed: int,
               context_tokens: int, output_tokens: int, options: dict | None = None) -> ProbeSpec:
    """One probe for `(task, depth, sample)`, deterministic in `seed` and `context_tokens`."""
    suite = suite_of(task)
    rng = probe_rng(seed, context_tokens, task, depth, sample)
    built = _BUILDERS[task][1](rng, haystack, depth, options or {})
    return ProbeSpec(
        suite=suite,
        task=task,
        sample=sample,
        depth=depth if is_depth_swept(task) else None,
        output_tokens=output_tokens,
        question=built["question"],
        truths=list(built.get("truths", [])),
        distractors=list(built.get("distractors", [])),
        inserts=list(built.get("inserts", [])),
        filler=built.get("filler"),
        scorer=built.get("scorer", SCORER_MATCH_ALL),
    )
