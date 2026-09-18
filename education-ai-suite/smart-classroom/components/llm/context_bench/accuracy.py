# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""The accuracy path: which probes a case runs, and what their answers score.

The throughput benchmark deliberately ignores answer content -- see benchmark.py. This module
is the opt-in accuracy path, and it answers the one question throughput cannot: does this
configuration still *use* the long context correctly, after whatever KV quantization, weight
compression or speculative decoding bought the speed.

Two suites, because that question has two halves that fail independently.

  * **Retrieval** -- RULER's synthetic probes (`tasks.py`): find a planted fact, find it among
    decoys, find all of them, follow a chain of assignments, count the whole context. Scored
    against known ground truth, so a number here is accuracy in the plain sense.

  * **Generation** -- the real classroom task at a realistic answer length, scored for
    *fidelity* against the baseline profile's answer to the same prompt. This is
    who_what_benchmark's measurement, and it is what catches the regression retrieval misses:
    a profile can still return every planted code and write visibly worse prose. Three
    families of number, deliberately kept side by side because each is blind to something:
    embedding **similarity** (semantic, via `wwb_adapter`, the only optional dependency),
    **ROUGE/chrF** (lexical -- did it say the same words), and **FDT/SDT** (token-exact -- how
    many tokens in did greedy decoding first disagree). A paraphrase scores ~1.0 on
    similarity, ~0.6 on ROUGE and 3 on FDT, and reporting only the first would call that
    "lossless".

Scoring runs in the orchestrator, not the child subprocess: the child returns only its
generated text, so a run can be re-scored without re-loading a 14-33 GB model, and
trial_runner never imports this module. The generation suite's fidelity scores need the
baseline profile's answers, which only exist once that case has run, so they are filled at
report time (`benchmark._fill_generation_fidelity`) rather than per case.
"""

from __future__ import annotations

from components.llm.context_bench import scoring, tasks, wwb_adapter
from components.llm.context_bench.context_builder import corpus_text
from components.llm.context_bench.tasks import (  # re-exported for benchmark.py and tests
    SUITE_GENERATION,
    SUITE_RETRIEVAL,
    ProbeSpec,
)

__all__ = [
    "SUITE_GENERATION", "SUITE_RETRIEVAL", "ProbeSpec",
    "RETRIEVAL_METRICS", "GENERATION_METRICS", "PROBE_SCORE_FIELDS",
    "build_probe_specs", "score_retrieval", "score_grounding", "score_fidelity",
    "aggregate", "delta", "metrics_for",
]

# Aggregate metric names, in report order. `exact_match_rate` and `recall_rate` keep the
# names the retrieval report has always used; the rest are named after the probe field.
RETRIEVAL_METRICS = ("exact_match_rate", "recall_rate", "token_f1", "distractor_rate")
GENERATION_METRICS = (
    "similarity", "rouge1", "rouge2", "rouge_l", "chrf",
    "fdt", "fdt_norm", "sdt_norm", "identical_rate", "fact_coverage",
)

# Probe-level field -> aggregate name. Anything not listed aggregates under its own name.
_AGGREGATE_NAMES = {
    "exact_match": "exact_match_rate",
    "recall": "recall_rate",
    "identical": "identical_rate",
}
_PROBE_FIELD = {aggregate: probe for probe, aggregate in _AGGREGATE_NAMES.items()}

# Every per-probe score column, so probes.csv has one stable schema across both suites and a
# row simply leaves blank the metrics its suite does not produce.
PROBE_SCORE_FIELDS = (
    "exact_match", "recall", "token_f1", "distractor_rate",
    "similarity", "rouge1", "rouge2", "rouge_l", "chrf",
    "fdt", "fdt_norm", "sdt", "sdt_norm", "identical", "fact_coverage",
)


def metrics_for(suite: str) -> tuple:
    return GENERATION_METRICS if suite == SUITE_GENERATION else RETRIEVAL_METRICS


# ---------------------------------------------------------------------------
# Probe construction
# ---------------------------------------------------------------------------
def build_probe_specs(accuracy_cfg: dict, context_tokens: int) -> list:
    """Every probe one (model, profile, context) case runs, in a deterministic order.

    Depth-swept tasks (`tasks.DEPTH_SWEPT_TASKS`) produce `depths x samples` probes -- the
    classic NIAH sweep, where position in the context is the measured variable. The rest place
    their needles by construction and produce `samples` probes: "the depth" of a probe with
    eight spread needles is not a thing that exists, and pretending otherwise would multiply
    the run time by `len(depths)` for identical prompts.

    Ordering is by suite then task then depth then sample, so a long run's log reads in a
    predictable order and a re-run is comparable line by line.
    """
    haystack = corpus_text()
    seed = accuracy_cfg.get("seed", 0)
    specs = []
    for suite in (SUITE_RETRIEVAL, SUITE_GENERATION):
        section = accuracy_cfg.get(suite)
        if not section:
            continue
        options = section.get("options") or {}
        for task in section["tasks"]:
            depths = section["depths"] if tasks.is_depth_swept(task) else [None]
            for depth in depths:
                for sample in range(section["samples"]):
                    specs.append(tasks.build_spec(
                        task=task,
                        sample=sample,
                        depth=depth,
                        haystack=haystack,
                        seed=seed,
                        context_tokens=context_tokens,
                        output_tokens=section["output_tokens"],
                        options=options.get(task) or {},
                    ))
    return specs


# ---------------------------------------------------------------------------
# Scoring one answer
# ---------------------------------------------------------------------------
def score_retrieval(prediction: str, spec: ProbeSpec) -> dict:
    """The retrieval scores for one answer, as they land in probes.csv.

    `recall` is the primary number and is RULER's own scorer for the task -- item recall for
    the NIAH and tracing tasks, intersection-over-union for the aggregation ones (which have
    to punish over-answering, since listing fifty words to be sure of ten is not counting),
    and any-acceptable-answer for the paraphrasable QA task. `exact_match` and `token_f1` are
    the SQuAD pair kept alongside it, and `distractor_rate` is the precision signal: a profile
    that has started confusing one planted key for another is still returning *a* value, and
    recall alone reads that as a plain miss.
    """
    if spec.scorer == tasks.SCORER_IOU:
        recall = scoring.intersection_over_union(
            scoring.split_items(prediction), spec.truths
        )
    elif spec.scorer == tasks.SCORER_MATCH_PART:
        recall = scoring.string_match_part(prediction, spec.truths)
    else:
        recall = scoring.string_match_all(prediction, spec.truths)
    return {
        "exact_match": scoring.exact_match(prediction, spec.truths),
        "recall": recall,
        "token_f1": scoring.token_f1(prediction, spec.truths),
        "distractor_rate": scoring.distractor_rate(prediction, spec.distractors),
    }


def score_grounding(prediction: str, spec: ProbeSpec) -> dict:
    """The generation suite's ground-truth score: the share of planted facts the answer names.

    `None` for the open-ended `summary` task, which has no ground truth by design -- its whole
    measurement is the fidelity comparison against the baseline. Aggregation skips the Nones,
    so a suite mixing both tasks reports `fact_coverage` over the probes that actually have a
    truth rather than diluting it with the ones that cannot.
    """
    if not spec.truths:
        return {"fact_coverage": None}
    return {"fact_coverage": scoring.string_match_all(prediction, spec.truths)}


def score_fidelity(prediction: str, reference: str,
                   prediction_ids: list | None = None,
                   reference_ids: list | None = None) -> dict:
    """How far one profile's answer drifted from the baseline profile's answer.

    Lexical (ROUGE/chrF) and token-exact (FDT/SDT) only -- embedding similarity is filled
    separately and in one batch by `wwb_adapter`, because loading its model per probe would
    dominate the scoring pass.

    `*_ids` are the two answers as model token sequences and are what FDT/SDT are defined
    over. When the orchestrator could not load the model's tokenizer they fall back to
    normalized words, which changes the unit of FDT (words, not tokens) but not its meaning;
    the report says which was used rather than leaving the reader to guess.
    """
    if prediction_ids is None:
        prediction_ids = scoring.lexical_tokens(prediction)
    if reference_ids is None:
        reference_ids = scoring.lexical_tokens(reference)
    return {
        "rouge1": scoring.rouge_n(prediction, reference, 1),
        "rouge2": scoring.rouge_n(prediction, reference, 2),
        "rouge_l": scoring.rouge_l(prediction, reference),
        "chrf": scoring.chrf(prediction, reference),
        "identical": prediction == reference,
        **scoring.divergent_tokens(reference_ids, prediction_ids),
    }


def fill_similarity(probes: list, references: list, model_id: str):
    """Attach WWB's embedding similarity to each probe in one batched call.

    Returns the `wwb_adapter.SimilarityResult` so the caller can surface, once, why the column
    is empty when who_what_benchmark is not installed. Probes are left untouched rather than
    zeroed in that case: an unmeasured similarity is not a similarity of 0.
    """
    result = wwb_adapter.similarity(references, [p.get("prediction", "") for p in probes],
                                    model_id)
    if result.available:
        for probe, value in zip(probes, result.values):
            probe["similarity"] = round(float(value), 4)
    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(scores: list, metrics: tuple) -> dict:
    """Mean of each metric over a set of scored probes, skipping the ones it does not apply to.

    Per-metric skipping matters here in a way it does not for the throughput aggregates: a
    generation suite mixing `summary` and `fact_sheet` probes has `fact_coverage` on only some
    of them, and a retrieval task with no decoys has no `distractor_rate` at all. Averaging a
    missing measurement as 0.0 would report a hallucination rate for a task that never planted
    a decoy.

    Empty input reports a `probe_count` of 0 with no rates rather than phantom zeros --
    "nothing measured" and "nothing retrieved" are different findings, exactly as
    `metrics.aggregate` treats an empty run.
    """
    out = {"probe_count": len(scores)}
    for metric in metrics:
        field = _PROBE_FIELD.get(metric, metric)
        values = [s[field] for s in scores if isinstance(s.get(field), (int, float, bool))]
        out[metric] = round(sum(float(v) for v in values) / len(values), 4) if values else None
    return out


def group_aggregate(scored: list, metrics: tuple, key) -> dict:
    """`aggregate` applied per group, keyed by `key(probe)` and sorted by that key."""
    groups: dict = {}
    for probe in scored:
        groups.setdefault(key(probe), []).append(probe)
    return {name: aggregate(rows, metrics) for name, rows in sorted(groups.items())}


def delta(profile_agg: dict, baseline_agg: dict, metrics: tuple) -> dict:
    """Each metric minus the baseline profile's, so a regression reads as a negative number.

    This is what turns the accuracy sweep into a precision-loss measurement: a u8-KV or MTP
    profile that retrieves the needle less often than the f16/no-MTP baseline shows up here as
    a negative delta, next to whatever TPOT it bought. `None` where either side lacks the
    metric, rather than a delta computed against a missing measurement.

    Note `distractor_rate` and `sdt_norm` are metrics where *lower* is better, so their sign
    reads the other way round; the report labels them rather than flipping them here, because
    a delta that silently negates some of its columns is a delta nobody can check.
    """
    out = {}
    for metric in metrics:
        mine, base = profile_agg.get(metric), baseline_agg.get(metric)
        out[f"{metric}_delta"] = (
            round(mine - base, 4)
            if isinstance(mine, (int, float)) and isinstance(base, (int, float))
            else None
        )
    return out
