# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Optional bridge to OpenVINO's who_what_benchmark for the embedding-similarity metric.

WWB is the tool the OpenVINO project itself validates compressed models with: it generates
from a baseline and an optimized model over the same prompts and reports how similar the two
sets of outputs are, where similarity is the cosine distance between sentence embeddings
(`sentence-transformers/all-mpnet-base-v2` by default). That is the one fidelity metric this
repo cannot reimplement honestly -- it is defined by a specific 420 MB embedding model, and a
hand-rolled substitute would produce a number that looks like WWB's and is not.

So this module delegates rather than reimplements, and is the *only* part of the accuracy path
that needs anything installed:

    pip install "whowhatbench @ git+https://github.com/openvinotoolkit/openvino.genai.git#subdirectory=tools/who_what_benchmark"

Everything else -- the RULER tasks, ROUGE/chrF, and the FDT/SDT divergent-token metrics WWB
also reports -- is computed natively in `scoring.py`, which keeps the suite fully runnable and
fully unit-testable with nothing installed. When WWB is absent the `similarity` column reports
`--` and the report says why; no other metric is affected and no run fails.

The return shape of WWB's evaluator has moved between versions (a metrics DataFrame, a
`(mean, per_sample)` pair, a plain dict), and it is not a published API. `_as_similarities`
therefore normalizes whatever comes back instead of assuming one shape, and an unrecognized
shape degrades to "unavailable" with the reason attached -- the accuracy run must not die
because an upstream tool refactored its return value.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Where `TextSimilarity` has lived; tried in order. `whowhatbench` re-exports it at the top
# level in recent versions, and `whowhat_metrics` is where it is actually defined.
_MODULE_CANDIDATES = ("whowhatbench.whowhat_metrics", "whowhatbench")

# One loaded embedding model per id: it is hundreds of megabytes and every case in the sweep
# scores against the same one.
_EVALUATOR_CACHE: dict = {}


@dataclass(frozen=True)
class SimilarityResult:
    """Per-probe similarities, or the reason there are none.

    `values` is one score per (reference, prediction) pair in input order, so a caller can put
    each probe's own similarity on its row; `mean` is what WWB prints. `reason` is non-None
    exactly when `values` is None, and is shown once in the report rather than per probe.
    """

    values: list | None
    mean: float | None
    reason: str | None = None

    @property
    def available(self) -> bool:
        return self.values is not None


def unavailable(reason: str) -> SimilarityResult:
    return SimilarityResult(values=None, mean=None, reason=reason)


def similarity(references: list, predictions: list,
               model_id: str = DEFAULT_EMBEDDING_MODEL) -> SimilarityResult:
    """WWB's embedding similarity for each (reference, prediction) pair, or why it is missing.

    `references` are the baseline profile's answers and `predictions` the ones being judged --
    the same direction WWB's `--base-model` / `--target-model` split uses, so a score here is
    comparable with one from the `wwb` CLI on the same pair of models.
    """
    if len(references) != len(predictions):
        raise ValueError(
            f"similarity needs paired inputs; got {len(references)} references "
            f"and {len(predictions)} predictions"
        )
    if not references:
        return unavailable("no probes to score")

    evaluator = _evaluator(model_id)
    if isinstance(evaluator, str):
        return unavailable(evaluator)
    try:
        raw = evaluator.evaluate(list(references), list(predictions))
    except Exception as exc:  # noqa: BLE001 - an optional metric must not end the run
        return unavailable(f"who_what_benchmark raised {type(exc).__name__}: {exc}")

    values = _as_similarities(raw, len(references))
    if values is None:
        return unavailable(
            f"could not read a similarity out of {type(raw).__name__} -- "
            "who_what_benchmark's return shape is not one this adapter recognizes"
        )
    return SimilarityResult(values=values, mean=round(sum(values) / len(values), 4))


def _evaluator(model_id: str):
    """A cached `TextSimilarity`, or a string explaining why there is none."""
    if model_id in _EVALUATOR_CACHE:
        return _EVALUATOR_CACHE[model_id]

    import importlib

    text_similarity = None
    for name in _MODULE_CANDIDATES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        text_similarity = getattr(module, "TextSimilarity", None)
        if text_similarity is not None:
            break

    if text_similarity is None:
        result = (
            "who_what_benchmark is not installed (no whowhatbench.TextSimilarity). "
            "Install it to get the embedding-similarity column; every other accuracy "
            "metric is computed natively and is unaffected."
        )
    else:
        try:
            result = text_similarity(model_id)
        except Exception as exc:  # noqa: BLE001
            result = (
                f"who_what_benchmark could not load the embedding model {model_id!r}: "
                f"{type(exc).__name__}: {exc}"
            )
    _EVALUATOR_CACHE[model_id] = result
    return result


def _as_similarities(raw, expected: int) -> list | None:
    """Pull `expected` per-pair similarities out of whatever WWB's evaluator returned.

    Handles the shapes seen upstream -- a `(aggregate, per_sample)` pair, a pandas DataFrame or
    dict with a `similarity` column, and a bare sequence of floats -- and returns None for
    anything else rather than guessing. A per-sample list of the wrong length is also None: it
    cannot be aligned to the probes, and aligning it wrongly would put one probe's score on
    another probe's row.
    """
    # `(metrics, per_sample)`: the per-sample half is the one that aligns with the probes.
    if isinstance(raw, tuple) and len(raw) == 2:
        for candidate in (raw[1], raw[0]):
            values = _as_similarities(candidate, expected)
            if values is not None:
                return values
        return None

    column = _similarity_column(raw)
    if column is None:
        return None
    try:
        values = [float(v) for v in column]
    except (TypeError, ValueError):
        return None
    # A single aggregate is a legitimate answer when there is exactly one pair; otherwise it
    # is the mean, which cannot be spread back across the probes.
    return values if len(values) == expected else None


def _similarity_column(raw):
    """The sequence of similarity numbers inside `raw`, without importing pandas."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return [raw]
    # DataFrame / dict / anything else indexable by the column name.
    for key in ("similarity", "Similarity"):
        try:
            if key in raw:
                column = raw[key]
                return list(column.values) if hasattr(column, "values") else list(column)
        except TypeError:
            break
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return None
