# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Pure scoring functions for the long-context accuracy path -- no model, no GPU, no I/O.

Everything here is `(strings or sequences) -> number`, so the whole accuracy contract is
unit-testable with a stub tokenizer and nothing installed. `accuracy.py` decides *which*
scores a probe gets; this module only knows how to compute them.

Three families, one per question the accuracy run asks:

  * **Retrieval** (RULER / NIAH). SQuAD normalization, exact match and token-F1, plus
    RULER's own `string_match_all` / `string_match_part` -- the share of ground-truth items
    that appear in the answer. RULER scores multi-value, multi-query, variable-tracking and
    word-extraction tasks with exactly this item-recall rule, so using it here means a number
    reported for `niah_multivalue` means what the paper means by it.

  * **Lexical task scoring** (ROUGE-1/2/L, chrF). Graded overlap for long-form answers, where
    exact match is useless and "did this profile still say the same things" is the question.
    Hand-rolled rather than pulled from `rouge-score`/`sacrebleu`: they are the standard
    implementations, but both are new dependencies for ~80 lines of n-gram and LCS counting,
    and the hardware-free test contract is worth more here than the last decimal place.

  * **Divergent token metrics** (FDT/SDT), from Divergent Token Metrics (arXiv:2311.01544)
    and reported by OpenVINO's who_what_benchmark. These are the fidelity metrics that catch
    what a similarity score smooths over: FDT says how many tokens two configurations agreed
    on before they first differed, SDT how much of the output differs overall. They need only
    two token sequences, which is why they are implemented here instead of being delegated to
    `wwb_adapter` along with the embedding similarity.

Every function returns `None` where the input cannot support a number, rather than 0.0 --
"nothing measured" and "nothing matched" are different findings, the same convention
`metrics.aggregate` keeps.
"""

from __future__ import annotations

import re
import string
from collections import Counter

# SQuAD's normalization, so a token-F1 here means what it means in the QA literature.
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    """SQuAD normalization: lowercase, drop punctuation and articles, collapse whitespace."""
    lowered = (text or "").lower()
    without_punct = lowered.translate(_PUNCT_TABLE)
    without_articles = _ARTICLES.sub(" ", without_punct)
    return " ".join(without_articles.split())


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def lexical_tokens(text: str) -> list[str]:
    """Words for the lexical metrics: lowercased, punctuation dropped, **articles kept**.

    Deliberately not `tokens()`. SQuAD normalization throws away "a"/"an"/"the" because a QA
    answer is no less correct for being preceded by an article -- but ROUGE exists to notice
    that a profile started writing "on a mat" where the baseline wrote "on the mat", and under
    SQuAD normalization those two are the same string. Using the retrieval tokenizer for the
    fidelity metrics would report 1.00 for a real rewording.
    """
    return re.findall(r"\w+", (text or "").lower())


# ---------------------------------------------------------------------------
# Retrieval scoring (SQuAD + RULER)
# ---------------------------------------------------------------------------
def exact_match(prediction: str, truths: list[str]) -> bool:
    """The answer is the ground truth and nothing else, as an order-insensitive token multiset.

    For a single-item truth this is SQuAD exact match unchanged: "the code is 7Q4M91" is a
    retrieval hit but not an exact match, and reporting both is what distinguishes "found the
    fact" from "answered with only the fact" -- the difference a question that says "answer
    with only the code" is trying to elicit. For a multi-item truth, order is not part of the
    answer (`X, Y` and `Y, X` are the same set of variables), so the multiset is compared
    rather than the string.
    """
    if not truths:
        return False
    return Counter(tokens(prediction)) == Counter(tokens(" ".join(truths)))


def string_match_all(prediction: str, truths: list[str]) -> float:
    """RULER's scorer: the share of ground-truth items that appear in the answer.

    This is item recall, not token recall -- for `niah_multivalue` the question is how many of
    the planted values came back, and a value is either there or it is not. Matching is done on
    the SQuAD-normalized strings so case and punctuation around the item do not decide a hit.
    """
    if not truths:
        return 0.0
    haystack = normalize(prediction)
    hits = sum(1 for truth in truths if normalize(truth) and normalize(truth) in haystack)
    return round(hits / len(truths), 4)


def string_match_part(prediction: str, truths: list[str]) -> float:
    """RULER's QA scorer: 1.0 if *any* acceptable answer appears. A paraphrase question has
    several correct surface forms, and requiring all of them would score a right answer wrong."""
    if not truths:
        return 0.0
    haystack = normalize(prediction)
    return 1.0 if any(normalize(t) and normalize(t) in haystack for t in truths) else 0.0


def token_f1(prediction: str, truths: list[str]) -> float:
    """SQuAD token-level F1 over the normalized token multisets, against the joined truth."""
    pred_tokens = tokens(prediction)
    truth_tokens = tokens(" ".join(truths or []))
    if not pred_tokens or not truth_tokens:
        # SQuAD's convention: F1 is 1.0 only when both sides are empty.
        return 1.0 if pred_tokens == truth_tokens else 0.0
    common = _multiset_overlap(pred_tokens, truth_tokens)
    if not common:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(truth_tokens)
    return round(2 * precision * recall / (precision + recall), 4)


def _multiset_overlap(a: list, b: list) -> int:
    counts = Counter(a)
    overlap = 0
    for item in b:
        if counts[item] > 0:
            counts[item] -= 1
            overlap += 1
    return overlap


def intersection_over_union(prediction_items: list[str], truths: list[str]) -> float:
    """RULER's CWE metric: |predicted ∩ truth| / |predicted ∪ truth|.

    Unlike `string_match_all` this punishes over-answering. A model that lists fifty words to
    be sure the ten frequent ones are among them has not done the aggregation task, and item
    recall alone would score it a perfect 1.0.
    """
    predicted = {_item_key(item) for item in prediction_items if _item_key(item)}
    truth = {_item_key(item) for item in truths if _item_key(item)}
    if not predicted and not truth:
        return 1.0
    union = predicted | truth
    return round(len(predicted & truth) / len(union), 4) if union else 0.0


def _item_key(text: str) -> str:
    """The comparison key for a list item: lowercased words, punctuation dropped.

    Not the SQuAD `normalize`, which deletes "a"/"an"/"the". That is right for matching a
    planted code inside a sentence and wrong here, where the items *are* words: a word-
    extraction task whose answer happens to include an article would silently lose it from
    both sides of the set, and score against a truth it never asked for.
    """
    return " ".join(lexical_tokens(text))


def distractor_rate(prediction: str, distractors: list[str]) -> float | None:
    """Share of the planted decoys that wrongly appear in the answer -- the precision signal
    item recall cannot give.

    A quantized profile that starts confusing one planted key for another shows up here before
    it shows up in recall: it is still returning *a* value, just the wrong one. `None` when the
    task planted no decoys, so aggregation does not average a rate that was never measured.
    """
    if not distractors:
        return None
    haystack = normalize(prediction)
    hits = sum(1 for d in distractors if normalize(d) and normalize(d) in haystack)
    return round(hits / len(distractors), 4)


def split_items(text: str) -> list[str]:
    """Split a free-text answer into the items a list-answer task asked for.

    Models answer "apple, banana and cherry", "apple\\nbanana\\ncherry" or "1. apple ..."
    interchangeably, and an aggregation task's IoU must not depend on which. Splits on commas,
    semicolons, newlines and the word "and", then strips list bullets and numbering.
    """
    parts = re.split(r"[,;\n]+|\band\b", text or "")
    items = []
    for part in parts:
        cleaned = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", part).strip()
        if cleaned:
            items.append(cleaned)
    return items


# ---------------------------------------------------------------------------
# Lexical task scoring (ROUGE / chrF)
# ---------------------------------------------------------------------------
def _ngrams(sequence: list, n: int) -> Counter:
    return Counter(tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1))


def _f_measure(overlap: int, pred_total: int, ref_total: int, beta: float = 1.0) -> float:
    """Precision/recall harmonic mean with a recall weight, shared by ROUGE and chrF."""
    if not overlap or not pred_total or not ref_total:
        return 0.0
    precision = overlap / pred_total
    recall = overlap / ref_total
    beta_sq = beta * beta
    return round(
        (1 + beta_sq) * precision * recall / (beta_sq * precision + recall), 4
    )


def rouge_n(prediction: str, reference: str, n: int = 1) -> float:
    """ROUGE-N F-measure: n-gram overlap over the normalized word sequences.

    F rather than ROUGE's traditional recall-only form: the comparison here is between two
    generations of similar length, not a summary against a long source, so a profile that
    pads its answer must not score higher for it.
    """
    pred_grams = _ngrams(lexical_tokens(prediction), n)
    ref_grams = _ngrams(lexical_tokens(reference), n)
    overlap = sum((pred_grams & ref_grams).values())
    return _f_measure(overlap, sum(pred_grams.values()), sum(ref_grams.values()))


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F-measure: longest-common-subsequence overlap over the normalized words.

    LCS rewards the two answers making the same points *in the same order* without requiring
    them to be contiguous, which is the useful signal when a decode-side change reworded a
    sentence but kept the content and its sequence.
    """
    pred_tokens = lexical_tokens(prediction)
    ref_tokens = lexical_tokens(reference)
    return _f_measure(_lcs_length(pred_tokens, ref_tokens), len(pred_tokens), len(ref_tokens))


def _lcs_length(a: list, b: list) -> int:
    """LCS length in O(len(a) * len(b)) time and O(min) space -- answers here are a few
    hundred tokens, so the quadratic table is cheap; only the rolling row is kept anyway."""
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = [0] * (len(b) + 1)
    for item_a in a:
        current = [0]
        for index, item_b in enumerate(b):
            current.append(
                previous[index] + 1 if item_a == item_b
                else max(previous[index + 1], current[index])
            )
        previous = current
    return previous[-1]


def _chrf_chars(text: str) -> list:
    """chrF's character sequence: lowercased, whitespace removed, punctuation kept.

    Punctuation stays because at character level it carries real signal -- a profile that
    stopped emitting the baseline's line breaks and commas is producing differently structured
    output, and that is the thing chrF is here to notice. Whitespace goes, as chrF specifies.
    """
    return [c for c in (text or "").lower() if not c.isspace()]


def chrf(prediction: str, reference: str, max_n: int = 6, beta: float = 2.0) -> float:
    """chrF: character n-gram F-score, averaged over n = 1..max_n, with recall weighted beta×.

    Character-level, so unlike ROUGE it still scores a near-miss: a profile whose answer
    differs by a suffix or a digit reads as "almost the same" rather than as a missed token.
    The defaults (n up to 6, beta 2) are chrF's standard settings.
    """
    pred_chars = _chrf_chars(prediction)
    ref_chars = _chrf_chars(reference)
    if not pred_chars or not ref_chars:
        return 1.0 if pred_chars == ref_chars else 0.0
    scores = []
    for n in range(1, max_n + 1):
        pred_grams, ref_grams = _ngrams(pred_chars, n), _ngrams(ref_chars, n)
        if not pred_grams or not ref_grams:
            continue
        scores.append(_f_measure(
            sum((pred_grams & ref_grams).values()),
            sum(pred_grams.values()), sum(ref_grams.values()), beta,
        ))
    return round(sum(scores) / len(scores), 4) if scores else 0.0


# ---------------------------------------------------------------------------
# Divergent token metrics (WWB fidelity)
# ---------------------------------------------------------------------------
def first_divergent_token(reference_ids: list, prediction_ids: list) -> int:
    """FDT: how many leading tokens the two generations agree on before the first difference.

    Higher is better, and the unit is tokens, not a rate -- which is the point. Greedy decoding
    is supposed to be deterministic, so two configurations of the same model should agree from
    the first token; an FDT of 3 on a 256-token answer says the divergence happened almost
    immediately, and an FDT equal to the answer length says the outputs are identical. A
    similarity score cannot make that distinction, because two answers that diverge at token 3
    and still discuss the same lesson embed almost identically.
    """
    limit = min(len(reference_ids), len(prediction_ids))
    for index in range(limit):
        if reference_ids[index] != prediction_ids[index]:
            return index
    return limit


def divergent_tokens(reference_ids: list, prediction_ids: list) -> dict:
    """The FDT/SDT pair and their length-normalized forms, as who_what_benchmark reports them.

    SDT is the count of positions at which the two sequences differ, aligned from the start and
    charging every token past the shorter sequence's end -- a profile that stops early has
    diverged for the whole remaining answer. Normalization is by the reference length, so
    `sdt_norm` is a comparable rate across probes whose answers differ in length and `fdt_norm`
    is "what fraction of the reference did the two agree on".

    `None` throughout when the reference is empty: there is nothing to have diverged from.
    """
    if not reference_ids:
        return {"fdt": None, "fdt_norm": None, "sdt": None, "sdt_norm": None}
    longest = max(len(reference_ids), len(prediction_ids))
    differing = sum(
        1 for index in range(longest)
        if _at(reference_ids, index) != _at(prediction_ids, index)
    )
    fdt = first_divergent_token(reference_ids, prediction_ids)
    return {
        "fdt": fdt,
        "fdt_norm": round(fdt / len(reference_ids), 4),
        "sdt": differing,
        "sdt_norm": round(differing / len(reference_ids), 4),
    }


def _at(sequence: list, index: int):
    """The token at `index`, or a sentinel past the end that equals nothing -- so a short
    answer counts as divergent for its missing tail rather than silently matching."""
    return sequence[index] if index < len(sequence) else _MISSING


_MISSING = object()
