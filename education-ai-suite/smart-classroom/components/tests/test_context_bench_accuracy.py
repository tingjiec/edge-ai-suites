# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""The accuracy suites -- RULER retrieval and who_what_benchmark generation fidelity --
exercised without a GPU, a model, or the OpenVINO stack, the same hardware-free contract the
other context_bench tests keep.

Four things are pinned here.

  * **Scoring** is pure arithmetic (SQuAD, RULER's item recall and IoU, ROUGE, chrF, FDT/SDT),
    so its edge cases are checked directly. The one that has bitten this code already gets its
    own test: the lexical metrics must NOT use the SQuAD tokenizer, which drops articles and
    would score "on a mat" against "on the mat" as a perfect match.
  * **Task construction** has to keep its ground truth true -- distinct values per probe, a
    corpus-disjoint alphabet, decoys that differ from the target, and, for the aggregation
    tasks, a word list whose top words survive being sliced to an exact token count.
  * **Prompt building** has to place every insert at its depth and land within tolerance of the
    target token count, for up to sixteen inserts, exercised with the word-level stub
    tokenizer. The tolerance itself is pinned: demanding exactness here threw away a whole
    case over a prompt of 8,001 tokens where 8,000 was asked for.
  * **The orchestration** -- config validation, per-case scoring, the cross-case fidelity fill
    and the report cells -- is dicts in, strings/dicts out, so it needs no runtime either.

`whowhatbench` is an optional dependency and is not installed in CI, so the adapter is tested
against a stub injected into `sys.modules`: what matters is that every return shape upstream
has used is read correctly and that an unrecognized one degrades to "unavailable" rather than
ending the run.

The load-bearing invariant -- that turning accuracy ON does not change the throughput path's
CSV headers -- is asserted at the end.
"""

import sys
import types
import unittest
from collections import Counter
from types import SimpleNamespace

from components.llm.context_bench import (
    accuracy,
    benchmark,
    scoring,
    tasks,
    wwb_adapter,
)
from components.llm.context_bench.context_builder import (
    _USER_SUFFIX,
    build_probe_prompt,
    corpus_text,
)


class FakeTokenizer:
    """Word-level stub with an exact encode/decode round trip -- identical to the one the
    context-builder test uses, copied so this file stands alone."""

    def __init__(self):
        self._id_to_word = []
        self._word_to_id = {}

    def encode(self, text, add_special_tokens=True):
        ids = []
        for word in text.split():
            if word not in self._word_to_id:
                self._word_to_id[word] = len(self._id_to_word)
                self._id_to_word.append(word)
            ids.append(self._word_to_id[word])
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self._id_to_word[i] for i in ids)

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        parts = [f"<{m['role']}> {m.get('content', '')}" for m in messages]
        return " ".join(parts)


class BoundaryDriftTokenizer(FakeTokenizer):
    """Adds a token at the transcript/template seam, so the correction loop has drift to null
    out -- a multi-insert probe has two seams per insert, the harder case."""

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        rendered = super().apply_chat_template(messages, tokenize=tokenize, **kwargs)
        if messages[-1].get("content"):
            rendered += " boundary-token"
        return rendered


# ---------------------------------------------------------------------------
# Scoring: retrieval
# ---------------------------------------------------------------------------
class TestRetrievalScoring(unittest.TestCase):
    def test_exact_match_is_squad_normalized(self):
        # Case and surrounding articles are normalized away before comparison.
        self.assertTrue(scoring.exact_match("7Q4M91", ["7q4m91"]))
        self.assertTrue(scoring.exact_match("The Code", ["code"]))
        self.assertFalse(scoring.exact_match("the code is 7Q4M91", ["7Q4M91"]))

    def test_exact_match_over_several_items_ignores_order(self):
        # `X, Y` and `Y, X` are the same set of variables; order is not part of the answer.
        self.assertTrue(scoring.exact_match("VAR_A, VAR_B", ["VAR_B", "VAR_A"]))
        self.assertFalse(scoring.exact_match("VAR_A", ["VAR_B", "VAR_A"]))

    def test_string_match_all_is_ruler_item_recall(self):
        # The share of planted items that came back, regardless of the words around them.
        self.assertEqual(scoring.string_match_all("codes: AAA and BBB", ["AAA", "BBB"]), 1.0)
        self.assertEqual(scoring.string_match_all("codes: AAA", ["AAA", "BBB"]), 0.5)
        self.assertEqual(scoring.string_match_all("nothing", ["AAA"]), 0.0)

    def test_string_match_part_accepts_any_correct_surface_form(self):
        # A paraphrasable question has several right answers; requiring all would fail a
        # correct one.
        self.assertEqual(scoring.string_match_part("It is in March.", ["March", "03"]), 1.0)
        self.assertEqual(scoring.string_match_part("It is in July.", ["March", "03"]), 0.0)

    def test_iou_punishes_over_answering(self):
        # Listing everything to be sure the right words are in there is not the aggregation
        # task, and plain item recall would score it 1.0.
        truths = ["pencil", "beaker"]
        self.assertEqual(scoring.intersection_over_union(truths, truths), 1.0)
        padded = ["pencil", "beaker", "comet", "glacier"]
        self.assertEqual(scoring.intersection_over_union(padded, truths), 0.5)
        self.assertEqual(scoring.string_match_all(" ".join(padded), truths), 1.0)

    def test_item_matching_keeps_a_word_that_happens_to_be_an_article(self):
        # SQuAD normalization deletes "a"/"an"/"the". Right for finding a code inside a
        # sentence, wrong for a word-extraction task whose items *are* words.
        self.assertEqual(scoring.intersection_over_union(["the", "beaker"],
                                                         ["the", "beaker"]), 1.0)
        self.assertEqual(scoring.intersection_over_union(["a"], ["beaker"]), 0.0)

    def test_token_f1_is_the_graded_overlap(self):
        # pred normalizes to {secret, code, is, 7q4m91} (article dropped); truth {7q4m91}.
        # precision 1/4, recall 1/1 -> F1 = 2*(0.25*1)/1.25 = 0.4.
        self.assertAlmostEqual(scoring.token_f1("the secret code is 7Q4M91", ["7Q4M91"]), 0.4)

    def test_distractor_rate_is_none_when_nothing_was_planted(self):
        # "No decoys planted" and "no decoys returned" are different findings; averaging the
        # first as 0.0 would report a precision figure for a task that never measured one.
        self.assertIsNone(scoring.distractor_rate("anything", []))
        self.assertEqual(scoring.distractor_rate("saw AAA", ["AAA", "ZZZ"]), 0.5)

    def test_split_items_handles_the_shapes_models_actually_answer_in(self):
        self.assertEqual(
            scoring.split_items("1. apple, banana and cherry\n- date"),
            ["apple", "banana", "cherry", "date"],
        )

    def test_empty_prediction_scores_zero_without_crashing(self):
        self.assertFalse(scoring.exact_match("", ["7Q4M91"]))
        self.assertEqual(scoring.string_match_all("", ["7Q4M91"]), 0.0)
        self.assertEqual(scoring.token_f1("", ["7Q4M91"]), 0.0)


# ---------------------------------------------------------------------------
# Scoring: lexical and divergence
# ---------------------------------------------------------------------------
class TestLexicalScoring(unittest.TestCase):
    def test_lexical_metrics_keep_articles_unlike_squad(self):
        """The regression this guards: ROUGE exists to notice a rewording, and the SQuAD
        tokenizer drops "a"/"an"/"the", which makes the two sentences below identical."""
        pred, ref = "the cat sat on a mat", "the cat sat on the mat"
        self.assertEqual(scoring.tokens(pred), scoring.tokens(ref))  # SQuAD: indistinguishable
        self.assertNotEqual(scoring.lexical_tokens(pred), scoring.lexical_tokens(ref))
        for score in (scoring.rouge_n(pred, ref, 1), scoring.rouge_l(pred, ref)):
            self.assertLess(score, 1.0)

    def test_identical_text_scores_one_everywhere(self):
        text = "Energy is conserved; it only changes form."
        self.assertEqual(scoring.rouge_n(text, text, 1), 1.0)
        self.assertEqual(scoring.rouge_n(text, text, 2), 1.0)
        self.assertEqual(scoring.rouge_l(text, text), 1.0)
        self.assertEqual(scoring.chrf(text, text), 1.0)

    def test_disjoint_text_scores_zero(self):
        self.assertEqual(scoring.rouge_n("alpha beta", "gamma delta", 1), 0.0)
        self.assertEqual(scoring.rouge_l("alpha beta", "gamma delta"), 0.0)

    def test_rouge_l_rewards_order_where_rouge_1_does_not(self):
        # Same bag of words, reversed. ROUGE-1 cannot tell them apart; ROUGE-L can.
        forward, backward = "a b c d", "d c b a"
        self.assertEqual(scoring.rouge_n(forward, backward, 1), 1.0)
        self.assertLess(scoring.rouge_l(forward, backward), 1.0)

    def test_chrf_still_scores_a_near_miss_that_rouge_calls_a_total_miss(self):
        # One character apart: token metrics see two different tokens, chrF sees "almost".
        self.assertEqual(scoring.rouge_n("7Q4M91", "7Q4M92", 1), 0.0)
        self.assertGreater(scoring.chrf("7Q4M91", "7Q4M92"), 0.5)


class TestDivergentTokenMetrics(unittest.TestCase):
    def test_fdt_is_the_length_of_the_agreeing_prefix(self):
        self.assertEqual(scoring.first_divergent_token([1, 2, 3, 4], [1, 2, 9, 4]), 2)
        self.assertEqual(scoring.first_divergent_token([1, 2, 3], [1, 2, 3]), 3)
        self.assertEqual(scoring.first_divergent_token([1, 2, 3], [9, 2, 3]), 0)

    def test_identical_sequences_have_no_divergence(self):
        self.assertEqual(
            scoring.divergent_tokens([1, 2, 3], [1, 2, 3]),
            {"fdt": 3, "fdt_norm": 1.0, "sdt": 0, "sdt_norm": 0.0},
        )

    def test_a_short_answer_is_divergent_for_its_missing_tail(self):
        # Stopping early is a divergence for every remaining position, not a silent match.
        self.assertEqual(
            scoring.divergent_tokens([1, 2, 3, 4, 5], [1, 2]),
            {"fdt": 2, "fdt_norm": 0.4, "sdt": 3, "sdt_norm": 0.6},
        )

    def test_an_empty_reference_reports_nothing_rather_than_zero(self):
        self.assertEqual(
            scoring.divergent_tokens([], [1, 2]),
            {"fdt": None, "fdt_norm": None, "sdt": None, "sdt_norm": None},
        )

    def test_fdt_separates_a_paraphrase_from_an_identical_answer(self):
        """The reason FDT is reported next to similarity: two answers that diverge at the
        first token and still mean the same thing are NOT a lossless configuration."""
        scores = accuracy.score_fidelity(
            "Energy is always conserved in a closed system.",
            "In a closed system, energy is always conserved.",
        )
        self.assertFalse(scores["identical"])
        self.assertEqual(scores["fdt"], 0)
        self.assertGreater(scores["rouge1"], 0.9)  # same words, so the bag-of-words agrees


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
def _spec(task, depth=0.5, sample=0, seed=123, context=8000, options=None):
    return tasks.build_spec(
        task=task, sample=sample, depth=depth if tasks.is_depth_swept(task) else None,
        haystack=corpus_text(), seed=seed, context_tokens=context,
        output_tokens=64, options=options or {},
    )


class TestTaskRegistry(unittest.TestCase):
    def test_the_ruler_behaviours_are_all_covered(self):
        """RULER's point is that retrieval, multi-hop tracing and aggregation fail
        independently. A suite with only `niah_single` cannot see two of the three."""
        for task in ("niah_single", "niah_multikey", "niah_multivalue", "niah_multiquery",
                     "vt", "cwe", "fwe", "qa"):
            self.assertIn(task, tasks.RETRIEVAL_TASKS, task)
        self.assertEqual(set(tasks.GENERATION_TASKS), {"summary", "fact_sheet"})

    def test_an_unknown_task_is_an_error_not_a_skipped_measurement(self):
        self.assertFalse(tasks.is_known("niah_sngle"))
        with self.assertRaisesRegex(KeyError, "unknown accuracy task"):
            tasks.suite_of("niah_sngle")

    def test_only_the_single_needle_tasks_are_depth_swept(self):
        # A probe that spreads eight needles by construction has no depth to put in a column.
        self.assertTrue(tasks.is_depth_swept("niah_single"))
        self.assertFalse(tasks.is_depth_swept("niah_multivalue"))
        self.assertFalse(tasks.is_depth_swept("cwe"))


class TestPlantedValues(unittest.TestCase):
    def test_values_are_deterministic_and_corpus_disjoint(self):
        import random

        haystack = corpus_text().upper()
        for seed in range(50):
            value = tasks.draw_value(random.Random(seed), corpus_text())
            self.assertNotIn(value.upper(), haystack)
            self.assertEqual(value, tasks.draw_value(random.Random(seed), corpus_text()))

    def test_the_same_config_reproduces_the_same_probe(self):
        self.assertEqual(_spec("niah_single").truths, _spec("niah_single").truths)
        self.assertEqual(_spec("vt").inserts, _spec("vt").inserts)

    def test_every_sample_and_depth_plants_a_different_value(self):
        # Otherwise a profile could score a hit by memorizing one answer across the sweep.
        values = [
            _spec("niah_single", depth=d, sample=s).truths[0]
            for d in (0.0, 0.5, 1.0) for s in range(3)
        ]
        self.assertEqual(len(set(values)), len(values))

    def test_a_different_context_length_plants_different_values(self):
        self.assertNotEqual(
            _spec("niah_single", context=8000).truths,
            _spec("niah_single", context=32000).truths,
        )


class TestRetrievalTaskShapes(unittest.TestCase):
    def test_multikey_plants_decoys_distinct_from_the_target(self):
        spec = _spec("niah_multikey", options={"num_distractors": 4})
        self.assertEqual(len(spec.truths), 1)
        self.assertEqual(len(spec.distractors), 4)
        self.assertEqual(len(spec.inserts), 5)
        self.assertNotIn(spec.truths[0], spec.distractors)

    def test_multivalue_and_multiquery_ask_for_every_planted_value(self):
        for task, option in (("niah_multivalue", "num_values"),
                             ("niah_multiquery", "num_queries")):
            spec = _spec(task, options={option: 5})
            self.assertEqual(len(spec.truths), 5, task)
            self.assertEqual(len(spec.inserts), 5, task)
            self.assertEqual(len(set(spec.truths)), 5, task)

    def test_vt_chains_every_target_name_to_one_seeded_value(self):
        spec = _spec("vt", options={"chain_length": 4, "num_chains": 3})
        self.assertEqual(len(spec.truths), 4)          # the whole target chain
        self.assertEqual(len(spec.distractors), 8)     # the two decoy chains
        self.assertEqual(len(spec.inserts), 12)        # every assignment statement
        # The seeded value appears in exactly one statement; the rest copy a variable.
        seeds = [s for _d, s in spec.inserts if "the value of" not in s]
        self.assertEqual(len(seeds), 3)                # one seed per chain
        self.assertIn(spec.truths[0], seeds[0] + seeds[1] + seeds[2])

    def test_aggregation_tasks_replace_the_haystack_rather_than_hide_in_it(self):
        for task in ("cwe", "fwe"):
            spec = _spec(task)
            self.assertEqual(spec.inserts, [], task)
            self.assertIn("WORD LIST", spec.filler or "", task)
            self.assertEqual(spec.scorer, tasks.SCORER_IOU, task)

    def test_qa_answers_with_a_word_not_a_code(self):
        spec = _spec("qa")
        self.assertEqual(spec.scorer, tasks.SCORER_MATCH_PART)
        # A natural-language answer, so the model has to understand the paragraph rather than
        # copy the one string in the context that looks unlike the rest.
        self.assertTrue(spec.truths[0].isalpha())
        self.assertIn(spec.truths[0], spec.inserts[0][1])

    def test_inserts_are_spread_across_the_transcript_not_clustered(self):
        depths = sorted(d for d, _ in _spec("niah_multivalue", options={"num_values": 4}).inserts)
        self.assertLess(depths[0], 0.35)
        self.assertGreater(depths[-1], 0.6)


class TestGenerationTaskShapes(unittest.TestCase):
    def test_summary_has_no_ground_truth_by_design(self):
        # Its whole measurement is the fidelity comparison against the baseline profile.
        spec = _spec("summary")
        self.assertEqual(spec.truths, [])
        self.assertEqual(spec.inserts, [])
        self.assertEqual(spec.suite, accuracy.SUITE_GENERATION)

    def test_each_sample_draws_a_different_real_task(self):
        questions = {_spec("summary", sample=s).question for s in range(5)}
        self.assertGreater(len(questions), 1)

    def test_fact_sheet_carries_its_own_ground_truth(self):
        spec = _spec("fact_sheet", options={"num_facts": 6})
        self.assertEqual(len(spec.truths), 6)
        self.assertEqual(len(spec.inserts), 6)


class TestAggregationGroundTruthSurvivesSlicing(unittest.TestCase):
    """`cwe`/`fwe` size their word list by slicing a repeated block to an exact token count.
    If that slice could reorder the word frequencies, the recorded answer would be wrong --
    which is a silently *incorrect* accuracy number, not a missing one."""

    def _rendered_word_counts(self, spec, target):
        prompt, tokens_seen = build_probe_prompt(
            FakeTokenizer(), target, spec.inserts, spec.question, spec.filler
        )
        self.assertEqual(tokens_seen, target)
        body = prompt.split("WORD LIST", 1)[1].split("---")[0]
        return Counter(word for word in body.split() if word.isalpha())

    def test_cwe_top_words_are_exactly_the_recorded_truths(self):
        spec = _spec("cwe", options={"num_target_words": 10})
        counts = self._rendered_word_counts(spec, 9000)
        top = {word for word, _ in counts.most_common(10)}
        self.assertEqual(top, set(spec.truths))

    def test_fwe_top_words_are_exactly_the_recorded_truths(self):
        spec = _spec("fwe", options={"top_k": 3})
        counts = self._rendered_word_counts(spec, 9000)
        self.assertEqual([word for word, _ in counts.most_common(3)], spec.truths)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
class TestProbePrompt(unittest.TestCase):
    def test_every_task_lands_on_an_exact_token_count_with_its_inserts_intact(self):
        for task in tasks.KNOWN_TASKS:
            spec = _spec(task)
            for target in (2000, 6000):
                prompt, seen = build_probe_prompt(
                    FakeTokenizer(), target, spec.inserts, spec.question, spec.filler
                )
                self.assertEqual(seen, target, f"{task} @ {target}")
                for _depth, sentence in spec.inserts:
                    self.assertIn(sentence, prompt, f"{task}: {sentence[:40]}")

    def test_plants_the_question_not_the_summarization_task(self):
        prompt, _ = build_probe_prompt(
            FakeTokenizer(), 500, [(0.5, "the secret access code is 7Q4M91")],
            "What is the access code? Answer with only the code.",
        )
        self.assertIn("the secret access code is 7Q4M91", prompt)
        self.assertIn("What is the access code", prompt)
        # The throughput summarization task must be gone -- this is a probe prompt.
        self.assertNotIn("Summarize the lesson", prompt)
        self.assertNotIn(_USER_SUFFIX.strip(), prompt)

    def test_depth_moves_an_insert_from_top_to_bottom(self):
        needle = "MARKERNEEDLEXYZ planted here"
        top, _ = build_probe_prompt(FakeTokenizer(), 2000, [(0.0, needle)], "Where?")
        bottom, _ = build_probe_prompt(FakeTokenizer(), 2000, [(1.0, needle)], "Where?")
        self.assertLess(top.index("MARKERNEEDLEXYZ") / len(top), 0.35)
        self.assertGreater(bottom.index("MARKERNEEDLEXYZ") / len(bottom), 0.6)

    def test_several_inserts_keep_their_relative_order(self):
        inserts = [(0.8, "MARKER_LAST here"), (0.1, "MARKER_FIRST here"),
                   (0.5, "MARKER_MID here")]
        prompt, _ = build_probe_prompt(FakeTokenizer(), 4000, inserts, "Where?")
        self.assertLess(prompt.index("MARKER_FIRST"), prompt.index("MARKER_MID"))
        self.assertLess(prompt.index("MARKER_MID"), prompt.index("MARKER_LAST"))

    def test_no_inserts_is_a_plain_transcript_with_the_task(self):
        prompt, seen = build_probe_prompt(FakeTokenizer(), 1000, [], "Summarize it.")
        self.assertEqual(seen, 1000)
        self.assertIn("Summarize it.", prompt)

    def test_a_custom_filler_replaces_the_classroom_dialog(self):
        prompt, seen = build_probe_prompt(
            FakeTokenizer(), 1000, [], "Count them.", filler="alpha beta gamma\n"
        )
        self.assertEqual(seen, 1000)
        self.assertIn("alpha beta gamma", prompt)
        self.assertNotIn("TEACHER:", prompt)

    def test_converges_through_boundary_token_drift_with_many_inserts(self):
        spec = _spec("vt", options={"chain_length": 4, "num_chains": 4})
        self.assertEqual(len(spec.inserts), 16)
        _prompt, seen = build_probe_prompt(
            BoundaryDriftTokenizer(), 6000, spec.inserts, spec.question
        )
        self.assertEqual(seen, 6000)

    def test_a_depth_outside_the_unit_interval_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "depth must be in"):
            build_probe_prompt(FakeTokenizer(), 400, [(1.5, "needle")], "q?")


class TestProbePromptTolerance(unittest.TestCase):
    """The contract that stopped a real run dying over one token.

    A probe has two tokenizer seams per insert -- thirty-four on a `vt` probe -- and the size
    correction moves one number against the sum of all that drift, so it can orbit the target
    by a token indefinitely. Throughput divides by the token count and needs it exact; a
    retrieval verdict does not, so the probe builder takes a tolerance and the count it
    actually used is what gets recorded.
    """

    class UnreachableTokenizer(FakeTokenizer):
        """A rendered prompt always measures an ODD number of tokens, so an even target can
        never be hit however the filler is sized.

        A constant offset would not do: the correction loop subtracts it on the next round and
        lands exactly. This reproduces the real failure, where merges at the seams move the
        total by a token in a way the loop's arithmetic cannot anticipate, and it orbits the
        target forever.
        """

        def encode(self, text, add_special_tokens=True):
            ids = super().encode(text, add_special_tokens=add_special_tokens)
            if "<system>" in text and len(ids) % 2 == 0:  # the fully rendered prompt
                ids = ids + [0]
            return ids

    def test_exact_is_still_the_default(self):
        # Nothing silently loosens: a caller that does not ask for tolerance gets the old
        # contract, an exact prompt or an exception.
        _prompt, tokens = build_probe_prompt(
            FakeTokenizer(), 2000, [(0.5, "needle here")], "q?"
        )
        self.assertEqual(tokens, 2000)

    def test_an_unreachable_target_raises_without_a_tolerance(self):
        with self.assertRaisesRegex(ValueError, "within 0 token"):
            build_probe_prompt(self.UnreachableTokenizer(), 2000, [(0.5, "needle")], "q?")

    def test_a_tolerance_accepts_the_near_miss_instead_of_losing_the_probe(self):
        _prompt, tokens = build_probe_prompt(
            self.UnreachableTokenizer(), 2000, [(0.5, "needle")], "q?", tolerance=16
        )
        self.assertNotEqual(tokens, 2000)
        self.assertLessEqual(abs(tokens - 2000), 16)

    def test_the_error_names_the_closest_it_reached(self):
        # So the message says how far off it was, not just that it failed.
        with self.assertRaisesRegex(ValueError, r"closest was \d+"):
            build_probe_prompt(self.UnreachableTokenizer(), 2000, [(0.5, "needle")], "q?")

    def test_the_child_uses_a_tolerance_and_the_throughput_path_does_not(self):
        from components.llm.context_bench import trial_runner

        self.assertGreater(trial_runner.PROBE_TOKEN_TOLERANCE, 0)
        # The throughput builder has no tolerance parameter at all -- its numbers are
        # per-token rates, so a token really does matter there.
        import inspect

        from components.llm.context_bench import context_builder

        self.assertNotIn(
            "tolerance",
            inspect.signature(context_builder.build_benchmark_prompt).parameters,
        )

    def test_one_unbuildable_probe_does_not_abort_the_rest(self):
        # The failure that cost a whole case twice: a single probe's geometry failing threw
        # away the model load and every probe that had already run.
        import inspect

        from components.llm.context_bench import trial_runner

        source = inspect.getsource(trial_runner._run_accuracy_probes)
        self.assertIn("continue", source)
        self.assertIn("skipping probe", source)


# ---------------------------------------------------------------------------
# Probe sets, scoring dispatch and aggregation
# ---------------------------------------------------------------------------
def _accuracy_cfg(**overrides):
    cfg = {
        "suites": [accuracy.SUITE_RETRIEVAL, accuracy.SUITE_GENERATION],
        "baseline_profile": "paged_min",
        "seed": 123,
        "embedding_model": None,
        accuracy.SUITE_RETRIEVAL: {
            "tasks": ["niah_single", "vt"], "depths": [0.0, 1.0], "samples": 2,
            "output_tokens": 64, "options": {},
        },
        accuracy.SUITE_GENERATION: {
            "tasks": ["summary", "fact_sheet"], "depths": [0.5], "samples": 2,
            "output_tokens": 256, "options": {},
        },
    }
    cfg.update(overrides)
    return cfg


class TestProbeSets(unittest.TestCase):
    def test_depth_swept_tasks_multiply_by_depth_and_the_rest_do_not(self):
        specs = accuracy.build_probe_specs(_accuracy_cfg(), 8000)
        counts = Counter(s.task for s in specs)
        self.assertEqual(counts["niah_single"], 4)   # 2 depths x 2 samples
        self.assertEqual(counts["vt"], 2)            # 2 samples, no depth sweep
        self.assertEqual(counts["summary"], 2)
        self.assertEqual(counts["fact_sheet"], 2)

    def test_every_probe_has_a_unique_coordinate(self):
        # The coordinate is what pairs an answer back to its spec, and a generation probe to
        # the baseline profile's answer for the same work. A collision would cross the wires.
        specs = accuracy.build_probe_specs(_accuracy_cfg(), 8000)
        self.assertEqual(len({s.coordinate for s in specs}), len(specs))

    def test_a_suite_left_out_runs_no_probes_for_it(self):
        cfg = _accuracy_cfg(suites=[accuracy.SUITE_RETRIEVAL])
        cfg[accuracy.SUITE_GENERATION] = None
        specs = accuracy.build_probe_specs(cfg, 8000)
        self.assertTrue(all(s.suite == accuracy.SUITE_RETRIEVAL for s in specs))

    def test_each_probe_carries_its_own_decode_budget(self):
        specs = accuracy.build_probe_specs(_accuracy_cfg(), 8000)
        by_task = {s.task: s.output_tokens for s in specs}
        self.assertEqual(by_task["niah_single"], 64)
        self.assertEqual(by_task["summary"], 256)


class TestScoringDispatch(unittest.TestCase):
    def test_the_task_picks_its_ruler_scorer(self):
        iou_spec = _spec("cwe")
        # Over-answering is punished for an aggregation task...
        over = accuracy.score_retrieval(", ".join(iou_spec.truths + ["extra", "words"]),
                                        iou_spec)
        self.assertLess(over["recall"], 1.0)
        # ...but a NIAH answer wrapped in a sentence is still a full hit.
        niah = _spec("niah_single")
        self.assertEqual(
            accuracy.score_retrieval(f"the code is {niah.truths[0]}.", niah)["recall"], 1.0
        )

    def test_a_decoy_in_the_answer_is_reported_separately_from_a_miss(self):
        spec = _spec("niah_multikey")
        confused = accuracy.score_retrieval(f"the code is {spec.distractors[0]}", spec)
        self.assertEqual(confused["recall"], 0.0)
        self.assertGreater(confused["distractor_rate"], 0.0)

    def test_grounding_is_none_for_the_open_ended_task(self):
        self.assertIsNone(accuracy.score_grounding("anything", _spec("summary"))["fact_coverage"])
        spec = _spec("fact_sheet", options={"num_facts": 4})
        self.assertEqual(
            accuracy.score_grounding(" ".join(spec.truths[:2]), spec)["fact_coverage"], 0.5
        )


class TestAggregation(unittest.TestCase):
    def test_rates_are_means_over_the_probes(self):
        scores = [
            {"exact_match": True, "recall": 1.0, "token_f1": 1.0, "distractor_rate": 0.0},
            {"exact_match": False, "recall": 1.0, "token_f1": 0.4, "distractor_rate": 0.0},
            {"exact_match": False, "recall": 0.0, "token_f1": 0.0, "distractor_rate": 1.0},
            {"exact_match": True, "recall": 1.0, "token_f1": 1.0, "distractor_rate": 0.0},
        ]
        agg = accuracy.aggregate(scores, accuracy.RETRIEVAL_METRICS)
        self.assertEqual(agg["probe_count"], 4)
        self.assertEqual(agg["exact_match_rate"], 0.5)
        self.assertEqual(agg["recall_rate"], 0.75)
        self.assertEqual(agg["token_f1"], 0.6)
        self.assertEqual(agg["distractor_rate"], 0.25)

    def test_a_metric_missing_from_some_probes_averages_over_the_rest(self):
        # `fact_coverage` exists only on `fact_sheet` probes; averaging the `summary` probes
        # in as 0.0 would report a coverage figure for a task that has no ground truth.
        agg = accuracy.aggregate(
            [{"fact_coverage": 0.5}, {"fact_coverage": None}, {"fact_coverage": 1.0}],
            ("fact_coverage",),
        )
        self.assertEqual(agg["probe_count"], 3)
        self.assertEqual(agg["fact_coverage"], 0.75)

    def test_a_metric_missing_everywhere_is_none_not_zero(self):
        agg = accuracy.aggregate([{"recall": 1.0}], accuracy.GENERATION_METRICS)
        self.assertIsNone(agg["similarity"])

    def test_empty_is_reported_not_zeroed(self):
        # "Nothing measured" and "nothing retrieved" are different findings, as elsewhere.
        agg = accuracy.aggregate([], accuracy.RETRIEVAL_METRICS)
        self.assertEqual(agg["probe_count"], 0)
        self.assertIsNone(agg["exact_match_rate"])

    def test_delta_is_signed_against_the_baseline(self):
        d = accuracy.delta(
            {"exact_match_rate": 0.5, "recall_rate": 0.6},
            {"exact_match_rate": 0.8, "recall_rate": 0.9},
            ("exact_match_rate", "recall_rate"),
        )
        self.assertAlmostEqual(d["exact_match_rate_delta"], -0.3)
        self.assertAlmostEqual(d["recall_rate_delta"], -0.3)

    def test_delta_is_none_when_a_side_is_missing(self):
        d = accuracy.delta({"recall_rate": None}, {"recall_rate": 0.8}, ("recall_rate",))
        self.assertIsNone(d["recall_rate_delta"])


# ---------------------------------------------------------------------------
# The who_what_benchmark bridge (optional dependency, stubbed)
# ---------------------------------------------------------------------------
class TestWwbAdapter(unittest.TestCase):
    """`whowhatbench` is not installed in CI, so the contract tested here is the adapter's:
    read every return shape upstream has used, and degrade to a reason rather than an
    exception when the shape is unrecognized or the package is absent."""

    def tearDown(self):
        wwb_adapter._EVALUATOR_CACHE.clear()
        sys.modules.pop("whowhatbench", None)
        sys.modules.pop("whowhatbench.whowhat_metrics", None)

    def _install(self, evaluate):
        wwb_adapter._EVALUATOR_CACHE.clear()
        package = types.ModuleType("whowhatbench")
        module = types.ModuleType("whowhatbench.whowhat_metrics")

        class TextSimilarity:
            def __init__(self, model_id):
                self.model_id = model_id

            def evaluate(self, gt, prediction):
                return evaluate(gt, prediction)

        module.TextSimilarity = TextSimilarity
        package.whowhat_metrics = module
        sys.modules["whowhatbench"] = package
        sys.modules["whowhatbench.whowhat_metrics"] = module

    def test_reads_the_metrics_per_sample_pair_upstream_returns(self):
        self._install(lambda gt, pred: ({"similarity": [0.9]},
                                        {"similarity": [0.95, 0.85]}))
        result = wwb_adapter.similarity(["a", "b"], ["c", "d"])
        self.assertTrue(result.available)
        self.assertEqual(result.values, [0.95, 0.85])
        self.assertEqual(result.mean, 0.9)

    def test_reads_a_plain_column_or_a_bare_sequence(self):
        self._install(lambda gt, pred: {"similarity": [0.5, 0.7]})
        self.assertEqual(wwb_adapter.similarity(["a", "b"], ["c", "d"]).values, [0.5, 0.7])
        self._install(lambda gt, pred: [0.1, 0.2])
        self.assertEqual(wwb_adapter.similarity(["a", "b"], ["c", "d"]).values, [0.1, 0.2])

    def test_an_unrecognized_shape_degrades_instead_of_guessing(self):
        self._install(lambda gt, pred: object())
        result = wwb_adapter.similarity(["a"], ["b"])
        self.assertFalse(result.available)
        self.assertIn("not one this adapter recognizes", result.reason)

    def test_a_misaligned_result_is_refused_rather_than_zipped(self):
        # One score for two probes cannot be aligned, and aligning it wrongly would put one
        # probe's similarity on another probe's row.
        self._install(lambda gt, pred: [0.1])
        self.assertFalse(wwb_adapter.similarity(["a", "b"], ["c", "d"]).available)

    def test_an_exception_inside_wwb_does_not_end_the_run(self):
        def _boom(gt, pred):
            raise RuntimeError("boom")

        self._install(_boom)
        result = wwb_adapter.similarity(["a"], ["b"])
        self.assertFalse(result.available)
        self.assertIn("boom", result.reason)

    def test_not_installed_is_a_reason_not_a_crash(self):
        wwb_adapter._EVALUATOR_CACHE.clear()
        sys.modules.pop("whowhatbench", None)
        result = wwb_adapter.similarity(["a"], ["b"])
        self.assertFalse(result.available)
        self.assertIn("not installed", result.reason)

    def test_unpaired_inputs_are_a_programming_error(self):
        with self.assertRaisesRegex(ValueError, "paired inputs"):
            wwb_adapter.similarity(["a"], ["b", "c"])


# ---------------------------------------------------------------------------
# Case scoring and reporting (pure dict/string, no hardware)
# ---------------------------------------------------------------------------
def _answers(specs, correct):
    """Fake child output: every probe answered either correctly or not at all."""
    rows = []
    for spec in specs:
        if spec.suite == accuracy.SUITE_GENERATION:
            text = ("Energy transfer was the topic. " + " ".join(spec.truths)) if correct \
                else "Something about a lesson, vaguely."
        else:
            text = ", ".join(spec.truths) if correct else "I could not find it"
        rows.append({
            "suite": spec.suite, "task": spec.task, "depth": spec.depth,
            "sample": spec.sample, "prediction_text": text,
        })
    return rows


def _case(profile, specs, correct):
    scored, summary = benchmark._score_case_accuracy(_answers(specs, correct), specs)
    return {
        "model": "Qwen3.8-27B", "profile": profile, "context_tokens": 8000, "status": "ok",
        "pipeline_mode": "paged", "mtp": profile != "paged_min",
        "num_assistant_tokens": None if profile == "paged_min" else 6,
        "accuracy": summary, "_probes": scored, "_model_dir": None,
    }


class TestCaseScoring(unittest.TestCase):
    def setUp(self):
        self.cfg = _accuracy_cfg()
        self.specs = accuracy.build_probe_specs(self.cfg, 8000)

    def test_probes_are_matched_to_their_spec_and_scored_per_suite(self):
        case = _case("paged_min", self.specs, correct=True)
        retrieval = case["accuracy"][accuracy.SUITE_RETRIEVAL]
        self.assertEqual(retrieval["recall_rate"], 1.0)
        self.assertEqual(set(retrieval["per_task"]), {"niah_single", "vt"})
        # The depth matrix covers the depth-swept tasks only.
        self.assertEqual(set(retrieval["per_depth"]), {"0.00", "1.00"})

    def test_an_answer_with_no_matching_spec_is_dropped_not_misattributed(self):
        stray = _answers(self.specs, correct=True)
        stray.append({"suite": "retrieval", "task": "niah_single", "depth": 0.42,
                      "sample": 99, "prediction_text": "junk"})
        scored, summary = benchmark._score_case_accuracy(stray, self.specs)
        self.assertEqual(len(scored), len(self.specs))
        self.assertEqual(summary[accuracy.SUITE_RETRIEVAL]["recall_rate"], 1.0)

    def test_the_probe_row_carries_the_ground_truth_it_was_scored_against(self):
        case = _case("paged_min", self.specs, correct=True)
        row = next(p for p in case["_probes"] if p["task"] == "niah_single")
        self.assertIn(row["truth"], row["prediction"])
        self.assertEqual(row["recall"], 1.0)


class TestGenerationFidelityFill(unittest.TestCase):
    """The fidelity scores are cross-case by construction -- the reference is another
    profile's answer -- so they are filled at report time, and that is where the pairing can
    go wrong."""

    def setUp(self):
        self.cfg = _accuracy_cfg()
        self.specs = accuracy.build_probe_specs(self.cfg, 8000)
        self.settings = {"accuracy": self.cfg}

    def test_a_matching_profile_scores_perfect_fidelity_against_the_baseline(self):
        cases = [_case("paged_min", self.specs, True), _case("mtp_k6", self.specs, True)]
        benchmark._fill_generation_fidelity(cases, self.settings)
        for case in cases:
            generation = case["accuracy"][accuracy.SUITE_GENERATION]
            self.assertEqual(generation["rouge_l"], 1.0)
            self.assertEqual(generation["identical_rate"], 1.0)
            self.assertEqual(generation["sdt_norm"], 0.0)

    def test_a_drifting_profile_is_graded_not_just_flagged(self):
        cases = [_case("paged_min", self.specs, True), _case("mtp_k6", self.specs, False)]
        benchmark._fill_generation_fidelity(cases, self.settings)
        drifted = cases[1]["accuracy"][accuracy.SUITE_GENERATION]
        self.assertEqual(drifted["identical_rate"], 0.0)   # the old sha256 check's answer
        self.assertGreater(drifted["sdt_norm"], 0.0)       # ...and how far it drifted
        self.assertLess(drifted["rouge_l"], 1.0)
        self.assertEqual(drifted["fact_coverage"], 0.0)

    def test_the_baseline_compares_against_itself(self):
        cases = [_case("paged_min", self.specs, True)]
        benchmark._fill_generation_fidelity(cases, self.settings)
        generation = cases[0]["accuracy"][accuracy.SUITE_GENERATION]
        self.assertEqual(generation["identical_rate"], 1.0)
        self.assertEqual(generation["fdt_norm"], 1.0)

    def test_running_it_twice_does_not_accumulate(self):
        # write_reports runs after every case, so this is re-entered on every pass.
        cases = [_case("paged_min", self.specs, True), _case("mtp_k6", self.specs, False)]
        benchmark._fill_generation_fidelity(cases, self.settings)
        first = dict(cases[1]["accuracy"][accuracy.SUITE_GENERATION])
        benchmark._fill_generation_fidelity(cases, self.settings)
        self.assertEqual(cases[1]["accuracy"][accuracy.SUITE_GENERATION], first)

    def test_without_a_baseline_case_nothing_is_invented(self):
        cases = [_case("mtp_k6", self.specs, False)]
        benchmark._fill_generation_fidelity(cases, self.settings)
        self.assertIsNone(cases[0]["accuracy"][accuracy.SUITE_GENERATION]["rouge_l"])

    def test_a_different_context_is_never_used_as_a_reference(self):
        # A profile's answers are only comparable on the identical prompt, and the prompt is
        # seeded from the context length.
        other = _case("paged_min", self.specs, True)
        other["context_tokens"] = 32000
        cases = [other, _case("mtp_k6", self.specs, False)]
        benchmark._fill_generation_fidelity(cases, self.settings)
        self.assertIsNone(cases[1]["accuracy"][accuracy.SUITE_GENERATION]["rouge_l"])


class TestAccuracyReport(unittest.TestCase):
    def setUp(self):
        self.cfg = _accuracy_cfg()
        self.specs = accuracy.build_probe_specs(self.cfg, 8000)
        self.settings = {"accuracy": self.cfg}
        self.cases = [
            _case("paged_min", self.specs, True),
            _case("mtp_k6", self.specs, False),
        ]
        benchmark._fill_generation_fidelity(self.cases, self.settings)
        benchmark._fill_accuracy_deltas(self.cases, self.settings)

    def test_retrieval_leaderboard_ranks_by_recall(self):
        ranked = benchmark._accuracy_leaderboard(self.cases, accuracy.SUITE_RETRIEVAL)
        self.assertEqual([c["profile"] for c in ranked], ["paged_min", "mtp_k6"])

    def test_generation_leaderboard_still_ranks_without_the_embedding_column(self):
        # who_what_benchmark is optional; requiring its column would leave the whole
        # generation finding silently empty on the common install.
        ranked = benchmark._accuracy_leaderboard(self.cases, accuracy.SUITE_GENERATION)
        self.assertEqual([c["profile"] for c in ranked], ["paged_min", "mtp_k6"])

    def test_deltas_are_filled_against_the_named_baseline(self):
        deltas = self.cases[1]["accuracy"][accuracy.SUITE_RETRIEVAL]["deltas"]
        self.assertAlmostEqual(deltas["recall_rate_delta"], -1.0)
        self.assertAlmostEqual(
            self.cases[0]["accuracy"][accuracy.SUITE_RETRIEVAL]["deltas"]["recall_rate_delta"],
            0.0,
        )

    def test_the_retrieval_section_renders_a_per_task_table_and_a_depth_sweep(self):
        text = "\n".join(benchmark._accuracy_section(self.cases, 8000, self.settings))
        self.assertIn("Retrieval Accuracy (RULER)", text)
        self.assertIn("niah_single", text)
        self.assertIn("vt", text)
        self.assertIn("Depth sweep", text)
        self.assertIn("d=0.00", text)
        self.assertIn("-100pp", text)   # mtp_k6's recall gap to the baseline

    def test_the_generation_section_renders_every_fidelity_family(self):
        text = "\n".join(benchmark._accuracy_section(self.cases, 8000, self.settings))
        self.assertIn("Generation Fidelity", text)
        for column in ("Similarity", "ROUGE-L", "chrF", "FDT", "SDT norm", "Identical",
                       "Coverage"):
            self.assertIn(column, text)

    def test_a_disabled_similarity_column_says_so(self):
        text = "\n".join(benchmark._accuracy_section(self.cases, 8000, self.settings))
        self.assertIn("Similarity not measured", text)

    def test_a_suite_that_did_not_run_renders_no_section(self):
        settings = {"accuracy": _accuracy_cfg(suites=[accuracy.SUITE_RETRIEVAL])}
        text = "\n".join(benchmark._accuracy_section(self.cases, 8000, settings))
        self.assertIn("Retrieval Accuracy", text)
        self.assertNotIn("Generation Fidelity", text)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
def _accuracy_args(**overrides):
    values = {"accuracy": True, "accuracy_suites": None, "accuracy_tasks": None,
              "accuracy_depths": None, "accuracy_samples": None}
    values.update(overrides)
    return SimpleNamespace(**values)


class TestAccuracyConfigResolution(unittest.TestCase):
    PROFILES = [{"name": "paged_min"}, {"name": "mtp_k3"}]

    def _resolve(self, section, **arg_overrides):
        cfg = SimpleNamespace(accuracy=section)
        return benchmark._resolve_accuracy(cfg, _accuracy_args(**arg_overrides), self.PROFILES)

    def test_off_by_default(self):
        cfg = SimpleNamespace(accuracy={"suites": ["retrieval"]})
        self.assertIsNone(
            benchmark._resolve_accuracy(cfg, _accuracy_args(accuracy=False), self.PROFILES)
        )

    def test_a_minimal_section_defaults_to_the_full_suite(self):
        resolved = self._resolve({"suites": ["retrieval"], "baseline_profile": "paged_min"})
        self.assertEqual(
            resolved[accuracy.SUITE_RETRIEVAL]["tasks"], list(tasks.RETRIEVAL_TASKS)
        )
        self.assertIsNone(resolved[accuracy.SUITE_GENERATION])

    def test_depths_are_sorted_and_deduplicated(self):
        resolved = self._resolve(
            {"suites": ["retrieval"], "retrieval": {"depths": [1.0, 0.0, 0.5, 0.0]}}
        )
        self.assertEqual(resolved[accuracy.SUITE_RETRIEVAL]["depths"], [0.0, 0.5, 1.0])

    def test_generation_gets_a_longer_decode_budget_than_retrieval(self):
        # Scoring prose at 24 tokens would measure truncation rather than fidelity.
        resolved = self._resolve({"suites": ["retrieval", "generation"]})
        self.assertGreater(
            resolved[accuracy.SUITE_GENERATION]["output_tokens"],
            resolved[accuracy.SUITE_RETRIEVAL]["output_tokens"],
        )

    def test_missing_section_is_rejected(self):
        with self.assertRaisesRegex(SystemExit, "needs an `accuracy` section"):
            self._resolve(None)

    def test_an_unknown_suite_is_rejected(self):
        with self.assertRaisesRegex(SystemExit, "accuracy.suites"):
            self._resolve({"suites": ["retreival"]})

    def test_an_unknown_task_names_the_ones_that_exist(self):
        with self.assertRaisesRegex(SystemExit, "unknown task"):
            self._resolve({"suites": ["retrieval"], "retrieval": {"tasks": ["niah_sngle"]}})

    def test_a_task_from_the_wrong_suite_is_rejected_in_the_config(self):
        with self.assertRaisesRegex(SystemExit, "selects no retrieval task"):
            self._resolve({"suites": ["retrieval"], "retrieval": {"tasks": ["summary"]}})

    def test_a_cli_filter_that_empties_a_suite_drops_it_instead_of_failing(self):
        # `--accuracy-tasks` spans every suite, so narrowing to retrieval tasks leaves
        # generation with nothing. That means "not this suite this time", not "refuse the run".
        resolved = self._resolve(
            {"suites": ["retrieval", "generation"]}, accuracy_tasks=["vt", "niah_multikey"]
        )
        self.assertEqual(resolved["suites"], [accuracy.SUITE_RETRIEVAL])
        self.assertIsNone(resolved[accuracy.SUITE_GENERATION])
        self.assertEqual(resolved[accuracy.SUITE_RETRIEVAL]["tasks"], ["vt", "niah_multikey"])

    def test_a_cli_filter_that_empties_every_suite_is_still_an_error(self):
        with self.assertRaisesRegex(SystemExit, "selects no task in any suite"):
            self._resolve({"suites": ["retrieval"]}, accuracy_tasks=["summary"])

    def test_depth_outside_unit_interval_is_rejected(self):
        with self.assertRaisesRegex(SystemExit, "fractions in .0.0, 1.0."):
            self._resolve({"suites": ["retrieval"], "retrieval": {"depths": [0.0, 1.5]}})

    def test_baseline_not_in_the_run_is_rejected(self):
        with self.assertRaisesRegex(SystemExit, "baseline_profile"):
            self._resolve({"suites": ["retrieval"], "baseline_profile": "does_not_exist"})

    def test_a_typo_in_task_options_is_rejected_not_silently_dropped(self):
        # Otherwise the run reports the defaults as though they had been configured.
        with self.assertRaisesRegex(SystemExit, "not a retrieval task"):
            self._resolve({"suites": ["retrieval"],
                           "retrieval": {"options": {"niah_multkey": {"num_distractors": 2}}}})

    def test_unknown_keys_name_the_accepted_shape(self):
        with self.assertRaisesRegex(SystemExit, "unknown key"):
            self._resolve({"suites": ["retrieval"], "nonsense": 1})
        with self.assertRaisesRegex(SystemExit, "unknown key"):
            self._resolve({"suites": ["retrieval"], "retrieval": {"nonsense": 1}})

    def test_cli_overrides_apply_across_the_suites_being_run(self):
        resolved = self._resolve(
            {"suites": ["retrieval", "generation"]},
            accuracy_tasks=["vt", "summary"], accuracy_depths=[0.2, 0.8], accuracy_samples=5,
        )
        self.assertEqual(resolved[accuracy.SUITE_RETRIEVAL]["tasks"], ["vt"])
        self.assertEqual(resolved[accuracy.SUITE_GENERATION]["tasks"], ["summary"])
        self.assertEqual(resolved[accuracy.SUITE_RETRIEVAL]["depths"], [0.2, 0.8])
        self.assertEqual(resolved[accuracy.SUITE_GENERATION]["samples"], 5)

    def test_cli_suite_selection_overrides_the_config(self):
        resolved = self._resolve(
            {"suites": ["retrieval", "generation"]}, accuracy_suites=["generation"]
        )
        self.assertEqual(resolved["suites"], [accuracy.SUITE_GENERATION])
        self.assertIsNone(resolved[accuracy.SUITE_RETRIEVAL])


# ---------------------------------------------------------------------------
# The load-bearing invariant: accuracy must not disturb the throughput headers
# ---------------------------------------------------------------------------
class TestThroughputHeadersAreUntouched(unittest.TestCase):
    ACCURACY_ONLY = {
        "suite", "task", "depth", "sample", "truth", "distractors", "prediction",
        "accuracy", *accuracy.PROBE_SCORE_FIELDS,
    }

    def test_case_fields_carry_no_accuracy_columns(self):
        self.assertFalse(self.ACCURACY_ONLY & set(benchmark.CASE_FIELDS))

    def test_iteration_csv_fields_carry_no_accuracy_columns(self):
        self.assertFalse(self.ACCURACY_ONLY & set(benchmark.ITERATION_CSV_FIELDS))

    def test_probes_csv_covers_both_suites_in_one_schema(self):
        for field in ("suite", "task", "depth", "sample", "truth", "prediction",
                      "exact_match", "recall", "token_f1", "distractor_rate",
                      "similarity", "rouge_l", "chrf", "fdt", "sdt_norm", "fact_coverage"):
            self.assertIn(field, benchmark.PROBE_CSV_FIELDS)

    def test_main_still_runs_the_matrix_and_not_only_list_profiles(self):
        """A structural assertion, because the failure it catches is silent.

        `--list-profiles` returns early from `main`, and a helper defined immediately after
        that `return` swallowed the rest of the function body into itself: `main` still
        parsed its arguments, still resolved its settings, and then fell off the end and
        exited 0 having benchmarked nothing. No error, no output, no report -- the run just
        came back instantly. Nothing else in this file would have noticed, because every
        other test calls the pieces directly.
        """
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(benchmark))
        main = next(n for n in tree.body if getattr(n, "name", "") == "main")
        called = {
            node.func.id for node in ast.walk(main)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        for required in ("_preflight_environment_check", "_run_case", "write_reports"):
            self.assertIn(required, called, f"main() no longer calls {required}")

    def test_a_throughput_run_never_carries_accuracy_settings(self):
        # --accuracy off -> settings["accuracy"] is None, so _run_case passes probes=None and
        # the child takes the unchanged throughput path.
        args = _accuracy_args(accuracy=False)
        self.assertIsNone(
            benchmark._resolve_accuracy(
                SimpleNamespace(accuracy=None), args, [{"name": "paged_min"}]
            )
        )


if __name__ == "__main__":
    unittest.main()
