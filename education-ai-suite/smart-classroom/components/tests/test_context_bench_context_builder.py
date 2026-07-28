# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import unittest

from components.llm.context_bench.context_builder import (
    build_benchmark_prompt,
    build_text_of_token_length,
    measure_template_overhead,
)
from components.llm.context_bench.trial_runner import count_pipeline_prompt_tokens


class FakeTokenizer:
    """Word-level stub tokenizer with a stable encode/decode round trip, so this
    test doesn't need the real (heavy) transformers dependency just to exercise
    the sizing/truncation logic.

    Each distinct whitespace-delimited word maps to one token id, so token counts
    are predictable and slicing an id list then decoding round-trips cleanly.
    """

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


class TestContextBuilder(unittest.TestCase):
    def test_zero_tokens(self):
        tok = FakeTokenizer()
        text, actual = build_text_of_token_length(tok, 0)
        self.assertEqual(text, "")
        self.assertEqual(actual, 0)

    def test_hits_exact_target_for_word_tokenizer(self):
        tok = FakeTokenizer()
        for target in (50, 500, 5000):
            text, actual = build_text_of_token_length(tok, target)
            # A word-level tokenizer round-trips exactly.
            self.assertEqual(actual, target, f"target={target}")
            self.assertEqual(len(tok.encode(text, add_special_tokens=False)), target)

    def test_large_target_beyond_single_corpus(self):
        tok = FakeTokenizer()
        _text, actual = build_text_of_token_length(tok, 160000)
        self.assertEqual(actual, 160000)

    def test_template_overhead_ignores_user_content(self):
        tok = FakeTokenizer()
        messages = [
            {"role": "system", "content": "one two three four"},
            {"role": "user", "content": "should be ignored for overhead"},
        ]
        overhead = measure_template_overhead(tok, messages, add_generation_prompt=True)
        empty_render = tok.apply_chat_template(
            [
                {"role": "system", "content": "one two three four"},
                {"role": "user", "content": ""},
            ],
            tokenize=False,
        )
        self.assertEqual(overhead, len(tok.encode(empty_render)))
        # User content must not inflate the overhead measurement.
        self.assertLess(overhead, 10)

    def test_build_benchmark_prompt_lands_on_target(self):
        tok = FakeTokenizer()
        for target in (2000, 80000, 160000):
            prompt, prompt_tokens = build_benchmark_prompt(tok, target)
            self.assertIn("<system>", prompt)
            self.assertIn("<user>", prompt)
            self.assertIn("CLASSROOM TRANSCRIPT", prompt)
            self.assertIn("TASK", prompt)
            self.assertIn("Summarize the lesson", prompt)
            # The rendered prompt (content + template overhead) should land on the
            # requested size for a round-tripping tokenizer.
            self.assertEqual(prompt_tokens, target, f"target={target}")

    def test_corrects_template_boundary_token_drift(self):
        class BoundaryDriftTokenizer(FakeTokenizer):
            def apply_chat_template(self, messages, tokenize=False, **kwargs):
                rendered = super().apply_chat_template(messages, tokenize=tokenize, **kwargs)
                if messages[-1].get("content"):
                    rendered += " boundary-token"
                return rendered

        tok = BoundaryDriftTokenizer()
        _prompt, prompt_tokens = build_benchmark_prompt(tok, 80000)
        self.assertEqual(prompt_tokens, 80000)

    def test_pipeline_tokenizer_count_uses_input_id_width(self):
        class InputIds:
            shape = (1, 160000)

        class PipelineTokenizer:
            def encode(self, _prompt):
                return type("Tokenized", (), {"input_ids": InputIds()})()

        class Pipeline:
            def get_tokenizer(self):
                return PipelineTokenizer()

        self.assertEqual(count_pipeline_prompt_tokens(Pipeline(), "prompt"), 160000)


if __name__ == "__main__":
    unittest.main()
