# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import unittest

from components.llm.context_bench.context_builder import (
    build_benchmark_prompt,
    build_text_of_token_length,
    render_prompt,
)
from components.llm.context_bench.trial_runner import prepare_pipeline_input


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
        self.assertEqual(build_text_of_token_length(FakeTokenizer(), 0), "")

    def test_hits_exact_target_for_word_tokenizer(self):
        tok = FakeTokenizer()
        for target in (50, 500, 5000):
            text = build_text_of_token_length(tok, target)
            # A word-level tokenizer round-trips exactly.
            self.assertEqual(len(tok.encode(text, add_special_tokens=False)), target)

    def test_large_target_beyond_single_corpus(self):
        tok = FakeTokenizer()
        text = build_text_of_token_length(tok, 160000)
        self.assertEqual(len(tok.encode(text, add_special_tokens=False)), 160000)

    def test_empty_transcript_prices_only_the_scaffolding(self):
        # This is how build_benchmark_prompt backs out the room left for content, so it
        # must cover the template + system prompt + task text and nothing else.
        tok = FakeTokenizer()
        prompt, overhead = render_prompt(tok, "", add_generation_prompt=True)

        self.assertEqual(overhead, len(tok.encode(prompt)))
        self.assertIn("CLASSROOM TRANSCRIPT", prompt)
        self.assertLess(overhead, 50)

    def test_transcript_length_moves_the_rendered_count_one_for_one(self):
        tok = FakeTokenizer()
        _, overhead = render_prompt(tok, "")
        _, with_content = render_prompt(tok, build_text_of_token_length(tok, 1000))

        self.assertEqual(with_content - overhead, 1000)

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
                return tokenized

        class Pipeline:
            def get_tokenizer(self):
                return PipelineTokenizer()

        tokenized = type("Tokenized", (), {"input_ids": InputIds()})()
        pipeline_input, count = prepare_pipeline_input(Pipeline(), "prompt", True)

        self.assertIs(pipeline_input, tokenized)
        self.assertEqual(count, 160000)

    def test_vlm_keeps_text_input_after_pipeline_token_count(self):
        class PipelineTokenizer:
            def encode(self, _prompt):
                input_ids = type("InputIds", (), {"shape": (1, 160000)})()
                return type("Tokenized", (), {"input_ids": input_ids})()

        class Pipeline:
            def get_tokenizer(self):
                return PipelineTokenizer()

        pipeline_input, count = prepare_pipeline_input(Pipeline(), "prompt", False)

        self.assertEqual(pipeline_input, "prompt")
        self.assertEqual(count, 160000)


class TestThroughputTaskFraming(unittest.TestCase):
    """A throughput task swaps the framing around the transcript, not the sizing contract."""

    APP_TASK = {"system_prompt": "Summarize the classroom transcript. ## Teacher Summary",
                "user_prefix": "/no_think\n", "suffix": ""}

    def test_no_task_is_the_built_in_prompt_byte_for_byte(self):
        self.assertEqual(build_benchmark_prompt(FakeTokenizer(), 5000),
                         build_benchmark_prompt(FakeTokenizer(), 5000, None))

    def test_app_task_hits_the_exact_size_with_its_own_framing(self):
        tok = FakeTokenizer()
        prompt, count = build_benchmark_prompt(tok, 5000, self.APP_TASK)

        self.assertEqual(count, 5000)
        self.assertTrue(prompt.startswith("<system> Summarize the classroom transcript."))
        self.assertIn("<user> /no_think", prompt)
        self.assertNotIn("CLASSROOM TRANSCRIPT", prompt)
        self.assertNotIn("two concise sentences", prompt)


class TestThroughputTaskPrompt(unittest.TestCase):
    """`classroom_summary` is rebuilt from the app config the way the summarizer builds it."""

    APP_CONFIG = (
        "app:\n  language: en\n"
        "models:\n  summarizer:\n    mode: {mode}\n    system_prompt:\n      en:\n"
        "        Dialog: dialog prompt\n        Teacher: teacher prompt\n"
        "        Hybrid: hybrid prompt\n"
    )

    def _resolve(self, mode="dialog", model="Qwen3.6-35B-A3B"):
        import tempfile
        from pathlib import Path

        from components.llm.context_bench import benchmark

        with tempfile.TemporaryDirectory() as root:
            path = Path(root, "config.yaml")
            path.write_text(self.APP_CONFIG.format(mode=mode), encoding="utf-8")
            return benchmark.throughput_task_prompt("classroom_summary", model, str(path))

    def test_the_default_task_needs_no_app_config(self):
        from components.llm.context_bench import benchmark

        self.assertIsNone(benchmark.throughput_task_prompt("summary_2s", "any", "missing.yaml"))
        self.assertIsNone(benchmark.throughput_task_prompt(None, "any", "missing.yaml"))

    def test_mode_selects_the_system_prompt_and_qwen3_gets_no_think(self):
        self.assertEqual(self._resolve()["system_prompt"], "dialog prompt")
        self.assertEqual(self._resolve("teacher")["system_prompt"], "teacher prompt")
        self.assertEqual(self._resolve("hybrid")["system_prompt"], "hybrid prompt")
        self.assertEqual(self._resolve()["user_prefix"], "/no_think\n")
        self.assertEqual(self._resolve(model="Llama-3")["user_prefix"], "")

    def test_a_missing_app_config_is_a_clear_configuration_error(self):
        from components.llm.context_bench import benchmark

        with self.assertRaisesRegex(SystemExit, "needs models.summarizer.system_prompt"):
            benchmark.throughput_task_prompt("classroom_summary", "Qwen3", "no/such/config.yaml")


if __name__ == "__main__":
    unittest.main()
