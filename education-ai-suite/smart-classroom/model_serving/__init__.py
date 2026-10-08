# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""VLM/LLM model serving for edu-ai-suite.

The OpenAI-compatible chat API (``openai_chat``), a standalone server
(``python -m model_serving``), and the client/supervisor the Smart Classroom
app uses when ``models.text_gen.serving.mode`` is ``managed`` or ``external``.
The server imports only the ``text_gen`` engine -- no ASR, OCR, video
analytics, Content Search or UI -- so it can run on its own.
"""
