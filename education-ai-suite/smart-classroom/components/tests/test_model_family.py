# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The thinking switch follows the model family, newer releases included."""

import os
import sys

import pytest

_SC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SC_ROOT not in sys.path:
    sys.path.insert(0, _SC_ROOT)

from utils.model_family import is_qwen3_5_family, is_qwen3_dense, qwen_version


@pytest.mark.parametrize("name, dense, newer", [
    ("Qwen/Qwen3-VL-8B-Instruct", True, False),
    ("Qwen/Qwen3-8B", True, False),
    ("Qwen/Qwen3.5-9B", False, True),
    ("Qwen/Qwen3.6-35B-A3B", False, True),
    ("Qwen/Qwen3.8-27B", False, True),   # dense, but the Qwen3.5 template
    ("Qwen/Qwen4-30B", False, True),     # unreleased: lands in the newer family
    ("Qwen/Qwen2.5-VL-7B-Instruct", False, False),
    ("meta-llama/Llama-3.1-8B", False, False),
])
def test_family(name, dense, newer):
    assert is_qwen3_dense(name) is dense
    assert is_qwen3_5_family(name) is newer


def test_version_is_parsed_from_the_name():
    assert qwen_version("OpenVINO/Qwen3.8-27B-int4-ov") == (3, 8)
    assert qwen_version("whisper-small") is None
