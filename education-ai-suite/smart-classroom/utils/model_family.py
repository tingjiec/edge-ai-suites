# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Model-family checks for the ``text_gen`` VLM.

Qwen3 comes in two shapes that need different prompting, and both match a naive
``"qwen3" in name`` test:

* **Qwen3** (Qwen3-8B, Qwen3-VL-8B-Instruct) — honours the ``/no_think`` soft
  switch in the user turn, and emits no thinking when it is present.
* **Qwen3.5 and later** (Qwen3.5-9B, Qwen3.6-35B-A3B, Qwen3.8-27B; HF
  ``model_type: qwen3_5`` / ``qwen3_5_moe``) — ignores ``/no_think`` and controls
  reasoning solely through the chat template's ``enable_thinking`` flag.

The family is matched by version, not by a list of names, so the next point
release (Qwen3.9, Qwen4, ...) lands in the newer family without a code change.
"""

import re

_QWEN_VERSION_RE = re.compile(r"qwen(\d+)(?:\.(\d+))?")

# First release that dropped ``/no_think`` for the template flag.
_TEMPLATE_THINKING_SINCE = (3, 5)


def qwen_version(model_name):
    """Return ``(major, minor)`` parsed from a Qwen model name, or None."""
    match = _QWEN_VERSION_RE.search(str(model_name).lower())
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


def is_qwen3_5_family(model_name) -> bool:
    """Qwen3.5 or newer: template-flag thinking, export needs transformers 5.2."""
    version = qwen_version(model_name)
    return version is not None and version >= _TEMPLATE_THINKING_SINCE


# Kept for callers written when the family was Qwen3.5/3.6 MoE only.
is_qwen3_moe_vlm = is_qwen3_5_family


def is_qwen3_dense(model_name) -> bool:
    name = str(model_name).lower()
    return "qwen3" in name and not is_qwen3_5_family(name)
