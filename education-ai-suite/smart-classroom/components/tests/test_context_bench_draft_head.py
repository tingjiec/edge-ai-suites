# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""The dFlash draft-head tool must find exactly the target's int8 lm_head, or refuse."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from components.llm.context_bench import dflash_draft_head


def _layer(layer_id, name, kind, **data):
    attrs = " ".join(f'{key}="{value}"' for key, value in data.items())
    return (f'\t\t<layer id="{layer_id}" name="{name}" type="{kind}" version="opset1">\n'
            f'\t\t\t<data {attrs} />\n\t\t</layer>\n')


def _ir(scale_type="f16", scale_shape="8, 1", transpose_b="true", weight_type="i8"):
    layers = (
        _layer(1, "self.model.lm_head.weight", "Const", element_type=weight_type,
               shape="8, 4", offset=100, size=32)
        + _layer(2, "lm_head/Convert", "Convert", destination_type="f16")
        + _layer(3, "self.model.lm_head.weight/scale", "Const", element_type=scale_type,
                 shape=scale_shape, offset=132, size=16)
        + _layer(4, "self.model.lm_head.weight/fq_weights_1", "Multiply", auto_broadcast="numpy")
        + _layer(5, "__module.model.lm_head/ov_ext::linear/Convert", "Convert",
                 destination_type="f32")
        + _layer(6, "__module.model.lm_head/ov_ext::linear/MatMul", "MatMul",
                 transpose_a="false", transpose_b=transpose_b)
    )
    edges = "".join(
        f'\t\t<edge from-layer="{a}" from-port="0" to-layer="{b}" to-port="{p}" />\n'
        for a, b, p in ((1, 2, 0), (2, 4, 0), (3, 4, 1), (4, 5, 0), (5, 6, 1))
    )
    return f'<net name="m" version="11">\n\t<layers>\n{layers}\t</layers>\n\t<edges>\n{edges}\t</edges>\n</net>\n'


class TestLocateLmHead(unittest.TestCase):
    def _locate(self, **ir_kwargs):
        with tempfile.TemporaryDirectory() as target:
            Path(target, "openvino_language_model.xml").write_text(_ir(**ir_kwargs), encoding="utf-8")
            return dflash_draft_head.locate_lm_head(target)

    def test_finds_the_int8_weight_and_per_row_scale_offsets(self):
        head = self._locate()

        self.assertEqual((head["weight_offset"], head["scale_offset"]), (100, 132))
        self.assertEqual((head["vocab"], head["hidden"]), (8, 4))
        self.assertTrue(head["bin"].endswith("openvino_language_model.bin"))

    def test_refuses_heads_it_does_not_understand(self):
        for kwargs, message in (
            ({"weight_type": "u8"}, "not int8 weights"),
            ({"scale_shape": "8, 2"}, "not per-row"),
            ({"transpose_b": "false"}, "transpose_b"),
        ):
            with self.subTest(**kwargs), self.assertRaisesRegex(ValueError, message):
                self._locate(**kwargs)

    def test_a_directory_without_an_ir_is_refused(self):
        with tempfile.TemporaryDirectory() as target, self.assertRaisesRegex(ValueError, "has no"):
            dflash_draft_head.locate_lm_head(target)


class TestInt4Packing(unittest.TestCase):
    def test_round_trip_is_within_half_a_quantization_step(self):
        rng = np.random.default_rng(0)
        weight = rng.integers(-127, 128, size=(6, 256), dtype=np.int8)
        scale = rng.uniform(0.001, 0.01, size=(6, 1)).astype(np.float16)

        packed, scales = dflash_draft_head._quantize_int4(weight, scale, 128)

        nibbles = np.stack([packed & 0x0F, packed >> 4], axis=-1).reshape(6, 2, 128)
        values = np.where(nibbles > 7, nibbles.astype(np.int16) - 16, nibbles)
        rebuilt = (values * scales.astype(np.float32)).reshape(6, 256)
        original = weight.astype(np.float32) * scale.astype(np.float32)
        step = np.repeat(scales.astype(np.float32).reshape(6, 2), 128, axis=1)
        self.assertTrue(np.all(np.abs(rebuilt - original) <= step / 2 + 1e-6))
        self.assertEqual(packed.shape, (6, 2, 64))


if __name__ == "__main__":
    unittest.main()
