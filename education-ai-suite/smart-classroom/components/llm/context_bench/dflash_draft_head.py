# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Give a dFlash draft its own int4 lm_head, so OpenVINO GenAI stops grafting the target's.

A dFlash draft exported by optimum-intel ends at `last_hidden_state`. GenAI then clones the
*target's* lm_head onto it at load time -- on Qwen3.6-35B-A3B int4 that is a 248320 x 2048
int8 matrix, 509 MB, streamed on every draft pass. A draft that already outputs `logits`
skips the graft (GenAI accepts exactly one of the two outputs), so this tool appends the same
head requantized to int4 symmetric, group 128: half the bytes per pass.

Measured on an Intel Arc B390 iGPU at 8K context, same session, stock vs this draft:
steady-state passes 52.5 -> 49.3 ms (k=3) and 66.0 -> 61.5 ms (k=7), with draft acceptance
unchanged on the benchmark prompt (71.4%, 34.1%) and within 2 points on math. An int8 copy
of the head (`--bits 8`) reproduces the graft exactly -- identical acceptance and pass cost --
and is the control that shows the gain is the head's bytes, not the graph change.

The target is never loaded: its head is read from the .bin by the offsets its XML declares,
and the tool refuses any head that is not the int8-per-row-symmetric chain optimum-intel
writes (Const i8 [V, H] -> Convert -> Multiply by Const f16 [V, 1] -> MatMul transpose_b).
The rebuilt head is checked against a NumPy reference on CPU before anything is written, and
the output is a new directory: neither input is modified.

    python -m components.llm.context_bench.dflash_draft_head \\
        --target models/openvino/Qwen3.6-35B-A3B_int4 \\
        --draft models/openvino/qwen3.6-35b-a3b-dflash-int4-ov \\
        --output models/openvino/qwen3.6-35b-a3b-dflash-int4-ov-int4head

Then point a profile's `dflash.model` at the output directory. The derived draft is tied to
the target export it was built from: rebuild it whenever the target is re-exported.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

# Rows quantized per chunk: the f32 copy of the whole head would be 2 GB.
_CHUNK_ROWS = 16384

_LANGUAGE_MODEL_IRS = ("openvino_language_model.xml", "openvino_model.xml")


def _layer_data(xml_text: str, layer_id: str) -> tuple[str, dict]:
    """(type, <data> attributes) of one IR layer, parsed from the XML text."""
    match = re.search(rf'<layer id="{layer_id}" name="[^"]*" type="([^"]+)"[^>]*>(.*?)</layer>',
                      xml_text, re.S)
    if match is None:
        raise ValueError(f"layer {layer_id} not found")
    data = re.search(r"<data ([^>]*?)/?>", match.group(2))
    return match.group(1), dict(re.findall(r'(\w+)="([^"]*)"', data.group(1) if data else ""))


def locate_lm_head(target_dir: str) -> dict:
    """Byte offsets and shapes of the target's int8 lm_head weight and per-row scale.

    Walks back from the MatMul named `lm_head` through the decompression chain optimum-intel
    emits. Raises ValueError for anything else, rather than quantizing the wrong tensor.
    """
    xml_path = next((os.path.join(target_dir, name) for name in _LANGUAGE_MODEL_IRS
                     if os.path.isfile(os.path.join(target_dir, name))), None)
    if xml_path is None:
        raise ValueError(f"{target_dir} has no {' or '.join(_LANGUAGE_MODEL_IRS)}")
    with open(xml_path, encoding="utf-8") as xml_file:
        text = xml_file.read()
    edges = {}
    for from_layer, to_layer in re.findall(r'<edge from-layer="(\d+)" from-port="\d+" '
                                           r'to-layer="(\d+)"', text):
        edges.setdefault(to_layer, []).append(from_layer)
    matmuls = re.findall(r'<layer id="(\d+)" name="[^"]*lm_head[^"]*" type="MatMul"', text)
    if len(matmuls) != 1:
        raise ValueError(f"expected one lm_head MatMul in {xml_path}, found {len(matmuls)}")
    _, matmul = _layer_data(text, matmuls[0])
    if matmul.get("transpose_b") != "true":
        raise ValueError("lm_head MatMul is not transpose_b; unsupported layout")

    def producers(layer_id):
        return [(source, *_layer_data(text, source)) for source in edges.get(layer_id, [])]

    # MatMul <- Convert(f32) <- Multiply(weight f16, scale f16) <- {Convert(i8 const), scale const}
    chain = [p for p in producers(matmuls[0]) if p[1] == "Convert"]
    multiply = [p for c in chain for p in producers(c[0]) if p[1] == "Multiply"]
    if len(multiply) != 1:
        raise ValueError("lm_head is not a Convert(Multiply(...)) decompression chain")
    weight = scale = None
    for source, kind, data in producers(multiply[0][0]):
        if kind == "Const" and data.get("element_type") == "f16":
            scale = data
        elif kind == "Convert":
            consts = [p for p in producers(source) if p[1] == "Const"]
            if consts and consts[0][2].get("element_type") == "i8":
                weight = consts[0][2]
    if weight is None or scale is None:
        raise ValueError("lm_head is not int8 weights with a per-row f16 scale")
    vocab, hidden = (int(x) for x in weight["shape"].split(","))
    if [int(x) for x in scale["shape"].split(",")] != [vocab, 1]:
        raise ValueError(f"lm_head scale shape {scale['shape']} is not per-row")
    return {
        "bin": os.path.splitext(xml_path)[0] + ".bin",
        "weight_offset": int(weight["offset"]),
        "scale_offset": int(scale["offset"]),
        "vocab": vocab,
        "hidden": hidden,
    }


def _quantize_int4(weight, scale, group: int):
    """Requantize the int8 head to int4 symmetric per `group` columns: (packed u8, f16 scales)."""
    import numpy as np

    vocab, hidden = weight.shape
    groups = hidden // group
    packed = np.empty((vocab, groups, group // 2), np.uint8)
    scales = np.empty((vocab, groups, 1), np.float16)
    for start in range(0, vocab, _CHUNK_ROWS):
        stop = min(vocab, start + _CHUNK_ROWS)
        rows = weight[start:stop].astype(np.float32) * scale[start:stop].astype(np.float32)
        rows = rows.reshape(stop - start, groups, group)
        step = np.abs(rows).max(axis=2, keepdims=True) / 7.0
        step[step == 0] = 1.0
        # Codes are chosen against the f16 scale actually stored, so dequantization agrees.
        step = step.astype(np.float16)
        codes = np.clip(np.rint(rows / step.astype(np.float32)), -8, 7).astype(np.int8)
        nibbles = (codes & 0x0F).astype(np.uint8)
        packed[start:stop] = nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)  # low nibble first
        scales[start:stop] = step
    return packed, scales


def build_head(hidden_state, weight, scale, bits: int, group: int):
    """Append `logits = hidden_state @ dequant(head).T` to a graph; returns the MatMul node."""
    import numpy as np
    import openvino as ov
    import openvino.opset13 as ops

    vocab, hidden = weight.shape
    if bits == 8:
        decompressed = ops.multiply(
            ops.convert(ops.constant(np.ascontiguousarray(weight), dtype=ov.Type.i8), ov.Type.f16),
            ops.constant(np.ascontiguousarray(scale)),
        )
    else:
        if hidden % group:
            raise ValueError(f"hidden size {hidden} is not a multiple of group {group}")
        packed, scales = _quantize_int4(weight, scale, group)
        tensor = ov.Tensor(ov.Type.i4, ov.Shape([vocab, hidden // group, group]))
        np.copyto(tensor.data.view(np.uint8).reshape(-1), packed.reshape(-1))
        decompressed = ops.reshape(
            ops.multiply(ops.convert(ops.constant(tensor), ov.Type.f16), ops.constant(scales)),
            ops.constant(np.array([vocab, hidden], np.int64)),
            special_zero=False,
        )
    return ops.matmul(hidden_state, ops.convert(decompressed, ov.Type.f32),
                      transpose_a=False, transpose_b=True)


def _self_check(weight, scale, bits: int, group: int) -> float:
    """Relative error of the compiled head against the same quantization done in NumPy."""
    import numpy as np
    import openvino as ov
    import openvino.opset13 as ops

    rows = min(4096, weight.shape[0])
    weight, scale = np.asarray(weight[:rows]), np.asarray(scale[:rows])
    hidden = ops.parameter([4, weight.shape[1]], ov.Type.f32)
    model = ov.Model([ops.result(build_head(hidden, weight, scale, bits, group))], [hidden])
    probe = np.random.default_rng(0).standard_normal((4, weight.shape[1])).astype(np.float32)
    got = ov.Core().compile_model(model, "CPU")(probe)[0]
    reference = weight.astype(np.float32) * scale.astype(np.float32)
    if bits == 4:
        packed, scales = _quantize_int4(weight, scale, group)
        nibbles = np.stack([packed & 0x0F, packed >> 4], axis=-1).reshape(rows, -1, group)
        values = np.where(nibbles > 7, nibbles.astype(np.int16) - 16, nibbles).astype(np.float32)
        reference = (values * scales.astype(np.float32)).reshape(rows, -1)
    expected = probe @ reference.T
    return float(np.abs(got - expected).max() / np.abs(expected).max())


def build_draft(target_dir: str, draft_dir: str, output_dir: str, bits: int = 4,
                group: int = 128) -> str:
    """Write `draft_dir` plus its own lm_head to `output_dir`; returns the output IR path."""
    import numpy as np
    import openvino as ov
    import openvino.opset13 as ops

    if os.path.abspath(output_dir) in (os.path.abspath(target_dir), os.path.abspath(draft_dir)):
        raise ValueError("--output must be a new directory, not one of the inputs")
    head = locate_lm_head(target_dir)
    weight = np.memmap(head["bin"], np.int8, "r", offset=head["weight_offset"],
                       shape=(head["vocab"], head["hidden"]))
    scale = np.memmap(head["bin"], np.float16, "r", offset=head["scale_offset"],
                      shape=(head["vocab"], 1))
    error = _self_check(weight, scale, bits, group)
    if error > 0.01:
        raise ValueError(f"rebuilt head disagrees with its NumPy reference (rel. error {error:.4f})")

    model = ov.Core().read_model(os.path.join(draft_dir, "openvino_model.xml"))
    if [output.get_names() & {"logits"} for output in model.outputs] != [set()]:
        raise ValueError("draft must have exactly one output and no `logits` yet")
    old_result = model.get_results()[0]
    hidden_state = old_result.input_value(0)
    if hidden_state.get_partial_shape()[-1].get_length() != head["hidden"]:
        raise ValueError("draft hidden size does not match the target lm_head")
    if hidden_state.get_element_type() != ov.Type.f32:
        hidden_state = ops.convert(hidden_state, ov.Type.f32)
    logits = ops.result(build_head(hidden_state, weight, scale, bits, group))
    logits.output(0).get_tensor().set_names({"logits"})
    model.add_results([logits])
    model.remove_result(old_result)
    model.validate_nodes_and_infer_types()

    os.makedirs(output_dir, exist_ok=True)
    output_xml = os.path.join(output_dir, "openvino_model.xml")
    ov.save_model(model, output_xml, compress_to_fp16=False)
    for name in os.listdir(draft_dir):
        source = os.path.join(draft_dir, name)
        if os.path.isfile(source) and not name.startswith("openvino_model."):
            shutil.copyfile(source, os.path.join(output_dir, name))
    return output_xml


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--target", required=True, help="target model IR directory")
    parser.add_argument("--draft", required=True, help="dFlash draft IR directory")
    parser.add_argument("--output", required=True, help="new directory for the derived draft")
    parser.add_argument("--bits", type=int, choices=(4, 8), default=4,
                        help="4: int4 sym head (faster); 8: exact copy of the graft (control)")
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args(argv)
    try:
        path = build_draft(args.target, args.draft, args.output, args.bits, args.group_size)
    except ValueError as exc:
        sys.exit(f"dflash_draft_head: {exc}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
