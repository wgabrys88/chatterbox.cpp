import json
import math
from pathlib import Path

import gguf
import numpy as np

WEIGHT_TYPES = tuple(json.loads(Path(__file__).with_name("precision_policy.json").read_text(encoding="utf-8"))["weight_types"])
QUANT_TYPES = {"q4_0": gguf.GGMLQuantizationType.Q4_0}


def add_weight(writer, name, array, weight_type, *, force_f32=False):
    if weight_type not in WEIGHT_TYPES:
        raise ValueError(f"unsupported weight type: {weight_type}")
    arr = np.ascontiguousarray(array)
    if arr.dtype.kind in "iu" or np.issubdtype(arr.dtype, np.integer):
        writer.add_tensor(name, arr)
        return "native"
    if force_f32 or weight_type == "f32":
        writer.add_tensor(name, np.ascontiguousarray(arr.astype(np.float32)))
        return "f32"
    if weight_type == "f16":
        writer.add_tensor(name, np.ascontiguousarray(arr.astype(np.float16)))
        return "f16"
    qtype = QUANT_TYPES[weight_type]
    block = gguf.GGML_QUANT_SIZES[qtype][0]
    if arr.ndim != 2 or math.prod(arr.shape) < 1024 or arr.shape[-1] % block:
        writer.add_tensor(name, np.ascontiguousarray(arr.astype(np.float16)))
        return "f16"
    qdata = gguf.quants.quantize(np.ascontiguousarray(arr.astype(np.float32)), qtype)
    writer.add_tensor(name, qdata, raw_shape=qdata.shape, raw_dtype=qtype)
    return weight_type
