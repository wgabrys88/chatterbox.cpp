#!/usr/bin/env python3


from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import gguf







_DENY_SUBSTRINGS = (

    "flow/input_embedding",
    "flow/spk_embed_affine/w",
    "/builtin/",

    "text_emb",
    "speech_emb",
    "wte",
    "wpe",

    "stft_basis",
    "mel_filterbank",
    "mel_fb",
    "pos_emb",
    "pe/pe",
    "pre_attention_query",












    "/b",
    "/bias",
    "/bn/",
    "/norm/",
    "/ln_",
    "/scale",















    "alpha",
    "beta",
    "gamma",








    "voice_encoder/",
    "campplus/",
    "s3tokv2/",
)









_DENY_SUFFIXES = (
    "/g",
)




_QUANTIZABLE_SRC_DTYPES = {
    gguf.GGMLQuantizationType.F32,
    gguf.GGMLQuantizationType.F16,
}


_QUANT_TYPE = {
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
    "q5_0": gguf.GGMLQuantizationType.Q5_0,
    "q4_0": gguf.GGMLQuantizationType.Q4_0,




    "f16":  gguf.GGMLQuantizationType.F16,
}


def should_quantize(name: str, shape: tuple[int, ...], qtype: gguf.GGMLQuantizationType) -> bool:

    n_elements = 1
    for d in shape:
        n_elements *= d
    if n_elements < 1024:
        return False


    for s in _DENY_SUBSTRINGS:
        if s in name:
            return False
    for s in _DENY_SUFFIXES:
        if name.endswith(s):
            return False

    block = gguf.GGML_QUANT_SIZES[qtype][0]





    if len(shape) == 2:
        return shape[-1] % block == 0



















    if len(shape) == 3:
        return shape[-1] % block == 0

    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="Source GGUF (F32/F16)")
    ap.add_argument("dst", type=Path, help="Output GGUF")
    ap.add_argument("dtype", choices=_QUANT_TYPE.keys(), help="Target quant dtype")
    ap.add_argument(
        "--name-filter",
        default=None,
        help=("Substring filter on tensor names; only tensors whose name "
              "contains this substring are touched.  All other tensors "
              "are passed through at their source dtype.  Useful for "
              "applying f16 to HiFT conv kernels in a Q4_0 source GGUF "
              "without disturbing the existing Q4_0 CFM weights."),
    )
    args = ap.parse_args()

    qtype = _QUANT_TYPE[args.dtype]
    name_filter = args.name_filter

    src = gguf.GGUFReader(args.src, "r")
    arch = src.fields.get("general.architecture")
    arch_name = ""
    if arch is not None:
        arch_name = bytes(arch.parts[arch.data[0]]).decode("utf-8")

    writer = gguf.GGUFWriter(args.dst, arch_name or "chatterbox-s3gen")



    _SKIP_KEYS = {
        "GGUF.version",
        "GGUF.tensor_count",
        "GGUF.kv_count",
        "general.architecture",
    }
    for key, field in src.fields.items():
        if key in _SKIP_KEYS:
            continue
        val_type = field.types[0] if field.types else None
        parts = [field.parts[i] for i in field.data]
        if val_type is None:
            continue
        if val_type == gguf.GGUFValueType.ARRAY:
            sub_type = field.types[1] if len(field.types) > 1 else None
            if sub_type == gguf.GGUFValueType.STRING:
                values = [bytes(p).decode("utf-8") for p in parts]
                writer.add_array(key, values)
            else:
                arr = np.concatenate([np.asarray(p) for p in parts]).tolist()
                writer.add_array(key, arr)
        elif val_type == gguf.GGUFValueType.STRING:
            writer.add_string(key, bytes(parts[0]).decode("utf-8"))
        elif val_type == gguf.GGUFValueType.BOOL:
            writer.add_bool(key, bool(parts[0][0]))
        elif val_type in (gguf.GGUFValueType.UINT8, gguf.GGUFValueType.UINT16,
                          gguf.GGUFValueType.UINT32, gguf.GGUFValueType.UINT64):
            writer.add_uint32(key, int(parts[0][0]))
        elif val_type in (gguf.GGUFValueType.INT8, gguf.GGUFValueType.INT16,
                          gguf.GGUFValueType.INT32, gguf.GGUFValueType.INT64):
            writer.add_int32(key, int(parts[0][0]))
        elif val_type in (gguf.GGUFValueType.FLOAT32, gguf.GGUFValueType.FLOAT64):
            writer.add_float32(key, float(parts[0][0]))

    quantized_count = 0
    kept_count = 0
    src_bytes = 0
    dst_bytes = 0

    for t in src.tensors:

        shape = tuple(int(d) for d in reversed(t.shape) if d > 0)
        if not shape:
            shape = (int(t.shape[0]),)

        data = np.asarray(t.data)
        src_bytes += data.nbytes

        in_filter = name_filter is None or name_filter in t.name
        if (in_filter and t.tensor_type in _QUANTIZABLE_SRC_DTYPES
                      and t.tensor_type != qtype
                      and should_quantize(t.name, shape, qtype)):



            arr = data.astype(np.float32).reshape(shape)
            qdata = gguf.quants.quantize(arr, qtype)
            writer.add_tensor(t.name, qdata, raw_shape=qdata.shape, raw_dtype=qtype)
            quantized_count += 1
            dst_bytes += qdata.nbytes
        else:








            block_size, type_size = gguf.GGML_QUANT_SIZES[t.tensor_type]
            if block_size == 1:
                arr = data.reshape(shape)
                writer.add_tensor(t.name, arr, raw_shape=arr.shape, raw_dtype=t.tensor_type)
            else:








                byte_inner = shape[-1] // block_size * type_size
                byte_shape = tuple(list(shape[:-1]) + [byte_inner])
                writer.add_tensor(t.name, data, raw_shape=byte_shape, raw_dtype=t.tensor_type)
            kept_count += 1
            dst_bytes += data.nbytes

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"arch: {arch_name!r}")
    print(f"quantized: {quantized_count} tensors to {args.dtype.upper()}")
    print(f"kept:      {kept_count} tensors as source dtype")
    print(f"size:      {src_bytes / 1e6:.1f} MB  →  {dst_bytes / 1e6:.1f} MB  "
          f"({dst_bytes / src_bytes * 100:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
