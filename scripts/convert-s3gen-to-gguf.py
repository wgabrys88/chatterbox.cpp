#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Optional
import gguf
import numpy as np
import torch
from safetensors.torch import load_file
VARIANTS = {
    "turbo": {
        "ckpt_filename": "s3gen_meanflow.safetensors",
        "loader": "safetensors",
        "gguf_name": "Chatterbox Turbo S3Gen",
        "gguf_description": "S3Gen flow + mel2wav (HiFT) for ggml port.",
        "meanflow": True,
        "n_timesteps": 2,
        "cfg_rate": 0.0,
    },
    "mtl": {
        "ckpt_filename": "s3gen.pt",
        "loader": "torch",
        "gguf_name": "Chatterbox Multilingual S3Gen",
        "gguf_description": "S3Gen standard-CFM (10-step Euler, CFG) + HiFT vocoder for ggml port.",
        "meanflow": False,
        "n_timesteps": 10,
        "cfg_rate": 0.7,
    },
}
QUANT_CHOICES = ("f16", "q4_0")
from quant_policy import QUANT_TYPE as _RQ_QUANT_TYPE, should_quantize as _SHOULD_QUANTIZE
_RAW_F32_SUBSTRINGS = (
    "flow/input_embedding",
    "flow/spk_embed_affine/",
    "/builtin/",
    "s3gen/mel_fb/",
    "campplus/",
    "s3tokv2/",
)
def _must_stay_f32(name: str) -> bool:
    return any(s in name for s in _RAW_F32_SUBSTRINGS)
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert Trident S3Gen weights to GGUF.")
    ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--quant", choices=QUANT_CHOICES, required=True)
    return ap.parse_args()
def as_numpy(tensor: torch.Tensor, *, dtype=None) -> np.ndarray:
    if dtype is not None:
        tensor = tensor.to(dtype)
    return np.ascontiguousarray(tensor.detach().cpu().numpy())
def resolve_weight_norm(state: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    g = state[f"{prefix}.parametrizations.weight.original0"]
    v = state[f"{prefix}.parametrizations.weight.original1"]
    norm = v.flatten(1).norm(dim=1).view(-1, *([1] * (v.ndim - 1)))
    return g * v / norm
def expand_weight_norm(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = dict(state)
    prefixes = set()
    for k in state:
        m = re.match(r"(.+)\.parametrizations\.weight\.original0$", k)
        if m:
            prefixes.add(m.group(1))
    for p in prefixes:
        out[f"{p}.weight"] = resolve_weight_norm(state, p)
        out.pop(f"{p}.parametrizations.weight.original0", None)
        out.pop(f"{p}.parametrizations.weight.original1", None)
    return out
def export(writer: gguf.GGUFWriter, state: dict, name: str, *, dtype=torch.float32):
    arr = as_numpy(state[name], dtype=dtype)
    gguf_name = name
    writer.add_tensor(gguf_name, arr)
    return arr.shape
def add_tensor_maybe_q(
    writer: gguf.GGUFWriter,
    name: str,
    arr: np.ndarray,
    quant: str,
    *,
    stats: Optional[dict[str, int]] = None,
) -> None:
    if arr.dtype.kind in "iu" or np.issubdtype(arr.dtype, np.integer):
        writer.add_tensor(name, arr)
        return
    if _must_stay_f32(name):
        writer.add_tensor(name, np.ascontiguousarray(arr.astype(np.float32)))
        return
    if quant in ("f16", "f32"):
        writer.add_tensor(name, arr)
        return
    qtype = _RQ_QUANT_TYPE[quant]
    if not _SHOULD_QUANTIZE(name, arr.shape, qtype):
        writer.add_tensor(name, np.ascontiguousarray(arr.astype(np.float16)) if arr.ndim == 3 else arr)
        return
    qdata = gguf.quants.quantize(np.ascontiguousarray(arr.astype(np.float32)), qtype)
    writer.add_tensor(name, qdata, raw_shape=qdata.shape, raw_dtype=qtype)
    if stats is not None:
        stats["n_quant"] = stats.get("n_quant", 0) + 1
def export_conformer_block(
    writer: gguf.GGUFWriter,
    state: dict,
    prefix: str,
    gguf_prefix: str,
    quant: str,
    *,
    stats: Optional[dict[str, int]] = None,
):
    mapping = {
        "norm_mha.weight":           ("norm_mha/w", torch.float32),
        "norm_mha.bias":             ("norm_mha/b", torch.float32),
        "norm_ff.weight":            ("norm_ff/w", torch.float32),
        "norm_ff.bias":              ("norm_ff/b", torch.float32),
        "self_attn.linear_q.weight": ("attn/q/w", torch.float32),
        "self_attn.linear_q.bias":   ("attn/q/b",   torch.float32),
        "self_attn.linear_k.weight": ("attn/k/w", torch.float32),
        "self_attn.linear_k.bias":   ("attn/k/b",   torch.float32),
        "self_attn.linear_v.weight": ("attn/v/w", torch.float32),
        "self_attn.linear_v.bias":   ("attn/v/b",   torch.float32),
        "self_attn.linear_out.weight": ("attn/o/w", torch.float32),
        "self_attn.linear_out.bias":   ("attn/o/b",   torch.float32),
        "self_attn.linear_pos.weight": ("attn/pos/w", torch.float32),
        "self_attn.pos_bias_u":      ("attn/pos_bias_u", torch.float32),
        "self_attn.pos_bias_v":      ("attn/pos_bias_v", torch.float32),
        "feed_forward.w_1.weight":   ("ff/w1/w", torch.float32),
        "feed_forward.w_1.bias":     ("ff/w1/b",   torch.float32),
        "feed_forward.w_2.weight":   ("ff/w2/w", torch.float32),
        "feed_forward.w_2.bias":     ("ff/w2/b",   torch.float32),
    }
    for src_suffix, (dst_suffix, dtype) in mapping.items():
        src = f"{prefix}.{src_suffix}"
        dst = f"{gguf_prefix}/{dst_suffix}"
        arr = as_numpy(state[src], dtype=dtype)
        add_tensor_maybe_q(writer, dst, arr, quant, stats=stats)
def main():
    args = parse_args()
    cfg = VARIANTS[args.variant]
    ckpt_dir = args.ckpt_dir
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / cfg["ckpt_filename"]
    print(f"Loading {ckpt_path}")
    if cfg["loader"] == "safetensors":
        raw = load_file(ckpt_path)
    elif cfg["loader"] == "torch":
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    else:
        raise ValueError(f"unknown loader: {cfg['loader']}")
    state = expand_weight_norm(raw)
    print(f"Resolved {len([k for k in raw if 'parametrizations' in k])} weight_norm entries")
    conds = torch.load(ckpt_dir / "conds.pt", map_location="cpu", weights_only=True)
    gen = conds["gen"]
    writer = gguf.GGUFWriter(str(args.out), "chatterbox-s3gen")
    writer.add_name(cfg["gguf_name"])
    writer.add_description(cfg["gguf_description"])
    writer.add_string("s3gen.quantization", args.quant)
    writer.add_string("s3gen.variant", args.variant)
    writer.add_bool("s3gen.meanflow", cfg["meanflow"])
    writer.add_uint32("s3gen.n_timesteps", cfg["n_timesteps"])
    writer.add_float32("s3gen.cfg_rate", cfg["cfg_rate"])
    qstats: Optional[dict[str, int]] = {"n_quant": 0} if args.quant == "q4_0" else None
    writer.add_uint32("s3gen.speech_vocab_size", 6561)
    writer.add_uint32("s3gen.input_size", 512)
    writer.add_uint32("s3gen.output_size", 80)
    writer.add_uint32("s3gen.encoder.n_blocks", 6)
    writer.add_uint32("s3gen.encoder.up_n_blocks", 4)
    writer.add_uint32("s3gen.encoder.attention_heads", 8)
    writer.add_uint32("s3gen.encoder.head_dim", 64)
    writer.add_uint32("s3gen.encoder.ff_size", 2048)
    writer.add_uint32("s3gen.encoder.token_mel_ratio", 2)
    writer.add_uint32("s3gen.encoder.pre_lookahead_len", 3)
    writer.add_float32("s3gen.layer_norm_eps", 1e-12)
    writer.add_uint32("s3gen.spk_embed_dim", 192)
    prompt_token = gen["prompt_token"].reshape(-1).to(torch.int32)
    prompt_feat = gen["prompt_feat"].squeeze(0)
    embedding = gen["embedding"].squeeze(0)
    writer.add_uint32("s3gen.builtin.prompt_token_len", int(prompt_token.numel()))
    writer.add_uint32("s3gen.builtin.prompt_feat_frames", int(prompt_feat.shape[0]))
    add_tensor_maybe_q(writer, "s3gen/builtin/prompt_token", as_numpy(prompt_token), args.quant, stats=qstats)
    add_tensor_maybe_q(writer, "s3gen/builtin/prompt_feat", as_numpy(prompt_feat, dtype=torch.float32), args.quant, stats=qstats)
    add_tensor_maybe_q(writer, "s3gen/builtin/embedding", as_numpy(embedding, dtype=torch.float32), args.quant, stats=qstats)
    add_tensor_maybe_q(writer, "flow/input_embedding",       as_numpy(state["flow.input_embedding.weight"]), args.quant, stats=qstats)
    add_tensor_maybe_q(writer, "flow/spk_embed_affine/w",    as_numpy(state["flow.spk_embed_affine_layer.weight"]), args.quant, stats=qstats)
    add_tensor_maybe_q(writer, "flow/spk_embed_affine/b",    as_numpy(state["flow.spk_embed_affine_layer.bias"]), args.quant, stats=qstats)
    add_tensor_maybe_q(writer, "flow/encoder_proj/w",        as_numpy(state["flow.encoder_proj.weight"]), args.quant, stats=qstats)
    add_tensor_maybe_q(writer, "flow/encoder_proj/b",        as_numpy(state["flow.encoder_proj.bias"]), args.quant, stats=qstats)
    add_tensor_maybe_q(writer, "flow/encoder/embed/linear/w",  as_numpy(state["flow.encoder.embed.out.0.weight"]), args.quant, stats=qstats)
    add_tensor_maybe_q(writer, "flow/encoder/embed/linear/b",  as_numpy(state["flow.encoder.embed.out.0.bias"]), args.quant, stats=qstats)
    add_tensor_maybe_q(writer, "flow/encoder/embed/norm/w",    as_numpy(state["flow.encoder.embed.out.1.weight"]), args.quant, stats=qstats)
    add_tensor_maybe_q(writer, "flow/encoder/embed/norm/b",    as_numpy(state["flow.encoder.embed.out.1.bias"]), args.quant, stats=qstats)
    add_tensor_maybe_q(writer, "flow/encoder/pre_lookahead/conv1/w", as_numpy(state["flow.encoder.pre_lookahead_layer.conv1.weight"]), args.quant, stats=qstats)
    add_tensor_maybe_q(writer, "flow/encoder/pre_lookahead/conv1/b", as_numpy(state["flow.encoder.pre_lookahead_layer.conv1.bias"]), args.quant, stats=qstats)
    add_tensor_maybe_q(writer, "flow/encoder/pre_lookahead/conv2/w", as_numpy(state["flow.encoder.pre_lookahead_layer.conv2.weight"]), args.quant, stats=qstats)
    add_tensor_maybe_q(writer, "flow/encoder/pre_lookahead/conv2/b", as_numpy(state["flow.encoder.pre_lookahead_layer.conv2.bias"]), args.quant, stats=qstats)
    for i in range(6):
        export_conformer_block(writer, state,
                               f"flow.encoder.encoders.{i}",
                               f"flow/encoder/block{i}",
                               args.quant,
                               stats=qstats)
    add_tensor_maybe_q(writer, "flow/encoder/up_layer/conv/w", as_numpy(state["flow.encoder.up_layer.conv.weight"]), args.quant, stats=qstats)
    add_tensor_maybe_q(writer, "flow/encoder/up_layer/conv/b", as_numpy(state["flow.encoder.up_layer.conv.bias"]), args.quant, stats=qstats)
    add_tensor_maybe_q(writer, "flow/encoder/up_embed/linear/w", as_numpy(state["flow.encoder.up_embed.out.0.weight"]), args.quant, stats=qstats)
    add_tensor_maybe_q(writer, "flow/encoder/up_embed/linear/b", as_numpy(state["flow.encoder.up_embed.out.0.bias"]), args.quant, stats=qstats)
    add_tensor_maybe_q(writer, "flow/encoder/up_embed/norm/w",   as_numpy(state["flow.encoder.up_embed.out.1.weight"]), args.quant, stats=qstats)
    add_tensor_maybe_q(writer, "flow/encoder/up_embed/norm/b",   as_numpy(state["flow.encoder.up_embed.out.1.bias"]), args.quant, stats=qstats)
    for i in range(4):
        export_conformer_block(writer, state,
                               f"flow.encoder.up_encoders.{i}",
                               f"flow/encoder/up_block{i}",
                               args.quant,
                               stats=qstats)
    add_tensor_maybe_q(writer, "flow/encoder/after_norm/w", as_numpy(state["flow.encoder.after_norm.weight"]), args.quant, stats=qstats)
    add_tensor_maybe_q(writer, "flow/encoder/after_norm/b", as_numpy(state["flow.encoder.after_norm.bias"]), args.quant, stats=qstats)
    decoder_keys = sorted(k for k in state if k.startswith("flow.decoder.estimator."))
    for k in decoder_keys:
        gguf_name = k.replace("flow.decoder.estimator.", "cfm/").replace(".", "/")
        add_tensor_maybe_q(writer, gguf_name, as_numpy(state[k], dtype=torch.float32), args.quant, stats=qstats)
    mel2wav_keys = sorted(k for k in state if k.startswith("mel2wav."))
    for k in mel2wav_keys:
        gguf_name = k.replace("mel2wav.", "hift/").replace(".", "/")
        add_tensor_maybe_q(writer, gguf_name, as_numpy(state[k], dtype=torch.float32), args.quant, stats=qstats)
    import librosa
    mel_fb_24k_80 = librosa.filters.mel(
        sr=24000, n_fft=1920, n_mels=80, fmin=0, fmax=8000,
    ).astype(np.float32)
    add_tensor_maybe_q(writer, "s3gen/mel_fb/24k_80", np.ascontiguousarray(mel_fb_24k_80), args.quant, stats=qstats)
    speaker_keys = [k for k in state if k.startswith("speaker_encoder.")]
    if not speaker_keys:
        print(f"warning: no speaker_encoder.* tensors found in {ckpt_path}")
    else:
        BN_EPS = 1e-5
        bn_groups: dict[str, dict[str, torch.Tensor]] = {}
        for k in speaker_keys:
            parts = k.rsplit(".", 1)
            if len(parts) == 2 and parts[1] in ("weight", "bias", "running_mean",
                                                "running_var", "num_batches_tracked"):
                bn_groups.setdefault(parts[0], {})[parts[1]] = state[k]
        bn_prefixes = {p for p, t in bn_groups.items()
                       if "running_mean" in t and "running_var" in t}
        n_bn = 0
        n_conv = 0
        for k in speaker_keys:
            parts = k.rsplit(".", 1)
            prefix, last = (parts[0], parts[1]) if len(parts) == 2 else (k, "")
            if last == "num_batches_tracked":
                continue
            gguf_base = "campplus/" + prefix.removeprefix("speaker_encoder.").replace(".", "/")
            if prefix in bn_prefixes:
                if last in ("weight", "bias"):
                    continue
                if last == "running_var":
                    continue
                if last == "running_mean":
                    grp = bn_groups[prefix]
                    mean = grp["running_mean"].float()
                    var  = grp["running_var"].float()
                    denom = torch.sqrt(var + BN_EPS)
                    if "weight" in grp and "bias" in grp:
                        gamma = grp["weight"].float()
                        beta  = grp["bias"].float()
                        scale = gamma / denom
                        shift = beta - mean * scale
                    else:
                        scale = 1.0 / denom
                        shift = -mean * scale
                    add_tensor_maybe_q(writer, gguf_base + "/s",
                                       np.ascontiguousarray(scale.numpy().astype(np.float32)),
                                       args.quant, stats=qstats)
                    add_tensor_maybe_q(writer, gguf_base + "/b",
                                       np.ascontiguousarray(shift.numpy().astype(np.float32)),
                                       args.quant, stats=qstats)
                    n_bn += 1
                continue
            gguf_name = "campplus/" + k.removeprefix("speaker_encoder.").replace(".", "/")
            add_tensor_maybe_q(writer, gguf_name, as_numpy(state[k], dtype=torch.float32), args.quant, stats=qstats)
            n_conv += 1
        writer.add_uint32("campplus.feat_dim",         80)
        writer.add_uint32("campplus.embedding_size",   192)
        writer.add_uint32("campplus.growth_rate",      32)
        writer.add_uint32("campplus.bn_size",          4)
        writer.add_uint32("campplus.init_channels",    128)
        writer.add_uint32("campplus.block1_layers",    12)
        writer.add_uint32("campplus.block2_layers",    24)
        writer.add_uint32("campplus.block3_layers",    16)
        writer.add_uint32("campplus.block1_dilation",  1)
        writer.add_uint32("campplus.block2_dilation",  2)
        writer.add_uint32("campplus.block3_dilation",  2)
        writer.add_uint32("campplus.kernel_size",      3)
        writer.add_uint32("campplus.seg_pool_len",     100)
        writer.add_uint32("campplus.sample_rate",      16000)
        SR = 16000
        NFFT = 512
        N_MELS = 80
        LOW = 20.0
        HIGH = 8000.0
        mel_low  = 1127.0 * np.log(1.0 + LOW  / 700.0)
        mel_high = 1127.0 * np.log(1.0 + HIGH / 700.0)
        mel_delta = (mel_high - mel_low) / (N_MELS + 1)
        bin_freq  = np.arange(NFFT // 2 + 1, dtype=np.float64) * SR / NFFT
        bin_mel   = 1127.0 * np.log(1.0 + bin_freq / 700.0)
        kaldi_fb  = np.zeros((N_MELS, NFFT // 2 + 1), dtype=np.float32)
        for m in range(N_MELS):
            mel_center = mel_low + (m + 1) * mel_delta
            mel_lo = mel_center - mel_delta
            mel_hi = mel_center + mel_delta
            for k, mb in enumerate(bin_mel):
                if mb < mel_lo or mb > mel_hi:
                    continue
                if mb <= mel_center:
                    kaldi_fb[m, k] = (mb - mel_lo) / (mel_center - mel_lo)
                else:
                    kaldi_fb[m, k] = (mel_hi - mb) / (mel_hi - mel_center)
        add_tensor_maybe_q(writer, "campplus/mel_fb_kaldi_80", np.ascontiguousarray(kaldi_fb), args.quant, stats=qstats)
        print(f"Embedded CAMPPlus: {n_conv} conv/linear tensors + {n_bn} fused BNs "
              f"+ kaldi mel filterbank {kaldi_fb.shape}")
    tok_keys = [k for k in state if k.startswith("tokenizer.")]
    if not tok_keys:
        print(f"warning: no tokenizer.* tensors found in {ckpt_path}")
    else:
        n_tok = 0
        for k in tok_keys:
            rest = k[len("tokenizer."):]
            if rest in ("window",):
                continue
            if rest == "_mel_filters":
                gguf_name = "s3tokv2/mel_fb"
            else:
                gguf_name = "s3tokv2/" + rest.replace(".", "/")
            add_tensor_maybe_q(writer, gguf_name, as_numpy(state[k], dtype=torch.float32), args.quant, stats=qstats)
            n_tok += 1
        if "tokenizer._mel_filters" not in state:
            fb = librosa.filters.mel(
                sr=16000, n_fft=400, n_mels=128, fmin=0, fmax=8000,
            ).astype(np.float32)
            add_tensor_maybe_q(writer, "s3tokv2/mel_fb", np.ascontiguousarray(fb), args.quant, stats=qstats)
            n_tok += 1
            print(f"Synthesized s3tokv2/mel_fb {fb.shape} (checkpoint has no tokenizer._mel_filters)")
        writer.add_uint32("s3tokv2.n_mels",        128)
        writer.add_uint32("s3tokv2.n_audio_state", 1280)
        writer.add_uint32("s3tokv2.n_audio_head",  20)
        writer.add_uint32("s3tokv2.n_audio_layer", 6)
        writer.add_uint32("s3tokv2.head_dim",      64)
        writer.add_uint32("s3tokv2.mlp_ratio",     4)
        writer.add_uint32("s3tokv2.fsmn_kernel",   31)
        writer.add_uint32("s3tokv2.fsq_levels",    3)
        writer.add_uint32("s3tokv2.fsq_dim",       8)
        writer.add_uint32("s3tokv2.codebook_size", 3 ** 8)
        writer.add_uint32("s3tokv2.conv_stride",   2)
        writer.add_uint32("s3tokv2.n_fft",         400)
        writer.add_uint32("s3tokv2.hop",           160)
        writer.add_uint32("s3tokv2.sample_rate",   16000)
        writer.add_float32("s3tokv2.rope_theta",   10000.0)
        writer.add_uint32("s3tokv2.rope_max_pos",  2048)
        print(f"Embedded S3TokenizerV2: {n_tok} tensors")
    n_flow = sum(1 for k in state if k.startswith("flow.")) - sum(1 for k in state if k.startswith("flow.decoder.estimator."))
    n_cfm  = len(decoder_keys)
    n_hift = len(mel2wav_keys)
    print(f"Wrote: encoder(+proj)~{n_flow} tensors, cfm={n_cfm}, hift={n_hift}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    out_size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"\nOutput: {args.out} ({out_size_mb:.0f} MB)")
    if args.quant == "q4_0" and qstats is not None:
        print(f"  --quant {args.quant}: {qstats['n_quant']} tensors block-quantized "
              f"(shared policy; embeddings, voice encoders, "
              f"norms/biases, and filterbanks kept at full precision)")
if __name__ == "__main__":
    main()
