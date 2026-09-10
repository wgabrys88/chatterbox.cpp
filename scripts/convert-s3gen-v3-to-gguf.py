#!/usr/bin/env python3
import json, re, sys
from pathlib import Path
import gguf, numpy as np, torch
from safetensors.torch import load_file
from quant_policy import QUANT_TYPE, should_quantize
QUANT = "q4_0"
TEXT_VOCAB_SIZE = 2454
def as_numpy(tensor, *, dtype=None):
    if dtype is not None: tensor = tensor.to(dtype)
    return np.ascontiguousarray(tensor.detach().cpu().numpy())
def resolve_weight_norm(state, prefix):
    g = state[f"{prefix}.parametrizations.weight.original0"]
    v = state[f"{prefix}.parametrizations.weight.original1"]
    norm = v.flatten(1).norm(dim=1).view(-1, *([1] * (v.ndim - 1)))
    return g * v / norm
def expand_weight_norm(state):
    out = dict(state)
    prefixes = {m.group(1) for k in state if (m := re.match(r"(.+)\.parametrizations\.weight\.original0$", k))}
    for p in prefixes:
        out[f"{p}.weight"] = resolve_weight_norm(state, p)
        out.pop(f"{p}.parametrizations.weight.original0", None)
        out.pop(f"{p}.parametrizations.weight.original1", None)
    return out
def must_f32(name):
    return any(s in name for s in ("flow/input_embedding", "flow/spk_embed_affine/", "/builtin/", "s3gen/mel_fb/", "campplus/", "s3tokv2/"))
def add(writer, name, arr):
    if arr.dtype.kind in "iu" or np.issubdtype(arr.dtype, np.integer):
        writer.add_tensor(name, arr); return
    if must_f32(name):
        writer.add_tensor(name, np.ascontiguousarray(arr.astype(np.float32))); return
    qtype = QUANT_TYPE[QUANT]
    if not should_quantize(name, arr.shape, qtype):
        writer.add_tensor(name, np.ascontiguousarray(arr.astype(np.float16)) if arr.ndim == 3 else arr)
        return
    qdata = gguf.quants.quantize(np.ascontiguousarray(arr.astype(np.float32)), qtype)
    writer.add_tensor(name, qdata, raw_shape=qdata.shape, raw_dtype=qtype)
def export_conformer_block(writer, state, prefix, gguf_prefix):
    mapping = {
        "norm_mha.weight": "norm_mha/w", "norm_mha.bias": "norm_mha/b",
        "norm_ff.weight": "norm_ff/w", "norm_ff.bias": "norm_ff/b",
        "self_attn.linear_q.weight": "attn/q/w", "self_attn.linear_q.bias": "attn/q/b",
        "self_attn.linear_k.weight": "attn/k/w", "self_attn.linear_k.bias": "attn/k/b",
        "self_attn.linear_v.weight": "attn/v/w", "self_attn.linear_v.bias": "attn/v/b",
        "self_attn.linear_out.weight": "attn/o/w", "self_attn.linear_out.bias": "attn/o/b",
        "self_attn.linear_pos.weight": "attn/pos/w",
        "self_attn.pos_bias_u": "attn/pos_bias_u", "self_attn.pos_bias_v": "attn/pos_bias_v",
        "feed_forward.w_1.weight": "ff/w1/w", "feed_forward.w_1.bias": "ff/w1/b",
        "feed_forward.w_2.weight": "ff/w2/w", "feed_forward.w_2.bias": "ff/w2/b",
    }
    for src_suffix, dst_suffix in mapping.items():
        add(writer, f"{gguf_prefix}/{dst_suffix}", as_numpy(state[f"{prefix}.{src_suffix}"], dtype=torch.float32))
def main():
    ckpt_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    tok = json.loads((ckpt_dir / "grapheme_mtl_merged_expanded_v1.json").read_text(encoding="utf-8"))
    vocab = tok["model"]["vocab"]
    n_tok = max(int(i) for i in vocab.values()) + 1
    if n_tok != TEXT_VOCAB_SIZE:
        raise SystemExit(f"tokenizer {n_tok} != {TEXT_VOCAB_SIZE}")
    raw = load_file(ckpt_dir / "s3gen.safetensors")
    allowed = ("flow.", "mel2wav.", "speaker_encoder.", "tokenizer.")
    for name in sorted(raw):
        print(f"{name}\t{tuple(raw[name].shape)}", flush=True)
    unknown = [name for name in raw if not name.startswith(allowed)]
    if unknown:
        print("STOP unknown s3gen keys:", file=sys.stderr)
        for name in unknown:
            print(f"  {name}\t{tuple(raw[name].shape)}", file=sys.stderr)
        raise SystemExit("s3gen keys differ")
    mixer = [name for name in raw if "time_embed_mixer" in name]
    if mixer:
        raise SystemExit("time_embed_mixer present")
    state = expand_weight_norm(raw)
    gen = torch.load(ckpt_dir / "conds.pt", map_location="cpu", weights_only=True)["gen"]
    writer = gguf.GGUFWriter(str(out), "chatterbox-s3gen")
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
    add(writer, "s3gen/builtin/prompt_token", as_numpy(prompt_token))
    add(writer, "s3gen/builtin/prompt_feat", as_numpy(prompt_feat, dtype=torch.float32))
    add(writer, "s3gen/builtin/embedding", as_numpy(embedding, dtype=torch.float32))
    add(writer, "flow/input_embedding", as_numpy(state["flow.input_embedding.weight"]))
    add(writer, "flow/spk_embed_affine/w", as_numpy(state["flow.spk_embed_affine_layer.weight"]))
    add(writer, "flow/spk_embed_affine/b", as_numpy(state["flow.spk_embed_affine_layer.bias"]))
    add(writer, "flow/encoder_proj/w", as_numpy(state["flow.encoder_proj.weight"]))
    add(writer, "flow/encoder_proj/b", as_numpy(state["flow.encoder_proj.bias"]))
    add(writer, "flow/encoder/embed/linear/w", as_numpy(state["flow.encoder.embed.out.0.weight"]))
    add(writer, "flow/encoder/embed/linear/b", as_numpy(state["flow.encoder.embed.out.0.bias"]))
    add(writer, "flow/encoder/embed/norm/w", as_numpy(state["flow.encoder.embed.out.1.weight"]))
    add(writer, "flow/encoder/embed/norm/b", as_numpy(state["flow.encoder.embed.out.1.bias"]))
    add(writer, "flow/encoder/pre_lookahead/conv1/w", as_numpy(state["flow.encoder.pre_lookahead_layer.conv1.weight"]))
    add(writer, "flow/encoder/pre_lookahead/conv1/b", as_numpy(state["flow.encoder.pre_lookahead_layer.conv1.bias"]))
    add(writer, "flow/encoder/pre_lookahead/conv2/w", as_numpy(state["flow.encoder.pre_lookahead_layer.conv2.weight"]))
    add(writer, "flow/encoder/pre_lookahead/conv2/b", as_numpy(state["flow.encoder.pre_lookahead_layer.conv2.bias"]))
    for i in range(6):
        export_conformer_block(writer, state, f"flow.encoder.encoders.{i}", f"flow/encoder/block{i}")
    add(writer, "flow/encoder/up_layer/conv/w", as_numpy(state["flow.encoder.up_layer.conv.weight"]))
    add(writer, "flow/encoder/up_layer/conv/b", as_numpy(state["flow.encoder.up_layer.conv.bias"]))
    add(writer, "flow/encoder/up_embed/linear/w", as_numpy(state["flow.encoder.up_embed.out.0.weight"]))
    add(writer, "flow/encoder/up_embed/linear/b", as_numpy(state["flow.encoder.up_embed.out.0.bias"]))
    add(writer, "flow/encoder/up_embed/norm/w", as_numpy(state["flow.encoder.up_embed.out.1.weight"]))
    add(writer, "flow/encoder/up_embed/norm/b", as_numpy(state["flow.encoder.up_embed.out.1.bias"]))
    for i in range(4):
        export_conformer_block(writer, state, f"flow.encoder.up_encoders.{i}", f"flow/encoder/up_block{i}")
    add(writer, "flow/encoder/after_norm/w", as_numpy(state["flow.encoder.after_norm.weight"]))
    add(writer, "flow/encoder/after_norm/b", as_numpy(state["flow.encoder.after_norm.bias"]))
    for k in sorted(k for k in state if k.startswith("flow.decoder.estimator.")):
        add(writer, k.replace("flow.decoder.estimator.", "cfm/").replace(".", "/"), as_numpy(state[k], dtype=torch.float32))
    for k in sorted(k for k in state if k.startswith("mel2wav.")):
        add(writer, k.replace("mel2wav.", "hift/").replace(".", "/"), as_numpy(state[k], dtype=torch.float32))
    import librosa
    add(writer, "s3gen/mel_fb/24k_80", np.ascontiguousarray(librosa.filters.mel(sr=24000, n_fft=1920, n_mels=80, fmin=0, fmax=8000).astype(np.float32)))
    speaker_keys = [k for k in state if k.startswith("speaker_encoder.")]
    BN_EPS = 1e-5
    bn_groups = {}
    for k in speaker_keys:
        parts = k.rsplit(".", 1)
        if len(parts) == 2 and parts[1] in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
            bn_groups.setdefault(parts[0], {})[parts[1]] = state[k]
    bn_prefixes = {p for p, t in bn_groups.items() if "running_mean" in t and "running_var" in t}
    for k in speaker_keys:
        parts = k.rsplit(".", 1)
        prefix, last = (parts[0], parts[1]) if len(parts) == 2 else (k, "")
        if last == "num_batches_tracked": continue
        gguf_base = "campplus/" + prefix.removeprefix("speaker_encoder.").replace(".", "/")
        if prefix in bn_prefixes:
            if last in ("weight", "bias", "running_var"): continue
            if last == "running_mean":
                grp = bn_groups[prefix]
                mean, var = grp["running_mean"].float(), grp["running_var"].float()
                denom = torch.sqrt(var + BN_EPS)
                gamma = grp["weight"].float() if "weight" in grp else torch.ones_like(mean)
                beta = grp["bias"].float() if "bias" in grp else torch.zeros_like(mean)
                scale = gamma / denom
                shift = beta - mean * scale
                add(writer, gguf_base + "/s", np.ascontiguousarray(scale.numpy().astype(np.float32)))
                add(writer, gguf_base + "/b", np.ascontiguousarray(shift.numpy().astype(np.float32)))
            continue
        add(writer, "campplus/" + k.removeprefix("speaker_encoder.").replace(".", "/"), as_numpy(state[k], dtype=torch.float32))
    writer.add_uint32("campplus.feat_dim", 80)
    writer.add_uint32("campplus.embedding_size", 192)
    writer.add_uint32("campplus.growth_rate", 32)
    writer.add_uint32("campplus.bn_size", 4)
    writer.add_uint32("campplus.init_channels", 128)
    writer.add_uint32("campplus.block1_layers", 12)
    writer.add_uint32("campplus.block2_layers", 24)
    writer.add_uint32("campplus.block3_layers", 16)
    writer.add_uint32("campplus.block1_dilation", 1)
    writer.add_uint32("campplus.block2_dilation", 2)
    writer.add_uint32("campplus.block3_dilation", 2)
    writer.add_uint32("campplus.kernel_size", 3)
    writer.add_uint32("campplus.seg_pool_len", 100)
    writer.add_uint32("campplus.sample_rate", 16000)
    SR, NFFT, N_MELS, LOW, HIGH = 16000, 512, 80, 20.0, 8000.0
    mel_low = 1127.0 * np.log(1.0 + LOW / 700.0)
    mel_high = 1127.0 * np.log(1.0 + HIGH / 700.0)
    mel_delta = (mel_high - mel_low) / (N_MELS + 1)
    bin_freq = np.arange(NFFT // 2 + 1, dtype=np.float64) * SR / NFFT
    bin_mel = 1127.0 * np.log(1.0 + bin_freq / 700.0)
    kaldi_fb = np.zeros((N_MELS, NFFT // 2 + 1), dtype=np.float32)
    for m in range(N_MELS):
        mel_center = mel_low + (m + 1) * mel_delta
        mel_lo, mel_hi = mel_center - mel_delta, mel_center + mel_delta
        for k, mb in enumerate(bin_mel):
            if mb < mel_lo or mb > mel_hi: continue
            kaldi_fb[m, k] = (mb - mel_lo) / (mel_center - mel_lo) if mb <= mel_center else (mel_hi - mb) / (mel_hi - mel_center)
    add(writer, "campplus/mel_fb_kaldi_80", np.ascontiguousarray(kaldi_fb))
    for k in [k for k in state if k.startswith("tokenizer.")]:
        rest = k[len("tokenizer."):]
        if rest in ("window", "_mel_filters"): continue
        add(writer, "s3tokv2/" + rest.replace(".", "/"), as_numpy(state[k], dtype=torch.float32))
    add(writer, "s3tokv2/mel_fb", np.ascontiguousarray(librosa.filters.mel(sr=16000, n_fft=400, n_mels=128, fmin=0, fmax=8000).astype(np.float32)))
    writer.add_uint32("s3tokv2.n_mels", 128)
    writer.add_uint32("s3tokv2.n_audio_state", 1280)
    writer.add_uint32("s3tokv2.n_audio_head", 20)
    writer.add_uint32("s3tokv2.n_audio_layer", 6)
    writer.add_uint32("s3tokv2.head_dim", 64)
    writer.add_uint32("s3tokv2.mlp_ratio", 4)
    writer.add_uint32("s3tokv2.fsmn_kernel", 31)
    writer.add_uint32("s3tokv2.fsq_levels", 3)
    writer.add_uint32("s3tokv2.fsq_dim", 8)
    writer.add_uint32("s3tokv2.codebook_size", 3 ** 8)
    writer.add_uint32("s3tokv2.conv_stride", 2)
    writer.add_uint32("s3tokv2.n_fft", 400)
    writer.add_uint32("s3tokv2.hop", 160)
    writer.add_uint32("s3tokv2.sample_rate", 16000)
    writer.add_float32("s3tokv2.rope_theta", 10000.0)
    writer.add_uint32("s3tokv2.rope_max_pos", 2048)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
if __name__ == "__main__":
    main()
