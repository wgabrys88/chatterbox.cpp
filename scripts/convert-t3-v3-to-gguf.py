#!/usr/bin/env python3
import argparse, json, math, re
from pathlib import Path
import gguf, numpy as np, torch
from quant_policy import WEIGHT_TYPES, add_weight
from tokenizers import Tokenizer
from safetensors.torch import load_file
SPEECH_VOCAB_SIZE = 8194
START_SPEECH_TOKEN, STOP_SPEECH_TOKEN, SPEAKER_EMBED_SIZE = 6561, 6562, 256
MAX_GENERATION_TOKENS = 4096
ROPE_THETA, ROPE_ORIG_CTX = 500000.0, 8192
ROPE_FACTOR, ROPE_HIGH, ROPE_LOW = 8.0, 4.0, 1.0
LAYER_RE = re.compile(r"^tfmr\.layers\.(\d+)\.(.+)$")
SKIP = {"tfmr.embed_tokens.weight", "text_head.weight"}
from convert_common import as_numpy, finish_t3

def llama3_freq_factors(n_dims, base, factor, low_freq_factor, high_freq_factor, old_context_len):
    n = n_dims // 2
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        inv = base ** (-2.0 * i / n_dims)
        wavelen = 2.0 * math.pi / inv
        low_wl = old_context_len / low_freq_factor
        high_wl = old_context_len / high_freq_factor
        if wavelen < high_wl:
            new_inv = inv
        elif wavelen > low_wl:
            new_inv = inv / factor
        else:
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_inv = (1.0 - smooth) * inv / factor + smooth * inv
        out[i] = np.float32(inv / new_inv)
    return out
MATRIX_SUFFIXES = ("/attn/q/w", "/attn/k/w", "/attn/v/w", "/attn/o/w", "/ffn/gate/w", "/ffn/up/w", "/ffn/down/w")

def add(writer, name, array, matrix_type):
    is_matrix = name.startswith("model/h") and name.endswith(MATRIX_SUFFIXES)
    add_weight(writer, name, array, matrix_type, force_f32=not is_matrix)

def tokenizer_contract(ckpt_dir, text_vocab_size):
    tokenizer_path = ckpt_dir / "grapheme_mtl_merged_expanded_v1.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab = tokenizer.get_vocab()
    language_tokens = sorted(t for t in vocab if re.fullmatch(r"\[[a-z]{2,3}\]", t))
    return {
        "language_tokens": language_tokens,
        "start": int(vocab["[START]"]),
        "stop": int(vocab["[STOP]"]),
    }
def map_name(name):
    table = {
        "tfmr.norm.weight": ("model/norm/g", torch.float32),
        "text_emb.weight": ("chatterbox/text_emb", torch.float32),
        "speech_emb.weight": ("chatterbox/speech_emb", torch.float32),
        "speech_head.weight": ("chatterbox/speech_head", torch.float32),
        "text_pos_emb.emb.weight": ("chatterbox/text_pos_emb", torch.float32),
        "speech_pos_emb.emb.weight": ("chatterbox/speech_pos_emb", torch.float32),
        "cond_enc.spkr_enc.weight": ("chatterbox/cond_spkr/w", torch.float32),
        "cond_enc.spkr_enc.bias": ("chatterbox/cond_spkr/b", torch.float32),
        "cond_enc.emotion_adv_fc.weight": ("chatterbox/emotion_adv_fc/w", torch.float32),
        "cond_enc.perceiver.pre_attention_query": ("chatterbox/perceiver/pre_attention_query", torch.float32),
        "cond_enc.perceiver.attn.norm.weight": ("chatterbox/perceiver/attn/norm/g", torch.float32),
        "cond_enc.perceiver.attn.norm.bias": ("chatterbox/perceiver/attn/norm/b", torch.float32),
        "cond_enc.perceiver.attn.to_q.weight": ("chatterbox/perceiver/attn/to_q/w", torch.float32),
        "cond_enc.perceiver.attn.to_q.bias": ("chatterbox/perceiver/attn/to_q/b", torch.float32),
        "cond_enc.perceiver.attn.to_k.weight": ("chatterbox/perceiver/attn/to_k/w", torch.float32),
        "cond_enc.perceiver.attn.to_k.bias": ("chatterbox/perceiver/attn/to_k/b", torch.float32),
        "cond_enc.perceiver.attn.to_v.weight": ("chatterbox/perceiver/attn/to_v/w", torch.float32),
        "cond_enc.perceiver.attn.to_v.bias": ("chatterbox/perceiver/attn/to_v/b", torch.float32),
        "cond_enc.perceiver.attn.proj_out.weight": ("chatterbox/perceiver/attn/proj_out/w", torch.float32),
        "cond_enc.perceiver.attn.proj_out.bias": ("chatterbox/perceiver/attn/proj_out/b", torch.float32),
    }
    if name in table: return table[name]
    if name in SKIP: return None
    m = LAYER_RE.match(name)
    if not m: return None
    layers = {
        "input_layernorm.weight": "model/h{}/attn_norm/g",
        "post_attention_layernorm.weight": "model/h{}/ffn_norm/g",
        "self_attn.q_proj.weight": "model/h{}/attn/q/w",
        "self_attn.k_proj.weight": "model/h{}/attn/k/w",
        "self_attn.v_proj.weight": "model/h{}/attn/v/w",
        "self_attn.o_proj.weight": "model/h{}/attn/o/w",
        "mlp.gate_proj.weight": "model/h{}/ffn/gate/w",
        "mlp.up_proj.weight": "model/h{}/ffn/up/w",
        "mlp.down_proj.weight": "model/h{}/ffn/down/w",
    }
    if m.group(2) not in layers: return None
    return layers[m.group(2)].format(int(m.group(1))), torch.float32
def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt_dir")
    p.add_argument("out")
    p.add_argument("t3_safetensors")
    p.add_argument("--matrix-type", required=True, choices=WEIGHT_TYPES)
    a = p.parse_args()
    ckpt_dir, out = Path(a.ckpt_dir), Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    state = load_file(ckpt_dir / a.t3_safetensors)
    conds = torch.load(ckpt_dir / "conds.pt", map_location="cpu", weights_only=True)
    n_embd = int(state["tfmr.norm.weight"].shape[0])
    n_layer = max(int(m.group(1)) for name in state if (m := LAYER_RE.match(name))) + 1
    n_head = n_embd // 64
    n_ff = int(state["tfmr.layers.0.mlp.gate_proj.weight"].shape[0])
    perceiver_len = int(state["cond_enc.perceiver.pre_attention_query"].shape[1])
    text_pos_len = int(state["text_pos_emb.emb.weight"].shape[0])
    text_vocab_size = int(state["text_emb.weight"].shape[0])
    speech_pos_len = int(state["speech_pos_emb.emb.weight"].shape[0])

    n_ctx = 1 + perceiver_len + 1 + text_pos_len + 2 + MAX_GENERATION_TOKENS
    tokenizer = tokenizer_contract(ckpt_dir, text_vocab_size)
    writer = gguf.GGUFWriter(str(out), "chatterbox")
    writer.add_uint32("chatterbox.n_ctx", n_ctx)
    writer.add_uint32("chatterbox.n_embd", n_embd)
    writer.add_uint32("chatterbox.n_head", n_head)
    writer.add_uint32("chatterbox.n_layer", n_layer)
    writer.add_uint32("chatterbox.n_ff", n_ff)
    writer.add_uint32("chatterbox.n_batch", 2)
    writer.add_uint32("chatterbox.perceiver_len", perceiver_len)
    writer.add_uint32("chatterbox.text_vocab_size", text_vocab_size)
    writer.add_uint32("chatterbox.speech_vocab_size", SPEECH_VOCAB_SIZE)
    writer.add_uint32("chatterbox.start_text_token", tokenizer["start"])
    writer.add_uint32("chatterbox.stop_text_token", tokenizer["stop"])
    writer.add_uint32("chatterbox.start_speech_token", START_SPEECH_TOKEN)
    writer.add_uint32("chatterbox.stop_speech_token", STOP_SPEECH_TOKEN)
    writer.add_uint32("chatterbox.speaker_embed_size", SPEAKER_EMBED_SIZE)
    writer.add_float32("chatterbox.layer_norm_eps", 1e-5)
    writer.add_float32("chatterbox.rope_theta", ROPE_THETA)
    writer.add_uint32("chatterbox.rope_orig_ctx", ROPE_ORIG_CTX)
    writer.add_uint32("chatterbox.text_frontend_version", 4)
    writer.add_string("chatterbox.conversion.matrix_type", a.matrix_type)
    writer.add_string("chatterbox.tokenizer.language_tokens", ",".join(tokenizer["language_tokens"]))
    for name, tensor in state.items():
        mapped = map_name(name)
        if mapped is None: continue
        gguf_name, dtype = mapped
        add(writer, gguf_name, as_numpy(tensor, dtype=dtype), a.matrix_type)
    writer.add_tensor("model/rope_freq_factors", llama3_freq_factors(64, ROPE_THETA, ROPE_FACTOR, ROPE_LOW, ROPE_HIGH, ROPE_ORIG_CTX))
    finish_t3(writer, ckpt_dir, conds)
if __name__ == "__main__":
    main()
