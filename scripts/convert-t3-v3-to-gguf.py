#!/usr/bin/env python3
import json, math, re, sys
from pathlib import Path
import gguf, numpy as np, torch
from safetensors.torch import load_file
TEXT_VOCAB_SIZE, SPEECH_VOCAB_SIZE = 2454, 8194
START_TEXT_TOKEN, STOP_TEXT_TOKEN = 255, 0
START_SPEECH_TOKEN, STOP_SPEECH_TOKEN, SPEAKER_EMBED_SIZE = 6561, 6562, 256
N_PREDICT = 1000
ROPE_THETA, ROPE_ORIG_CTX = 500000.0, 8192
ROPE_FACTOR, ROPE_HIGH, ROPE_LOW = 8.0, 4.0, 1.0
LAYER_RE = re.compile(r"^tfmr\.layers\.(\d+)\.(.+)$")
QTYPE = gguf.GGMLQuantizationType.Q8_0
SKIP = {"tfmr.embed_tokens.weight", "text_head.weight"}
def as_numpy(tensor, *, dtype=None):
    if dtype is not None: tensor = tensor.to(dtype)
    return np.ascontiguousarray(tensor.detach().cpu().numpy())
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
def quantizable(name):
    if name == "chatterbox/speech_head": return True
    if not name.startswith("model/h"): return False
    return name.endswith(("/attn/q/w", "/attn/k/w", "/attn/v/w", "/attn/o/w", "/ffn/gate/w", "/ffn/up/w", "/ffn/down/w"))
def add(writer, name, array):
    if not quantizable(name):
        writer.add_tensor(name, array)
        return
    qdata = gguf.quants.quantize(array.astype(np.float32), QTYPE)
    writer.add_tensor(name, qdata, raw_shape=qdata.shape, raw_dtype=QTYPE)
def tokenizer(ckpt_dir):
    tok = json.loads((ckpt_dir / "grapheme_mtl_merged_expanded_v1.json").read_text(encoding="utf-8"))
    vocab = tok["model"]["vocab"]
    if not isinstance(vocab, dict):
        raise SystemExit("tokenizer vocab")
    id_to_tok = {int(i): t for t, i in vocab.items()}
    added_ids = set()
    for a in tok.get("added_tokens", []):
        added_ids.add(int(a["id"]))
        id_to_tok[int(a["id"])] = a["content"]
    n = max(id_to_tok) + 1
    if n != TEXT_VOCAB_SIZE:
        raise SystemExit(f"tokenizer {n} != {TEXT_VOCAB_SIZE}")
    tokens, types = [], []
    for i in range(n):
        tok_s = id_to_tok[i]
        tokens.append(tok_s)
        types.append(int(gguf.TokenType.USER_DEFINED if i in added_ids else gguf.TokenType.NORMAL))
    merges = []
    for m in tok["model"]["merges"]:
        if isinstance(m, str):
            merges.append(m)
        elif isinstance(m, list) and len(m) == 2:
            merges.append(m[0] + " " + m[1])
        else:
            raise SystemExit("merge")
    return tokens, types, merges
def map_name(name):
    table = {
        "tfmr.norm.weight": ("model/norm/g", torch.float32),
        "text_emb.weight": ("chatterbox/text_emb", torch.float16),
        "speech_emb.weight": ("chatterbox/speech_emb", torch.float16),
        "speech_head.weight": ("chatterbox/speech_head", torch.float16),
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
    return layers[m.group(2)].format(int(m.group(1))), torch.float16
def main():
    ckpt_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    state = load_file(ckpt_dir / "t3_mtl23ls_v3.safetensors")
    for name in sorted(state):
        print(f"{name}\t{tuple(state[name].shape)}", flush=True)
    unknown = [name for name in state if name not in SKIP and map_name(name) is None]
    if unknown:
        print("STOP unknown keys:", file=sys.stderr)
        for name in unknown:
            print(f"  {name}\t{tuple(state[name].shape)}", file=sys.stderr)
        raise SystemExit("v3 t3 keys differ")
    conds = torch.load(ckpt_dir / "conds.pt", map_location="cpu", weights_only=True)
    n_embd = int(state["tfmr.norm.weight"].shape[0])
    n_layer = max(int(m.group(1)) for name in state if (m := LAYER_RE.match(name))) + 1
    n_head = n_embd // 64
    if n_embd % 64:
        raise SystemExit(f"n_head {n_embd}//64")
    n_ff = int(state["tfmr.layers.0.mlp.gate_proj.weight"].shape[0])
    perceiver_len = int(state["cond_enc.perceiver.pre_attention_query"].shape[1])
    text_pos_len = int(state["text_pos_emb.emb.weight"].shape[0])
    n_ctx = 1 + perceiver_len + 1 + text_pos_len + 2 + N_PREDICT
    tokens, types, merges = tokenizer(ckpt_dir)
    writer = gguf.GGUFWriter(str(out), "chatterbox")
    writer.add_uint32("chatterbox.n_ctx", n_ctx)
    writer.add_uint32("chatterbox.n_embd", n_embd)
    writer.add_uint32("chatterbox.n_head", n_head)
    writer.add_uint32("chatterbox.n_layer", n_layer)
    writer.add_uint32("chatterbox.n_ff", n_ff)
    writer.add_uint32("chatterbox.n_batch", 2)
    writer.add_uint32("chatterbox.perceiver_len", perceiver_len)
    writer.add_uint32("chatterbox.text_vocab_size", TEXT_VOCAB_SIZE)
    writer.add_uint32("chatterbox.speech_vocab_size", SPEECH_VOCAB_SIZE)
    writer.add_uint32("chatterbox.start_text_token", START_TEXT_TOKEN)
    writer.add_uint32("chatterbox.stop_text_token", STOP_TEXT_TOKEN)
    writer.add_uint32("chatterbox.start_speech_token", START_SPEECH_TOKEN)
    writer.add_uint32("chatterbox.stop_speech_token", STOP_SPEECH_TOKEN)
    writer.add_uint32("chatterbox.speaker_embed_size", SPEAKER_EMBED_SIZE)
    writer.add_float32("chatterbox.layer_norm_eps", 1e-5)
    writer.add_float32("chatterbox.rope_theta", ROPE_THETA)
    writer.add_uint32("chatterbox.rope_orig_ctx", ROPE_ORIG_CTX)
    writer.add_tokenizer_model("hf-bpe")
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    writer.add_token_merges(merges)
    for name, tensor in state.items():
        mapped = map_name(name)
        if mapped is None: continue
        gguf_name, dtype = mapped
        add(writer, gguf_name, as_numpy(tensor, dtype=dtype))
    writer.add_tensor("model/rope_freq_factors", llama3_freq_factors(64, ROPE_THETA, ROPE_FACTOR, ROPE_LOW, ROPE_HIGH, ROPE_ORIG_CTX))
    builtin_tokens = conds["t3"]["cond_prompt_speech_tokens"].reshape(-1).to(torch.int32)
    writer.add_uint32("chatterbox.cond_prompt_max", int(builtin_tokens.numel()))
    writer.add_uint32("chatterbox.cond_prompt_length", int(builtin_tokens.numel()))
    writer.add_tensor("chatterbox/builtin/speaker_emb", as_numpy(conds["t3"]["speaker_emb"].reshape(1, SPEAKER_EMBED_SIZE), dtype=torch.float32))
    writer.add_tensor("chatterbox/builtin/cond_prompt_speech_tokens", as_numpy(builtin_tokens))
    emotion = conds["t3"]["emotion_adv"].reshape(1).to(torch.float32)
    writer.add_tensor("chatterbox/builtin/emotion_adv", as_numpy(emotion))
    ve = load_file(ckpt_dir / "ve.safetensors")
    writer.add_uint32("voice_encoder.n_mels", 40)
    writer.add_uint32("voice_encoder.hidden_size", 256)
    writer.add_uint32("voice_encoder.num_layers", 3)
    writer.add_uint32("voice_encoder.embedding_size", 256)
    writer.add_uint32("voice_encoder.partial_frames", 160)
    writer.add_uint32("voice_encoder.sample_rate", 16000)
    writer.add_uint32("voice_encoder.n_fft", 400)
    writer.add_uint32("voice_encoder.hop_size", 160)
    writer.add_uint32("voice_encoder.win_size", 400)
    writer.add_float32("voice_encoder.overlap", 0.5)
    writer.add_float32("voice_encoder.rate", 1.3)
    writer.add_float32("voice_encoder.min_coverage", 0.8)
    for k, t in ve.items():
        if not k.startswith("similarity_"):
            writer.add_tensor(f"voice_encoder/{k.replace('.', '/')}", as_numpy(t, dtype=torch.float32))
    import librosa
    writer.add_tensor("voice_encoder/mel_fb", np.ascontiguousarray(librosa.filters.mel(sr=16000, n_fft=400, n_mels=40, fmin=0, fmax=8000).astype(np.float32)))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
if __name__ == "__main__":
    main()
