#!/usr/bin/env python3
import json, re, sys
from pathlib import Path
import gguf, numpy as np, torch
from safetensors.torch import load_file
TEXT_VOCAB_SIZE, SPEECH_VOCAB_SIZE = 50276, 6563
START_SPEECH_TOKEN, STOP_SPEECH_TOKEN, SPEAKER_EMBED_SIZE = 6561, 6562, 256
LAYER_RE = re.compile(r"^tfmr\.h\.(\d+)\.(.+)$")
QTYPE = gguf.GGMLQuantizationType.Q8_0
def as_numpy(tensor, *, dtype=None, transpose=False):
    if dtype is not None: tensor = tensor.to(dtype)
    array = tensor.detach().cpu().numpy()
    if transpose: array = array.T
    return np.ascontiguousarray(array)
def quantizable(name):
    if name == "chatterbox/speech_head": return True
    return name.startswith("model/h") and name.endswith(("/attn/c_attn/w", "/attn/c_proj/w", "/mlp/c_fc/w", "/mlp/c_proj/w"))
def add(writer, name, array):
    if not quantizable(name):
        writer.add_tensor(name, array)
        return
    qdata = gguf.quants.quantize(array.astype(np.float32), QTYPE)
    writer.add_tensor(name, qdata, raw_shape=qdata.shape, raw_dtype=QTYPE)
def tokenizer(ckpt_dir):
    vocab = json.loads((ckpt_dir / "vocab.json").read_text(encoding="utf-8"))
    added = json.loads((ckpt_dir / "added_tokens.json").read_text(encoding="utf-8"))
    id_to_tok = {int(i): t for t, i in vocab.items()}
    for t, i in added.items(): id_to_tok[int(i)] = t
    tokens, types = [], []
    for i in range(max(id_to_tok) + 1):
        tok = id_to_tok[i]
        tokens.append(tok)
        types.append(int(gguf.TokenType.USER_DEFINED if tok in added else gguf.TokenType.NORMAL))
    merges = [ln.rstrip("\r\n") for ln in (ckpt_dir / "merges.txt").read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")]
    return tokens, types, merges
def map_name(name):
    table = {
        "tfmr.wpe.weight": ("model/wpe", torch.float32, False),
        "tfmr.ln_f.weight": ("model/ln_f/g", torch.float32, False),
        "tfmr.ln_f.bias": ("model/ln_f/b", torch.float32, False),
        "text_emb.weight": ("chatterbox/text_emb", torch.float16, False),
        "speech_emb.weight": ("chatterbox/speech_emb", torch.float16, False),
        "speech_head.weight": ("chatterbox/speech_head", torch.float16, False),
        "speech_head.bias": ("chatterbox/speech_head_bias", torch.float32, False),
        "cond_enc.spkr_enc.weight": ("chatterbox/cond_spkr/w", torch.float32, False),
        "cond_enc.spkr_enc.bias": ("chatterbox/cond_spkr/b", torch.float32, False),
    }
    if name in table: return table[name]
    if name == "tfmr.wte.weight": return None
    # official T3 3f35dfc8: text_head is training forward()/loss only; inference_turbo never reads it
    if name == "text_head.weight": return None
    m = LAYER_RE.match(name)
    if not m: return None
    layers = {
        "ln_1.weight": ("model/h{}/ln_1/g", torch.float32, False),
        "ln_1.bias": ("model/h{}/ln_1/b", torch.float32, False),
        "ln_2.weight": ("model/h{}/ln_2/g", torch.float32, False),
        "ln_2.bias": ("model/h{}/ln_2/b", torch.float32, False),
        "attn.c_attn.weight": ("model/h{}/attn/c_attn/w", torch.float16, True),
        "attn.c_attn.bias": ("model/h{}/attn/c_attn/b", torch.float32, False),
        "attn.c_proj.weight": ("model/h{}/attn/c_proj/w", torch.float16, True),
        "attn.c_proj.bias": ("model/h{}/attn/c_proj/b", torch.float32, False),
        "mlp.c_fc.weight": ("model/h{}/mlp/c_fc/w", torch.float16, True),
        "mlp.c_fc.bias": ("model/h{}/mlp/c_fc/b", torch.float32, False),
        "mlp.c_proj.weight": ("model/h{}/mlp/c_proj/w", torch.float16, True),
        "mlp.c_proj.bias": ("model/h{}/mlp/c_proj/b", torch.float32, False),
    }
    if m.group(2) not in layers: return None
    fmt, dtype, transpose = layers[m.group(2)]
    return fmt.format(int(m.group(1))), dtype, transpose
def main():
    ckpt_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    state = load_file(ckpt_dir / "t3_turbo_v1.safetensors")
    for name in sorted(state):
        print(f"{name}\t{tuple(state[name].shape)}", flush=True)
    skip = {"tfmr.wte.weight", "text_head.weight"}
    unknown = [name for name in state if name not in skip and map_name(name) is None]
    if unknown:
        print("STOP unknown keys:", file=sys.stderr)
        for name in unknown:
            print(f"  {name}\t{tuple(state[name].shape)}", file=sys.stderr)
        raise SystemExit("turbo is not a nano swap")
    conds = torch.load(ckpt_dir / "conds.pt", map_location="cpu", weights_only=True)
    n_embd = int(state["tfmr.ln_f.weight"].shape[0])
    n_ctx = int(state["tfmr.wpe.weight"].shape[0])
    n_layer = max(int(m.group(1)) for name in state if (m := LAYER_RE.match(name))) + 1
    n_head = n_embd // 64
    if n_embd % 64:
        raise SystemExit(f"n_head {n_embd}//64")
    writer = gguf.GGUFWriter(str(out), "chatterbox")
    writer.add_uint32("chatterbox.n_ctx", n_ctx)
    writer.add_uint32("chatterbox.n_embd", n_embd)
    writer.add_uint32("chatterbox.n_head", n_head)
    writer.add_uint32("chatterbox.n_layer", n_layer)
    writer.add_uint32("chatterbox.text_vocab_size", TEXT_VOCAB_SIZE)
    writer.add_uint32("chatterbox.speech_vocab_size", SPEECH_VOCAB_SIZE)
    writer.add_uint32("chatterbox.start_speech_token", START_SPEECH_TOKEN)
    writer.add_uint32("chatterbox.stop_speech_token", STOP_SPEECH_TOKEN)
    writer.add_uint32("chatterbox.speaker_embed_size", SPEAKER_EMBED_SIZE)
    writer.add_float32("chatterbox.layer_norm_eps", 1e-5)
    tokens, types, merges = tokenizer(ckpt_dir)
    if len(tokens) != TEXT_VOCAB_SIZE:
        raise SystemExit(f"tokenizer {len(tokens)} != {TEXT_VOCAB_SIZE}")
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    writer.add_token_merges(merges)
    for name, tensor in state.items():
        mapped = map_name(name)
        if mapped is None: continue
        gguf_name, dtype, transpose = mapped
        add(writer, gguf_name, as_numpy(tensor, dtype=dtype, transpose=transpose))
    builtin_tokens = conds["t3"]["cond_prompt_speech_tokens"].reshape(-1).to(torch.int32)
    writer.add_uint32("chatterbox.cond_prompt_max", int(builtin_tokens.numel()))
    writer.add_uint32("chatterbox.cond_prompt_length", int(builtin_tokens.numel()))
    writer.add_tensor("chatterbox/builtin/speaker_emb", as_numpy(conds["t3"]["speaker_emb"].reshape(1, SPEAKER_EMBED_SIZE), dtype=torch.float32))
    writer.add_tensor("chatterbox/builtin/cond_prompt_speech_tokens", as_numpy(builtin_tokens))
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
