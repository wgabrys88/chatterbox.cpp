#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
import gguf, numpy as np, torch
from quant_policy import WEIGHT_TYPES, add_weight
from safetensors.torch import load_file
TEXT_VOCAB_SIZE, SPEECH_VOCAB_SIZE = 50276, 6563
START_SPEECH_TOKEN, STOP_SPEECH_TOKEN, SPEAKER_EMBED_SIZE = 6561, 6562, 256
LAYER_RE = re.compile(r"^tfmr\.h\.(\d+)\.(.+)$")
SKIP = {"tfmr.wte.weight", "text_head.weight"}
from convert_common import as_numpy, finish_t3

MATRIX_SUFFIXES = ("/attn/c_attn/w", "/attn/c_proj/w", "/mlp/c_fc/w", "/mlp/c_proj/w")
def add(writer, name, array, matrix_type):
    is_matrix = name.startswith("model/h") and name.endswith(MATRIX_SUFFIXES)
    add_weight(writer, name, array, matrix_type, force_f32=not is_matrix)

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
        "text_emb.weight": ("chatterbox/text_emb", torch.float32, False),
        "speech_emb.weight": ("chatterbox/speech_emb", torch.float32, False),
        "speech_head.weight": ("chatterbox/speech_head", torch.float32, False),
        "speech_head.bias": ("chatterbox/speech_head_bias", torch.float32, False),
        "cond_enc.spkr_enc.weight": ("chatterbox/cond_spkr/w", torch.float32, False),
        "cond_enc.spkr_enc.bias": ("chatterbox/cond_spkr/b", torch.float32, False),
    }
    if name in table: return table[name]
    if name in SKIP: return None
    m = LAYER_RE.match(name)
    if not m: return None
    layers = {
        "ln_1.weight": ("model/h{}/ln_1/g", torch.float32, False),
        "ln_1.bias": ("model/h{}/ln_1/b", torch.float32, False),
        "ln_2.weight": ("model/h{}/ln_2/g", torch.float32, False),
        "ln_2.bias": ("model/h{}/ln_2/b", torch.float32, False),
        "attn.c_attn.weight": ("model/h{}/attn/c_attn/w", torch.float32, True),
        "attn.c_attn.bias": ("model/h{}/attn/c_attn/b", torch.float32, False),
        "attn.c_proj.weight": ("model/h{}/attn/c_proj/w", torch.float32, True),
        "attn.c_proj.bias": ("model/h{}/attn/c_proj/b", torch.float32, False),
        "mlp.c_fc.weight": ("model/h{}/mlp/c_fc/w", torch.float32, True),
        "mlp.c_fc.bias": ("model/h{}/mlp/c_fc/b", torch.float32, False),
        "mlp.c_proj.weight": ("model/h{}/mlp/c_proj/w", torch.float32, True),
        "mlp.c_proj.bias": ("model/h{}/mlp/c_proj/b", torch.float32, False),
    }
    if m.group(2) not in layers: return None
    fmt, dtype, transpose = layers[m.group(2)]
    return fmt.format(int(m.group(1))), dtype, transpose
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
    n_embd = int(state["tfmr.ln_f.weight"].shape[0])
    n_ctx = int(state["tfmr.wpe.weight"].shape[0])
    n_layer = max(int(m.group(1)) for name in state if (m := LAYER_RE.match(name))) + 1
    n_head = n_embd // 64
    writer = gguf.GGUFWriter(str(out), "chatterbox")
    writer.add_string("chatterbox.conversion.matrix_type", a.matrix_type)
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
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    writer.add_token_merges(merges)
    for name, tensor in state.items():
        mapped = map_name(name)
        if mapped is None: continue
        gguf_name, dtype, transpose = mapped
        add(writer, gguf_name, as_numpy(tensor, dtype=dtype, transpose=transpose), a.matrix_type)
    finish_t3(writer, ckpt_dir, conds)
if __name__ == "__main__":
    main()
