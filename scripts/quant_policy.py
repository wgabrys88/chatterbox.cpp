import gguf
DENY = (
    "flow/input_embedding", "flow/spk_embed_affine/w", "/builtin/",
    "text_emb", "speech_emb", "wte", "wpe", "stft_basis", "mel_filterbank",
    "mel_fb", "pos_emb", "pe/pe", "pre_attention_query", "/b", "/bias",
    "/bn/", "/norm/", "/ln_", "/scale", "alpha", "beta", "gamma",
    "voice_encoder/", "campplus/", "s3tokv2/",
)
QUANT_TYPE = {
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
    "q5_0": gguf.GGMLQuantizationType.Q5_0,
    "q4_0": gguf.GGMLQuantizationType.Q4_0,
    "f16": gguf.GGMLQuantizationType.F16,
}
def should_quantize(name, shape, qtype):
    if __import__("math").prod(shape) < 1024 or any(s in name for s in DENY) or name.endswith("/g"):
        return False
    block = gguf.GGML_QUANT_SIZES[qtype][0]
    return len(shape) in (2, 3) and shape[-1] % block == 0
