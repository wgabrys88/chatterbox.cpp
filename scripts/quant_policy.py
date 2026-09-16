import gguf
# Substring denylist. Do not put "/b" here: it is a prefix of "block" and
# would skip every conformer block weight. Biases use DENY_SUFFIX instead.
DENY = (
    "flow/input_embedding", "flow/spk_embed_affine/w", "/builtin/",
    "text_emb", "speech_emb", "wte", "wpe", "stft_basis", "mel_filterbank",
    "mel_fb", "pos_emb", "pe/pe", "pre_attention_query", "/bias",
    "/bn/", "/norm/", "/ln_", "/scale", "alpha", "beta", "gamma",
    "voice_encoder/", "campplus/", "s3tokv2/",
    # Gauge is the vocoder's own fuel plan. Encoder mu and F0 stay F32.
    "flow/encoder/", "flow/encoder_proj", "hift/f0_predictor",
)
DENY_SUFFIX = ("/b", "/bias", "/g")
QUANT_TYPE = {
    "q4_0": gguf.GGMLQuantizationType.Q4_0,
}
def should_quantize(name, shape, qtype):
    if __import__("math").prod(shape) < 1024:
        return False
    if name.endswith(DENY_SUFFIX):
        return False
    if any(s in name for s in DENY):
        return False
    block = gguf.GGML_QUANT_SIZES[qtype][0]
    return len(shape) in (2, 3) and shape[-1] % block == 0
