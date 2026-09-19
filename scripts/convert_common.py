import numpy as np
import torch
from safetensors.torch import load_file

def as_numpy(tensor, *, dtype=None, transpose=False):
    if dtype is not None: tensor = tensor.to(dtype)
    array = tensor.detach().cpu().numpy()
    if transpose: array = array.T
    return np.ascontiguousarray(array)

def finish_t3(writer, ckpt_dir, conds):
    builtin_tokens = conds["t3"]["cond_prompt_speech_tokens"].reshape(-1).to(torch.int32)
    writer.add_uint32("chatterbox.cond_prompt_max", int(builtin_tokens.numel()))
    writer.add_uint32("chatterbox.cond_prompt_length", int(builtin_tokens.numel()))
    writer.add_tensor("chatterbox/builtin/speaker_emb", as_numpy(conds["t3"]["speaker_emb"].reshape(1, 256), dtype=torch.float32))
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
