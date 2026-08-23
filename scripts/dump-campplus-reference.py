#!/usr/bin/env python3


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", type=Path, help="Reference wav (any SR; resampled to 16 kHz)")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output dir; creates fbank.npy + embedding.npy inside it")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from chatterbox.tts_turbo import ChatterboxTurboTTS

    tts = ChatterboxTurboTTS.from_pretrained("cpu")
    speaker_encoder = tts.s3gen.speaker_encoder
    speaker_encoder.eval()

    wav, sr = torchaudio.load(str(args.wav))
    wav = wav.mean(dim=0) if wav.ndim == 2 and wav.shape[0] > 1 else wav.squeeze(0)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)




    import torchaudio.compliance.kaldi as Kaldi
    fbank_raw = Kaldi.fbank(wav.unsqueeze(0), num_mel_bins=80, dither=0.0)


    fbank_centered = fbank_raw - fbank_raw.mean(dim=0, keepdim=True)

    np.save(args.out / "fbank_raw.npy",      np.ascontiguousarray(fbank_raw.numpy().astype(np.float32)))
    np.save(args.out / "fbank.npy",          np.ascontiguousarray(fbank_centered.numpy().astype(np.float32)))

    with torch.no_grad():
        emb = speaker_encoder.forward(fbank_centered.unsqueeze(0).to(torch.float32))
    emb = emb[0].cpu().numpy().astype(np.float32)
    np.save(args.out / "embedding.npy", np.ascontiguousarray(emb))


    np.save(args.out / "wav_16k.npy", np.ascontiguousarray(wav.numpy().astype(np.float32)))

    print(f"fbank_raw.npy  shape={fbank_raw.shape}")
    print(f"fbank.npy      shape={fbank_centered.shape}")
    print(f"embedding.npy  shape={emb.shape}  norm={np.linalg.norm(emb):.4f}")
    print(f"wav_16k.npy    shape={wav.shape}")


if __name__ == "__main__":
    main()
