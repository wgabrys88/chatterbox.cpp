#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import soundfile as sf


def _patch_dtype_bugs():
    from chatterbox.models.s3tokenizer.s3tokenizer import S3Tokenizer
    _orig_mel = S3Tokenizer.log_mel_spectrogram
    def _p_mel(self, audio, padding=0):
        if not torch.is_tensor(audio): audio = torch.from_numpy(audio)
        audio = audio.float()
        self._mel_filters.data = self._mel_filters.data.float()
        self.window.data = self.window.data.float()
        return _orig_mel(self, audio, padding)
    S3Tokenizer.log_mel_spectrogram = _p_mel

    from chatterbox.models.voice_encoder.voice_encoder import VoiceEncoder
    _orig_fwd = VoiceEncoder.forward
    VoiceEncoder.forward = lambda self, mels: _orig_fwd(self, mels.float())


def _patch_flow_finalize_bug():
    from chatterbox.models.s3gen.flow import _repeat_batch_dim
    from chatterbox.models.s3gen.utils.mask import make_pad_mask
    import torch.nn.functional as F
    from chatterbox.models.s3gen.flow import CausalMaskedDiffWithXvec

    def inference_patched(self, token, token_len, prompt_token, prompt_token_len,
                          prompt_feat, prompt_feat_len, embedding,
                          finalize, n_timesteps=10, noised_mels=None, meanflow=False):
        B = token.size(0)
        embedding = torch.atleast_2d(embedding)
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        prompt_token = _repeat_batch_dim(prompt_token, B, ndim=2)
        prompt_token_len = _repeat_batch_dim(prompt_token_len, B, ndim=1)
        prompt_feat = _repeat_batch_dim(prompt_feat, B, ndim=3)
        prompt_feat_len = _repeat_batch_dim(prompt_feat_len, B, ndim=1)
        embedding = _repeat_batch_dim(embedding, B, ndim=2)
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(token.long()) * mask
        h, h_masks = self.encoder(token, token_len)






        if finalize is False:
            trim = self.pre_lookahead_len * self.token_mel_ratio
            h = h[:, :-trim]
            h_masks = h_masks[..., :-trim]

        h_lengths = h_masks.sum(dim=-1).squeeze(dim=-1)
        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]
        h = self.encoder_proj(h)
        conds = torch.zeros([B, mel_len1 + mel_len2, self.output_size],
                            device=token.device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)
        mask = (~make_pad_mask(h_lengths)).unsqueeze(1).to(h)
        if mask.shape[0] != B:
            mask = mask.repeat(B, 1, 1)

        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask, spks=embedding, cond=conds,
            n_timesteps=n_timesteps, noised_mels=noised_mels, meanflow=meanflow,
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2, f"feat {feat.shape} vs mel_len2 {mel_len2}"
        return feat, None

    CausalMaskedDiffWithXvec.inference = inference_patched


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_wav",   type=Path, help="Reference wav for voice cloning")
    ap.add_argument("--text",    type=str, default="Hello in my voice.")
    ap.add_argument("--chunk-tokens", type=int, default=25,
                    help="Number of new speech tokens per streaming chunk (default: 25 ≈ 1 s of audio)")
    ap.add_argument("--out",     type=Path, required=True)
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    _patch_dtype_bugs()
    _patch_flow_finalize_bug()

    from chatterbox.tts_turbo import ChatterboxTurboTTS

    torch.manual_seed(args.seed)
    tts = ChatterboxTurboTTS.from_pretrained("cpu")
    tts.prepare_conditionals(str(args.ref_wav), exaggeration=0.5)
    cond_t3  = tts.conds.t3
    cond_gen = tts.conds.gen


    print(f"[1/3] T3 generating speech tokens for text: {args.text!r}")
    from chatterbox.tts_turbo import punc_norm
    text = punc_norm(args.text)
    text_tokens = tts.tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids
    speech_tokens = tts.t3.inference_turbo(
        t3_cond=cond_t3,
        text_tokens=text_tokens,
        temperature=0.8, top_k=1000, top_p=0.95, repetition_penalty=1.2,
    )
    speech_tokens = speech_tokens[speech_tokens < 6561]
    S3GEN_SIL = 4299
    silence = torch.tensor([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL]).long()
    speech_tokens = torch.cat([speech_tokens, silence])
    print(f"      {len(speech_tokens)} speech tokens")
    np.save(args.out / "speech_tokens.npy",
            np.ascontiguousarray(speech_tokens.cpu().numpy().astype(np.int32)))




    np.save(args.out / "speaker_emb.npy",
            np.ascontiguousarray(cond_t3.speaker_emb.detach().squeeze().cpu().numpy().astype(np.float32)))
    np.save(args.out / "cond_prompt_speech_tokens.npy",
            np.ascontiguousarray(cond_t3.cond_prompt_speech_tokens.detach().squeeze().cpu().numpy().astype(np.int32)))
    np.save(args.out / "embedding.npy",
            np.ascontiguousarray(cond_gen["embedding"].detach().squeeze().cpu().numpy().astype(np.float32)))
    np.save(args.out / "prompt_token.npy",
            np.ascontiguousarray(cond_gen["prompt_token"].detach().squeeze().cpu().numpy().astype(np.int32)))
    np.save(args.out / "prompt_feat.npy",
            np.ascontiguousarray(cond_gen["prompt_feat"].detach().squeeze().cpu().numpy().astype(np.float32)))


    print("[2/3] batch (non-streamed) reference")
    with torch.no_grad():
        batch_mels = tts.s3gen.flow_inference(
            speech_tokens, speech_token_lens=None,
            ref_dict=cond_gen, n_cfm_timesteps=2, finalize=True,
        ).to(dtype=tts.s3gen.dtype)
        batch_wavs, batch_source = tts.s3gen.hift_inference(batch_mels, None)
    batch_wavs = batch_wavs.clone()
    batch_wavs[:, :len(tts.s3gen.trim_fade)] *= tts.s3gen.trim_fade
    batch_wav = batch_wavs.squeeze().cpu().numpy().astype(np.float32)
    np.save(args.out / "batch_mels.npy",
            np.ascontiguousarray(batch_mels.squeeze(0).transpose(0, 1).cpu().numpy().astype(np.float32)))
    np.save(args.out / "batch_wav.npy", np.ascontiguousarray(batch_wav))
    print(f"      batch: {batch_mels.shape[-1]} mel frames → {len(batch_wav)} wav samples")






















    TOKEN_MEL_RATIO  = tts.s3gen.flow.token_mel_ratio
    PRE_LOOKAHEAD    = tts.s3gen.flow.pre_lookahead_len
    LOOKAHEAD_MELS   = PRE_LOOKAHEAD * TOKEN_MEL_RATIO

    MEL_TO_WAV       = 480



    SOURCE_OVERLAP   = MEL_TO_WAV

    chunk_tokens = args.chunk_tokens
    n_speech = len(speech_tokens)

    boundaries = list(range(0, n_speech, chunk_tokens)) + [n_speech]

    streamed_wav = np.zeros(0, dtype=np.float32)
    prev_source_tail = torch.zeros(1, 1, 0)
    prev_mels_emitted = 0
    per_chunk_stats = []

    print(f"[3/3] streaming loop: {len(boundaries)-1} chunks × {chunk_tokens} tokens")







    captured_mu = {"mu": None}
    def _enc_proj_hook(mod, inp, out):
        captured_mu["mu"] = out.detach().clone().cpu()
    _mu_hook_handle = tts.s3gen.flow.encoder_proj.register_forward_hook(_enc_proj_hook)









    captured_decoder = {"mu": None, "mask": None, "spks": None, "cond": None}
    captured_step0 = {"x_in": None, "mask": None, "mu": None, "t": None,
                      "r": None, "spks": None, "cond": None, "dxdt": None}

    _orig_basic_euler = tts.s3gen.flow.decoder.basic_euler
    def _basic_euler_capture(x, t_span, mu, mask, spks, cond):

        captured_decoder["mu"]   = mu.detach().clone().cpu()
        captured_decoder["mask"] = mask.detach().clone().cpu()
        captured_decoder["spks"] = None if spks is None else spks.detach().clone().cpu()
        captured_decoder["cond"] = None if cond is None else cond.detach().clone().cpu()

        _orig_fwd = tts.s3gen.flow.decoder.estimator.forward
        def _est(x, mask, mu, t, spks=None, cond=None, r=None):
            out = _orig_fwd(x, mask, mu, t, spks=spks, cond=cond, r=r)
            if captured_step0["dxdt"] is None:
                captured_step0["x_in"] = x.detach().clone().cpu()
                captured_step0["mask"] = mask.detach().clone().cpu()
                captured_step0["mu"]   = mu.detach().clone().cpu()
                captured_step0["t"]    = t.detach().clone().cpu()
                captured_step0["r"]    = None if r is None else r.detach().clone().cpu()
                captured_step0["spks"] = None if spks is None else spks.detach().clone().cpu()
                captured_step0["cond"] = None if cond is None else cond.detach().clone().cpu()
                captured_step0["dxdt"] = out.detach().clone().cpu()
            return out
        tts.s3gen.flow.decoder.estimator.forward = _est
        try:
            return _orig_basic_euler(x, t_span, mu, mask, spks, cond)
        finally:
            tts.s3gen.flow.decoder.estimator.forward = _orig_fwd

    tts.s3gen.flow.decoder.basic_euler = _basic_euler_capture



    cfm_stage_captures = {}
    cfm_stage_hooks = []
    def _stage_hook(key):
        def _h(mod, inp, out):
            if key not in cfm_stage_captures:
                if isinstance(out, torch.Tensor):
                    cfm_stage_captures[key] = out.detach().clone().cpu()
        return _h

    est = tts.s3gen.flow.decoder.estimator
    cfm_stage_hooks.append(est.down_blocks[0][0].register_forward_hook(_stage_hook("cfm_d0_rn")))
    cfm_stage_hooks.append(est.down_blocks[0][1][0].register_forward_hook(_stage_hook("cfm_d0_tfm0")))
    cfm_stage_hooks.append(est.down_blocks[0][2].register_forward_hook(_stage_hook("cfm_d0_downsample")))

    for k, end in enumerate(boundaries[1:], start=1):
        is_last = (end == n_speech)
        tokens_so_far = speech_tokens[:end]

        np.save(args.out / f"chunk_{k:02d}_tokens.npy",
                np.ascontiguousarray(tokens_so_far.cpu().numpy().astype(np.int32)))

        captured_mu["mu"] = None
        captured_step0["dxdt"] = None
        with torch.no_grad():
            mels = tts.s3gen.flow_inference(
                tokens_so_far, speech_token_lens=None,
                ref_dict=cond_gen, n_cfm_timesteps=2, finalize=is_last,
            ).to(dtype=tts.s3gen.dtype)

        if captured_mu["mu"] is not None:
            mu_np = captured_mu["mu"].squeeze(0).numpy().astype(np.float32)
            np.save(args.out / f"chunk_{k:02d}_mu.npy", np.ascontiguousarray(mu_np))



        for name in ("mu", "mask", "spks", "cond"):
            t = captured_decoder[name]
            if t is None: continue
            arr = t.squeeze(0).numpy().astype(np.float32)
            np.save(args.out / f"chunk_{k:02d}_dec_{name}.npy", np.ascontiguousarray(arr))



        for name in ("x_in", "mask", "mu", "t", "r", "spks", "cond", "dxdt"):
            t = captured_step0[name]
            if t is None: continue
            arr = t.numpy().astype(np.float32)
            if arr.ndim >= 3:
                arr = arr.squeeze(0)
            np.save(args.out / f"chunk_{k:02d}_step0_{name}.npy", np.ascontiguousarray(arr))
            captured_step0[name] = None

        for stage_key, tval in list(cfm_stage_captures.items()):
            if tval is None: continue
            arr = tval.squeeze(0).numpy().astype(np.float32)
            np.save(args.out / f"chunk_{k:02d}_{stage_key}.npy", np.ascontiguousarray(arr))
        cfm_stage_captures.clear()



        mels_new_count = mels.shape[-1] - prev_mels_emitted
        mels_new = mels[..., prev_mels_emitted:]
        if mels_new.shape[-1] <= 0:


            continue





        with torch.no_grad():
            wav_k, src_k = tts.s3gen.hift_inference(mels_new, prev_source_tail)
        wav_k_np = wav_k.squeeze().cpu().numpy().astype(np.float32)


        if k == 1:
            tf = tts.s3gen.trim_fade.cpu().numpy()
            wav_k_np[:len(tf)] *= tf



        streamed_wav = np.concatenate([streamed_wav, wav_k_np])

        np.save(args.out / f"chunk_{k:02d}_mels_new.npy",
                np.ascontiguousarray(mels_new.squeeze(0).transpose(0, 1).cpu().numpy().astype(np.float32)))
        np.save(args.out / f"chunk_{k:02d}_wav.npy",
                np.ascontiguousarray(wav_k_np))
        np.save(args.out / f"chunk_{k:02d}_source.npy",
                np.ascontiguousarray(src_k.squeeze(0).cpu().numpy().astype(np.float32)))


        prev_source_tail = src_k[..., -SOURCE_OVERLAP:].clone()
        np.save(args.out / f"chunk_{k:02d}_source_tail.npy",
                np.ascontiguousarray(prev_source_tail.squeeze(0).cpu().numpy().astype(np.float32)))

        prev_mels_emitted = mels.shape[-1]
        per_chunk_stats.append({
            "chunk": k,
            "is_last": is_last,
            "tokens_consumed_total": end,
            "mels_new": mels_new_count,
            "wav_samples": len(wav_k_np),
        })
        print(f"    chunk {k:02d}: finalize={is_last}  tokens_total={end}  "
              f"mels_new={mels_new_count}  wav={len(wav_k_np)} samples")

    _mu_hook_handle.remove()
    tts.s3gen.flow.decoder.basic_euler = _orig_basic_euler
    for h in cfm_stage_hooks: h.remove()
    np.save(args.out / "streamed_wav.npy", np.ascontiguousarray(streamed_wav))




    n = min(len(streamed_wav), len(batch_wav))
    diff = streamed_wav[:n] - batch_wav[:n]
    stats = {
        "chunk_tokens": chunk_tokens,
        "n_speech_tokens": int(n_speech),
        "n_chunks": len(per_chunk_stats),
        "batch_wav_samples": int(len(batch_wav)),
        "streamed_wav_samples": int(len(streamed_wav)),
        "rms_err_vs_batch": float(np.sqrt(np.mean(diff ** 2))),
        "max_abs_err_vs_batch": float(np.max(np.abs(diff))),
        "batch_rms": float(np.sqrt(np.mean(batch_wav ** 2))),
        "streamed_rms": float(np.sqrt(np.mean(streamed_wav ** 2))),
        "per_chunk": per_chunk_stats,
    }
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2))


    sf.write(args.out / "batch.wav",    batch_wav,    24000, subtype="PCM_16")
    sf.write(args.out / "streamed.wav", streamed_wav, 24000, subtype="PCM_16")

    print()
    print(f"rms(streamed - batch) = {stats['rms_err_vs_batch']:.5f}  "
          f"(batch_rms={stats['batch_rms']:.4f})")
    print(f"→ ratio = {stats['rms_err_vs_batch'] / max(stats['batch_rms'], 1e-9):.3%}")
    print(f"dumped to {args.out}")


if __name__ == "__main__":
    main()
