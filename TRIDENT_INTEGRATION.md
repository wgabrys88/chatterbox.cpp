# chatterbox.cpp ↔ Trident integration

This document is the **API contract** that the Trident runtime depends on. It is
maintained in chatterbox.cpp so that any change to the wire protocol, command-line
knobs, or T3 / S3Gen pipeline is reflected here. Trident pins a specific commit of
this repo in `config.py:CHATTERBOX_REV` and rebuilds the native server when the pin
moves.

## What Trident builds and runs

Trident's `python main.py install` clones this repo at the pinned SHA, runs
`cmake -S . -B tools/tts-build -A x64 -DGGML_VULKAN=ON ...`, and links
`chatterbox-server.exe` from `src/server.cpp`. The server speaks the binary
protocol below on TCP port `17933` (loopback only).

## Wire protocol v2 (TCP, little-endian)

Defined in `src/server.cpp` and parsed on the Python side by `runtime.WireProtocol`.

```
Request header (28 bytes):
  uint32 magic   // 0x32525454 ("TTR2")
  uint32 version // 2
  uint32 kind    // request_kind: 1=synthesize, 2=advance_epoch, 3=close
  uint32 epoch
  uint32 response
  uint32 piece
  uint32 text_len

Request body:
  uint8 text[text_len]   // UTF-8, no NUL terminator

Response header (32 bytes):
  uint32 magic, version, kind, epoch, response, piece, chunk_id, length

Response body:
  uint8 payload[length]   // PCM int16 little-endian for RESP_PCM
```

Response kinds: `1=RESP_PCM`, `2=RESP_DONE`, `3=RESP_CANCELLED`, `4=RESP_ERROR`,
`5=RESP_CLOSED`.

## Command-line knobs (consumed by `EngineOptions` in `include/tts-cpp/chatterbox/engine.h`)

Trident always passes these (see `runtime.TTSProfile`):

| Flag                 | TTSProfile field | Default |
|----------------------|------------------|---------|
| `--n-gpu-layers`     | gpu_layers       | 99      |
| `--context`          | context          | 2048    |
| `--threads`          | threads          | 4       |
| `--seed`             | seed             | 42      |
| `--max-tokens`       | max_tokens       | 1000    |
| `--top-k`            | top_k            | 1000    |
| `--top-p`            | top_p            | 0.95    |
| `--min-p`            | min_p            | 0.05    |
| `--temperature`      | temperature      | 0.8     |
| `--repeat-penalty`   | repeat_penalty   | 1.2     |
| `--cfg-weight`       | cfg_weight       | 0.5     |
| `--exaggeration`     | exaggeration     | 0.5     |
| `--cfm-steps`        | cfm_steps        | 2       |
| `--fastconv`         | fastconv         | 1       |

Other required flags: `--run-id`, `--family nano`, `--model <t3.gguf>`,
`--s3gen-gguf <s3gen.gguf>`, `--reference <voice.wav>`, `--language en`,
`--port 17933`.

## Per-piece S3Gen pipeline (the cost model)

`Engine::piece_streaming` in `src/chatterbox_engine.cpp:135` runs:

1. **T3 forward pass** (GPU, Vulkan): `eval_prompt` (full prompt) + N×`eval_step`.
   Cost scales with text-token count.
2. **S3Gen encoder** (GPU, Vulkan): 6 conformer blocks + lookahead + 4 up-blocks.
   Graph is rebuilt when sequence length T or embedding dim D changes.
3. **CFM estimator** (GPU, Vulkan): 4-block UNet with attention. Meanflow mode
   uses `cfm_steps=2` (down from 10 in classic S3Gen).
4. **HiFT decode** (GPU, Vulkan): mel → F0 → STFT → iSTFT → 24 kHz int16 PCM.

**Per-piece fixed cost is ~500 ms on Intel Iris Xe (Xe1, SIMD8).** Trident
chunks text into pieces of ~75 chars so that 30 short sentences become 4 chunks
(see `tts._text_chunks`); this amortizes the per-piece overhead.

## What Trident relies on that is NOT in the public API

- The first 20 ms of `RESP_PCM` for `piece_id == 0` is all zeros (S3Gen warm-up
  artifact). Trident strips this in `_pcm_for_wav`.
- `begin_synthesis()` only resets the `cancelled` flag; the T3 KV buffer is
  cleared by `warm_up()` and persists across pieces within a session.
- `model.buffer_kv` is large (hundreds of MB on Iris Xe); reallocating it kills
  performance. Do not free it between `begin_synthesis()` calls.
- The server's synthesis thread **cannot safely batch more than ~1 piece** on
  Iris Xe — the Vulkan shader pool hits a `Missing multi_add` error when the
  compute graph grows beyond a few pieces. Trident's Python client sends one
  request per piece and lets the server process them in a tight loop.

## Tested on

- **GPU**: Intel Iris Xe (Xe1, `minSubgroupSize == 8`, integer dot product).
  Architecture detection lives in `ggml-vulkan.cpp:INTEL_XE1`. Xe2 (SIMD16) is
  expected to be ~2× faster per cycle.
- **Backend**: Vulkan only (`TTS_BACKEND = "vulkan"`). CUDA is not built.
- **OS**: Windows 10/11, Python 3.11+, MSVC.
