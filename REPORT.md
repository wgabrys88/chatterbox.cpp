# Nano pin and the last speech run

Trident does not follow this branch tip. `main.py` has `CHATTERBOX_REV`. On install it downloads that commit and builds `chatterbox-server.exe`. That is luck in the good sense: while this branch went six commits forward into experiments, Trident kept building the older good snapshot. Those six were then undone (`back to m36`), so the engine files here are that snapshot again. This report is the extra file so this can be a normal commit on the same code. After push, Trident’s pin should be this commit. Do not reset those six away. Leave the GGML pin alone.

The last investigated run used the pin that was in `main.py` that day, Vulkan, MTL off. `--s3-reset` is not in that binary.

## What spoke

`python tts_nano.py --text ...` — no Gemma. Fresh SaT process (`chunk.py`, `sat-12l-sm`, CPU, threshold 0.25, newlines stay breaks). Do not add a splitter by character, by dot, or by max length. TCP v4 to the Nano server. T3 (GPT-2, seed 42 every piece) then S3Gen meanflow, two CFM steps, last 25 speech tokens as history. Python concatenates PCM. Piece 0: server holds 20 ms for the next piece, zeros/fades the start, Python drops 20 ms of leading zeros. After that trim the wave matches the native piece ledger exactly.

SaT is on this path. Turbo, V3, Gemma, Parakeet, and `validate.py` are not. `main.py` is on the path because `tts_nano.py` imports it. Python is the installer, pin, SaT, TCP client, and wav concat. A C++-only Trident is a new project, not a cleanup. SaT has no C++ port here.

## The wave

`Trident/out_09-09-26-09-47-40_tts.wav` — 173.56 s, 24 kHz, mono, 16-bit, incomplete. Started 09:47:06, ready 09:47:16, died 09:51:05 on piece 87: `T3 stopped without EOS`. Pieces 0–86 spoke. Pieces 87–121 did not. No `tts_out.wav` (that copy happens only after a full success). `main.log` is a later unload, ignore it for this synth. Python `synth.*` went to stdout and was not saved. Read `chunk.log` and `tts.log`.

SaT: 5013 chars, 1 line, 122 pieces. The spoken text was the agent prompt plus count 1–30 twice (`And now, we will start over again:`). Same number spoken twice with seed 42 makes the same speech. Expected, not a bug.

Piece 87 was a SHA/path sentence. T3 hit max-tokens 1000 without EOS. No JSON ledger for the failed piece. EOS-hold was tried later and undone with the rest. Do not put it back without a new user run.

## 00:33 and 00:39

First 30 s sounded good. Both faults sit inside piece 5 only (25.36 s–55.72 s, 30.36 s, 251 chars, 759 speech tokens, stop=eos):

`Read every git tracked file in full in Trident repository, then must-read later in full end to end: in Trident, chunk.py, main.py, tts_nano.py; in chatterbox, chatterbox_engine.cpp, t3_turbo.cpp, chatterbox_t3_internal.h, server.cpp, s3gen_pipeline.h.`

~00:33 is after the colon, around `chunk.py`. Source has one `chunk`. Ear heard two. Not copied PCM (no strong 180 ms self-match). Repeat penalty only looks at the last four speech tokens. A word is 10–25. The sixteen-identical abort did not fire (`stop=eos`). This is T3 over-speaking a filename, or reading `chunk.py` as two chunk-like syllables.

~00:39: source has `chatterbox` three times in a row. Hearing it three times can be a correct reading. Treat 00:39 as mostly source, 00:33 as the real over-speak.

Not SaT duplicating words. Not Python concat (no piece boundary there). Not S3 history (that only colours the first second of the piece). Not a buffer overflow. SaT packed identifiers into one long piece. T3 then spent 30 s on it. You forbade extra splitters, so do not “fix” it by chopping filenames in Python.

Next proof: you run one `--audit` Nano pass of **only** that piece-5 sentence. Then read `04-sample.jsonl` and `03-speech-tokens.bin`. Sampling knobs stay. Audit is slow; do not use it for RTF.

## Knobs (leave them)

Nano: min-p 0.05, temperature 0.8, top-k 1000, top-p 0.95, repeat 1.2 on the last four tokens, stop after sixteen identical, context 2048, threads 4, seed 42, max-tokens 1000, GPU layers 99, fastconv on, cfg-weight 0, exaggeration 0, CFM steps 2. History 25 is compile-time.

This server applies min-p. Official Python Nano ignores min-p > 0. That fork is intentional. Not proven as 00:33. Do not edit min-p, repeat, or context on this evidence.

Two CFM steps is correct. GGUF says meanflow and 2. Official Python uses 2. Marketing “one step” is the lie. Dropping to 1 would change speech. cfg-weight and exaggeration do nothing on Nano (no RTF save). No unused flag that only saves RTF. `--s3-reset` is not in this code. `--audit` slows things.

## Logs

Piece JSON that is real: text, counts, hashes, stop, T3/S3 times, S3 history/new, emitted samples. Missing: speech token ids, failed-piece JSON, Python `synth.*` on disk, CFM steps in the ledger. Do not rewrite logging from zero. Add ids (or a short repeat summary) to the existing JSONL, and persist Python `synth.*` next to the native log. JSON lines that exist match the wav.

KV leak did not cause 00:33 (fault is mid-piece, 8 s after the boundary). S3 history did not (1 s at the start of piece 5). Cause is T3 inside piece 5.

## What not to do

Do not rebuild unless chatterbox itself must change. Changing `CHATTERBOX_REV` will make the next Trident run rebuild; that is expected after a pin bump, same engine files. Do not convert GGUF, start servers, or run probes unless asked. Do not commit secrets, wav, models, exe, `nano-bare`, or the zip. One meaningful change, then a user run, then read the new logs.

Fat you can delete later without touching Nano speech: Turbo/V3 scripts, Brain, Parakeet, Validate, MTL (already compiled out). Do not “slim” the S3 file as a free win.

## nano-bare

Untracked under Trident: `nano-bare/` (flat, 31 files) and `nano-bare.zip`. Sources only, no blobs. `main.py` in that snapshot is the full Trident file that ran, not a slim TTS-only copy. The zip is for reading. CMake still wants the GGML checkout.

## Files from that run

Wave: `Trident/out_09-09-26-09-47-40_tts.wav`  
Logs: `Trident/.runtime-logs/chunk.log`, `tts.log`  
Pin in Trident `main.py` after this pairing: this commit, plus GGML `58c3805840b516b2a88ff867ccf7bb41dba79951`  
The run itself built pin `519861682a187a2696fb98eb881052ad91a70470` with that same GGML.
