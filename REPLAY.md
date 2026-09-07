# Replay capture (schema 2)

Use the Trident Nano capture command. It records the exact source, runtime/model
SHA-256 values, native revision, backend/device, settings and one binary capture
inside a lossless ZIP next to the WAV. Keep the referenced GGUF files in models:
weights are identified by hash, not copied into every recording.

Each capture record is little-endian uint32 JSON length, uint64 payload length,
UTF-8 JSON and raw payload. JSON contains schema, sequence, UTC nanoseconds,
scope, event; tensor records also contain name, dtype, shape and C layout.
Floating point payloads retain their bits (including filtered negative infinity).
All dimensions are NumPy order. Scope .r0_p0 is warmup; .r1_p0 is the first request.
Step scopes append .t3-s0, .t3-s1, etc. Repeated scope/name is distinguished by
record sequence. A recording is complete only when run.json says CAPTURED.

T3 speaker and conditioning tokens at root are the actual baked overrides.
Each evaluation records input speech token and absolute positions. Prompt
attention-mask is FP16, row=query and column=key; step attention is unmasked
over the active prefix. Each kv-k-in / kv-v-in contains the complete actual active incoming cache,
with shape [layer, head, position, head_dimension]. The prompt has zero active
positions. Every later step reads the active cache from the backend before
evaluation, including earlier rows; nothing is reconstructed or deduplicated.
Unused capacity and allocator memory are not replay inputs.
Sampling records include raw/filtered scores, probabilities, full mt19937 states,
selected token and the effective/published decision (including forced EOS).

S3 records the complete incoming mel/source/phase/pending state and token range,
voice conditioning, speech window, padded flow tokens, encoder positions and
outputs, CFM initial noise and each integration step, conditioned mel, F0,
excitation and waveform stages. source-noise is [harmonic,new_sample], starting
at history_samples. source-harmonics is [harmonic,all_samples]; its history
region is unused zeros, with source history copied from state-source-in.
source-parameters = [sine amplitude, noise std, voiced threshold, linear bias].
source-phase-offsets contains actual uniform draws; state-phase-in is double.
STFT/ISTFT kernels/windows are recorded as their actual generated inputs.
The pinned source defines fixed layer operations and derived masks/constants.
Graph/allocator caches contain no additional recurrent semantic state.

Native wire PCM16 is recorded before sending. run.json emission entries map
received frames to WAV sample ranges and record the existing leading-zero trim.
The final WAV SHA-256 ties those ranges to the audio. ZIP manifest hashes cover
the evidence stream; read binary tensors using NumPy frombuffer with the recorded
dtype and shape. Capture performs no ASR, extra tokenization, reference inference,
post-hoc deduplication, retries or sampling modifications.

Capture supplies replay inputs and outputs, not a claim of speech correctness
or a replay/reference implementation.
