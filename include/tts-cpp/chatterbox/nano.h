#pragma once
namespace tts_cpp::chatterbox {
inline constexpr int SEED = 42;
inline constexpr int N_PREDICT = 8196;
inline constexpr int TOP_K = 1000;
inline constexpr float TOP_P = 0.95f;
inline constexpr float TEMPERATURE = 0.8f;
inline constexpr float REPEAT_PENALTY = 1.2f;
inline constexpr int CFM_STEPS = 2;
inline constexpr int SILENCE_TOKEN = 4299;
inline constexpr int SILENCE_COUNT = 3;
inline constexpr int STREAM_TOKENS = 24;
inline constexpr int STREAM_CROSSFADE_SAMPLES = 288;
// Splitter budget (text BPE tokens per utterance). 0 = unsplit. Set from
// listen.py measurement, never from a derived speech-token number.
inline constexpr int SPLIT_TOKENS = 0;
// KV cap. 0 = wpe rows (8196).
inline constexpr int N_CTX = 0;
inline constexpr bool MODE_STREAMING = true;
}
