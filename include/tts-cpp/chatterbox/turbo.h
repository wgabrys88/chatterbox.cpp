#pragma once
namespace tts_cpp::chatterbox {
inline constexpr int SEED = 42;
inline constexpr int N_PREDICT = 1000;
inline constexpr int TOP_K = 1000;
inline constexpr float TOP_P = 0.95f;
inline constexpr float TEMPERATURE = 0.8f;
inline constexpr float REPEAT_PENALTY = 1.2f;
inline constexpr int REPEAT_LAST_N = 1000;
inline constexpr int CFM_STEPS = 2;
inline constexpr int SILENCE_TOKEN = 4299;
inline constexpr int SILENCE_COUNT = 3;
}