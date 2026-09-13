#pragma once
namespace tts_cpp::chatterbox {
inline constexpr int SEED = 42;
inline constexpr int N_PREDICT = 1000;
inline constexpr int TOP_K = 0;
inline constexpr float TOP_P = 1.0f;
inline constexpr float MIN_P = 0.05f;
inline constexpr float TEMPERATURE = 0.8f;
inline constexpr float REPEAT_PENALTY = 1.2f;
inline constexpr int REPEAT_LAST_N = 1000;
inline constexpr float CFG_WEIGHT = 0.5f;
inline constexpr int CFM_STEPS = 10;
inline constexpr float CFM_CFG = 0.7f;
inline constexpr int SILENCE_TOKEN = 4299;
}
