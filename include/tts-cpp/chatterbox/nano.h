#pragma once
namespace tts_cpp::chatterbox {
inline constexpr int N_THREADS = 4;
inline constexpr int SEED = 42;
inline constexpr int N_PREDICT = 8196;
inline constexpr int TOP_K = 1000;
inline constexpr float TOP_P = 0.95f;
inline constexpr float TEMPERATURE = 0.5f;
inline constexpr float REPEAT_PENALTY = 1.2f;
inline constexpr int REPEAT_LAST_N = 4;
inline constexpr int REPEAT_STOP = 16;
inline constexpr int CFM_STEPS = 1;
}
