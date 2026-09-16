#pragma once
namespace tts_cpp::chatterbox {
inline constexpr int SEED = 42;
inline constexpr int N_PREDICT = 1000;
inline constexpr int TOP_K = 1000;
inline constexpr float TOP_P = 0.95f;
inline constexpr float TEMPERATURE = 0.8f;
inline constexpr float REPEAT_PENALTY = 1.2f;
inline constexpr int S3GEN_SIL = 4299;
inline constexpr int SIL_COUNT = 3;
inline constexpr int CFM_STEPS = 2;
inline constexpr int SAMPLES_PER_TOKEN = 960;
inline constexpr int TRIM_FADE = 480;
}
