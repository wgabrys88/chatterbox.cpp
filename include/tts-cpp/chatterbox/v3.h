#pragma once
namespace tts_cpp::chatterbox {
inline constexpr int SEED = 42;
inline constexpr int N_PREDICT = 1000;
inline constexpr float TOP_P = 1.0f;
inline constexpr float MIN_P = 0.05f;
inline constexpr float TEMPERATURE = 0.8f;
// mtl_tts.generate passes 2.0; t3.inference() default is 1.2. Multilingual API uses 2.0.
inline constexpr float REPEAT_PENALTY = 2.0f;
inline constexpr float CFG_WEIGHT = 0.5f;
inline constexpr int CFM_STEPS = 10;
inline constexpr float CFM_CFG = 0.7f;
inline constexpr int TRIM_FADE = 480;
}
