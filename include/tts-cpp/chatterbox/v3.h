#pragma once
namespace tts_cpp::chatterbox {
inline constexpr int SEED = 42;
inline constexpr int N_PREDICT = 1000;
inline constexpr float TOP_P = 1.0f;
inline constexpr float MIN_P = 0.05f;
inline constexpr float TEMPERATURE = 0.8f;
// ResembleAI chatterbox mtl_tts.ChatterboxMultilingualTTS.generate default
// after PR 516 / commit 3f35dfc8 (v3 checkpoint; lowered from v2-era 2.0).
// Omitted CLI knobs use this header. Trident must not force 2.0.
inline constexpr float REPEAT_PENALTY = 1.2f;
inline constexpr float CFG_WEIGHT = 0.5f;
inline constexpr int CFM_STEPS = 10;
inline constexpr float CFM_CFG = 0.7f;
inline constexpr int TRIM_FADE = 480;
}
