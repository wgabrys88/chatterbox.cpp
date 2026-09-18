#pragma once
#include <climits>
#include <limits>
namespace tts_cpp::chatterbox::detail {
struct RuntimeKnobs {
    float repeat_penalty = std::numeric_limits<float>::quiet_NaN();
    float temperature = std::numeric_limits<float>::quiet_NaN();
    float top_p = std::numeric_limits<float>::quiet_NaN();
    int seed = INT_MIN;
    int n_predict = INT_MIN;
    int cfm_steps = INT_MIN;
    int trim_fade = INT_MIN;
#if defined(TTS_FAMILY_GPT2)
    int top_k = INT_MIN;
#elif defined(TTS_FAMILY_V3)
    float min_p = std::numeric_limits<float>::quiet_NaN();
    float cfg_weight = std::numeric_limits<float>::quiet_NaN();
    float exaggeration = std::numeric_limits<float>::quiet_NaN();
    float cfm_cfg = std::numeric_limits<float>::quiet_NaN();
#else
#error TTS_FAMILY must be gpt2 or v3
#endif
};
inline RuntimeKnobs& runtime_knobs() { static RuntimeKnobs k; return k; }
inline float effective_repeat_penalty() { return runtime_knobs().repeat_penalty; }
inline float effective_temperature() { return runtime_knobs().temperature; }
inline float effective_top_p() { return runtime_knobs().top_p; }
inline int effective_seed() { return runtime_knobs().seed; }
inline int effective_n_predict() { return runtime_knobs().n_predict; }
inline int effective_cfm_steps() { return runtime_knobs().cfm_steps; }
inline int effective_trim_fade() { return runtime_knobs().trim_fade; }
#if defined(TTS_FAMILY_GPT2)
inline int effective_top_k() { return runtime_knobs().top_k; }
#elif defined(TTS_FAMILY_V3)
inline float effective_min_p() { return runtime_knobs().min_p; }
inline float effective_cfg_weight() { return runtime_knobs().cfg_weight; }
inline float effective_exaggeration() { return runtime_knobs().exaggeration; }
inline float effective_cfm_cfg() { return runtime_knobs().cfm_cfg; }
#endif
}
