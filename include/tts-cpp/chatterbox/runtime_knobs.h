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
}
