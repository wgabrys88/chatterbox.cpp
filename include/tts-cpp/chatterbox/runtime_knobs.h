#pragma once
#if defined(TTS_FAMILY_GPT2)
#include "tts-cpp/chatterbox/gpt2.h"
#elif defined(TTS_FAMILY_V3)
#include "tts-cpp/chatterbox/v3.h"
#else
#error TTS_FAMILY must be gpt2 or v3
#endif
namespace tts_cpp::chatterbox::detail {
struct RuntimeKnobs {
    float repeat_penalty = REPEAT_PENALTY;
    float temperature = TEMPERATURE;
    float top_p = TOP_P;
    int seed = SEED;
    int n_predict = N_PREDICT;
    int cfm_steps = CFM_STEPS;
    int trim_fade = TRIM_FADE;
#if defined(TTS_FAMILY_GPT2)
    int top_k = TOP_K;
#elif defined(TTS_FAMILY_V3)
    float min_p = MIN_P;
    float cfg_weight = CFG_WEIGHT;
    float exaggeration = EXAGGERATION;
    float cfm_cfg = CFM_CFG;
#endif
};
inline RuntimeKnobs& runtime_knobs() {
    static RuntimeKnobs k;
    return k;
}
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
