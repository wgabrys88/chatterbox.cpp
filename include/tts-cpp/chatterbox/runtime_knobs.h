#pragma once
#if defined(TTS_FAMILY_TURBO)
#include "tts-cpp/chatterbox/turbo.h"
#elif defined(TTS_FAMILY_V3)
#include "tts-cpp/chatterbox/v3.h"
#else
#include "tts-cpp/chatterbox/nano.h"
#endif
namespace tts_cpp::chatterbox::detail {
struct RuntimeKnobs {
    float repeat_penalty = REPEAT_PENALTY;
    float temperature = TEMPERATURE;
    int top_k = TOP_K;
    float top_p = TOP_P;
    int seed = SEED;
    int n_predict = N_PREDICT;
    int cfm_steps = CFM_STEPS;
    int silence_token = SILENCE_TOKEN;
#if defined(TTS_FAMILY_V3)
    float min_p = MIN_P;
    float cfg_weight = CFG_WEIGHT;
    float cfm_cfg = CFM_CFG;
#else
    int silence_count = SILENCE_COUNT;
#endif
    bool sampler_log = false;
};
inline RuntimeKnobs& runtime_knobs() {
    static RuntimeKnobs k;
    return k;
}
inline bool sampler_log_enabled() { return runtime_knobs().sampler_log; }
inline float effective_repeat_penalty() { return runtime_knobs().repeat_penalty; }
inline float effective_temperature() { return runtime_knobs().temperature; }
inline int effective_top_k() { return runtime_knobs().top_k; }
inline float effective_top_p() { return runtime_knobs().top_p; }
inline int effective_seed() { return runtime_knobs().seed; }
inline int effective_n_predict() { return runtime_knobs().n_predict; }
inline int effective_cfm_steps() { return runtime_knobs().cfm_steps; }
inline int effective_silence_token() { return runtime_knobs().silence_token; }
#if defined(TTS_FAMILY_V3)
inline float effective_min_p() { return runtime_knobs().min_p; }
inline float effective_cfg_weight() { return runtime_knobs().cfg_weight; }
inline float effective_cfm_cfg() { return runtime_knobs().cfm_cfg; }
#else
inline int effective_silence_count() { return runtime_knobs().silence_count; }
#endif
}
