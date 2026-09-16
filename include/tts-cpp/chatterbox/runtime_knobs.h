#pragma once
#include <string>
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
    float min_p = MIN_P;
    int seed = SEED;
    int n_predict = N_PREDICT;
    int cfm_steps = CFM_STEPS;
    int trim_fade = TRIM_FADE;
    // 0 one-shot burn, 1 tape GGUF only (instrument), 2 steal gauge, 3 lung tank then burn every slice
    int stage = 0;
    // 0 = no extra token cap. Tank C = prompt_fuel_mean * 100. Else also cap slice length.
    int cut_x = 0;
    std::string artifact_path;
#if defined(TTS_FAMILY_GPT2)
    int top_k = TOP_K;
    int sil_count = SIL_COUNT;
    int s3gen_sil = S3GEN_SIL;
#elif defined(TTS_FAMILY_V3)
    float cfg_weight = CFG_WEIGHT;
    float cfm_cfg = CFM_CFG;
    float exaggeration = EXAGGERATION;
#endif
};
inline RuntimeKnobs& runtime_knobs() {
    static RuntimeKnobs k;
    return k;
}
inline float effective_repeat_penalty() { return runtime_knobs().repeat_penalty; }
inline float effective_temperature() { return runtime_knobs().temperature; }
inline float effective_top_p() { return runtime_knobs().top_p; }
inline float effective_min_p() { return runtime_knobs().min_p; }
inline int effective_seed() { return runtime_knobs().seed; }
inline int effective_n_predict() { return runtime_knobs().n_predict; }
inline int effective_cfm_steps() { return runtime_knobs().cfm_steps; }
inline int effective_trim_fade() { return runtime_knobs().trim_fade; }
inline int effective_stage() { return runtime_knobs().stage; }
inline int effective_cut_x() { return runtime_knobs().cut_x; }
#if defined(TTS_FAMILY_GPT2)
inline int effective_top_k() { return runtime_knobs().top_k; }
inline int effective_sil_count() { return runtime_knobs().sil_count; }
inline int effective_s3gen_sil() { return runtime_knobs().s3gen_sil; }
#elif defined(TTS_FAMILY_V3)
inline float effective_cfg_weight() { return runtime_knobs().cfg_weight; }
inline float effective_cfm_cfg() { return runtime_knobs().cfm_cfg; }
inline float effective_exaggeration() { return runtime_knobs().exaggeration; }
#endif
}
