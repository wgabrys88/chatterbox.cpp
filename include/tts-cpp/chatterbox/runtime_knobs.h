#pragma once
#if defined(TTS_FAMILY_TURBO)
#include "tts-cpp/chatterbox/turbo.h"
#elif defined(TTS_FAMILY_V3)
#include "tts-cpp/chatterbox/v3.h"
#else
#include "tts-cpp/chatterbox/nano.h"
#endif
namespace tts_cpp::chatterbox::detail {
// One product flag. streaming = framed PCM emitted while T3 runs.
// batching = one complete WAV after the whole text. Weights, knobs, GGUF
// are identical in both modes.
enum class Mode { Streaming, Batching };
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
    // Utterance splitter budget in text BPE tokens per T3 KV. 0 = no split.
    int split_tokens = SPLIT_TOKENS;
    // KV allocation cap. 0 = architecture (wpe rows / GGUF n_ctx).
    int n_ctx = N_CTX;
    Mode mode = MODE_STREAMING ? Mode::Streaming : Mode::Batching;
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
inline int effective_split_tokens() { return runtime_knobs().split_tokens; }
inline int effective_n_ctx() { return runtime_knobs().n_ctx; }
inline Mode effective_mode() { return runtime_knobs().mode; }
inline const char* mode_name(Mode m) { return m == Mode::Streaming ? "streaming" : "batching"; }
#if defined(TTS_FAMILY_V3)
inline float effective_min_p() { return runtime_knobs().min_p; }
inline float effective_cfg_weight() { return runtime_knobs().cfg_weight; }
inline float effective_cfm_cfg() { return runtime_knobs().cfm_cfg; }
#else
inline int effective_silence_count() { return runtime_knobs().silence_count; }
#endif
}
