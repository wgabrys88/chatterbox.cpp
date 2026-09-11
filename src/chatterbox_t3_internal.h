#pragma once
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <ostream>
#include <random>
#include <set>
#include <string>
#include <vector>
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "tts-cpp/chatterbox/turbo.h"
namespace tts_cpp::chatterbox::detail {
constexpr int CHBX_MAX_NODES = 8192;
inline bool sampler_log_enabled() {
    const char * v = std::getenv("CHATTERBOX_SAMPLER_LOG");
    return v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y');
}
inline float effective_repeat_penalty() {
    const char * v = std::getenv("CHATTERBOX_REPEAT_PENALTY");
    if (v && *v) {
        char * end = nullptr;
        const float f = std::strtof(v, &end);
        if (end != v && f > 0.f) return f;
    }
    return REPEAT_PENALTY;
}
inline void apply_speech_repeat_penalty(float * scores, int vocab,
                                        const std::vector<int32_t> & generated) {
    if (generated.empty() || vocab <= 0) return;
    const float penalty = effective_repeat_penalty();
    const size_t start = generated.size() > (size_t)REPEAT_LAST_N
        ? generated.size() - (size_t)REPEAT_LAST_N : 0;
    std::set<int32_t> seen;
    for (size_t i = start; i < generated.size(); ++i) seen.insert(generated[i]);
    for (int32_t t : seen) {
        if (t < 0 || t >= vocab) continue;
        float & s = scores[t];
        if (s == -INFINITY) continue;
        s = s > 0.0f ? s / penalty : s * penalty;
    }
}
constexpr const char * KEY_TEXT_VOCAB_SIZE   = "chatterbox.text_vocab_size";
constexpr const char * KEY_SPEECH_VOCAB_SIZE = "chatterbox.speech_vocab_size";
constexpr const char * KEY_START_SPEECH      = "chatterbox.start_speech_token";
constexpr const char * KEY_STOP_SPEECH       = "chatterbox.stop_speech_token";
constexpr const char * KEY_SPEAKER_EMBED     = "chatterbox.speaker_embed_size";
constexpr const char * KEY_LAYER_NORM_EPS    = "chatterbox.layer_norm_eps";
constexpr const char * KEY_COND_PROMPT_LEN   = "chatterbox.cond_prompt_length";
constexpr const char * KEY_N_EMBD            = "chatterbox.n_embd";
constexpr const char * KEY_N_HEAD            = "chatterbox.n_head";
constexpr const char * KEY_N_LAYER           = "chatterbox.n_layer";
struct chatterbox_hparams {
    int32_t n_text_vocab = 0, n_speech_vocab = 0, start_speech_token = 0, stop_speech_token = 0;
    int32_t n_ctx = 0, n_embd = 0, n_head = 0, n_layer = 0, speaker_embed_size = 0, cond_prompt_len = 0;
    float eps = 1e-5f;
};
struct gpt2_layer {
    ggml_tensor * ln_1_g = nullptr, * ln_1_b = nullptr, * ln_2_g = nullptr, * ln_2_b = nullptr;
    ggml_tensor * c_attn_attn_w = nullptr, * c_attn_attn_b = nullptr;
    ggml_tensor * c_attn_proj_w = nullptr, * c_attn_proj_b = nullptr;
    ggml_tensor * c_mlp_fc_w = nullptr, * c_mlp_fc_b = nullptr;
    ggml_tensor * c_mlp_proj_w = nullptr, * c_mlp_proj_b = nullptr;
};
struct chatterbox_model {
    chatterbox_hparams hparams;
    ggml_tensor * wpe = nullptr, * ln_f_g = nullptr, * ln_f_b = nullptr;
    ggml_tensor * text_emb = nullptr, * speech_emb = nullptr, * speech_head = nullptr, * speech_head_bias = nullptr;
    ggml_tensor * cond_spkr_w = nullptr, * cond_spkr_b = nullptr;
    ggml_tensor * builtin_speaker_emb = nullptr, * builtin_cond_prompt_tokens = nullptr;
    std::vector<gpt2_layer> layers;
    ggml_tensor * memory_k = nullptr, * memory_v = nullptr;
    ggml_context * ctx_w = nullptr, * ctx_kv = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer_w = nullptr, buffer_kv = nullptr;
    std::map<std::string, ggml_tensor *> tensors;
    std::vector<std::string> tok_tokens, tok_merges;
};
ggml_backend_t init_backend();
void load_model_gguf(const std::string & path, chatterbox_model & model);
void eval_prompt(const chatterbox_model &, ggml_gallocr_t, const std::vector<int32_t> &, std::vector<float> &, int &);
void eval_step(const chatterbox_model &, ggml_gallocr_t, int, int32_t, std::vector<float> &);
int32_t sample_next_token_ex(const std::vector<float> &, const std::vector<int32_t> &, std::mt19937 &);
extern std::ostream * g_sampler_log;
extern int g_sampler_step;
}
