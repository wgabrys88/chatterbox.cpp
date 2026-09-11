#pragma once
#include <cmath>
#include <cstdint>
#include <map>
#include <ostream>
#include <random>
#include <set>
#include <string>
#include <vector>
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "tts-cpp/chatterbox/v3.h"
namespace tts_cpp::chatterbox::detail {
constexpr int CHBX_MAX_NODES = 8192;
constexpr int CFG_BATCH = 2;
inline void apply_speech_repeat_penalty(float * scores, int vocab,
                                        const std::vector<int32_t> & generated) {
    if (generated.empty() || vocab <= 0) return;
    const size_t start = generated.size() > (size_t)REPEAT_LAST_N
        ? generated.size() - (size_t)REPEAT_LAST_N : 0;
    std::set<int32_t> seen;
    for (size_t i = start; i < generated.size(); ++i) seen.insert(generated[i]);
    for (int32_t t : seen) {
        if (t < 0 || t >= vocab) continue;
        float & s = scores[t];
        if (s == -INFINITY) continue;
        s = s > 0.0f ? s / REPEAT_PENALTY : s * REPEAT_PENALTY;
    }
}
constexpr const char * KEY_TEXT_VOCAB_SIZE   = "chatterbox.text_vocab_size";
constexpr const char * KEY_SPEECH_VOCAB_SIZE = "chatterbox.speech_vocab_size";
constexpr const char * KEY_START_TEXT        = "chatterbox.start_text_token";
constexpr const char * KEY_STOP_TEXT         = "chatterbox.stop_text_token";
constexpr const char * KEY_START_SPEECH      = "chatterbox.start_speech_token";
constexpr const char * KEY_STOP_SPEECH       = "chatterbox.stop_speech_token";
constexpr const char * KEY_SPEAKER_EMBED     = "chatterbox.speaker_embed_size";
constexpr const char * KEY_LAYER_NORM_EPS    = "chatterbox.layer_norm_eps";
constexpr const char * KEY_COND_PROMPT_LEN   = "chatterbox.cond_prompt_length";
constexpr const char * KEY_N_EMBD            = "chatterbox.n_embd";
constexpr const char * KEY_N_HEAD            = "chatterbox.n_head";
constexpr const char * KEY_N_LAYER           = "chatterbox.n_layer";
constexpr const char * KEY_N_FF              = "chatterbox.n_ff";
constexpr const char * KEY_N_CTX             = "chatterbox.n_ctx";
constexpr const char * KEY_PERCEIVER_LEN     = "chatterbox.perceiver_len";
constexpr const char * KEY_ROPE_THETA        = "chatterbox.rope_theta";
constexpr const char * KEY_ROPE_ORIG_CTX     = "chatterbox.rope_orig_ctx";
struct chatterbox_hparams {
    int32_t n_text_vocab = 0, n_speech_vocab = 0;
    int32_t start_text_token = 0, stop_text_token = 0;
    int32_t start_speech_token = 0, stop_speech_token = 0;
    int32_t n_ctx = 0, n_embd = 0, n_head = 0, n_layer = 0, n_ff = 0;
    int32_t speaker_embed_size = 0, cond_prompt_len = 0, perceiver_len = 0;
    int32_t rope_orig_ctx = 0;
    float eps = 1e-5f, rope_theta = 0.0f;
};
struct llama_layer {
    ggml_tensor * attn_norm = nullptr, * ffn_norm = nullptr;
    ggml_tensor * wq = nullptr, * wk = nullptr, * wv = nullptr, * wo = nullptr;
    ggml_tensor * gate = nullptr, * up = nullptr, * down = nullptr;
};
struct perceiver_w {
    ggml_tensor * query = nullptr;
    ggml_tensor * norm_g = nullptr, * norm_b = nullptr;
    ggml_tensor * to_q_w = nullptr, * to_q_b = nullptr;
    ggml_tensor * to_k_w = nullptr, * to_k_b = nullptr;
    ggml_tensor * to_v_w = nullptr, * to_v_b = nullptr;
    ggml_tensor * proj_w = nullptr, * proj_b = nullptr;
};
struct chatterbox_model {
    chatterbox_hparams hparams;
    ggml_tensor * rms_out = nullptr;
    ggml_tensor * text_emb = nullptr, * speech_emb = nullptr, * speech_head = nullptr;
    ggml_tensor * text_pos_emb = nullptr, * speech_pos_emb = nullptr;
    ggml_tensor * cond_spkr_w = nullptr, * cond_spkr_b = nullptr;
    ggml_tensor * emotion_adv_fc_w = nullptr;
    ggml_tensor * builtin_speaker_emb = nullptr;
    ggml_tensor * builtin_cond_prompt_tokens = nullptr;
    ggml_tensor * builtin_emotion_adv = nullptr;
    ggml_tensor * rope_freq_factors = nullptr;
    perceiver_w perceiver;
    std::vector<llama_layer> layers;
    ggml_tensor * memory_k = nullptr, * memory_v = nullptr;
    ggml_context * ctx_w = nullptr, * ctx_kv = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer_w = nullptr, buffer_kv = nullptr;
    std::map<std::string, ggml_tensor *> tensors;
    std::vector<std::string> tok_tokens, tok_merges;
    std::vector<int> tok_types;
};
ggml_backend_t init_backend();
void load_model_gguf(const std::string & path, chatterbox_model & model);
void eval_prompt(const chatterbox_model &, ggml_gallocr_t, const std::vector<int32_t> &, std::vector<float> &, int &);
void eval_step(const chatterbox_model &, ggml_gallocr_t, int, int32_t, int, std::vector<float> &);
int32_t sample_next_token_ex(const std::vector<float> &, const std::vector<int32_t> &, std::mt19937 &);
extern std::ostream * g_sampler_log;
extern int g_sampler_step;
}
