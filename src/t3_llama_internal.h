#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <vector>
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "tts-cpp/chatterbox/runtime_knobs.h"
#include "t3_repeat_penalty.h"
namespace tts_cpp::chatterbox::detail {
constexpr int CHBX_MAX_NODES = 8192;
constexpr int CFG_BATCH = 2;
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
constexpr const char * KEY_TEXT_FRONTEND     = "chatterbox.text_frontend_version";
constexpr const char * KEY_TOKENIZER_SHA      = "chatterbox.tokenizer.source_sha256";
constexpr const char * KEY_TOKENIZER_JSON     = "chatterbox.tokenizer.json";
constexpr const char * KEY_CANGJIE_SHA        = "chatterbox.tokenizer.cangjie_sha256";
constexpr const char * KEY_OFFICIAL_TOKENIZER_SHA = "chatterbox.tokenizer.official_source_sha256";
constexpr const char * KEY_OFFICIAL_TTS_SHA   = "chatterbox.tokenizer.official_tts_source_sha256";
constexpr const char * KEY_LANGUAGE_TOKENS    = "chatterbox.tokenizer.language_tokens";
struct chatterbox_hparams {
    int32_t n_text_vocab = 0, n_speech_vocab = 0;
    int32_t start_text_token = 0, stop_text_token = 0;
    int32_t start_speech_token = 0, stop_speech_token = 0;
    int32_t n_ctx = 0, n_embd = 0, n_head = 0, n_layer = 0, n_ff = 0;
    int32_t cond_prompt_len = 0, perceiver_len = 0;
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
    ggml_tensor * rope_freq_factors = nullptr;
    perceiver_w perceiver;
    std::vector<llama_layer> layers;
    int kv_rows = 0;
    ggml_tensor * memory_k = nullptr, * memory_v = nullptr;
    ggml_context * ctx_w = nullptr, * ctx_kv = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer_w = nullptr, buffer_kv = nullptr;
    std::map<std::string, ggml_tensor *> tensors;
    std::string tokenizer_sha256, tokenizer_json, cangjie_sha256, official_tokenizer_sha256, official_tts_sha256, language_tokens;
};
ggml_backend_t init_backend();
void load_model_gguf(const std::string & path, chatterbox_model & model);
void eval_prompt(chatterbox_model &, ggml_gallocr_t, const std::vector<int32_t> &, std::vector<float> &, int &);
void eval_step(const chatterbox_model &, ggml_gallocr_t, int, int32_t, int, std::vector<float> &);
int32_t sample_next_token_ex(const std::vector<float> &, const std::vector<int32_t> &, std::mt19937 &);
}
