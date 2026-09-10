#pragma once
#include <cmath>
#include <cstdint>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "tts-cpp/chatterbox/nano.h"
namespace tts_cpp::chatterbox::detail {
constexpr int CHBX_MAX_NODES = 8192;

inline void apply_speech_repeat_penalty(float * scores, int vocab,
                                        const std::vector<int32_t> & generated,
                                        float penalty) {
    if (penalty == 1.0f || generated.empty() || vocab <= 0) return;
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
    int32_t n_text_vocab       = 0;
    int32_t n_speech_vocab     = 0;
    int32_t start_speech_token = 0;
    int32_t stop_speech_token  = 0;
    int32_t n_ctx              = 0;
    int32_t n_embd             = 0;
    int32_t n_head             = 0;
    int32_t n_layer            = 0;
    int32_t speaker_embed_size = 0;
    int32_t cond_prompt_len    = 0;
    float   eps                = 1e-5f;
};
struct gpt2_layer {
    ggml_tensor * ln_1_g = nullptr;
    ggml_tensor * ln_1_b = nullptr;
    ggml_tensor * ln_2_g = nullptr;
    ggml_tensor * ln_2_b = nullptr;
    ggml_tensor * c_attn_attn_w = nullptr;
    ggml_tensor * c_attn_attn_b = nullptr;
    ggml_tensor * c_attn_proj_w = nullptr;
    ggml_tensor * c_attn_proj_b = nullptr;
    ggml_tensor * c_mlp_fc_w   = nullptr;
    ggml_tensor * c_mlp_fc_b   = nullptr;
    ggml_tensor * c_mlp_proj_w = nullptr;
    ggml_tensor * c_mlp_proj_b = nullptr;
};
struct chatterbox_model {
    chatterbox_hparams hparams;
    ggml_tensor * wpe              = nullptr;
    ggml_tensor * ln_f_g           = nullptr;
    ggml_tensor * ln_f_b           = nullptr;
    ggml_tensor * text_emb         = nullptr;
    ggml_tensor * speech_emb       = nullptr;
    ggml_tensor * speech_head      = nullptr;
    ggml_tensor * speech_head_bias = nullptr;
    ggml_tensor * cond_spkr_w      = nullptr;
    ggml_tensor * cond_spkr_b      = nullptr;
    ggml_tensor * builtin_speaker_emb        = nullptr;
    ggml_tensor * builtin_cond_prompt_tokens = nullptr;
    std::vector<gpt2_layer>  layers;
    ggml_tensor * memory_k = nullptr;
    ggml_tensor * memory_v = nullptr;
    ggml_context * ctx_w  = nullptr;
    ggml_context * ctx_kv = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer_w  = nullptr;
    ggml_backend_buffer_t buffer_kv = nullptr;
    ggml_context *        ctx_override    = nullptr;
    ggml_backend_buffer_t buffer_override = nullptr;
    std::map<std::string, ggml_tensor *> tensors;
    std::vector<std::string> tok_tokens;
    std::vector<std::string> tok_merges;
};
struct chatterbox_sampling_params {
    int32_t top_k          = 1000;
    float   top_p          = 0.95f;
    float   temp           = 0.5f;
    float   repeat_penalty = 1.2f;
};
ggml_backend_t init_backend();
bool load_model_gguf(
    const std::string & path,
    chatterbox_model &  model);
bool eval_prompt(
    const chatterbox_model &     model,
    ggml_gallocr_t               allocr,
    const std::vector<int32_t> & text_tokens,
    std::vector<float> &         logits_out,
    int &                        prompt_len);
bool eval_step(
    const chatterbox_model & model,
    ggml_gallocr_t           allocr,
    int                      n_past,
    int32_t                  token,
    std::vector<float> &     logits_out);
int32_t sample_next_token_ex(
    const std::vector<float> &          logits,
    const std::vector<int32_t> &        generated,
    const chatterbox_sampling_params &  params,
    std::mt19937 &                      rng);
void chatterbox_log_cb(ggml_log_level level, const char * text, void * ud);
bool compute_prompt_feat_native(
    const std::string &  wav_path,
    const std::string &  s3gen_gguf,
    std::vector<float> & prompt_feat,
    int &                prompt_feat_rows,
    ggml_backend_t       backend);
bool compute_embedding_native(
    const std::string &  wav_path,
    const std::string &  s3gen_gguf,
    std::vector<float> & embedding,
    ggml_backend_t       backend);
bool compute_speech_tokens_native(
    const std::string &    wav_path,
    const std::string &    s3gen_gguf,
    int                    max_cond_tokens,
    std::vector<int32_t> & prompt_token,
    std::vector<int32_t> & cond_prompt_tokens,
    ggml_backend_t         backend);
bool validate_reference_audio(const std::string & path);
}
