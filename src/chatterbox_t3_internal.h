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
#if defined(TTS_FAMILY_V3)
constexpr int CFG_BATCH = 2;
#endif
struct chatterbox_hparams {
#if defined(TTS_FAMILY_V3)
    int32_t n_text_vocab = 0, n_speech_vocab = 0;
    int32_t start_text_token = 0, stop_text_token = 0;
    int32_t start_speech_token = 0, stop_speech_token = 0;
    int32_t n_ctx = 0, n_embd = 0, n_head = 0, n_layer = 0, n_ff = 0;
    int32_t cond_prompt_len = 0, perceiver_len = 0;
    int32_t rope_orig_ctx = 0;
    float eps = 1e-5f, rope_theta = 0.0f;
#else
    int32_t n_text_vocab = 0, n_speech_vocab = 0, start_speech_token = 0, stop_speech_token = 0;
    int32_t n_ctx = 0, n_embd = 0, n_head = 0, n_layer = 0, cond_prompt_len = 0;
    float eps = 1e-5f;
#endif
};
#if defined(TTS_FAMILY_V3)
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
#else
struct gpt2_layer {
    ggml_tensor * ln_1_g = nullptr, * ln_1_b = nullptr, * ln_2_g = nullptr, * ln_2_b = nullptr;
    ggml_tensor * c_attn_attn_w = nullptr, * c_attn_attn_b = nullptr;
    ggml_tensor * c_attn_proj_w = nullptr, * c_attn_proj_b = nullptr;
    ggml_tensor * c_mlp_fc_w = nullptr, * c_mlp_fc_b = nullptr;
    ggml_tensor * c_mlp_proj_w = nullptr, * c_mlp_proj_b = nullptr;
#endif
};
struct chatterbox_model {
    chatterbox_hparams hparams;
#if defined(TTS_FAMILY_V3)
    ggml_tensor * rms_out = nullptr;
    ggml_tensor * text_emb = nullptr, * speech_emb = nullptr, * speech_head = nullptr;
    ggml_tensor * text_pos_emb = nullptr, * speech_pos_emb = nullptr;
#else
    ggml_tensor * wpe = nullptr, * ln_f_g = nullptr, * ln_f_b = nullptr;
    ggml_tensor * text_emb = nullptr, * speech_emb = nullptr, * speech_head = nullptr, * speech_head_bias = nullptr;
#endif
    ggml_tensor * cond_spkr_w = nullptr, * cond_spkr_b = nullptr;
#if defined(TTS_FAMILY_V3)
    ggml_tensor * emotion_adv_fc_w = nullptr;
    ggml_tensor * builtin_speaker_emb = nullptr;
    ggml_tensor * builtin_cond_prompt_tokens = nullptr;
    ggml_tensor * rope_freq_factors = nullptr;
    perceiver_w perceiver;
    std::vector<llama_layer> layers;
#else
    ggml_tensor * builtin_speaker_emb = nullptr, * builtin_cond_prompt_tokens = nullptr;
    std::vector<gpt2_layer> layers;
#endif
    int kv_rows = 0;
    ggml_tensor * memory_k = nullptr, * memory_v = nullptr;
    ggml_context * ctx_w = nullptr, * ctx_kv = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer_w = nullptr, buffer_kv = nullptr;
    std::map<std::string, ggml_tensor *> tensors;
#if defined(TTS_FAMILY_V3)
    std::string language_tokens;
#else
    std::vector<std::string> tok_tokens, tok_merges;
#endif
};
inline void prepare_kv(chatterbox_model& model, int prompt_len, int batches) {
    const int rows = int(std::min<int64_t>(int64_t(prompt_len) + runtime_knobs().n_predict + 1, model.hparams.n_ctx));
    if (rows <= model.kv_rows) return;
    if (model.buffer_kv) ggml_backend_buffer_free(model.buffer_kv);
    if (model.ctx_kv) ggml_free(model.ctx_kv);
    model.ctx_kv = ggml_init({ggml_tensor_overhead() * 2, nullptr, true});
    const int64_t size = int64_t(model.hparams.n_embd) * model.hparams.n_layer * rows * batches;
    model.memory_k = ggml_new_tensor_1d(model.ctx_kv, GGML_TYPE_F32, size);
    model.memory_v = ggml_new_tensor_1d(model.ctx_kv, GGML_TYPE_F32, size);
    model.buffer_kv = ggml_backend_alloc_ctx_tensors(model.ctx_kv, model.backend);
    model.kv_rows = rows;
    ggml_backend_buffer_clear(model.buffer_kv, 0);
}
inline void set_causal_mask(ggml_cgraph* graph, int n) {
    std::vector<ggml_fp16_t> mask(size_t(n) * n, ggml_fp32_to_fp16(0.0f));
    for (int q = 0; q < n; ++q)
        for (int k = q + 1; k < n; ++k) mask[size_t(q) * n + k] = ggml_fp32_to_fp16(-INFINITY);
    ggml_backend_tensor_set(ggml_graph_get_tensor(graph, "kq_mask"), mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
}
inline std::vector<float> probabilities(const std::vector<float>& scores) {
    float mx = -INFINITY;
    for (float s : scores) if (s != -INFINITY) mx = std::max(mx, s);
    std::vector<float> probs(scores.size());
    float sum = 0;
    for (size_t i = 0; i < scores.size(); ++i) {
        probs[i] = scores[i] == -INFINITY ? 0.0f : std::exp(scores[i] - mx);
        sum += probs[i];
    }
    for (float& p : probs) p /= sum;
    return probs;
}
ggml_backend_t init_backend();
void load_model_gguf(const std::string & path, chatterbox_model & model);
void eval_prompt(chatterbox_model &, ggml_gallocr_t, const std::vector<int32_t> &, std::vector<float> &, int &);
#if defined(TTS_FAMILY_V3)
void eval_step(const chatterbox_model &, ggml_gallocr_t, int, int32_t, int, std::vector<float> &);
#else
void eval_step(const chatterbox_model &, ggml_gallocr_t, int, int32_t, std::vector<float> &);
#endif
int32_t sample_next_token_ex(const std::vector<float> &, const std::vector<int32_t> &, std::mt19937 &);
}
