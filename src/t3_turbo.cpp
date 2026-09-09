#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#elif defined(GGML_USE_CUDA)
#include "ggml-cuda.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include "chatterbox_t3_internal.h"

using namespace tts_cpp::chatterbox::detail;
namespace tts_cpp::chatterbox::detail {

static int64_t require_key(const gguf_context * ctx, const char * key) {
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) throw std::runtime_error(std::string("missing GGUF key: ") + key);
    return id;
}
static ggml_tensor * require_tensor(const chatterbox_model & m, const char * name) {
    auto it = m.tensors.find(name);
    if (it == m.tensors.end() || !it->second) throw std::runtime_error(std::string("missing tensor: ") + name);
    return it->second;
}
ggml_backend_t init_backend(int n_gpu_layers) {
    if (n_gpu_layers <= 0) throw std::runtime_error("GPU layers required");
#ifdef GGML_USE_VULKAN
    auto * b = ggml_backend_vk_init(0);
    if (!b) throw std::runtime_error("Vulkan backend init failed");
#elif defined(GGML_USE_CUDA)
    auto * b = ggml_backend_cuda_init(0);
    if (!b) throw std::runtime_error("CUDA backend init failed");
#else
#error "No Chatterbox GPU backend selected"
#endif
    return b;
}
bool load_model_gguf(const std::string & path, chatterbox_model & model, int requested_ctx, int n_gpu_layers) {
    {
        gguf_init_params peek_params = {  true,  nullptr };
        gguf_context * peek_ctx = gguf_init_from_file(path.c_str(), peek_params);
        if (peek_ctx) {
            std::string variant = "t3_turbo";
            const int64_t vk = gguf_find_key(peek_ctx, KEY_VARIANT);
            if (vk >= 0 && gguf_get_kv_type(peek_ctx, vk) == GGUF_TYPE_STRING) {
                const char * v = gguf_get_val_str(peek_ctx, vk);
                if (v) variant = v;
            } else if (vk >= 0) {
                fprintf(stderr, "%s: %s has unexpected GGUF type %d (expected STRING); refusing to load\n",
                        __func__, KEY_VARIANT, (int) gguf_get_kv_type(peek_ctx, vk));
                gguf_free(peek_ctx);
                return false;
            }
            gguf_free(peek_ctx);
            if (variant == "t3_mtl") {
#ifdef TTS_CPP_MTL
                return load_model_gguf_mtl(path, model, requested_ctx, n_gpu_layers);
#else
                fprintf(stderr, "%s: multilingual T3 was not compiled (TTS_CPP_MTL=OFF)\n", __func__);
                return false;
#endif
            }
        }
    }
    ggml_context * tmp_ctx = nullptr;
    gguf_init_params gguf_params = {  false,  &tmp_ctx };
    gguf_context * gguf_ctx = gguf_init_from_file(path.c_str(), gguf_params);
    if (!gguf_ctx) { fprintf(stderr, "%s: failed to open '%s'\n", __func__, path.c_str()); return false; }
    try {
        auto & hp = model.hparams;
        hp.variant = CHBX_VARIANT_TURBO;
        hp.n_text_vocab       = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_TEXT_VOCAB_SIZE));
        hp.n_speech_vocab     = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_SPEECH_VOCAB_SIZE));
        hp.start_speech_token = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_START_SPEECH));
        hp.stop_speech_token  = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_STOP_SPEECH));
        hp.speaker_embed_size = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_SPEAKER_EMBED));
        hp.cond_prompt_len    = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_COND_PROMPT_LEN));
        hp.eps                = gguf_get_val_f32(gguf_ctx, require_key(gguf_ctx, KEY_LAYER_NORM_EPS));
        hp.n_ctx   = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_N_CTX));
        hp.n_embd  = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_N_EMBD));
        hp.n_head  = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_N_HEAD));
        hp.n_layer = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_N_LAYER));
        if (requested_ctx > 0) hp.n_ctx = std::min(hp.n_ctx, requested_ctx);
        model.backend = init_backend(n_gpu_layers);
        const int64_t num_tensors = gguf_get_n_tensors(gguf_ctx);
        ggml_init_params params = { ggml_tensor_overhead() * (size_t) num_tensors, nullptr, true };
        model.ctx_w = ggml_init(params);
        if (!model.ctx_w) throw std::runtime_error("ggml_init() failed");
        for (int64_t i = 0; i < num_tensors; ++i) {
            const char * name = gguf_get_tensor_name(gguf_ctx, i);
            ggml_tensor * src = ggml_get_tensor(tmp_ctx, name);
            ggml_tensor * dst = ggml_dup_tensor(model.ctx_w, src);
            ggml_set_name(dst, name);
            model.tensors[name] = dst;
        }
        model.buffer_w = ggml_backend_alloc_ctx_tensors(model.ctx_w, model.backend);
        for (ggml_tensor * cur = ggml_get_first_tensor(model.ctx_w); cur; cur = ggml_get_next_tensor(model.ctx_w, cur)) {
            ggml_tensor * src = ggml_get_tensor(tmp_ctx, ggml_get_name(cur));
            ggml_backend_tensor_set(cur, ggml_get_data(src), 0, ggml_nbytes(src));
        }
        model.wpe              = require_tensor(model, "model/wpe");
        model.ln_f_g           = require_tensor(model, "model/ln_f/g");
        model.ln_f_b           = require_tensor(model, "model/ln_f/b");
        model.text_emb         = require_tensor(model, "chatterbox/text_emb");
        model.speech_emb       = require_tensor(model, "chatterbox/speech_emb");
        model.speech_head      = require_tensor(model, "chatterbox/speech_head");
        model.speech_head_bias = require_tensor(model, "chatterbox/speech_head_bias");
        model.cond_spkr_w      = require_tensor(model, "chatterbox/cond_spkr/w");
        model.cond_spkr_b      = require_tensor(model, "chatterbox/cond_spkr/b");
        model.builtin_speaker_emb        = require_tensor(model, "chatterbox/builtin/speaker_emb");
        model.builtin_cond_prompt_tokens = require_tensor(model, "chatterbox/builtin/cond_prompt_speech_tokens");
        model.layers.resize(hp.n_layer);
        for (int i = 0; i < hp.n_layer; ++i) {
            auto & l = model.layers[i];
            std::string p = "model/h" + std::to_string(i);
            l.ln_1_g        = require_tensor(model, (p + "/ln_1/g").c_str());
            l.ln_1_b        = require_tensor(model, (p + "/ln_1/b").c_str());
            l.ln_2_g        = require_tensor(model, (p + "/ln_2/g").c_str());
            l.ln_2_b        = require_tensor(model, (p + "/ln_2/b").c_str());
            l.c_attn_attn_w = require_tensor(model, (p + "/attn/c_attn/w").c_str());
            l.c_attn_attn_b = require_tensor(model, (p + "/attn/c_attn/b").c_str());
            l.c_attn_proj_w = require_tensor(model, (p + "/attn/c_proj/w").c_str());
            l.c_attn_proj_b = require_tensor(model, (p + "/attn/c_proj/b").c_str());
            l.c_mlp_fc_w    = require_tensor(model, (p + "/mlp/c_fc/w").c_str());
            l.c_mlp_fc_b    = require_tensor(model, (p + "/mlp/c_fc/b").c_str());
            l.c_mlp_proj_w  = require_tensor(model, (p + "/mlp/c_proj/w").c_str());
            l.c_mlp_proj_b  = require_tensor(model, (p + "/mlp/c_proj/b").c_str());
        }
        ggml_init_params kv_params = { ggml_tensor_overhead() * 2, nullptr, true };
        model.ctx_kv = ggml_init(kv_params);
        int64_t n_elements = (int64_t) hp.n_embd * hp.n_layer * hp.n_ctx;
        model.memory_k = ggml_new_tensor_1d(model.ctx_kv, GGML_TYPE_F32, n_elements);
        model.memory_v = ggml_new_tensor_1d(model.ctx_kv, GGML_TYPE_F32, n_elements);
        model.buffer_kv = ggml_backend_alloc_ctx_tensors(model.ctx_kv, model.backend);
        if (g_log_verbose) fprintf(stderr, "%s: ctx=%d embd=%d layers=%d heads=%d text_vocab=%d speech_vocab=%d cond_prompt=%d\n",
                __func__, hp.n_ctx, hp.n_embd, hp.n_layer, hp.n_head,
                hp.n_text_vocab, hp.n_speech_vocab, hp.cond_prompt_len);
        if (g_log_verbose) fprintf(stderr, "%s: weights=%.2f MB  KV=%.2f MB\n", __func__,
                ggml_backend_buffer_get_size(model.buffer_w) / (1024.0*1024.0),
                ggml_backend_buffer_get_size(model.buffer_kv) / (1024.0*1024.0));
        {
            const int64_t tok_kid = gguf_find_key(gguf_ctx, "tokenizer.ggml.tokens");
            const int64_t mer_kid = gguf_find_key(gguf_ctx, "tokenizer.ggml.merges");
            if (tok_kid >= 0 && mer_kid >= 0) {
                const size_t n_tok = gguf_get_arr_n(gguf_ctx, tok_kid);
                const size_t n_mer = gguf_get_arr_n(gguf_ctx, mer_kid);
                model.tok_tokens.reserve(n_tok);
                for (size_t i = 0; i < n_tok; ++i) {
                    model.tok_tokens.emplace_back(gguf_get_arr_str(gguf_ctx, tok_kid, i));
                }
                model.tok_merges.reserve(n_mer);
                for (size_t i = 0; i < n_mer; ++i) {
                    model.tok_merges.emplace_back(gguf_get_arr_str(gguf_ctx, mer_kid, i));
                }
                if (g_log_verbose) fprintf(stderr, "%s: tokenizer embedded (%zu tokens, %zu merges)\n",
                        __func__, n_tok, n_mer);
            } else {
                fprintf(stderr, "%s: no embedded tokenizer; --tokenizer-dir will be required for --text\n",
                        __func__);
            }
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "%s: %s\n", __func__, e.what());
        gguf_free(gguf_ctx); if (tmp_ctx) ggml_free(tmp_ctx);
        return false;
    }
    gguf_free(gguf_ctx);
    ggml_free(tmp_ctx);
    return true;
}
static ggml_tensor * build_transformer_core(
    ggml_context * ctx, ggml_cgraph * gf,
    const chatterbox_model & model,
    ggml_tensor * inpL, int n_past, int N) {
    const auto & hp = model.hparams;
    const int n_embd = hp.n_embd, n_head = hp.n_head, n_layer = hp.n_layer, n_ctx = hp.n_ctx;
    const int HD = n_embd / n_head;
    const int64_t L = n_past + N;
    const size_t kv_layer_elems  = (size_t) HD * n_ctx * n_head;
    const size_t kv_head_stride  = (size_t) HD * n_ctx * sizeof(float);
    const size_t kv_pos_stride   = (size_t) HD * sizeof(float);
    ggml_tensor * kq_mask = nullptr;
    if (N > 1) {
        kq_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, L, N);
        ggml_set_name(kq_mask, "kq_mask");
        ggml_set_input(kq_mask);
    }
    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * cur;
        cur = ggml_norm(ctx, inpL, hp.eps);
        cur = ggml_add(ctx, ggml_mul(ctx, cur, model.layers[il].ln_1_g), model.layers[il].ln_1_b);
        cur = ggml_add(ctx, ggml_mul_mat(ctx, model.layers[il].c_attn_attn_w, cur), model.layers[il].c_attn_attn_b);
        const size_t qkv_col_stride  = cur->nb[1];
        const size_t qkv_head_stride = (size_t) HD * sizeof(float);
        ggml_tensor * Q = ggml_view_3d(ctx, cur, HD, N, n_head,
            qkv_col_stride,
            qkv_head_stride,
            0);
        ggml_tensor * Kcur_QNH = ggml_view_3d(ctx, cur, HD, N, n_head,
            qkv_col_stride,
            qkv_head_stride,
            (size_t) n_embd * sizeof(float));
        ggml_tensor * Vcur_QNH = ggml_view_3d(ctx, cur, HD, N, n_head,
            qkv_col_stride,
            qkv_head_stride,
            (size_t) 2 * n_embd * sizeof(float));
        const size_t layer_off = (size_t) il * kv_layer_elems * sizeof(float);
        {
            ggml_tensor * k_dst = ggml_view_3d(ctx, model.memory_k,
                HD, N, n_head,
                kv_pos_stride,
                kv_head_stride,
                layer_off + (size_t) n_past * kv_pos_stride);
            ggml_tensor * v_dst = ggml_view_3d(ctx, model.memory_v,
                HD, N, n_head,
                kv_pos_stride,
                kv_head_stride,
                layer_off + (size_t) n_past * kv_pos_stride);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, Kcur_QNH, k_dst));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, Vcur_QNH, v_dst));
        }
        ggml_tensor * K = ggml_view_3d(ctx, model.memory_k,
            HD, L, n_head,
            kv_pos_stride,
            kv_head_stride,
            layer_off);
        ggml_tensor * V = ggml_view_3d(ctx, model.memory_v,
            HD, L, n_head,
            kv_pos_stride,
            kv_head_stride,
            layer_off);
        ggml_tensor * attn = ggml_flash_attn_ext(ctx, Q, K, V, kq_mask,
            1.0f / std::sqrt((float) HD), 0.0f, 0.0f);
        cur = ggml_reshape_2d(ctx, attn, n_embd, N);
        cur = ggml_add(ctx, ggml_add(ctx, ggml_mul_mat(ctx, model.layers[il].c_attn_proj_w, cur), model.layers[il].c_attn_proj_b), inpL);
        ggml_tensor * inpFF = cur;
        cur = ggml_norm(ctx, inpFF, hp.eps);
        cur = ggml_add(ctx, ggml_mul(ctx, cur, model.layers[il].ln_2_g), model.layers[il].ln_2_b);
        cur = ggml_gelu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, model.layers[il].c_mlp_fc_w, cur), model.layers[il].c_mlp_fc_b));
        cur = ggml_add(ctx, ggml_mul_mat(ctx, model.layers[il].c_mlp_proj_w, cur), model.layers[il].c_mlp_proj_b);
        inpL = ggml_add(ctx, cur, inpFF);
    }
    inpL = ggml_norm(ctx, inpL, hp.eps);
    inpL = ggml_add(ctx, ggml_mul(ctx, inpL, model.ln_f_g), model.ln_f_b);
    ggml_tensor * logits = ggml_add(ctx, ggml_mul_mat(ctx, model.speech_head, inpL), model.speech_head_bias);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);
    return logits;
}
static ggml_cgraph * build_prompt_graph(const chatterbox_model & model, int n_text_tokens) {
    const int N = 1 + model.hparams.cond_prompt_len + n_text_tokens + 1;
    static size_t buf_size = ggml_tensor_overhead()*CHBX_MAX_NODES + ggml_graph_overhead_custom(CHBX_MAX_NODES, false);
    thread_local std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, CHBX_MAX_NODES, false);
    ggml_tensor * text_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_text_tokens);
    ggml_set_name(text_tokens, "text_tokens"); ggml_set_input(text_tokens);
    ggml_tensor * start_token = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(start_token, "speech_token"); ggml_set_input(start_token);
    ggml_tensor * position = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_set_name(position, "position"); ggml_set_input(position);
    ggml_tensor * spkr = ggml_add(ctx, ggml_mul_mat(ctx, model.cond_spkr_w, model.builtin_speaker_emb), model.cond_spkr_b);
    ggml_tensor * cond = ggml_get_rows(ctx, model.speech_emb, model.builtin_cond_prompt_tokens);
    ggml_tensor * temb = ggml_get_rows(ctx, model.text_emb, text_tokens);
    ggml_tensor * semb = ggml_get_rows(ctx, model.speech_emb, start_token);
    ggml_tensor * inp = ggml_concat(ctx, spkr, cond, 1);
    inp = ggml_concat(ctx, inp, temb, 1);
    inp = ggml_concat(ctx, inp, semb, 1);
    inp = ggml_add(ctx, inp, ggml_get_rows(ctx, model.wpe, position));
    build_transformer_core(ctx, gf, model, inp, 0, N);
    ggml_free(ctx);
    return gf;
}
static ggml_cgraph * build_step_graph(const chatterbox_model & model, int n_past) {
    static size_t buf_size = ggml_tensor_overhead()*CHBX_MAX_NODES + ggml_graph_overhead_custom(CHBX_MAX_NODES, false);
    thread_local std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, CHBX_MAX_NODES, false);
    ggml_tensor * speech_token = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(speech_token, "speech_token"); ggml_set_input(speech_token);
    ggml_tensor * position = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(position, "position"); ggml_set_input(position);
    ggml_tensor * inp = ggml_add(ctx,
        ggml_get_rows(ctx, model.speech_emb, speech_token),
        ggml_get_rows(ctx, model.wpe, position));
    build_transformer_core(ctx, gf, model, inp, n_past, 1);
    ggml_free(ctx);
    return gf;
}
bool eval_prompt(
    const chatterbox_model & model, ggml_gallocr_t allocr, int n_threads,
    const std::vector<int32_t> & text_tokens, std::vector<float> & logits_out, int & prompt_len) {
    prompt_len = 1 + model.hparams.cond_prompt_len + (int)text_tokens.size() + 1;
    if (prompt_len > model.hparams.n_ctx) {
        fprintf(stderr, "%s: prompt %d exceeds context %d\n", __func__, prompt_len, model.hparams.n_ctx);
        return false;
    }
    ggml_cgraph * gf = build_prompt_graph(model, (int)text_tokens.size());
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "text_tokens"), text_tokens.data(), 0, text_tokens.size()*sizeof(int32_t));
    int32_t st = model.hparams.start_speech_token;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "speech_token"), &st, 0, sizeof(st));
    std::vector<int32_t> pos(prompt_len);
    for (int i = 0; i < prompt_len; ++i) pos[i] = i;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "position"), pos.data(), 0, pos.size()*sizeof(int32_t));
    {
        const int N = prompt_len;
        ggml_tensor * kq_mask = ggml_graph_get_tensor(gf, "kq_mask");
        if (kq_mask) {
            const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
            const ggml_fp16_t ninf_h = ggml_fp32_to_fp16(-INFINITY);
            std::vector<ggml_fp16_t> mask((size_t)N * N, zero_h);
            for (int q = 0; q < N; ++q) {
                for (int k = 0; k < N; ++k) {
                    if (k > q) mask[(size_t)q * N + k] = ninf_h;
                }
            }
            ggml_backend_tensor_set(kq_mask, mask.data(), 0, mask.size()*sizeof(ggml_fp16_t));
        }
    }
    const auto status = ggml_backend_graph_compute(model.backend, gf);
    if (status != GGML_STATUS_SUCCESS) return false;
    ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    logits_out.resize(model.hparams.n_speech_vocab);
    ggml_backend_tensor_get(logits, logits_out.data(),
        (size_t)model.hparams.n_speech_vocab*(prompt_len-1)*sizeof(float),
        (size_t)model.hparams.n_speech_vocab*sizeof(float));
    return true;
}
bool eval_step(
    const chatterbox_model & model, ggml_gallocr_t allocr, int n_threads,
    int n_past, int32_t token, std::vector<float> & logits_out) {
    ggml_cgraph * gf = build_step_graph(model, n_past);
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "speech_token"), &token, 0, sizeof(token));
    int32_t position = n_past;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "position"), &position, 0, sizeof(position));
    const auto status = ggml_backend_graph_compute(model.backend, gf);
    if (status != GGML_STATUS_SUCCESS) return false;
    ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    logits_out.resize(model.hparams.n_speech_vocab);
    ggml_backend_tensor_get(logits, logits_out.data(), 0, (size_t)model.hparams.n_speech_vocab*sizeof(float));
    return true;
}
static void fill_sample_decision(t3_sample_decision * decision,
                                 const std::vector<float> & scores,
                                 const std::vector<int32_t> & generated) {
    if (!decision) return;
    const int n = (int)scores.size();
    decision->candidates = 0;
    struct IS { int idx; float s; };
    std::vector<IS> ranked;
    ranked.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (scores[i] != -INFINITY) {
            decision->candidates++;
            ranked.push_back({i, scores[i]});
        }
    }
    const int top_n = std::min(5, (int)ranked.size());
    if (top_n > 0) {
        std::partial_sort(ranked.begin(), ranked.begin() + top_n, ranked.end(),
            [](const IS& a, const IS& b) { return a.s > b.s; });
    }
    for (int i = 0; i < 5; ++i) {
        if (i < top_n) {
            decision->top5_ids[i] = ranked[i].idx;
            decision->top5_logprobs[i] = ranked[i].s;
        } else {
            decision->top5_ids[i] = 0;
            decision->top5_logprobs[i] = -INFINITY;
        }
    }
    const int m = std::min(4, (int)generated.size());
    for (int i = 0; i < 4; ++i) {
        decision->repeat_last4[i] = i < m ? generated[generated.size() - (size_t)m + i] : 0;
    }
}

int32_t sample_next_token_ex(
    const std::vector<float> & logits,
    const std::vector<int32_t> & generated,
    const chatterbox_sampling_params & params,
    std::mt19937 & rng,
    t3_sample_decision * decision) {
    const int n = (int)logits.size();
    
    std::vector<float> scores(logits.begin(), logits.end());
    if (params.temp > 0.0f && params.temp != 1.0f) {
        float inv_t = 1.0f / params.temp;
        for (float & s : scores) s *= inv_t;
    }
    apply_min_p(scores.data(), n, params.min_p);
    if (params.top_k > 0 && params.top_k < n) {
        std::vector<float> tmp(scores);
        std::nth_element(tmp.begin(), tmp.begin() + params.top_k, tmp.end(), std::greater<float>());
        float threshold = tmp[params.top_k];
        int kept = 0;
        for (float s : scores) if (s > threshold) ++kept;
        if (kept < params.top_k) threshold -= 1e-10f;
        for (float & s : scores) if (s <= threshold) s = -INFINITY;
    }
    if (params.top_p < 1.0f) {
        struct IS { int idx; float s; };
        std::vector<IS> sorted;
        sorted.reserve(n);
        for (int i = 0; i < n; ++i) if (scores[i] != -INFINITY) sorted.push_back({i, scores[i]});
        std::sort(sorted.begin(), sorted.end(), [](const IS& a, const IS& b){ return a.s > b.s; });
        float mx = sorted.empty() ? 0.0f : sorted[0].s;
        std::vector<float> probs(sorted.size());
        float psum = 0;
        for (size_t i = 0; i < sorted.size(); ++i) { probs[i] = std::exp(sorted[i].s - mx); psum += probs[i]; }
        for (float & p : probs) p /= psum;
        float cum = 0;
        std::set<int> keep_set;
        for (size_t i = 0; i < sorted.size(); ++i) {
            cum += probs[i];
            keep_set.insert(sorted[i].idx);
            if (cum >= params.top_p) break;
        }
        if (keep_set.empty() && !sorted.empty()) keep_set.insert(sorted[0].idx);
        for (int i = 0; i < n; ++i) if (keep_set.find(i) == keep_set.end()) scores[i] = -INFINITY;
    }
    apply_speech_repeat_penalty(scores.data(), n, generated, params.repeat_penalty);
    fill_sample_decision(decision, scores, generated);
    float mx = -INFINITY;
    for (float s : scores) if (s != -INFINITY) mx = std::max(mx, s);
    std::vector<float> probs(n);
    float psum = 0;
    for (int i = 0; i < n; ++i) {
        probs[i] = (scores[i] == -INFINITY) ? 0.0f : std::exp(scores[i] - mx);
        psum += probs[i];
    }
    if (psum == 0.0f) {
        if (decision) decision->chosen_id = 0;
        return 0;
    }
    for (float & p : probs) p /= psum;
    int32_t chosen = 0;
    if (params.temp <= 0.0f) {
        chosen = (int32_t)std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
    } else {
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        chosen = (int32_t)dist(rng);
    }
    if (decision) decision->chosen_id = chosen;
    return chosen;
}

}
