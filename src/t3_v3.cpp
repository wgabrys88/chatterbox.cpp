#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "ggml-vulkan.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
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
ggml_backend_t init_backend() {
    auto * b = ggml_backend_vk_init(0);
    if (!b) throw std::runtime_error("Vulkan backend init failed");
    return b;
}
void load_model_gguf(const std::string & path, chatterbox_model & model) {
    ggml_context * tmp_ctx = nullptr;
    gguf_init_params gguf_params = { false, &tmp_ctx };
    gguf_context * gguf_ctx = gguf_init_from_file(path.c_str(), gguf_params);
    if (!gguf_ctx) throw std::runtime_error("T3 GGUF open failed");
    try {
        auto & hp = model.hparams;
        hp.n_text_vocab       = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_TEXT_VOCAB_SIZE));
        hp.n_speech_vocab     = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_SPEECH_VOCAB_SIZE));
        hp.start_text_token   = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_START_TEXT));
        hp.stop_text_token    = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_STOP_TEXT));
        hp.start_speech_token = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_START_SPEECH));
        hp.stop_speech_token  = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_STOP_SPEECH));
        hp.speaker_embed_size = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_SPEAKER_EMBED));
        hp.cond_prompt_len    = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_COND_PROMPT_LEN));
        hp.eps                = gguf_get_val_f32(gguf_ctx, require_key(gguf_ctx, KEY_LAYER_NORM_EPS));
        hp.n_embd  = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_N_EMBD));
        hp.n_head  = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_N_HEAD));
        hp.n_layer = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_N_LAYER));
        hp.n_ff    = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_N_FF));
        hp.n_ctx   = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_N_CTX));
        hp.perceiver_len = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_PERCEIVER_LEN));
        hp.rope_theta    = gguf_get_val_f32(gguf_ctx, require_key(gguf_ctx, KEY_ROPE_THETA));
        hp.rope_orig_ctx = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_ROPE_ORIG_CTX));
        if (!model.backend) throw std::runtime_error("Vulkan backend required");
        if (hp.n_embd % hp.n_head) throw std::runtime_error("n_head");
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
        model.rms_out          = require_tensor(model, "model/norm/g");
        model.text_emb         = require_tensor(model, "chatterbox/text_emb");
        model.speech_emb       = require_tensor(model, "chatterbox/speech_emb");
        model.speech_head      = require_tensor(model, "chatterbox/speech_head");
        model.text_pos_emb     = require_tensor(model, "chatterbox/text_pos_emb");
        model.speech_pos_emb   = require_tensor(model, "chatterbox/speech_pos_emb");
        model.cond_spkr_w      = require_tensor(model, "chatterbox/cond_spkr/w");
        model.cond_spkr_b      = require_tensor(model, "chatterbox/cond_spkr/b");
        model.emotion_adv_fc_w = require_tensor(model, "chatterbox/emotion_adv_fc/w");
        model.builtin_speaker_emb        = require_tensor(model, "chatterbox/builtin/speaker_emb");
        model.builtin_cond_prompt_tokens = require_tensor(model, "chatterbox/builtin/cond_prompt_speech_tokens");
        model.builtin_emotion_adv        = require_tensor(model, "chatterbox/builtin/emotion_adv");
        model.rope_freq_factors          = require_tensor(model, "model/rope_freq_factors");
        model.perceiver.query   = require_tensor(model, "chatterbox/perceiver/pre_attention_query");
        model.perceiver.norm_g  = require_tensor(model, "chatterbox/perceiver/attn/norm/g");
        model.perceiver.norm_b  = require_tensor(model, "chatterbox/perceiver/attn/norm/b");
        model.perceiver.to_q_w  = require_tensor(model, "chatterbox/perceiver/attn/to_q/w");
        model.perceiver.to_q_b  = require_tensor(model, "chatterbox/perceiver/attn/to_q/b");
        model.perceiver.to_k_w  = require_tensor(model, "chatterbox/perceiver/attn/to_k/w");
        model.perceiver.to_k_b  = require_tensor(model, "chatterbox/perceiver/attn/to_k/b");
        model.perceiver.to_v_w  = require_tensor(model, "chatterbox/perceiver/attn/to_v/w");
        model.perceiver.to_v_b  = require_tensor(model, "chatterbox/perceiver/attn/to_v/b");
        model.perceiver.proj_w  = require_tensor(model, "chatterbox/perceiver/attn/proj_out/w");
        model.perceiver.proj_b  = require_tensor(model, "chatterbox/perceiver/attn/proj_out/b");
        hp.cond_prompt_len = (int32_t) ggml_nelements(model.builtin_cond_prompt_tokens);
        model.layers.resize(hp.n_layer);
        for (int i = 0; i < hp.n_layer; ++i) {
            auto & l = model.layers[i];
            std::string p = "model/h" + std::to_string(i);
            l.attn_norm = require_tensor(model, (p + "/attn_norm/g").c_str());
            l.ffn_norm  = require_tensor(model, (p + "/ffn_norm/g").c_str());
            l.wq = require_tensor(model, (p + "/attn/q/w").c_str());
            l.wk = require_tensor(model, (p + "/attn/k/w").c_str());
            l.wv = require_tensor(model, (p + "/attn/v/w").c_str());
            l.wo = require_tensor(model, (p + "/attn/o/w").c_str());
            l.gate = require_tensor(model, (p + "/ffn/gate/w").c_str());
            l.up   = require_tensor(model, (p + "/ffn/up/w").c_str());
            l.down = require_tensor(model, (p + "/ffn/down/w").c_str());
        }
        ggml_init_params kv_params = { ggml_tensor_overhead() * 2, nullptr, true };
        model.ctx_kv = ggml_init(kv_params);
        const int HD = hp.n_embd / hp.n_head;
        int64_t n_elements = (int64_t) HD * hp.n_ctx * hp.n_head * CFG_BATCH * hp.n_layer;
        model.memory_k = ggml_new_tensor_1d(model.ctx_kv, GGML_TYPE_F32, n_elements);
        model.memory_v = ggml_new_tensor_1d(model.ctx_kv, GGML_TYPE_F32, n_elements);
        model.buffer_kv = ggml_backend_alloc_ctx_tensors(model.ctx_kv, model.backend);
        {
            const int64_t tok_kid = require_key(gguf_ctx, "tokenizer.ggml.tokens");
            const int64_t mer_kid = require_key(gguf_ctx, "tokenizer.ggml.merges");
            const int64_t typ_kid = require_key(gguf_ctx, "tokenizer.ggml.token_type");
            const size_t n_tok = gguf_get_arr_n(gguf_ctx, tok_kid);
            const size_t n_mer = gguf_get_arr_n(gguf_ctx, mer_kid);
            const size_t n_typ = gguf_get_arr_n(gguf_ctx, typ_kid);
            if (n_typ != n_tok) throw std::runtime_error("token_type");
            if (gguf_get_arr_type(gguf_ctx, typ_kid) != GGUF_TYPE_INT32) throw std::runtime_error("token_type dtype");
            const int32_t * types = (const int32_t *) gguf_get_arr_data(gguf_ctx, typ_kid);
            model.tok_tokens.reserve(n_tok);
            model.tok_types.reserve(n_typ);
            for (size_t i = 0; i < n_tok; ++i) {
                model.tok_tokens.emplace_back(gguf_get_arr_str(gguf_ctx, tok_kid, i));
                model.tok_types.push_back(types[i]);
            }
            model.tok_merges.reserve(n_mer);
            for (size_t i = 0; i < n_mer; ++i)
                model.tok_merges.emplace_back(gguf_get_arr_str(gguf_ctx, mer_kid, i));
        }
    } catch (...) {
        gguf_free(gguf_ctx); if (tmp_ctx) ggml_free(tmp_ctx);
        throw;
    }
    gguf_free(gguf_ctx);
    ggml_free(tmp_ctx);
}

static ggml_tensor * rms(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
}
static ggml_tensor * linear(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, ggml_tensor * b) {
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    return b ? ggml_add(ctx, y, b) : y;
}
static ggml_tensor * perceiver_attn(ggml_context * ctx, const perceiver_w & w, ggml_tensor * x1, ggml_tensor * x2) {
    const int n_embd = (int)x1->ne[0];
    const int n_q = (int)x1->ne[1];
    const int n_kv = (int)x2->ne[1];
    const int n_head = 4;
    const int HD = n_embd / n_head;
    ggml_tensor * n1 = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x1, 1e-5f), w.norm_g), w.norm_b);
    ggml_tensor * n2 = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x2, 1e-5f), w.norm_g), w.norm_b);
    ggml_tensor * q = linear(ctx, w.to_q_w, n1, w.to_q_b);
    ggml_tensor * k = linear(ctx, w.to_k_w, n2, w.to_k_b);
    ggml_tensor * v = linear(ctx, w.to_v_w, n2, w.to_v_b);
    q = ggml_reshape_3d(ctx, q, HD, n_q, n_head);
    k = ggml_reshape_3d(ctx, k, HD, n_kv, n_head);
    v = ggml_reshape_3d(ctx, v, HD, n_kv, n_head);
    ggml_tensor * attn = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f / std::sqrt((float)HD), 0.0f, 0.0f);
    ggml_tensor * flat = ggml_reshape_2d(ctx, attn, n_embd, n_q);
    return ggml_add(ctx, x1, linear(ctx, w.proj_w, flat, w.proj_b));
}
static ggml_tensor * run_perceiver(ggml_context * ctx, const chatterbox_model & model, ggml_tensor * h) {
    ggml_tensor * query = ggml_reshape_2d(ctx, model.perceiver.query, model.hparams.n_embd, model.hparams.perceiver_len);
    ggml_tensor * pre = perceiver_attn(ctx, model.perceiver, query, h);
    return perceiver_attn(ctx, model.perceiver, pre, pre);
}
static ggml_tensor * build_transformer_core(
    ggml_context * ctx, ggml_cgraph * gf,
    const chatterbox_model & model,
    ggml_tensor * inpL, int n_past, int N) {
    const auto & hp = model.hparams;
    const int n_embd = hp.n_embd, n_head = hp.n_head, n_layer = hp.n_layer, n_ctx = hp.n_ctx;
    const int HD = n_embd / n_head;
    const int64_t L = n_past + N;
    const size_t kv_pos_stride   = (size_t) HD * sizeof(float);
    const size_t kv_head_stride  = (size_t) HD * n_ctx * sizeof(float);
    const size_t kv_batch_stride = (size_t) HD * n_ctx * n_head * sizeof(float);
    const size_t kv_layer_elems  = (size_t) HD * n_ctx * n_head * CFG_BATCH;
    ggml_tensor * kq_mask = nullptr;
    if (N > 1) {
        kq_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, L, N);
        ggml_set_name(kq_mask, "kq_mask");
        ggml_set_input(kq_mask);
    }
    ggml_tensor * position = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_set_name(position, "position");
    ggml_set_input(position);
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * cur = rms(ctx, inpL, layer.attn_norm, hp.eps);
        ggml_tensor * Q = linear(ctx, layer.wq, cur, nullptr);
        ggml_tensor * K = linear(ctx, layer.wk, cur, nullptr);
        ggml_tensor * V = linear(ctx, layer.wv, cur, nullptr);
        Q = ggml_reshape_4d(ctx, Q, HD, n_head, N, CFG_BATCH);
        K = ggml_reshape_4d(ctx, K, HD, n_head, N, CFG_BATCH);
        V = ggml_reshape_4d(ctx, V, HD, n_head, N, CFG_BATCH);
        Q = ggml_rope_ext(ctx, Q, position, model.rope_freq_factors, HD, GGML_ROPE_TYPE_NEOX,
                          hp.rope_orig_ctx, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(ctx, K, position, model.rope_freq_factors, HD, GGML_ROPE_TYPE_NEOX,
                          hp.rope_orig_ctx, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
        K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
        V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));
        const size_t layer_off = (size_t) il * kv_layer_elems * sizeof(float);
        {
            ggml_tensor * k_dst = ggml_view_4d(ctx, model.memory_k,
                HD, N, n_head, CFG_BATCH,
                kv_pos_stride, kv_head_stride, kv_batch_stride,
                layer_off + (size_t) n_past * kv_pos_stride);
            ggml_tensor * v_dst = ggml_view_4d(ctx, model.memory_v,
                HD, N, n_head, CFG_BATCH,
                kv_pos_stride, kv_head_stride, kv_batch_stride,
                layer_off + (size_t) n_past * kv_pos_stride);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, K, k_dst));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, V, v_dst));
        }
        ggml_tensor * Kc = ggml_view_4d(ctx, model.memory_k,
            HD, L, n_head, CFG_BATCH,
            kv_pos_stride, kv_head_stride, kv_batch_stride, layer_off);
        ggml_tensor * Vc = ggml_view_4d(ctx, model.memory_v,
            HD, L, n_head, CFG_BATCH,
            kv_pos_stride, kv_head_stride, kv_batch_stride, layer_off);
        ggml_tensor * attn = ggml_flash_attn_ext(ctx, Q, Kc, Vc, kq_mask,
            1.0f / std::sqrt((float) HD), 0.0f, 0.0f);
        cur = ggml_reshape_3d(ctx, attn, n_embd, N, CFG_BATCH);
        cur = ggml_add(ctx, linear(ctx, layer.wo, cur, nullptr), inpL);
        ggml_tensor * inpFF = cur;
        cur = rms(ctx, inpFF, layer.ffn_norm, hp.eps);
        ggml_tensor * gate = linear(ctx, layer.gate, cur, nullptr);
        ggml_tensor * up = linear(ctx, layer.up, cur, nullptr);
        cur = ggml_swiglu_split(ctx, gate, up);
        cur = linear(ctx, layer.down, cur, nullptr);
        inpL = ggml_add(ctx, cur, inpFF);
    }
    inpL = rms(ctx, inpL, model.rms_out, hp.eps);
    ggml_tensor * logits = ggml_mul_mat(ctx, model.speech_head, inpL);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);
    return logits;
}
static ggml_tensor * repeat_batch(ggml_context * ctx, ggml_tensor * x, int n_embd, int T) {
    ggml_tensor * x3 = ggml_reshape_3d(ctx, x, n_embd, T, 1);
    ggml_tensor * tmpl = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, T, CFG_BATCH);
    return ggml_repeat(ctx, x3, tmpl);
}
static ggml_cgraph * build_prompt_graph(const chatterbox_model & model, int n_text_tokens) {
    const int n_embd = model.hparams.n_embd;
    const int cond_len = 1 + model.hparams.perceiver_len + 1;
    const int N = cond_len + n_text_tokens + 2;
    static size_t buf_size = ggml_tensor_overhead()*CHBX_MAX_NODES + ggml_graph_overhead_custom(CHBX_MAX_NODES, false);
    thread_local std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, CHBX_MAX_NODES, false);
    ggml_tensor * text_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_text_tokens);
    ggml_set_name(text_tokens, "text_tokens"); ggml_set_input(text_tokens);
    ggml_tensor * text_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_text_tokens);
    ggml_set_name(text_pos, "text_pos"); ggml_set_input(text_pos);
    ggml_tensor * cond_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, model.hparams.cond_prompt_len);
    ggml_set_name(cond_pos, "cond_pos"); ggml_set_input(cond_pos);
    ggml_tensor * bos_tok = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(bos_tok, "bos_tok"); ggml_set_input(bos_tok);
    ggml_tensor * bos_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(bos_pos, "bos_pos"); ggml_set_input(bos_pos);
    ggml_tensor * spkr = ggml_add(ctx, ggml_mul_mat(ctx, model.cond_spkr_w, model.builtin_speaker_emb), model.cond_spkr_b);
    ggml_tensor * h = ggml_add(ctx,
        ggml_get_rows(ctx, model.speech_emb, model.builtin_cond_prompt_tokens),
        ggml_get_rows(ctx, model.speech_pos_emb, cond_pos));
    ggml_tensor * perc = run_perceiver(ctx, model, h);
    ggml_tensor * emo = ggml_mul_mat(ctx, model.emotion_adv_fc_w, model.builtin_emotion_adv);
    ggml_tensor * cond = ggml_concat(ctx, spkr, perc, 1);
    cond = ggml_concat(ctx, cond, emo, 1);
    cond = repeat_batch(ctx, cond, n_embd, cond_len);
    ggml_tensor * temb = ggml_add(ctx,
        ggml_get_rows(ctx, model.text_emb, text_tokens),
        ggml_get_rows(ctx, model.text_pos_emb, text_pos));
    ggml_tensor * tpos = ggml_get_rows(ctx, model.text_pos_emb, text_pos);
    ggml_tensor * t0 = ggml_reshape_3d(ctx, temb, n_embd, n_text_tokens, 1);
    ggml_tensor * t1 = ggml_reshape_3d(ctx, tpos, n_embd, n_text_tokens, 1);
    ggml_tensor * text = ggml_concat(ctx, t0, t1, 2);
    ggml_tensor * bos = ggml_add(ctx,
        ggml_get_rows(ctx, model.speech_emb, bos_tok),
        ggml_get_rows(ctx, model.speech_pos_emb, bos_pos));
    bos = repeat_batch(ctx, bos, n_embd, 1);
    ggml_tensor * inp = ggml_concat(ctx, cond, text, 1);
    inp = ggml_concat(ctx, inp, bos, 1);
    inp = ggml_concat(ctx, inp, bos, 1);
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
    ggml_tensor * speech_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(speech_pos, "speech_pos"); ggml_set_input(speech_pos);
    ggml_tensor * inp = ggml_add(ctx,
        ggml_get_rows(ctx, model.speech_emb, speech_token),
        ggml_get_rows(ctx, model.speech_pos_emb, speech_pos));
    inp = repeat_batch(ctx, inp, model.hparams.n_embd, 1);
    build_transformer_core(ctx, gf, model, inp, n_past, 1);
    ggml_free(ctx);
    return gf;
}
static void cfg_last_logits(ggml_tensor * logits, int N, int vocab, std::vector<float> & out) {
    std::vector<float> cond((size_t)vocab), uncond((size_t)vocab);
    ggml_backend_tensor_get(logits, cond.data(), (size_t)(N - 1) * logits->nb[1], (size_t)vocab * sizeof(float));
    ggml_backend_tensor_get(logits, uncond.data(), logits->nb[2] + (size_t)(N - 1) * logits->nb[1], (size_t)vocab * sizeof(float));
    out.resize((size_t)vocab);
    for (int i = 0; i < vocab; ++i)
        out[i] = cond[i] + CFG_WEIGHT * (cond[i] - uncond[i]);
}
void eval_prompt(
    const chatterbox_model & model, ggml_gallocr_t allocr,
    const std::vector<int32_t> & text_tokens, std::vector<float> & logits_out, int & prompt_len) {
    const int cond_len = 1 + model.hparams.perceiver_len + 1;
    prompt_len = cond_len + (int)text_tokens.size() + 2;
    if (prompt_len > model.hparams.n_ctx) throw std::runtime_error("T3 prompt exceeds context");
    ggml_cgraph * gf = build_prompt_graph(model, (int)text_tokens.size());
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "text_tokens"), text_tokens.data(), 0, text_tokens.size()*sizeof(int32_t));
    std::vector<int32_t> tpos((size_t)text_tokens.size());
    for (int i = 0; i < (int)text_tokens.size(); ++i) tpos[i] = i;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "text_pos"), tpos.data(), 0, tpos.size()*sizeof(int32_t));
    std::vector<int32_t> cpos((size_t)model.hparams.cond_prompt_len);
    for (int i = 0; i < model.hparams.cond_prompt_len; ++i) cpos[i] = i;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "cond_pos"), cpos.data(), 0, cpos.size()*sizeof(int32_t));
    int32_t bos = model.hparams.start_speech_token;
    int32_t bos_p = 0;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "bos_tok"), &bos, 0, sizeof(bos));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "bos_pos"), &bos_p, 0, sizeof(bos_p));
    std::vector<int32_t> pos((size_t)prompt_len);
    for (int i = 0; i < prompt_len; ++i) pos[i] = i;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "position"), pos.data(), 0, pos.size()*sizeof(int32_t));
    {
        const int N = prompt_len;
        ggml_tensor * kq_mask = ggml_graph_get_tensor(gf, "kq_mask");
        if (!kq_mask) throw std::runtime_error("T3 kq_mask missing");
        const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t ninf_h = ggml_fp32_to_fp16(-INFINITY);
        std::vector<ggml_fp16_t> mask((size_t)N * N, zero_h);
        for (int q = 0; q < N; ++q)
            for (int k = 0; k < N; ++k)
                if (k > q) mask[(size_t)q * N + k] = ninf_h;
        ggml_backend_tensor_set(kq_mask, mask.data(), 0, mask.size()*sizeof(ggml_fp16_t));
    }
    if (ggml_backend_graph_compute(model.backend, gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("T3 prompt failed");
    cfg_last_logits(ggml_graph_get_tensor(gf, "logits"), prompt_len, model.hparams.n_speech_vocab, logits_out);
}
void eval_step(
    const chatterbox_model & model, ggml_gallocr_t allocr,
    int n_past, int32_t token, int speech_pos, std::vector<float> & logits_out) {
    ggml_cgraph * gf = build_step_graph(model, n_past);
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "speech_token"), &token, 0, sizeof(token));
    int32_t sp = speech_pos;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "speech_pos"), &sp, 0, sizeof(sp));
    int32_t position = n_past;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "position"), &position, 0, sizeof(position));
    if (ggml_backend_graph_compute(model.backend, gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("T3 step failed");
    cfg_last_logits(ggml_graph_get_tensor(gf, "logits"), 1, model.hparams.n_speech_vocab, logits_out);
}
int32_t sample_next_token_ex(
    const std::vector<float> & logits,
    const std::vector<int32_t> & generated,
    std::mt19937 & rng) {
    const int n = (int)logits.size();
    std::vector<float> scores(logits.begin(), logits.end());
    apply_speech_repeat_penalty(scores.data(), n, generated);
    if (TEMPERATURE > 0.0f && TEMPERATURE != 1.0f) {
        float inv_t = 1.0f / TEMPERATURE;
        for (float & s : scores) s *= inv_t;
    }
    {
        float mx = -INFINITY;
        for (float s : scores) if (s != -INFINITY) mx = std::max(mx, s);
        std::vector<float> probs((size_t)n);
        float psum = 0;
        for (int i = 0; i < n; ++i) {
            probs[i] = (scores[i] == -INFINITY) ? 0.0f : std::exp(scores[i] - mx);
            psum += probs[i];
        }
        if (psum == 0.0f) throw std::runtime_error("sampler produced empty distribution");
        for (float & p : probs) p /= psum;
        float pmax = 0;
        for (float p : probs) pmax = std::max(pmax, p);
        const float limit = MIN_P * pmax;
        for (int i = 0; i < n; ++i) if (probs[i] < limit) scores[i] = -INFINITY;
    }
    if (TOP_K > 0 && TOP_K < n) {
        std::vector<float> tmp(scores);
        std::nth_element(tmp.begin(), tmp.begin() + TOP_K, tmp.end(), std::greater<float>());
        float threshold = tmp[TOP_K];
        int kept = 0;
        for (float s : scores) if (s > threshold) ++kept;
        if (kept < TOP_K) threshold -= 1e-10f;
        for (float & s : scores) if (s <= threshold) s = -INFINITY;
    }
    if (TOP_P < 1.0f) {
        struct IS { int idx; float s; };
        std::vector<IS> sorted;
        sorted.reserve(n);
        for (int i = 0; i < n; ++i) if (scores[i] != -INFINITY) sorted.push_back({i, scores[i]});
        std::sort(sorted.begin(), sorted.end(), [](const IS& a, const IS& b){ return a.s > b.s; });
        float mx = sorted[0].s;
        std::vector<float> probs(sorted.size());
        float psum = 0;
        for (size_t i = 0; i < sorted.size(); ++i) { probs[i] = std::exp(sorted[i].s - mx); psum += probs[i]; }
        for (float & p : probs) p /= psum;
        float cum = 0;
        std::set<int> keep_set;
        for (size_t i = 0; i < sorted.size(); ++i) {
            cum += probs[i];
            keep_set.insert(sorted[i].idx);
            if (cum >= TOP_P) break;
        }
        for (int i = 0; i < n; ++i) if (keep_set.find(i) == keep_set.end()) scores[i] = -INFINITY;
    }
    float mx = -INFINITY;
    for (float s : scores) if (s != -INFINITY) mx = std::max(mx, s);
    std::vector<float> probs((size_t)n);
    float psum = 0;
    for (int i = 0; i < n; ++i) {
        probs[i] = (scores[i] == -INFINITY) ? 0.0f : std::exp(scores[i] - mx);
        psum += probs[i];
    }
    if (psum == 0.0f) throw std::runtime_error("sampler produced empty distribution");
    for (float & p : probs) p /= psum;
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return (int32_t)dist(rng);
}

}
