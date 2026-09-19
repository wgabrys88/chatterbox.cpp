#include "gguf_weights.h"
#include "ggml-vulkan.h"
#include <set>
#include <stdexcept>
#include "chatterbox_t3_internal.h"
#if !defined(TTS_FAMILY_V3)
#error t3_v3.cpp is the Llama T3 backend; configure -DTTS_FAMILY=v3
#endif

using namespace tts_cpp::chatterbox::detail;
namespace tts_cpp::chatterbox::detail {

ggml_backend_t init_backend() { return ggml_backend_vk_init(0); }
void load_model_gguf(const std::string & path, chatterbox_model & model) {
    GgufWeights weights(path);
    auto* gguf_ctx = weights.file;
    auto & hp = model.hparams;
    hp.n_text_vocab       = (int32_t) gguf_get_val_u32(gguf_ctx, gguf_find_key(gguf_ctx, "chatterbox.text_vocab_size"));
    hp.n_speech_vocab     = (int32_t) gguf_get_val_u32(gguf_ctx, gguf_find_key(gguf_ctx, "chatterbox.speech_vocab_size"));
    hp.start_text_token   = (int32_t) gguf_get_val_u32(gguf_ctx, gguf_find_key(gguf_ctx, "chatterbox.start_text_token"));
    hp.stop_text_token    = (int32_t) gguf_get_val_u32(gguf_ctx, gguf_find_key(gguf_ctx, "chatterbox.stop_text_token"));
    hp.start_speech_token = (int32_t) gguf_get_val_u32(gguf_ctx, gguf_find_key(gguf_ctx, "chatterbox.start_speech_token"));
    hp.stop_speech_token  = (int32_t) gguf_get_val_u32(gguf_ctx, gguf_find_key(gguf_ctx, "chatterbox.stop_speech_token"));
    hp.cond_prompt_len    = (int32_t) gguf_get_val_u32(gguf_ctx, gguf_find_key(gguf_ctx, "chatterbox.cond_prompt_length"));
    hp.eps                = gguf_get_val_f32(gguf_ctx, gguf_find_key(gguf_ctx, "chatterbox.layer_norm_eps"));
    hp.n_embd  = (int32_t) gguf_get_val_u32(gguf_ctx, gguf_find_key(gguf_ctx, "chatterbox.n_embd"));
    hp.n_head  = (int32_t) gguf_get_val_u32(gguf_ctx, gguf_find_key(gguf_ctx, "chatterbox.n_head"));
    hp.n_layer = (int32_t) gguf_get_val_u32(gguf_ctx, gguf_find_key(gguf_ctx, "chatterbox.n_layer"));
    hp.n_ff    = (int32_t) gguf_get_val_u32(gguf_ctx, gguf_find_key(gguf_ctx, "chatterbox.n_ff"));
    hp.n_ctx   = (int32_t) gguf_get_val_u32(gguf_ctx, gguf_find_key(gguf_ctx, "chatterbox.n_ctx"));
    hp.perceiver_len = (int32_t) gguf_get_val_u32(gguf_ctx, gguf_find_key(gguf_ctx, "chatterbox.perceiver_len"));
    hp.rope_theta    = gguf_get_val_f32(gguf_ctx, gguf_find_key(gguf_ctx, "chatterbox.rope_theta"));
    hp.rope_orig_ctx = (int32_t) gguf_get_val_u32(gguf_ctx, gguf_find_key(gguf_ctx, "chatterbox.rope_orig_ctx"));
    model.language_tokens = gguf_get_val_str(gguf_ctx, gguf_find_key(gguf_ctx, "chatterbox.tokenizer.language_tokens"));
    weights.upload(model);
    model.rms_out          = model.tensors["model/norm/g"];
    model.text_emb         = model.tensors["chatterbox/text_emb"];
    model.speech_emb       = model.tensors["chatterbox/speech_emb"];
    model.speech_head      = model.tensors["chatterbox/speech_head"];
    model.text_pos_emb     = model.tensors["chatterbox/text_pos_emb"];
    model.speech_pos_emb   = model.tensors["chatterbox/speech_pos_emb"];
    model.cond_spkr_w      = model.tensors["chatterbox/cond_spkr/w"];
    model.cond_spkr_b      = model.tensors["chatterbox/cond_spkr/b"];
    model.emotion_adv_fc_w = model.tensors["chatterbox/emotion_adv_fc/w"];
    model.builtin_speaker_emb        = model.tensors["chatterbox/builtin/speaker_emb"];
    model.builtin_cond_prompt_tokens = model.tensors["chatterbox/builtin/cond_prompt_speech_tokens"];
    model.rope_freq_factors          = model.tensors["model/rope_freq_factors"];
    model.perceiver.query   = model.tensors["chatterbox/perceiver/pre_attention_query"];
    model.perceiver.norm_g  = model.tensors["chatterbox/perceiver/attn/norm/g"];
    model.perceiver.norm_b  = model.tensors["chatterbox/perceiver/attn/norm/b"];
    model.perceiver.to_q_w  = model.tensors["chatterbox/perceiver/attn/to_q/w"];
    model.perceiver.to_q_b  = model.tensors["chatterbox/perceiver/attn/to_q/b"];
    model.perceiver.to_k_w  = model.tensors["chatterbox/perceiver/attn/to_k/w"];
    model.perceiver.to_k_b  = model.tensors["chatterbox/perceiver/attn/to_k/b"];
    model.perceiver.to_v_w  = model.tensors["chatterbox/perceiver/attn/to_v/w"];
    model.perceiver.to_v_b  = model.tensors["chatterbox/perceiver/attn/to_v/b"];
    model.perceiver.proj_w  = model.tensors["chatterbox/perceiver/attn/proj_out/w"];
    model.perceiver.proj_b  = model.tensors["chatterbox/perceiver/attn/proj_out/b"];
    hp.cond_prompt_len = (int32_t) ggml_nelements(model.builtin_cond_prompt_tokens);
    model.layers.resize(hp.n_layer);
    for (int i = 0; i < hp.n_layer; ++i) {
        auto & l = model.layers[i];
        std::string p = "model/h" + std::to_string(i);
        l.attn_norm = model.tensors[(p + "/attn_norm/g").c_str()];
        l.ffn_norm  = model.tensors[(p + "/ffn_norm/g").c_str()];
        l.wq = model.tensors[(p + "/attn/q/w").c_str()];
        l.wk = model.tensors[(p + "/attn/k/w").c_str()];
        l.wv = model.tensors[(p + "/attn/v/w").c_str()];
        l.wo = model.tensors[(p + "/attn/o/w").c_str()];
        l.gate = model.tensors[(p + "/ffn/gate/w").c_str()];
        l.up   = model.tensors[(p + "/ffn/up/w").c_str()];
        l.down = model.tensors[(p + "/ffn/down/w").c_str()];
    }

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
    q = ggml_reshape_3d(ctx, q, HD, n_head, n_q);
    k = ggml_reshape_3d(ctx, k, HD, n_head, n_kv);
    v = ggml_reshape_3d(ctx, v, HD, n_head, n_kv);
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
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
    const int n_embd = hp.n_embd, n_head = hp.n_head, n_layer = hp.n_layer, n_ctx = model.kv_rows;
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
    ggml_tensor * emotion_adv = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(emotion_adv, "emotion_adv"); ggml_set_input(emotion_adv);
    ggml_tensor * spkr = ggml_add(ctx, ggml_mul_mat(ctx, model.cond_spkr_w, model.builtin_speaker_emb), model.cond_spkr_b);
    ggml_tensor * h = ggml_add(ctx,
        ggml_get_rows(ctx, model.speech_emb, model.builtin_cond_prompt_tokens),
        ggml_get_rows(ctx, model.speech_pos_emb, cond_pos));
    ggml_tensor * perc = run_perceiver(ctx, model, h);
    ggml_tensor * emo = ggml_mul_mat(ctx, model.emotion_adv_fc_w, emotion_adv);
    ggml_tensor * cond = ggml_concat(ctx, spkr, perc, 1);
    cond = ggml_concat(ctx, cond, emo, 1);
    cond = repeat_batch(ctx, cond, n_embd, cond_len);
    ggml_tensor * tpos = ggml_get_rows(ctx, model.text_pos_emb, text_pos);
    ggml_tensor * temb = ggml_add(ctx, ggml_get_rows(ctx, model.text_emb, text_tokens), tpos);

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
    const float w = runtime_knobs().cfg_weight;
    for (int i = 0; i < vocab; ++i)
        out[i] = cond[i] + w * (cond[i] - uncond[i]);
}
void eval_prompt(
    chatterbox_model & model, ggml_gallocr_t allocr,
    const std::vector<int32_t> & text_tokens, std::vector<float> & logits_out, int & prompt_len) {
    const int cond_len = 1 + model.hparams.perceiver_len + 1;
    prompt_len = cond_len + (int)text_tokens.size() + 2;
    prepare_kv(model, prompt_len, 2);
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
    const float exaggeration = runtime_knobs().exaggeration;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "emotion_adv"), &exaggeration, 0, sizeof(exaggeration));
    std::vector<int32_t> pos((size_t)prompt_len);
    for (int i = 0; i < prompt_len; ++i) pos[i] = i;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "position"), pos.data(), 0, pos.size()*sizeof(int32_t));
    set_causal_mask(gf, prompt_len);
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
    const float temperature = runtime_knobs().temperature;
    const float min_p = runtime_knobs().min_p;
    const float top_p = runtime_knobs().top_p;
    std::vector<float> scores(logits.begin(), logits.end());
    apply_speech_repeat_penalty(scores.data(), n, generated);
    if (temperature > 0.0f && temperature != 1.0f) {
        float inv_t = 1.0f / temperature;
        for (float & s : scores) s *= inv_t;
    }
    {
        auto probs = probabilities(scores);
        float pmax = 0;
        for (float p : probs) pmax = std::max(pmax, p);
        const float limit = min_p * pmax;
        for (int i = 0; i < n; ++i) if (probs[i] < limit) scores[i] = -INFINITY;
    }
    if (top_p < 1.0f) {
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
            if (cum >= top_p) break;
        }
        for (int i = 0; i < n; ++i) if (keep_set.find(i) == keep_set.end()) scores[i] = -INFINITY;
    }
    auto probs = probabilities(scores);
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    int32_t chosen = (int32_t)dist(rng);
    return chosen;
}

}
