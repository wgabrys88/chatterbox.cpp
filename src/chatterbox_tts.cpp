#include "tts-cpp/chatterbox/log.h"
#include "chatterbox_t3_internal.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "ggml-vulkan.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
static int g_n_threads = 1;
static double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}
static void check_cancel(const std::atomic<bool>* cancel) {
    if (cancel && cancel->load(std::memory_order_relaxed)) throw std::runtime_error("synthesis cancelled");
}
static void compute(ggml_backend_t backend, ggml_cgraph * gf) {
    ggml_backend_graph_compute(backend, gf);
}
struct model_ctx {
    ggml_backend_t backend = nullptr;
    ggml_context * ctx_w = nullptr;
    ggml_backend_buffer_t buffer_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
    bool meanflow = true;
    int n_timesteps = 2;
    float cfg_rate = 0.0f;
};
static ggml_backend_t s3gen_init_backend(int n_gpu_layers) {
    if (n_gpu_layers <= 0) throw std::runtime_error("Vulkan GPU layers required");
    auto * b = ggml_backend_vk_init(0);
    if (!b) throw std::runtime_error("Vulkan S3Gen backend init failed");
    char desc[256] = {0};
    ggml_backend_vk_get_device_description(0, desc, sizeof(desc));
    tts_emit("tts.backend", ",\"role\":\"s3gen\",\"backend\":\"Vulkan\",\"gpu_layers\":" + std::to_string(n_gpu_layers) + ",\"device\":" + tts_json_escape(desc));
    return b;
}
static model_ctx load_s3gen_gguf(const std::string&, int, bool);
namespace {
struct s3gen_cache_entry { std::string path; int gpu = 0; bool fastconv = false; std::unique_ptr<model_ctx> m; };
static std::mutex                            g_s3gen_cache_mu;
static std::unique_ptr<s3gen_cache_entry>    g_s3gen_cache_entry;
static double                                g_s3gen_cache_last_load_ms = 0.0;
}
static void s3gen_model_cache_release() {
    std::lock_guard<std::mutex> lk(g_s3gen_cache_mu);
    if (!g_s3gen_cache_entry) return;
    model_ctx * m = g_s3gen_cache_entry->m.get();
    if (m) {
        if (m->buffer_w) { ggml_backend_buffer_free(m->buffer_w); m->buffer_w = nullptr; }
        if (m->ctx_w)    { ggml_free(m->ctx_w);                   m->ctx_w    = nullptr; }
        if (m->backend)  { ggml_backend_free(m->backend);         m->backend  = nullptr; }
        m->tensors.clear();
    }
    g_s3gen_cache_entry.reset();
}
static model_ctx * s3gen_model_cache_get(const std::string& path, int n_gpu_layers, bool fastconv) {
    std::lock_guard<std::mutex> lk(g_s3gen_cache_mu);
    if (g_s3gen_cache_entry &&
        g_s3gen_cache_entry->path == path &&
        g_s3gen_cache_entry->gpu  == n_gpu_layers &&
        g_s3gen_cache_entry->fastconv == fastconv) {
        g_s3gen_cache_last_load_ms = 0.0;
        return g_s3gen_cache_entry->m.get();
    }
    double t0 = now_ms();
    auto m = std::make_unique<model_ctx>(load_s3gen_gguf(path, n_gpu_layers, fastconv));
    g_s3gen_cache_last_load_ms = now_ms() - t0;
    g_s3gen_cache_entry = std::make_unique<s3gen_cache_entry>(
        s3gen_cache_entry{path, n_gpu_layers, fastconv, std::move(m)});
    static bool registered = false;
    if (!registered) {
        std::atexit(s3gen_model_cache_release);
        registered = true;
    }
    return g_s3gen_cache_entry->m.get();
}
static double s3gen_model_cache_last_load_ms() { return g_s3gen_cache_last_load_ms; }
static model_ctx load_s3gen_gguf(const std::string& path, int n_gpu_layers, bool fastconv) {
    model_ctx m;
    ggml_context * tmp_ctx = nullptr;
    gguf_init_params gp = {  false,  &tmp_ctx };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) throw std::runtime_error("gguf_init_from_file failed: " + path);
    m.backend = s3gen_init_backend(n_gpu_layers);
    int64_t n_tensors = gguf_get_n_tensors(g);
    ggml_init_params p = { ggml_tensor_overhead() * (size_t)n_tensors, nullptr, true };
    m.ctx_w = ggml_init(p);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(g, i);
        ggml_tensor * src = ggml_get_tensor(tmp_ctx, name);
        ggml_tensor * dst = fastconv && src->type == GGML_TYPE_F16 && ggml_is_3d(src)
            ? ggml_new_tensor(m.ctx_w, GGML_TYPE_F32, ggml_n_dims(src), src->ne)
            : ggml_dup_tensor(m.ctx_w, src);
        ggml_set_name(dst, name);
        m.tensors[name] = dst;
    }
    m.buffer_w = ggml_backend_alloc_ctx_tensors(m.ctx_w, m.backend);
    size_t baked = 0, baked_bytes = 0;
    for (ggml_tensor * cur = ggml_get_first_tensor(m.ctx_w); cur; cur = ggml_get_next_tensor(m.ctx_w, cur)) {
        ggml_tensor * src = ggml_get_tensor(tmp_ctx, ggml_get_name(cur));
        if (cur->type == GGML_TYPE_F32 && src->type == GGML_TYPE_F16 && ggml_is_3d(src)) {
            std::vector<float> f32((size_t) ggml_nelements(src));
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) ggml_get_data(src), f32.data(), ggml_nelements(src));
            ggml_backend_tensor_set(cur, f32.data(), 0, f32.size() * sizeof(float));
            ++baked;
            baked_bytes += f32.size() * sizeof(float);
        } else {
            ggml_backend_tensor_set(cur, ggml_get_data(src), 0, ggml_nbytes(src));
        }
    }
    int64_t k = gguf_find_key(g, "s3gen.meanflow");
    if (k >= 0) m.meanflow = gguf_get_val_bool(g, k);
    k = gguf_find_key(g, "s3gen.n_timesteps");
    m.n_timesteps = k >= 0 ? (int)gguf_get_val_u32(g, k) : (m.meanflow ? 2 : 10);
    k = gguf_find_key(g, "s3gen.cfg_rate");
    m.cfg_rate = k >= 0 ? gguf_get_val_f32(g, k) : (m.meanflow ? 0.0f : 0.7f);
    gguf_free(g);
    ggml_free(tmp_ctx);
    return m;
}
static ggml_tensor * find_tensor(const model_ctx & m, const std::string & name) {
    auto it = m.tensors.find(name);
    if (it == m.tensors.end()) throw std::runtime_error("tensor not found: " + name);
    return it->second;
}
static ggml_tensor * conv1d_f32(ggml_context * ctx, ggml_tensor * kernel, ggml_tensor * input,
                                int stride, int padding, int dilation) {
    ggml_tensor * im2col = ggml_im2col(ctx, kernel, input, stride, 0, padding, 0, dilation, 0, false, GGML_TYPE_F32);
    ggml_tensor * result = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
        ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]));
    return ggml_reshape_3d(ctx, result, im2col->ne[1], kernel->ne[2], im2col->ne[2]);
}
static ggml_tensor * conv1d_f32_b(ggml_context * ctx, ggml_tensor * kernel, ggml_tensor * input,
                                  int stride, int padding, int dilation) {
    ggml_tensor * im2col = ggml_im2col(ctx, kernel, input, stride, 0, padding, 0, dilation, 0, false, GGML_TYPE_F32);
    ggml_tensor * k_flat = ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]);
    ggml_tensor * prod = ggml_mul_mat(ctx, k_flat, im2col);
    return ggml_cont(ctx, ggml_permute(ctx, prod, 1, 0, 2, 3));
}
static ggml_tensor * conv_transpose_1d_f32(ggml_context * ctx, ggml_tensor * kernel,
                                           ggml_tensor * input, int stride, int padding) {
    ggml_tensor * out = ggml_conv_transpose_1d(ctx, kernel, input, stride, 0, 1);
    if (padding == 0) return out;
    int64_t L_new = out->ne[0] - 2 * padding;
    ggml_tensor * v = ggml_view_3d(ctx, out, L_new, out->ne[1], out->ne[2],
                                   out->nb[1], out->nb[2], (size_t)padding * out->nb[0]);
    return ggml_cont(ctx, v);
}
static ggml_tensor * zero_pad_dim0(ggml_context * ctx, ggml_tensor * x, int p_front, int p_back) {
    if (p_front <= 0 && p_back <= 0) return x;
    ggml_tensor * y = x;
    if (p_front > 0) {
        GGML_ASSERT(p_front <= (int)x->ne[0]);
        ggml_tensor * head = ggml_view_4d(ctx, x, p_front, x->ne[1], x->ne[2], x->ne[3],
                                           x->nb[1], x->nb[2], x->nb[3], 0);
        ggml_tensor * z = ggml_scale(ctx, ggml_cont(ctx, head), 0.0f);
        y = ggml_concat(ctx, z, y, 0);
    }
    if (p_back > 0) {
        GGML_ASSERT(p_back <= (int)x->ne[0]);
        ggml_tensor * tail = ggml_view_4d(ctx, x, p_back, x->ne[1], x->ne[2], x->ne[3],
                                           x->nb[1], x->nb[2], x->nb[3],
                                           (size_t)(x->ne[0] - p_back) * x->nb[0]);
        ggml_tensor * z = ggml_scale(ctx, ggml_cont(ctx, tail), 0.0f);
        y = ggml_concat(ctx, y, z, 0);
    }
    return y;
}
static ggml_tensor * ggml_mish_fn(ggml_context * ctx, ggml_tensor * x) {
    ggml_tensor * sp = ggml_unary(ctx, x, GGML_UNARY_OP_SOFTPLUS);
    ggml_tensor * th = ggml_unary(ctx, sp, GGML_UNARY_OP_TANH);
    return ggml_mul(ctx, x, th);
}
static ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x,
                                ggml_tensor * w, ggml_tensor * b, float eps = 1e-5f) {
    ggml_tensor * y = ggml_norm(ctx, x, eps);
    y = ggml_mul(ctx, y, w);
    return ggml_add(ctx, y, b);
}
static ggml_tensor * layer_norm_on_channel(ggml_context * ctx, ggml_tensor * x,
                                           ggml_tensor * w, ggml_tensor * b, float eps = 1e-5f) {
    ggml_tensor * xt = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    xt = ggml_norm(ctx, xt, eps);
    xt = ggml_mul(ctx, xt, w);
    xt = ggml_add(ctx, xt, b);
    return ggml_cont(ctx, ggml_permute(ctx, xt, 1, 0, 2, 3));
}
static ggml_tensor * reflect_pad_1d(ggml_context * ctx, ggml_tensor * x, int p_left, int p_right) {
    ggml_tensor * y = x;
    for (int i = 0; i < p_left; ++i) {
        int src_idx = p_left - i;
        ggml_tensor * s = ggml_view_3d(ctx, x, 1, x->ne[1], x->ne[2], x->nb[1], x->nb[2], (size_t)src_idx * x->nb[0]);
        s = ggml_cont(ctx, s);
        y = ggml_concat(ctx, s, y, 0);
    }
    int L_orig = (int)x->ne[0];
    for (int i = 0; i < p_right; ++i) {
        int src_idx = L_orig - 2 - i;
        ggml_tensor * s = ggml_view_3d(ctx, x, 1, x->ne[1], x->ne[2], x->nb[1], x->nb[2], (size_t)src_idx * x->nb[0]);
        s = ggml_cont(ctx, s);
        y = ggml_concat(ctx, y, s, 0);
    }
    return y;
}
struct conformer_w {
    ggml_tensor *norm_mha_w, *norm_mha_b, *norm_ff_w, *norm_ff_b;
    ggml_tensor *q_w, *q_b, *k_w, *k_b, *v_w, *v_b, *o_w, *o_b;
    ggml_tensor *pos_w, *pos_bias_u, *pos_bias_v;
    ggml_tensor *ff1_w, *ff1_b, *ff2_w, *ff2_b;
};
static conformer_w load_conformer(const model_ctx & m, const std::string & pfx) {
    conformer_w w;
    w.norm_mha_w = find_tensor(m, pfx + "/norm_mha/w");
    w.norm_mha_b = find_tensor(m, pfx + "/norm_mha/b");
    w.norm_ff_w  = find_tensor(m, pfx + "/norm_ff/w");
    w.norm_ff_b  = find_tensor(m, pfx + "/norm_ff/b");
    w.q_w = find_tensor(m, pfx + "/attn/q/w"); w.q_b = find_tensor(m, pfx + "/attn/q/b");
    w.k_w = find_tensor(m, pfx + "/attn/k/w"); w.k_b = find_tensor(m, pfx + "/attn/k/b");
    w.v_w = find_tensor(m, pfx + "/attn/v/w"); w.v_b = find_tensor(m, pfx + "/attn/v/b");
    w.o_w = find_tensor(m, pfx + "/attn/o/w"); w.o_b = find_tensor(m, pfx + "/attn/o/b");
    w.pos_w = find_tensor(m, pfx + "/attn/pos/w");
    w.pos_bias_u = find_tensor(m, pfx + "/attn/pos_bias_u");
    w.pos_bias_v = find_tensor(m, pfx + "/attn/pos_bias_v");
    w.ff1_w = find_tensor(m, pfx + "/ff/w1/w"); w.ff1_b = find_tensor(m, pfx + "/ff/w1/b");
    w.ff2_w = find_tensor(m, pfx + "/ff/w2/w"); w.ff2_b = find_tensor(m, pfx + "/ff/w2/b");
    return w;
}
static ggml_tensor * conformer_block(ggml_context * ctx, const conformer_w & w,
                                     ggml_tensor * x, ggml_tensor * pos_emb,
                                     int D, int T, int H, int HD, float eps = 1e-12f) {
    ggml_tensor * residual = x;
    ggml_tensor * xn = ggml_norm(ctx, x, eps);
    xn = ggml_add(ctx, ggml_mul(ctx, xn, w.norm_mha_w), w.norm_mha_b);
    ggml_tensor * q = ggml_add(ctx, ggml_mul_mat(ctx, w.q_w, xn), w.q_b);
    ggml_tensor * k = ggml_add(ctx, ggml_mul_mat(ctx, w.k_w, xn), w.k_b);
    ggml_tensor * v = ggml_add(ctx, ggml_mul_mat(ctx, w.v_w, xn), w.v_b);
    ggml_tensor * p = ggml_mul_mat(ctx, w.pos_w, pos_emb);
    q = ggml_reshape_3d(ctx, q, HD, H, T);
    k = ggml_reshape_3d(ctx, k, HD, H, T);
    v = ggml_reshape_3d(ctx, v, HD, H, T);
    p = ggml_reshape_3d(ctx, p, HD, H, pos_emb->ne[1]);
    ggml_tensor * q_perm = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    ggml_tensor * k_perm = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    ggml_tensor * v_perm = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
    ggml_tensor * p_perm = ggml_cont(ctx, ggml_permute(ctx, p, 0, 2, 1, 3));
    ggml_tensor * u_bias = ggml_reshape_3d(ctx, w.pos_bias_u, HD, 1, H);
    ggml_tensor * v_bias = ggml_reshape_3d(ctx, w.pos_bias_v, HD, 1, H);
    ggml_tensor * q_plus_u = ggml_add(ctx, q_perm, u_bias);
    ggml_tensor * q_plus_v = ggml_add(ctx, q_perm, v_bias);
    ggml_tensor * ac = ggml_mul_mat(ctx, k_perm, q_plus_u);
    ggml_tensor * bd = ggml_mul_mat(ctx, p_perm, q_plus_v);
    ggml_tensor * bd_padded = zero_pad_dim0(ctx, bd, 1, 0);
    ggml_tensor * bd_viewed = ggml_reshape_3d(ctx, bd_padded, T, 2*T, H);
    ggml_tensor * bd_sliced = ggml_view_3d(ctx, bd_viewed, T, 2*T - 1, H,
                                           bd_viewed->nb[1], bd_viewed->nb[2], bd_viewed->nb[1]);
    ggml_tensor * bd_reshaped = ggml_reshape_3d(ctx, ggml_cont(ctx, bd_sliced), 2*T - 1, T, H);
    ggml_tensor * bd_final = ggml_view_3d(ctx, bd_reshaped, T, T, H,
                                          bd_reshaped->nb[1], bd_reshaped->nb[2], 0);
    bd_final = ggml_cont(ctx, bd_final);
    ggml_tensor * scores = ggml_add(ctx, ac, bd_final);
    scores = ggml_scale(ctx, scores, 1.0f / std::sqrt((float)HD));
    ggml_tensor * attn = ggml_soft_max(ctx, scores);
    ggml_tensor * v_for_mm = ggml_cont(ctx, ggml_permute(ctx, v_perm, 1, 0, 2, 3));
    ggml_tensor * attn_v = ggml_mul_mat(ctx, v_for_mm, attn);
    ggml_tensor * merged = ggml_cont(ctx, ggml_permute(ctx, attn_v, 0, 2, 1, 3));
    ggml_tensor * flat = ggml_reshape_2d(ctx, merged, HD * H, T);
    ggml_tensor * attn_out = ggml_add(ctx, ggml_mul_mat(ctx, w.o_w, flat), w.o_b);
    x = ggml_add(ctx, residual, attn_out);
    residual = x;
    xn = ggml_norm(ctx, x, eps);
    xn = ggml_add(ctx, ggml_mul(ctx, xn, w.norm_ff_w), w.norm_ff_b);
    ggml_tensor * ff = ggml_add(ctx, ggml_mul_mat(ctx, w.ff1_w, xn), w.ff1_b);
    ff = ggml_silu(ctx, ff);
    ff = ggml_add(ctx, ggml_mul_mat(ctx, w.ff2_w, ff), w.ff2_b);
    return ggml_add(ctx, residual, ff);
}
static void compute_pos_emb(std::vector<float> & pe, int T, int D) {
    int L = 2 * T - 1;
    pe.assign(L * D, 0.0f);
    const float log10000 = std::log(10000.0f);
    std::vector<float> div_term(D / 2);
    for (int i = 0; i < D / 2; ++i) div_term[i] = std::exp(-((float)(2*i) * log10000 / (float)D));
    std::vector<std::vector<float>> pos_pe(T, std::vector<float>(D, 0.0f));
    std::vector<std::vector<float>> neg_pe(T, std::vector<float>(D, 0.0f));
    for (int i = 0; i < T; ++i) {
        for (int k = 0; k < D / 2; ++k) {
            pos_pe[i][2*k]     = std::sin((float)i * div_term[k]);
            pos_pe[i][2*k + 1] = std::cos((float)i * div_term[k]);
            neg_pe[i][2*k]     = std::sin(-(float)i * div_term[k]);
            neg_pe[i][2*k + 1] = std::cos(-(float)i * div_term[k]);
        }
    }
    for (int t = 0; t < T; ++t) {
        int src = T - 1 - t;
        for (int d = 0; d < D; ++d) pe[t*D + d] = pos_pe[src][d];
    }
    for (int t = 1; t < T; ++t) {
        for (int d = 0; d < D; ++d) pe[(T - 1 + t)*D + d] = neg_pe[t][d];
    }
}
static std::vector<float> run_encoder(const model_ctx & m, const std::vector<float> & input_embed, int T, int D = 512) {
    const int H = 8, HEAD_DIM = 64;
    const int T2 = 2 * T;
    static size_t buf_size = 64 * 1024 * 1024;
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params gp = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 32768, false);
    ggml_tensor * x_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, T);
    ggml_set_name(x_in, "x_in"); ggml_set_input(x_in);
    ggml_tensor * pos1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, 2*T - 1);
    ggml_set_name(pos1, "pos1"); ggml_set_input(pos1);
    ggml_tensor * pos2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, 2*T2 - 1);
    ggml_set_name(pos2, "pos2"); ggml_set_input(pos2);
    ggml_tensor * elw = find_tensor(m, "flow/encoder/embed/linear/w");
    ggml_tensor * elb = find_tensor(m, "flow/encoder/embed/linear/b");
    ggml_tensor * enw = find_tensor(m, "flow/encoder/embed/norm/w");
    ggml_tensor * enb = find_tensor(m, "flow/encoder/embed/norm/b");
    ggml_tensor * x = ggml_add(ctx, ggml_mul_mat(ctx, elw, x_in), elb);
    x = ggml_norm(ctx, x, 1e-5f);
    x = ggml_add(ctx, ggml_mul(ctx, x, enw), enb);
    x = ggml_scale(ctx, x, std::sqrt((float)D));
    ggml_tensor * residual = x;
    ggml_tensor * xt = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    ggml_tensor * pw1 = find_tensor(m, "flow/encoder/pre_lookahead/conv1/w");
    ggml_tensor * pb1 = find_tensor(m, "flow/encoder/pre_lookahead/conv1/b");
    ggml_tensor * pw2 = find_tensor(m, "flow/encoder/pre_lookahead/conv2/w");
    ggml_tensor * pb2 = find_tensor(m, "flow/encoder/pre_lookahead/conv2/b");
    xt = zero_pad_dim0(ctx, xt, 0, 3);
    xt = conv1d_f32(ctx, pw1, xt, 1, 0, 1);
    xt = ggml_add(ctx, xt, ggml_reshape_2d(ctx, pb1, 1, D));
    xt = ggml_leaky_relu(ctx, xt, 0.01f, false);
    xt = zero_pad_dim0(ctx, xt, 2, 0);
    xt = conv1d_f32(ctx, pw2, xt, 1, 0, 1);
    xt = ggml_add(ctx, xt, ggml_reshape_2d(ctx, pb2, 1, D));
    xt = ggml_cont(ctx, ggml_permute(ctx, xt, 1, 0, 2, 3));
    x = ggml_add(ctx, xt, residual);
    for (int i = 0; i < 6; ++i) {
        auto w = load_conformer(m, "flow/encoder/block" + std::to_string(i));
        x = conformer_block(ctx, w, x, pos1, D, T, H, HEAD_DIM);
    }
    ggml_tensor * up_w = find_tensor(m, "flow/encoder/up_layer/conv/w");
    ggml_tensor * up_b = find_tensor(m, "flow/encoder/up_layer/conv/b");
    ggml_tensor * xu = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    ggml_tensor * xu_3d = ggml_reshape_3d(ctx, xu, 1, xu->ne[0], xu->ne[1]);
    ggml_tensor * xu_2x = ggml_concat(ctx, xu_3d, xu_3d, 0);
    xu = ggml_cont(ctx, ggml_reshape_2d(ctx, xu_2x, xu_3d->ne[1]*2, xu_3d->ne[2]));
    xu = zero_pad_dim0(ctx, xu, 4, 0);
    xu = conv1d_f32(ctx, up_w, xu, 1, 0, 1);
    xu = ggml_add(ctx, xu, ggml_reshape_2d(ctx, up_b, 1, D));
    x = ggml_cont(ctx, ggml_permute(ctx, xu, 1, 0, 2, 3));
    ggml_tensor * ulw = find_tensor(m, "flow/encoder/up_embed/linear/w");
    ggml_tensor * ulb = find_tensor(m, "flow/encoder/up_embed/linear/b");
    ggml_tensor * unw = find_tensor(m, "flow/encoder/up_embed/norm/w");
    ggml_tensor * unb = find_tensor(m, "flow/encoder/up_embed/norm/b");
    x = ggml_add(ctx, ggml_mul_mat(ctx, ulw, x), ulb);
    x = ggml_norm(ctx, x, 1e-5f);
    x = ggml_add(ctx, ggml_mul(ctx, x, unw), unb);
    x = ggml_scale(ctx, x, std::sqrt((float)D));
    for (int i = 0; i < 4; ++i) {
        auto w = load_conformer(m, "flow/encoder/up_block" + std::to_string(i));
        x = conformer_block(ctx, w, x, pos2, D, T2, H, HEAD_DIM);
    }
    ggml_tensor * anw = find_tensor(m, "flow/encoder/after_norm/w");
    ggml_tensor * anb = find_tensor(m, "flow/encoder/after_norm/b");
    x = ggml_norm(ctx, x, 1e-5f);
    x = ggml_add(ctx, ggml_mul(ctx, x, anw), anb);
    ggml_tensor * epw = find_tensor(m, "flow/encoder_proj/w");
    ggml_tensor * epb = find_tensor(m, "flow/encoder_proj/b");
    ggml_tensor * mu = ggml_add(ctx, ggml_mul_mat(ctx, epw, x), epb);
    ggml_set_name(mu, "mu"); ggml_set_output(mu);
    ggml_build_forward_expand(gf, mu);
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x_in"), input_embed.data(), 0, input_embed.size()*sizeof(float));
    std::vector<float> pe1, pe2;
    compute_pos_emb(pe1, T, D);
    compute_pos_emb(pe2, T2, D);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos1"), pe1.data(), 0, pe1.size()*sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos2"), pe2.data(), 0, pe2.size()*sizeof(float));
    compute(m.backend, gf);
    std::vector<float> mu_data(ggml_nelements(mu));
    ggml_backend_tensor_get(mu, mu_data.data(), 0, ggml_nbytes(mu));
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    return mu_data;
}
struct cfm_resnet_w {
    ggml_tensor *b1_conv_w, *b1_conv_b, *b1_ln_w, *b1_ln_b;
    ggml_tensor *b2_conv_w, *b2_conv_b, *b2_ln_w, *b2_ln_b;
    ggml_tensor *mlp_w, *mlp_b, *res_w, *res_b;
};
static cfm_resnet_w load_cfm_resnet(const model_ctx & m, const std::string & pfx) {
    cfm_resnet_w w;
    w.b1_conv_w = find_tensor(m, pfx + "/block1/block/0/weight");
    w.b1_conv_b = find_tensor(m, pfx + "/block1/block/0/bias");
    w.b1_ln_w   = find_tensor(m, pfx + "/block1/block/2/weight");
    w.b1_ln_b   = find_tensor(m, pfx + "/block1/block/2/bias");
    w.b2_conv_w = find_tensor(m, pfx + "/block2/block/0/weight");
    w.b2_conv_b = find_tensor(m, pfx + "/block2/block/0/bias");
    w.b2_ln_w   = find_tensor(m, pfx + "/block2/block/2/weight");
    w.b2_ln_b   = find_tensor(m, pfx + "/block2/block/2/bias");
    w.mlp_w     = find_tensor(m, pfx + "/mlp/1/weight");
    w.mlp_b     = find_tensor(m, pfx + "/mlp/1/bias");
    w.res_w     = find_tensor(m, pfx + "/res_conv/weight");
    w.res_b     = find_tensor(m, pfx + "/res_conv/bias");
    return w;
}
static ggml_tensor * cfm_causal_block(ggml_context * ctx, ggml_tensor * x,
                                      ggml_tensor * conv_w, ggml_tensor * conv_b,
                                      ggml_tensor * ln_w, ggml_tensor * ln_b, int64_t C_out) {
    ggml_tensor * xp = zero_pad_dim0(ctx, x, 2, 0);
    ggml_tensor * y = conv1d_f32(ctx, conv_w, xp, 1, 0, 1);
    y = ggml_add(ctx, y, ggml_reshape_2d(ctx, conv_b, 1, C_out));
    y = layer_norm_on_channel(ctx, y, ln_w, ln_b);
    return ggml_mish_fn(ctx, y);
}
static ggml_tensor * cfm_resnet(ggml_context * ctx, const cfm_resnet_w & w,
                                ggml_tensor * x, ggml_tensor * t_emb, int64_t C_out) {
    ggml_tensor * h = cfm_causal_block(ctx, x, w.b1_conv_w, w.b1_conv_b, w.b1_ln_w, w.b1_ln_b, C_out);
    ggml_tensor * t_feat = ggml_mish_fn(ctx, t_emb);
    ggml_tensor * t_proj = ggml_add(ctx, ggml_mul_mat(ctx, w.mlp_w, t_feat), w.mlp_b);
    h = ggml_add(ctx, h, ggml_reshape_2d(ctx, t_proj, 1, C_out));
    h = cfm_causal_block(ctx, h, w.b2_conv_w, w.b2_conv_b, w.b2_ln_w, w.b2_ln_b, C_out);
    ggml_tensor * res = conv1d_f32(ctx, w.res_w, x, 1, 0, 1);
    res = ggml_add(ctx, res, ggml_reshape_2d(ctx, w.res_b, 1, C_out));
    return ggml_add(ctx, h, res);
}
struct basic_tfm_w {
    ggml_tensor *norm1_w, *norm1_b;
    ggml_tensor *to_q, *to_k, *to_v;
    ggml_tensor *to_out_w, *to_out_b;
    ggml_tensor *norm3_w, *norm3_b;
    ggml_tensor *ff0_w, *ff0_b, *ff2_w, *ff2_b;
};
static basic_tfm_w load_basic_tfm(const model_ctx & m, const std::string & pfx) {
    basic_tfm_w w;
    w.norm1_w = find_tensor(m, pfx + "/norm1/weight");
    w.norm1_b = find_tensor(m, pfx + "/norm1/bias");
    w.to_q = find_tensor(m, pfx + "/attn1/to_q/weight");
    w.to_k = find_tensor(m, pfx + "/attn1/to_k/weight");
    w.to_v = find_tensor(m, pfx + "/attn1/to_v/weight");
    w.to_out_w = find_tensor(m, pfx + "/attn1/to_out/0/weight");
    w.to_out_b = find_tensor(m, pfx + "/attn1/to_out/0/bias");
    w.norm3_w = find_tensor(m, pfx + "/norm3/weight");
    w.norm3_b = find_tensor(m, pfx + "/norm3/bias");
    w.ff0_w = find_tensor(m, pfx + "/ff/net/0/proj/weight");
    w.ff0_b = find_tensor(m, pfx + "/ff/net/0/proj/bias");
    w.ff2_w = find_tensor(m, pfx + "/ff/net/2/weight");
    w.ff2_b = find_tensor(m, pfx + "/ff/net/2/bias");
    return w;
}
static ggml_tensor * basic_tfm(ggml_context * ctx, const basic_tfm_w & w,
                               ggml_tensor * x, int T, int C, bool f16_kv_attn, int H = 8, int HD = 64) {
    int INNER = H * HD;
    ggml_tensor * nx = layer_norm(ctx, x, w.norm1_w, w.norm1_b);
    ggml_tensor * q = ggml_mul_mat(ctx, w.to_q, nx);
    ggml_tensor * k = ggml_mul_mat(ctx, w.to_k, nx);
    ggml_tensor * v = ggml_mul_mat(ctx, w.to_v, nx);
    const size_t col_stride  = (size_t) INNER * sizeof(float);
    const size_t head_stride = (size_t) HD    * sizeof(float);
    q = ggml_view_3d(ctx, q, HD, T, H, col_stride, head_stride, 0);
    k = ggml_view_3d(ctx, k, HD, T, H, col_stride, head_stride, 0);
    v = ggml_view_3d(ctx, v, HD, T, H, col_stride, head_stride, 0);
    if (f16_kv_attn) {
        ggml_tensor * k_f16 = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, HD, T, H);
        ggml_tensor * v_f16 = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, HD, T, H);
        k = ggml_cpy(ctx, k, k_f16);
        v = ggml_cpy(ctx, v, v_f16);
    }
    ggml_tensor * attn_fa = ggml_flash_attn_ext(ctx, q, k, v, nullptr,
                                                1.0f / std::sqrt((float)HD),
                                                0.0f,
                                                0.0f);
    ggml_tensor * flat = ggml_reshape_2d(ctx, attn_fa, INNER, T);
    ggml_tensor * attn_out = ggml_add(ctx, ggml_mul_mat(ctx, w.to_out_w, flat), w.to_out_b);
    x = ggml_add(ctx, x, attn_out);
    ggml_tensor * nx2 = layer_norm(ctx, x, w.norm3_w, w.norm3_b);
    ggml_tensor * ff = ggml_add(ctx, ggml_mul_mat(ctx, w.ff0_w, nx2), w.ff0_b);
    ff = ggml_gelu_erf(ctx, ff);
    ff = ggml_add(ctx, ggml_mul_mat(ctx, w.ff2_w, ff), w.ff2_b);
    return ggml_add(ctx, x, ff);
}
struct cfm_tfm_stack { std::vector<basic_tfm_w> blocks; };
static cfm_tfm_stack load_tfm_stack(const model_ctx & m, const std::string & pfx, int n) {
    cfm_tfm_stack s;
    for (int i = 0; i < n; ++i) s.blocks.push_back(load_basic_tfm(m, pfx + "/" + std::to_string(i)));
    return s;
}
static ggml_tensor * apply_tfm_stack(ggml_context * ctx, const cfm_tfm_stack & s,
                                     ggml_tensor * x, int T, int C, bool f16_kv_attn) {
    ggml_tensor * xt = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    for (const auto & b : s.blocks) xt = basic_tfm(ctx, b, xt, T, C, f16_kv_attn);
    return ggml_cont(ctx, ggml_permute(ctx, xt, 1, 0, 2, 3));
}
static ggml_tensor * cfm_causal_k3(ggml_context * ctx, ggml_tensor * x,
                                   ggml_tensor * w, ggml_tensor * b, int C_out) {
    ggml_tensor * xp = zero_pad_dim0(ctx, x, 2, 0);
    ggml_tensor * y = conv1d_f32(ctx, w, xp, 1, 0, 1);
    return ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, C_out));
}
static ggml_tensor * cfm_causal_block_b(ggml_context * ctx, ggml_tensor * x,
                                        ggml_tensor * conv_w, ggml_tensor * conv_b,
                                        ggml_tensor * ln_w, ggml_tensor * ln_b, int64_t C_out) {
    ggml_tensor * xp = zero_pad_dim0(ctx, x, 2, 0);
    ggml_tensor * y = conv1d_f32_b(ctx, conv_w, xp, 1, 0, 1);
    y = ggml_add(ctx, y, ggml_reshape_2d(ctx, conv_b, 1, C_out));
    y = layer_norm_on_channel(ctx, y, ln_w, ln_b);
    return ggml_mish_fn(ctx, y);
}
static ggml_tensor * cfm_resnet_b(ggml_context * ctx, const cfm_resnet_w & w,
                                  ggml_tensor * x, ggml_tensor * t_emb_b, int64_t C_out) {
    ggml_tensor * h = cfm_causal_block_b(ctx, x, w.b1_conv_w, w.b1_conv_b, w.b1_ln_w, w.b1_ln_b, C_out);
    ggml_tensor * t_feat = ggml_mish_fn(ctx, t_emb_b);
    ggml_tensor * t_proj = ggml_add(ctx, ggml_mul_mat(ctx, w.mlp_w, t_feat),
                                    w.mlp_b);
    const int64_t B = t_proj->ne[1];
    h = ggml_add(ctx, h, ggml_reshape_3d(ctx, t_proj, 1, C_out, B));
    h = cfm_causal_block_b(ctx, h, w.b2_conv_w, w.b2_conv_b, w.b2_ln_w, w.b2_ln_b, C_out);
    ggml_tensor * res = conv1d_f32_b(ctx, w.res_w, x, 1, 0, 1);
    res = ggml_add(ctx, res, ggml_reshape_2d(ctx, w.res_b, 1, C_out));
    return ggml_add(ctx, h, res);
}
static ggml_tensor * basic_tfm_b(ggml_context * ctx, const basic_tfm_w & w,
                                 ggml_tensor * x, int T, int C, int B,
                                 bool f16_kv_attn,
                                 int H = 8, int HD = 64) {
    int INNER = H * HD;
    ggml_tensor * nx = layer_norm(ctx, x, w.norm1_w, w.norm1_b);
    ggml_tensor * q = ggml_mul_mat(ctx, w.to_q, nx);
    ggml_tensor * k = ggml_mul_mat(ctx, w.to_k, nx);
    ggml_tensor * v = ggml_mul_mat(ctx, w.to_v, nx);
    const size_t col_stride   = (size_t) INNER   * sizeof(float);
    const size_t head_stride  = (size_t) HD      * sizeof(float);
    const size_t batch_stride = (size_t) INNER * T * sizeof(float);
    q = ggml_view_4d(ctx, q, HD, T, H, B, col_stride, head_stride, batch_stride, 0);
    k = ggml_view_4d(ctx, k, HD, T, H, B, col_stride, head_stride, batch_stride, 0);
    v = ggml_view_4d(ctx, v, HD, T, H, B, col_stride, head_stride, batch_stride, 0);
    if (f16_kv_attn) {
        ggml_tensor * k_f16 = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, HD, T, H, B);
        ggml_tensor * v_f16 = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, HD, T, H, B);
        k = ggml_cpy(ctx, k, k_f16);
        v = ggml_cpy(ctx, v, v_f16);
    }
    ggml_tensor * attn_fa = ggml_flash_attn_ext(ctx, q, k, v, nullptr,
                                                1.0f / std::sqrt((float)HD), 0.0f, 0.0f);
    ggml_tensor * flat = ggml_reshape_3d(ctx, attn_fa, INNER, T, B);
    ggml_tensor * attn_out = ggml_add(ctx, ggml_mul_mat(ctx, w.to_out_w, flat), w.to_out_b);
    x = ggml_add(ctx, x, attn_out);
    ggml_tensor * nx2 = layer_norm(ctx, x, w.norm3_w, w.norm3_b);
    ggml_tensor * ff = ggml_add(ctx, ggml_mul_mat(ctx, w.ff0_w, nx2), w.ff0_b);
    ff = ggml_gelu_erf(ctx, ff);
    ff = ggml_add(ctx, ggml_mul_mat(ctx, w.ff2_w, ff), w.ff2_b);
    return ggml_add(ctx, x, ff);
}
static ggml_tensor * apply_tfm_stack_b(ggml_context * ctx, const cfm_tfm_stack & s,
                                       ggml_tensor * x, int T, int C, int B,
                                       bool f16_kv_attn) {
    ggml_tensor * xt = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    for (const auto & b : s.blocks) xt = basic_tfm_b(ctx, b, xt, T, C, B, f16_kv_attn);
    return ggml_cont(ctx, ggml_permute(ctx, xt, 1, 0, 2, 3));
}
static ggml_tensor * cfm_causal_k3_b(ggml_context * ctx, ggml_tensor * x,
                                     ggml_tensor * w, ggml_tensor * b, int C_out) {
    ggml_tensor * xp = zero_pad_dim0(ctx, x, 2, 0);
    ggml_tensor * y = conv1d_f32_b(ctx, w, xp, 1, 0, 1);
    return ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, C_out));
}
struct time_mlp_cache {
    ggml_backend_t  backend = nullptr;
    std::vector<uint8_t> buf;
    ggml_context *  ctx    = nullptr;
    ggml_cgraph *   gf     = nullptr;
    ggml_gallocr_t  allocr = nullptr;
    ggml_tensor *   x_in   = nullptr;
    ggml_tensor *   y_out  = nullptr;
    ~time_mlp_cache() {
        if (allocr) ggml_gallocr_free(allocr);
        if (ctx)    ggml_free(ctx);
    }
};
static std::vector<float> compute_time_mlp(const model_ctx & m, float t_val) {
    const int TDIM = 320;
    std::vector<float> t_sin(TDIM);
    float log_factor = std::log(10000.0f) / (float)(TDIM/2 - 1);
    for (int i = 0; i < TDIM/2; ++i) {
        float freq = std::exp(-(float)i * log_factor);
        float arg = 1000.0f * t_val * freq;
        t_sin[i] = std::sin(arg);
        t_sin[i + TDIM/2] = std::cos(arg);
    }
    thread_local time_mlp_cache cache;
    if (cache.ctx == nullptr || cache.backend != m.backend) {
        if (cache.allocr) { ggml_gallocr_free(cache.allocr); cache.allocr = nullptr; }
        if (cache.ctx)    { ggml_free(cache.ctx); cache.ctx = nullptr; }
        cache.buf.assign(4 * 1024 * 1024, 0);
        ggml_init_params gp = { cache.buf.size(), cache.buf.data(), true };
        cache.ctx = ggml_init(gp);
        cache.gf  = ggml_new_graph(cache.ctx);
        cache.x_in = ggml_new_tensor_1d(cache.ctx, GGML_TYPE_F32, TDIM);
        ggml_set_name(cache.x_in, "x"); ggml_set_input(cache.x_in);
        ggml_tensor * l1w = find_tensor(m, "cfm/time_mlp/linear_1/weight");
        ggml_tensor * l1b = find_tensor(m, "cfm/time_mlp/linear_1/bias");
        ggml_tensor * l2w = find_tensor(m, "cfm/time_mlp/linear_2/weight");
        ggml_tensor * l2b = find_tensor(m, "cfm/time_mlp/linear_2/bias");
        ggml_tensor * y = ggml_add(cache.ctx, ggml_mul_mat(cache.ctx, l1w, cache.x_in), l1b);
        y = ggml_silu(cache.ctx, y);
        y = ggml_add(cache.ctx, ggml_mul_mat(cache.ctx, l2w, y), l2b);
        ggml_set_name(y, "out"); ggml_set_output(y);
        cache.y_out = y;
        ggml_build_forward_expand(cache.gf, cache.y_out);
        cache.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
        ggml_gallocr_reserve(cache.allocr, cache.gf);
        cache.backend = m.backend;
    }
    ggml_gallocr_alloc_graph(cache.allocr, cache.gf);
    ggml_backend_tensor_set(cache.x_in, t_sin.data(), 0, t_sin.size() * sizeof(float));
    compute(m.backend, cache.gf);
    std::vector<float> out(ggml_nelements(cache.y_out));
    ggml_backend_tensor_get(cache.y_out, out.data(), 0, ggml_nbytes(cache.y_out));
    return out;
}
static std::vector<float> compute_time_mixed(const model_ctx & m,
                                             const std::vector<float> & t_mlp,
                                             const std::vector<float> & r_mlp) {
    static size_t buf_size = 4 * 1024 * 1024;
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params gp = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    int TOT = (int)t_mlp.size();
    ggml_tensor * t_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, TOT);
    ggml_set_name(t_in, "t_in"); ggml_set_input(t_in);
    ggml_tensor * r_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, TOT);
    ggml_set_name(r_in, "r_in"); ggml_set_input(r_in);
    ggml_tensor * cat = ggml_concat(ctx, t_in, r_in, 0);
    ggml_tensor * mix_w = find_tensor(m, "cfm/time_embed_mixer/weight");
    ggml_tensor * mixed = ggml_mul_mat(ctx, mix_w, cat);
    ggml_set_name(mixed, "out"); ggml_set_output(mixed);
    ggml_build_forward_expand(gf, mixed);
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "t_in"), t_mlp.data(), 0, t_mlp.size()*sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "r_in"), r_mlp.data(), 0, r_mlp.size()*sizeof(float));
    compute(m.backend, gf);
    std::vector<float> out(ggml_nelements(mixed));
    ggml_backend_tensor_get(mixed, out.data(), 0, ggml_nbytes(mixed));
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    return out;
}
struct cfm_estimator_cache {
    int  T  = -1;
    bool b2 = false;
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t allocr = nullptr;
    std::vector<uint8_t> buf;
    ~cfm_estimator_cache() {
        if (allocr) ggml_gallocr_free(allocr);
        if (ctx) ggml_free(ctx);
    }
};
static std::vector<float> cfm_estimator_forward(
    const model_ctx & m,
    cfm_estimator_cache & cache,
    const std::vector<float> & x,
    const std::vector<float> & mu,
    const std::vector<float> & t_emb,
    const std::vector<float> & spks,
    const std::vector<float> & cond,
    int T,
    bool f16_kv_attn) {
    const int MEL = 80, CH = 256, TIME_DIM = 1024;
    const int N_MID = 12, N_BLOCKS = 4;
    const bool build_graph = (cache.T != T) || cache.b2;
    if (build_graph) {
        if (cache.allocr) { ggml_gallocr_free(cache.allocr); cache.allocr = nullptr; }
        if (cache.ctx) { ggml_free(cache.ctx); cache.ctx = nullptr; }
        cache.buf.resize(64 * 1024 * 1024);
        ggml_init_params gp = { cache.buf.size(), cache.buf.data(), true };
        cache.ctx = ggml_init(gp);
        cache.gf = ggml_new_graph_custom(cache.ctx, 65536, false);
        cache.T  = T;
        cache.b2 = false;
    }
    ggml_context * ctx = cache.ctx;
    ggml_cgraph * gf = cache.gf;
    if (build_graph) {
    ggml_tensor * x_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, MEL); ggml_set_name(x_in, "x_in"); ggml_set_input(x_in);
    ggml_tensor * mu_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, MEL); ggml_set_name(mu_in, "mu_in"); ggml_set_input(mu_in);
    ggml_tensor * spks_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, MEL); ggml_set_name(spks_in, "spks_in"); ggml_set_input(spks_in);
    ggml_tensor * cond_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, MEL); ggml_set_name(cond_in, "cond_in"); ggml_set_input(cond_in);
    ggml_tensor * t_emb_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, TIME_DIM); ggml_set_name(t_emb_in, "t_emb"); ggml_set_input(t_emb_in);
    ggml_tensor * spks_bc = ggml_repeat(ctx, ggml_reshape_2d(ctx, spks_in, 1, MEL), x_in);
    ggml_tensor * xc = ggml_concat(ctx, x_in, mu_in, 1);
    xc = ggml_concat(ctx, xc, spks_bc, 1);
    xc = ggml_concat(ctx, xc, cond_in, 1);
    auto down_rn = load_cfm_resnet(m, "cfm/down_blocks/0/0");
    auto down_tfms = load_tfm_stack(m, "cfm/down_blocks/0/1", N_BLOCKS);
    ggml_tensor * down_conv_w = find_tensor(m, "cfm/down_blocks/0/2/weight");
    ggml_tensor * down_conv_b = find_tensor(m, "cfm/down_blocks/0/2/bias");
    ggml_tensor * z = cfm_resnet(ctx, down_rn, xc, t_emb_in, CH);
    z = apply_tfm_stack(ctx, down_tfms, z, T, CH, f16_kv_attn);
    ggml_tensor * hidden = z;
    z = cfm_causal_k3(ctx, z, down_conv_w, down_conv_b, CH);
    for (int i = 0; i < N_MID; ++i) {
        auto rn = load_cfm_resnet(m, "cfm/mid_blocks/" + std::to_string(i) + "/0");
        auto tfms = load_tfm_stack(m, "cfm/mid_blocks/" + std::to_string(i) + "/1", N_BLOCKS);
        z = cfm_resnet(ctx, rn, z, t_emb_in, CH);
        z = apply_tfm_stack(ctx, tfms, z, T, CH, f16_kv_attn);
    }
    auto up_rn = load_cfm_resnet(m, "cfm/up_blocks/0/0");
    auto up_tfms = load_tfm_stack(m, "cfm/up_blocks/0/1", N_BLOCKS);
    ggml_tensor * up_conv_w = find_tensor(m, "cfm/up_blocks/0/2/weight");
    ggml_tensor * up_conv_b = find_tensor(m, "cfm/up_blocks/0/2/bias");
    z = ggml_concat(ctx, z, hidden, 1);
    z = cfm_resnet(ctx, up_rn, z, t_emb_in, CH);
    z = apply_tfm_stack(ctx, up_tfms, z, T, CH, f16_kv_attn);
    z = cfm_causal_k3(ctx, z, up_conv_w, up_conv_b, CH);
    ggml_tensor * fb_conv_w = find_tensor(m, "cfm/final_block/block/0/weight");
    ggml_tensor * fb_conv_b = find_tensor(m, "cfm/final_block/block/0/bias");
    ggml_tensor * fb_ln_w   = find_tensor(m, "cfm/final_block/block/2/weight");
    ggml_tensor * fb_ln_b   = find_tensor(m, "cfm/final_block/block/2/bias");
    z = cfm_causal_block(ctx, z, fb_conv_w, fb_conv_b, fb_ln_w, fb_ln_b, CH);
    ggml_tensor * fp_w = find_tensor(m, "cfm/final_proj/weight");
    ggml_tensor * fp_b = find_tensor(m, "cfm/final_proj/bias");
    ggml_tensor * out = conv1d_f32(ctx, fp_w, z, 1, 0, 1);
    out = ggml_add(ctx, out, ggml_reshape_2d(ctx, fp_b, 1, MEL));
    ggml_set_name(out, "out"); ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    cache.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(cache.allocr, gf);
    }
    ggml_gallocr_alloc_graph(cache.allocr, gf);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x_in"), x.data(), 0, x.size()*sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mu_in"), mu.data(), 0, mu.size()*sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "spks_in"), spks.data(), 0, spks.size()*sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "cond_in"), cond.data(), 0, cond.size()*sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "t_emb"), t_emb.data(), 0, t_emb.size()*sizeof(float));
    compute(m.backend, gf);
    ggml_tensor * out_t = ggml_graph_get_tensor(gf, "out");
    std::vector<float> out_data(ggml_nelements(out_t));
    ggml_backend_tensor_get(out_t, out_data.data(), 0, ggml_nbytes(out_t));
    return out_data;
}
static void cfm_estimator_forward_b2(
    const model_ctx & m,
    cfm_estimator_cache & cache,
    const std::vector<float> & x_c,     const std::vector<float> & x_u,
    const std::vector<float> & mu_c,    const std::vector<float> & mu_u,
    const std::vector<float> & t_emb_c, const std::vector<float> & t_emb_u,
    const std::vector<float> & spks_c,  const std::vector<float> & spks_u,
    const std::vector<float> & cond_c,  const std::vector<float> & cond_u,
    std::vector<float> & out_c, std::vector<float> & out_u,
    int T,
    bool f16_kv_attn) {
    const int MEL = 80, CH = 256, TIME_DIM = 1024;
    const int N_MID = 12, N_BLOCKS = 4;
    const int B = 2;
    const bool build_graph = (cache.T != T) || !cache.b2;
    if (build_graph) {
        if (cache.allocr) { ggml_gallocr_free(cache.allocr); cache.allocr = nullptr; }
        if (cache.ctx) { ggml_free(cache.ctx); cache.ctx = nullptr; }
        cache.buf.resize(64 * 1024 * 1024);
        ggml_init_params gp = { cache.buf.size(), cache.buf.data(), true };
        cache.ctx = ggml_init(gp);
        cache.gf = ggml_new_graph_custom(cache.ctx, 65536, false);
        cache.T  = T;
        cache.b2 = true;
    }
    ggml_context * ctx = cache.ctx;
    ggml_cgraph * gf = cache.gf;
    if (build_graph) {
    ggml_tensor * x_in    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T, MEL, B); ggml_set_name(x_in, "x_in");       ggml_set_input(x_in);
    ggml_tensor * mu_in   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T, MEL, B); ggml_set_name(mu_in, "mu_in");     ggml_set_input(mu_in);
    ggml_tensor * spks_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, MEL, B);    ggml_set_name(spks_in, "spks_in"); ggml_set_input(spks_in);
    ggml_tensor * cond_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T, MEL, B); ggml_set_name(cond_in, "cond_in"); ggml_set_input(cond_in);
    ggml_tensor * t_emb_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, TIME_DIM, B); ggml_set_name(t_emb_in, "t_emb"); ggml_set_input(t_emb_in);
    ggml_tensor * spks_bc = ggml_repeat(ctx,
        ggml_reshape_3d(ctx, spks_in, 1, MEL, B), x_in);
    ggml_tensor * xc = ggml_concat(ctx, x_in, mu_in, 1);
    xc = ggml_concat(ctx, xc, spks_bc, 1);
    xc = ggml_concat(ctx, xc, cond_in, 1);
    auto down_rn = load_cfm_resnet(m, "cfm/down_blocks/0/0");
    auto down_tfms = load_tfm_stack(m, "cfm/down_blocks/0/1", N_BLOCKS);
    ggml_tensor * down_conv_w = find_tensor(m, "cfm/down_blocks/0/2/weight");
    ggml_tensor * down_conv_b = find_tensor(m, "cfm/down_blocks/0/2/bias");
    ggml_tensor * z = cfm_resnet_b(ctx, down_rn, xc, t_emb_in, CH);
    z = apply_tfm_stack_b(ctx, down_tfms, z, T, CH, B, f16_kv_attn);
    ggml_tensor * hidden = z;
    z = cfm_causal_k3_b(ctx, z, down_conv_w, down_conv_b, CH);
    for (int i = 0; i < N_MID; ++i) {
        auto rn = load_cfm_resnet(m, "cfm/mid_blocks/" + std::to_string(i) + "/0");
        auto tfms = load_tfm_stack(m, "cfm/mid_blocks/" + std::to_string(i) + "/1", N_BLOCKS);
        z = cfm_resnet_b(ctx, rn, z, t_emb_in, CH);
        z = apply_tfm_stack_b(ctx, tfms, z, T, CH, B, f16_kv_attn);
    }
    auto up_rn = load_cfm_resnet(m, "cfm/up_blocks/0/0");
    auto up_tfms = load_tfm_stack(m, "cfm/up_blocks/0/1", N_BLOCKS);
    ggml_tensor * up_conv_w = find_tensor(m, "cfm/up_blocks/0/2/weight");
    ggml_tensor * up_conv_b = find_tensor(m, "cfm/up_blocks/0/2/bias");
    z = ggml_concat(ctx, z, hidden, 1);
    z = cfm_resnet_b(ctx, up_rn, z, t_emb_in, CH);
    z = apply_tfm_stack_b(ctx, up_tfms, z, T, CH, B, f16_kv_attn);
    z = cfm_causal_k3_b(ctx, z, up_conv_w, up_conv_b, CH);
    ggml_tensor * fb_conv_w = find_tensor(m, "cfm/final_block/block/0/weight");
    ggml_tensor * fb_conv_b = find_tensor(m, "cfm/final_block/block/0/bias");
    ggml_tensor * fb_ln_w   = find_tensor(m, "cfm/final_block/block/2/weight");
    ggml_tensor * fb_ln_b   = find_tensor(m, "cfm/final_block/block/2/bias");
    z = cfm_causal_block_b(ctx, z, fb_conv_w, fb_conv_b, fb_ln_w, fb_ln_b, CH);
    ggml_tensor * fp_w = find_tensor(m, "cfm/final_proj/weight");
    ggml_tensor * fp_b = find_tensor(m, "cfm/final_proj/bias");
    ggml_tensor * out = conv1d_f32_b(ctx, fp_w, z, 1, 0, 1);
    out = ggml_add(ctx, out, ggml_reshape_2d(ctx, fp_b, 1, MEL));
    ggml_set_name(out, "out"); ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    cache.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(cache.allocr, gf);
    }
    ggml_gallocr_alloc_graph(cache.allocr, gf);
    const size_t one_tm = (size_t) T * MEL * sizeof(float);
    const size_t one_m  = (size_t) MEL * sizeof(float);
    const size_t one_td = (size_t) TIME_DIM * sizeof(float);
    ggml_tensor * x_t    = ggml_graph_get_tensor(gf, "x_in");
    ggml_tensor * mu_t   = ggml_graph_get_tensor(gf, "mu_in");
    ggml_tensor * spks_t = ggml_graph_get_tensor(gf, "spks_in");
    ggml_tensor * cond_t = ggml_graph_get_tensor(gf, "cond_in");
    ggml_tensor * te_t   = ggml_graph_get_tensor(gf, "t_emb");
    ggml_backend_tensor_set(x_t,     x_c.data(),     0 * one_tm, one_tm);
    ggml_backend_tensor_set(x_t,     x_u.data(),     1 * one_tm, one_tm);
    ggml_backend_tensor_set(mu_t,    mu_c.data(),    0 * one_tm, one_tm);
    ggml_backend_tensor_set(mu_t,    mu_u.data(),    1 * one_tm, one_tm);
    ggml_backend_tensor_set(cond_t,  cond_c.data(),  0 * one_tm, one_tm);
    ggml_backend_tensor_set(cond_t,  cond_u.data(),  1 * one_tm, one_tm);
    ggml_backend_tensor_set(spks_t,  spks_c.data(),  0 * one_m,  one_m);
    ggml_backend_tensor_set(spks_t,  spks_u.data(),  1 * one_m,  one_m);
    ggml_backend_tensor_set(te_t,    t_emb_c.data(), 0 * one_td, one_td);
    ggml_backend_tensor_set(te_t,    t_emb_u.data(), 1 * one_td, one_td);
    compute(m.backend, gf);
    ggml_tensor * out_t = ggml_graph_get_tensor(gf, "out");
    const size_t half = (size_t) T * MEL;
    thread_local std::vector<float> both;
    both.resize((size_t) ggml_nelements(out_t));
    const size_t want = both.size() * sizeof(float);
    if (want > ggml_nbytes(out_t) || both.size() < 2 * half) {
        throw std::runtime_error("cfm b2 out size mismatch");
    }
    ggml_backend_tensor_get(out_t, both.data(), 0, want);
    out_c.assign(both.begin(), both.begin() + half);
    out_u.assign(both.begin() + half, both.begin() + 2 * half);
}
static std::vector<float> build_hann_window(int n, bool periodic = true) {
    std::vector<float> w(n);
    double N = periodic ? (double)n : (double)(n - 1);
    const double two_pi = 2.0 * M_PI;
    for (int i = 0; i < n; ++i) w[i] = (float)(0.5 * (1.0 - std::cos(two_pi * (double)i / N)));
    return w;
}
static std::vector<float> build_stft_kernel(int n_fft, const std::vector<float> & window) {
    int F = n_fft / 2 + 1;
    std::vector<float> K((size_t)n_fft * 1 * (2 * F), 0.0f);
    const double two_pi = 2.0 * M_PI;
    for (int f = 0; f < F; ++f) {
        for (int n = 0; n < n_fft; ++n) {
            double th = two_pi * f * n / n_fft;
            float w = window[n];
            K[n + f       * n_fft] = (float)(std::cos(th) * w);
            K[n + (F + f) * n_fft] = (float)(-std::sin(th) * w);
        }
    }
    return K;
}
static std::vector<float> build_istft_kernel(int n_fft, const std::vector<float> & window) {
    int F = n_fft / 2 + 1;
    std::vector<float> K((size_t)n_fft * 1 * (2 * F), 0.0f);
    const double two_pi = 2.0 * M_PI;
    const double inv_N = 1.0 / (double)n_fft;
    for (int f = 0; f < F; ++f) {
        double coef_re = (f == 0 || f == n_fft/2) ? 1.0 : 2.0;
        double coef_im = (f == 0 || f == n_fft/2) ? 0.0 : 2.0;
        for (int n = 0; n < n_fft; ++n) {
            double th = two_pi * f * n / n_fft;
            float w = window[n];
            K[n + f       * n_fft] = (float)(coef_re * std::cos(th) * w * inv_N);
            K[n + (F + f) * n_fft] = (float)(-coef_im * std::sin(th) * w * inv_N);
        }
    }
    return K;
}
static std::vector<float> build_window_sum(int T_stft, int n_fft, int hop,
                                           const std::vector<float> & window) {
    int L = (T_stft - 1) * hop + n_fft;
    std::vector<float> ws(L, 0.0f);
    for (int t = 0; t < T_stft; ++t) {
        int base = t * hop;
        for (int n = 0; n < n_fft; ++n) ws[base + n] += window[n] * window[n];
    }
    return ws;
}
static ggml_tensor * snake(ggml_context * ctx, ggml_tensor * x,
                           ggml_tensor * alpha, ggml_tensor * inv_alpha) {
    ggml_tensor * a  = ggml_reshape_2d(ctx, alpha,     1, alpha->ne[0]);
    ggml_tensor * ia = ggml_reshape_2d(ctx, inv_alpha, 1, inv_alpha->ne[0]);
    ggml_tensor * ax = ggml_mul(ctx, x, a);
    ggml_tensor * s  = ggml_sin(ctx, ax);
    ggml_tensor * s2 = ggml_mul(ctx, s, s);
    return ggml_add(ctx, x, ggml_mul(ctx, s2, ia));
}
static std::vector<float> invert_alpha_cpu(const model_ctx & m, const std::string & name) {
    ggml_tensor * t = find_tensor(m, name);
    std::vector<float> a(ggml_nelements(t));
    ggml_backend_tensor_get(t, a.data(), 0, ggml_nbytes(t));
    std::vector<float> inv(a.size());
    for (size_t i = 0; i < a.size(); ++i) inv[i] = 1.0f / (a[i] + 1e-9f);
    return inv;
}
static std::vector<float> run_f0_predictor(const model_ctx & m, const std::vector<float> & mel, int T_mel) {
    static size_t buf_size = 8 * 1024 * 1024;
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params gp = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1024, false);
    ggml_tensor * mel_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_mel, 80);
    ggml_set_name(mel_in, "mel_in"); ggml_set_input(mel_in);
    ggml_tensor * x = mel_in;
    for (int i = 0; i < 5; ++i) {
        std::string pfx = "hift/f0_predictor/condnet/" + std::to_string(i * 2);
        ggml_tensor * w = find_tensor(m, pfx + "/weight");
        ggml_tensor * b = find_tensor(m, pfx + "/bias");
        int C_out = (int)w->ne[2];
        x = conv1d_f32(ctx, w, x, 1, 1, 1);
        x = ggml_add(ctx, x, ggml_reshape_2d(ctx, b, 1, C_out));
        x = ggml_unary(ctx, x, GGML_UNARY_OP_ELU);
    }
    ggml_tensor * xp = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    ggml_tensor * cw = find_tensor(m, "hift/f0_predictor/classifier/weight");
    ggml_tensor * cb = find_tensor(m, "hift/f0_predictor/classifier/bias");
    ggml_tensor * y = ggml_mul_mat(ctx, cw, xp);
    y = ggml_add(ctx, y, cb);
    y = ggml_abs(ctx, y);
    y = ggml_reshape_1d(ctx, y, T_mel);
    ggml_set_name(y, "out"); ggml_set_output(y);
    ggml_build_forward_expand(gf, y);
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mel_in"), mel.data(), 0, mel.size()*sizeof(float));
    compute(m.backend, gf);
    std::vector<float> f0(T_mel);
    ggml_backend_tensor_get(y, f0.data(), 0, ggml_nbytes(y));
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    return f0;
}
static std::vector<float> sinegen_source(const std::vector<float> & f0_wav, int sr,
                                         int harmonic_num, float sine_amp, float noise_std,
                                         float voiced_threshold,
                                         const std::vector<float> & l_w, float l_b,
                                         uint32_t seed) {
    int T_wav = (int)f0_wav.size();
    int H = harmonic_num + 1;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(-(float)M_PI, (float)M_PI);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> phase_vec(H, 0.0f);
    for (int h = 1; h < H; ++h) phase_vec[h] = uniform(rng);
    std::vector<float> sine_waves((size_t)H * T_wav, 0.0f);
    std::vector<double> cum_phase(H, 0.0);
    for (int t = 0; t < T_wav; ++t) {
        float f0 = f0_wav[t];
        bool voiced = f0 > voiced_threshold;
        for (int h = 0; h < H; ++h) {
            double inc = (double)f0 * (h + 1) / (double)sr;
            cum_phase[h] += inc;
            double theta = 2.0 * M_PI * (cum_phase[h] - std::floor(cum_phase[h]));
            float sine = sine_amp * std::sin((float)theta + phase_vec[h]);
            float namp = voiced ? noise_std : sine_amp / 3.0f;
            float uv = voiced ? 1.0f : 0.0f;
            sine_waves[(size_t)h * T_wav + t] = sine * uv + namp * gauss(rng);
        }
    }
    std::vector<float> src(T_wav, 0.0f);
    for (int t = 0; t < T_wav; ++t) {
        float s = l_b;
        for (int h = 0; h < H; ++h) s += l_w[h] * sine_waves[(size_t)h * T_wav + t];
        src[t] = std::tanh(s);
    }
    return src;
}
static std::vector<float> run_stft(const model_ctx & m, const std::vector<float> & src) {
    const int n_fft = 16, hop = 4;
    const int F = n_fft / 2 + 1;
    int T_src = (int)src.size();
    auto window = build_hann_window(n_fft, true);
    auto kernel = build_stft_kernel(n_fft, window);
    static size_t buf_size = 4 * 1024 * 1024;
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params gp = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_tensor * s = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_src, 1);
    ggml_set_name(s, "s"); ggml_set_input(s);
    ggml_tensor * s_pad = reflect_pad_1d(ctx, s, n_fft/2, n_fft/2);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_fft, 1, 2*F);
    ggml_set_name(k, "k"); ggml_set_input(k);
    ggml_tensor * spec = conv1d_f32(ctx, k, s_pad, hop, 0, 1);
    ggml_set_name(spec, "out"); ggml_set_output(spec);
    ggml_build_forward_expand(gf, spec);
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "s"), src.data(), 0, src.size()*sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "k"), kernel.data(), 0, kernel.size()*sizeof(float));
    compute(m.backend, gf);
    std::vector<float> out(ggml_nelements(spec));
    ggml_backend_tensor_get(spec, out.data(), 0, ggml_nbytes(spec));
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    return out;
}
static std::vector<float> run_hift_decode(const model_ctx & m,
                                          const std::vector<float> & mel, int T_mel,
                                          const std::vector<float> & s_stft, int T_stft) {
    const int MEL = 80, NFFT2 = 18, BASE_CH = 512, n_fft = 16, hop = 4;
    const int F = n_fft / 2 + 1;
    std::vector<int> ups_rates  = {8, 5, 3};
    std::vector<int> ups_ksizes = {16, 11, 7};
    std::vector<int> ups_ch     = {256, 128, 64};
    std::vector<int> rb_ksizes  = {3, 7, 11};
    std::vector<std::vector<int>> rb_dils = {{1,3,5},{1,3,5},{1,3,5}};
    std::vector<int> src_rb_ksizes = {7, 7, 11};
    std::vector<std::vector<int>> src_rb_dils = {{1,3,5},{1,3,5},{1,3,5}};
    static const size_t buf_size = 64 * 1024 * 1024;
    thread_local std::vector<uint8_t> buf(buf_size);
    ggml_init_params gp = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 131072, false);
    ggml_tensor * mel_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_mel, MEL);
    ggml_set_name(mel_in, "mel_in"); ggml_set_input(mel_in);
    ggml_tensor * s_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_stft, NFFT2);
    ggml_set_name(s_in, "s_in"); ggml_set_input(s_in);
    struct inv_entry { std::string gn; std::vector<float> data; };
    std::vector<inv_entry> inv_alphas;
    auto mk_inv = [&](const std::string & pref, int C) {
        std::string gn = "inv_" + pref;
        auto inv = invert_alpha_cpu(m, pref);
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
        ggml_set_name(t, gn.c_str()); ggml_set_input(t);
        inv_alphas.push_back({gn, std::move(inv)});
        return t;
    };
    auto load_rb = [&](const std::string & pref, int C) {
        struct pd { ggml_tensor *a1, *c1w, *c1b, *a2, *c2w, *c2b, *ia1, *ia2; };
        std::vector<pd> p(3);
        for (int i = 0; i < 3; ++i) {
            p[i].a1 = find_tensor(m, pref + "/activations1/" + std::to_string(i) + "/alpha");
            p[i].c1w = find_tensor(m, pref + "/convs1/" + std::to_string(i) + "/weight");
            p[i].c1b = find_tensor(m, pref + "/convs1/" + std::to_string(i) + "/bias");
            p[i].a2 = find_tensor(m, pref + "/activations2/" + std::to_string(i) + "/alpha");
            p[i].c2w = find_tensor(m, pref + "/convs2/" + std::to_string(i) + "/weight");
            p[i].c2b = find_tensor(m, pref + "/convs2/" + std::to_string(i) + "/bias");
            p[i].ia1 = mk_inv(pref + "/activations1/" + std::to_string(i) + "/alpha", C);
            p[i].ia2 = mk_inv(pref + "/activations2/" + std::to_string(i) + "/alpha", C);
        }
        return p;
    };
    auto rb_fwd = [&](auto & rb, ggml_tensor * x, int C, const std::vector<int> & dils, int ks) {
        for (int i = 0; i < 3; ++i) {
            auto & p = rb[i];
            int d = dils[i];
            int pad1 = (ks * d - d) / 2;
            int pad2 = (ks - 1) / 2;
            ggml_tensor * xt = snake(ctx, x, p.a1, p.ia1);
            xt = conv1d_f32(ctx, p.c1w, xt, 1, pad1, d);
            xt = ggml_add(ctx, xt, ggml_reshape_2d(ctx, p.c1b, 1, C));
            xt = snake(ctx, xt, p.a2, p.ia2);
            xt = conv1d_f32(ctx, p.c2w, xt, 1, pad2, 1);
            xt = ggml_add(ctx, xt, ggml_reshape_2d(ctx, p.c2b, 1, C));
            x = ggml_add(ctx, x, xt);
        }
        return x;
    };
    ggml_tensor * cpw = find_tensor(m, "hift/conv_pre/weight");
    ggml_tensor * cpb = find_tensor(m, "hift/conv_pre/bias");
    ggml_tensor * x = conv1d_f32(ctx, cpw, mel_in, 1, 3, 1);
    x = ggml_add(ctx, x, ggml_reshape_2d(ctx, cpb, 1, BASE_CH));
    for (int i = 0; i < 3; ++i) {
        x = ggml_leaky_relu(ctx, x, 0.1f, false);
        ggml_tensor * uw = find_tensor(m, "hift/ups/" + std::to_string(i) + "/weight");
        ggml_tensor * ub = find_tensor(m, "hift/ups/" + std::to_string(i) + "/bias");
        int up_pad = (ups_ksizes[i] - ups_rates[i]) / 2;
        x = conv_transpose_1d_f32(ctx, uw, x, ups_rates[i], up_pad);
        x = ggml_add(ctx, x, ggml_reshape_2d(ctx, ub, 1, ups_ch[i]));
        if (i == 2) {
            ggml_tensor * xs = ggml_view_3d(ctx, x, 1, x->ne[1], x->ne[2], x->nb[1], x->nb[2], 1 * x->nb[0]);
            xs = ggml_cont(ctx, xs);
            x = ggml_concat(ctx, xs, x, 0);
        }
        ggml_tensor * sw = find_tensor(m, "hift/source_downs/" + std::to_string(i) + "/weight");
        ggml_tensor * sb = find_tensor(m, "hift/source_downs/" + std::to_string(i) + "/bias");
        int sd_stride = (i == 0) ? 15 : (i == 1) ? 3 : 1;
        int sd_pad    = (i == 0) ? 7  : (i == 1) ? 1 : 0;
        ggml_tensor * si = conv1d_f32(ctx, sw, s_in, sd_stride, sd_pad, 1);
        si = ggml_add(ctx, si, ggml_reshape_2d(ctx, sb, 1, (int)sw->ne[2]));
        auto srb = load_rb("hift/source_resblocks/" + std::to_string(i), ups_ch[i]);
        si = rb_fwd(srb, si, ups_ch[i], src_rb_dils[i], src_rb_ksizes[i]);
        x = ggml_add(ctx, x, si);
        ggml_tensor * xs = nullptr;
        for (int j = 0; j < 3; ++j) {
            auto rb = load_rb("hift/resblocks/" + std::to_string(i * 3 + j), ups_ch[i]);
            ggml_tensor * rb_out = rb_fwd(rb, x, ups_ch[i], rb_dils[j], rb_ksizes[j]);
            xs = (xs == nullptr) ? rb_out : ggml_add(ctx, xs, rb_out);
        }
        x = ggml_scale(ctx, xs, 1.0f / 3.0f);
    }
    x = ggml_leaky_relu(ctx, x, 0.01f, false);
    ggml_tensor * cp2w = find_tensor(m, "hift/conv_post/weight");
    ggml_tensor * cp2b = find_tensor(m, "hift/conv_post/bias");
    x = conv1d_f32(ctx, cp2w, x, 1, 3, 1);
    x = ggml_add(ctx, x, ggml_reshape_2d(ctx, cp2b, 1, NFFT2));
    size_t col_stride = x->nb[1];
    ggml_tensor * mag_log = ggml_cont(ctx, ggml_view_2d(ctx, x, T_stft, F, col_stride, 0));
    mag_log = ggml_clamp(ctx, mag_log, -1e6f, 1e2f);
    ggml_tensor * mag = ggml_exp(ctx, mag_log);
    ggml_tensor * ph_in = ggml_cont(ctx, ggml_view_2d(ctx, x, T_stft, F, col_stride, (size_t)F * col_stride));
    ggml_tensor * ph = ggml_sin(ctx, ph_in);
    ggml_tensor * real = ggml_mul(ctx, mag, ggml_cos(ctx, ph));
    ggml_tensor * imag = ggml_mul(ctx, mag, ggml_sin(ctx, ph));
    ggml_tensor * spec = ggml_concat(ctx, real, imag, 1);
    auto window = build_hann_window(n_fft, true);
    auto ik = build_istft_kernel(n_fft, window);
    auto ws = build_window_sum(T_stft, n_fft, hop, window);
    ggml_tensor * istft_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_fft, 1, 2 * F);
    ggml_set_name(istft_k, "istft_k"); ggml_set_input(istft_k);
    ggml_tensor * ws_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int)ws.size(), 1);
    ggml_set_name(ws_in, "w_sum"); ggml_set_input(ws_in);
    ggml_tensor * y = ggml_conv_transpose_1d(ctx, istft_k, spec, hop, 0, 1);
    y = ggml_div(ctx, y, ws_in);
    int pad_amt = n_fft / 2;
    int L_wav = (int)ws.size() - n_fft;
    ggml_tensor * y_trim = ggml_cont(ctx, ggml_view_2d(ctx, y, L_wav, y->ne[1], y->nb[1],
                                                       (size_t)pad_amt * y->nb[0]));
    y_trim = ggml_clamp(ctx, y_trim, -0.99f, 0.99f);
    ggml_set_name(y_trim, "wav"); ggml_set_output(y_trim);
    ggml_build_forward_expand(gf, y_trim);
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mel_in"), mel.data(), 0, mel.size()*sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "s_in"), s_stft.data(), 0, s_stft.size()*sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "istft_k"), ik.data(), 0, ik.size()*sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "w_sum"), ws.data(), 0, ws.size()*sizeof(float));
    for (auto & ia : inv_alphas)
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, ia.gn.c_str()), ia.data.data(), 0, ia.data.size()*sizeof(float));
    compute(m.backend, gf);
    std::vector<float> wav(ggml_nelements(y_trim));
    ggml_backend_tensor_get(y_trim, wav.data(), 0, ggml_nbytes(y_trim));
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    return wav;
}
void s3gen_synthesize(const std::vector<int32_t>& speech_tokens, const s3gen_synthesize_opts& opts) {
    check_cancel(opts.cancel);
    if (speech_tokens.empty()) throw std::runtime_error("S3Gen speech tokens empty");
    if (!opts.pcm_out) throw std::runtime_error("S3Gen PCM output missing");
    if (opts.prompt_token.empty() || opts.embedding.empty() || opts.prompt_feat.empty() || opts.prompt_rows <= 0)
        throw std::runtime_error("S3Gen voice conditioning missing");
    g_n_threads = opts.n_threads;
    constexpr int sr = 24000;
    constexpr int pre_lookahead_len = 3;
    const int seed = opts.seed;
    std::vector<float> emb_data = opts.embedding;
    std::vector<int32_t> pt_data = opts.prompt_token;
    std::vector<float> pf_data = opts.prompt_feat;
    int pf_rows = opts.prompt_rows;
    std::vector<int32_t> padded;
    for (int32_t token : speech_tokens) if (token >= 0 && token < 6561) padded.push_back(token);
    if (padded.empty()) throw std::runtime_error("S3Gen speech tokens invalid");
    padded.insert(padded.end(), pre_lookahead_len, 4299);
    model_ctx& m = *s3gen_model_cache_get(opts.s3gen_gguf_path, opts.n_gpu_layers, opts.fastconv);
    const double load_ms = s3gen_model_cache_last_load_ms();
    const model_ctx& m_hift = m;
    double pipeline_t0 = now_ms();
    const int D = 512;
    const int MEL = 80;
    int n_prompt = (int)pt_data.size();
    int n_total = n_prompt + (int)padded.size();
    std::vector<int32_t> flow_tokens(n_total);
    std::memcpy(flow_tokens.data(), pt_data.data(), n_prompt * sizeof(int32_t));
    std::memcpy(flow_tokens.data() + n_prompt, padded.data(), padded.size() * sizeof(int32_t));
    ggml_tensor * emb_w = find_tensor(m, "flow/input_embedding");
    std::vector<float> emb_w_data(ggml_nelements(emb_w));
    ggml_backend_tensor_get(emb_w, emb_w_data.data(), 0, ggml_nbytes(emb_w));
    int vocab_size = (int)emb_w->ne[1];
    std::vector<float> input_embed(n_total * D);
    for (int i = 0; i < n_total; ++i) {
        int32_t tok = flow_tokens[i];
        if (tok < 0 || tok >= vocab_size) throw std::runtime_error("S3Gen token out of range");
        std::memcpy(input_embed.data() + i * D, emb_w_data.data() + (size_t)tok * D, D * sizeof(float));
    }
    std::vector<float> mu_T = run_encoder(m, input_embed, n_total, D);
    check_cancel(opts.cancel);
    int T_mu = 2 * n_total;
    if (!opts.final) {
        T_mu -= 2 * pre_lookahead_len;
        mu_T.resize((size_t)T_mu * MEL);
    }
    std::vector<float> mu(T_mu * MEL);
    for (int m2 = 0; m2 < MEL; ++m2)
        for (int t = 0; t < T_mu; ++t)
            mu[m2 * T_mu + t] = mu_T[t * MEL + m2];
    const float * emb_raw = emb_data.data();
    float norm = 0.0f;
    for (int i = 0; i < 192; ++i) norm += emb_raw[i] * emb_raw[i];
    norm = std::sqrt(norm + 1e-12f);
    std::vector<float> emb_norm(192);
    for (int i = 0; i < 192; ++i) emb_norm[i] = emb_raw[i] / norm;
    ggml_tensor * saw = find_tensor(m, "flow/spk_embed_affine/w");
    ggml_tensor * sab = find_tensor(m, "flow/spk_embed_affine/b");
    std::vector<float> saw_data(ggml_nelements(saw)), sab_data(ggml_nelements(sab));
    ggml_backend_tensor_get(saw, saw_data.data(), 0, ggml_nbytes(saw));
    ggml_backend_tensor_get(sab, sab_data.data(), 0, ggml_nbytes(sab));
    std::vector<float> spks(MEL, 0.0f);
    for (int o = 0; o < MEL; ++o) {
        float acc = sab_data[o];
        for (int i = 0; i < 192; ++i) acc += saw_data[o * 192 + i] * emb_norm[i];
        spks[o] = acc;
    }
    int mel_len1 = pf_rows;
    if (mel_len1 > T_mu) throw std::runtime_error("S3Gen prompt exceeds output");
    std::vector<float> cond(T_mu * MEL, 0.0f);
    const float * pf_raw = pf_data.data();
    for (int m2 = 0; m2 < MEL; ++m2)
        for (int t = 0; t < mel_len1; ++t)
            cond[m2 * T_mu + t] = pf_raw[t * MEL + m2];
    const bool meanflow = m.meanflow;
    std::vector<float> z(T_mu * MEL);
    const int prompt_len_in_mu = T_mu - 2 * (int)padded.size();
    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    for (float& v : z) v = gauss(rng);
    if (meanflow) {
        std::mt19937 rng2(seed + 2);
        for (int m2 = 0; m2 < MEL; ++m2)
            for (int t = prompt_len_in_mu; t < T_mu; ++t) z[m2 * T_mu + t] = gauss(rng2);
    }
    const int cfm_steps = opts.cfm_steps > 0 ? opts.cfm_steps : (meanflow ? 2 : m.n_timesteps);
    if (!meanflow && cfm_steps < 5) throw std::runtime_error("non-meanflow CFM requires at least 5 steps");
    std::vector<float> t_span;
    t_span.reserve(cfm_steps + 1);
    for (int i = 0; i <= cfm_steps; ++i) {
        float t = (float)i / (float)cfm_steps;
        t_span.push_back(meanflow ? t : 1.0f - std::cos(t * .5f * (float)M_PI));
    }
    const std::vector<float> zeros_tm(T_mu * MEL, 0.0f), zeros_m(MEL, 0.0f);
    cfm_estimator_cache cfm_cache;
    for (size_t step = 0; step + 1 < t_span.size(); ++step) {
        check_cancel(opts.cancel);
        const float t = t_span[step], r = t_span[step + 1], dt = r - t;
        auto t_emb = compute_time_mlp(m, t);
        if (meanflow) t_emb = compute_time_mixed(m, t_emb, compute_time_mlp(m, r));
        std::vector<float> dxdt;
        if (!meanflow && m.cfg_rate != 0.0f) {
            std::vector<float> uncond;
            cfm_estimator_forward_b2(m, cfm_cache, z, z, mu, zeros_tm, t_emb, t_emb,
                spks, zeros_m, cond, zeros_tm, dxdt, uncond, T_mu, false);
            for (size_t i = 0; i < dxdt.size(); ++i) dxdt[i] = (1.0f + m.cfg_rate) * dxdt[i] - m.cfg_rate * uncond[i];
        } else {
            dxdt = cfm_estimator_forward(m, cfm_cache, z, mu, t_emb, spks, cond, T_mu, false);
        }
        check_cancel(opts.cancel);
        for (size_t i = 0; i < z.size(); ++i) z[i] += dt * dxdt[i];
    }
    check_cancel(opts.cancel);
    const int T_mel = T_mu - mel_len1 - opts.skip_mel_frames;
    if (T_mel <= 0) throw std::runtime_error("S3Gen streaming mel range empty");
    std::vector<float> mel(MEL * T_mel);
    const int mel_off = mel_len1 + opts.skip_mel_frames;
    for (int m2 = 0; m2 < MEL; ++m2)
        for (int t = 0; t < T_mel; ++t)
            mel[m2 * T_mel + t] = z[m2 * T_mu + (t + mel_off)];
    auto f0 = run_f0_predictor(m_hift, mel, T_mel);
    check_cancel(opts.cancel);
    int upsample = 8 * 5 * 3 * 4;
    int T_wav = T_mel * upsample;
    std::vector<float> f0_up(T_wav);
    for (int i = 0; i < T_mel; ++i)
        for (int j = 0; j < upsample; ++j) f0_up[i * upsample + j] = f0[i];
    std::vector<float> l_linear_w(9);
    ggml_tensor * llw = find_tensor(m_hift, "hift/m_source/l_linear/weight");
    ggml_tensor * llb = find_tensor(m_hift, "hift/m_source/l_linear/bias");
    ggml_backend_tensor_get(llw, l_linear_w.data(), 0, 9 * sizeof(float));
    float l_linear_b;
    ggml_backend_tensor_get(llb, &l_linear_b, 0, sizeof(float));
    auto src = sinegen_source(f0_up, sr, 8, 0.1f, 0.003f, 10.0f, l_linear_w, l_linear_b, (uint32_t)(seed + 1));
    std::copy_n(opts.hift_cache_source.begin(), std::min(opts.hift_cache_source.size(), src.size()), src.begin());
    if (opts.hift_source_tail) {
        const size_t tail = std::min<std::size_t>(480, src.size());
        opts.hift_source_tail->assign(src.end() - tail, src.end());
    }
    auto s_stft = run_stft(m_hift, src);
    check_cancel(opts.cancel);
    int T_stft = (int)(s_stft.size() / 18);
    auto wav = run_hift_decode(m_hift, mel, T_mel, s_stft, T_stft);
    check_cancel(opts.cancel);
    const int n_trim = sr / 50;
    const int fade_len = 2 * n_trim;
    if ((int)wav.size() >= fade_len) {
        for (int i = 0; i < n_trim; ++i) wav[i] = 0.0f;
        for (int i = 0; i < n_trim; ++i) {
            float theta = (float)M_PI * (1.0f - (float)i / (float)n_trim);
            float w = 0.5f * (std::cos(theta) + 1.0f);
            wav[n_trim + i] *= w;
        }
    }
    const double pipeline_total = now_ms() - pipeline_t0;
    const double audio_ms = 1000.0 * wav.size() / sr;
    char rtf[32];
    std::snprintf(rtf, sizeof(rtf), "%.3f", audio_ms > 0.0 ? pipeline_total / audio_ms : 0.0);
    tts_emit("s3gen",
        ",\"infer_ms\":" + std::to_string((int)(pipeline_total + 0.5)) +
        ",\"audio_ms\":" + std::to_string((int)(audio_ms + 0.5)) +
        ",\"rtf\":" + rtf +
        ",\"tokens\":" + std::to_string(speech_tokens.size()) +
        ",\"cfm_steps\":" + std::to_string(cfm_steps) +
        ",\"meanflow\":" + std::string(meanflow ? "true" : "false") +
        ",\"load_ms\":" + std::to_string((int)(load_ms + 0.5)) +
        ",\"chunk_id\":" + std::to_string(opts.chunk_id) +
        ",\"final\":" + std::string(opts.final ? "true" : "false") +
        ",\"samples\":" + std::to_string(wav.size()));
    *opts.pcm_out = std::move(wav);
}
void s3gen_preload(const std::string& path, int n_gpu_layers, bool fastconv) {
    (void)s3gen_model_cache_get(path, n_gpu_layers, fastconv);
}
void s3gen_unload() {
    s3gen_model_cache_release();
}
