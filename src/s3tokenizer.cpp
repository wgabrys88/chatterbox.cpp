#include "s3tokenizer.h"
#include "voice_features.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf_weights.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
bool s3tokv2_load(const std::string & path, s3tokv2_weights & w)
{
    GgufWeights weights(path);
        w.n_mels       = (int)weights.u32("s3tokv2.n_mels");
        w.n_state      = (int)weights.u32("s3tokv2.n_audio_state");
        w.n_head       = (int)weights.u32("s3tokv2.n_audio_head");
        w.n_layer      = (int)weights.u32("s3tokv2.n_audio_layer");
        w.head_dim     = (int)weights.u32("s3tokv2.head_dim");
        w.mlp_ratio    = (int)weights.u32("s3tokv2.mlp_ratio");
        w.fsmn_kernel  = (int)weights.u32("s3tokv2.fsmn_kernel");
        w.fsq_levels   = (int)weights.u32("s3tokv2.fsq_levels");
        w.fsq_dim      = (int)weights.u32("s3tokv2.fsq_dim");
        w.codebook_size= (int)weights.u32("s3tokv2.codebook_size");
        w.conv_stride  = (int)weights.u32("s3tokv2.conv_stride");
        w.n_fft        = (int)weights.u32("s3tokv2.n_fft");
        w.hop          = (int)weights.u32("s3tokv2.hop");
        w.sample_rate  = (int)weights.u32("s3tokv2.sample_rate");
        w.rope_theta   = weights.f32("s3tokv2.rope_theta");
        w.rope_max_pos = (int)weights.u32("s3tokv2.rope_max_pos");
    weights.copy( "s3tokv2/mel_fb",              w.mel_fb);
    weights.copy( "s3tokv2/encoder/conv1/weight", w.conv1_w);
    weights.copy( "s3tokv2/encoder/conv1/bias",   w.conv1_b);
    weights.copy( "s3tokv2/encoder/conv2/weight", w.conv2_w);
    weights.copy( "s3tokv2/encoder/conv2/bias",   w.conv2_b);
    w.blocks.clear(); w.blocks.resize(w.n_layer);
    for (int i = 0; i < w.n_layer; ++i) {
        auto & b = w.blocks[i];
        const std::string p = "s3tokv2/encoder/blocks/" + std::to_string(i);
        weights.copy( (p + "/attn_ln/weight").c_str(), b.attn_ln_w);
        weights.copy( (p + "/attn_ln/bias").c_str(),   b.attn_ln_b);
        weights.copy( (p + "/attn/query/weight").c_str(), b.q_w);
        weights.copy( (p + "/attn/query/bias").c_str(),   b.q_b);
        weights.copy( (p + "/attn/key/weight").c_str(),   b.k_w);
        weights.copy( (p + "/attn/value/weight").c_str(), b.v_w);
        weights.copy( (p + "/attn/value/bias").c_str(),   b.v_b);
        weights.copy( (p + "/attn/out/weight").c_str(),   b.out_w);
        weights.copy( (p + "/attn/out/bias").c_str(),     b.out_b);
        weights.copy( (p + "/attn/fsmn_block/weight").c_str(), b.fsmn_w);
        weights.copy( (p + "/mlp_ln/weight").c_str(), b.mlp_ln_w);
        weights.copy( (p + "/mlp_ln/bias").c_str(),   b.mlp_ln_b);
        weights.copy( (p + "/mlp/0/weight").c_str(), b.mlp0_w);
        weights.copy( (p + "/mlp/0/bias").c_str(),   b.mlp0_b);
        weights.copy( (p + "/mlp/2/weight").c_str(), b.mlp2_w);
        weights.copy( (p + "/mlp/2/bias").c_str(),   b.mlp2_b);
    }
    weights.copy( "s3tokv2/quantizer/_codebook/project_down/weight", w.fsq_w);
    weights.copy( "s3tokv2/quantizer/_codebook/project_down/bias",   w.fsq_b);
    return true;
}
static void reflect_pad(const float * in, int L, int left, int right,
                        std::vector<float> & out)
{
    out.resize((size_t)(L + left + right));
    for (int i = 0; i < left;  ++i) out[i]          = in[left  - i];
    for (int i = 0; i < L;     ++i) out[left + i]   = in[i];
    for (int i = 0; i < right; ++i) out[left + L + i] = in[L - 2 - i];
}
static std::vector<float> s3tokv2_log_mel(const std::vector<float> & wav,
                                   const s3tokv2_weights & w,
                                   ggml_backend_t backend,
                                   int & out_T)
{
    const int n_fft  = w.n_fft;
    const int hop    = w.hop;
    const int F      = n_fft / 2 + 1;
    const int n_mels = w.n_mels;
    const int L = (int)wav.size();
    const int pad = n_fft / 2;
    std::vector<float> padded;
    reflect_pad(wav.data(), L, pad, pad, padded);
    const int n_frames = ((int)padded.size() - n_fft) / hop + 1;
    const int T = n_frames - 1;
    std::vector<float> hann(n_fft);
    for (int n = 0; n < n_fft; ++n)
        hann[n] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * (float)n / (float)n_fft));
    std::vector<float> frames((size_t)T * n_fft, 0.0f);
    for (int t = 0; t < T; ++t) {
        const float * x = padded.data() + t * hop;
        float * f = frames.data() + (size_t)t * n_fft;
        for (int n = 0; n < n_fft; ++n) f[n] = x[n] * hann[n];
    }
    std::vector<float> mel_tm = mel_graph_run(frames, w.mel_fb, T, n_fft, F, n_mels, 2.0f, -1.0f, backend);
    std::vector<float> mel((size_t)n_mels * T);
    for (int t = 0; t < T; ++t)
        for (int m = 0; m < n_mels; ++m)
            mel[(size_t)m * T + t] = mel_tm[(size_t)t * n_mels + m];
    const float log10_inv = 1.0f / std::log(10.0f);
    float max_v = -std::numeric_limits<float>::infinity();
    for (float & v : mel) {
        v = std::log(std::max(v, 1e-10f)) * log10_inv;
        if (v > max_v) max_v = v;
    }
    const float floor_v = max_v - 8.0f;
    for (float & v : mel) {
        if (v < floor_v) v = floor_v;
        v = (v + 4.0f) / 4.0f;
    }
    out_T = T;
    return mel;
}
namespace {
struct encoder_ctx {
    ggml_backend_t          backend      = nullptr;
    ggml_context         *  ctx          = nullptr;
    ggml_backend_buffer_t   buffer       = nullptr;
    ggml_gallocr_t          alloc        = nullptr;
    ggml_tensor * mel_in = nullptr;
    ggml_tensor * pos    = nullptr;
    ggml_tensor * conv1_w = nullptr, * conv1_b = nullptr;
    ggml_tensor * conv2_w = nullptr, * conv2_b = nullptr;
    struct block_t {
        ggml_tensor * attn_ln_w, * attn_ln_b;
        ggml_tensor * q_w, * q_b;
        ggml_tensor * k_w;
        ggml_tensor * v_w, * v_b;
        ggml_tensor * out_w, * out_b;
        ggml_tensor * fsmn_w;
        ggml_tensor * mlp_ln_w, * mlp_ln_b;
        ggml_tensor * mlp0_w, * mlp0_b;
        ggml_tensor * mlp2_w, * mlp2_b;
    };
    std::vector<block_t> blocks;
};
static ggml_tensor * linear(ggml_context * ctx,
                            ggml_tensor * x,
                            ggml_tensor * w,
                            ggml_tensor * b )
{
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    if (b) {
        y = ggml_add(ctx, y, b);
    }
    return y;
}
static ggml_tensor * conv1d_f32(ggml_context * ctx,
                                ggml_tensor * kernel, ggml_tensor * input,
                                int stride, int padding, int dilation)
{
    ggml_tensor * im2col = ggml_im2col(ctx, kernel, input,
                                       stride, 0, padding, 0, dilation, 0,
                                       false, GGML_TYPE_F32);
    ggml_tensor * r = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
        ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]));
    return ggml_reshape_3d(ctx, r, im2col->ne[1], kernel->ne[2], im2col->ne[2]);
}
static ggml_tensor * conv1d_dw_f32(ggml_context * ctx,
                                   ggml_tensor * kernel, ggml_tensor * input,
                                   int stride, int padding, int dilation)
{
    ggml_tensor * new_b = ggml_reshape_4d(ctx, input, input->ne[0], 1, input->ne[1], input->ne[2]);
    ggml_tensor * im2col = ggml_im2col(ctx, kernel, new_b,
                                       stride, 0, padding, 0, dilation, 0,
                                       false, GGML_TYPE_F32);
    ggml_tensor * result = ggml_mul_mat(ctx, im2col, kernel);
    return ggml_reshape_3d(ctx, result, result->ne[0], result->ne[2], 1);
}
static ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x,
                                ggml_tensor * gamma, ggml_tensor * beta,
                                float eps = 1e-5f)
{
    ggml_tensor * y = ggml_norm(ctx, x, eps);
    y = ggml_mul(ctx, y, gamma);
    y = ggml_add(ctx, y, beta);
    return y;
}

}
static bool build_encoder_ctx(encoder_ctx & ec, const s3tokv2_weights & w,
                               ggml_backend_t backend)
{
    ec.backend = backend;
    const int n_tensors = 4 + 16 * w.n_layer + 8;
    ggml_init_params ip = {
         (size_t)n_tensors * ggml_tensor_overhead(),
         nullptr,
         true,
    };
    ec.ctx = ggml_init(ip);
    ec.conv1_w = ggml_new_tensor_3d(ec.ctx, GGML_TYPE_F32, 3, w.n_mels, w.n_state);
    ec.conv1_b = ggml_new_tensor_1d(ec.ctx, GGML_TYPE_F32, w.n_state);
    ec.conv2_w = ggml_new_tensor_3d(ec.ctx, GGML_TYPE_F32, 3, w.n_state, w.n_state);
    ec.conv2_b = ggml_new_tensor_1d(ec.ctx, GGML_TYPE_F32, w.n_state);
    ec.blocks.resize(w.n_layer);
    for (int i = 0; i < w.n_layer; ++i) {
        auto & B = ec.blocks[i];
        B.attn_ln_w = ggml_new_tensor_1d(ec.ctx, GGML_TYPE_F32, w.n_state);
        B.attn_ln_b = ggml_new_tensor_1d(ec.ctx, GGML_TYPE_F32, w.n_state);
        B.q_w       = ggml_new_tensor_2d(ec.ctx, GGML_TYPE_F32, w.n_state, w.n_state);
        B.q_b       = ggml_new_tensor_1d(ec.ctx, GGML_TYPE_F32, w.n_state);
        B.k_w       = ggml_new_tensor_2d(ec.ctx, GGML_TYPE_F32, w.n_state, w.n_state);
        B.v_w       = ggml_new_tensor_2d(ec.ctx, GGML_TYPE_F32, w.n_state, w.n_state);
        B.v_b       = ggml_new_tensor_1d(ec.ctx, GGML_TYPE_F32, w.n_state);
        B.out_w     = ggml_new_tensor_2d(ec.ctx, GGML_TYPE_F32, w.n_state, w.n_state);
        B.out_b     = ggml_new_tensor_1d(ec.ctx, GGML_TYPE_F32, w.n_state);
        B.fsmn_w    = ggml_new_tensor_3d(ec.ctx, GGML_TYPE_F32, w.fsmn_kernel, 1, w.n_state);
        B.mlp_ln_w  = ggml_new_tensor_1d(ec.ctx, GGML_TYPE_F32, w.n_state);
        B.mlp_ln_b  = ggml_new_tensor_1d(ec.ctx, GGML_TYPE_F32, w.n_state);
        const int mlp_hidden = w.n_state * w.mlp_ratio;
        B.mlp0_w    = ggml_new_tensor_2d(ec.ctx, GGML_TYPE_F32, w.n_state, mlp_hidden);
        B.mlp0_b    = ggml_new_tensor_1d(ec.ctx, GGML_TYPE_F32, mlp_hidden);
        B.mlp2_w    = ggml_new_tensor_2d(ec.ctx, GGML_TYPE_F32, mlp_hidden, w.n_state);
        B.mlp2_b    = ggml_new_tensor_1d(ec.ctx, GGML_TYPE_F32, w.n_state);
    }
    ec.buffer = ggml_backend_alloc_ctx_tensors(ec.ctx, ec.backend);
    auto set = [&](ggml_tensor * t, const std::vector<float> & src) {
        size_t bytes = src.size() * sizeof(float);
        ggml_backend_tensor_set(t, src.data(), 0, bytes);
        return true;
    };
    set(ec.conv1_w, w.conv1_w);
    set(ec.conv1_b, w.conv1_b);
    set(ec.conv2_w, w.conv2_w);
    set(ec.conv2_b, w.conv2_b);
    for (int i = 0; i < w.n_layer; ++i) {
        auto & B = ec.blocks[i];
        const auto & src = w.blocks[i];
        set(B.attn_ln_w, src.attn_ln_w);
        set(B.attn_ln_b, src.attn_ln_b);
        set(B.q_w, src.q_w); set(B.q_b, src.q_b);
        set(B.k_w, src.k_w);
        set(B.v_w, src.v_w); set(B.v_b, src.v_b);
        set(B.out_w, src.out_w); set(B.out_b, src.out_b);
        set(B.fsmn_w, src.fsmn_w);
        set(B.mlp_ln_w, src.mlp_ln_w);
        set(B.mlp_ln_b, src.mlp_ln_b);
        set(B.mlp0_w, src.mlp0_w); set(B.mlp0_b, src.mlp0_b);
        set(B.mlp2_w, src.mlp2_w); set(B.mlp2_b, src.mlp2_b);
    }
    return true;
}
static void free_encoder_ctx(encoder_ctx & ec) {
    if (ec.alloc)  { ggml_gallocr_free(ec.alloc);  ec.alloc = nullptr; }
    if (ec.buffer) { ggml_backend_buffer_free(ec.buffer); ec.buffer = nullptr; }
    if (ec.ctx)    { ggml_free(ec.ctx); ec.ctx = nullptr; }
    ec.backend = nullptr;
}
static ggml_tensor * build_encoder_graph(encoder_ctx & ec,
                                         ggml_context * ctx,
                                         const s3tokv2_weights & w,
                                         int T_mel)
{
    ggml_tensor * x = conv1d_f32(ctx, ec.conv1_w, ec.mel_in, w.conv_stride, 1, 1);
    x = ggml_add(ctx, x, ggml_reshape_2d(ctx, ec.conv1_b, 1, w.n_state));
    x = ggml_gelu_erf(ctx, x);
    ggml_tensor * y = conv1d_f32(ctx, ec.conv2_w, x, w.conv_stride, 1, 1);
    y = ggml_add(ctx, y, ggml_reshape_2d(ctx, ec.conv2_b, 1, w.n_state));
    y = ggml_gelu_erf(ctx, y);
    ggml_tensor * h = ggml_cont(ctx, ggml_transpose(ctx, y));
    const int n_head   = w.n_head;
    const int head_dim = w.head_dim;
    const int n_state  = w.n_state;
    for (int i = 0; i < w.n_layer; ++i) {
        auto & B = ec.blocks[i];
        ggml_tensor * ln = layer_norm(ctx, h, B.attn_ln_w, B.attn_ln_b);
        ggml_tensor * q = linear(ctx, ln, B.q_w, B.q_b);
        ggml_tensor * k = linear(ctx, ln, B.k_w, nullptr);
        ggml_tensor * v = linear(ctx, ln, B.v_w, B.v_b);
        const int T = (int)q->ne[1];
        q = ggml_reshape_3d(ctx, q, head_dim, n_head, T);
        k = ggml_reshape_3d(ctx, k, head_dim, n_head, T);
        v = ggml_reshape_3d(ctx, v, head_dim, n_head, T);
        q = ggml_rope_ext(ctx, q, ec.pos, nullptr, head_dim,
                          GGML_ROPE_TYPE_NEOX, w.rope_max_pos,
                          w.rope_theta, 1.0f,
                          0.0f, 1.0f,
                          32.0f, 1.0f);
        k = ggml_rope_ext(ctx, k, ec.pos, nullptr, head_dim,
                          GGML_ROPE_TYPE_NEOX, w.rope_max_pos, w.rope_theta, 1.0f,
                          0.0f, 1.0f, 32.0f, 1.0f);
        ggml_tensor * v_flat = ggml_reshape_2d(ctx, ggml_cont(ctx, v), n_state, T);
        ggml_tensor * v_tn   = ggml_cont(ctx, ggml_transpose(ctx, v_flat));
        ggml_tensor * fsmn   = conv1d_dw_f32(ctx, B.fsmn_w, v_tn,
                                             1, (w.fsmn_kernel - 1) / 2, 1);
        fsmn = ggml_add(ctx, fsmn, v_tn);
        ggml_tensor * fsmn_memory = ggml_cont(ctx, ggml_transpose(ctx, fsmn));
        ggml_tensor * q_perm = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        ggml_tensor * k_perm = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        ggml_tensor * v_perm = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
        ggml_tensor * scores = ggml_mul_mat(ctx, k_perm, q_perm);
        const float scale = 1.0f / std::sqrt((float)head_dim);
        scores = ggml_scale(ctx, scores, scale);
        scores = ggml_soft_max(ctx, scores);
        ggml_tensor * attn_out = ggml_mul_mat(ctx, v_perm, scores);
        attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));
        attn_out = ggml_reshape_2d(ctx, attn_out, n_state, T);
        ggml_tensor * out_proj = linear(ctx, attn_out, B.out_w, B.out_b);
        h = ggml_add(ctx, h, ggml_add(ctx, out_proj, fsmn_memory));
        ggml_tensor * ln2 = layer_norm(ctx, h, B.mlp_ln_w, B.mlp_ln_b);
        ggml_tensor * m = linear(ctx, ln2, B.mlp0_w, B.mlp0_b);
        m = ggml_gelu_erf(ctx, m);
        m = linear(ctx, m, B.mlp2_w, B.mlp2_b);
        h = ggml_add(ctx, h, m);
    }
    (void)T_mel;
    return h;
}
bool s3tokv2_tokenize(const std::vector<float> & wav,
                      const s3tokv2_weights & w,
                      int max_tokens,
                      std::vector<int32_t> & out_tokens,
                      ggml_backend_t backend)
{
    int T_mel = 0;
    std::vector<float> mel = s3tokv2_log_mel(wav, w, backend, T_mel);
    if (mel.empty()) return false;
    const int T1 = (T_mel + 2 - 2 - 1) / 2 + 1;
    const int T2 = (T1    + 2 - 2 - 1) / 2 + 1;
    encoder_ctx ec;
    if (!build_encoder_ctx(ec, w, backend)) { free_encoder_ctx(ec); return false; }
    ggml_context * input_ctx = nullptr;
    {
        ggml_init_params ip2 = {
             4 * ggml_tensor_overhead(),
             nullptr,
             true,
        };
        input_ctx = ggml_init(ip2);
    }
    ec.mel_in = ggml_new_tensor_2d(input_ctx, GGML_TYPE_F32, T_mel, w.n_mels);
    ggml_set_name(ec.mel_in, "mel_in");
    ec.pos = ggml_new_tensor_1d(input_ctx, GGML_TYPE_I32, T2);
    ggml_set_name(ec.pos, "pos");
    ggml_backend_buffer_t input_buf = ggml_backend_alloc_ctx_tensors(input_ctx, ec.backend);
    if (!input_buf) { free_encoder_ctx(ec); ggml_free(input_ctx); return false; }
    ggml_context * run_ctx = nullptr;
    {
        ggml_init_params ip3 = {
             ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(4096, false),
             nullptr,
             true,
        };
        run_ctx = ggml_init(ip3);
        if (!run_ctx) {
            throw std::runtime_error("s3tokv2 run_ctx");
        }
    }
    ggml_backend_tensor_set(ec.mel_in, mel.data(), 0, mel.size() * sizeof(float));
    std::vector<int32_t> pos(T2);
    for (int i = 0; i < T2; ++i) pos[i] = i;
    ggml_backend_tensor_set(ec.pos, pos.data(), 0, pos.size() * sizeof(int32_t));
    ggml_cgraph * gf = ggml_new_graph_custom(run_ctx, 4096, false);
    ggml_tensor * h_out = build_encoder_graph(ec, run_ctx, w, T_mel);
    ggml_build_forward_expand(gf, h_out);
    ec.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ec.backend));
    if (!ggml_gallocr_alloc_graph(ec.alloc, gf)) {
        throw std::runtime_error("s3tokv2 alloc");
    }
    if (ggml_backend_graph_compute(ec.backend, gf) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("s3tokv2 compute");
    }
    const int T_out = (int)h_out->ne[1];
    const int D_out = (int)h_out->ne[0];
    std::vector<float> hidden((size_t)T_out * D_out);
    ggml_backend_tensor_get(h_out, hidden.data(), 0, hidden.size() * sizeof(float));
    free_encoder_ctx(ec);
    ggml_backend_buffer_free(input_buf); ggml_free(input_ctx);
    ggml_free(run_ctx);
    const int fsq_dim = w.fsq_dim;
    std::vector<int32_t> tokens(T_out);
    for (int t = 0; t < T_out; ++t) {
        const float * h = hidden.data() + (size_t)t * D_out;
        int32_t code = 0;
        int32_t power = 1;
        for (int o = 0; o < fsq_dim; ++o) {
            float acc = w.fsq_b[o];
            const float * row = w.fsq_w.data() + (size_t)o * D_out;
            for (int d = 0; d < D_out; ++d) acc += row[d] * h[d];
            float q = std::tanh(acc) * 0.9990000128746033f;
            int32_t r = (int32_t)std::lround(q) + 1;
            if (r < 0) r = 0;
            if (r > w.fsq_levels - 1) r = w.fsq_levels - 1;
            code += r * power;
            power *= w.fsq_levels;
        }
        tokens[t] = code;
    }
    if (max_tokens > 0 && (int)tokens.size() > max_tokens) tokens.resize(max_tokens);
    out_tokens = std::move(tokens);
    return true;
}
