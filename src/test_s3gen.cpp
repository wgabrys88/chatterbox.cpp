


#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include "npy.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>


struct model_ctx {
    ggml_backend_t backend = nullptr;
    ggml_context * ctx_w = nullptr;
    ggml_backend_buffer_t buffer_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
};

static model_ctx load_s3gen_gguf(const std::string & path) {
    model_ctx m;

    ggml_context * tmp_ctx = nullptr;
    gguf_init_params gp = {  false,  &tmp_ctx };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) throw std::runtime_error("gguf_init_from_file failed: " + path);

    m.backend = ggml_backend_cpu_init();
    int64_t n_tensors = gguf_get_n_tensors(g);

    ggml_init_params p = { ggml_tensor_overhead() * (size_t)n_tensors, nullptr, true };
    m.ctx_w = ggml_init(p);

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(g, i);
        ggml_tensor * src = ggml_get_tensor(tmp_ctx, name);
        ggml_tensor * dst = ggml_dup_tensor(m.ctx_w, src);
        ggml_set_name(dst, name);
        m.tensors[name] = dst;
    }
    m.buffer_w = ggml_backend_alloc_ctx_tensors(m.ctx_w, m.backend);
    for (ggml_tensor * cur = ggml_get_first_tensor(m.ctx_w); cur; cur = ggml_get_next_tensor(m.ctx_w, cur)) {
        ggml_tensor * src = ggml_get_tensor(tmp_ctx, ggml_get_name(cur));
        ggml_backend_tensor_set(cur, ggml_get_data(src), 0, ggml_nbytes(src));
    }
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
    result = ggml_reshape_3d(ctx, result, im2col->ne[1], kernel->ne[2], im2col->ne[2]);
    return result;
}


static std::vector<float> compute_and_read(
    ggml_backend_t backend, ggml_cgraph * gf, ggml_tensor * out,
    ggml_gallocr_t allocr, int n_threads = 4) {
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);
    if (ggml_backend_is_cpu(backend)) ggml_backend_cpu_set_n_threads(backend, n_threads);
    ggml_backend_graph_compute(backend, gf);
    std::vector<float> data(ggml_nelements(out));
    ggml_backend_tensor_get(out, data.data(), 0, ggml_nbytes(out));
    return data;
}


static void stage_A(const model_ctx & m, const std::string & ref_dir) {
    fprintf(stderr, "\n=== Stage A: speaker_emb projection ===\n");
    npy_array emb = npy_load(ref_dir + "/embedding.npy");
    npy_array expect = npy_load(ref_dir + "/speaker_emb_affine.npy");

    static size_t buf_size = ggml_tensor_overhead()*64 + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);
    ggml_init_params gp = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * inp = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 192);
    ggml_set_name(inp, "embedding");
    ggml_set_input(inp);


    ggml_tensor * sq = ggml_sqr(ctx, inp);
    ggml_tensor * sum_sq = ggml_sum(ctx, sq);

    ggml_tensor * rsqrt_ns = ggml_sqrt(ctx, sum_sq);
    ggml_tensor * one = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(one, "one");
    ggml_set_input(one);
    ggml_tensor * inv_norm = ggml_div(ctx, one, rsqrt_ns);
    ggml_tensor * normed = ggml_scale(ctx, inp, 1.0f);
    normed = ggml_mul(ctx, inp, ggml_repeat(ctx, inv_norm, inp));


    ggml_tensor * W = find_tensor(m, "flow/spk_embed_affine/w");
    ggml_tensor * b = find_tensor(m, "flow/spk_embed_affine/b");
    ggml_tensor * y = ggml_mul_mat(ctx, W, normed);
    y = ggml_add(ctx, y, b);
    ggml_set_name(y, "out");
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);


    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "embedding"),
                            npy_as_f32(emb), 0, emb.data.size());
    float one_val = 1.0f;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "one"), &one_val, 0, sizeof(float));

    ggml_backend_graph_compute(m.backend, gf);
    std::vector<float> out(ggml_nelements(y));
    ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));

    auto stats = compare_f32(out.data(), npy_as_f32(expect), std::min(out.size(), expect.n_elements()));
    print_compare("speaker_emb_affine", stats);
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
}


static void stage_B(const model_ctx & m, const std::string & ref_dir) {
    fprintf(stderr, "\n=== Stage B: input_embedding ===\n");
    npy_array tokens = npy_load(ref_dir + "/flow_input_tokens.npy");
    npy_array expect = npy_load(ref_dir + "/input_embedded.npy");
    int64_t N = tokens.shape[0];
    fprintf(stderr, "  N=%lld\n", (long long)N);

    static size_t buf_size = ggml_tensor_overhead()*64 + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);
    ggml_init_params gp = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_set_name(ids, "ids");
    ggml_set_input(ids);

    ggml_tensor * W = find_tensor(m, "flow/input_embedding");
    ggml_tensor * y = ggml_get_rows(ctx, W, ids);
    ggml_set_name(y, "out");
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "ids"), npy_as_i32(tokens), 0, tokens.data.size());
    ggml_backend_graph_compute(m.backend, gf);

    std::vector<float> out(ggml_nelements(y));
    ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));

    auto stats = compare_f32(out.data(), npy_as_f32(expect), std::min(out.size(), expect.n_elements()));
    print_compare("input_embedded", stats);
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
}


static void stage_C(const model_ctx & m, const std::string & ref_dir) {
    fprintf(stderr, "\n=== Stage C: encoder_embed ===\n");
    npy_array input = npy_load(ref_dir + "/input_embedded.npy");
    npy_array exp_x = npy_load(ref_dir + "/encoder_embed_tup0.npy");
    npy_array exp_p = npy_load(ref_dir + "/encoder_embed_tup1.npy");
    int64_t N = input.shape[0];
    int64_t D = input.shape[1];
    fprintf(stderr, "  N=%lld D=%lld pos_emb_len=%lld\n", (long long)N, (long long)D, (long long)exp_p.shape[0]);


    int64_t L = 2 * N - 1;
    std::vector<float> pe(L * D, 0.0f);


    {
        const float log10000 = std::log(10000.0f);
        std::vector<float> div_term(D / 2);
        for (int i = 0; i < D / 2; ++i) div_term[i] = std::exp(-((float)(2*i) * log10000 / (float)D));

        std::vector<std::vector<float>> pos_pe(N, std::vector<float>(D, 0.0f));
        std::vector<std::vector<float>> neg_pe(N, std::vector<float>(D, 0.0f));
        for (int t = 0; t < N; ++t) {
            for (int i = 0; i < D / 2; ++i) {
                float angle_pos = (float)t * div_term[i];
                float angle_neg = -(float)t * div_term[i];
                pos_pe[t][2*i]     = std::sin(angle_pos);
                pos_pe[t][2*i + 1] = std::cos(angle_pos);
                neg_pe[t][2*i]     = std::sin(angle_neg);
                neg_pe[t][2*i + 1] = std::cos(angle_neg);
            }
        }


        for (int t = 0; t < N; ++t) {
            int src = N - 1 - t;
            for (int d = 0; d < D; ++d) pe[t*D + d] = pos_pe[src][d];
        }
        for (int t = 1; t < N; ++t) {
            for (int d = 0; d < D; ++d) pe[(N - 1 + t - 1 + 1)*D + d] = neg_pe[t][d];


        }

        for (int t = 1; t < N; ++t) {
            for (int d = 0; d < D; ++d) pe[(N - 1 + t)*D + d] = neg_pe[t][d];
        }
    }


    auto pe_stats = compare_f32(pe.data(), npy_as_f32(exp_p), pe.size());
    print_compare("pos_emb", pe_stats);


    static size_t buf_size = ggml_tensor_overhead()*64 + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);
    ggml_init_params gp = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, N);
    ggml_set_name(x, "x");
    ggml_set_input(x);

    ggml_tensor * lw = find_tensor(m, "flow/encoder/embed/linear/w");
    ggml_tensor * lb = find_tensor(m, "flow/encoder/embed/linear/b");
    ggml_tensor * nw = find_tensor(m, "flow/encoder/embed/norm/w");
    ggml_tensor * nb = find_tensor(m, "flow/encoder/embed/norm/b");


    ggml_tensor * y = ggml_mul_mat(ctx, lw, x);
    y = ggml_add(ctx, y, lb);

    y = ggml_norm(ctx, y, 1e-5f);
    y = ggml_mul(ctx, y, nw);
    y = ggml_add(ctx, y, nb);

    y = ggml_scale(ctx, y, std::sqrt((float)D));
    ggml_set_name(y, "out");
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x"), npy_as_f32(input), 0, input.data.size());
    ggml_backend_graph_compute(m.backend, gf);
    std::vector<float> out(ggml_nelements(y));
    ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));

    auto xs = compare_f32(out.data(), npy_as_f32(exp_x), std::min(out.size(), exp_x.n_elements()));
    print_compare("encoder_embed_x", xs);

    ggml_gallocr_free(allocr);
    ggml_free(ctx);
}


static void stage_D(const model_ctx & m, const std::string & ref_dir) {
    fprintf(stderr, "\n=== Stage D: pre_lookahead ===\n");
    npy_array input_npy = npy_load(ref_dir + "/encoder_embed_tup0.npy");
    npy_array exp_npy   = npy_load(ref_dir + "/pre_lookahead.npy");
    int64_t N = input_npy.shape[0];
    int64_t D = input_npy.shape[1];
    fprintf(stderr, "  N=%lld D=%lld\n", (long long)N, (long long)D);


    static size_t buf_size = ggml_tensor_overhead()*64 + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);
    ggml_init_params gp = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph(ctx);


    ggml_tensor * x_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, N);
    ggml_set_name(x_in, "x_in");
    ggml_set_input(x_in);


    ggml_tensor * x = ggml_cont(ctx, ggml_permute(ctx, x_in, 1, 0, 2, 3));


    ggml_tensor * w1 = find_tensor(m, "flow/encoder/pre_lookahead/conv1/w");
    ggml_tensor * b1 = find_tensor(m, "flow/encoder/pre_lookahead/conv1/b");
    ggml_tensor * w2 = find_tensor(m, "flow/encoder/pre_lookahead/conv2/w");
    ggml_tensor * b2 = find_tensor(m, "flow/encoder/pre_lookahead/conv2/b");

    ggml_tensor * x_padded1 = ggml_pad_ext(ctx, x, 0, 3, 0, 0, 0, 0, 0, 0);
    ggml_tensor * y = conv1d_f32(ctx, w1, x_padded1, 1, 0, 1);

    ggml_tensor * b1_r = ggml_reshape_2d(ctx, b1, 1, D);
    y = ggml_add(ctx, y, b1_r);
    y = ggml_leaky_relu(ctx, y, 0.01f, false);

    ggml_tensor * y_padded = ggml_pad_ext(ctx, y, 2, 0, 0, 0, 0, 0, 0, 0);
    y = conv1d_f32(ctx, w2, y_padded, 1, 0, 1);
    ggml_tensor * b2_r = ggml_reshape_2d(ctx, b2, 1, D);
    y = ggml_add(ctx, y, b2_r);


    ggml_tensor * y_transposed = ggml_cont(ctx, ggml_permute(ctx, y, 1, 0, 2, 3));
    ggml_tensor * out = ggml_add(ctx, y_transposed, x_in);
    ggml_set_name(out, "out");
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x_in"), npy_as_f32(input_npy), 0, input_npy.data.size());
    ggml_backend_graph_compute(m.backend, gf);

    std::vector<float> out_data(ggml_nelements(out));
    ggml_backend_tensor_get(out, out_data.data(), 0, ggml_nbytes(out));

    auto s = compare_f32(out_data.data(), npy_as_f32(exp_npy), std::min(out_data.size(), exp_npy.n_elements()));
    print_compare("pre_lookahead", s);
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
}


struct conformer_block_weights {
    ggml_tensor * norm_mha_w, * norm_mha_b;
    ggml_tensor * norm_ff_w,  * norm_ff_b;

    ggml_tensor * q_w, * q_b;
    ggml_tensor * k_w, * k_b;
    ggml_tensor * v_w, * v_b;
    ggml_tensor * o_w, * o_b;
    ggml_tensor * pos_w;
    ggml_tensor * pos_bias_u;
    ggml_tensor * pos_bias_v;

    ggml_tensor * ff1_w, * ff1_b;
    ggml_tensor * ff2_w, * ff2_b;
};

static conformer_block_weights load_conformer_block(const model_ctx & m, const std::string & prefix) {
    conformer_block_weights w;
    w.norm_mha_w = find_tensor(m, prefix + "/norm_mha/w");
    w.norm_mha_b = find_tensor(m, prefix + "/norm_mha/b");
    w.norm_ff_w  = find_tensor(m, prefix + "/norm_ff/w");
    w.norm_ff_b  = find_tensor(m, prefix + "/norm_ff/b");
    w.q_w = find_tensor(m, prefix + "/attn/q/w");
    w.q_b = find_tensor(m, prefix + "/attn/q/b");
    w.k_w = find_tensor(m, prefix + "/attn/k/w");
    w.k_b = find_tensor(m, prefix + "/attn/k/b");
    w.v_w = find_tensor(m, prefix + "/attn/v/w");
    w.v_b = find_tensor(m, prefix + "/attn/v/b");
    w.o_w = find_tensor(m, prefix + "/attn/o/w");
    w.o_b = find_tensor(m, prefix + "/attn/o/b");
    w.pos_w = find_tensor(m, prefix + "/attn/pos/w");
    w.pos_bias_u = find_tensor(m, prefix + "/attn/pos_bias_u");
    w.pos_bias_v = find_tensor(m, prefix + "/attn/pos_bias_v");
    w.ff1_w = find_tensor(m, prefix + "/ff/w1/w");
    w.ff1_b = find_tensor(m, prefix + "/ff/w1/b");
    w.ff2_w = find_tensor(m, prefix + "/ff/w2/w");
    w.ff2_b = find_tensor(m, prefix + "/ff/w2/b");
    return w;
}


static ggml_tensor * conformer_block(ggml_context * ctx,
                                     const conformer_block_weights & w,
                                     ggml_tensor * x,
                                     ggml_tensor * pos_emb,
                                     int D, int T, int H, int HEAD_DIM,
                                     float eps = 1e-12f) {

    ggml_tensor * residual = x;
    ggml_tensor * xn = ggml_norm(ctx, x, eps);
    xn = ggml_add(ctx, ggml_mul(ctx, xn, w.norm_mha_w), w.norm_mha_b);


    ggml_tensor * q = ggml_add(ctx, ggml_mul_mat(ctx, w.q_w, xn), w.q_b);
    ggml_tensor * k = ggml_add(ctx, ggml_mul_mat(ctx, w.k_w, xn), w.k_b);
    ggml_tensor * v = ggml_add(ctx, ggml_mul_mat(ctx, w.v_w, xn), w.v_b);
    ggml_tensor * p = ggml_mul_mat(ctx, w.pos_w, pos_emb);


    q = ggml_reshape_3d(ctx, q, HEAD_DIM, H, T);
    k = ggml_reshape_3d(ctx, k, HEAD_DIM, H, T);
    v = ggml_reshape_3d(ctx, v, HEAD_DIM, H, T);
    p = ggml_reshape_3d(ctx, p, HEAD_DIM, H, pos_emb->ne[1]);


    ggml_tensor * q_perm = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    ggml_tensor * k_perm = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    ggml_tensor * v_perm = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
    ggml_tensor * p_perm = ggml_cont(ctx, ggml_permute(ctx, p, 0, 2, 1, 3));


    ggml_tensor * u_bias = ggml_reshape_3d(ctx, w.pos_bias_u, HEAD_DIM, 1, H);
    ggml_tensor * v_bias = ggml_reshape_3d(ctx, w.pos_bias_v, HEAD_DIM, 1, H);

    ggml_tensor * q_plus_u = ggml_add(ctx, q_perm, u_bias);
    ggml_tensor * q_plus_v = ggml_add(ctx, q_perm, v_bias);


    ggml_tensor * ac = ggml_mul_mat(ctx, k_perm, q_plus_u);

    ggml_tensor * bd = ggml_mul_mat(ctx, p_perm, q_plus_v);


    ggml_tensor * bd_padded = ggml_pad_ext(ctx, bd, 1, 0, 0, 0, 0, 0, 0, 0);
    ggml_tensor * bd_viewed = ggml_reshape_3d(ctx, bd_padded, T, 2*T, H);

    ggml_tensor * bd_sliced = ggml_view_3d(ctx, bd_viewed,
                                           T, 2*T - 1, H,
                                           bd_viewed->nb[1],
                                           bd_viewed->nb[2],
                                           bd_viewed->nb[1]);
    ggml_tensor * bd_reshaped = ggml_reshape_3d(ctx, ggml_cont(ctx, bd_sliced),
                                                2*T - 1, T, H);
    ggml_tensor * bd_final = ggml_view_3d(ctx, bd_reshaped,
                                          T, T, H,
                                          bd_reshaped->nb[1], bd_reshaped->nb[2], 0);
    bd_final = ggml_cont(ctx, bd_final);


    ggml_tensor * scores = ggml_add(ctx, ac, bd_final);
    scores = ggml_scale(ctx, scores, 1.0f / std::sqrt((float)HEAD_DIM));


    ggml_tensor * attn = ggml_soft_max(ctx, scores);


    ggml_tensor * v_for_mm = ggml_cont(ctx, ggml_permute(ctx, v_perm, 1, 0, 2, 3));
    ggml_tensor * attn_v = ggml_mul_mat(ctx, v_for_mm, attn);


    ggml_tensor * heads_merged = ggml_cont(ctx, ggml_permute(ctx, attn_v, 0, 2, 1, 3));
    ggml_tensor * heads_flat = ggml_reshape_2d(ctx, heads_merged, HEAD_DIM * H, T);


    ggml_tensor * attn_out = ggml_add(ctx, ggml_mul_mat(ctx, w.o_w, heads_flat), w.o_b);

    x = ggml_add(ctx, residual, attn_out);


    residual = x;
    xn = ggml_norm(ctx, x, eps);
    xn = ggml_add(ctx, ggml_mul(ctx, xn, w.norm_ff_w), w.norm_ff_b);

    ggml_tensor * ff = ggml_add(ctx, ggml_mul_mat(ctx, w.ff1_w, xn), w.ff1_b);
    ff = ggml_silu(ctx, ff);
    ff = ggml_add(ctx, ggml_mul_mat(ctx, w.ff2_w, ff), w.ff2_b);

    x = ggml_add(ctx, residual, ff);
    return x;
}


static void stage_E0(const model_ctx & m, const std::string & ref_dir) {
    fprintf(stderr, "\n=== Stage E0: block 0 sub-pieces ===\n");
    npy_array x_npy = npy_load(ref_dir + "/pre_lookahead.npy");
    int T = (int)x_npy.shape[0];
    int D = (int)x_npy.shape[1];

    static size_t buf_size = ggml_tensor_overhead()*128 + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    auto check = [&](const std::string & name, std::function<ggml_tensor*(ggml_context*, ggml_tensor*)> build_fn, const std::string & exp_file) {
        ggml_init_params gp = { buf_size, buf.data(), true };
        ggml_context * ctx = ggml_init(gp);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, T);
        ggml_set_name(x, "x"); ggml_set_input(x);
        ggml_tensor * y = build_fn(ctx, x);
        ggml_set_name(y, "out"); ggml_set_output(y);
        ggml_build_forward_expand(gf, y);

        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
        ggml_gallocr_reserve(allocr, gf);
        ggml_gallocr_alloc_graph(allocr, gf);
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x"), npy_as_f32(x_npy), 0, x_npy.data.size());
        ggml_backend_graph_compute(m.backend, gf);

        std::vector<float> out(ggml_nelements(y));
        ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));

        npy_array exp = npy_load(ref_dir + "/" + exp_file);
        auto s = compare_f32(out.data(), npy_as_f32(exp), std::min(out.size(), exp.n_elements()));
        print_compare(name.c_str(), s);
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
    };

    auto w = load_conformer_block(m, "flow/encoder/block0");

    check("norm_mha", [&](ggml_context * ctx, ggml_tensor * x) {
        ggml_tensor * y = ggml_norm(ctx, x, 1e-12f);
        y = ggml_add(ctx, ggml_mul(ctx, y, w.norm_mha_w), w.norm_mha_b);
        return y;
    }, "b0_norm_mha.npy");

    check("q_proj", [&](ggml_context * ctx, ggml_tensor * x) {
        ggml_tensor * y = ggml_norm(ctx, x, 1e-12f);
        y = ggml_add(ctx, ggml_mul(ctx, y, w.norm_mha_w), w.norm_mha_b);
        y = ggml_add(ctx, ggml_mul_mat(ctx, w.q_w, y), w.q_b);
        return y;
    }, "b0_q.npy");

    check("k_proj", [&](ggml_context * ctx, ggml_tensor * x) {
        ggml_tensor * y = ggml_norm(ctx, x, 1e-12f);
        y = ggml_add(ctx, ggml_mul(ctx, y, w.norm_mha_w), w.norm_mha_b);
        y = ggml_add(ctx, ggml_mul_mat(ctx, w.k_w, y), w.k_b);
        return y;
    }, "b0_k.npy");
}


static void stage_E(const model_ctx & m, const std::string & ref_dir) {
    fprintf(stderr, "\n=== Stage E: Conformer block 0 ===\n");
    npy_array x_npy      = npy_load(ref_dir + "/pre_lookahead.npy");
    npy_array pos_npy    = npy_load(ref_dir + "/encoder_embed_tup1.npy");
    npy_array expect_npy = npy_load(ref_dir + "/enc_block0_tup0.npy");
    int T = (int)x_npy.shape[0];
    int D = (int)x_npy.shape[1];
    int L = (int)pos_npy.shape[0];
    int H = 8, HEAD_DIM = 64;
    fprintf(stderr, "  T=%d D=%d L=%d H=%d HD=%d\n", T, D, L, H, HEAD_DIM);

    static size_t buf_size = ggml_tensor_overhead()*1024 + ggml_graph_overhead_custom(2048, false);
    static std::vector<uint8_t> buf(buf_size);
    ggml_init_params gp = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 2048, false);

    ggml_tensor * x_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, T);
    ggml_set_name(x_in, "x_in"); ggml_set_input(x_in);
    ggml_tensor * pos_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, L);
    ggml_set_name(pos_in, "pos_in"); ggml_set_input(pos_in);

    auto w = load_conformer_block(m, "flow/encoder/block0");
    ggml_tensor * out = conformer_block(ctx, w, x_in, pos_in, D, T, H, HEAD_DIM);
    ggml_set_name(out, "out"); ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x_in"), npy_as_f32(x_npy), 0, x_npy.data.size());
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos_in"), npy_as_f32(pos_npy), 0, pos_npy.data.size());
    ggml_backend_graph_compute(m.backend, gf);

    std::vector<float> out_data(ggml_nelements(out));
    ggml_backend_tensor_get(out, out_data.data(), 0, ggml_nbytes(out));

    auto s = compare_f32(out_data.data(), npy_as_f32(expect_npy), std::min(out_data.size(), expect_npy.n_elements()));
    print_compare("enc_block0_out", s);
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
}


static void compute_pos_emb(std::vector<float> & pe, int T, int D) {
    int L = 2 * T - 1;
    pe.assign(L * D, 0.0f);
    const float log10000 = std::log(10000.0f);
    std::vector<float> div_term(D / 2);
    for (int i = 0; i < D / 2; ++i) div_term[i] = std::exp(-((float)(2*i) * log10000 / (float)D));

    std::vector<std::vector<float>> pos_pe(T, std::vector<float>(D, 0.0f));
    std::vector<std::vector<float>> neg_pe(T, std::vector<float>(D, 0.0f));
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < D / 2; ++i) {
            float ap = (float)t * div_term[i];
            pos_pe[t][2*i]     = std::sin(ap);
            pos_pe[t][2*i + 1] = std::cos(ap);
            neg_pe[t][2*i]     = std::sin(-ap);
            neg_pe[t][2*i + 1] = std::cos(-ap);
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


static void stage_F(const model_ctx & m, const std::string & ref_dir) {
    fprintf(stderr, "\n=== Stage F: full encoder + encoder_proj ===\n");
    npy_array inp_npy       = npy_load(ref_dir + "/input_embedded.npy");
    npy_array expect_out    = npy_load(ref_dir + "/after_norm.npy");
    npy_array expect_proj   = npy_load(ref_dir + "/encoder_proj.npy");
    int T = (int)inp_npy.shape[0];
    int D = (int)inp_npy.shape[1];
    int H = 8, HEAD_DIM = 64;
    int T2 = 2 * T;
    fprintf(stderr, "  T=%d D=%d T2=%d\n", T, D, T2);

    static size_t buf_size = ggml_tensor_overhead()*4096 + ggml_graph_overhead_custom(16384, false);
    static std::vector<uint8_t> buf(buf_size);
    ggml_init_params gp = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);

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
    xt = ggml_pad_ext(ctx, xt, 0, 3, 0, 0, 0, 0, 0, 0);
    xt = conv1d_f32(ctx, pw1, xt, 1, 0, 1);
    xt = ggml_add(ctx, xt, ggml_reshape_2d(ctx, pb1, 1, D));
    xt = ggml_leaky_relu(ctx, xt, 0.01f, false);
    xt = ggml_pad_ext(ctx, xt, 2, 0, 0, 0, 0, 0, 0, 0);
    xt = conv1d_f32(ctx, pw2, xt, 1, 0, 1);
    xt = ggml_add(ctx, xt, ggml_reshape_2d(ctx, pb2, 1, D));
    xt = ggml_cont(ctx, ggml_permute(ctx, xt, 1, 0, 2, 3));
    x = ggml_add(ctx, xt, residual);


    for (int i = 0; i < 6; ++i) {
        auto w = load_conformer_block(m, "flow/encoder/block" + std::to_string(i));
        x = conformer_block(ctx, w, x, pos1, D, T, H, HEAD_DIM);
    }


    ggml_set_name(x, "enc_block5"); ggml_set_output(x);


    ggml_tensor * up_w = find_tensor(m, "flow/encoder/up_layer/conv/w");
    ggml_tensor * up_b = find_tensor(m, "flow/encoder/up_layer/conv/b");


    ggml_tensor * xu = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    ggml_tensor * xu_3d = ggml_reshape_3d(ctx, xu, 1, xu->ne[0], xu->ne[1]);
    ggml_tensor * xu_2x = ggml_concat(ctx, xu_3d, xu_3d, 0);
    xu = ggml_cont(ctx, ggml_reshape_2d(ctx, xu_2x, xu_3d->ne[1]*2, xu_3d->ne[2]));

    xu = ggml_pad_ext(ctx, xu, 4, 0, 0, 0, 0, 0, 0, 0);
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
        auto w = load_conformer_block(m, "flow/encoder/up_block" + std::to_string(i));
        x = conformer_block(ctx, w, x, pos2, D, T2, H, HEAD_DIM);
    }


    ggml_tensor * anw = find_tensor(m, "flow/encoder/after_norm/w");
    ggml_tensor * anb = find_tensor(m, "flow/encoder/after_norm/b");
    x = ggml_norm(ctx, x, 1e-5f);
    x = ggml_add(ctx, ggml_mul(ctx, x, anw), anb);
    ggml_set_name(x, "encoder_out"); ggml_set_output(x);


    ggml_tensor * epw = find_tensor(m, "flow/encoder_proj/w");
    ggml_tensor * epb = find_tensor(m, "flow/encoder_proj/b");
    ggml_tensor * mu = ggml_add(ctx, ggml_mul_mat(ctx, epw, x), epb);
    ggml_set_name(mu, "mu"); ggml_set_output(mu);

    ggml_build_forward_expand(gf, mu);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x_in"), npy_as_f32(inp_npy), 0, inp_npy.data.size());


    std::vector<float> pe1, pe2;
    compute_pos_emb(pe1, T, D);
    compute_pos_emb(pe2, T2, D);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos1"), pe1.data(), 0, pe1.size()*sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos2"), pe2.data(), 0, pe2.size()*sizeof(float));

    ggml_backend_graph_compute(m.backend, gf);


    ggml_tensor * b5 = ggml_graph_get_tensor(gf, "enc_block5");
    if (b5) {
        std::vector<float> b5d(ggml_nelements(b5));
        ggml_backend_tensor_get(b5, b5d.data(), 0, ggml_nbytes(b5));
        npy_array exp5 = npy_load(ref_dir + "/enc_block5_tup0.npy");
        auto s5 = compare_f32(b5d.data(), npy_as_f32(exp5), std::min(b5d.size(), exp5.n_elements()));
        print_compare("enc_block5", s5);
    }

    std::vector<float> enc_out_data(ggml_nelements(x));
    ggml_backend_tensor_get(x, enc_out_data.data(), 0, ggml_nbytes(x));
    auto s1 = compare_f32(enc_out_data.data(), npy_as_f32(expect_out), std::min(enc_out_data.size(), expect_out.n_elements()));
    print_compare("encoder_out (after_norm)", s1);

    std::vector<float> mu_data(ggml_nelements(mu));
    ggml_backend_tensor_get(mu, mu_data.data(), 0, ggml_nbytes(mu));
    auto s2 = compare_f32(mu_data.data(), npy_as_f32(expect_proj), std::min(mu_data.size(), expect_proj.n_elements()));
    print_compare("encoder_proj (mu)", s2);

    ggml_gallocr_free(allocr);
    ggml_free(ctx);
}


static void compute_t_sin_emb(float t_val, int dim, float scale, std::vector<float> & out) {
    int half = dim / 2;
    out.assign(dim, 0.0f);
    float log_factor = std::log(10000.0f) / (float)(half - 1);
    for (int i = 0; i < half; ++i) {
        float freq = std::exp(-(float)i * log_factor);
        float arg = scale * t_val * freq;
        out[i] = std::sin(arg);
        out[i + half] = std::cos(arg);
    }
}

static void stage_G1(const model_ctx & m, const std::string & ref_dir) {
    fprintf(stderr, "\n=== Stage G1: CFM time embedding (step 0) ===\n");
    npy_array t_npy      = npy_load(ref_dir + "/cfm_step0_t.npy");
    npy_array r_npy      = npy_load(ref_dir + "/cfm_step0_r.npy");

    npy_array t_sin_exp  = npy_load(ref_dir + "/cfm_t_sinemb_call0.npy");
    npy_array r_sin_exp  = npy_load(ref_dir + "/cfm_t_sinemb_call1.npy");
    npy_array t_mlp_exp  = npy_load(ref_dir + "/cfm_t_mlp_call0.npy");
    npy_array r_mlp_exp  = npy_load(ref_dir + "/cfm_t_mlp_call1.npy");
    npy_array t_mix_exp  = npy_load(ref_dir + "/cfm_t_mix_call0.npy");

    float t_val = ((const float*)t_npy.data.data())[0];
    float r_val = ((const float*)r_npy.data.data())[0];
    fprintf(stderr, "  t=%g r=%g\n", t_val, r_val);

    const int TDIM = 320;
    std::vector<float> t_sin(TDIM), r_sin(TDIM);
    compute_t_sin_emb(t_val, TDIM, 1000.0f, t_sin);
    compute_t_sin_emb(r_val, TDIM, 1000.0f, r_sin);

    auto s1 = compare_f32(t_sin.data(), npy_as_f32(t_sin_exp), t_sin.size());
    print_compare("t_sinemb", s1);
    auto s2 = compare_f32(r_sin.data(), npy_as_f32(r_sin_exp), r_sin.size());
    print_compare("r_sinemb", s2);


    static size_t buf_size = ggml_tensor_overhead()*64 + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);
    ggml_init_params gp = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * t_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, TDIM);
    ggml_set_name(t_in, "t_in"); ggml_set_input(t_in);
    ggml_tensor * r_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, TDIM);
    ggml_set_name(r_in, "r_in"); ggml_set_input(r_in);

    ggml_tensor * l1_w = find_tensor(m, "cfm/time_mlp/linear_1/weight");
    ggml_tensor * l1_b = find_tensor(m, "cfm/time_mlp/linear_1/bias");
    ggml_tensor * l2_w = find_tensor(m, "cfm/time_mlp/linear_2/weight");
    ggml_tensor * l2_b = find_tensor(m, "cfm/time_mlp/linear_2/bias");

    auto time_mlp = [&](ggml_tensor * x) {
        ggml_tensor * y = ggml_add(ctx, ggml_mul_mat(ctx, l1_w, x), l1_b);
        y = ggml_silu(ctx, y);
        y = ggml_add(ctx, ggml_mul_mat(ctx, l2_w, y), l2_b);
        return y;
    };
    ggml_tensor * t_mlp_out = time_mlp(t_in);
    ggml_tensor * r_mlp_out = time_mlp(r_in);


    ggml_tensor * concatted = ggml_concat(ctx, t_mlp_out, r_mlp_out, 0);

    ggml_tensor * mix_w = find_tensor(m, "cfm/time_embed_mixer/weight");
    ggml_tensor * mixed = ggml_mul_mat(ctx, mix_w, concatted);

    ggml_set_name(t_mlp_out, "t_mlp_out"); ggml_set_output(t_mlp_out);
    ggml_set_name(r_mlp_out, "r_mlp_out"); ggml_set_output(r_mlp_out);
    ggml_set_name(mixed, "mixed"); ggml_set_output(mixed);
    ggml_build_forward_expand(gf, mixed);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "t_in"), t_sin.data(), 0, t_sin.size()*sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "r_in"), r_sin.data(), 0, r_sin.size()*sizeof(float));
    ggml_backend_graph_compute(m.backend, gf);

    std::vector<float> t_mlp_out_data(ggml_nelements(t_mlp_out));
    ggml_backend_tensor_get(t_mlp_out, t_mlp_out_data.data(), 0, ggml_nbytes(t_mlp_out));
    auto s3 = compare_f32(t_mlp_out_data.data(), npy_as_f32(t_mlp_exp), t_mlp_out_data.size());
    print_compare("t_mlp", s3);

    std::vector<float> r_mlp_out_data(ggml_nelements(r_mlp_out));
    ggml_backend_tensor_get(r_mlp_out, r_mlp_out_data.data(), 0, ggml_nbytes(r_mlp_out));
    auto s4 = compare_f32(r_mlp_out_data.data(), npy_as_f32(r_mlp_exp), r_mlp_out_data.size());
    print_compare("r_mlp", s4);

    std::vector<float> mix_data(ggml_nelements(mixed));
    ggml_backend_tensor_get(mixed, mix_data.data(), 0, ggml_nbytes(mixed));
    auto s5 = compare_f32(mix_data.data(), npy_as_f32(t_mix_exp), std::min(mix_data.size(), t_mix_exp.n_elements()));
    print_compare("time_mixer", s5);

    ggml_gallocr_free(allocr);
    ggml_free(ctx);
}


static ggml_tensor * ggml_mish(ggml_context * ctx, ggml_tensor * x) {
    ggml_tensor * sp = ggml_unary(ctx, x, GGML_UNARY_OP_SOFTPLUS);
    ggml_tensor * th = ggml_unary(ctx, sp, GGML_UNARY_OP_TANH);
    return ggml_mul(ctx, x, th);
}


static ggml_tensor * layer_norm(ggml_context * ctx,
                                ggml_tensor * x,
                                ggml_tensor * w,
                                ggml_tensor * b,
                                float eps = 1e-5f) {
    ggml_tensor * y = ggml_norm(ctx, x, eps);
    y = ggml_mul(ctx, y, w);
    y = ggml_add(ctx, y, b);
    return y;
}


static ggml_tensor * layer_norm_on_channel(ggml_context * ctx,
                                           ggml_tensor * x,
                                           ggml_tensor * w,
                                           ggml_tensor * b,
                                           float eps = 1e-5f) {
    ggml_tensor * xt = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    xt = ggml_norm(ctx, xt, eps);
    xt = ggml_mul(ctx, xt, w);
    xt = ggml_add(ctx, xt, b);
    return ggml_cont(ctx, ggml_permute(ctx, xt, 1, 0, 2, 3));
}


static ggml_tensor * cfm_causal_block(ggml_context * ctx,
                                      ggml_tensor * x,
                                      ggml_tensor * conv_w,
                                      ggml_tensor * conv_b,
                                      ggml_tensor * ln_w,
                                      ggml_tensor * ln_b,
                                      int64_t C_out) {

    ggml_tensor * x_padded = ggml_pad_ext(ctx, x, 2, 0, 0, 0, 0, 0, 0, 0);

    ggml_tensor * y = conv1d_f32(ctx, conv_w, x_padded, 1, 0, 1);

    ggml_tensor * b_r = ggml_reshape_2d(ctx, conv_b, 1, C_out);
    y = ggml_add(ctx, y, b_r);

    y = layer_norm_on_channel(ctx, y, ln_w, ln_b);

    y = ggml_mish(ctx, y);
    return y;
}

struct cfm_resnet_w {
    ggml_tensor *b1_conv_w, *b1_conv_b, *b1_ln_w, *b1_ln_b;
    ggml_tensor *b2_conv_w, *b2_conv_b, *b2_ln_w, *b2_ln_b;
    ggml_tensor *mlp_w, *mlp_b;
    ggml_tensor *res_w, *res_b;
};

static cfm_resnet_w load_cfm_resnet(const model_ctx & m, const std::string & prefix) {
    cfm_resnet_w w;
    w.b1_conv_w = find_tensor(m, prefix + "/block1/block/0/weight");
    w.b1_conv_b = find_tensor(m, prefix + "/block1/block/0/bias");
    w.b1_ln_w   = find_tensor(m, prefix + "/block1/block/2/weight");
    w.b1_ln_b   = find_tensor(m, prefix + "/block1/block/2/bias");
    w.b2_conv_w = find_tensor(m, prefix + "/block2/block/0/weight");
    w.b2_conv_b = find_tensor(m, prefix + "/block2/block/0/bias");
    w.b2_ln_w   = find_tensor(m, prefix + "/block2/block/2/weight");
    w.b2_ln_b   = find_tensor(m, prefix + "/block2/block/2/bias");
    w.mlp_w     = find_tensor(m, prefix + "/mlp/1/weight");
    w.mlp_b     = find_tensor(m, prefix + "/mlp/1/bias");
    w.res_w     = find_tensor(m, prefix + "/res_conv/weight");
    w.res_b     = find_tensor(m, prefix + "/res_conv/bias");
    return w;
}


static ggml_tensor * cfm_resnet(ggml_context * ctx, const cfm_resnet_w & w,
                                ggml_tensor * x,
                                ggml_tensor * t_emb,
                                int64_t C_out) {
    ggml_tensor * h = cfm_causal_block(ctx, x, w.b1_conv_w, w.b1_conv_b, w.b1_ln_w, w.b1_ln_b, C_out);


    ggml_tensor * t_feat = ggml_mish(ctx, t_emb);
    ggml_tensor * t_proj = ggml_add(ctx, ggml_mul_mat(ctx, w.mlp_w, t_feat), w.mlp_b);

    ggml_tensor * t_proj_bc = ggml_reshape_2d(ctx, t_proj, 1, C_out);
    h = ggml_add(ctx, h, t_proj_bc);

    h = cfm_causal_block(ctx, h, w.b2_conv_w, w.b2_conv_b, w.b2_ln_w, w.b2_ln_b, C_out);


    ggml_tensor * res = conv1d_f32(ctx, w.res_w, x, 1, 0, 1);
    ggml_tensor * res_b_r = ggml_reshape_2d(ctx, w.res_b, 1, C_out);
    res = ggml_add(ctx, res, res_b_r);

    return ggml_add(ctx, h, res);
}


static void stage_G2(const model_ctx & m, const std::string & ref_dir) {
    fprintf(stderr, "\n=== Stage G2: CFM down_block[0] resnet (step 0) ===\n");
    npy_array x_npy    = npy_load(ref_dir + "/cfm_step0_x_in.npy");
    npy_array mu_npy   = npy_load(ref_dir + "/cfm_step0_mu.npy");
    npy_array spks_npy = npy_load(ref_dir + "/cfm_step0_spks.npy");
    npy_array cond_npy = npy_load(ref_dir + "/cfm_step0_cond.npy");
    npy_array t_mix_exp = npy_load(ref_dir + "/cfm_t_mix_call0.npy");
    npy_array exp_npy  = npy_load(ref_dir + "/cfm_d0_rn_call0.npy");

    int MEL = 80;
    int T = (int)x_npy.shape[1];
    int CIN = 320;
    int COUT = 256;
    int TIME_EMB_DIM = 1024;
    fprintf(stderr, "  T=%d CIN=%d COUT=%d time_dim=%d\n", T, CIN, COUT, TIME_EMB_DIM);

    static size_t buf_size = ggml_tensor_overhead()*256 + ggml_graph_overhead_custom(1024, false);
    static std::vector<uint8_t> buf(buf_size);
    ggml_init_params gp = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1024, false);


    ggml_tensor * x_in    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, MEL);
    ggml_set_name(x_in, "x_in");    ggml_set_input(x_in);
    ggml_tensor * mu_in   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, MEL);
    ggml_set_name(mu_in, "mu_in");  ggml_set_input(mu_in);
    ggml_tensor * spks_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, MEL);
    ggml_set_name(spks_in, "spks_in"); ggml_set_input(spks_in);
    ggml_tensor * cond_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, MEL);
    ggml_set_name(cond_in, "cond_in"); ggml_set_input(cond_in);
    ggml_tensor * t_emb_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, TIME_EMB_DIM);
    ggml_set_name(t_emb_in, "t_emb"); ggml_set_input(t_emb_in);

    ggml_tensor * spks_r = ggml_reshape_2d(ctx, spks_in, 1, MEL);
    ggml_tensor * spks_bc = ggml_repeat(ctx, spks_r, x_in);
    ggml_tensor * xc = ggml_concat(ctx, x_in, mu_in, 1);
    xc = ggml_concat(ctx, xc, spks_bc, 1);
    xc = ggml_concat(ctx, xc, cond_in, 1);

    auto rn_w = load_cfm_resnet(m, "cfm/down_blocks/0/0");


    ggml_tensor * xpad = ggml_pad_ext(ctx, xc, 2, 0, 0, 0, 0, 0, 0, 0);
    ggml_tensor * h_conv = conv1d_f32(ctx, rn_w.b1_conv_w, xpad, 1, 0, 1);
    h_conv = ggml_add(ctx, h_conv, ggml_reshape_2d(ctx, rn_w.b1_conv_b, 1, COUT));
    ggml_set_name(h_conv, "h_conv"); ggml_set_output(h_conv);
    ggml_tensor * h_ln = layer_norm_on_channel(ctx, h_conv, rn_w.b1_ln_w, rn_w.b1_ln_b);
    ggml_set_name(h_ln, "h_ln"); ggml_set_output(h_ln);
    ggml_tensor * h_b1 = ggml_mish(ctx, h_ln);
    ggml_set_name(h_b1, "h_b1"); ggml_set_output(h_b1);

    ggml_tensor * t_feat = ggml_mish(ctx, t_emb_in);
    ggml_tensor * t_proj = ggml_add(ctx, ggml_mul_mat(ctx, rn_w.mlp_w, t_feat), rn_w.mlp_b);
    ggml_set_name(t_proj, "t_proj"); ggml_set_output(t_proj);

    ggml_tensor * t_proj_bc = ggml_reshape_2d(ctx, t_proj, 1, COUT);
    ggml_tensor * h_with_t = ggml_add(ctx, h_b1, t_proj_bc);

    ggml_tensor * h_b2 = cfm_causal_block(ctx, h_with_t, rn_w.b2_conv_w, rn_w.b2_conv_b, rn_w.b2_ln_w, rn_w.b2_ln_b, COUT);
    ggml_set_name(h_b2, "h_b2"); ggml_set_output(h_b2);

    ggml_tensor * res = conv1d_f32(ctx, rn_w.res_w, xc, 1, 0, 1);
    ggml_tensor * res_b_r = ggml_reshape_2d(ctx, rn_w.res_b, 1, COUT);
    res = ggml_add(ctx, res, res_b_r);
    ggml_set_name(res, "res"); ggml_set_output(res);

    ggml_tensor * rn_out = ggml_add(ctx, h_b2, res);
    ggml_set_name(rn_out, "out"); ggml_set_output(rn_out);
    ggml_build_forward_expand(gf, rn_out);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x_in"),    npy_as_f32(x_npy),    0, x_npy.data.size());
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mu_in"),   npy_as_f32(mu_npy),   0, mu_npy.data.size());
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "spks_in"), npy_as_f32(spks_npy), 0, spks_npy.data.size());
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "cond_in"), npy_as_f32(cond_npy), 0, cond_npy.data.size());
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "t_emb"),   npy_as_f32(t_mix_exp), 0, t_mix_exp.data.size());

    ggml_backend_graph_compute(m.backend, gf);

    auto check_stage = [&](const char * name, ggml_tensor * t, const std::string & ref_file) {
        npy_array exp = npy_load(ref_dir + "/" + ref_file);
        std::vector<float> data(ggml_nelements(t));
        ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
        auto s = compare_f32(data.data(), npy_as_f32(exp), std::min(data.size(), exp.n_elements()));
        print_compare(name, s);
    };
    {
        npy_array exp = npy_load(ref_dir + "/cfm_concat.npy");
        std::vector<float> data(ggml_nelements(xc));
        ggml_backend_tensor_get(xc, data.data(), 0, ggml_nbytes(xc));
        fprintf(stderr, "  got xc[0..5]: %.4f %.4f %.4f %.4f %.4f\n",
                data[0], data[1], data[2], data[3], data[4]);
        const float * ref = npy_as_f32(exp);
        fprintf(stderr, "  ref xc[0..5]: %.4f %.4f %.4f %.4f %.4f\n",
                ref[0], ref[1], ref[2], ref[3], ref[4]);


        size_t mismatch_at = data.size();
        for (size_t i = 0; i < std::min(data.size(), exp.n_elements()); ++i) {
            if (std::fabs(data[i] - ref[i]) > 1e-4) { mismatch_at = i; break; }
        }
        fprintf(stderr, "  first mismatch at index %zu (channel=%zu t=%zu)\n",
                mismatch_at, mismatch_at / T, mismatch_at % T);
        if (mismatch_at < data.size()) {
            fprintf(stderr, "    got: %.4f  ref: %.4f\n", data[mismatch_at], ref[mismatch_at]);
        }
        auto s = compare_f32(data.data(), ref, std::min(data.size(), exp.n_elements()));
        print_compare("xc", s);
    }
    check_stage("h_conv", h_conv, "cfm_h_conv.npy");
    {
        npy_array exp = npy_load(ref_dir + "/cfm_h_ln.npy");
        std::vector<float> data(ggml_nelements(h_ln));
        ggml_backend_tensor_get(h_ln, data.data(), 0, ggml_nbytes(h_ln));
        const float * ref = npy_as_f32(exp);
        fprintf(stderr, "  got h_ln[0..5]: %.4f %.4f %.4f %.4f %.4f\n",
                data[0], data[1], data[2], data[3], data[4]);
        fprintf(stderr, "  ref h_ln[0..5]: %.4f %.4f %.4f %.4f %.4f\n",
                ref[0], ref[1], ref[2], ref[3], ref[4]);
        fprintf(stderr, "  got h_ln[T..T+4]: %.4f %.4f %.4f %.4f %.4f\n",
                data[T], data[T+1], data[T+2], data[T+3], data[T+4]);
        fprintf(stderr, "  ref h_ln[T..T+4]: %.4f %.4f %.4f %.4f %.4f\n",
                ref[T], ref[T+1], ref[T+2], ref[T+3], ref[T+4]);
        auto s = compare_f32(data.data(), ref, std::min(data.size(), exp.n_elements()));
        print_compare("h_ln", s);
    }
    check_stage("h_b1", h_b1, "cfm_d0_rn_b1_call0.npy");
    check_stage("t_proj", t_proj, "cfm_d0_rn_mlp_call0.npy");
    check_stage("h_b2", h_b2, "cfm_d0_rn_b2_call0.npy");
    check_stage("res", res, "cfm_d0_rn_res_call0.npy");
    check_stage("cfm_resnet_out", rn_out, "cfm_d0_rn_call0.npy");

    ggml_gallocr_free(allocr);
    ggml_free(ctx);
}


struct basic_tfm_w {
    ggml_tensor *norm1_w, *norm1_b;
    ggml_tensor *to_q, *to_k, *to_v;
    ggml_tensor *to_out_w, *to_out_b;
    ggml_tensor *norm3_w, *norm3_b;
    ggml_tensor *ff0_w, *ff0_b;
    ggml_tensor *ff2_w, *ff2_b;
};

static basic_tfm_w load_basic_tfm(const model_ctx & m, const std::string & prefix) {
    basic_tfm_w w;
    w.norm1_w = find_tensor(m, prefix + "/norm1/weight");
    w.norm1_b = find_tensor(m, prefix + "/norm1/bias");
    w.to_q    = find_tensor(m, prefix + "/attn1/to_q/weight");
    w.to_k    = find_tensor(m, prefix + "/attn1/to_k/weight");
    w.to_v    = find_tensor(m, prefix + "/attn1/to_v/weight");
    w.to_out_w = find_tensor(m, prefix + "/attn1/to_out/0/weight");
    w.to_out_b = find_tensor(m, prefix + "/attn1/to_out/0/bias");
    w.norm3_w = find_tensor(m, prefix + "/norm3/weight");
    w.norm3_b = find_tensor(m, prefix + "/norm3/bias");
    w.ff0_w  = find_tensor(m, prefix + "/ff/net/0/proj/weight");
    w.ff0_b  = find_tensor(m, prefix + "/ff/net/0/proj/bias");
    w.ff2_w  = find_tensor(m, prefix + "/ff/net/2/weight");
    w.ff2_b  = find_tensor(m, prefix + "/ff/net/2/bias");
    return w;
}


static ggml_tensor * basic_tfm_forward(ggml_context * ctx, const basic_tfm_w & w,
                                       ggml_tensor * x, int T, int C,
                                       int H = 8, int HD = 64,
                                       ggml_tensor ** out_nx = nullptr,
                                       ggml_tensor ** out_attn = nullptr,
                                       ggml_tensor ** out_nx2 = nullptr,
                                       ggml_tensor ** out_ff = nullptr) {
    int INNER = H * HD;


    ggml_tensor * nx = layer_norm(ctx, x, w.norm1_w, w.norm1_b);
    if (out_nx) *out_nx = nx;

    ggml_tensor * q = ggml_mul_mat(ctx, w.to_q, nx);
    ggml_tensor * k = ggml_mul_mat(ctx, w.to_k, nx);
    ggml_tensor * v = ggml_mul_mat(ctx, w.to_v, nx);


    q = ggml_reshape_3d(ctx, q, HD, H, T);
    k = ggml_reshape_3d(ctx, k, HD, H, T);
    v = ggml_reshape_3d(ctx, v, HD, H, T);
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));


    ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    scores = ggml_scale(ctx, scores, 1.0f / std::sqrt((float)HD));
    ggml_tensor * attn = ggml_soft_max(ctx, scores);


    ggml_tensor * v_for_mm = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
    ggml_tensor * attn_v = ggml_mul_mat(ctx, v_for_mm, attn);


    ggml_tensor * merged = ggml_cont(ctx, ggml_permute(ctx, attn_v, 0, 2, 1, 3));
    ggml_tensor * flat = ggml_reshape_2d(ctx, merged, INNER, T);


    ggml_tensor * attn_out = ggml_add(ctx, ggml_mul_mat(ctx, w.to_out_w, flat), w.to_out_b);
    if (out_attn) *out_attn = attn_out;
    x = ggml_add(ctx, x, attn_out);


    ggml_tensor * nx2 = layer_norm(ctx, x, w.norm3_w, w.norm3_b);
    if (out_nx2) *out_nx2 = nx2;
    ggml_tensor * ff = ggml_add(ctx, ggml_mul_mat(ctx, w.ff0_w, nx2), w.ff0_b);
    ff = ggml_gelu_erf(ctx, ff);
    ff = ggml_add(ctx, ggml_mul_mat(ctx, w.ff2_w, ff), w.ff2_b);
    if (out_ff) *out_ff = ff;
    x = ggml_add(ctx, x, ff);
    return x;
}


static void stage_G3(const model_ctx & m, const std::string & ref_dir) {
    fprintf(stderr, "\n=== Stage G3: CFM BasicTransformerBlock ===\n");
    npy_array inp_npy = npy_load(ref_dir + "/cfm_d0_rn_call0.npy");
    npy_array exp_npy = npy_load(ref_dir + "/cfm_d0_t0_call0.npy");

    int T = (int)exp_npy.shape[0];
    int C = (int)exp_npy.shape[1];
    fprintf(stderr, "  T=%d C=%d\n", T, C);


    static size_t buf_size = ggml_tensor_overhead()*256 + ggml_graph_overhead_custom(1024, false);
    static std::vector<uint8_t> buf(buf_size);
    ggml_init_params gp = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1024, false);


    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, C);
    ggml_set_name(inp, "inp"); ggml_set_input(inp);


    ggml_tensor * x = ggml_cont(ctx, ggml_permute(ctx, inp, 1, 0, 2, 3));

    auto tw = load_basic_tfm(m, "cfm/down_blocks/0/1/0");
    ggml_tensor *nx, *attn, *nx2, *ff_out;
    ggml_tensor * out = basic_tfm_forward(ctx, tw, x, T, C, 8, 64, &nx, &attn, &nx2, &ff_out);
    ggml_set_name(nx, "nx"); ggml_set_output(nx);
    ggml_set_name(attn, "attn"); ggml_set_output(attn);
    ggml_set_name(nx2, "nx2"); ggml_set_output(nx2);
    ggml_set_name(ff_out, "ff_out"); ggml_set_output(ff_out);
    ggml_set_name(out, "out"); ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inp"), npy_as_f32(inp_npy), 0, inp_npy.data.size());
    ggml_backend_graph_compute(m.backend, gf);

    auto check_stage = [&](const char * name, ggml_tensor * t, const std::string & ref_file) {
        npy_array exp = npy_load(ref_dir + "/" + ref_file);
        std::vector<float> data(ggml_nelements(t));
        ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
        auto s = compare_f32(data.data(), npy_as_f32(exp), std::min(data.size(), exp.n_elements()));
        print_compare(name, s);
    };
    check_stage("nx (norm1)", nx, "cfm_d0_t0_n1_call0.npy");
    check_stage("attn_out", attn, "cfm_d0_t0_attn_call0.npy");
    check_stage("nx2 (norm3)", nx2, "cfm_d0_t0_n3_call0.npy");
    check_stage("ff_out", ff_out, "cfm_d0_t0_ff_call0.npy");
    check_stage("tfm_out", out, "cfm_d0_t0_call0.npy");

    ggml_gallocr_free(allocr);
    ggml_free(ctx);
}


struct cfm_tfm_stack {
    std::vector<basic_tfm_w> blocks;
};

static cfm_tfm_stack load_tfm_stack(const model_ctx & m, const std::string & prefix, int n) {
    cfm_tfm_stack s;
    for (int i = 0; i < n; ++i) {
        s.blocks.push_back(load_basic_tfm(m, prefix + "/" + std::to_string(i)));
    }
    return s;
}


static ggml_tensor * apply_tfm_stack(ggml_context * ctx, const cfm_tfm_stack & s,
                                     ggml_tensor * x, int T, int C) {

    ggml_tensor * xt = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    for (const auto & b : s.blocks) {
        xt = basic_tfm_forward(ctx, b, xt, T, C);
    }

    return ggml_cont(ctx, ggml_permute(ctx, xt, 1, 0, 2, 3));
}


static ggml_tensor * cfm_causal_conv_1d_k3(ggml_context * ctx,
                                           ggml_tensor * x,
                                           ggml_tensor * conv_w,
                                           ggml_tensor * conv_b,
                                           int C_out) {
    ggml_tensor * xp = ggml_pad_ext(ctx, x, 2, 0, 0, 0, 0, 0, 0, 0);
    ggml_tensor * y = conv1d_f32(ctx, conv_w, xp, 1, 0, 1);
    y = ggml_add(ctx, y, ggml_reshape_2d(ctx, conv_b, 1, C_out));
    return y;
}


static void stage_G4(const model_ctx & m, const std::string & ref_dir) {
    fprintf(stderr, "\n=== Stage G4: CFM decoder one step (full) ===\n");

    npy_array x_npy      = npy_load(ref_dir + "/cfm_step0_x_in.npy");
    npy_array mu_npy     = npy_load(ref_dir + "/cfm_step0_mu.npy");
    npy_array spks_npy   = npy_load(ref_dir + "/cfm_step0_spks.npy");
    npy_array cond_npy   = npy_load(ref_dir + "/cfm_step0_cond.npy");
    npy_array t_mix_exp  = npy_load(ref_dir + "/cfm_t_mix_call0.npy");
    npy_array dxdt_exp   = npy_load(ref_dir + "/cfm_step0_dxdt.npy");

    int MEL = 80;
    int T = (int)x_npy.shape[1];
    int CIN = 320, CH = 256;
    int TIME_EMB_DIM = 1024;
    int N_MID = 12, N_BLOCKS = 4;
    fprintf(stderr, "  T=%d CIN=%d CH=%d\n", T, CIN, CH);

    static size_t buf_size = ggml_tensor_overhead()*16384 + ggml_graph_overhead_custom(65536, false);
    static std::vector<uint8_t> buf(buf_size);
    ggml_init_params gp = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 65536, false);

    ggml_tensor * x_in    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, MEL);
    ggml_set_name(x_in, "x_in");    ggml_set_input(x_in);
    ggml_tensor * mu_in   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, MEL);
    ggml_set_name(mu_in, "mu_in");  ggml_set_input(mu_in);
    ggml_tensor * spks_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, MEL);
    ggml_set_name(spks_in, "spks_in"); ggml_set_input(spks_in);
    ggml_tensor * cond_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, MEL);
    ggml_set_name(cond_in, "cond_in"); ggml_set_input(cond_in);
    ggml_tensor * t_emb_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, TIME_EMB_DIM);
    ggml_set_name(t_emb_in, "t_emb"); ggml_set_input(t_emb_in);


    ggml_tensor * spks_bc = ggml_repeat(ctx, ggml_reshape_2d(ctx, spks_in, 1, MEL), x_in);
    ggml_tensor * xc = ggml_concat(ctx, x_in, mu_in, 1);
    xc = ggml_concat(ctx, xc, spks_bc, 1);
    xc = ggml_concat(ctx, xc, cond_in, 1);


    auto down_rn = load_cfm_resnet(m, "cfm/down_blocks/0/0");
    auto down_tfms = load_tfm_stack(m, "cfm/down_blocks/0/1", N_BLOCKS);
    ggml_tensor * down_conv_w = find_tensor(m, "cfm/down_blocks/0/2/weight");
    ggml_tensor * down_conv_b = find_tensor(m, "cfm/down_blocks/0/2/bias");

    ggml_tensor * x = cfm_resnet(ctx, down_rn, xc, t_emb_in, CH);
    ggml_tensor * skip = x;
    x = apply_tfm_stack(ctx, down_tfms, x, T, CH);

    ggml_tensor * hidden = x;
    x = cfm_causal_conv_1d_k3(ctx, x, down_conv_w, down_conv_b, CH);


    for (int i = 0; i < N_MID; ++i) {
        auto mid_rn = load_cfm_resnet(m, "cfm/mid_blocks/" + std::to_string(i) + "/0");
        auto mid_tfms = load_tfm_stack(m, "cfm/mid_blocks/" + std::to_string(i) + "/1", N_BLOCKS);
        x = cfm_resnet(ctx, mid_rn, x, t_emb_in, CH);
        x = apply_tfm_stack(ctx, mid_tfms, x, T, CH);
    }


    auto up_rn = load_cfm_resnet(m, "cfm/up_blocks/0/0");
    auto up_tfms = load_tfm_stack(m, "cfm/up_blocks/0/1", N_BLOCKS);
    ggml_tensor * up_conv_w = find_tensor(m, "cfm/up_blocks/0/2/weight");
    ggml_tensor * up_conv_b = find_tensor(m, "cfm/up_blocks/0/2/bias");


    x = ggml_concat(ctx, x, hidden, 1);
    x = cfm_resnet(ctx, up_rn, x, t_emb_in, CH);
    x = apply_tfm_stack(ctx, up_tfms, x, T, CH);
    x = cfm_causal_conv_1d_k3(ctx, x, up_conv_w, up_conv_b, CH);


    ggml_tensor * fb_conv_w = find_tensor(m, "cfm/final_block/block/0/weight");
    ggml_tensor * fb_conv_b = find_tensor(m, "cfm/final_block/block/0/bias");
    ggml_tensor * fb_ln_w   = find_tensor(m, "cfm/final_block/block/2/weight");
    ggml_tensor * fb_ln_b   = find_tensor(m, "cfm/final_block/block/2/bias");
    x = cfm_causal_block(ctx, x, fb_conv_w, fb_conv_b, fb_ln_w, fb_ln_b, CH);


    ggml_tensor * fp_w = find_tensor(m, "cfm/final_proj/weight");
    ggml_tensor * fp_b = find_tensor(m, "cfm/final_proj/bias");
    ggml_tensor * out = conv1d_f32(ctx, fp_w, x, 1, 0, 1);
    out = ggml_add(ctx, out, ggml_reshape_2d(ctx, fp_b, 1, MEL));

    (void)skip;

    ggml_set_name(out, "out"); ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x_in"),    npy_as_f32(x_npy),    0, x_npy.data.size());
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mu_in"),   npy_as_f32(mu_npy),   0, mu_npy.data.size());
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "spks_in"), npy_as_f32(spks_npy), 0, spks_npy.data.size());
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "cond_in"), npy_as_f32(cond_npy), 0, cond_npy.data.size());
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "t_emb"),   npy_as_f32(t_mix_exp), 0, t_mix_exp.data.size());

    fprintf(stderr, "  computing...\n");
    ggml_backend_graph_compute(m.backend, gf);

    std::vector<float> out_data(ggml_nelements(out));
    ggml_backend_tensor_get(out, out_data.data(), 0, ggml_nbytes(out));
    auto s = compare_f32(out_data.data(), npy_as_f32(dxdt_exp), std::min(out_data.size(), dxdt_exp.n_elements()));
    print_compare("cfm_step0_dxdt", s);

    ggml_gallocr_free(allocr);
    ggml_free(ctx);
}


static ggml_tensor * snake(ggml_context * ctx, ggml_tensor * x, ggml_tensor * alpha) {

    ggml_tensor * a = ggml_reshape_2d(ctx, alpha, 1, alpha->ne[0]);

    ggml_tensor * ax = ggml_mul(ctx, x, a);

    ggml_tensor * sin_ax = ggml_unary(ctx, ax, GGML_UNARY_OP_STEP);

    (void)sin_ax;

    ggml_tensor * s = ggml_sin(ctx, ax);

    ggml_tensor * s2 = ggml_mul(ctx, s, s);


    (void)s2;
    return x;
}


static ggml_tensor * snake_with_inv(ggml_context * ctx, ggml_tensor * x,
                                    ggml_tensor * alpha, ggml_tensor * inv_alpha) {
    ggml_tensor * a  = ggml_reshape_2d(ctx, alpha,     1, alpha->ne[0]);
    ggml_tensor * ia = ggml_reshape_2d(ctx, inv_alpha, 1, inv_alpha->ne[0]);
    ggml_tensor * ax = ggml_mul(ctx, x, a);
    ggml_tensor * s  = ggml_sin(ctx, ax);
    ggml_tensor * s2 = ggml_mul(ctx, s, s);
    ggml_tensor * t  = ggml_mul(ctx, s2, ia);
    return ggml_add(ctx, x, t);
}


static void stage_H1(const model_ctx & m, const std::string & ref_dir) {
    fprintf(stderr, "\n=== Stage H1: f0_predictor ===\n");
    npy_array mel_npy = npy_load(ref_dir + "/mel_output.npy");
    npy_array exp_npy = npy_load(ref_dir + "/hift_f0.npy");
    int MEL = 80;
    int T = (int)mel_npy.shape[1];
    fprintf(stderr, "  T=%d\n", T);

    static size_t buf_size = ggml_tensor_overhead()*128 + ggml_graph_overhead_custom(512, false);
    static std::vector<uint8_t> buf(buf_size);
    ggml_init_params gp = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 512, false);

    ggml_tensor * mel_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, MEL);
    ggml_set_name(mel_in, "mel_in"); ggml_set_input(mel_in);


    ggml_tensor * x = mel_in;
    int C = MEL;
    for (int i = 0; i < 5; ++i) {
        int layer_idx = i * 2;
        std::string pfx = "hift/f0_predictor/condnet/" + std::to_string(layer_idx);
        ggml_tensor * w = find_tensor(m, pfx + "/weight");
        ggml_tensor * b = find_tensor(m, pfx + "/bias");
        int C_out = (int)w->ne[2];

        ggml_tensor * xp = ggml_pad_ext(ctx, x, 1, 1, 0, 0, 0, 0, 0, 0);
        x = conv1d_f32(ctx, w, xp, 1, 0, 1);
        x = ggml_add(ctx, x, ggml_reshape_2d(ctx, b, 1, C_out));
        x = ggml_unary(ctx, x, GGML_UNARY_OP_ELU);
        C = C_out;
    }


    ggml_tensor * xp = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    ggml_tensor * cw = find_tensor(m, "hift/f0_predictor/classifier/weight");
    ggml_tensor * cb = find_tensor(m, "hift/f0_predictor/classifier/bias");
    ggml_tensor * y = ggml_mul_mat(ctx, cw, xp);
    y = ggml_add(ctx, y, cb);
    y = ggml_abs(ctx, y);

    y = ggml_reshape_1d(ctx, y, T);

    ggml_set_name(y, "out"); ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mel_in"), npy_as_f32(mel_npy), 0, mel_npy.data.size());
    ggml_backend_graph_compute(m.backend, gf);

    std::vector<float> out_data(ggml_nelements(y));
    ggml_backend_tensor_get(y, out_data.data(), 0, ggml_nbytes(y));
    auto s = compare_f32(out_data.data(), npy_as_f32(exp_npy), std::min(out_data.size(), exp_npy.n_elements()));
    print_compare("f0", s);

    ggml_gallocr_free(allocr);
    ggml_free(ctx);
}


static std::vector<float> invert_alpha(const model_ctx & m, const std::string & name) {
    ggml_tensor * t = find_tensor(m, name);
    std::vector<float> alpha(ggml_nelements(t));
    ggml_backend_tensor_get(t, alpha.data(), 0, ggml_nbytes(t));
    std::vector<float> inv(alpha.size());
    for (size_t i = 0; i < alpha.size(); ++i) inv[i] = 1.0f / (alpha[i] + 1e-9f);
    return inv;
}


struct resblock_w {
    struct pair_w {
        ggml_tensor *alpha1, *conv1_w, *conv1_b;
        ggml_tensor *alpha2, *conv2_w, *conv2_b;

    };
    std::vector<pair_w> pairs;
    std::vector<std::string> inv_alpha1_names;
    std::vector<std::string> inv_alpha2_names;
};

static resblock_w load_resblock(const model_ctx & m, const std::string & prefix, int n_pairs = 3) {
    resblock_w rb;
    for (int i = 0; i < n_pairs; ++i) {
        resblock_w::pair_w p;
        p.alpha1  = find_tensor(m, prefix + "/activations1/" + std::to_string(i) + "/alpha");
        p.conv1_w = find_tensor(m, prefix + "/convs1/" + std::to_string(i) + "/weight");
        p.conv1_b = find_tensor(m, prefix + "/convs1/" + std::to_string(i) + "/bias");
        p.alpha2  = find_tensor(m, prefix + "/activations2/" + std::to_string(i) + "/alpha");
        p.conv2_w = find_tensor(m, prefix + "/convs2/" + std::to_string(i) + "/weight");
        p.conv2_b = find_tensor(m, prefix + "/convs2/" + std::to_string(i) + "/bias");
        rb.pairs.push_back(p);
    }
    return rb;
}

static ggml_tensor * resblock_forward(ggml_context * ctx, const resblock_w & rb,
                                      ggml_tensor * x, int C,
                                      const std::vector<int> & dilations,
                                      int kernel_size,
                                      ggml_tensor ** inv_a1_tensors,
                                      ggml_tensor ** inv_a2_tensors) {
    for (size_t i = 0; i < rb.pairs.size(); ++i) {
        const auto & p = rb.pairs[i];
        int dilation = dilations[i];

        int pad1 = (kernel_size * dilation - dilation) / 2;
        int pad2 = (kernel_size - 1) / 2;

        ggml_tensor * xt = snake_with_inv(ctx, x, p.alpha1, inv_a1_tensors[i]);

        xt = ggml_pad_ext(ctx, xt, pad1, pad1, 0, 0, 0, 0, 0, 0);
        xt = conv1d_f32(ctx, p.conv1_w, xt, 1, 0, dilation);
        xt = ggml_add(ctx, xt, ggml_reshape_2d(ctx, p.conv1_b, 1, C));

        xt = snake_with_inv(ctx, xt, p.alpha2, inv_a2_tensors[i]);

        xt = ggml_pad_ext(ctx, xt, pad2, pad2, 0, 0, 0, 0, 0, 0);
        xt = conv1d_f32(ctx, p.conv2_w, xt, 1, 0, 1);
        xt = ggml_add(ctx, xt, ggml_reshape_2d(ctx, p.conv2_b, 1, C));
        x = ggml_add(ctx, x, xt);
    }
    return x;
}


static ggml_tensor * conv_transpose_1d_f32(ggml_context * ctx,
                                           ggml_tensor * kernel,
                                           ggml_tensor * input,
                                           int stride, int padding) {


    ggml_tensor * out = ggml_conv_transpose_1d(ctx, kernel, input, stride, 0, 1);
    if (padding == 0) return out;

    int64_t L_full = out->ne[0];
    int64_t L_new = L_full - 2 * padding;
    ggml_tensor * sliced = ggml_view_3d(ctx, out,
                                        L_new, out->ne[1], out->ne[2],
                                        out->nb[1], out->nb[2],
                                        (size_t)padding * out->nb[0]);
    return ggml_cont(ctx, sliced);
}


static void stage_H3(const model_ctx & m, const std::string & ref_dir) {
    fprintf(stderr, "\n=== Stage H3: HiFT decode body (conv_pre -> conv_post) ===\n");
    npy_array mel_npy     = npy_load(ref_dir + "/mel_output.npy");
    npy_array s_stft_npy  = npy_load(ref_dir + "/hift_s_stft.npy");
    npy_array exp_npy     = npy_load(ref_dir + "/hift_conv_post.npy");

    int MEL = 80;
    int T_mel = (int)mel_npy.shape[1];
    int T_stft = (int)s_stft_npy.shape[1];
    int NFFT2 = 18;
    int BASE_CH = 512;
    fprintf(stderr, "  T_mel=%d T_stft=%d\n", T_mel, T_stft);


    std::vector<int> ups_rates = {8, 5, 3};
    std::vector<int> ups_ksizes = {16, 11, 7};
    std::vector<int> ups_ch = {256, 128, 64};
    std::vector<int> rb_ksizes = {3, 7, 11};
    std::vector<std::vector<int>> rb_dilations = {{1,3,5}, {1,3,5}, {1,3,5}};
    std::vector<int> src_rb_ksizes = {7, 11};

    src_rb_ksizes = {7, 7, 11};
    std::vector<std::vector<int>> src_rb_dilations = {{1,3,5}, {1,3,5}, {1,3,5}};

    static size_t buf_size = 8*1024*1024;
    static std::vector<uint8_t> buf(buf_size);
    ggml_init_params gp = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 131072, false);

    ggml_tensor * mel_in    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_mel,  MEL);    ggml_set_name(mel_in, "mel_in"); ggml_set_input(mel_in);
    ggml_tensor * s_stft_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_stft, NFFT2);  ggml_set_name(s_stft_in, "s_stft_in"); ggml_set_input(s_stft_in);


    struct inv_alpha_entry {
        std::string graph_name;
        std::vector<float> data;
        ggml_tensor * tensor;
    };
    std::vector<inv_alpha_entry> inv_alphas;
    auto make_inv_alpha_input = [&](const std::string & name_prefix, int C) {
        std::string gn = "inv_" + name_prefix;
        std::vector<float> inv = invert_alpha(m, name_prefix);
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
        ggml_set_name(t, gn.c_str());
        ggml_set_input(t);
        inv_alphas.push_back({gn, std::move(inv), t});
        return t;
    };


    ggml_tensor * conv_pre_w = find_tensor(m, "hift/conv_pre/weight");
    ggml_tensor * conv_pre_b = find_tensor(m, "hift/conv_pre/bias");
    ggml_tensor * xp = ggml_pad_ext(ctx, mel_in, 3, 3, 0, 0, 0, 0, 0, 0);
    ggml_tensor * x = conv1d_f32(ctx, conv_pre_w, xp, 1, 0, 1);
    x = ggml_add(ctx, x, ggml_reshape_2d(ctx, conv_pre_b, 1, BASE_CH));
    ggml_set_name(x, "conv_pre_out"); ggml_set_output(x);


    for (int i = 0; i < 3; ++i) {

        x = ggml_leaky_relu(ctx, x, 0.1f, false);


        ggml_tensor * up_w = find_tensor(m, "hift/ups/" + std::to_string(i) + "/weight");
        ggml_tensor * up_b = find_tensor(m, "hift/ups/" + std::to_string(i) + "/bias");
        int stride = ups_rates[i];
        int k = ups_ksizes[i];
        int up_pad = (k - stride) / 2;
        (void)up_pad;


        x = conv_transpose_1d_f32(ctx, up_w, x, stride, up_pad);

        x = ggml_add(ctx, x, ggml_reshape_2d(ctx, up_b, 1, ups_ch[i]));
        if (i == 0) { ggml_set_name(x, "ups0_out"); ggml_set_output(x); }
        if (i == 1) { ggml_set_name(x, "ups1_out"); ggml_set_output(x); }
        if (i == 2) { ggml_set_name(x, "ups2_out"); ggml_set_output(x); }


        if (i == 2) {

            ggml_tensor * x_slice = ggml_view_3d(ctx, x,
                                                 1, x->ne[1], x->ne[2],
                                                 x->nb[1], x->nb[2],
                                                 1 * x->nb[0]);
            x_slice = ggml_cont(ctx, x_slice);
            x = ggml_concat(ctx, x_slice, x, 0);
        }


        ggml_tensor * sd_w = find_tensor(m, "hift/source_downs/" + std::to_string(i) + "/weight");
        ggml_tensor * sd_b = find_tensor(m, "hift/source_downs/" + std::to_string(i) + "/bias");


        int sd_k = (int)sd_w->ne[0];
        int sd_ic = (int)sd_w->ne[1];
        int sd_oc = (int)sd_w->ne[2];
        (void)sd_ic;


        int sd_stride, sd_pad;
        if (i == 0) { sd_stride = 15; sd_pad = 7; }
        else if (i == 1) { sd_stride = 3; sd_pad = 1; }
        else { sd_stride = 1; sd_pad = 0; }
        ggml_tensor * sd_in = ggml_pad_ext(ctx, s_stft_in, sd_pad, sd_pad, 0, 0, 0, 0, 0, 0);
        ggml_tensor * si = conv1d_f32(ctx, sd_w, sd_in, sd_stride, 0, 1);
        si = ggml_add(ctx, si, ggml_reshape_2d(ctx, sd_b, 1, sd_oc));
        if (i == 0) { ggml_set_name(si, "sd0_out"); ggml_set_output(si); }
        (void)sd_k;


        std::string srb_prefix = "hift/source_resblocks/" + std::to_string(i);
        auto srb = load_resblock(m, srb_prefix);
        ggml_tensor * srb_a1_invs[3], * srb_a2_invs[3];
        for (int j = 0; j < 3; ++j) {
            srb_a1_invs[j] = make_inv_alpha_input(srb_prefix + "/activations1/" + std::to_string(j) + "/alpha", ups_ch[i]);
            srb_a2_invs[j] = make_inv_alpha_input(srb_prefix + "/activations2/" + std::to_string(j) + "/alpha", ups_ch[i]);
        }
        si = resblock_forward(ctx, srb, si, ups_ch[i], src_rb_dilations[i], src_rb_ksizes[i], srb_a1_invs, srb_a2_invs);
        if (i == 0) { ggml_set_name(si, "sr0_out"); ggml_set_output(si); }


        x = ggml_add(ctx, x, si);


        ggml_tensor * xs = nullptr;
        for (int j = 0; j < 3; ++j) {
            int rb_idx = i * 3 + j;
            std::string rb_prefix = "hift/resblocks/" + std::to_string(rb_idx);
            auto rb = load_resblock(m, rb_prefix);
            ggml_tensor * rb_a1_invs[3], * rb_a2_invs[3];
            for (int k2 = 0; k2 < 3; ++k2) {
                rb_a1_invs[k2] = make_inv_alpha_input(rb_prefix + "/activations1/" + std::to_string(k2) + "/alpha", ups_ch[i]);
                rb_a2_invs[k2] = make_inv_alpha_input(rb_prefix + "/activations2/" + std::to_string(k2) + "/alpha", ups_ch[i]);
            }
            ggml_tensor * rb_out = resblock_forward(ctx, rb, x, ups_ch[i], rb_dilations[j], rb_ksizes[j], rb_a1_invs, rb_a2_invs);
            if (i == 0 && j == 0) { ggml_set_name(rb_out, "rb0_out"); ggml_set_output(rb_out); }
            if (xs == nullptr) xs = rb_out;
            else xs = ggml_add(ctx, xs, rb_out);
        }
        x = ggml_scale(ctx, xs, 1.0f / 3.0f);
    }


    x = ggml_leaky_relu(ctx, x, 0.01f, false);


    ggml_tensor * cp_w = find_tensor(m, "hift/conv_post/weight");
    ggml_tensor * cp_b = find_tensor(m, "hift/conv_post/bias");
    ggml_tensor * xp2 = ggml_pad_ext(ctx, x, 3, 3, 0, 0, 0, 0, 0, 0);
    x = conv1d_f32(ctx, cp_w, xp2, 1, 0, 1);
    x = ggml_add(ctx, x, ggml_reshape_2d(ctx, cp_b, 1, NFFT2));

    ggml_set_name(x, "out"); ggml_set_output(x);
    ggml_build_forward_expand(gf, x);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mel_in"), npy_as_f32(mel_npy), 0, mel_npy.data.size());
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "s_stft_in"), npy_as_f32(s_stft_npy), 0, s_stft_npy.data.size());

    for (const auto & ia : inv_alphas) {
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, ia.graph_name.c_str()),
                                ia.data.data(), 0, ia.data.size() * sizeof(float));
    }

    fprintf(stderr, "  computing (%zu 1/alpha inputs)...\n", inv_alphas.size());
    ggml_backend_graph_compute(m.backend, gf);

    std::vector<float> out_data(ggml_nelements(x));
    ggml_backend_tensor_get(x, out_data.data(), 0, ggml_nbytes(x));

    fprintf(stderr, "  out shape ne=[%lld, %lld] vs ref (%lld, %lld)\n",
            (long long)x->ne[0], (long long)x->ne[1], (long long)exp_npy.shape[0], (long long)exp_npy.shape[1]);

    auto check = [&](const char * name, const std::string & ne_name, const std::string & ref_file) {
        ggml_tensor * t = ggml_graph_get_tensor(gf, ne_name.c_str());
        if (!t) { fprintf(stderr, "  [%s] tensor not in graph\n", name); return; }
        npy_array exp = npy_load(ref_dir + "/" + ref_file);
        std::vector<float> data(ggml_nelements(t));
        ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
        fprintf(stderr, "  [%s] got shape ne=[%lld,%lld] ref=(%lld,%lld)\n",
                name, (long long)t->ne[0], (long long)t->ne[1],
                (long long)exp.shape[0], (long long)exp.shape[1]);
        auto ss = compare_f32(data.data(), npy_as_f32(exp), std::min(data.size(), exp.n_elements()));
        print_compare(name, ss);
    };
    check("conv_pre", "conv_pre_out", "hift_conv_pre.npy");
    check("ups0",     "ups0_out",     "hift_ups0.npy");
    check("sd0",      "sd0_out",      "hift_sd0.npy");
    check("sr0",      "sr0_out",      "hift_sr0.npy");
    check("rb0",      "rb0_out",      "hift_rb0.npy");
    check("ups1",     "ups1_out",     "hift_ups1.npy");
    check("ups2",     "ups2_out",     "hift_ups2.npy");

    auto s = compare_f32(out_data.data(), npy_as_f32(exp_npy), std::min(out_data.size(), exp_npy.n_elements()));
    print_compare("conv_post", s);

    ggml_gallocr_free(allocr);
    ggml_free(ctx);
}


static std::vector<float> build_stft_kernel(int n_fft, const std::vector<float> & window) {
    int F = n_fft / 2 + 1;
    int OC = 2 * F;
    std::vector<float> K((size_t)n_fft * 1 * OC, 0.0f);
    const double two_pi = 2.0 * M_PI;
    for (int f = 0; f < F; ++f) {
        int oc_re = f;
        int oc_im = F + f;
        for (int n = 0; n < n_fft; ++n) {
            double theta = two_pi * (double)f * (double)n / (double)n_fft;
            float w = window[n];
            K[n + oc_re * n_fft] = (float)(std::cos(theta) * w);
            K[n + oc_im * n_fft] = (float)(-std::sin(theta) * w);
        }
    }
    return K;
}

static std::vector<float> build_hann_window(int n, bool periodic = true) {
    std::vector<float> w(n);
    double N = periodic ? (double)n : (double)(n - 1);
    const double two_pi = 2.0 * M_PI;
    for (int i = 0; i < n; ++i) {
        w[i] = (float)(0.5 * (1.0 - std::cos(two_pi * (double)i / N)));
    }
    return w;
}


static ggml_tensor * reflect_pad_1d(ggml_context * ctx, ggml_tensor * x, int p_left, int p_right) {
    ggml_tensor * y = x;
    if (p_left > 0) {
        for (int i = 0; i < p_left; ++i) {
            int src_idx = p_left - i;
            ggml_tensor * s = ggml_view_3d(ctx, x,
                                           1, x->ne[1], x->ne[2],
                                           x->nb[1], x->nb[2],
                                           (size_t)src_idx * x->nb[0]);
            s = ggml_cont(ctx, s);
            y = ggml_concat(ctx, s, y, 0);
        }
    }
    if (p_right > 0) {
        int L_orig = (int)x->ne[0];
        for (int i = 0; i < p_right; ++i) {
            int src_idx = L_orig - 2 - i;
            ggml_tensor * s = ggml_view_3d(ctx, x,
                                           1, x->ne[1], x->ne[2],
                                           x->nb[1], x->nb[2],
                                           (size_t)src_idx * x->nb[0]);
            s = ggml_cont(ctx, s);
            y = ggml_concat(ctx, y, s, 0);
        }
    }
    return y;
}


static void stage_H4(const model_ctx & m, const std::string & ref_dir) {
    (void)m;
    fprintf(stderr, "\n=== Stage H4: STFT ===\n");
    npy_array src_npy = npy_load(ref_dir + "/hift_msource_tup0.npy");
    npy_array exp_npy = npy_load(ref_dir + "/hift_s_stft.npy");

    int T_src = (int)src_npy.shape[0];
    int n_fft = 16;
    int hop = 4;
    int F = n_fft / 2 + 1;

    auto window = build_hann_window(n_fft, true);
    auto kernel_data = build_stft_kernel(n_fft, window);

    static size_t buf_size = 4 * 1024 * 1024;
    static std::vector<uint8_t> buf(buf_size);
    ggml_init_params gp = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

    ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_src, 1);
    ggml_set_name(src, "src"); ggml_set_input(src);

    int pad_amt = n_fft / 2;
    ggml_tensor * src_padded = reflect_pad_1d(ctx, src, pad_amt, pad_amt);

    ggml_tensor * kernel = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_fft, 1, 2 * F);
    ggml_set_name(kernel, "kernel"); ggml_set_input(kernel);

    ggml_tensor * spec = conv1d_f32(ctx, kernel, src_padded, hop, 0, 1);
    ggml_set_name(spec, "spec"); ggml_set_output(spec);
    ggml_build_forward_expand(gf, spec);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "src"), npy_as_f32(src_npy), 0, src_npy.data.size());
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "kernel"), kernel_data.data(), 0, kernel_data.size() * sizeof(float));
    ggml_backend_graph_compute(cpu, gf);

    std::vector<float> out_data(ggml_nelements(spec));
    ggml_backend_tensor_get(spec, out_data.data(), 0, ggml_nbytes(spec));

    fprintf(stderr, "  spec ne=[%lld, %lld] ref=(%lld, %lld)\n",
            (long long)spec->ne[0], (long long)spec->ne[1],
            (long long)exp_npy.shape[0], (long long)exp_npy.shape[1]);

    auto s = compare_f32(out_data.data(), npy_as_f32(exp_npy), std::min(out_data.size(), exp_npy.n_elements()));
    print_compare("stft", s);

    ggml_gallocr_free(allocr);
    ggml_backend_free(cpu);
    ggml_free(ctx);
}


static std::vector<float> build_istft_kernel(int n_fft, const std::vector<float> & window) {
    int F = n_fft / 2 + 1;
    int IC = 2 * F;

    std::vector<float> K((size_t)n_fft * 1 * IC, 0.0f);
    const double two_pi = 2.0 * M_PI;
    const double inv_N = 1.0 / (double)n_fft;
    for (int f = 0; f < F; ++f) {
        double coef_re = (f == 0 || f == n_fft / 2) ? 1.0 : 2.0;
        double coef_im = (f == 0 || f == n_fft / 2) ? 0.0 : 2.0;
        for (int n = 0; n < n_fft; ++n) {
            double theta = two_pi * (double)f * (double)n / (double)n_fft;
            float w = window[n];
            K[n + 0 * n_fft + f       * n_fft] = (float)(coef_re * std::cos(theta) * w * inv_N);
            K[n + 0 * n_fft + (F + f) * n_fft] = (float)(-coef_im * std::sin(theta) * w * inv_N);
        }
    }
    return K;
}


static std::vector<float> build_window_sum(int T_stft, int n_fft, int hop, const std::vector<float> & window) {
    int L = (T_stft - 1) * hop + n_fft;
    std::vector<float> ws(L, 0.0f);
    for (int t = 0; t < T_stft; ++t) {
        int base = t * hop;
        for (int n = 0; n < n_fft; ++n) {
            ws[base + n] += window[n] * window[n];
        }
    }
    return ws;
}


static void stage_H5(const model_ctx & m, const std::string & ref_dir) {
    (void)m;
    fprintf(stderr, "\n=== Stage H5: ISTFT (conv_post -> waveform) ===\n");
    npy_array cp_npy  = npy_load(ref_dir + "/hift_conv_post.npy");
    npy_array wav_npy = npy_load(ref_dir + "/waveform.npy");

    int n_fft = 16;
    int hop = 4;
    int F = n_fft / 2 + 1;
    int T_stft = (int)cp_npy.shape[1];
    int L_raw = (T_stft - 1) * hop + n_fft;
    int L_wav = L_raw - n_fft;
    fprintf(stderr, "  T_stft=%d L_raw=%d L_wav=%d (ref=%lld)\n", T_stft, L_raw, L_wav, (long long)wav_npy.shape[0]);


    auto window = build_hann_window(n_fft, true);
    auto istft_kernel = build_istft_kernel(n_fft, window);
    auto w_sum = build_window_sum(T_stft, n_fft, hop, window);

    static size_t buf_size = 16 * 1024 * 1024;
    static std::vector<uint8_t> buf(buf_size);
    ggml_init_params gp = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);


    ggml_tensor * cp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_stft, 2 * F);
    ggml_set_name(cp, "cp"); ggml_set_input(cp);


    size_t row_stride = cp->nb[0];
    size_t col_stride = cp->nb[1];

    ggml_tensor * mag_log = ggml_view_2d(ctx, cp, T_stft, F, col_stride, 0);
    mag_log = ggml_cont(ctx, mag_log);

    mag_log = ggml_clamp(ctx, mag_log, -1e6f, 1e2f);
    ggml_tensor * mag = ggml_exp(ctx, mag_log);


    ggml_tensor * ph_in = ggml_view_2d(ctx, cp, T_stft, F, col_stride, (size_t)F * col_stride);
    ph_in = ggml_cont(ctx, ph_in);
    ggml_tensor * ph = ggml_sin(ctx, ph_in);


    ggml_tensor * cos_ph = ggml_cos(ctx, ph);
    ggml_tensor * sin_ph = ggml_sin(ctx, ph);
    ggml_tensor * real = ggml_mul(ctx, mag, cos_ph);
    ggml_tensor * imag = ggml_mul(ctx, mag, sin_ph);


    ggml_tensor * spec = ggml_concat(ctx, real, imag, 1);


    ggml_tensor * kernel = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_fft, 1, 2 * F);
    ggml_set_name(kernel, "kernel"); ggml_set_input(kernel);


    ggml_tensor * y = ggml_conv_transpose_1d(ctx, kernel, spec, hop, 0, 1);


    ggml_tensor * ws_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L_raw, 1);
    ggml_set_name(ws_in, "w_sum"); ggml_set_input(ws_in);
    y = ggml_div(ctx, y, ws_in);


    int pad_amt = n_fft / 2;
    ggml_tensor * y_trim = ggml_view_2d(ctx, y,
                                        L_wav, y->ne[1],
                                        y->nb[1],
                                        (size_t)pad_amt * y->nb[0]);
    y_trim = ggml_cont(ctx, y_trim);


    y_trim = ggml_clamp(ctx, y_trim, -0.99f, 0.99f);

    ggml_set_name(y_trim, "wav"); ggml_set_output(y_trim);
    ggml_build_forward_expand(gf, y_trim);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "cp"),     npy_as_f32(cp_npy),        0, cp_npy.data.size());
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "kernel"), istft_kernel.data(),       0, istft_kernel.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "w_sum"),  w_sum.data(),              0, w_sum.size() * sizeof(float));

    ggml_backend_graph_compute(cpu, gf);

    std::vector<float> out_data(ggml_nelements(y_trim));
    ggml_backend_tensor_get(y_trim, out_data.data(), 0, ggml_nbytes(y_trim));

    fprintf(stderr, "  wav ne=[%lld, %lld] ref=(%lld,)\n",
            (long long)y_trim->ne[0], (long long)y_trim->ne[1], (long long)wav_npy.shape[0]);

    auto s = compare_f32(out_data.data(), npy_as_f32(wav_npy), std::min(out_data.size(), wav_npy.n_elements()));
    print_compare("waveform", s);

    ggml_gallocr_free(allocr);
    ggml_backend_free(cpu);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s S3GEN_GGUF REFERENCE_DIR [stage=A|B|C|D|E|E0|F|G1|G2|G3|G4|H1|H3|H4|H5|ALL]\n", argv[0]);
        return 1;
    }
    const std::string gguf_path = argv[1];
    const std::string ref_dir   = argv[2];
    const std::string stage     = argc > 3 ? argv[3] : "ALL";

    fprintf(stderr, "Loading %s\n", gguf_path.c_str());
    model_ctx m = load_s3gen_gguf(gguf_path);
    fprintf(stderr, "  %zu tensors loaded, backend=%s\n", m.tensors.size(),
            ggml_backend_name(m.backend));

    try {
        if (stage == "A" || stage == "ALL") stage_A(m, ref_dir);
        if (stage == "B" || stage == "ALL") stage_B(m, ref_dir);
        if (stage == "C" || stage == "ALL") stage_C(m, ref_dir);
        if (stage == "D" || stage == "ALL") stage_D(m, ref_dir);
        if (stage == "E0" || stage == "ALL") stage_E0(m, ref_dir);
        if (stage == "E" || stage == "ALL") stage_E(m, ref_dir);
        if (stage == "F" || stage == "ALL") stage_F(m, ref_dir);
        if (stage == "G1" || stage == "ALL") stage_G1(m, ref_dir);
        if (stage == "G2" || stage == "ALL") stage_G2(m, ref_dir);
        if (stage == "G3" || stage == "ALL") stage_G3(m, ref_dir);
        if (stage == "G4" || stage == "ALL") stage_G4(m, ref_dir);
        if (stage == "H1" || stage == "ALL") stage_H1(m, ref_dir);
        if (stage == "H3" || stage == "ALL") stage_H3(m, ref_dir);
        if (stage == "H4" || stage == "ALL") stage_H4(m, ref_dir);
        if (stage == "H5" || stage == "ALL") stage_H5(m, ref_dir);
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    ggml_backend_buffer_free(m.buffer_w);
    ggml_backend_free(m.backend);
    ggml_free(m.ctx_w);
    return 0;
}
