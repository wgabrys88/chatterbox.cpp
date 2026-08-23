#include "voice_encoder.h"
#include "voice_features.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>


static bool copy_tensor_f32(ggml_context * ctx, const char * name,
                            std::vector<float> & out)
{
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) return false;
    out.resize(ggml_nelements(t));
    std::memcpy(out.data(), ggml_get_data(t), ggml_nbytes(t));
    return true;
}

bool voice_encoder_load(const std::string & t3_gguf_path,
                        voice_encoder_weights & out)
{
    ggml_context * tmp_ctx = nullptr;
    gguf_init_params gp = {  false,  &tmp_ctx };
    gguf_context * g = gguf_init_from_file(t3_gguf_path.c_str(), gp);
    if (!g) {
        fprintf(stderr, "voice_encoder_load: failed to open %s\n", t3_gguf_path.c_str());
        return false;
    }

    auto cleanup = [&](bool ok) {
        gguf_free(g);
        if (tmp_ctx) ggml_free(tmp_ctx);
        return ok;
    };


    if (gguf_find_key(g, "voice_encoder.hidden_size") < 0) {
        return cleanup(false);
    }

    auto get_u32 = [&](const char * k, uint32_t fallback) -> uint32_t {
        int64_t id = gguf_find_key(g, k);
        return id < 0 ? fallback : gguf_get_val_u32(g, id);
    };
    auto get_f32 = [&](const char * k, float fallback) -> float {
        int64_t id = gguf_find_key(g, k);
        return id < 0 ? fallback : gguf_get_val_f32(g, id);
    };

    out.n_layers       = (int)get_u32("voice_encoder.num_layers",    3);
    out.n_mels         = (int)get_u32("voice_encoder.n_mels",       40);
    out.hidden         = (int)get_u32("voice_encoder.hidden_size",  256);
    out.embedding      = (int)get_u32("voice_encoder.embedding_size", out.hidden);
    out.partial_frames = (int)get_u32("voice_encoder.partial_frames", 160);
    out.overlap        = get_f32("voice_encoder.overlap",            0.5f);
    out.rate           = get_f32("voice_encoder.rate",               1.3f);
    out.min_coverage   = get_f32("voice_encoder.min_coverage",       0.8f);

    auto load_or_fail = [&](const char * name, std::vector<float> & dst) {
        if (copy_tensor_f32(tmp_ctx, name, dst)) return true;
        fprintf(stderr, "voice_encoder_load: missing expected tensor '%s' in %s\n",
                name, t3_gguf_path.c_str());
        return false;
    };

    out.lstm.clear();
    out.lstm.resize(out.n_layers);
    for (int l = 0; l < out.n_layers; ++l) {
        auto & L = out.lstm[l];
        L.H = out.hidden;
        L.I = (l == 0) ? out.n_mels : out.hidden;
        char name[128];
        std::snprintf(name, sizeof(name), "voice_encoder/lstm/weight_ih_l%d", l);
        if (!load_or_fail(name, L.w_ih)) return cleanup(false);
        std::snprintf(name, sizeof(name), "voice_encoder/lstm/weight_hh_l%d", l);
        if (!load_or_fail(name, L.w_hh)) return cleanup(false);
        std::snprintf(name, sizeof(name), "voice_encoder/lstm/bias_ih_l%d", l);
        if (!load_or_fail(name, L.b_ih)) return cleanup(false);
        std::snprintf(name, sizeof(name), "voice_encoder/lstm/bias_hh_l%d", l);
        if (!load_or_fail(name, L.b_hh)) return cleanup(false);
    }
    if (!load_or_fail("voice_encoder/proj/weight", out.proj_w)) return cleanup(false);
    if (!load_or_fail("voice_encoder/proj/bias",   out.proj_b)) return cleanup(false);
    if (!load_or_fail("voice_encoder/mel_fb",      out.mel_fb)) return cleanup(false);

    return cleanup(true);
}


static void compute_partials(int n_frames, int partial, float rate,
                             int sample_rate_hz,
                             float overlap, float min_coverage,
                             int & n_wins, int & step, int & target_n)
{


    if (rate > 0.0f) {
        step = (int)std::lround(((double)sample_rate_hz / (double)rate) / (double)partial);
    } else {
        step = (int)std::lround((double)partial * (1.0 - overlap));
    }
    if (step <= 0) step = 1;
    if (step > partial) step = partial;


    int a = std::max(n_frames - partial + step, 0);
    int nw = a / step;
    int remainder = a - nw * step;
    if (nw == 0 || ((double)(remainder + (partial - step)) / (double)partial) >= (double)min_coverage) {
        nw += 1;
    }
    n_wins   = nw;
    target_n = partial + step * (nw - 1);
}


struct ve_graph {
    ggml_backend_t           backend      = nullptr;
    bool                     owns_backend = false;
    ggml_context           * weights_ctx  = nullptr;
    ggml_backend_buffer_t    weights_buf  = nullptr;
    ggml_gallocr_t           allocr       = nullptr;


    std::vector<ggml_tensor *> w_ih;
    std::vector<ggml_tensor *> w_hh;
    std::vector<ggml_tensor *> b_ih;
    std::vector<ggml_tensor *> b_hh;
    ggml_tensor *             proj_w  = nullptr;
    ggml_tensor *             proj_b  = nullptr;


    int H        = 0;
    int E        = 0;
    int partial  = 0;
    int n_mels   = 0;
    int n_layers = 0;
    int n_wins   = 0;
};

static void ve_graph_free(ve_graph & g) {
    if (g.allocr)      { ggml_gallocr_free(g.allocr);                g.allocr = nullptr; }
    if (g.weights_buf) { ggml_backend_buffer_free(g.weights_buf);    g.weights_buf = nullptr; }
    if (g.weights_ctx) { ggml_free(g.weights_ctx);                   g.weights_ctx = nullptr; }
    if (g.owns_backend && g.backend) { ggml_backend_free(g.backend); g.backend = nullptr; }
}

static bool ve_graph_init_weights(ve_graph & G, const voice_encoder_weights & w)
{
    const int n_layers = w.n_layers;
    const int H = w.hidden;
    const int E = w.embedding;

    G.H        = H;
    G.E        = E;
    G.partial  = w.partial_frames;
    G.n_mels   = w.n_mels;
    G.n_layers = n_layers;

    G.w_ih.assign(n_layers, nullptr);
    G.w_hh.assign(n_layers, nullptr);
    G.b_ih.assign(n_layers, nullptr);
    G.b_hh.assign(n_layers, nullptr);

    const int n_tensors = n_layers * 4 + 2 + 8;
    ggml_init_params ip = {
         (size_t) n_tensors * ggml_tensor_overhead(),
         nullptr,
         true,
    };
    G.weights_ctx = ggml_init(ip);
    if (!G.weights_ctx) {
        fprintf(stderr, "voice_encoder: ggml_init for weights failed\n");
        return false;
    }


    const int G4 = 4 * H;
    for (int l = 0; l < n_layers; ++l) {
        const int I_l = (l == 0) ? w.n_mels : H;
        char name[64];
        std::snprintf(name, sizeof(name), "ve/l%d/w_ih", l);
        G.w_ih[l] = ggml_new_tensor_2d(G.weights_ctx, GGML_TYPE_F32, I_l, G4);
        ggml_set_name(G.w_ih[l], name);

        std::snprintf(name, sizeof(name), "ve/l%d/w_hh", l);
        G.w_hh[l] = ggml_new_tensor_2d(G.weights_ctx, GGML_TYPE_F32, H, G4);
        ggml_set_name(G.w_hh[l], name);

        std::snprintf(name, sizeof(name), "ve/l%d/b_ih", l);
        G.b_ih[l] = ggml_new_tensor_1d(G.weights_ctx, GGML_TYPE_F32, G4);
        ggml_set_name(G.b_ih[l], name);

        std::snprintf(name, sizeof(name), "ve/l%d/b_hh", l);
        G.b_hh[l] = ggml_new_tensor_1d(G.weights_ctx, GGML_TYPE_F32, G4);
        ggml_set_name(G.b_hh[l], name);
    }
    G.proj_w = ggml_new_tensor_2d(G.weights_ctx, GGML_TYPE_F32, H, E);
    ggml_set_name(G.proj_w, "ve/proj_w");
    G.proj_b = ggml_new_tensor_1d(G.weights_ctx, GGML_TYPE_F32, E);
    ggml_set_name(G.proj_b, "ve/proj_b");


    G.weights_buf = ggml_backend_alloc_ctx_tensors(G.weights_ctx, G.backend);
    if (!G.weights_buf) {
        fprintf(stderr, "voice_encoder: backend weight alloc failed\n");
        return false;
    }

    auto set_tensor = [](ggml_tensor * t, const std::vector<float> & src) -> bool {
        const size_t bytes = src.size() * sizeof(float);
        if (bytes != ggml_nbytes(t)) {
            fprintf(stderr, "voice_encoder: size mismatch for %s: expected %zu, got %zu\n",
                    ggml_get_name(t), ggml_nbytes(t), bytes);
            return false;
        }
        ggml_backend_tensor_set(t, src.data(), 0, bytes);
        return true;
    };
    for (int l = 0; l < n_layers; ++l) {
        if (!set_tensor(G.w_ih[l], w.lstm[l].w_ih)) return false;
        if (!set_tensor(G.w_hh[l], w.lstm[l].w_hh)) return false;
        if (!set_tensor(G.b_ih[l], w.lstm[l].b_ih)) return false;
        if (!set_tensor(G.b_hh[l], w.lstm[l].b_hh)) return false;
    }
    if (!set_tensor(G.proj_w, w.proj_w)) return false;
    if (!set_tensor(G.proj_b, w.proj_b)) return false;
    return true;
}

static ggml_cgraph * build_ve_batched_graph(const ve_graph & G) {


    const int max_nodes = 32 * G.partial * G.n_layers + 256;

    const size_t buf_size =
        ggml_tensor_overhead() * max_nodes +
        ggml_graph_overhead_custom(max_nodes, false);
    static std::vector<uint8_t> buf;
    buf.resize(buf_size);
    ggml_init_params p = { buf_size, buf.data(),  true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf   = ggml_new_graph_custom(ctx, max_nodes, false);

    const int H  = G.H;
    const int E  = G.E;
    const int T  = G.partial;
    const int G4 = 4 * H;
    const int B  = G.n_wins;


    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, G.n_mels, T, B);
    ggml_set_name(x, "x"); ggml_set_input(x);


    ggml_tensor * h0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, B);
    ggml_set_name(h0, "h0"); ggml_set_input(h0);
    ggml_tensor * c0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, B);
    ggml_set_name(c0, "c0"); ggml_set_input(c0);

    ggml_tensor * x_layer = x;

    for (int l = 0; l < G.n_layers; ++l) {


        ggml_tensor * gates_ih_seq = ggml_mul_mat(ctx, G.w_ih[l], x_layer);
        gates_ih_seq = ggml_add(ctx, gates_ih_seq, G.b_ih[l]);


        const bool need_seq = (l + 1 < G.n_layers);
        ggml_tensor * h_seq = nullptr;
        if (need_seq) {
            h_seq = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H, T, B);
            char name[32]; std::snprintf(name, sizeof(name), "h_seq_l%d", l);
            ggml_set_name(h_seq, name);
        }

        ggml_tensor * h_prev = h0;
        ggml_tensor * c_prev = c0;


        const size_t gates_batch_stride = (size_t) G4 * T * sizeof(float);
        const size_t gates_t_offset     = (size_t) G4 *     sizeof(float);


        const size_t hseq_batch_stride  = (size_t) H  * T * sizeof(float);
        const size_t hseq_t_offset      = (size_t) H  *     sizeof(float);

        for (int t = 0; t < T; ++t) {

            ggml_tensor * gates_ih_t = ggml_view_2d(ctx, gates_ih_seq,
                                                     G4, B,
                                                     gates_batch_stride,
                                                     gates_t_offset * (size_t) t);


            ggml_tensor * gates_hh = ggml_mul_mat(ctx, G.w_hh[l], h_prev);
            gates_hh = ggml_add(ctx, gates_hh, G.b_hh[l]);

            ggml_tensor * gates = ggml_add(ctx, gates_ih_t, gates_hh);


            const size_t gates_col_stride = (size_t) G4 * sizeof(float);
            ggml_tensor * i_raw = ggml_view_2d(ctx, gates, H, B, gates_col_stride,
                                                0 * (size_t) H * sizeof(float));
            ggml_tensor * f_raw = ggml_view_2d(ctx, gates, H, B, gates_col_stride,
                                                1 * (size_t) H * sizeof(float));
            ggml_tensor * g_raw = ggml_view_2d(ctx, gates, H, B, gates_col_stride,
                                                2 * (size_t) H * sizeof(float));
            ggml_tensor * o_raw = ggml_view_2d(ctx, gates, H, B, gates_col_stride,
                                                3 * (size_t) H * sizeof(float));

            ggml_tensor * i_t = ggml_sigmoid(ctx, i_raw);
            ggml_tensor * f_t = ggml_sigmoid(ctx, f_raw);
            ggml_tensor * g_t = ggml_tanh   (ctx, g_raw);
            ggml_tensor * o_t = ggml_sigmoid(ctx, o_raw);

            ggml_tensor * fc  = ggml_mul(ctx, f_t, c_prev);
            ggml_tensor * ig  = ggml_mul(ctx, i_t, g_t);
            ggml_tensor * c_t = ggml_add(ctx, fc, ig);
            ggml_tensor * h_t = ggml_mul(ctx, o_t, ggml_tanh(ctx, c_t));

            if (need_seq) {
                ggml_tensor * dst = ggml_view_2d(ctx, h_seq, H, B,
                                                  hseq_batch_stride,
                                                  hseq_t_offset * (size_t) t);
                ggml_build_forward_expand(gf, ggml_cpy(ctx, h_t, dst));
            }

            h_prev = h_t;
            c_prev = c_t;
        }

        x_layer = need_seq ? h_seq : h_prev;
    }


    ggml_tensor * emb = ggml_mul_mat(ctx, G.proj_w, x_layer);
    emb = ggml_add(ctx, emb, G.proj_b);
    emb = ggml_relu(ctx, emb);
    ggml_set_name(emb, "emb"); ggml_set_output(emb);
    ggml_build_forward_expand(gf, emb);

    ggml_free(ctx);
    return gf;
}


bool voice_encoder_embed(const std::vector<float> & wav_16k,
                         const voice_encoder_weights & w,
                         ggml_backend_t backend,
                         std::vector<float> & out)
{
    if (w.mel_fb.empty() || w.lstm.size() != (size_t)w.n_layers) {
        fprintf(stderr, "voice_encoder_embed: weights are incomplete\n");
        return false;
    }


    std::vector<float> mel = mel_extract_16k_40(wav_16k, w.mel_fb);
    if (mel.empty()) {
        fprintf(stderr, "voice_encoder_embed: mel extraction failed\n");
        return false;
    }
    const int T_mel = (int)(mel.size() / w.n_mels);


    int n_wins, step, target_n;
    compute_partials(T_mel, w.partial_frames, w.rate, w.partial_frames,
                     w.overlap, w.min_coverage, n_wins, step, target_n);
    if (target_n > T_mel) mel.resize((size_t) target_n * w.n_mels, 0.0f);
    else if (target_n < T_mel) mel.resize((size_t) target_n * w.n_mels);


    ve_graph G;
    G.backend = backend;
    if (!G.backend) {
        G.backend = ggml_backend_cpu_init();
        if (!G.backend) {
            fprintf(stderr, "voice_encoder_embed: ggml_backend_cpu_init failed\n");
            return false;
        }
        G.owns_backend = true;
    }

    if (!ve_graph_init_weights(G, w)) {
        ve_graph_free(G);
        return false;
    }
    G.n_wins = n_wins;


    ggml_cgraph * gf = build_ve_batched_graph(G);
    G.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(G.backend));
    if (!G.allocr || !ggml_gallocr_reserve(G.allocr, gf)) {
        fprintf(stderr, "voice_encoder_embed: gallocr_reserve failed\n");
        ve_graph_free(G);
        return false;
    }
    ggml_gallocr_alloc_graph(G.allocr, gf);

    const int H       = w.hidden;
    const int E       = w.embedding;
    const int partial = w.partial_frames;
    const int n_mels  = w.n_mels;


    std::vector<float> x_buf((size_t) n_mels * partial * n_wins);
    for (int wi = 0; wi < n_wins; ++wi) {
        const int t0 = wi * step;
        std::memcpy(x_buf.data() + (size_t) wi * partial * n_mels,
                    mel.data()   + (size_t) t0 * n_mels,
                    (size_t) partial * n_mels * sizeof(float));
    }

    ggml_tensor * x_t = ggml_graph_get_tensor(gf, "x");
    ggml_tensor * h0  = ggml_graph_get_tensor(gf, "h0");
    ggml_tensor * c0  = ggml_graph_get_tensor(gf, "c0");
    ggml_backend_tensor_set(x_t, x_buf.data(), 0, x_buf.size() * sizeof(float));

    std::vector<float> zeros((size_t) H * n_wins, 0.0f);
    ggml_backend_tensor_set(h0, zeros.data(), 0, zeros.size() * sizeof(float));
    ggml_backend_tensor_set(c0, zeros.data(), 0, zeros.size() * sizeof(float));

    ggml_backend_graph_compute(G.backend, gf);


    std::vector<float> emb_buf((size_t) E * n_wins);
    ggml_tensor * emb_tensor = ggml_graph_get_tensor(gf, "emb");
    ggml_backend_tensor_get(emb_tensor, emb_buf.data(), 0, emb_buf.size() * sizeof(float));

    std::vector<float> emb_accum(E, 0.0f);
    for (int wi = 0; wi < n_wins; ++wi) {
        float * v = emb_buf.data() + (size_t) wi * E;
        double sq = 0.0;
        for (int o = 0; o < E; ++o) sq += (double) v[o] * (double) v[o];
        double nrm = std::sqrt(sq);
        if (nrm > 1e-12) {
            float s = (float) (1.0 / nrm);
            for (int o = 0; o < E; ++o) v[o] *= s;
        }
        for (int o = 0; o < E; ++o) emb_accum[o] += v[o];
    }

    float inv_n = 1.0f / (float) n_wins;
    for (int o = 0; o < E; ++o) emb_accum[o] *= inv_n;

    double sq = 0.0;
    for (int o = 0; o < E; ++o) sq += (double) emb_accum[o] * (double) emb_accum[o];
    double nrm = std::sqrt(sq);
    if (nrm > 1e-12) {
        float s = (float) (1.0 / nrm);
        for (int o = 0; o < E; ++o) emb_accum[o] *= s;
    }
    out = std::move(emb_accum);

    ve_graph_free(G);
    return true;
}
