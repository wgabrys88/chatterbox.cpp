#include "tts-cpp/chatterbox/engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <set>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "chatterbox_t3_internal.h"
#include "gpt2_bpe.h"
#include "mtl_tokenizer.h"
#include "npy.h"
#include "t3_mtl.h"
#include "tts-cpp/chatterbox/s3gen_pipeline.h"
#include "voice_encoder.h"
#include "voice_features.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

namespace tts_cpp::chatterbox {

using namespace detail;

namespace {

int resolve_thread_count(int requested) {
    if (requested > 0) return requested;
    const int hw = (int) std::thread::hardware_concurrency();
    return hw > 0 ? std::min(hw, 4) : 4;
}

void wait_for_preload(std::thread & t) {
    if (t.joinable()) t.join();
}

// ---------------------------------------------------------------------------
// Trident debug fingerprints.  Every conditioning tensor gets a deterministic
// FNV-1a hash plus range stats so trident.log can prove, per run, which
// reference was actually baked and that the write reached the backend.
// ---------------------------------------------------------------------------
std::uint64_t tts_fnv1a(const void * data, size_t bytes) {
    const unsigned char * p = (const unsigned char *) data;
    std::uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < bytes; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

void tts_log_vec(const char * tag, const std::vector<float> & v) {
    float mn = 0.0f, mx = 0.0f, l2 = 0.0f;
    bool finite = true;
    for (float x : v) {
        if (!std::isfinite(x)) finite = false;
        mn = std::min(mn, x);
        mx = std::max(mx, x);
        l2 += x * x;
    }
    fprintf(stderr,
            "tts event=vec tag=%s dim=%zu l2=%.6e min=%.6e max=%.6e fnv=%016llx finite=%d\n",
            tag, v.size(), (double) std::sqrt(l2), (double) mn, (double) mx,
            (unsigned long long) tts_fnv1a(v.data(), v.size() * sizeof(float)),
            finite ? 1 : 0);
}

void tts_log_tokens(const char * tag, const std::vector<int32_t> & t) {
    fprintf(stderr,
            "tts event=tokens tag=%s n=%zu first=%d mid=%d last=%d fnv=%016llx\n",
            tag, t.size(),
            t.empty() ? -1 : t.front(),
            t.empty() ? -1 : t[t.size() / 2],
            t.empty() ? -1 : t.back(),
            (unsigned long long) tts_fnv1a(t.data(), t.size() * sizeof(int32_t)));
}

void tts_dump_tokens(const char * tag, const std::vector<int32_t> & t) {
    fprintf(stderr, "tts event=tokens_all tag=%s n=%zu v=", tag, t.size());
    for (size_t i = 0; i < t.size(); ++i) fprintf(stderr, "%s%d", i ? "," : "", (int) t[i]);
    fprintf(stderr, "\n");
    fflush(stderr);
}

int tts_count_mismatch(const std::vector<float> & a, const std::vector<float> & b) {
    size_t n = std::min(a.size(), b.size());
    int bad = (int) (a.size() != b.size());
    for (size_t i = 0; i < n; ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) ++bad;
    }
    return bad;
}

// One deterministic forward per critical op on the active backend compared
// against a CPU reference graph of the same op.  A single number per boot
// turns "Vulkan might miscompute" into measured fact.
void tts_backend_parity_probe(ggml_backend_t backend) {
    fprintf(stderr, "tts event=parity begin backend=%s\n", ggml_backend_name(backend));
    fflush(stderr);
    auto rel_err = [](const std::vector<float> & got, const std::vector<float> & ref) {
        double worst = 0.0;
        size_t n = std::min(got.size(), ref.size());
        for (size_t i = 0; i < n; ++i) {
            double m = std::max(std::fabs((double) got[i]), std::fabs((double) ref[i]));
            if (m <= 1e-12) continue;
            worst = std::max(worst, std::fabs((double) got[i] - (double) ref[i]) / m);
        }
        return worst;
    };
    auto dump_out = [&](int side, ggml_backend_t be, const std::vector<float> & out) {
        float mn = 0, mx = 0; double l2 = 0; bool fin = true;
        for (float x : out) { if (!std::isfinite(x)) fin = false; mn = std::min(mn, x); mx = std::max(mx, x); l2 += (double) x * x; }
        fprintf(stderr,
                "tts event=parity out side=%d be=%s dim=%zu l2=%.6e min=%.6e max=%.6e fnv=%016llx finite=%d\n",
                side, ggml_backend_name(be), out.size(), std::sqrt(l2), (double) mn, (double) mx,
                (unsigned long long) tts_fnv1a(out.data(), out.size() * sizeof(float)), fin ? 1 : 0);
        fflush(stderr);
    };
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) { fprintf(stderr, "tts event=parity error=cpu_init_failed\n"); return; }
    fprintf(stderr, "tts event=parity stage=cpu_ok\n");
    fflush(stderr);

    // mul_mat: A(K,M) x B(K,N).
    {
        fprintf(stderr, "tts event=parity block=mul_mat begin\n");
        fflush(stderr);
        const int M = 48, N = 48, K = 96;
        ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(128, false), nullptr, true };
        std::vector<float> a(K * M), b(K * N);
        for (size_t i = 0; i < a.size(); ++i) a[i] = std::sin((float) i * 0.37f) * 0.5f;
        for (size_t i = 0; i < b.size(); ++i) b[i] = std::cos((float) i * 0.53f) * 0.5f;
        std::vector<float> got, want;
        for (int side = 0; side < 2; ++side) {
            ggml_context * ctx = ggml_init(ip);
            fprintf(stderr, "tts event=parity stage=ctx side=%d\n", side);
            fflush(stderr);
            fprintf(stderr, "tts event=parity call=new_tensor_A\n"); fflush(stderr);
            ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
            fprintf(stderr, "tts event=parity call=new_tensor_B\n"); fflush(stderr);
            ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
            fprintf(stderr, "tts event=parity call=mul_mat\n"); fflush(stderr);
            ggml_tensor * Y = ggml_mul_mat(ctx, A, B);
            fprintf(stderr, "tts event=parity call=new_graph\n"); fflush(stderr);
            ggml_cgraph * gf = ggml_new_graph_custom(ctx, 128, false);
            fprintf(stderr, "tts event=parity call=expand\n"); fflush(stderr);
            ggml_build_forward_expand(gf, Y);
            fprintf(stderr, "tts event=parity stage=graph side=%d\n", side);
            fflush(stderr);
            ggml_backend_t be = side ? cpu : backend;
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
            fprintf(stderr, "tts event=parity stage=alloc side=%d be=%s\n", side, ggml_backend_name(be));
            fflush(stderr);
            ggml_backend_tensor_set(A, a.data(), 0, ggml_nbytes(A));
            ggml_backend_tensor_set(B, b.data(), 0, ggml_nbytes(B));
            fprintf(stderr, "tts event=parity stage=compute side=%d\n", side);
            fflush(stderr);
            ggml_backend_graph_compute(be, gf);
            std::vector<float> out(ggml_nelements(Y));
            ggml_backend_tensor_get(Y, out.data(), 0, ggml_nbytes(Y));
            {
                float mn = 0, mx = 0; double l2 = 0; bool fin = true;
                for (float x : out) { if (!std::isfinite(x)) fin = false; mn = std::min(mn, x); mx = std::max(mx, x); l2 += (double) x * x; }
                fprintf(stderr,
                        "tts event=parity out side=%d be=%s dim=%zu l2=%.6e min=%.6e max=%.6e fnv=%016llx finite=%d\n",
                        side, ggml_backend_name(be), out.size(), std::sqrt(l2), (double) mn, (double) mx,
                        (unsigned long long) tts_fnv1a(out.data(), out.size() * sizeof(float)), fin ? 1 : 0);
                fflush(stderr);
            }
            (side ? want : got) = std::move(out);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
        fprintf(stderr, "tts event=parity op=mul_mat n=%d max_rel=%.3e\n", M * N, rel_err(got, want));
        fflush(stderr);
    }

    // conv_transpose_1d: kernel(Cout,Cin,K) x src(Cin,T), s=1 p=0 d=1.
    {
        fprintf(stderr, "tts event=parity block=conv_transpose_1d begin\n");
        fflush(stderr);
        const int Cin = 2, Cout = 4, K = 3, T = 16, S = 1, P = 0, D = 1;
        ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(128, false), nullptr, true };
        std::vector<float> k(Cout * Cin * K), x(Cin * T);
        for (size_t i = 0; i < k.size(); ++i) k[i] = std::sin((float) i * 0.71f) * 0.5f;
        for (size_t i = 0; i < x.size(); ++i) x[i] = std::cos((float) i * 0.29f) * 0.5f;
        std::vector<float> got, want;
        for (int side = 0; side < 2; ++side) {
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * W = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, Cout, Cin);
            ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, Cin);
            ggml_tensor * Y = ggml_conv_transpose_1d(ctx, W, X, S, P, D);
            ggml_cgraph * gf = ggml_new_graph_custom(ctx, 128, false);
            ggml_build_forward_expand(gf, Y);
            ggml_backend_t be = side ? cpu : backend;
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
            fprintf(stderr, "tts event=parity stage=alloc side=%d\n", side);
            fflush(stderr);
            ggml_backend_tensor_set(W, k.data(), 0, ggml_nbytes(W));
            ggml_backend_tensor_set(X, x.data(), 0, ggml_nbytes(X));
            fprintf(stderr, "tts event=parity stage=compute side=%d\n", side);
            fflush(stderr);
            ggml_backend_graph_compute(be, gf);
            std::vector<float> out(ggml_nelements(Y));
            ggml_backend_tensor_get(Y, out.data(), 0, ggml_nbytes(Y));
            {
                float mn = 0, mx = 0; double l2 = 0; bool fin = true;
                for (float x : out) { if (!std::isfinite(x)) fin = false; mn = std::min(mn, x); mx = std::max(mx, x); l2 += (double) x * x; }
                fprintf(stderr,
                        "tts event=parity out side=%d be=%s dim=%zu l2=%.6e min=%.6e max=%.6e fnv=%016llx finite=%d\n",
                        side, ggml_backend_name(be), out.size(), std::sqrt(l2), (double) mn, (double) mx,
                        (unsigned long long) tts_fnv1a(out.data(), out.size() * sizeof(float)), fin ? 1 : 0);
                fflush(stderr);
            }
            (side ? want : got) = std::move(out);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
        fprintf(stderr, "tts event=parity op=conv_transpose_1d n=%d max_rel=%.3e\n",
                (int) want.size(), rel_err(got, want));
        fflush(stderr);
    }

    // soft_max over 1024 elements.
    {
        fprintf(stderr, "tts event=parity block=soft_max begin\n");
        fflush(stderr);
        const int n = 1024;
        ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(128, false), nullptr, true };
        std::vector<float> v(n);
        for (int i = 0; i < n; ++i) v[i] = std::sin((float) i * 0.11f);
        std::vector<float> got, want;
        for (int side = 0; side < 2; ++side) {
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * V = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
            ggml_tensor * Y = ggml_soft_max(ctx, V);
            ggml_cgraph * gf = ggml_new_graph_custom(ctx, 128, false);
            ggml_build_forward_expand(gf, Y);
            ggml_backend_t be = side ? cpu : backend;
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
            fprintf(stderr, "tts event=parity stage=alloc side=%d\n", side);
            fflush(stderr);
            ggml_backend_tensor_set(V, v.data(), 0, ggml_nbytes(V));
            fprintf(stderr, "tts event=parity stage=compute side=%d\n", side);
            fflush(stderr);
            ggml_backend_graph_compute(be, gf);
            std::vector<float> out(ggml_nelements(Y));
            ggml_backend_tensor_get(Y, out.data(), 0, ggml_nbytes(Y));
            {
                float mn = 0, mx = 0; double l2 = 0; bool fin = true;
                for (float x : out) { if (!std::isfinite(x)) fin = false; mn = std::min(mn, x); mx = std::max(mx, x); l2 += (double) x * x; }
                fprintf(stderr,
                        "tts event=parity out side=%d be=%s dim=%zu l2=%.6e min=%.6e max=%.6e fnv=%016llx finite=%d\n",
                        side, ggml_backend_name(be), out.size(), std::sqrt(l2), (double) mn, (double) mx,
                        (unsigned long long) tts_fnv1a(out.data(), out.size() * sizeof(float)), fin ? 1 : 0);
                fflush(stderr);
            }
            (side ? want : got) = std::move(out);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
        fprintf(stderr, "tts event=parity op=soft_max n=%d max_rel=%.3e\n", n, rel_err(got, want));
        fflush(stderr);
    }

    // group_norm over (32,8) with 8 groups.
    {
        fprintf(stderr, "tts event=parity block=group_norm begin\n");
        fflush(stderr);
        const int C = 32, Tg = 8, G = 8;
        ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(128, false), nullptr, true };
        std::vector<float> v(C * Tg);
        for (size_t i = 0; i < v.size(); ++i) v[i] = std::cos((float) i * 0.19f) * 0.5f;
        std::vector<float> got, want;
        for (int side = 0; side < 2; ++side) {
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * V = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Tg, C);
            ggml_tensor * Y = ggml_group_norm(ctx, V, G, 1e-5f);
            ggml_cgraph * gf = ggml_new_graph_custom(ctx, 128, false);
            ggml_build_forward_expand(gf, Y);
            ggml_backend_t be = side ? cpu : backend;
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
            fprintf(stderr, "tts event=parity stage=alloc side=%d\n", side);
            fflush(stderr);
            ggml_backend_tensor_set(V, v.data(), 0, ggml_nbytes(V));
            fprintf(stderr, "tts event=parity stage=compute side=%d\n", side);
            fflush(stderr);
            ggml_backend_graph_compute(be, gf);
            std::vector<float> out(ggml_nelements(Y));
            ggml_backend_tensor_get(Y, out.data(), 0, ggml_nbytes(Y));
            {
                float mn = 0, mx = 0; double l2 = 0; bool fin = true;
                for (float x : out) { if (!std::isfinite(x)) fin = false; mn = std::min(mn, x); mx = std::max(mx, x); l2 += (double) x * x; }
                fprintf(stderr,
                        "tts event=parity out side=%d be=%s dim=%zu l2=%.6e min=%.6e max=%.6e fnv=%016llx finite=%d\n",
                        side, ggml_backend_name(be), out.size(), std::sqrt(l2), (double) mn, (double) mx,
                        (unsigned long long) tts_fnv1a(out.data(), out.size() * sizeof(float)), fin ? 1 : 0);
                fflush(stderr);
            }
            (side ? want : got) = std::move(out);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
        fprintf(stderr, "tts event=parity op=group_norm n=%d max_rel=%.3e\n",
                (int) want.size(), rel_err(got, want));
        fflush(stderr);
    }

    // GELU over 2048 elements.
    {
        fprintf(stderr, "tts event=parity block=gelu begin\n");
        fflush(stderr);
        const int n = 2048;
        ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(128, false), nullptr, true };
        std::vector<float> v(n);
        for (int i = 0; i < n; ++i) v[i] = std::sin((float) i * 0.07f) * 2.0f;
        std::vector<float> got, want;
        for (int side = 0; side < 2; ++side) {
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * V = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
            ggml_tensor * Y = ggml_gelu(ctx, V);
            ggml_cgraph * gf = ggml_new_graph_custom(ctx, 128, false);
            ggml_build_forward_expand(gf, Y);
            ggml_backend_t be = side ? cpu : backend;
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
            ggml_backend_tensor_set(V, v.data(), 0, ggml_nbytes(V));
            ggml_backend_graph_compute(be, gf);
            std::vector<float> out(ggml_nelements(Y));
            ggml_backend_tensor_get(Y, out.data(), 0, ggml_nbytes(Y));
            dump_out(side, be, out);
            (side ? want : got) = std::move(out);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
        fprintf(stderr, "tts event=parity op=gelu n=%d max_rel=%.3e\n", n, rel_err(got, want));
        fflush(stderr);
        double vg = 0, cg = 0;
        for (int i = 0; i < n; ++i) {
            const double tr = v[i] * 0.5 * (1.0 + std::erf(v[i] / std::sqrt(2.0)));
            const double m = std::max(1e-9, std::fabs(tr));
            vg = std::max(vg, std::fabs((double) got[i] - tr) / m);
            cg = std::max(cg, std::fabs((double) want[i] - tr) / m);
        }
        fprintf(stderr, "tts event=parity op=gelu truth vulkan=%.3e cpu=%.3e\n", vg, cg);
        fflush(stderr);
    }

    // Conv1d(s=2,p=1) via F32 im2col + mul_mat — S3TokenizerV2 stem path.
    {
        fprintf(stderr, "tts event=parity block=conv1d_im2col begin\n");
        fflush(stderr);
        const int K = 3, Cin = 16, Cout = 64, T = 512;
        ggml_init_params ip = { ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(256, false), nullptr, true };
        std::vector<float> w(K * Cin * Cout), x(Cin * T);
        for (size_t i = 0; i < w.size(); ++i) w[i] = std::sin((float) i * 0.13f) * 0.25f;
        for (size_t i = 0; i < x.size(); ++i) x[i] = std::cos((float) i * 0.31f) * 0.5f;
        std::vector<float> got, want;
        for (int side = 0; side < 2; ++side) {
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * W = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, Cin, Cout);
            ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, Cin);
            ggml_tensor * im2col = ggml_im2col(ctx, W, X, 2, 0, 1, 0, 1, 0, false, GGML_TYPE_F32);
            ggml_tensor * R = ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
                ggml_reshape_2d(ctx, W, W->ne[0] * W->ne[1], Cout));
            ggml_tensor * Y = ggml_reshape_3d(ctx, R, im2col->ne[1], Cout, im2col->ne[2]);
            ggml_cgraph * gf = ggml_new_graph_custom(ctx, 256, false);
            ggml_build_forward_expand(gf, Y);
            ggml_backend_t be = side ? cpu : backend;
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
            ggml_backend_tensor_set(W, w.data(), 0, ggml_nbytes(W));
            ggml_backend_tensor_set(X, x.data(), 0, ggml_nbytes(X));
            ggml_backend_graph_compute(be, gf);
            std::vector<float> out(ggml_nelements(Y));
            ggml_backend_tensor_get(Y, out.data(), 0, ggml_nbytes(Y));
            dump_out(side, be, out);
            (side ? want : got) = std::move(out);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
        fprintf(stderr, "tts event=parity op=conv1d_im2col n=%d max_rel=%.3e\n",
                (int) want.size(), rel_err(got, want));
        fflush(stderr);
    }

    // Depth-wise Conv1d(k=31,p=15) via F32 im2col — S3TokenizerV2 FSMN path.
    {
        fprintf(stderr, "tts event=parity block=conv1d_dw_im2col begin\n");
        fflush(stderr);
        const int K = 31, C = 32, T = 256;
        ggml_init_params ip = { ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(256, false), nullptr, true };
        std::vector<float> w(K * C), x(T * C);
        for (size_t i = 0; i < w.size(); ++i) w[i] = std::sin((float) i * 0.17f) * 0.2f;
        for (size_t i = 0; i < x.size(); ++i) x[i] = std::cos((float) i * 0.29f) * 0.5f;
        std::vector<float> got, want;
        for (int side = 0; side < 2; ++side) {
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * W = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, C);
            ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, C);
            ggml_tensor * X4 = ggml_reshape_4d(ctx, X, T, 1, C, 1);
            ggml_tensor * im2col = ggml_im2col(ctx, W, X4, 1, 0, (K - 1) / 2, 0, 1, 0, false, GGML_TYPE_F32);
            ggml_tensor * R = ggml_mul_mat(ctx, im2col, W);
            ggml_tensor * Y = ggml_reshape_3d(ctx, R, R->ne[0], W->ne[2], 1);
            ggml_cgraph * gf = ggml_new_graph_custom(ctx, 256, false);
            ggml_build_forward_expand(gf, Y);
            ggml_backend_t be = side ? cpu : backend;
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
            ggml_backend_tensor_set(W, w.data(), 0, ggml_nbytes(W));
            ggml_backend_tensor_set(X, x.data(), 0, ggml_nbytes(X));
            ggml_backend_graph_compute(be, gf);
            std::vector<float> out(ggml_nelements(Y));
            ggml_backend_tensor_get(Y, out.data(), 0, ggml_nbytes(Y));
            dump_out(side, be, out);
            (side ? want : got) = std::move(out);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
        fprintf(stderr, "tts event=parity op=conv1d_dw_im2col n=%d max_rel=%.3e\n",
                (int) want.size(), rel_err(got, want));
        fflush(stderr);
    }

    // RoPE NEOX over (64,20,T) — S3TokenizerV2 attention q/k path.
    {
        fprintf(stderr, "tts event=parity block=rope_neox begin\n");
        fflush(stderr);
        const int D = 64, H = 20, Tq = 96;
        ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(128, false), nullptr, true };
        std::vector<float> q(D * H * Tq);
        for (size_t i = 0; i < q.size(); ++i) q[i] = std::sin((float) i * 0.23f) * 0.5f;
        std::vector<double> tru(q.size());
        const double ts = std::pow(10000.0, -2.0 / D);
        for (int t = 0; t < Tq; ++t)
            for (int h = 0; h < H; ++h)
                for (int p = 0; p < D / 2; ++p) {
                    const size_t ia = ((size_t) t * H + h) * D + p, ib = ia + D / 2;
                    const double th = (double) t * std::pow(ts, (double) p);
                    const double c = std::cos(th), sn = std::sin(th);
                    tru[ia] = (double) q[ia] * c - (double) q[ib] * sn;
                    tru[ib] = (double) q[ia] * sn + (double) q[ib] * c;
                }
        std::vector<float> got, want;
        for (int side = 0; side < 2; ++side) {
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * Q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H, Tq);
            ggml_tensor * P = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Tq);
            ggml_tensor * Y = ggml_rope_ext(ctx, Q, P, nullptr, D,
                                            GGML_ROPE_TYPE_NEOX, /*n_ctx_orig=*/2048,
                                            /*freq_base=*/10000.0f, /*freq_scale=*/1.0f,
                                            /*ext_factor=*/0.0f, /*attn_factor=*/1.0f,
                                            /*beta_fast=*/32.0f, /*beta_slow=*/1.0f);
            ggml_cgraph * gf = ggml_new_graph_custom(ctx, 128, false);
            ggml_build_forward_expand(gf, Y);
            ggml_backend_t be = side ? cpu : backend;
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
            std::vector<int32_t> pos(Tq);
            for (int i = 0; i < Tq; ++i) pos[i] = i;
            ggml_backend_tensor_set(Q, q.data(), 0, ggml_nbytes(Q));
            ggml_backend_tensor_set(P, pos.data(), 0, ggml_nbytes(P));
            ggml_backend_graph_compute(be, gf);
            std::vector<float> out(ggml_nelements(Y));
            ggml_backend_tensor_get(Y, out.data(), 0, ggml_nbytes(Y));
            dump_out(side, be, out);
            (side ? want : got) = std::move(out);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
        fprintf(stderr, "tts event=parity op=rope_neox n=%d max_rel=%.3e\n",
                (int) want.size(), rel_err(got, want));
        fflush(stderr);
        double vg = 0, cg = 0;
        size_t wi = 0;
        for (size_t i = 0; i < tru.size(); ++i) {
            const double m = std::max(1e-9, std::fabs(tru[i]));
            const double rv = std::fabs((double) got[i] - tru[i]) / m;
            const double rc = std::fabs((double) want[i] - tru[i]) / m;
            if (rv > vg) { vg = rv; wi = i; }
            if (rc > cg) cg = rc;
        }
        fprintf(stderr,
                "tts event=parity op=rope_neox truth vulkan=%.3e cpu=%.3e worst_dim=%d t=%d h=%d got=%.7f want=%.7f true=%.7f\n",
                vg, cg, (int)((wi % ((size_t) H * D)) % D), (int)(wi / ((size_t) H * D)),
                (int)((wi % ((size_t) H * D)) / D), got[wi], want[wi], tru[wi]);
        fflush(stderr);
    }

    // Row-wise soft_max over (512,64) — realistic attention-score shape.
    {
        fprintf(stderr, "tts event=parity block=soft_max_attn begin\n");
        fflush(stderr);
        const int Tk = 512, Tq = 64;
        ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(128, false), nullptr, true };
        std::vector<float> v(Tk * Tq);
        for (size_t i = 0; i < v.size(); ++i) v[i] = std::sin((float) i * 0.05f) * 4.0f;
        std::vector<float> got, want;
        for (int side = 0; side < 2; ++side) {
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * V = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Tk, Tq);
            ggml_tensor * Y = ggml_soft_max(ctx, V);
            ggml_cgraph * gf = ggml_new_graph_custom(ctx, 128, false);
            ggml_build_forward_expand(gf, Y);
            ggml_backend_t be = side ? cpu : backend;
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
            ggml_backend_tensor_set(V, v.data(), 0, ggml_nbytes(V));
            ggml_backend_graph_compute(be, gf);
            std::vector<float> out(ggml_nelements(Y));
            ggml_backend_tensor_get(Y, out.data(), 0, ggml_nbytes(Y));
            dump_out(side, be, out);
            (side ? want : got) = std::move(out);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
        fprintf(stderr, "tts event=parity op=soft_max_attn n=%d max_rel=%.3e\n",
                (int) want.size(), rel_err(got, want));
        fflush(stderr);
    }

    // Plain F32 mul_mat, no im2col -- isolates the matmul core from the packing op.
    {
        fprintf(stderr, "tts event=parity block=mul_mat_plain begin\n");
        fflush(stderr);
        const int Km = 48, Mm = 4096, Nm = 64;
        ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(128, false), nullptr, true };
        std::vector<float> a(Km * Mm), b(Km * Nm);
        for (size_t i = 0; i < a.size(); ++i) a[i] = std::sin((float) i * 0.19f) * 0.3f;
        for (size_t i = 0; i < b.size(); ++i) b[i] = std::cos((float) i * 0.23f) * 0.3f;
        std::vector<float> got, want;
        for (int side = 0; side < 2; ++side) {
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Km, Mm);
            ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Km, Nm);
            ggml_tensor * Y = ggml_mul_mat(ctx, A, B);
            ggml_cgraph * gf = ggml_new_graph_custom(ctx, 128, false);
            ggml_build_forward_expand(gf, Y);
            ggml_backend_t be = side ? cpu : backend;
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
            ggml_backend_tensor_set(A, a.data(), 0, ggml_nbytes(A));
            ggml_backend_tensor_set(B, b.data(), 0, ggml_nbytes(B));
            ggml_backend_graph_compute(be, gf);
            std::vector<float> out(ggml_nelements(Y));
            ggml_backend_tensor_get(Y, out.data(), 0, ggml_nbytes(Y));
            dump_out(side, be, out);
            (side ? want : got) = std::move(out);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
        fprintf(stderr, "tts event=parity op=mul_mat_plain n=%d max_rel=%.3e\n",
                (int) want.size(), rel_err(got, want));
        fflush(stderr);
    }

    ggml_backend_free(cpu);
}

} // namespace

struct Engine::Impl {
    EngineOptions opts;

    chatterbox_model     model{};
    ggml_gallocr_t       allocr = nullptr;
    std::thread          s3gen_preload_thread;

    // Baked voice-conditioning state.  Populated at construction when
    // `reference_audio` or `voice_dir` is set, then reused by every
    // synthesize() call so we never re-run VoiceEncoder / CAMPPlus /
    // S3TokenizerV2 / mel extraction more than once.
    bool                 voice_overridden = false;
    std::vector<float>   s3gen_prompt_feat;
    int                  s3gen_prompt_feat_rows = 0;
    std::vector<float>   s3gen_embedding;
    std::vector<int32_t> s3gen_prompt_token;

    std::atomic<bool>    cancel_flag{false};

    explicit Impl(const EngineOptions & o)
        : opts(o) {
        if (opts.t3_gguf_path.empty()) {
            throw std::runtime_error("Engine: t3_gguf_path is required");
        }
        if (opts.s3gen_gguf_path.empty()) {
            throw std::runtime_error("Engine: s3gen_gguf_path is required");
        }
        if (!std::filesystem::exists(opts.t3_gguf_path)) {
            throw std::runtime_error("Engine: T3 GGUF not found: " + opts.t3_gguf_path);
        }
        if (!std::filesystem::exists(opts.s3gen_gguf_path)) {
            throw std::runtime_error("Engine: S3Gen GGUF not found: " + opts.s3gen_gguf_path);
        }
        if (!opts.reference_audio.empty() &&
            !std::filesystem::exists(opts.reference_audio)) {
            throw std::runtime_error("Engine: reference_audio not found: " + opts.reference_audio);
        }
        if (!opts.voice_dir.empty() &&
            !std::filesystem::is_directory(opts.voice_dir)) {
            throw std::runtime_error("Engine: voice_dir not found: " + opts.voice_dir);
        }

        ggml_time_init();
        g_log_verbose = opts.verbose ? 1 : 0;
        ggml_log_set(chatterbox_log_cb, nullptr);

        if (!opts.reference_audio.empty() &&
            !validate_reference_audio(opts.reference_audio)) {
            throw std::runtime_error("Engine: reference_audio failed validation: " + opts.reference_audio);
        }

        if (!load_model_gguf(opts.t3_gguf_path, model, opts.n_ctx, opts.n_gpu_layers)) {
            throw std::runtime_error("Engine: failed to load T3 GGUF: " + opts.t3_gguf_path);
        }

        if (model.hparams.variant != CHBX_VARIANT_TURBO &&
            model.hparams.variant != CHBX_VARIANT_MTL) {
            free_model();
            throw std::runtime_error(
                "Engine: unknown chatterbox.variant in " + opts.t3_gguf_path);
        }

        s3gen_preload_thread = std::thread([path = opts.s3gen_gguf_path,
                                            ngpu = opts.n_gpu_layers]() {
            s3gen_preload(path, ngpu);
        });

        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!allocr) {
            wait_for_preload(s3gen_preload_thread);
            s3gen_unload();
            free_model();
            throw std::runtime_error("Engine: ggml_gallocr_new failed");
        }

        try {
            bake_voice_conditioning();
        } catch (...) {
            wait_for_preload(s3gen_preload_thread);
            s3gen_unload();
            if (allocr) { ggml_gallocr_free(allocr); allocr = nullptr; }
            free_model();
            throw;
        }
        wait_for_preload(s3gen_preload_thread);
        const char * variant = model.hparams.variant == CHBX_VARIANT_MTL ? "t3_mtl" : "t3_turbo";
        fprintf(stderr,
                "tts event=engine variant=%s gpu_layers=%d ctx=%d lang=%s voice_overridden=%d "
                "prompt_tok=%zu feat_rows=%d embed=%zu ref=%s\n",
                variant, opts.n_gpu_layers, model.hparams.n_ctx, opts.language.c_str(),
                (int) voice_overridden, s3gen_prompt_token.size(), s3gen_prompt_feat_rows,
                s3gen_embedding.size(), opts.reference_audio.c_str());
        tts_backend_parity_probe(model.backend);
    }

    ~Impl() {
        wait_for_preload(s3gen_preload_thread);
        // Release the S3Gen cache (which holds its own backend + buffers)
        // BEFORE freeing the T3 backend.  If we don't, the cache's
        // backend resources get torn down by static destructors at
        // process exit, after ggml-metal's global device has already
        // been finalised, tripping its "rsets count == 0" assertion.
        s3gen_unload();
        if (allocr) {
            ggml_gallocr_free(allocr);
            allocr = nullptr;
        }
        free_model();
    }

    Impl(const Impl &)             = delete;
    Impl & operator=(const Impl &) = delete;

    void free_model() {
        // Pull (buffer_stack, ctx_stack) out of the process-wide t3_stack_registry
        // BEFORE freeing them locally — otherwise the atexit hook installed
        // by load_model_gguf_mtl on non-CPU backends would later double-free
        // (or, worse, ggml_backend_buffer_free a buffer whose backend has
        // already been destroyed below).  Mirrors the free_t3 lambda in
        // src/chatterbox_cli.cpp.  No-op on CPU backends and on Turbo
        // (those code paths never allocate buffer_stack / ctx_stack).
        if (model.buffer_stack || model.ctx_stack) {
            t3_stack_unregister(model.buffer_stack, model.ctx_stack);
        }
        if (model.buffer_w)        { ggml_backend_buffer_free(model.buffer_w);        model.buffer_w        = nullptr; }
        if (model.buffer_kv)       { ggml_backend_buffer_free(model.buffer_kv);       model.buffer_kv       = nullptr; }
        if (model.buffer_stack)    { ggml_backend_buffer_free(model.buffer_stack);    model.buffer_stack    = nullptr; }
        if (model.buffer_override) { ggml_backend_buffer_free(model.buffer_override); model.buffer_override = nullptr; }
        if (model.backend)         { ggml_backend_free(model.backend);                model.backend         = nullptr; }
        if (model.ctx_w)           { ggml_free(model.ctx_w);                          model.ctx_w           = nullptr; }
        if (model.ctx_kv)          { ggml_free(model.ctx_kv);                         model.ctx_kv          = nullptr; }
        if (model.ctx_stack)       { ggml_free(model.ctx_stack);                      model.ctx_stack       = nullptr; }
        if (model.ctx_override)    { ggml_free(model.ctx_override);                   model.ctx_override    = nullptr; }
    }

    // Loads speaker_emb + cond_prompt_speech_tokens from voice_dir when
    // available, computes them from reference_audio otherwise, and writes
    // the results into model.builtin_speaker_emb / builtin_cond_prompt_tokens
    // so subsequent T3 graphs pick up the cloned voice.  Also stashes the
    // three S3Gen-side tensors (prompt_feat, embedding, prompt_token) on
    // the Impl for reuse in synthesize().
    void bake_voice_conditioning() {
        if (opts.reference_audio.empty() && opts.voice_dir.empty()) {
            return;
        }

        const int n_threads = resolve_thread_count(opts.n_threads);

        bool have_se = false;
        bool have_ct = false;
        std::vector<float>   se_data;
        std::vector<int32_t> ct_data;

        if (!opts.voice_dir.empty()) {
            const std::string se_path = opts.voice_dir + "/speaker_emb.npy";
            const std::string ct_path = opts.voice_dir + "/cond_prompt_speech_tokens.npy";
            const std::string emb_path = opts.voice_dir + "/embedding.npy";
            const std::string pt_path  = opts.voice_dir + "/prompt_token.npy";
            const std::string pf_path  = opts.voice_dir + "/prompt_feat.npy";

            if (std::filesystem::exists(se_path)) {
                npy_array a = npy_load(se_path);
                se_data.assign((const float *) a.data.data(),
                               (const float *) a.data.data() + a.n_elements());
                have_se = true;
            }
            if (std::filesystem::exists(ct_path)) {
                npy_array a = npy_load(ct_path);
                ct_data.assign((const int32_t *) a.data.data(),
                               (const int32_t *) a.data.data() + a.n_elements());
                have_ct = true;
            }
            if (std::filesystem::exists(emb_path)) {
                npy_array a = npy_load(emb_path);
                s3gen_embedding.assign((const float *) a.data.data(),
                                      (const float *) a.data.data() + a.n_elements());
            }
            if (std::filesystem::exists(pt_path)) {
                npy_array a = npy_load(pt_path);
                s3gen_prompt_token.assign((const int32_t *) a.data.data(),
                                          (const int32_t *) a.data.data() + a.n_elements());
            }
            if (std::filesystem::exists(pf_path)) {
                npy_array a = npy_load(pf_path);
                s3gen_prompt_feat.assign((const float *) a.data.data(),
                                         (const float *) a.data.data() + a.n_elements());
                // prompt_feat.npy shape is (T_mel, 80)
                if (a.shape.size() >= 1) {
                    s3gen_prompt_feat_rows = (int) a.shape[0];
                }
            }
        }

        if (!have_se && !opts.reference_audio.empty()) {
            voice_encoder_weights vew;
            if (!voice_encoder_load(opts.t3_gguf_path, vew)) {
                throw std::runtime_error("Engine: VoiceEncoder weights unavailable");
            }
            std::vector<float> wav;
            int sr = 0;
            if (!wav_load(opts.reference_audio, wav, sr)) {
                throw std::runtime_error("Engine: failed to load reference_audio");
            }
            fprintf(stderr,
                    "tts event=voice_bake stage=se_wav ref=%s sr=%d samples=%zu seconds=%.3f\n",
                    opts.reference_audio.c_str(), sr, wav.size(),
                    sr > 0 ? (double) wav.size() / (double) sr : 0.0);
            const double lufs_in = measure_lufs(wav, sr);
            normalise_lufs(wav, sr, -27.0);
            fprintf(stderr,
                    "tts event=voice_bake stage=se_lufs in=%.3f out=%.3f target=-27.000\n",
                    lufs_in, measure_lufs(wav, sr));
            if (sr != 16000) {
                const size_t before = wav.size();
                wav = resample_sinc(wav, sr, 16000);
                fprintf(stderr,
                        "tts event=voice_bake stage=se_resample from=%d to=16000 before=%zu after=%zu\n",
                        sr, before, wav.size());
            }
            const size_t kVeMaxSamples = (size_t) 30 * 16000;
            if (wav.size() > kVeMaxSamples) {
                fprintf(stderr,
                        "tts event=voice_bake stage=se_cap seconds=30 before=%zu after=%zu\n",
                        wav.size(), kVeMaxSamples);
                wav.resize(kVeMaxSamples);
            }
            if (!voice_encoder_embed(wav, vew, model.backend, se_data)) {
                throw std::runtime_error("Engine: VoiceEncoder forward failed");
            }
            tts_log_vec("se_ref", se_data);
            have_se = true;
        }

        std::vector<int32_t> prompt_token_from_ref;
        if (!have_ct && !opts.reference_audio.empty()) {
            std::vector<int32_t> cond_tokens;
            if (!compute_speech_tokens_native(
                    opts.reference_audio, opts.s3gen_gguf_path,
                    /*max_cond_tokens=*/ model.hparams.cond_prompt_len,
                    prompt_token_from_ref, cond_tokens,
                    n_threads, /*backend=*/ model.backend, opts.verbose)) {
                throw std::runtime_error("Engine: S3TokenizerV2 reference tokens failed");
            }
            ct_data = std::move(cond_tokens);
            have_ct = true;
            if (prompt_token_from_ref.empty() || ct_data.empty()) {
                throw std::runtime_error("Engine: S3TokenizerV2 returned empty conditioning");
            }
            tts_log_tokens("ct_ref", ct_data);
            {
                int32_t mn = *std::min_element(ct_data.begin(), ct_data.end());
                int32_t mx = *std::max_element(ct_data.begin(), ct_data.end());
                std::set<int32_t> uniq(ct_data.begin(), ct_data.end());
                fprintf(stderr, "tts event=tokens tag=ct_stats n=%zu min=%d max=%d distinct=%zu\n",
                        ct_data.size(), (int) mn, (int) mx, uniq.size());
            }
        }

        if (have_se) {
            if ((int64_t) se_data.size() != ggml_nelements(model.builtin_speaker_emb)) {
                throw std::runtime_error(
                    "Engine: speaker_emb size mismatch with builtin tensor");
            }
            std::vector<float> se_builtin(ggml_nelements(model.builtin_speaker_emb));
            ggml_backend_tensor_get(
                model.builtin_speaker_emb, se_builtin.data(), 0,
                ggml_nbytes(model.builtin_speaker_emb));
            tts_log_vec("se_builtin", se_builtin);
            ggml_backend_tensor_set(
                model.builtin_speaker_emb, se_data.data(), 0,
                ggml_nbytes(model.builtin_speaker_emb));
            voice_overridden = true;
            std::vector<float> se_readback(se_data.size());
            ggml_backend_tensor_get(
                model.builtin_speaker_emb, se_readback.data(), 0,
                ggml_nbytes(model.builtin_speaker_emb));
            tts_log_vec("se_readback", se_readback);
            fprintf(stderr,
                    "tts event=voice_bake stage=se_write mismatches=%d dims=%zu\n",
                    tts_count_mismatch(se_data, se_readback), se_data.size());
        }

        if (have_ct) {
            if ((int64_t) ct_data.size() == ggml_nelements(model.builtin_cond_prompt_tokens)) {
                ggml_backend_tensor_set(
                    model.builtin_cond_prompt_tokens, ct_data.data(), 0,
                    ggml_nbytes(model.builtin_cond_prompt_tokens));
            } else {
                ggml_init_params op = { ggml_tensor_overhead() * 2, nullptr, true };
                model.ctx_override = ggml_init(op);
                if (!model.ctx_override) {
                    throw std::runtime_error("Engine: ggml_init(ctx_override) failed");
                }
                ggml_tensor * new_ct = ggml_new_tensor_1d(
                    model.ctx_override, GGML_TYPE_I32, (int64_t) ct_data.size());
                ggml_set_name(new_ct,
                              "chatterbox/builtin/cond_prompt_speech_tokens_override");
                model.buffer_override = ggml_backend_alloc_ctx_tensors(
                    model.ctx_override, model.backend);
                if (!model.buffer_override) {
                    throw std::runtime_error("Engine: alloc override buffer failed");
                }
                ggml_backend_tensor_set(
                    new_ct, ct_data.data(), 0, ct_data.size() * sizeof(int32_t));
                model.builtin_cond_prompt_tokens = new_ct;
                model.hparams.cond_prompt_len = (int32_t) ct_data.size();
            }
            voice_overridden = true;
        }

        if (!opts.reference_audio.empty()) {
            if (s3gen_prompt_feat.empty()) {
                int rows = 0;
                if (!compute_prompt_feat_native(
                        opts.reference_audio, opts.s3gen_gguf_path,
                        s3gen_prompt_feat, rows, opts.verbose)) {
                    throw std::runtime_error("Engine: prompt_feat from reference_audio failed");
                }
                if (rows < 1 || s3gen_prompt_feat.empty()) {
                    throw std::runtime_error("Engine: prompt_feat is empty");
                }
                s3gen_prompt_feat_rows = rows;
            }
            tts_log_vec("s3_prompt_feat", s3gen_prompt_feat);
            fprintf(stderr, "tts event=voice_bake stage=prompt_feat rows=%d\n",
                    s3gen_prompt_feat_rows);
            if (s3gen_embedding.empty()) {
                if (!compute_embedding_native(
                        opts.reference_audio, opts.s3gen_gguf_path,
                        s3gen_embedding,
                        /*backend=*/ model.backend, opts.verbose)) {
                    throw std::runtime_error("Engine: CAMPPlus embedding failed");
                }
                if (s3gen_embedding.empty()) {
                    throw std::runtime_error("Engine: CAMPPlus embedding is empty");
                }
            }
            tts_log_vec("campplus_emb", s3gen_embedding);
            if (s3gen_prompt_token.empty() && !prompt_token_from_ref.empty()) {
                s3gen_prompt_token = std::move(prompt_token_from_ref);
            }
            if (s3gen_prompt_token.empty()) {
                throw std::runtime_error("Engine: prompt_token unavailable");
            }
            tts_log_tokens("s3_prompt_token", s3gen_prompt_token);
        tts_dump_tokens("s3_prompt_token", s3gen_prompt_token);
            if (s3gen_prompt_feat.empty()) {
                throw std::runtime_error("Engine: prompt_feat unavailable");
            }
            if (s3gen_embedding.empty()) {
                throw std::runtime_error("Engine: embedding unavailable");
            }
        }
    }

    std::vector<int32_t> run_t3(const std::string & text) {
        const int n_threads = resolve_thread_count(opts.n_threads);
        std::mt19937 rng(opts.seed);
        chatterbox_sampling_params sp;
        sp.top_k          = opts.top_k;
        sp.top_p          = opts.top_p;
        sp.temp           = opts.temperature;
        sp.repeat_penalty = opts.repeat_penalty;
        sp.min_p          = opts.min_p;
        sp.cfg_weight     = opts.cfg_weight;

        std::vector<int32_t> text_tokens;
        if (model.hparams.variant == CHBX_VARIANT_MTL) {
            if (model.mtl_tokenizer_json.empty()) {
                throw std::runtime_error("Engine: MTL GGUF has no tokenizer json");
            }
            mtl_tokenizer tok;
            if (!tok.load_from_json(model.mtl_tokenizer_json)) {
                throw std::runtime_error("Engine: MTL tokenizer json failed to load");
            }
            text_tokens = tok.encode(text, opts.language);
            // Python ChatterboxMultilingualTTS.generate and tts-cli pad
            // start_text_token (255) + ids + stop_text_token (0). The Engine
            // path used by Trident omitted this; German/Polish then start on
            // the [de]/[pl] language token with no SOT, which collapses the
            // decode into English-accented garbage.
            std::vector<int32_t> padded;
            padded.reserve(text_tokens.size() + 2);
            padded.push_back(model.hparams.start_text_token);
            padded.insert(padded.end(), text_tokens.begin(), text_tokens.end());
            padded.push_back(model.hparams.stop_text_token);
            fprintf(stderr, "tts event=text_tokens tokenizer=mtl lang=%s raw=%zu padded=%zu sot=%d eot=%d ids=",
                    opts.language.c_str(), text_tokens.size(), padded.size(),
                    model.hparams.start_text_token, model.hparams.stop_text_token);
            const size_t show = padded.size() < 12 ? padded.size() : 12;
            for (size_t i = 0; i < show; ++i) fprintf(stderr, "%d ", padded[i]);
            fprintf(stderr, "%s\n", padded.size() > show ? "..." : "");
            text_tokens = std::move(padded);
        } else {
            if (model.tok_tokens.empty()) {
                throw std::runtime_error(
                    "Engine: T3 GGUF has no embedded tokenizer; "
                    "re-run scripts/convert-t3-turbo-to-gguf.py");
            }
            gpt2_bpe bpe;
            bpe.load_from_arrays(model.tok_tokens, model.tok_merges);
            text_tokens = bpe.tokenize(gpt2_bpe::punc_norm(text));
            fprintf(stderr, "tts event=text_tokens tokenizer=gpt2 tokens=%zu\n", text_tokens.size());
        }
        if (text_tokens.empty()) {
            throw std::runtime_error("Engine: text tokenised to empty sequence");
        }
        tts_log_tokens("text", text_tokens);
        tts_dump_tokens("text", text_tokens);

        if (model.hparams.variant == CHBX_VARIANT_MTL) {
            std::vector<float> logits_c, logits_u;
            int prompt_len = 0;
            if (!eval_prompt_mtl(model, allocr, n_threads, text_tokens,
                                 opts.exaggeration, logits_c, logits_u, prompt_len)) {
                throw std::runtime_error("Engine: T3 MTL prompt eval failed");
            }
            int n_past = prompt_len;
            std::vector<int32_t> generated;
            generated.reserve((size_t) opts.n_predict + 1);
            int32_t current = sample_next_token_mtl(
                logits_c, logits_u, generated, sp, rng, model.hparams.stop_speech_token);
            generated.push_back(current);
            fprintf(stderr, "tts event=tok step=0 id=%d\n", (int) current);
            for (int i = 0; i < opts.n_predict; ++i) {
                if (cancel_flag.load(std::memory_order_relaxed)) {
                    throw std::runtime_error("Engine: synthesis cancelled during T3 decode");
                }
                if (current == model.hparams.stop_speech_token) break;
                if (n_past + 1 > model.hparams.n_ctx) break;
                if (!eval_step_mtl(model, allocr, n_threads, n_past, current, logits_c, logits_u)) {
                    throw std::runtime_error("Engine: T3 MTL step eval failed");
                }
                ++n_past;
                current = sample_next_token_mtl(
                    logits_c, logits_u, generated, sp, rng, model.hparams.stop_speech_token);
                generated.push_back(current);
                fprintf(stderr, "tts event=tok step=%d id=%d\n", i + 1, (int) current);
            }
            const bool eos = !generated.empty() && generated.back() == model.hparams.stop_speech_token;
            const bool cap = !eos && (int) generated.size() >= opts.n_predict;
            fprintf(stderr, "tts event=t3_decode speech_tokens=%zu eos=%d cap=%d max=%d\n",
                    generated.size(), (int) eos, (int) cap, opts.n_predict);
            if (eos) generated.pop_back();
            // Official MTL: the token just before EOS decodes to ~40 ms of noise.
            if (generated.size() > 1) generated.pop_back();
            tts_log_tokens("speech_final_mtl", generated);
            tts_dump_tokens("speech_final_mtl", generated);
            return generated;
        }

        std::vector<float> logits;
        int prompt_len = 0;
        if (!eval_prompt(model, allocr, n_threads, text_tokens, logits, prompt_len)) {
            throw std::runtime_error("Engine: T3 prompt eval failed");
        }

        int n_past = prompt_len;
        std::vector<int32_t> generated;
        generated.reserve((size_t) opts.n_predict + 1);

        int32_t current = sample_next_token_ex(logits, generated, sp, rng);
        generated.push_back(current);
        fprintf(stderr, "tts event=tok step=0 id=%d\n", (int) current);

        for (int i = 0; i < opts.n_predict; ++i) {
            if (cancel_flag.load(std::memory_order_relaxed)) {
                throw std::runtime_error("Engine: synthesis cancelled during T3 decode");
            }
            if (current == model.hparams.stop_speech_token) break;
            if (n_past + 1 > model.hparams.n_ctx) break;
            if (!eval_step(model, allocr, n_threads, n_past, current, logits)) {
                throw std::runtime_error("Engine: T3 step eval failed");
            }
            ++n_past;
            current = sample_next_token_ex(logits, generated, sp, rng);
            generated.push_back(current);
            fprintf(stderr, "tts event=tok step=%d id=%d\n", i + 1, (int) current);
        }

        {
            const bool eos = !generated.empty() && generated.back() == model.hparams.stop_speech_token;
            const bool cap = !eos && (int) generated.size() >= opts.n_predict;
            fprintf(stderr, "tts event=t3_decode speech_tokens=%zu eos=%d cap=%d max=%d\n",
                    generated.size(), (int) eos, (int) cap, opts.n_predict);
            if (eos) generated.pop_back();
        }
        {
            const int32_t oov = model.hparams.start_speech_token;
            size_t w = 0;
            for (size_t r = 0; r < generated.size(); ++r)
                if (generated[r] >= 0 && generated[r] < oov) generated[w++] = generated[r];
            generated.resize(w);
        }
        tts_log_tokens("speech_final", generated);
        tts_dump_tokens("speech_final", generated);
        return generated;
    }

    // Populate the fixed fields of s3gen_synthesize_opts (paths, threads,
    // seed, voice overrides, backend hints).  Streaming-specific fields
    // (finalize, hift_cache_source, skip_mel_frames, ...) are set per
    // chunk by `synthesize_streaming` below.
    void fill_common_s3gen_opts(s3gen_synthesize_opts & sopts) {
        sopts.s3gen_gguf_path = opts.s3gen_gguf_path;
        sopts.out_wav_path    = "";
        sopts.seed            = opts.seed;
        sopts.n_threads       = resolve_thread_count(opts.n_threads);
        sopts.verbose         = opts.verbose;
        sopts.n_gpu_layers    = opts.n_gpu_layers;

        if (!s3gen_prompt_feat.empty()) {
            sopts.prompt_feat_override      = s3gen_prompt_feat;
            sopts.prompt_feat_rows_override = s3gen_prompt_feat_rows;
        }
        if (!s3gen_embedding.empty()) {
            sopts.embedding_override = s3gen_embedding;
        }
        if (!s3gen_prompt_token.empty()) {
            sopts.prompt_token_override = s3gen_prompt_token;
        }
    }

    SynthesisResult synthesize_batch(const std::vector<int32_t> & speech_tokens,
                                     SynthesisResult && partial) {
        s3gen_synthesize_opts sopts;
        fill_common_s3gen_opts(sopts);
        sopts.cfm_steps = opts.cfm_steps;

        SynthesisResult result = std::move(partial);
        sopts.pcm_out = &result.pcm;

        const auto s3_t0 = std::chrono::steady_clock::now();
        const int rc = s3gen_synthesize_to_wav(speech_tokens, sopts);
        const auto s3_t1 = std::chrono::steady_clock::now();
        if (rc != 0) {
            throw std::runtime_error("Engine: s3gen_synthesize_to_wav failed with code "
                                     + std::to_string(rc));
        }

        result.sample_rate   = 24000;
        result.t3_tokens     = (int) speech_tokens.size();
        result.audio_samples = (int) result.pcm.size();
        result.s3gen_ms      = std::chrono::duration<double, std::milli>(s3_t1 - s3_t0).count();
        return result;
    }

    // Ports main.cpp's --stream-chunk-tokens loop.  Splits speech_tokens
    // into chunks of stream_chunk_tokens (with an optional smaller first
    // chunk), carries `hift_cache_source` and `skip_mel_frames` across
    // chunks for phase-continuous seams, and invokes `on_chunk` with
    // each chunk's PCM as it's produced.  Accumulates the full PCM in
    // the returned SynthesisResult so batch callers get the same shape.
    SynthesisResult synthesize_streaming(
        const std::vector<int32_t> & speech_tokens,
        const StreamCallback & on_chunk,
        SynthesisResult && partial) {

        std::vector<int32_t> seg_toks = speech_tokens;
        for (int i = 0; i < kS3GenLookaheadTokens; ++i) {
            seg_toks.push_back(kS3GenSilenceToken);
        }
        const int total_n = (int) seg_toks.size();

        const int chunk_n       = opts.stream_chunk_tokens;
        const int first_chunk_n = opts.stream_first_chunk_tokens > 0
                                    ? opts.stream_first_chunk_tokens
                                    : chunk_n;

        std::vector<int> boundaries = {0};
        int cursor = std::min(first_chunk_n, total_n);
        boundaries.push_back(cursor);
        while (cursor < total_n) {
            cursor = std::min(cursor + chunk_n, total_n);
            boundaries.push_back(cursor);
        }
        // Absorb a tiny trailing chunk into the previous one (avoids
        // paying the full encoder+CFM cost for a handful of new tokens;
        // matches main.cpp's tail-merge heuristic).
        const int min_tail = std::max(6, chunk_n / 3);
        if (boundaries.size() >= 3) {
            const int tail_len = boundaries.back() - boundaries[boundaries.size() - 2];
            if (tail_len < min_tail) boundaries.erase(boundaries.end() - 2);
        }

        std::vector<float> hift_cache_source;
        int prev_mels_emitted = 0;

        SynthesisResult result = std::move(partial);
        result.pcm.clear();

        const int n_chunks = (int) boundaries.size() - 1;
        double s3gen_ms_total = 0.0;

        for (int k = 1; k <= n_chunks; ++k) {
            if (cancel_flag.load(std::memory_order_relaxed)) {
                throw std::runtime_error("Engine: synthesis cancelled during streaming");
            }
            const int end              = boundaries[k];
            const bool is_last_in_seg  = (end == total_n);
            std::vector<int32_t> toks(seg_toks.begin(), seg_toks.begin() + end);

            s3gen_synthesize_opts copts;
            fill_common_s3gen_opts(copts);
            std::vector<float> chunk_pcm;
            copts.pcm_out                   = &chunk_pcm;
            copts.append_lookahead_silence  = false;
            copts.finalize                  = is_last_in_seg;
            copts.skip_mel_frames           = prev_mels_emitted;
            copts.apply_trim_fade           = (k == 1);
            copts.hift_cache_source         = hift_cache_source;
            std::vector<float> tail_out;
            copts.hift_source_tail_out      = &tail_out;
            copts.source_tail_samples       = 480;
            copts.cfm_steps                 = opts.stream_cfm_steps;

            const auto s3_t0 = std::chrono::steady_clock::now();
            const int rc = s3gen_synthesize_to_wav(toks, copts);
            const auto s3_t1 = std::chrono::steady_clock::now();
            if (rc != 0) {
                throw std::runtime_error(
                    "Engine: streaming chunk " + std::to_string(k) +
                    " failed with code " + std::to_string(rc));
            }
            s3gen_ms_total += std::chrono::duration<double, std::milli>(s3_t1 - s3_t0).count();

            on_chunk(chunk_pcm.data(), chunk_pcm.size(), k - 1, is_last_in_seg);

            result.pcm.insert(result.pcm.end(), chunk_pcm.begin(), chunk_pcm.end());
            hift_cache_source = std::move(tail_out);
            const size_t chunk_samples = chunk_pcm.size();
            prev_mels_emitted += (int)(chunk_samples / 480);
        }

        result.sample_rate   = 24000;
        result.t3_tokens     = (int) speech_tokens.size();
        result.audio_samples = (int) result.pcm.size();
        result.s3gen_ms      = s3gen_ms_total;
        return result;
    }

    SynthesisResult synthesize(const std::string & text,
                               const StreamCallback & on_chunk) {
        if (text.empty()) {
            throw std::runtime_error("Engine: text is empty");
        }
        cancel_flag.store(false, std::memory_order_relaxed);

        const auto t3_t0 = std::chrono::steady_clock::now();
        std::vector<int32_t> speech_tokens = run_t3(text);
        const auto t3_t1 = std::chrono::steady_clock::now();

        wait_for_preload(s3gen_preload_thread);

        SynthesisResult partial;
        partial.t3_ms = std::chrono::duration<double, std::milli>(t3_t1 - t3_t0).count();

        const bool use_streaming = on_chunk && opts.stream_chunk_tokens > 0;
        return use_streaming
            ? synthesize_streaming(speech_tokens, on_chunk, std::move(partial))
            : synthesize_batch(speech_tokens, std::move(partial));
    }

    SynthesisResult synthesize(const std::string & text) {
        return synthesize(text, StreamCallback{});
    }
};

Engine::Engine(const EngineOptions & opts)
    : pimpl_(std::make_unique<Impl>(opts)) {}

Engine::~Engine() = default;
Engine::Engine(Engine &&) noexcept            = default;
Engine & Engine::operator=(Engine &&) noexcept = default;

SynthesisResult Engine::synthesize(const std::string & text) {
    return pimpl_->synthesize(text);
}

SynthesisResult Engine::synthesize(const std::string & text,
                                   const StreamCallback & on_chunk) {
    return pimpl_->synthesize(text, on_chunk);
}

void Engine::cancel() {
    if (pimpl_) pimpl_->cancel_flag.store(true, std::memory_order_relaxed);
}

const EngineOptions & Engine::options() const {
    return pimpl_->opts;
}

} // namespace tts_cpp::chatterbox
