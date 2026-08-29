#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <limits>
static void reflect_pad_1d(const std::vector<float> & in, int p_left, int p_right,
                           std::vector<float> & out)
{
    const int L = (int)in.size();
    out.resize((size_t)(L + p_left + p_right));
    for (int i = 0; i < p_left; ++i) {
        const int src = p_left - i;
        out[i] = (src >= 0 && src < L) ? in[src] : 0.0f;
    }
    if (L > 0) std::memcpy(out.data() + p_left, in.data(), (size_t)L * sizeof(float));
    for (int i = 0; i < p_right; ++i) {
        const int src = L - 2 - i;
        out[(size_t)(L + p_left + i)] = (src >= 0 && src < L) ? in[src] : 0.0f;
    }
}
namespace {
struct ggml_ctx {
    ggml_backend_t backend = nullptr;
    bool owns_backend = false;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t weights_buf = nullptr;
    ggml_gallocr_t alloc = nullptr;
};
static void ctx_free(ggml_ctx & c) {
    if (c.alloc)       { ggml_gallocr_free(c.alloc);              c.alloc = nullptr; }
    if (c.weights_buf) { ggml_backend_buffer_free(c.weights_buf); c.weights_buf = nullptr; }
    if (c.ctx)         { ggml_free(c.ctx);                        c.ctx = nullptr; }
    if (c.backend && c.owns_backend) ggml_backend_free(c.backend);
    c.backend = nullptr;
}
static void make_dft_basis(int n_fft, int F,
                           std::vector<float> & cos_basis,
                           std::vector<float> & neg_sin_basis)
{
    cos_basis.resize((size_t) F * n_fft);
    neg_sin_basis.resize((size_t) F * n_fft);
    for (int k = 0; k < F; ++k) {
        for (int n = 0; n < n_fft; ++n) {
            const double th = 2.0 * M_PI * (double) k * (double) n / (double) n_fft;
            cos_basis[(size_t) k * n_fft + n]     = (float) std::cos(th);
            neg_sin_basis[(size_t) k * n_fft + n] = -(float) std::sin(th);
        }
    }
}
}
static std::vector<float> build_windowed_frames(
    const std::vector<float> & src_signal, int T, int hop, int win, int n_fft,
    const std::vector<float> & window)
{
    std::vector<float> frames((size_t) T * n_fft, 0.0f);
    for (int t = 0; t < T; ++t) {
        const float * x = src_signal.data() + (size_t) t * hop;
        float * f       = frames.data() + (size_t) t * n_fft;
        for (int n = 0; n < win; ++n) f[n] = x[n] * window[n];
    }
    return frames;
}
static std::vector<float> build_kaldi_frames(
    const std::vector<float> & wav, int T, int hop,
    int frame_len, int n_fft,
    const std::vector<float> & povey, float preemph)
{
    std::vector<float> frames((size_t) T * n_fft, 0.0f);
    for (int t = 0; t < T; ++t) {
        const float * src = wav.data() + (size_t) t * hop;
        float * f         = frames.data() + (size_t) t * n_fft;
        for (int n = 0; n < frame_len; ++n) f[n] = src[n];
        double acc = 0.0;
        for (int n = 0; n < frame_len; ++n) acc += f[n];
        const float dc = (float) (acc / frame_len);
        for (int n = 0; n < frame_len; ++n) f[n] -= dc;
        for (int n = frame_len - 1; n >= 1; --n) {
            f[n] = f[n] - preemph * f[n - 1];
        }
        f[0] = f[0] * (1.0f - preemph);
        for (int n = 0; n < frame_len; ++n) f[n] *= povey[n];
    }
    return frames;
}
static std::vector<float> mel_graph_run(
    const std::vector<float> & frames_TC,
    const std::vector<float> & mel_fb,
    int T, int n_fft, int F, int n_mels,
    float power_exp, float log_floor)
{
    const std::vector<float> * frames = &frames_TC;
    std::vector<float> cos_basis, neg_sin_basis;
    make_dft_basis(n_fft, F, cos_basis, neg_sin_basis);
    ggml_ctx gc;
    gc.backend      = ggml_backend_cpu_init();
    gc.owns_backend = true;
    if (!gc.backend) {
        fprintf(stderr, "mel_graph_run: ggml_backend_cpu_init failed\n");
        return {};
    }
    ggml_init_params ip = {
         64 * ggml_tensor_overhead(),
         nullptr,
         true,
    };
    gc.ctx = ggml_init(ip);
    ggml_tensor * t_frames   = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, n_fft, T);
    ggml_tensor * t_cos      = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, n_fft, F);
    ggml_tensor * t_neg_sin  = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, n_fft, F);
    ggml_tensor * t_mel_fb   = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, F, n_mels);
    ggml_set_name(t_frames,  "frames");  ggml_set_input(t_frames);
    ggml_set_name(t_cos,     "cos");
    ggml_set_name(t_neg_sin, "nsin");
    ggml_set_name(t_mel_fb,  "mel_fb");
    gc.weights_buf = ggml_backend_alloc_ctx_tensors(gc.ctx, gc.backend);
    if (!gc.weights_buf) {
        fprintf(stderr, "mel_graph_run: weights alloc failed\n"); ctx_free(gc); return {};
    }
    ggml_backend_tensor_set(t_cos,     cos_basis.data(),     0, cos_basis.size()     * sizeof(float));
    ggml_backend_tensor_set(t_neg_sin, neg_sin_basis.data(), 0, neg_sin_basis.size() * sizeof(float));
    ggml_backend_tensor_set(t_mel_fb,  mel_fb.data(),        0, mel_fb.size()        * sizeof(float));
    const int max_nodes = 32;
    const size_t buf_size = ggml_tensor_overhead() * max_nodes +
                            ggml_graph_overhead_custom(max_nodes, false);
    static std::vector<uint8_t> buf;
    buf.resize(buf_size);
    ggml_init_params gp = { buf_size, buf.data(),  true };
    ggml_context * gctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, max_nodes, false);
    ggml_tensor * spec_re = ggml_mul_mat(gctx, t_cos,     t_frames);
    ggml_tensor * spec_im = ggml_mul_mat(gctx, t_neg_sin, t_frames);
    ggml_tensor * re2 = ggml_sqr(gctx, spec_re);
    ggml_tensor * im2 = ggml_sqr(gctx, spec_im);
    ggml_tensor * pow_ = ggml_add(gctx, re2, im2);
    ggml_tensor * mag = pow_;
    if (power_exp == 1.0f) {
        mag = ggml_sqrt(gctx, pow_);
    }
    ggml_tensor * mel_FT = ggml_mul_mat(gctx, t_mel_fb, mag);
    ggml_tensor * out = mel_FT;
    if (log_floor > 0.0f) {
        ggml_tensor * clamped = ggml_clamp(gctx, out, log_floor, 1e30f);
        out = ggml_log(gctx, clamped);
    }
    ggml_set_name(out, "out"); ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    gc.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gc.backend));
    if (!gc.alloc || !ggml_gallocr_reserve(gc.alloc, gf) ||
        !ggml_gallocr_alloc_graph(gc.alloc, gf))
    {
        fprintf(stderr, "mel_graph_run: graph alloc failed\n");
        ggml_free(gctx); ctx_free(gc); return {};
    }
    ggml_backend_tensor_set(t_frames, frames->data(), 0, frames->size() * sizeof(float));
    if (ggml_backend_graph_compute(gc.backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "mel_graph_run: compute failed\n");
        ggml_free(gctx); ctx_free(gc); return {};
    }
    std::vector<float> out_TM((size_t) T * n_mels);
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "out"),
                            out_TM.data(), 0, out_TM.size() * sizeof(float));
    ggml_free(gctx);
    ctx_free(gc);
    return out_TM;
}
std::vector<float> mel_extract_stft_hann_ggml(
    const std::vector<float> & wav,
    const std::vector<float> & mel_fb,
    int n_fft, int hop, int win, int n_mels,
    int center_mode,
    float power_exp,
    float log_floor)
{
    const int F = n_fft / 2 + 1;
    if (mel_fb.size() != (size_t) n_mels * F) {
        fprintf(stderr, "mel_extract_stft_hann_ggml: filterbank has %zu, expected %d\n",
                mel_fb.size(), n_mels * F);
        return {};
    }
    const int pad = (center_mode == 0) ? (n_fft - hop) / 2 : n_fft / 2;
    std::vector<float> padded;
    reflect_pad_1d(wav, pad, pad, padded);
    if ((int) padded.size() < win) return {};
    const int T = (center_mode == 0)
        ? ((int) padded.size() - win) / hop + 1
        : 1 + (int) wav.size() / hop;
    std::vector<float> hann(win);
    for (int n = 0; n < win; ++n) {
        hann[n] = 0.5f * (1.0f - std::cos(2.0f * (float) M_PI * (float) n / (float) win));
    }
    std::vector<float> frames = build_windowed_frames(padded, T, hop, win, n_fft, hann);
    return mel_graph_run(frames, mel_fb, T, n_fft, F, n_mels, power_exp, log_floor);
}
std::vector<float> fbank_kaldi_80_ggml(const std::vector<float> & wav_16k,
                                       const std::vector<float> & mel_fb)
{
    const int n_fft     = 512;
    const int frame_len = 400;
    const int hop       = 160;
    const int n_mels    = 80;
    const int F         = n_fft / 2 + 1;
    const float preemph = 0.97f;
    if (mel_fb.size() != (size_t) n_mels * F) {
        fprintf(stderr, "fbank_kaldi_80_ggml: filterbank has %zu, expected %d\n",
                mel_fb.size(), n_mels * F);
        return {};
    }
    const int L = (int) wav_16k.size();
    if (L < frame_len) return {};
    const int T = (L - frame_len) / hop + 1;
    std::vector<float> povey(frame_len);
    for (int n = 0; n < frame_len; ++n) {
        const double a = 0.5 - 0.5 * std::cos(2.0 * M_PI * (double) n / (double) (frame_len - 1));
        povey[n] = (float) std::pow(a, 0.85);
    }
    std::vector<float> frames = build_kaldi_frames(wav_16k, T, hop, frame_len, n_fft, povey, preemph);
    return mel_graph_run(frames, mel_fb, T, n_fft, F, n_mels,
                         2.0f,
                          std::numeric_limits<float>::epsilon());
}
