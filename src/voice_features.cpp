#include "chatterbox_t3_internal.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <cstring>
#include <limits>
#include <vector>
static uint16_t u16(const unsigned char* p) { return (uint16_t)(p[0] | p[1] << 8); }
static uint32_t u32(const unsigned char* p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }
bool wav_load(const std::string& path, std::vector<float>& out, int& sr) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::vector<unsigned char> b((std::istreambuf_iterator<char>(f)), {});
    if (b.size() < 44 || std::memcmp(b.data(), "RIFF", 4) || std::memcmp(b.data() + 8, "WAVE", 4)) return false;
    uint16_t format = 0, channels = 0, bits = 0, block = 0;
    const unsigned char* data = nullptr;
    size_t bytes = 0;
    for (size_t p = 12; p + 8 <= b.size();) {
        const uint32_t n = u32(b.data() + p + 4);
        if (p + 8ull + n > b.size()) return false;
        const unsigned char* q = b.data() + p + 8;
        if (!std::memcmp(b.data() + p, "fmt ", 4) && n >= 16) {
            format = u16(q); channels = u16(q + 2); sr = (int)u32(q + 4); block = u16(q + 12); bits = u16(q + 14);
            if (format == 0xfffe && n >= 40) format = u16(q + 24);
        }
        if (!std::memcmp(b.data() + p, "data", 4)) { data = q; bytes = n; }
        p += 8 + n + (n & 1u);
    }
    if (!data || !channels || !sr || !block || !bits || bytes % block) return false;
    const size_t frames = bytes / block;
    out.assign(frames, 0.f);
    const size_t sample_bytes = bits / 8;
    if (!sample_bytes || sample_bytes * channels > block) return false;
    for (size_t i = 0; i < frames; ++i) {
        float sum = 0.f;
        for (uint16_t c = 0; c < channels; ++c) {
            const unsigned char* q = data + i * block + c * sample_bytes;
            float v = 0.f;
            if (format == 1 && bits == 8) v = ((int)q[0] - 128) / 128.f;
            else if (format == 1 && bits == 16) v = (int16_t)u16(q) / 32768.f;
            else if (format == 1 && bits == 24) { int32_t x = q[0] | q[1] << 8 | q[2] << 16; if (x & 0x800000) x |= ~0xffffff; v = x / 8388608.f; }
            else if (format == 1 && bits == 32) v = (int32_t)u32(q) / 2147483648.f;
            else if (format == 3 && bits == 32) std::memcpy(&v, q, 4);
            else return false;
            sum += v;
        }
        out[i] = sum / channels;
    }
    return true;
}
static double bessel_i0(double x) {
    double sum = 1.0;
    double term = 1.0;
    double half = 0.5 * x;
    for (int k = 1; k < 30; ++k) {
        term *= (half / (double)k) * (half / (double)k);
        sum += term;
        if (term < 1e-12 * sum) break;
    }
    return sum;
}
std::vector<float> resample_sinc(const std::vector<float> & in,
                                 int sr_in, int sr_out,
                                 int taps_half)
{
    if (sr_in == sr_out) return in;
    if (in.empty()) return {};
    const double fc  = 0.5 * std::min(sr_in, sr_out) / (double)sr_in;
    const double beta = 8.6;
    const double inv_i0_beta = 1.0 / bessel_i0(beta);
    const double rate  = (double)sr_out / (double)sr_in;
    const size_t L_in  = in.size();
    const size_t L_out = (size_t)std::floor((double)L_in * rate);
    std::vector<float> out(L_out, 0.0f);
    for (size_t n = 0; n < L_out; ++n) {
        const double t_in  = (double)n / rate;
        const long long center = (long long)std::floor(t_in);
        const double frac  = t_in - (double)center;
        float acc = 0.0f;
        for (int k = -taps_half; k <= taps_half; ++k) {
            const long long idx = center + k;
            if (idx < 0 || idx >= (long long)L_in) continue;
            const double offset = frac - (double)k;
            const double sinc_arg = 2.0 * M_PI * fc * offset;
            const double sinc = (std::fabs(offset) < 1e-12)
                ? 1.0
                : std::sin(sinc_arg) / sinc_arg;
            const double wrel = offset / (double)taps_half;
            const double win  = (std::fabs(wrel) <= 1.0)
                ? bessel_i0(beta * std::sqrt(1.0 - wrel * wrel)) * inv_i0_beta
                : 0.0;
            acc += (float)(2.0 * fc * sinc * win) * in[(size_t)idx];
        }
        out[n] = acc;
    }
    return out;
}
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

std::vector<float> mel_extract_24k_80(const std::vector<float> & wav_24k,
                                      const std::vector<float> & mel_filterbank)
{
    return mel_extract_stft_hann_ggml(wav_24k, mel_filterbank,
        1920, 480, 1920, 80,
        0, 1.0f, 1e-5f);
}
struct _biquad {
    double b0, b1, b2, a1, a2;
    double x1 = 0.0, x2 = 0.0, y1 = 0.0, y2 = 0.0;
    inline double process(double x) {
        double y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = x1; x1 = x; y2 = y1; y1 = y;
        return y;
    }
};
static _biquad _kweight_shelf(int sr) {
    const double f0 = 1681.97453899761;
    const double G  = 3.999843853973347;
    const double Q  = 0.7071752369554196;
    const double K  = std::tan(M_PI * f0 / (double)sr);
    const double Vh = std::pow(10.0, G / 20.0);
    const double Vb = std::pow(Vh, 0.499666774155719);
    const double a0 = 1.0 + K / Q + K * K;
    _biquad q;
    q.b0 = (Vh + Vb * K / Q + K * K) / a0;
    q.b1 = 2.0 * (K * K - Vh) / a0;
    q.b2 = (Vh - Vb * K / Q + K * K) / a0;
    q.a1 = 2.0 * (K * K - 1.0) / a0;
    q.a2 = (1.0 - K / Q + K * K) / a0;
    return q;
}
static _biquad _kweight_hipass(int sr) {
    const double f0 = 38.13547087602444;
    const double Q  = 0.5003270373238773;
    const double K  = std::tan(M_PI * f0 / (double)sr);
    const double a0 = 1.0 + K / Q + K * K;
    _biquad q;
    q.b0 = 1.0;
    q.b1 = -2.0;
    q.b2 = 1.0;
    q.a1 = 2.0 * (K * K - 1.0) / a0;
    q.a2 = (1.0 - K / Q + K * K) / a0;
    q.b0 /= a0; q.b1 /= a0; q.b2 /= a0;
    return q;
}
double measure_lufs(const std::vector<float> & wav, int sr)
{
    if ((int)wav.size() < (int)(0.4 * sr)) {
        return -std::numeric_limits<double>::infinity();
    }
    std::vector<double> filt(wav.size());
    {
        _biquad s1 = _kweight_shelf(sr);
        _biquad s2 = _kweight_hipass(sr);
        for (size_t i = 0; i < wav.size(); ++i) {
            double y = s1.process((double)wav[i]);
            y        = s2.process(y);
            filt[i]  = y;
        }
    }
    const int block_size = (int)std::round(0.4 * sr);
    const int hop        = (int)std::round(0.1 * sr);
    const int n_blocks   = std::max(0, ((int)filt.size() - block_size) / hop + 1);
    if (n_blocks <= 0) return -std::numeric_limits<double>::infinity();
    std::vector<double> Z(n_blocks);
    std::vector<double> L(n_blocks);
    for (int b = 0; b < n_blocks; ++b) {
        double sum = 0.0;
        const double * p = filt.data() + (size_t)b * hop;
        for (int i = 0; i < block_size; ++i) sum += p[i] * p[i];
        Z[b] = sum / block_size;
        L[b] = -0.691 + 10.0 * std::log10(std::max(Z[b], 1e-30));
    }
    double sum_abs = 0.0;  int n_abs = 0;
    for (int b = 0; b < n_blocks; ++b) {
        if (L[b] >= -70.0) { sum_abs += Z[b]; ++n_abs; }
    }
    if (n_abs == 0) return -std::numeric_limits<double>::infinity();
    double mean_abs = sum_abs / n_abs;
    double L_rel_thresh = -0.691 + 10.0 * std::log10(std::max(mean_abs, 1e-30)) - 10.0;
    double sum_rel = 0.0;  int n_rel = 0;
    for (int b = 0; b < n_blocks; ++b) {
        if (L[b] >= -70.0 && L[b] >= L_rel_thresh) {
            sum_rel += Z[b]; ++n_rel;
        }
    }
    if (n_rel == 0) return -std::numeric_limits<double>::infinity();
    double mean_rel = sum_rel / n_rel;
    return -0.691 + 10.0 * std::log10(std::max(mean_rel, 1e-30));
}
void normalise_lufs(std::vector<float> & wav, int sr)
{
    constexpr double target_lufs = -27.0;
    double loudness = measure_lufs(wav, sr);
    if (!std::isfinite(loudness)) return;
    double gain_db  = target_lufs - loudness;
    double gain_lin = std::pow(10.0, gain_db / 20.0);
    if (!std::isfinite(gain_lin) || gain_lin <= 0.0) return;
    for (float & v : wav) v = (float)((double)v * gain_lin);
}
std::vector<float> mel_extract_16k_40(const std::vector<float> & wav_16k,
                                      const std::vector<float> & mel_filterbank)
{
    return mel_extract_stft_hann_ggml(wav_16k, mel_filterbank,
        400, 160, 400, 40,
        1, 2.0f, -1.0f);
}
std::vector<float> fbank_kaldi_80(const std::vector<float> & wav_16k,
                                  const std::vector<float> & mel_filterbank)
{
    return fbank_kaldi_80_ggml(wav_16k, mel_filterbank);
}
