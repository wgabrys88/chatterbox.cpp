#include "voice_features.h"
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
std::vector<float> mel_extract_stft_hann_ggml(
    const std::vector<float> & wav,
    const std::vector<float> & mel_fb,
    int n_fft, int hop, int win, int n_mels,
    int center_mode, float power_exp, float log_floor);
std::vector<float> fbank_kaldi_80_ggml(const std::vector<float> & wav_16k,
                                       const std::vector<float> & mel_fb);
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
void normalise_lufs(std::vector<float> & wav, int sr, double target_lufs)
{
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
