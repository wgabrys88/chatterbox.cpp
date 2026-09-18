#include "voice_features.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static uint16_t u16(const unsigned char* p) { return (uint16_t)(p[0] | p[1] << 8); }
static uint32_t u32(const unsigned char* p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }
void wav_load(const std::string& path, std::vector<float>& out, int& sr) {
    std::ifstream f(path, std::ios::binary);
    std::vector<unsigned char> b((std::istreambuf_iterator<char>(f)), {});
    if (b.size() < 44 || std::memcmp(b.data(), "RIFF", 4) || std::memcmp(b.data() + 8, "WAVE", 4))
        throw std::runtime_error("WAV");
    uint16_t format = 0, channels = 0, bits = 0, block = 0;
    const unsigned char* data = nullptr;
    size_t bytes = 0;
    for (size_t p = 12; p + 8 <= b.size();) {
        const uint32_t n = u32(b.data() + p + 4);
        if (p + 8ull + n > b.size()) throw std::runtime_error("WAV");
        const unsigned char* q = b.data() + p + 8;
        if (!std::memcmp(b.data() + p, "fmt ", 4) && n >= 16) {
            format = u16(q); channels = u16(q + 2); sr = (int)u32(q + 4); block = u16(q + 12); bits = u16(q + 14);
            if (format == 0xfffe && n >= 40) format = u16(q + 24);
        }
        if (!std::memcmp(b.data() + p, "data", 4)) { data = q; bytes = n; }
        p += 8 + n + (n & 1u);
    }
    if (!data || !channels || !sr || !block || !bits || bytes % block) throw std::runtime_error("WAV");
    const size_t frames = bytes / block;
    out.assign(frames, 0.f);
    const size_t sample_bytes = bits / 8;
    if (!sample_bytes || sample_bytes * channels > block) throw std::runtime_error("WAV");
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
            else throw std::runtime_error("WAV");
            sum += v;
        }
        out[i] = sum / channels;
    }
}
static double sinc_pi(double x) {
    if (std::fabs(x) < 1e-20) return 1.0;
    return std::sin(M_PI * x) / (M_PI * x);
}

static std::vector<double> firls_lowpass(int n_taps, double f_pass, double f_stop)
{
    if (n_taps < 3 || (n_taps % 2) == 0) throw std::runtime_error("resample firls taps");
    if (!(f_pass > 0.0) || !(f_stop > f_pass) || !(f_stop <= 1.0)) throw std::runtime_error("resample firls bands");
    const int M = (n_taps - 1) / 2;
    const int nq = M + 1;
    std::vector<double> q((size_t)n_taps, 0.0);
    const double bands[2][2] = { { 0.0, f_pass }, { f_stop, 1.0 } };
    for (int i = 0; i < n_taps; ++i) {
        const double n = (double)i;
        q[(size_t)i] =
            (bands[0][1] * sinc_pi(bands[0][1] * n) - bands[0][0] * sinc_pi(bands[0][0] * n)) +
            (bands[1][1] * sinc_pi(bands[1][1] * n) - bands[1][0] * sinc_pi(bands[1][0] * n));
    }
    std::vector<double> Q((size_t)nq * (size_t)nq, 0.0);
    auto at = [&](int i, int j) -> double & { return Q[(size_t)i * (size_t)nq + (size_t)j]; };
    for (int i = 0; i < nq; ++i) {
        for (int j = 0; j < nq; ++j) {
            at(i, j) = q[(size_t)std::abs(i - j)] + q[(size_t)(i + j)];
        }
    }
    std::vector<double> b((size_t)nq);
    for (int i = 0; i < nq; ++i) b[(size_t)i] = f_pass * sinc_pi(f_pass * (double)i);

    std::vector<double> L((size_t)nq * (size_t)nq, 0.0);
    auto Lat = [&](int i, int j) -> double & { return L[(size_t)i * (size_t)nq + (size_t)j]; };
    bool chol = true;
    for (int k = 0; k < nq && chol; ++k) {
        double acc = at(k, k);
        for (int p = 0; p < k; ++p) acc -= Lat(k, p) * Lat(k, p);
        if (!(acc > 1e-18) || !std::isfinite(acc)) { chol = false; break; }
        Lat(k, k) = std::sqrt(acc);
        for (int i = k + 1; i < nq; ++i) {
            double s = at(i, k);
            for (int p = 0; p < k; ++p) s -= Lat(i, p) * Lat(k, p);
            Lat(i, k) = s / Lat(k, k);
        }
    }
    std::vector<double> a((size_t)nq, 0.0);
    if (chol) {
        std::vector<double> y((size_t)nq, 0.0);
        for (int i = 0; i < nq; ++i) {
            double s = b[(size_t)i];
            for (int p = 0; p < i; ++p) s -= Lat(i, p) * y[(size_t)p];
            y[(size_t)i] = s / Lat(i, i);
        }
        for (int i = nq - 1; i >= 0; --i) {
            double s = y[(size_t)i];
            for (int p = i + 1; p < nq; ++p) s -= Lat(p, i) * a[(size_t)p];
            a[(size_t)i] = s / Lat(i, i);
        }
    } else {

        std::vector<int> piv((size_t)nq);
        for (int i = 0; i < nq; ++i) piv[(size_t)i] = i;
        for (int k = 0; k < nq; ++k) {
            int best = k;
            double bestv = std::fabs(at(k, k));
            for (int i = k + 1; i < nq; ++i) {
                const double v = std::fabs(at(i, k));
                if (v > bestv) { bestv = v; best = i; }
            }
            if (!(bestv > 1e-18)) throw std::runtime_error("resample firls solve");
            if (best != k) {
                for (int j = 0; j < nq; ++j) std::swap(at(k, j), at(best, j));
                std::swap(b[(size_t)k], b[(size_t)best]);
            }
            for (int i = k + 1; i < nq; ++i) {
                const double f = at(i, k) / at(k, k);
                at(i, k) = 0.0;
                for (int j = k + 1; j < nq; ++j) at(i, j) -= f * at(k, j);
                b[(size_t)i] -= f * b[(size_t)k];
            }
        }
        for (int i = nq - 1; i >= 0; --i) {
            double s = b[(size_t)i];
            for (int j = i + 1; j < nq; ++j) s -= at(i, j) * a[(size_t)j];
            a[(size_t)i] = s / at(i, i);
        }
    }
    std::vector<double> h((size_t)n_taps);
    h[(size_t)M] = 2.0 * a[0];
    for (int i = 1; i <= M; ++i) {
        h[(size_t)(M - i)] = a[(size_t)i];
        h[(size_t)(M + i)] = a[(size_t)i];
    }
    return h;
}

static size_t upfirdn_len(size_t len_h, size_t in_len, int up, int down)
{
    return (((in_len - 1) * (size_t)up + len_h) - 1) / (size_t)down + 1;
}

std::vector<float> resample_sinc(const std::vector<float> & in,
                                 int sr_in, int sr_out,
                                 int taps_half)
{
    (void)taps_half;
    if (sr_in == sr_out) return in;
    if (in.empty()) return {};
    if (sr_in <= 0 || sr_out <= 0) throw std::runtime_error("resample");
    const int g = std::gcd(sr_in, sr_out);
    const int up = sr_out / g;
    const int down = sr_in / g;
    if (up == 1 && down == 1) return in;

    const int n_taps = 641;
    const int th = (n_taps - 1) / 2;
    const double f_stop = 1.0 / (double)down;
    const double f_pass = 0.915 * f_stop;
    std::vector<double> h = firls_lowpass(n_taps, f_pass, f_stop);
    for (double & v : h) v *= (double)up;

    const int n_pre_pad = down - (th % down);
    int n_post_pad = 0;
    const size_t n_in = in.size();
    const size_t n_up = n_in * (size_t)up;
    const size_t n_out = n_up / (size_t)down + ((n_up % (size_t)down) ? 1u : 0u);
    const size_t n_pre_remove = (size_t)((th + n_pre_pad) / down);
    while (upfirdn_len(h.size() + (size_t)n_pre_pad + (size_t)n_post_pad, n_in, up, down)
           < n_out + n_pre_remove) {
        ++n_post_pad;
    }
    std::vector<double> hpad((size_t)n_pre_pad + h.size() + (size_t)n_post_pad, 0.0);
    std::copy(h.begin(), h.end(), hpad.begin() + n_pre_pad);

    std::vector<float> out(n_out, 0.0f);
    for (size_t n = 0; n < n_out; ++n) {
        const long long i = (long long)(n + n_pre_remove);
        const long long t = i * (long long)down;
        double acc = 0.0;
        for (size_t k = 0; k < hpad.size(); ++k) {
            const long long idx_up = t - (long long)k;
            if (idx_up < 0 || (idx_up % (long long)up) != 0) continue;
            const long long xi = idx_up / (long long)up;
            if (xi >= (long long)n_in) continue;
            acc += hpad[k] * (double)in[(size_t)xi];
        }
        out[n] = (float)acc;
    }
    return out;
}

std::vector<float> trim_silence(const std::vector<float> & wav, float top_db,
                                int frame_length, int hop_length)
{

    if (wav.empty()) return {};
    if (frame_length <= 0 || hop_length <= 0) throw std::runtime_error("trim");
    const int n = (int)wav.size();
    const int pad = frame_length / 2;
    const int n_padded = n + 2 * pad;
    if (n_padded < frame_length) return {};
    const int n_frames = 1 + (n_padded - frame_length) / hop_length;
    if (n_frames <= 0) return {};
    std::vector<float> padded((size_t)n_padded, 0.0f);
    std::copy(wav.begin(), wav.end(), padded.begin() + pad);
    std::vector<float> rms((size_t)n_frames);
    const float inv_len = 1.0f / (float)frame_length;
    for (int i = 0; i < n_frames; ++i) {
        const float * sl = padded.data() + (size_t)i * (size_t)hop_length;
        float acc = 0.0f;
        for (int k = 0; k < frame_length; ++k) {
            const float v = sl[k];
            acc += v * v;
        }
        rms[(size_t)i] = std::sqrt(acc * inv_len);
    }
    const float amin2 = 1e-5f * 1e-5f;
    float ref = 0.0f;
    for (float v : rms) {
        const float a = std::fabs(v);
        if (a > ref) ref = a;
    }
    const float log_ref = 10.0f * std::log10(std::max(amin2, ref * ref));
    int first = -1;
    int last = -1;
    for (int i = 0; i < n_frames; ++i) {
        const float mag = std::fabs(rms[(size_t)i]);
        const float db = 10.0f * std::log10(std::max(amin2, mag * mag)) - log_ref;
        if (db > -top_db) {
            if (first < 0) first = i;
            last = i;
        }
    }
    if (first < 0) return {};
    const int start = first * hop_length;
    const int end = std::min(n, (last + 1) * hop_length);
    if (start >= end || start < 0) return {};
    return std::vector<float>(wav.begin() + start, wav.begin() + end);
}

std::vector<float> mel_extract_24k_80(const std::vector<float> & wav_24k,
                                      const std::vector<float> & mel_filterbank,
                                      ggml_backend_t backend)
{
    return mel_extract_stft_hann_ggml(wav_24k, mel_filterbank,
        1920, 480, 1920, 80,
        0, 1.0f, 1e-5f, backend);
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
static double measure_lufs(const std::vector<float> & wav, int sr)
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
                                      const std::vector<float> & mel_filterbank,
                                      ggml_backend_t backend)
{
    return mel_extract_stft_hann_ggml(wav_16k, mel_filterbank,
        400, 160, 400, 40,
        1, 2.0f, -1.0f, backend);
}
