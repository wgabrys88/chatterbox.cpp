#include "voice_features.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"


bool wav_load(const std::string & path,
              std::vector<float> & out_samples,
              int & out_sr)
{
    drwav wav;
    if (!drwav_init_file(&wav, path.c_str(), nullptr)) {
        fprintf(stderr, "wav_load: failed to open %s\n", path.c_str());
        return false;
    }
    out_sr = (int)wav.sampleRate;

    std::vector<float> interleaved(wav.totalPCMFrameCount * wav.channels);
    drwav_uint64 frames = drwav_read_pcm_frames_f32(&wav, wav.totalPCMFrameCount, interleaved.data());
    if (frames != wav.totalPCMFrameCount) {
        fprintf(stderr, "wav_load: short read (%llu / %llu)\n",
                (unsigned long long)frames, (unsigned long long)wav.totalPCMFrameCount);
    }


    out_samples.resize(frames);
    if (wav.channels == 1) {
        std::memcpy(out_samples.data(), interleaved.data(), frames * sizeof(float));
    } else {
        const int ch = (int)wav.channels;
        for (drwav_uint64 i = 0; i < frames; ++i) {
            float acc = 0.0f;
            for (int c = 0; c < ch; ++c) acc += interleaved[i * ch + c];
            out_samples[i] = acc / (float)ch;
        }
    }

    drwav_uninit(&wav);
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


static int gcd_int(int a, int b) {
    while (b) { int t = a % b; a = b; b = t; }
    return a;
}

std::vector<float> resample_sinc(const std::vector<float> & in,
                                 int sr_in, int sr_out,
                                 int taps_half)
{
    if (sr_in == sr_out) return in;
    if (in.empty()) return {};
    (void)gcd_int;


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


void reflect_pad_1d(const std::vector<float> & in, int p_left, int p_right,
                    std::vector<float> & out)
{
    const int L = (int)in.size();
    out.resize((size_t)(L + p_left + p_right));

    for (int i = 0; i < p_left; ++i) {
        int src = p_left - i;
        out[i] = (src >= 0 && src < L) ? in[src] : 0.0f;
    }
    std::memcpy(out.data() + p_left, in.data(), L * sizeof(float));

    for (int i = 0; i < p_right; ++i) {
        int src = L - 2 - i;
        out[(size_t)(L + p_left + i)] = (src >= 0 && src < L) ? in[src] : 0.0f;
    }
}


static std::vector<float> mel_extract_generic(
    const std::vector<float> & wav,
    const std::vector<float> & mel_filterbank,
    int n_fft, int hop, int win, int n_mels,
    int center_mode,
    float power_exponent,
    float log_floor,
    bool transpose_to_T_M)
{
    const int F = n_fft / 2 + 1;
    if (mel_filterbank.size() != (size_t)(n_mels * F)) {
        fprintf(stderr,
            "mel_extract_generic: filterbank has %zu elements, expected %d (n_mels * F)\n",
            mel_filterbank.size(), n_mels * F);
        return {};
    }


    const int pad = (center_mode == 0) ? (n_fft - hop) / 2 : n_fft / 2;
    std::vector<float> padded;
    reflect_pad_1d(wav, pad, pad, padded);
    const int L = (int)padded.size();

    if (L < win) return {};
    const int T = (center_mode == 0)
        ? (L - win) / hop + 1
        : 1 + (int)wav.size() / hop;

    std::vector<float> hann(win);
    for (int n = 0; n < win; ++n)
        hann[n] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * (float)n / (float)win));

    std::vector<float> cos_tbl((size_t)F * n_fft);
    std::vector<float> sin_tbl((size_t)F * n_fft);
    for (int k = 0; k < F; ++k) {
        for (int n = 0; n < n_fft; ++n) {
            double th = 2.0 * M_PI * (double)k * (double)n / (double)n_fft;
            cos_tbl[(size_t)k * n_fft + n] = (float)std::cos(th);
            sin_tbl[(size_t)k * n_fft + n] = (float)std::sin(th);
        }
    }

    std::vector<float> spec((size_t)F * T);
    std::vector<float> frame(win);
    for (int t = 0; t < T; ++t) {
        const float * x = padded.data() + t * hop;
        for (int n = 0; n < win; ++n) frame[n] = x[n] * hann[n];
        for (int k = 0; k < F; ++k) {
            const float * cs = cos_tbl.data() + (size_t)k * n_fft;
            const float * sn = sin_tbl.data() + (size_t)k * n_fft;
            float re = 0.0f, im = 0.0f;
            for (int n = 0; n < win; ++n) {
                re += frame[n] * cs[n];
                im -= frame[n] * sn[n];
            }
            float mag = std::sqrt(re * re + im * im + 1e-9f);
            if (power_exponent == 2.0f) mag = mag * mag;
            else if (power_exponent != 1.0f) mag = std::pow(mag, power_exponent);
            spec[(size_t)k * T + t] = mag;
        }
    }

    std::vector<float> mel((size_t)n_mels * T);
    for (int m = 0; m < n_mels; ++m) {
        const float * fb_row = mel_filterbank.data() + (size_t)m * F;
        for (int t = 0; t < T; ++t) {
            float acc = 0.0f;
            for (int k = 0; k < F; ++k) acc += fb_row[k] * spec[(size_t)k * T + t];
            mel[(size_t)m * T + t] = acc;
        }
    }

    if (log_floor > 0.0f)
        for (float & v : mel) v = std::log(std::max(v, log_floor));

    if (!transpose_to_T_M) return mel;

    std::vector<float> out((size_t)T * n_mels);
    for (int m = 0; m < n_mels; ++m)
        for (int t = 0; t < T; ++t)
            out[(size_t)t * n_mels + m] = mel[(size_t)m * T + t];
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


#if 0
static std::vector<float> fbank_kaldi_80_scalar(const std::vector<float> & wav_16k,
                                                const std::vector<float> & mel_filterbank)
{
    const int n_fft      = 512;
    const int frame_len  = 400;
    const int hop        = 160;
    const int n_mels     = 80;
    const int F          = n_fft / 2 + 1;
    const float preemph  = 0.97f;


    if (mel_filterbank.size() != (size_t)(n_mels * F)) {
        fprintf(stderr,
            "fbank_kaldi_80: filterbank has %zu elements, expected %d (n_mels * F)\n",
            mel_filterbank.size(), n_mels * F);
        return {};
    }

    const int L = (int)wav_16k.size();
    if (L < frame_len) return {};
    const int T = (L - frame_len) / hop + 1;


    std::vector<float> povey(frame_len);
    for (int n = 0; n < frame_len; ++n) {
        double a = 0.5 - 0.5 * std::cos(2.0 * M_PI * (double)n / (double)(frame_len - 1));
        povey[n] = (float)std::pow(a, 0.85);
    }


    std::vector<float> cos_tbl((size_t)F * n_fft);
    std::vector<float> sin_tbl((size_t)F * n_fft);
    for (int k = 0; k < F; ++k) {
        for (int n = 0; n < n_fft; ++n) {
            double th = 2.0 * M_PI * (double)k * (double)n / (double)n_fft;
            cos_tbl[(size_t)k * n_fft + n] = (float)std::cos(th);
            sin_tbl[(size_t)k * n_fft + n] = (float)std::sin(th);
        }
    }


    std::vector<float> out((size_t)T * n_mels);

    std::vector<float> frame(n_fft, 0.0f);
    std::vector<float> power(F);
    for (int t = 0; t < T; ++t) {


        const float * src = wav_16k.data() + t * hop;
        for (int n = 0; n < frame_len; ++n) frame[n] = src[n];
        for (int n = frame_len; n < n_fft; ++n) frame[n] = 0.0f;


        double acc = 0.0;
        for (int n = 0; n < frame_len; ++n) acc += frame[n];
        float dc = (float)(acc / frame_len);
        for (int n = 0; n < frame_len; ++n) frame[n] -= dc;


        for (int n = frame_len - 1; n >= 1; --n) {
            frame[n] = frame[n] - preemph * frame[n - 1];
        }
        frame[0] = frame[0] * (1.0f - preemph);


        for (int n = 0; n < frame_len; ++n) frame[n] *= povey[n];


        for (int k = 0; k < F; ++k) {
            const float * cs = cos_tbl.data() + (size_t)k * n_fft;
            const float * sn = sin_tbl.data() + (size_t)k * n_fft;
            float re = 0.0f, im = 0.0f;
            for (int n = 0; n < n_fft; ++n) {
                re += frame[n] * cs[n];
                im -= frame[n] * sn[n];
            }
            power[k] = re * re + im * im;
        }


        for (int m = 0; m < n_mels; ++m) {
            const float * fb_row = mel_filterbank.data() + (size_t)m * F;
            float mel_e = 0.0f;
            for (int k = 0; k < F; ++k) mel_e += fb_row[k] * power[k];

            if (mel_e < std::numeric_limits<float>::epsilon())
                mel_e = std::numeric_limits<float>::epsilon();
            out[(size_t)t * n_mels + m] = std::log(mel_e);
        }
    }

    return out;
}
#endif
