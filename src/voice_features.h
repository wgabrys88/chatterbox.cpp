#pragma once
#include <cstdint>
#include <string>
#include <vector>
struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;
bool wav_load(const std::string & path,
              std::vector<float> & out_samples,
              int & out_sr);
std::vector<float> resample_sinc(const std::vector<float> & in,
                                 int sr_in, int sr_out,
                                 int taps_half = 16);
double measure_lufs(const std::vector<float> & wav, int sr);
void normalise_lufs(std::vector<float> & wav, int sr, double target_lufs = -27.0);
std::vector<float> mel_graph_run(const std::vector<float> & frames_TC,
                                 const std::vector<float> & mel_fb,
                                 int T, int n_fft, int F, int n_mels,
                                 float power_exp, float log_floor,
                                 ggml_backend_t backend);
std::vector<float> mel_extract_24k_80(const std::vector<float> & wav_24k,
                                      const std::vector<float> & mel_filterbank,
                                      ggml_backend_t backend);
std::vector<float> mel_extract_16k_40(const std::vector<float> & wav_16k,
                                      const std::vector<float> & mel_filterbank,
                                      ggml_backend_t backend);
std::vector<float> fbank_kaldi_80(const std::vector<float> & wav_16k,
                                  const std::vector<float> & mel_filterbank,
                                  ggml_backend_t backend);
std::vector<float> fbank_kaldi_80_ggml(const std::vector<float> & wav_16k,
                                       const std::vector<float> & mel_fb,
                                       ggml_backend_t backend);
std::vector<float> mel_extract_stft_hann_ggml(
    const std::vector<float> & wav,
    const std::vector<float> & mel_fb,
    int n_fft, int hop, int win, int n_mels,
    int center_mode, float power_exp, float log_floor,
    ggml_backend_t backend);
