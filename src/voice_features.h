#pragma once


#include <cstdint>
#include <string>
#include <vector>


bool wav_load(const std::string & path,
              std::vector<float> & out_samples,
              int & out_sr);


std::vector<float> resample_sinc(const std::vector<float> & in,
                                 int sr_in, int sr_out,
                                 int taps_half = 16);


double measure_lufs(const std::vector<float> & wav, int sr);


void normalise_lufs(std::vector<float> & wav, int sr, double target_lufs = -27.0);


std::vector<float> mel_extract_24k_80(const std::vector<float> & wav_24k,
                                      const std::vector<float> & mel_filterbank);


std::vector<float> mel_extract_16k_40(const std::vector<float> & wav_16k,
                                      const std::vector<float> & mel_filterbank);


std::vector<float> fbank_kaldi_80(const std::vector<float> & wav_16k,
                                  const std::vector<float> & mel_filterbank);
