#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "bake_native.h"
#include "voice_features.h"
#include "campplus.h"
#include "s3tokenizer.h"
using namespace tts_cpp::chatterbox::detail;
namespace tts_cpp::chatterbox::detail {
void compute_prompt_feat_native(const std::string & wav_path, const std::string & s3gen_gguf_path,
                                std::vector<float> & out_feat, int & out_rows, ggml_backend_t backend) {
    std::vector<float> wav;
    int sr = 0;
    wav_load(wav_path, wav, sr);
    if (sr != 24000) wav = resample_sinc(wav, sr, 24000);
    normalise_lufs(wav, 24000, -27.0);
    if ((int)wav.size() > 10 * 24000) wav.resize(10 * 24000);
    ggml_context * tmp_ctx = nullptr;
    gguf_init_params gp = { false, &tmp_ctx };
    gguf_context * g = gguf_init_from_file(s3gen_gguf_path.c_str(), gp);
    ggml_tensor * fb = ggml_get_tensor(tmp_ctx, "s3gen/mel_fb/24k_80");
    std::vector<float> mel_fb(ggml_nelements(fb));
    std::memcpy(mel_fb.data(), ggml_get_data(fb), ggml_nbytes(fb));
    gguf_free(g);
    ggml_free(tmp_ctx);
    out_feat = mel_extract_24k_80(wav, mel_fb, backend);
    out_rows = (int)(out_feat.size() / 80);
}
void compute_embedding_native(const std::string & wav_path, const std::string & s3gen_gguf_path,
                              std::vector<float> & out_emb, ggml_backend_t backend) {
    campplus_weights w;
    campplus_load(s3gen_gguf_path, w);
    ggml_context * tmp_ctx = nullptr;
    gguf_init_params gp = { false, &tmp_ctx };
    gguf_context * g = gguf_init_from_file(s3gen_gguf_path.c_str(), gp);
    ggml_tensor * fb_t = ggml_get_tensor(tmp_ctx, "campplus/mel_fb_kaldi_80");
    std::vector<float> mel_fb(ggml_nelements(fb_t));
    std::memcpy(mel_fb.data(), ggml_get_data(fb_t), ggml_nbytes(fb_t));
    gguf_free(g);
    ggml_free(tmp_ctx);
    std::vector<float> wav;
    int sr = 0;
    wav_load(wav_path, wav, sr);
    normalise_lufs(wav, sr, -27.0);
    if (sr != 16000) wav = resample_sinc(wav, sr, 16000);
    if ((int)wav.size() > 10 * 16000) wav.resize(10 * 16000);
    std::vector<float> fbank = fbank_kaldi_80(wav, mel_fb, backend);
    const int T = (int)(fbank.size() / 80);
    std::vector<float> col_mean(80, 0.0f);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < 80; ++c) col_mean[c] += fbank[(size_t)t * 80 + c];
    for (int c = 0; c < 80; ++c) col_mean[c] /= (float)T;
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < 80; ++c) fbank[(size_t)t * 80 + c] -= col_mean[c];
    campplus_embed(fbank, T, w, backend, out_emb);
}
void compute_speech_tokens_native(const std::string & wav_path, const std::string & s3gen_gguf_path,
                                  int max_cond_tokens, std::vector<int32_t> & out_prompt_tokens,
                                  std::vector<int32_t> & out_cond_tokens, ggml_backend_t backend) {
    s3tokv2_weights w;
    s3tokv2_load(s3gen_gguf_path, w);
    std::vector<float> wav;
    int sr = 0;
    wav_load(wav_path, wav, sr);
    normalise_lufs(wav, sr, -27.0);
    if (sr != 16000) wav = resample_sinc(wav, sr, 16000);
    std::vector<float> prompt_wav(wav.begin(), wav.begin() + std::min((int)wav.size(), 10 * 16000));
    s3tokv2_tokenize(prompt_wav, w, -1, out_prompt_tokens, backend);
    std::vector<float> cond_wav(wav.begin(), wav.begin() + std::min((int)wav.size(), 15 * 16000));
    s3tokv2_tokenize(cond_wav, w, max_cond_tokens, out_cond_tokens, backend);
}
}
