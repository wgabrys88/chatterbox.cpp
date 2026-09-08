#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "chatterbox_t3_internal.h"
#include "voice_features.h"
#include "campplus.h"
#include "s3tokenizer.h"
using namespace tts_cpp::chatterbox::detail;
namespace tts_cpp::chatterbox::detail {
bool validate_reference_audio(const std::string & path) {
    std::vector<float> wav;
    int sr = 0;
    if (!wav_load(path, wav, sr)) {
        fprintf(stderr, "error: failed to load --reference-audio: %s\n", path.c_str());
        return false;
    }
    const double secs = (double)wav.size() / (double)sr;
    if (secs <= 5.0) {
        fprintf(stderr,
            "error: --reference-audio is only %.2f s; Chatterbox requires strictly more "
            "than 5 s of clean mono speech.  Shorter references produce undersized "
            "conditioning tensors and the model falls back on the built-in voice.\n"
            "  Recommended length: 10–15 s.\n", secs);
        return false;
    }
    if (secs < 10.0) {
        fprintf(stderr,
            "warning: --reference-audio is %.2f s; 10–15 s is recommended for best "
            "voice similarity.\n", secs);
    }
    return true;
}
bool compute_prompt_feat_native(const std::string & wav_path,
                                       const std::string & s3gen_gguf_path,
                                       std::vector<float> & out_feat,
                                       int & out_rows,
                                       bool verbose)
{
    if (verbose) fprintf(stderr, "voice: loading %s\n", wav_path.c_str());
    std::vector<float> wav;
    int sr = 0;
    if (!wav_load(wav_path, wav, sr)) return false;
    if (verbose) fprintf(stderr, "voice:   sr=%d samples=%zu (%.2f s)\n", sr, wav.size(), (double)wav.size() / sr);
    if (sr != 24000) {
        if (verbose) fprintf(stderr, "voice: resampling %d -> 24000\n", sr);
        wav = resample_sinc(wav, sr, 24000);
    }
    double pre  = measure_lufs(wav, 24000);
    normalise_lufs(wav, 24000, -27.0);
    if (verbose) fprintf(stderr, "voice:   loudness %.2f LUFS → -27 LUFS (+%.2f dB)\n", pre, -27.0 - pre);
    const int dec_cond_samples = 10 * 24000;
    if ((int)wav.size() > dec_cond_samples) wav.resize(dec_cond_samples);
    ggml_context * tmp_ctx = nullptr;
    gguf_init_params gp = {  false,  &tmp_ctx };
    gguf_context * g = gguf_init_from_file(s3gen_gguf_path.c_str(), gp);
    if (!g) {
        fprintf(stderr, "voice: failed to open %s\n", s3gen_gguf_path.c_str());
        return false;
    }
    ggml_tensor * fb = ggml_get_tensor(tmp_ctx, "s3gen/mel_fb/24k_80");
    if (!fb) {
        fprintf(stderr, "voice: s3gen/mel_fb/24k_80 missing from GGUF; re-run convert-s3gen-to-gguf.py\n");
        gguf_free(g); if (tmp_ctx) ggml_free(tmp_ctx);
        return false;
    }
    std::vector<float> mel_fb(ggml_nelements(fb));
    std::memcpy(mel_fb.data(), ggml_get_data(fb), ggml_nbytes(fb));
    gguf_free(g);
    if (tmp_ctx) ggml_free(tmp_ctx);
    out_feat = mel_extract_24k_80(wav, mel_fb);
    if (out_feat.empty()) return false;
    out_rows = (int)(out_feat.size() / 80);
    if (verbose) fprintf(stderr, "voice: prompt_feat shape=(%d, 80)\n", out_rows);
    return true;
}
bool compute_embedding_native(const std::string & wav_path,
                                     const std::string & s3gen_gguf_path,
                                     std::vector<float> & out_emb,
                                     bool verbose)
{
    campplus_weights w;
    if (!campplus_load(s3gen_gguf_path, w)) {
        fprintf(stderr, "voice: s3gen GGUF has no CAMPPlus weights; cannot synthesise "
                        "embedding natively (re-run convert-s3gen-to-gguf.py)\n");
        return false;
    }
    ggml_context * tmp_ctx = nullptr;
    gguf_init_params gp = {  false,  &tmp_ctx };
    gguf_context * g = gguf_init_from_file(s3gen_gguf_path.c_str(), gp);
    if (!g) return false;
    ggml_tensor * fb_t = ggml_get_tensor(tmp_ctx, "campplus/mel_fb_kaldi_80");
    if (!fb_t) {
        fprintf(stderr, "voice: campplus/mel_fb_kaldi_80 missing; rerun converter\n");
        gguf_free(g); if (tmp_ctx) ggml_free(tmp_ctx);
        return false;
    }
    std::vector<float> mel_fb(ggml_nelements(fb_t));
    std::memcpy(mel_fb.data(), ggml_get_data(fb_t), ggml_nbytes(fb_t));
    gguf_free(g); if (tmp_ctx) ggml_free(tmp_ctx);
    std::vector<float> wav;
    int sr = 0;
    if (!wav_load(wav_path, wav, sr)) return false;
    normalise_lufs(wav, sr, -27.0);
    if (sr != 16000) wav = resample_sinc(wav, sr, 16000);
    const int dec_cond_samples_16k = 10 * 16000;
    if ((int)wav.size() > dec_cond_samples_16k) wav.resize(dec_cond_samples_16k);
    std::vector<float> fbank = fbank_kaldi_80(wav, mel_fb);
    if (fbank.empty()) return false;
    const int T = (int)(fbank.size() / 80);
    std::vector<float> col_mean(80, 0.0f);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < 80; ++c) col_mean[c] += fbank[(size_t)t * 80 + c];
    for (int c = 0; c < 80; ++c) col_mean[c] /= (float)T;
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < 80; ++c) fbank[(size_t)t * 80 + c] -= col_mean[c];
    if (!campplus_embed(fbank, T, w, out_emb)) return false;
    if (verbose) fprintf(stderr, "voice: embedding shape=(%zu,) via CAMPPlus (%d fbank frames)\n",
            out_emb.size(), T);
    return true;
}
bool compute_speech_tokens_native(const std::string & wav_path,
                                         const std::string & s3gen_gguf_path,
                                         int max_cond_tokens,
                                         std::vector<int32_t> & out_prompt_tokens,
                                         std::vector<int32_t> & out_cond_tokens,
                                         int n_threads,
                                         ggml_backend_t backend,
                                         bool verbose = false)
{
    s3tokv2_weights w;
    if (!s3tokv2_load(s3gen_gguf_path, w)) {
        fprintf(stderr, "voice: s3gen GGUF has no S3TokenizerV2 weights; cannot "
                        "synthesise speech tokens natively (re-run converter)\n");
        return false;
    }
    std::vector<float> wav;
    int sr = 0;
    if (!wav_load(wav_path, wav, sr)) return false;
    normalise_lufs(wav, sr, -27.0);
    if (sr != 16000) wav = resample_sinc(wav, sr, 16000);
    const int dec_cond_samples = 10 * 16000;
    std::vector<float> prompt_wav(wav.begin(), wav.begin() + std::min((int)wav.size(), dec_cond_samples));
    if (!s3tokv2_tokenize(prompt_wav, w, -1, out_prompt_tokens, n_threads, backend)) return false;
    const int enc_cond_samples = 15 * 16000;
    std::vector<float> cond_wav(wav.begin(), wav.begin() + std::min((int)wav.size(), enc_cond_samples));
    if (!s3tokv2_tokenize(cond_wav, w, max_cond_tokens, out_cond_tokens, n_threads, backend)) return false;
    if (verbose) fprintf(stderr, "voice: prompt_token=(%zu,) cond_prompt_speech_tokens=(%zu,) via S3TokenizerV2\n",
            out_prompt_tokens.size(), out_cond_tokens.size());
    return true;
}
int g_log_verbose = 0;
void chatterbox_log_cb(ggml_log_level level, const char * text, void * ) {
    if (level >= GGML_LOG_LEVEL_ERROR && text) fputs(text, stderr);
}
}
