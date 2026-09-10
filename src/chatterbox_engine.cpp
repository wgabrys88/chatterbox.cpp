#include "tts-cpp/chatterbox/engine.h"
#include "tts-cpp/chatterbox/nano.h"
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "chatterbox_t3_internal.h"
#include "gpt2_bpe.h"
#include "s3gen_pipeline.h"
#include "voice_encoder.h"
#include "voice_features.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
namespace tts_cpp::chatterbox {
using namespace detail;
struct Engine::Impl {
    EngineOptions opts;
    chatterbox_model model{};
    ggml_gallocr_t allocr = nullptr;
    std::vector<float> prompt_feat, embedding;
    int prompt_rows = 0;
    std::vector<int32_t> prompt_token;
    explicit Impl(const EngineOptions& o) : opts(o) {}
    void init() {
        ggml_time_init();
        ggml_log_set([](ggml_log_level, const char*, void*) {}, nullptr);
        model.backend = init_backend();
        load_model_gguf(opts.t3_gguf_path, model);
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        bake_voice();
        s3gen_preload(opts.s3gen_gguf_path, model.backend);
    }
    ~Impl() {
        s3gen_unload();
        if (allocr) ggml_gallocr_free(allocr);
        if (model.buffer_w) ggml_backend_buffer_free(model.buffer_w);
        if (model.buffer_kv) ggml_backend_buffer_free(model.buffer_kv);
        if (model.buffer_override) ggml_backend_buffer_free(model.buffer_override);
        if (model.backend) ggml_backend_free(model.backend);
        if (model.ctx_w) ggml_free(model.ctx_w);
        if (model.ctx_kv) ggml_free(model.ctx_kv);
        if (model.ctx_override) ggml_free(model.ctx_override);
    }
    void bake_voice() {
        voice_encoder_weights ve;
        voice_encoder_load(opts.t3_gguf_path, ve);
        std::vector<float> wav, speaker;
        int sr = 0;
        wav_load(opts.reference_audio, wav, sr);
        normalise_lufs(wav, sr, -27.0);
        if (sr != 16000) wav = resample_sinc(wav, sr, 16000);
        if (wav.size() > 30u * 16000u) wav.resize(30u * 16000u);
        voice_encoder_embed(wav, ve, model.backend, speaker);
        ggml_backend_tensor_set(model.builtin_speaker_emb, speaker.data(), 0, ggml_nbytes(model.builtin_speaker_emb));
        std::vector<int32_t> cond;
        compute_speech_tokens_native(opts.reference_audio, opts.s3gen_gguf_path, model.hparams.cond_prompt_len,
            prompt_token, cond, model.backend);
        ggml_init_params p = {ggml_tensor_overhead() * 2, nullptr, true};
        model.ctx_override = ggml_init(p);
        auto* t = ggml_new_tensor_1d(model.ctx_override, GGML_TYPE_I32, (int64_t)cond.size());
        model.buffer_override = ggml_backend_alloc_ctx_tensors(model.ctx_override, model.backend);
        ggml_backend_tensor_set(t, cond.data(), 0, cond.size() * sizeof(int32_t));
        model.builtin_cond_prompt_tokens = t;
        model.hparams.cond_prompt_len = (int32_t)cond.size();
        compute_prompt_feat_native(opts.reference_audio, opts.s3gen_gguf_path, prompt_feat, prompt_rows, model.backend);
        compute_embedding_native(opts.reference_audio, opts.s3gen_gguf_path, embedding, model.backend);
    }
    std::vector<int32_t> generate_t3(const std::string& text) {
        std::mt19937 rng(SEED);
        gpt2_bpe bpe;
        bpe.load_from_arrays(model.tok_tokens, model.tok_merges);
        auto text_tokens = bpe.tokenize(gpt2_bpe::punc_norm(text));
        int n_past = 0;
        int32_t token = 0;
        std::vector<int32_t> out, tokens;
        out.reserve((size_t)N_PREDICT + 1);
        tokens.reserve((size_t)N_PREDICT + (size_t)SILENCE_COUNT);
        const int32_t stop = model.hparams.stop_speech_token;
        std::vector<float> logits;
        eval_prompt(model, allocr, text_tokens, logits, n_past);
        token = sample_next_token_ex(logits, out, rng);
        out.push_back(token);
        if (token >= 0 && token < model.hparams.start_speech_token) tokens.push_back(token);
        for (int step = 1; step < N_PREDICT && token != stop && n_past + 1 <= model.hparams.n_ctx; ++step) {
            eval_step(model, allocr, n_past++, token, logits);
            token = sample_next_token_ex(logits, out, rng);
            out.push_back(token);
            if (token >= 0 && token < model.hparams.start_speech_token) tokens.push_back(token);
        }
        if (token != stop) throw std::runtime_error("T3 stopped without EOS");
        tokens.insert(tokens.end(), (size_t)SILENCE_COUNT, SILENCE_TOKEN);
        return tokens;
    }
};
Engine::Engine(const EngineOptions& o) : pimpl_(std::make_unique<Impl>(o)) { pimpl_->init(); }
Engine::~Engine() = default;
std::vector<float> Engine::synthesize(const std::string& text) {
    s3gen_synthesize_opts s;
    s.s3gen_gguf_path = pimpl_->opts.s3gen_gguf_path;
    s.prompt_feat = pimpl_->prompt_feat;
    s.prompt_rows = pimpl_->prompt_rows;
    s.embedding = pimpl_->embedding;
    s.prompt_token = pimpl_->prompt_token;
    return s3gen_synthesize(pimpl_->generate_t3(text), s);
}
}
