#include "tts-cpp/chatterbox/engine.h"
#include "tts-cpp/chatterbox/gpt2.h"
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "chatterbox_t3_internal.h"
#include "gpt2_bpe.h"
#include "s3gen_pipeline.h"
#include "utterance_split.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
namespace tts_cpp::chatterbox {
using namespace detail;
struct Engine::Impl {
    EngineOptions opts;
    chatterbox_model model{};
    ggml_gallocr_t allocr = nullptr;
    explicit Impl(const EngineOptions& o) : opts(o) {}
    void init() {
        ggml_time_init();
        ggml_log_set([](ggml_log_level level, const char * text, void *) {
            if (level >= GGML_LOG_LEVEL_WARN && text) {
                fputs(text, stderr);
                fflush(stderr);
            }
        }, nullptr);
        model.backend = init_backend();
        load_model_gguf(opts.t3_gguf_path, model);
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        s3gen_preload(opts.s3gen_gguf_path, model.backend);
    }
    ~Impl() {
        s3gen_unload();
        if (allocr) ggml_gallocr_free(allocr);
        if (model.buffer_w) ggml_backend_buffer_free(model.buffer_w);
        if (model.buffer_kv) ggml_backend_buffer_free(model.buffer_kv);
        if (model.backend) ggml_backend_free(model.backend);
        if (model.ctx_w) ggml_free(model.ctx_w);
        if (model.ctx_kv) ggml_free(model.ctx_kv);
    }
    void synthesize(const std::string& text, std::vector<float>& pcm, SynthesizeStats * stats) {
        pcm.clear();
        gpt2_bpe bpe;
        bpe.load_from_arrays(model.tok_tokens, model.tok_merges);
        const TokenCount count = [&](const std::string& s) { return (int)bpe.tokenize(gpt2_bpe::punc_norm(s)).size(); };
        const auto units = split_utterances(text, effective_split_tokens(), count);
        std::mt19937 rng(effective_seed());
        if (stats) *stats = SynthesizeStats{};
        for (size_t i = 0; i < units.size(); ++i) {
            SynthesizeStats unit;
            auto tokens = generate_t3(units[i], bpe, rng, &unit);
            auto wav = s3gen_synthesize(tokens);
            std::fill_n(wav.begin(), std::min(wav.size(), (size_t)TRIM_FADE), 0.0f);
            for (size_t j = TRIM_FADE; j < std::min(wav.size(), (size_t)SAMPLES_PER_TOKEN); ++j)
                wav[j] *= 0.5f * (1.0f - std::cos((float)M_PI * (float)(j - TRIM_FADE) / (float)(TRIM_FADE - 1)));
            pcm.insert(pcm.end(), wav.begin(), wav.end());
            accumulate_unit(stats, unit, (int)i, (int)units.size(), units[i]);
        }
    }
    std::vector<int32_t> generate_t3(const std::string& text, const gpt2_bpe& bpe, std::mt19937& rng, SynthesizeStats * stats) {
        const int n_predict = effective_n_predict();
        auto text_tokens = bpe.tokenize(gpt2_bpe::punc_norm(text));
        if (model.buffer_kv) ggml_backend_buffer_clear(model.buffer_kv, 0);
        int n_past = 0;
        int32_t token = 0;
        std::vector<int32_t> predicted, speech;
        predicted.reserve((size_t)n_predict + 1);
        speech.reserve((size_t)n_predict + (size_t)SIL_COUNT);
        const int32_t stop = model.hparams.stop_speech_token;
        std::vector<float> logits;
        eval_prompt(model, allocr, text_tokens, logits, n_past);
        const std::vector<int32_t> first_pen = { model.hparams.start_speech_token };
        token = sample_next_token_ex(logits, first_pen, rng);
        predicted.push_back(token);
        for (int step = 1; step < n_predict && token != stop && n_past + 1 <= model.hparams.n_ctx; ++step) {
            eval_step(model, allocr, n_past++, token, logits);
            token = sample_next_token_ex(logits, predicted, rng);
            predicted.push_back(token);
        }
        if (token != stop) {
            char msg[256];
            std::snprintf(msg, sizeof(msg), "T3 no EOS: predicted=%d n_past=%d n_predict=%d n_ctx=%d text_tokens=%d",
                (int)predicted.size(), n_past, n_predict, model.hparams.n_ctx, (int)text_tokens.size());
            throw std::runtime_error(msg);
        }
        for (int32_t next : predicted) if (next >= 0 && next < 6561) speech.push_back(next);
        speech.insert(speech.end(), (size_t)SIL_COUNT, S3GEN_SIL);
        if (stats) {
            stats->predicted_count = (int)predicted.size();
            stats->dropped_count = (int)speech.size();
            stats->eos = token == stop ? 1 : 0;
            stats->n_past = n_past;
            stats->text_tokens = (int)text_tokens.size();
        }
        return speech;
    }
};
Engine::Engine(const EngineOptions& o) : pimpl_(std::make_unique<Impl>(o)) { pimpl_->init(); }
Engine::~Engine() = default;
void Engine::synthesize(const std::string& text, std::vector<float>& pcm, SynthesizeStats * stats) {
    pimpl_->synthesize(text, pcm, stats);
}
}
