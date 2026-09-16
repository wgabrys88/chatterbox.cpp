#include "tts-cpp/chatterbox/engine.h"
#include "tts-cpp/chatterbox/v3.h"
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "chatterbox_t3_internal.h"
#include "mtl_bpe.h"
#include "s3gen_pipeline.h"
#include "utterance_split.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
namespace tts_cpp::chatterbox {
using namespace detail;
static std::vector<int32_t> drop_invalid_tokens(const std::vector<int32_t> & x, int32_t sos, int32_t eos) {
    size_t s = 0, e = x.size();
    for (size_t i = 0; i < x.size(); ++i) if (x[i] == sos) { s = i + 1; break; }
    for (size_t i = 0; i < x.size(); ++i) if (x[i] == eos) { e = i; break; }
    if (s > e) throw std::runtime_error("drop_invalid");
    return std::vector<int32_t>(x.begin() + (std::ptrdiff_t)s, x.begin() + (std::ptrdiff_t)e);
}
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
        if (opts.language_id.empty()) throw std::runtime_error("language");
        mtl_bpe bpe;
        if (!bpe.load_from_arrays(model.tok_tokens, model.tok_types, model.tok_merges))
            throw std::runtime_error("tokenizer");
        const TokenCount count = [&](const std::string& s) { return (int)bpe.encode(s, opts.language_id).size(); };
        const auto units = split_utterances(text, effective_split_tokens(), count);
        std::mt19937 rng(effective_seed());
        if (stats) *stats = SynthesizeStats{};
        for (size_t i = 0; i < units.size(); ++i) {
            SynthesizeStats unit;
            auto tokens = generate_t3(units[i], bpe, rng, &unit);
            auto wav = s3gen_synthesize(tokens);
            const int n_tokens = (int)tokens.size();
            if (n_tokens < 1) throw std::runtime_error("S3Gen empty tokens");
            const int st_len = n_tokens - 1;
            wav.resize((size_t)st_len * (size_t)kSamplesPerToken);
            std::fill_n(wav.begin(), std::min(wav.size(), (size_t)TRIM_FADE), 0.0f);
            for (size_t j = TRIM_FADE; j < std::min(wav.size(), (size_t)(2 * TRIM_FADE)); ++j)
                wav[j] *= 0.5f * (1.0f - std::cos((float)M_PI * (float)(j - TRIM_FADE) / (float)(TRIM_FADE - 1)));
            pcm.insert(pcm.end(), wav.begin(), wav.end());
            accumulate_unit(stats, unit, (int)i, (int)units.size(), units[i]);
        }
    }
    std::vector<int32_t> generate_t3(const std::string& text, const mtl_bpe& bpe, std::mt19937& rng, SynthesizeStats * stats) {
        auto ids = bpe.encode(text, opts.language_id);
        std::vector<int32_t> text_tokens;
        text_tokens.reserve(ids.size() + 2);
        text_tokens.push_back(model.hparams.start_text_token);
        text_tokens.insert(text_tokens.end(), ids.begin(), ids.end());
        text_tokens.push_back(model.hparams.stop_text_token);
        if (model.buffer_kv) ggml_backend_buffer_clear(model.buffer_kv, 0);
        int n_past = 0;
        const int32_t stop = model.hparams.stop_speech_token;
        const int32_t sos = model.hparams.start_speech_token;
        std::vector<float> logits;
        eval_prompt(model, allocr, text_tokens, logits, n_past);
        std::vector<int32_t> generated;
        generated.push_back(sos);
        std::vector<int32_t> predicted;
        const int n_predict = effective_n_predict();
        predicted.reserve((size_t)n_predict);
        for (int i = 0; i < n_predict && n_past + 1 <= model.hparams.n_ctx; ++i) {
            int32_t token = sample_next_token_ex(logits, generated, rng);
            predicted.push_back(token);
            generated.push_back(token);
            if (token == stop) break;
            eval_step(model, allocr, n_past++, token, i + 1, logits);
        }
        if (predicted.empty()) throw std::runtime_error("T3 produced no tokens");
        if (predicted.back() != stop) {
            char msg[256];
            std::snprintf(msg, sizeof(msg), "T3 no EOS: predicted=%d n_past=%d n_predict=%d n_ctx=%d text_tokens=%d",
                (int)predicted.size(), n_past, n_predict, model.hparams.n_ctx, (int)text_tokens.size());
            throw std::runtime_error(msg);
        }
        auto dropped = drop_invalid_tokens(predicted, sos, stop);
        if (stats) {
            stats->predicted_count = (int)predicted.size();
            stats->dropped_count = (int)dropped.size();
            stats->eos = predicted.back() == stop ? 1 : 0;
            stats->n_past = n_past;
            stats->text_tokens = (int)text_tokens.size();
        }
        return dropped;
    }
};
Engine::Engine(const EngineOptions& o) : pimpl_(std::make_unique<Impl>(o)) { pimpl_->init(); }
Engine::~Engine() = default;
void Engine::synthesize(const std::string& text, std::vector<float>& pcm, SynthesizeStats * stats) {
    pimpl_->synthesize(text, pcm, stats);
}
}
