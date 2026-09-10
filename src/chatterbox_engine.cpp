#include "tts-cpp/chatterbox/engine.h"
#include "tts-cpp/chatterbox/turbo.h"
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "chatterbox_t3_internal.h"
#include "gpt2_bpe.h"
#include "s3gen_pipeline.h"
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
        ggml_log_set([](ggml_log_level, const char*, void*) {}, nullptr);
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
    return s3gen_synthesize(pimpl_->generate_t3(text));
}
}
