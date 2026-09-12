#include "tts-cpp/chatterbox/engine.h"
#include "tts-cpp/chatterbox/nano.h"
#include <fstream>
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
        const int n_predict = effective_n_predict();
        const int sil_n = effective_silence_count();
        const int sil = effective_silence_token();
        std::mt19937 rng(effective_seed());
        gpt2_bpe bpe;
        bpe.load_from_arrays(model.tok_tokens, model.tok_merges);
        auto text_tokens = bpe.tokenize(gpt2_bpe::punc_norm(text));
        int n_past = 0;
        int32_t token = 0;
        std::vector<int32_t> out, tokens;
        out.reserve((size_t)n_predict + 1);
        tokens.reserve((size_t)n_predict + (size_t)sil_n);
        const int32_t stop = model.hparams.stop_speech_token;
        std::vector<float> logits;
        std::ofstream slog;
        if (sampler_log_enabled()) {
            std::string p = opts.t3_gguf_path;
            auto slash = p.find_last_of("\\/");
            if (slash != std::string::npos)
                slog.open(p.substr(0, slash) + "/nano_sample_dump.csv");
        }
        g_sampler_log = slog.is_open() ? &slog : nullptr;
        g_sampler_step = 0;
        if (g_sampler_log)
            *g_sampler_log << "step,chosen,chosen_prob,sil4299_prob,sil4299_rank,sil4299_seen,gen_len,top0_id,top0_prob,top1_id,top1_prob,top2_id,top2_prob,top3_id,top3_prob,top4_id,top4_prob,top5_id,top5_prob,top6_id,top6_prob,top7_id,top7_prob,top8_id,top8_prob,top9_id,top9_prob\n";
        eval_prompt(model, allocr, text_tokens, logits, n_past);
        token = sample_next_token_ex(logits, out, rng);
        out.push_back(token);
        if (token >= 0 && token < model.hparams.start_speech_token) tokens.push_back(token);
        for (int step = 1; step < n_predict && token != stop && n_past + 1 <= model.hparams.n_ctx; ++step) {
            eval_step(model, allocr, n_past++, token, logits);
            token = sample_next_token_ex(logits, out, rng);
            out.push_back(token);
            if (token >= 0 && token < model.hparams.start_speech_token) tokens.push_back(token);
        }
        tokens.insert(tokens.end(), (size_t)sil_n, sil);
        g_sampler_log = nullptr;
        if (sampler_log_enabled()) {
            std::string p = opts.t3_gguf_path;
            auto slash = p.find_last_of("\\/");
            if (slash != std::string::npos) {
                std::ofstream f(p.substr(0, slash) + "/nano_t3_dump.txt");
                if (f) {
                    f << "bpe";
                    for (int32_t id : text_tokens) f << " " << id;
                    f << "\ntext";
                    for (int32_t id : text_tokens) f << " " << id;
                    f << "\npredicted";
                    for (int32_t id : out) f << " " << id;
                    f << "\ndropped";
                    for (int32_t id : tokens) f << " " << id;
                    f << "\npredicted_count " << out.size();
                    f << "\ndropped_count " << tokens.size();
                    f << "\neos " << (token == stop ? 1 : 0);
                    f << "\nn_embd " << model.hparams.n_embd;
                    f << "\nn_head " << model.hparams.n_head;
                    f << "\nn_layer " << model.hparams.n_layer;
                    f << "\nn_ctx " << model.hparams.n_ctx;
                    f << "\ntext_vocab " << model.hparams.n_text_vocab;
                    f << "\nspeech_vocab " << model.hparams.n_speech_vocab;
                    f << "\nstart_speech " << model.hparams.start_speech_token;
                    f << "\nstop_speech " << model.hparams.stop_speech_token;
                    f << "\ncond_prompt_len " << model.hparams.cond_prompt_len << "\n";
                }
            }
        }
        return tokens;
    }
};
Engine::Engine(const EngineOptions& o) : pimpl_(std::make_unique<Impl>(o)) { pimpl_->init(); }
Engine::~Engine() = default;
std::vector<float> Engine::synthesize(const std::string& text) {
    return s3gen_synthesize(pimpl_->generate_t3(text));
}
}
