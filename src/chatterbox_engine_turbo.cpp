#include "tts-cpp/chatterbox/engine.h"
#include <cstdio>
#include <fstream>
#include <iomanip>
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
    void synthesize(const std::string& text, Engine::AudioCallback cb, void * user, SynthesizeStats * stats) {
        if (!cb) throw std::runtime_error("audio callback");
        gpt2_bpe bpe;
        bpe.load_from_arrays(model.tok_tokens, model.tok_merges);
        const TokenCount count = [&](const std::string& s) { return (int)bpe.tokenize(gpt2_bpe::punc_norm(s)).size(); };
        const auto units = split_utterances(text, effective_split_tokens(), count);
        std::mt19937 rng(effective_seed());
        if (stats) *stats = SynthesizeStats{};
        for (size_t i = 0; i < units.size(); ++i) {
            SynthesizeStats unit;
            const auto wav = s3gen_synthesize(generate_t3(units[i], bpe, rng, &unit));
            if (!wav.empty()) cb(wav.data(), wav.size(), user);
            accumulate_unit(stats, unit, (int)i, (int)units.size(), units[i]);
        }
    }
    std::vector<int32_t> generate_t3(const std::string& text, const gpt2_bpe& bpe, std::mt19937& rng, SynthesizeStats * stats) {
        const int n_predict = effective_n_predict();
        const int sil_n = effective_silence_count();
        const int sil = effective_silence_token();
        auto text_tokens = bpe.tokenize(gpt2_bpe::punc_norm(text));
        if (model.buffer_kv) ggml_backend_buffer_clear(model.buffer_kv, 0);
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
                slog.open(p.substr(0, slash) + "/turbo_sample_dump.csv");
        }
        g_sampler_log = slog.is_open() ? &slog : nullptr;
        g_sampler_step = 0;
        if (g_sampler_log)
            *g_sampler_log << "step,chosen,raw_argmax,raw_argmax_logit,raw0_id,raw0_logit,raw1_id,raw1_logit,raw2_id,raw2_logit,raw3_id,raw3_logit,raw4_id,raw4_logit,raw5_id,raw5_logit,raw6_id,raw6_logit,raw7_id,raw7_logit,raw8_id,raw8_logit,raw9_id,raw9_logit,chosen_prob,sil4299_prob,sil4299_rank,sil4299_seen,gen_len,top0_id,top0_prob,top1_id,top1_prob,top2_id,top2_prob,top3_id,top3_prob,top4_id,top4_prob,top5_id,top5_prob,top6_id,top6_prob,top7_id,top7_prob,top8_id,top8_prob,top9_id,top9_prob\n";
        eval_prompt(model, allocr, text_tokens, logits, n_past);
        const int prompt_len = n_past;
        const std::vector<int32_t> first_pen = { model.hparams.start_speech_token };
        token = sample_next_token_ex(logits, first_pen, rng);
        out.push_back(token);
        if (token >= 0 && token < model.hparams.start_speech_token) tokens.push_back(token);
        for (int step = 1; step < n_predict && token != stop && n_past + 1 <= model.hparams.n_ctx; ++step) {
            eval_step(model, allocr, n_past++, token, logits);
            token = sample_next_token_ex(logits, out, rng);
            out.push_back(token);
            if (token >= 0 && token < model.hparams.start_speech_token) tokens.push_back(token);
        }
        if (token != stop) {
            g_sampler_log = nullptr;
            char msg[256];
            std::snprintf(msg, sizeof(msg), "T3 no EOS: predicted=%d n_past=%d n_predict=%d n_ctx=%d text_tokens=%d",
                (int)out.size(), n_past, n_predict, model.hparams.n_ctx, (int)text_tokens.size());
            throw std::runtime_error(msg);
        }
        tokens.insert(tokens.end(), (size_t)sil_n, sil);
        g_sampler_log = nullptr;
        if (sampler_log_enabled()) {
            std::string p = opts.t3_gguf_path;
            auto slash = p.find_last_of("\\/");
            if (slash != std::string::npos) {
                std::ofstream f(p.substr(0, slash) + "/turbo_t3_dump.txt");
                if (f) {
                    std::vector<int32_t> cond((size_t)model.hparams.cond_prompt_len);
                    ggml_backend_tensor_get(model.builtin_cond_prompt_tokens, cond.data(), 0, cond.size() * sizeof(int32_t));
                    std::vector<float> speaker((size_t)ggml_nelements(model.builtin_speaker_emb));
                    ggml_backend_tensor_get(model.builtin_speaker_emb, speaker.data(), 0, speaker.size() * sizeof(float));
                    f << "punc_norm " << gpt2_bpe::punc_norm(text);
                    f << "\nbpe";
                    for (int32_t id : text_tokens) f << " " << id;
                    f << "\ntext";
                    for (int32_t id : text_tokens) f << " " << id;
                    f << "\ncond";
                    for (int32_t id : cond) f << " " << id;
                    f << "\nspeaker";
                    f << std::setprecision(9);
                    for (float v : speaker) f << " " << v;
                    f << "\npredicted";
                    for (int32_t id : out) f << " " << id;
                    f << "\ndropped";
                    for (int32_t id : tokens) f << " " << id;
                    f << "\npredicted_count " << out.size();
                    f << "\ndropped_count " << tokens.size();
                    f << "\neos " << (token == stop ? 1 : 0);
                    f << "\nprompt_len " << prompt_len;
                    f << "\nn_past " << n_past;
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
        if (stats) {
            stats->predicted_count = (int)out.size();
            stats->dropped_count = (int)tokens.size();
            stats->eos = token == stop ? 1 : 0;
            stats->n_past = n_past;
            stats->text_tokens = (int)text_tokens.size();
        }
        return tokens;
    }
};
Engine::Engine(const EngineOptions& o) : pimpl_(std::make_unique<Impl>(o)) { pimpl_->init(); }
Engine::~Engine() = default;
void Engine::synthesize(const std::string& text, Engine::AudioCallback cb, void * user, SynthesizeStats * stats) {
    pimpl_->synthesize(text, cb, user, stats);
}
}
