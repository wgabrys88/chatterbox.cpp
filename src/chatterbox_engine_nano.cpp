#include "tts-cpp/chatterbox/engine.h"
#include "tts-cpp/chatterbox/nano.h"
#include <algorithm>
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
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
namespace tts_cpp::chatterbox {
using namespace detail;
namespace {
struct NanoAudioStream {
    std::vector<float> source_cache;
    std::vector<float> pending_tail;
    std::size_t emitted_samples = 0;

    static void emit(Engine::AudioCallback cb, void * user, const float * data, std::size_t n) {
        if (n) cb(data, n, user);
    }

    void emit_body(const float * body, std::size_t n, Engine::AudioCallback cb, void * user) {
        if (pending_tail.empty()) {
            emit(cb, user, body, n);
            return;
        }
        const std::size_t overlap = std::min({
            (std::size_t)STREAM_CROSSFADE_SAMPLES,
            pending_tail.size(),
            n,
        });
        if (pending_tail.size() > overlap)
            emit(cb, user, pending_tail.data(), pending_tail.size() - overlap);
        if (overlap) {
            std::vector<float> crossed(overlap);
            for (std::size_t i = 0; i < overlap; ++i) {
                const float in = overlap == 1 ? 1.0f : (float)i / (float)(overlap - 1);
                crossed[i] = pending_tail[pending_tail.size() - overlap + i] * (1.0f - in) + body[i] * in;
            }
            emit(cb, user, crossed.data(), crossed.size());
        }
        if (n > overlap) emit(cb, user, body + overlap, n - overlap);
    }

    void flush(const std::vector<int32_t>& tokens, bool final, Engine::AudioCallback cb, void * user) {
        std::vector<float> wav, source;
        s3gen_synthesize_stream(tokens, final, source_cache, wav, source);
        source_cache = std::move(source);
        if (wav.size() <= emitted_samples) return;
        const float * chunk = wav.data() + emitted_samples;
        const std::size_t chunk_len = wav.size() - emitted_samples;
        if (final) {
            emit_body(chunk, chunk_len, cb, user);
            pending_tail.clear();
            emitted_samples += chunk_len;
            return;
        }
        if (chunk_len <= (std::size_t)STREAM_CROSSFADE_SAMPLES) return;
        const std::size_t body_len = chunk_len - (std::size_t)STREAM_CROSSFADE_SAMPLES;
        emit_body(chunk, body_len, cb, user);
        pending_tail.assign(chunk + body_len, chunk + chunk_len);
        emitted_samples += body_len;
    }
};
}
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
    void synthesize(const std::string& text, Engine::AudioCallback cb, void * user) {
        if (!cb) throw std::runtime_error("audio callback");
        const int n_predict = effective_n_predict();
        const int sil_n = effective_silence_count();
        const int sil = effective_silence_token();
        std::mt19937 rng(effective_seed());
        gpt2_bpe bpe;
        bpe.load_from_arrays(model.tok_tokens, model.tok_merges);
        auto text_tokens = bpe.tokenize(gpt2_bpe::punc_norm(text));
        if (model.buffer_kv) ggml_backend_buffer_clear(model.buffer_kv, 0);
        int n_past = 0;
        int32_t token = 0;
        std::vector<int32_t> predicted, speech;
        predicted.reserve((size_t)n_predict + 1);
        speech.reserve((size_t)n_predict + (size_t)sil_n);
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
            *g_sampler_log << "step,chosen,raw_argmax,raw_argmax_logit,raw0_id,raw0_logit,raw1_id,raw1_logit,raw2_id,raw2_logit,raw3_id,raw3_logit,raw4_id,raw4_logit,raw5_id,raw5_logit,raw6_id,raw6_logit,raw7_id,raw7_logit,raw8_id,raw8_logit,raw9_id,raw9_logit,chosen_prob,sil4299_prob,sil4299_rank,sil4299_seen,gen_len,top0_id,top0_prob,top1_id,top1_prob,top2_id,top2_prob,top3_id,top3_prob,top4_id,top4_prob,top5_id,top5_prob,top6_id,top6_prob,top7_id,top7_prob,top8_id,top8_prob,top9_id,top9_prob\n";
        eval_prompt(model, allocr, text_tokens, logits, n_past);
        const int prompt_len = n_past;
        NanoAudioStream stream;
        auto accept = [&](int32_t next) {
            predicted.push_back(next);
            if (next >= 0 && next < model.hparams.start_speech_token) {
                speech.push_back(next);
                if (speech.size() % (size_t)STREAM_TOKENS == 0)
                    stream.flush(speech, false, cb, user);
            }
        };
        const std::vector<int32_t> first_pen = { model.hparams.start_speech_token };
        token = sample_next_token_ex(logits, first_pen, rng);
        accept(token);
        for (int step = 1; step < n_predict && token != stop && n_past + 1 <= model.hparams.n_ctx; ++step) {
            eval_step(model, allocr, n_past++, token, logits);
            token = sample_next_token_ex(logits, predicted, rng);
            accept(token);
        }
        speech.insert(speech.end(), (size_t)sil_n, sil);
        stream.flush(speech, true, cb, user);
        g_sampler_log = nullptr;
        if (sampler_log_enabled()) {
            std::string p = opts.t3_gguf_path;
            auto slash = p.find_last_of("\\/");
            if (slash != std::string::npos) {
                std::ofstream f(p.substr(0, slash) + "/nano_t3_dump.txt");
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
                    for (int32_t id : predicted) f << " " << id;
                    f << "\ndropped";
                    for (int32_t id : speech) f << " " << id;
                    f << "\npredicted_count " << predicted.size();
                    f << "\ndropped_count " << speech.size();
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
    }
};
Engine::Engine(const EngineOptions& o) : pimpl_(std::make_unique<Impl>(o)) { pimpl_->init(); }
Engine::~Engine() = default;
void Engine::synthesize(const std::string& text, Engine::AudioCallback cb, void * user) {
    pimpl_->synthesize(text, cb, user);
}
}
