#include "tts-cpp/chatterbox/engine.h"
#include "tts-cpp/chatterbox/nano.h"
#include <algorithm>
#include <cstdio>
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
namespace {
struct NanoAudioStream {
    std::vector<float> source_cache;
    std::vector<float> source_scratch;
    std::vector<float> wav_scratch;
    std::vector<float> pending_tail;
    std::vector<float> cross_scratch;
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
            cross_scratch.resize(overlap);
            for (std::size_t i = 0; i < overlap; ++i) {
                const float in = overlap == 1 ? 1.0f : (float)i / (float)(overlap - 1);
                cross_scratch[i] = pending_tail[pending_tail.size() - overlap + i] * (1.0f - in) + body[i] * in;
            }
            emit(cb, user, cross_scratch.data(), cross_scratch.size());
        }
        if (n > overlap) emit(cb, user, body + overlap, n - overlap);
    }

    void flush(const std::vector<int32_t>& tokens, bool final, Engine::AudioCallback cb, void * user) {
        s3gen_synthesize_stream(tokens, final, source_cache, wav_scratch, source_scratch);
        source_cache.swap(source_scratch);
        source_scratch.clear();
        if (wav_scratch.size() <= emitted_samples) return;
        const float * chunk = wav_scratch.data() + emitted_samples;
        const std::size_t chunk_len = wav_scratch.size() - emitted_samples;
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
            synthesize_unit(units[i], bpe, rng, cb, user, &unit);
            accumulate_unit(stats, unit, (int)i, (int)units.size(), units[i]);
        }
    }
    void synthesize_unit(const std::string& text, const gpt2_bpe& bpe, std::mt19937& rng, Engine::AudioCallback cb, void * user, SynthesizeStats * stats) {
        const int n_predict = effective_n_predict();
        const int sil_n = effective_silence_count();
        const int sil = effective_silence_token();
        auto text_tokens = bpe.tokenize(gpt2_bpe::punc_norm(text));
        if (model.buffer_kv) ggml_backend_buffer_clear(model.buffer_kv, 0);
        int n_past = 0;
        int32_t token = 0;
        std::vector<int32_t> predicted, speech;
        predicted.reserve((size_t)n_predict + 1);
        speech.reserve((size_t)n_predict + (size_t)sil_n);
        const int32_t stop = model.hparams.stop_speech_token;
        // Streaming: every STREAM_TOKENS the S3 decoder re-runs over the whole
        // prefix (prompt + all speech so far) and emits the new tail, so audio
        // starts early at O(n^2) decode cost. Batching: one S3 decode at EOS.
        const bool streaming = effective_mode() == Mode::Streaming;
        std::vector<float> logits;
        eval_prompt(model, allocr, text_tokens, logits, n_past);
        NanoAudioStream stream;
        auto accept = [&](int32_t next) {
            predicted.push_back(next);
            if (next >= 0 && next < model.hparams.start_speech_token) {
                speech.push_back(next);
                if (streaming && speech.size() % (size_t)STREAM_TOKENS == 0)
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
        if (token != stop) {
            char msg[256];
            std::snprintf(msg, sizeof(msg), "T3 no EOS: predicted=%d n_past=%d n_predict=%d n_ctx=%d text_tokens=%d",
                (int)predicted.size(), n_past, n_predict, model.hparams.n_ctx, (int)text_tokens.size());
            throw std::runtime_error(msg);
        }
        speech.insert(speech.end(), (size_t)sil_n, sil);
        stream.flush(speech, true, cb, user);
        if (stats) {
            stats->predicted_count = (int)predicted.size();
            stats->dropped_count = (int)speech.size();
            stats->eos = token == stop ? 1 : 0;
            stats->n_past = n_past;
            stats->text_tokens = (int)text_tokens.size();
        }
    }
};
Engine::Engine(const EngineOptions& o) : pimpl_(std::make_unique<Impl>(o)) { pimpl_->init(); }
Engine::~Engine() = default;
void Engine::synthesize(const std::string& text, Engine::AudioCallback cb, void * user, SynthesizeStats * stats) {
    pimpl_->synthesize(text, cb, user, stats);
}
}
