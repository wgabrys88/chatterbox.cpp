#include "tts-cpp/chatterbox/engine.h"
#if defined(TTS_FAMILY_V3)
#include "mtl_numbers.h"
#include "mtl_external_tokenizer.h"
#else
#include "tts-cpp/chatterbox/gpt2.h"
#include "gpt2_bpe.h"
#endif
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <memory>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "chatterbox_t3_internal.h"
#include "s3gen_pipeline.h"
#include "utterance_split.h"
#include "text_prepare.h"
#include <limits>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
namespace tts_cpp::chatterbox {
using namespace detail;
#if defined(TTS_FAMILY_V3)
static std::vector<int32_t> drop_invalid_tokens(const std::vector<int32_t> & x, int32_t sos, int32_t eos) {
    size_t s = 0, e = x.size();
    for (size_t i = 0; i < x.size(); ++i) if (x[i] == sos) { s = i + 1; break; }
    for (size_t i = 0; i < x.size(); ++i) if (x[i] == eos) { e = i; break; }
    if (s > e) throw std::runtime_error("drop_invalid");
    return std::vector<int32_t>(x.begin() + (std::ptrdiff_t)s, x.begin() + (std::ptrdiff_t)e);
}
#endif
struct Engine::Impl {
    EngineOptions opts;
    chatterbox_model model{};
    ggml_gallocr_t allocr = nullptr;
#if defined(TTS_FAMILY_V3)
    std::unique_ptr<mtl_external_tokenizer> tokenizer;
#endif
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
#if defined(TTS_FAMILY_V3)
        if (!std::filesystem::is_regular_file(std::filesystem::u8path(opts.tokenizer_python)) ||
            !std::filesystem::is_regular_file(std::filesystem::u8path(opts.tokenizer_script)) ||
            !std::filesystem::is_regular_file(std::filesystem::u8path(opts.tokenizer_source)) ||
            !std::filesystem::is_regular_file(std::filesystem::u8path(opts.tokenizer_tts_source)) ||
            !std::filesystem::is_regular_file(std::filesystem::u8path(opts.tokenizer_json)) ||
            !std::filesystem::is_regular_file(std::filesystem::u8path(opts.cangjie_json)) ||
            !std::filesystem::is_regular_file(std::filesystem::u8path(opts.dicta_model)))
            throw std::runtime_error("official tokenizer asset missing");
        tokenizer=std::make_unique<mtl_external_tokenizer>(mtl_external_tokenizer_options{opts.tokenizer_python,opts.tokenizer_script,opts.tokenizer_source,opts.tokenizer_tts_source,opts.tokenizer_json,opts.cangjie_json,opts.dicta_model},opts.language_id);
#endif
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        s3gen_preload(opts.s3gen_gguf_path, model.backend);
    }
    ~Impl() {
#if defined(TTS_FAMILY_V3)
        tokenizer.reset();
#endif
        s3gen_unload();
        if (allocr) ggml_gallocr_free(allocr);
        if (model.buffer_w) ggml_backend_buffer_free(model.buffer_w);
        if (model.buffer_kv) ggml_backend_buffer_free(model.buffer_kv);
        if (model.backend) ggml_backend_free(model.backend);
        if (model.ctx_w) ggml_free(model.ctx_w);
        if (model.ctx_kv) ggml_free(model.ctx_kv);
    }
    void synthesize(const std::string& text, std::vector<float>& pcm, SynthesizeStats* stats) {
        pcm.clear(); if(stats)*stats=SynthesizeStats{};
#if defined(TTS_FAMILY_V3)
        if(opts.language_id.empty()) throw std::runtime_error("language");
        const std::string language_token = "[" + opts.language_id + "]";
        const std::string language_list = "," + model.language_tokens + ",";
        if (language_list.find("," + language_token + ",") == std::string::npos) throw std::runtime_error("language token not present in converted tokenizer");
        if(!tokenizer) throw std::runtime_error("official tokenizer worker unavailable");
        const std::string punctuated = tokenizer->punctuation(text);
        const auto numbers = mtl_numbers{}.verbalize_numbers(punctuated, opts.language_id);
        const auto official = tokenizer->tokenize(numbers.text);
        const EncodeText encode = [&](const std::string& input) {
            if (input != numbers.text) throw std::runtime_error("unexpected tokenizer input mutation");
            auto ids=official.ids;
            ids.insert(ids.begin(),model.hparams.start_text_token);
            ids.push_back(model.hparams.stop_text_token);
            return ids;
        };
        PreparedText prepared;
        prepared.text = numbers.text;
#else
        gpt2_bpe bpe;
        if(!bpe.load_from_arrays(model.tok_tokens, model.tok_merges)) throw std::runtime_error("tokenizer");
        const EncodeText encode = [&](const std::string& input) {return bpe.tokenize(gpt2_bpe::punc_norm(input));};
        const auto prepared=prepare_text(text);
#endif
        const auto u=encode_utterance(prepared,encode);
        std::mt19937 rng(effective_seed());
        SynthesizeStats unit;
        auto tokens=generate_t3(u.ids,rng,&unit);
        auto wav=s3gen_synthesize(tokens);
#if defined(TTS_FAMILY_V3)
        const size_t crop=960;
        if(tokens.size()<2 || wav.size()!=tokens.size()*size_t(960)) throw std::runtime_error("S3 sample/token length mismatch");
        wav.resize(wav.size()-crop);
#else
        if(wav.size()!=tokens.size()*size_t(960)) throw std::runtime_error("S3 sample/token length mismatch");
#endif
        for(float v:wav) if(!std::isfinite(v)) throw std::runtime_error("non-finite S3 sample");
        const size_t trim_fade=(size_t)effective_trim_fade();
        std::fill_n(wav.begin(),std::min(wav.size(),trim_fade),0.0f);
        for(size_t j=trim_fade;j<std::min(wav.size(),2*trim_fade);++j)
            wav[j]*=trim_fade>1?0.5f*(1.0f-std::cos(float(M_PI)*float(j-trim_fade)/float(trim_fade-1))):1.0f;
        const size_t offset=pcm.size();
        if(wav.empty() || wav.size()>(size_t(UINT32_MAX)-36)/2 || offset>(size_t(UINT32_MAX)-36)/2-wav.size())
            throw std::runtime_error("empty audio or RIFF size limit");
        pcm.insert(pcm.end(),wav.begin(),wav.end());
        accumulate_unit(stats,unit);
    }
    std::vector<int32_t> generate_t3(const std::vector<int32_t>& text_tokens, std::mt19937& rng, SynthesizeStats* stats) {
        const int n_predict=effective_n_predict();
        if(n_predict<1)throw std::runtime_error("n-predict must be positive");
        if (model.buffer_kv) ggml_backend_buffer_clear(model.buffer_kv, 0);
        int n_past = 0;
        std::vector<int32_t> predicted;
        std::vector<float> logits;
        eval_prompt(model, allocr, text_tokens, logits, n_past);
#if defined(TTS_FAMILY_V3)
        const int32_t stop = model.hparams.stop_speech_token;
        const int32_t sos = model.hparams.start_speech_token;
        std::vector<int32_t> generated;
        generated.push_back(sos);
        predicted.reserve((size_t)n_predict);
        for (int i = 0; i < n_predict && n_past + 1 <= model.hparams.n_ctx; ++i) {
            int32_t token = sample_next_token_ex(logits, generated, rng);
            predicted.push_back(token);
            generated.push_back(token);
            if (token == stop) break;
            eval_step(model, allocr, n_past++, token, i + 1, logits);
        }
#else
        int32_t token = 0;
        predicted.reserve((size_t)n_predict + 1);
        const int32_t stop = model.hparams.stop_speech_token;
        const std::vector<int32_t> first_pen = { model.hparams.start_speech_token };
        token = sample_next_token_ex(logits, first_pen, rng);
        predicted.push_back(token);
        for (int step = 1; step < n_predict && token != stop && n_past + 1 <= model.hparams.n_ctx; ++step) {
            eval_step(model, allocr, n_past++, token, logits);
            token = sample_next_token_ex(logits, predicted, rng);
            predicted.push_back(token);
        }
#endif
#if defined(TTS_FAMILY_V3)
        if (predicted.empty()) throw std::runtime_error("T3 produced no tokens");
        if (predicted.back() != model.hparams.stop_speech_token) {
            char msg[256];
            std::snprintf(msg, sizeof(msg), "T3 no EOS: predicted=%d n_past=%d n_predict=%d n_ctx=%d text_tokens=%d",
                (int)predicted.size(), n_past, n_predict, model.hparams.n_ctx, (int)text_tokens.size());
            throw std::runtime_error(msg);
        }
        auto dropped = drop_invalid_tokens(predicted, model.hparams.start_speech_token, model.hparams.stop_speech_token);
        if (stats) {
            stats->predicted_count = (int)predicted.size();
            stats->dropped_count = (int)dropped.size();
            stats->eos = predicted.back() == model.hparams.stop_speech_token ? 1 : 0;
            stats->n_past = n_past;
            stats->text_tokens = (int)text_tokens.size();
        }
        return dropped;
#else
        if (predicted.empty() || predicted.back() != model.hparams.stop_speech_token) {
            char msg[256];
            std::snprintf(msg, sizeof(msg), "T3 no EOS: predicted=%d n_past=%d n_predict=%d n_ctx=%d text_tokens=%d",
                (int)predicted.size(), n_past, n_predict, model.hparams.n_ctx, (int)text_tokens.size());
            throw std::runtime_error(msg);
        }
        std::vector<int32_t> speech;
        speech.reserve((size_t)n_predict + (size_t)SIL_COUNT);
        for (int32_t next : predicted) if (next >= 0 && next < 6561) speech.push_back(next);
        speech.insert(speech.end(), (size_t)SIL_COUNT, S3GEN_SIL);
        if (stats) {
            stats->predicted_count = (int)predicted.size();
            stats->dropped_count = (int)speech.size();
            stats->eos = predicted.back() == model.hparams.stop_speech_token ? 1 : 0;
            stats->n_past = n_past;
            stats->text_tokens = (int)text_tokens.size();
        }
        return speech;
#endif
    }
};
Engine::Engine(const EngineOptions& o) : pimpl_(std::make_unique<Impl>(o)) { pimpl_->init(); }
Engine::~Engine() = default;
void Engine::synthesize(const std::string& text, std::vector<float>& pcm, SynthesizeStats * stats) {
    pimpl_->synthesize(text, pcm, stats);
}
}
