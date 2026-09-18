#include "tts-cpp/chatterbox/engine.h"
#if defined(TTS_FAMILY_V3)
#include "tts-cpp/chatterbox/v3.h"
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
#include "execution_trace.h"
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
static std::string json_number_rewrites(const std::vector<mtl_number_rewrite> & rewrites) {
    std::string out = "[";
    for (size_t i = 0; i < rewrites.size(); ++i) {
        if (i) out += ',';
        out += "{\"kind\":\"icu_spellout\",\"source\":" + json_string(rewrites[i].source) + ",\"target\":" + json_string(rewrites[i].target) + "}";
    }
    return out + "]";
}
#endif
struct Engine::Impl {
    EngineOptions opts;
    chatterbox_model model{};
    ggml_gallocr_t allocr = nullptr;
    explicit Impl(const EngineOptions& o) : opts(o) {}
    void init(ExecutionTrace* trace) {
        ggml_time_init();
        ggml_log_set([](ggml_log_level level, const char * text, void *) {
            if (level >= GGML_LOG_LEVEL_WARN && text) {
                fputs(text, stderr);
                fflush(stderr);
            }
        }, nullptr);
        trace_event(trace,"backend_load_start","model_load");
        model.backend = init_backend();
        trace_event(trace,"backend_load_end","model_load",{{"backend",json_string(ggml_backend_name(model.backend))},
            {"device",json_string(ggml_backend_dev_description(ggml_backend_get_device(model.backend)))}});
        auto load_start=TraceClock::now();
        trace_event(trace,"t3_load_start","model_load",{{"path",json_string(opts.t3_gguf_path)}});
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
        if (sha256_text(model.tokenizer_json) != model.tokenizer_sha256) throw std::runtime_error("embedded tokenizer JSON does not match converted model");
        if (sha256_file(opts.tokenizer_json) != model.tokenizer_sha256) throw std::runtime_error("tokenizer JSON does not match converted model");
        if (sha256_file(opts.cangjie_json) != model.cangjie_sha256) throw std::runtime_error("Cangjie mapping does not match converted model");
        if (sha256_file(opts.tokenizer_source) != model.official_tokenizer_sha256) throw std::runtime_error("official tokenizer source does not match converted model");
        if (sha256_file(opts.tokenizer_tts_source) != model.official_tts_sha256) throw std::runtime_error("official TTS frontend source does not match converted model");
        trace_event(trace,"tokenizer_contract","model_load",{{"frontend_version","4"},{"tokenizer_sha256",json_string(model.tokenizer_sha256)},
            {"cangjie_sha256",json_string(model.cangjie_sha256)},{"language_tokens",json_string(model.language_tokens)},
            {"official_source_sha256",json_string(sha256_file(opts.tokenizer_source))},{"official_tts_source_sha256",json_string(sha256_file(opts.tokenizer_tts_source))},{"adapter_sha256",json_string(sha256_file(opts.tokenizer_script))}});
#endif
        trace_event(trace,"t3_load_end","model_load",{{"host_wall_s",json_number(elapsed(load_start))}});
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        load_start=TraceClock::now();trace_event(trace,"s3_load_start","model_load",{{"path",json_string(opts.s3gen_gguf_path)}});
        s3gen_preload(opts.s3gen_gguf_path, model.backend);
        trace_event(trace,"s3_load_end","model_load",{{"host_wall_s",json_number(elapsed(load_start))}});
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
    void synthesize(const std::string& text, std::vector<float>& pcm, SynthesizeStats* stats, ExecutionTrace* trace) {
        pcm.clear(); if(stats)*stats=SynthesizeStats{};
        auto begin=TraceClock::now();
        trace_event(trace,"backend_identity","model",{{"backend",json_string(ggml_backend_name(model.backend))},
            {"device",json_string(ggml_backend_dev_description(ggml_backend_get_device(model.backend)))},
            {"vulkan_device_index","0"},{"selected_backend_verified","true"},
            {"driver_source",json_string("server.log Vulkan initialization and client host_inventory")},
            {"layers",std::to_string(model.hparams.n_layer)},{"embedding",std::to_string(model.hparams.n_embd)},
            {"heads",std::to_string(model.hparams.n_head)}});
#if defined(TTS_FAMILY_V3)
        if(opts.language_id.empty()) throw std::runtime_error("language");
        const std::string language_token = "[" + opts.language_id + "]";
        const std::string language_list = "," + model.language_tokens + ",";
        if (language_list.find("," + language_token + ",") == std::string::npos) throw std::runtime_error("language token not present in converted tokenizer");
        const mtl_external_tokenizer_options tokenizer_opts{opts.tokenizer_python,opts.tokenizer_script,opts.tokenizer_source,opts.tokenizer_tts_source,opts.tokenizer_json,opts.cangjie_json,opts.dicta_model};
        const std::string punctuated = mtl_external_punc_norm(tokenizer_opts, text);
        const auto numbers = mtl_numbers{}.verbalize_numbers(punctuated, opts.language_id);
        trace_event(trace, "number_verbalized", "prepare", {
            {"original_text", json_string(text)},
            {"punctuation_text", json_string(punctuated)},
            {"transport_text", json_string(numbers.text)},
            {"changed", numbers.rewrites.empty() ? "false" : "true"},
            {"changes", json_number_rewrites(numbers.rewrites)},
            {"language_id", json_string(opts.language_id)},
            {"provider", json_string(numbers.provider)},
            {"policy", json_string("official_punc_then_icu_cldr_spellout")},
        });
        const auto official = mtl_external_tokenize(tokenizer_opts, numbers.text, opts.language_id);
        trace_event(trace,"official_tokenizer","encode",{{"language_id",json_string(opts.language_id)},
            {"source_text",json_string(numbers.text)},{"tokenizer_input",json_string(official.tokenizer_input)},
            {"token_count",std::to_string(official.ids.size())},{"implementation",json_string("upstream MTLTokenizer + Hugging Face tokenizers")}});
        const EncodeText encode = [&](const std::string& input) {
            if (input != numbers.text) throw std::runtime_error("unexpected tokenizer input mutation");
            auto ids=official.ids;
            ids.insert(ids.begin(),model.hparams.start_text_token);
            ids.push_back(model.hparams.stop_text_token);
            return ids;
        };
        PreparedText prepared;
        prepared.text = numbers.text;
        trace_event(trace, "text_prepared", "prepare", {
            {"original_sha256", json_string(sha256_text(text))},
            {"prepared_sha256", json_string(sha256_text(numbers.text))},
            {"text", json_string(numbers.text)},
            {"edits", "[]"},
            {"unhandled_spans", "[]"},
            {"explicit_boundaries", "[]"},
            {"policy", json_string("upstream_multilingual_frontend")},
            {"language_id", json_string(opts.language_id)},
        });
#else
        gpt2_bpe bpe;
        if(!bpe.load_from_arrays(model.tok_tokens, model.tok_merges)) throw std::runtime_error("tokenizer");
        const EncodeText encode = [&](const std::string& input) {return bpe.tokenize(gpt2_bpe::punc_norm(input));};
        const auto prepared=prepare_text(text, true, trace);
#endif
        const auto u=encode_utterance(prepared,encode,trace);
        trace_tokens(trace,"text",u.ids);
        trace_event(trace,"preparation_complete","prepare",{{"host_wall_s",json_number(elapsed(begin))}});
        std::mt19937 rng(effective_seed());
        auto unit_start=TraceClock::now();
        trace_event(trace,"unit_start","unit",{{"index","0"},{"count","1"},
            {"begin",std::to_string(u.begin)},{"end",std::to_string(u.end)},
            {"text",json_string(u.text)},{"tokenizer_preprocessed",json_string(
#if defined(TTS_FAMILY_V3)
                official.tokenizer_input
#else
                gpt2_bpe::punc_norm(u.text)
#endif
            )},{"text_ids",json_ids(u.ids)},
            {"text_tokens",std::to_string(u.ids.size())},{"conditioning_tokens",std::to_string(model.hparams.cond_prompt_len)},
            {"architectural_context",std::to_string(model.hparams.n_ctx)},{"n_predict",std::to_string(effective_n_predict())},
            {"seed",std::to_string(effective_seed())},{"rng",json_string("mt19937 request scope")}});
        try {
            SynthesizeStats unit;
            auto tokens=generate_t3(u.ids,rng,&unit,trace);
            auto decode_start=TraceClock::now();
            trace_event(trace,"s3_start","s3",{{"index","0"},{"s3_input_ids",json_ids(tokens)},
                {"cfm_steps",std::to_string(effective_cfm_steps())}
#if defined(TTS_FAMILY_V3)
                ,{"cfm_cfg",json_number(effective_cfm_cfg())}
#endif
            });
            auto wav=s3gen_synthesize(tokens);
            trace_tokens(trace,"s3",tokens);
            const size_t raw=wav.size();
            trace_event(trace,"s3_complete","s3",{{"index","0"},{"host_wall_s",json_number(elapsed(decode_start))},
                {"raw_samples",std::to_string(raw)},{"completed_invocations","1"}});
#if defined(TTS_FAMILY_V3)
            const size_t crop=960;
            if(tokens.size()<2 || wav.size()!=tokens.size()*size_t(960)) throw std::runtime_error("S3 sample/token length mismatch");
            wav.resize(wav.size()-crop);
#else
            const size_t crop=0;
            if(wav.size()!=tokens.size()*size_t(960)) throw std::runtime_error("S3 sample/token length mismatch");
#endif
            auto assembly_start=TraceClock::now();
            for(float v:wav) if(!std::isfinite(v)) throw std::runtime_error("non-finite S3 sample");
            const size_t trim_fade=(size_t)effective_trim_fade();
            std::fill_n(wav.begin(),std::min(wav.size(),trim_fade),0.0f);
            for(size_t j=trim_fade;j<std::min(wav.size(),2*trim_fade);++j)
                wav[j]*=trim_fade>1?0.5f*(1.0f-std::cos(float(M_PI)*float(j-trim_fade)/float(trim_fade-1))):1.0f;
            const size_t offset=pcm.size();
            if(wav.empty() || wav.size()>(size_t(UINT32_MAX)-36)/2 || offset>(size_t(UINT32_MAX)-36)/2-wav.size())
                throw std::runtime_error("empty audio or RIFF size limit");
            pcm.insert(pcm.end(),wav.begin(),wav.end());
            accumulate_unit(stats,unit,0,1,u.text);
            trace_event(trace,"unit_complete","unit",{{"index","0"},
                {"output_begin_sample",std::to_string(offset)},{"output_end_sample",std::to_string(pcm.size())},
                {"unit_samples",std::to_string(wav.size())},{"raw_samples",std::to_string(raw)},
                {"cropped_samples",std::to_string(crop)},{"onset_zero_samples",std::to_string(std::min(wav.size(),trim_fade))},
                {"faded_samples",std::to_string(wav.size()>trim_fade?std::min(wav.size()-trim_fade,trim_fade):0)},
                {"assembly_host_wall_s",json_number(elapsed(assembly_start))},{"host_wall_s",json_number(elapsed(unit_start))},
                {"t3_completed_invocations","1"},{"s3_completed_invocations","1"},{"eos","true"}});
        } catch(const std::exception& e) {
            trace_event(trace,"unit_failed","unit",{{"index","0"},{"error",json_string(e.what())}});throw;
        }
        trace_event(trace,"synthesis_complete","synthesis",{{"host_wall_s",json_number(elapsed(begin))},
            {"units","1"},{"samples",std::to_string(pcm.size())}});
    }
    std::vector<int32_t> generate_t3(const std::vector<int32_t>& text_tokens, std::mt19937& rng, SynthesizeStats* stats, ExecutionTrace* trace) {
        const int n_predict=effective_n_predict();
        if(n_predict<1)throw std::runtime_error("n-predict must be positive");
        auto generation_start=TraceClock::now();
        if (model.buffer_kv) ggml_backend_buffer_clear(model.buffer_kv, 0);
        int n_past = 0;
        std::vector<int32_t> predicted;
        std::vector<float> logits;
        auto decode_start=TraceClock::now();
        trace_event(trace,"t3_start","t3",{{"attempted_invocations","1"},{"requested_prediction_cap",std::to_string(n_predict)}});
        try {
        auto prefill_start=TraceClock::now();
        eval_prompt(model, allocr, text_tokens, logits, n_past);
        trace_event(trace,"prefill_complete","t3",{{"host_wall_s",json_number(elapsed(prefill_start))},
            {"prompt_length",std::to_string(n_past)},{"kv_rows",std::to_string(model.kv_rows)},
            {"sampler_order",json_string(
#if defined(TTS_FAMILY_V3)
                "cfg,repetition_penalty,temperature,min_p,top_p,sample"
#else
                "temperature,top_k,top_p,repetition_penalty,sample"
#endif
            )}});
        decode_start=TraceClock::now();
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
        } catch(const std::exception& e) {
            trace_event(trace,"t3_failed","t3",{{"raw_ids",json_ids(predicted)},{"raw_count_including_eos",std::to_string(predicted.size())},
                {"n_past",std::to_string(n_past)},{"kv_rows",std::to_string(model.kv_rows)},
                {"stop_reason",json_string("model_error")},{"error",json_string(e.what())},
                {"host_wall_s",json_number(elapsed(generation_start))}});throw;
        }
        const bool reached_eos=!predicted.empty()&&predicted.back()==model.hparams.stop_speech_token;
        trace_event(trace,"t3_result","t3",{{"raw_ids",json_ids(predicted)},
            {"raw_count_including_eos",std::to_string(predicted.size())},{"eos",reached_eos?"true":"false"},
            {"stop_reason",json_string(reached_eos?"eos":n_past+1>model.hparams.n_ctx?"context_limit":"prediction_limit")},
            {"n_past",std::to_string(n_past)},{"generation_host_wall_s",json_number(elapsed(decode_start))},
            {"eos_index",reached_eos?std::to_string(predicted.size()-1):"null"},
            {"host_wall_s",json_number(elapsed(generation_start))}});
        trace_tokens(trace,"t3",predicted);
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
        trace_event(trace,"t3_complete","t3",{{"valid_speech_count",std::to_string(dropped.size()-0)},
            {"removed_count",std::to_string(predicted.size()-(dropped.size()-0))},
            {"appended_silence_count","0"},{"s3_input_count",std::to_string(dropped.size())},
            {"legacy_dropped_definition",json_string("S3 input count, not omitted words")},
            {"completed_invocations","1"}});
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
        trace_event(trace,"t3_complete","t3",{{"valid_speech_count",std::to_string(speech.size()-3)},
            {"removed_count",std::to_string(predicted.size()-(speech.size()-3))},
            {"appended_silence_count","3"},{"s3_input_count",std::to_string(speech.size())},
            {"legacy_dropped_definition",json_string("S3 input count, not omitted words")},
            {"completed_invocations","1"}});
        return speech;
#endif
    }
};
Engine::Engine(const EngineOptions& o, ExecutionTrace* trace) : pimpl_(std::make_unique<Impl>(o)) { pimpl_->init(trace); }
Engine::~Engine() = default;
void Engine::synthesize(const std::string& text, std::vector<float>& pcm, SynthesizeStats * stats, ExecutionTrace* trace) {
    pimpl_->synthesize(text, pcm, stats, trace);
}
}
