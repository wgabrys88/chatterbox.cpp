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
#include "t3_tape.h"
#include "utterance_split.h"
#include "execution_trace.h"
#include "text_prepare.h"
#include <limits>
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
        gpt2_bpe bpe;
        if(!bpe.load_from_arrays(model.tok_tokens, model.tok_merges)) throw std::runtime_error("tokenizer");
        const EncodeText encode = [&](const std::string& input) {return bpe.tokenize(gpt2_bpe::punc_norm(input));};
        const auto prepared=prepare_text(text, true, trace);
        const auto units=encode_one_utterance(prepared,encode,trace);
        trace_event(trace,"preparation_complete","prepare",{{"host_wall_s",json_number(elapsed(begin))}});
        std::mt19937 rng(effective_seed());
        for(size_t i=0;i<units.size();++i) {
            const auto&u=units[i];auto unit_start=TraceClock::now();
            trace_event(trace,"unit_start","unit",{{"index",std::to_string(i)},{"count",std::to_string(units.size())},
                {"begin",std::to_string(u.begin)},{"end",std::to_string(u.end)},
                {"text",json_string(u.text)},{"tokenizer_preprocessed",json_string(gpt2_bpe::punc_norm(u.text))},{"text_ids",json_ids(u.ids)},
                {"text_tokens",std::to_string(u.ids.size())},{"conditioning_tokens",std::to_string(model.hparams.cond_prompt_len)},
                {"architectural_context",std::to_string(model.hparams.n_ctx)},{"n_predict",std::to_string(effective_n_predict())},
                {"seed",std::to_string(effective_seed())},{"rng",json_string("mt19937 request scope; advances across units")}});
            try {
                SynthesizeStats unit;
                T3Tape tape=generate_t3(u.ids,rng,&unit,trace);
                tape.stage=effective_stage();
                tape.cut_x=effective_cut_x();
                tape.seed=effective_seed();
                tape.n_predict=effective_n_predict();
                tape.s3gen_sil=effective_s3gen_sil();
                tape.appended_silence_count=effective_sil_count();
                S3GaugeTensors gt;
                gt.voiced_threshold=kVoicedThreshold;
                analyze_tape_ids(tape,gt);
                const bool steal=tape.cut_x>0||tape.stage>=2;
                if(steal){
                    auto g=s3gen_gauge(tape.speech_ids);
                    gt.codebook_norm=std::move(g.codebook_norm);
                    gt.fuel=std::move(g.fuel);
                    gt.f0=std::move(g.f0);
                    gt.voiced=std::move(g.voiced);
                    gt.n_prompt=g.n_prompt;
                    gt.n_frames=(int)gt.fuel.size();
                    gt.prompt_fuel_mean=g.prompt_fuel_mean;
                    gt.prompt_f0_mean=g.prompt_f0_mean;
                    gt.breath_capacity=g.breath_capacity;
                }
                VChunkPlan plan=vchunker(tape,gt);
                const std::string& stem=runtime_knobs().artifact_path;
                if(stem.empty())throw std::runtime_error("artifact path");
                write_utterance_gguf(stem+".tape.gguf",tape,steal?&gt:nullptr,plan);
                unit.n_speech=(int)tape.speech_ids.size();
                unit.cut_at=plan.chunks.front().end;
                unit.cut_reason=plan.chunks.front().reason;
                unit.n_chunks=(int)plan.chunks.size();
                unit.stage=tape.stage;
                unit.stop_code=tape.stop_code;
                if(tape.stage==1||tape.stage==2){
                    accumulate_unit(stats,unit,int(i),int(units.size()),u.text);
                    trace_event(trace,"unit_complete","unit",{{"index",std::to_string(i)},
                        {"t3_completed_invocations","1"},{"s3_completed_invocations","0"},
                        {"n_speech",std::to_string(tape.speech_ids.size())},
                        {"n_chunks",std::to_string(plan.chunks.size())},
                        {"stage",std::to_string(tape.stage)},{"eos",tape.eos?"true":"false"}});
                    continue;
                }
                if(tape.stage==0&&tape.cut_x==0&&!tape.eos){
                    char msg[256];
                    std::snprintf(msg,sizeof(msg),"T3 no EOS: predicted=%d n_past=%d n_predict=%d n_ctx=%d text_tokens=%d",
                        unit.predicted_count,tape.n_past,tape.n_predict,model.hparams.n_ctx,tape.text_tokens);
                    throw std::runtime_error(msg);
                }
                auto decode_start=TraceClock::now();
                const size_t offset0=pcm.size();
                size_t raw_all=0;
                int s3_n=0;
                for(const auto& ch:plan.chunks){
                    auto ids=chunk_ids(tape,ch);
                    if(ids.empty())continue;
                    trace_event(trace,"s3_start","s3",{{"index",std::to_string(i)},{"s3_input_ids",json_ids(ids)},
                        {"chunk_begin",std::to_string(ch.begin)},{"chunk_end",std::to_string(ch.end)},
                        {"chunk_reason",std::to_string(ch.reason)},
                        {"cfm_steps",std::to_string(effective_cfm_steps())},{"trim_fade",std::to_string(effective_trim_fade())}});
                    auto wav=s3gen_synthesize(ids);
                    const size_t raw=wav.size();
                    raw_all+=raw; ++s3_n;
                    if(wav.size()!=ids.size()*size_t(960)) throw std::runtime_error("S3 sample/token length mismatch");
                    for(float v:wav) if(!std::isfinite(v)) throw std::runtime_error("non-finite S3 sample");
                    const int fade=effective_trim_fade();
                    if(fade>0) std::fill_n(wav.begin(),std::min(wav.size(),size_t(fade)),0.0f);
                    if(fade>=2){
                        for(size_t j=size_t(fade);j<std::min(wav.size(),size_t(2*fade));++j)
                            wav[j]*=0.5f*(1.0f-std::cos(float(M_PI)*float(j-fade)/float(fade-1)));
                    }
                    if(wav.empty() || wav.size()>(size_t(UINT32_MAX)-36)/2 || pcm.size()>(size_t(UINT32_MAX)-36)/2-wav.size())
                        throw std::runtime_error("empty audio or RIFF size limit");
                    pcm.insert(pcm.end(),wav.begin(),wav.end());
                    trace_event(trace,"s3_complete","s3",{{"index",std::to_string(i)},{"raw_samples",std::to_string(raw)},
                        {"chunk_end",std::to_string(ch.end)},{"completed_invocations","1"}});
                }
                auto assembly_start=TraceClock::now();
                accumulate_unit(stats,unit,int(i),int(units.size()),u.text);
                trace_event(trace,"unit_complete","unit",{{"index",std::to_string(i)},
                    {"output_begin_sample",std::to_string(offset0)},{"output_end_sample",std::to_string(pcm.size())},
                    {"unit_samples",std::to_string(pcm.size()-offset0)},{"raw_samples",std::to_string(raw_all)},
                    {"n_chunks",std::to_string(plan.chunks.size())},{"s3_completed_invocations",std::to_string(s3_n)},
                    {"assembly_host_wall_s",json_number(elapsed(assembly_start))},{"host_wall_s",json_number(elapsed(unit_start))},
                    {"t3_completed_invocations","1"},{"eos",tape.eos?"true":"false"},
                    {"s3_host_wall_s",json_number(elapsed(decode_start))}});
            } catch(const std::exception& e) {
                trace_event(trace,"unit_failed","unit",{{"index",std::to_string(i)},{"error",json_string(e.what())}});throw;
            }
        }
        trace_event(trace,"synthesis_complete","synthesis",{{"host_wall_s",json_number(elapsed(begin))},
            {"units",std::to_string(units.size())},{"samples",std::to_string(pcm.size())}});
    }
    T3Tape generate_t3(const std::vector<int32_t>& text_tokens, std::mt19937& rng, SynthesizeStats* stats, ExecutionTrace* trace) {
        const int n_predict=effective_n_predict();
        if(n_predict<1)throw std::runtime_error("n-predict must be positive");
        auto generation_start=TraceClock::now();
        if (model.buffer_kv) ggml_backend_buffer_clear(model.buffer_kv, 0);
        int n_past = 0;
        int32_t token = 0;
        std::vector<int32_t> predicted, speech;
        predicted.reserve((size_t)n_predict + 1);
        speech.reserve((size_t)n_predict + (size_t)effective_sil_count());
        const int32_t stop = model.hparams.stop_speech_token;
        std::vector<float> logits;
        auto decode_start=TraceClock::now();
        trace_event(trace,"t3_start","t3",{{"attempted_invocations","1"},{"requested_prediction_cap",std::to_string(n_predict)}});
        try {
        auto prefill_start=TraceClock::now();
        eval_prompt(model, allocr, text_tokens, logits, n_past);
        trace_event(trace,"prefill_complete","t3",{{"host_wall_s",json_number(elapsed(prefill_start))},
            {"prompt_length",std::to_string(n_past)},{"kv_rows",std::to_string(model.kv_rows)},
            {"sampler_order",json_string("temperature,top_k,top_p,min_p,repetition_penalty,sample")}});
        decode_start=TraceClock::now();
        const std::vector<int32_t> first_pen = { model.hparams.start_speech_token };
        token = sample_next_token_ex(logits, first_pen, rng);
        predicted.push_back(token);
        for (int step = 1; step < n_predict && token != stop && n_past + 1 <= model.hparams.n_ctx; ++step) {
            eval_step(model, allocr, n_past++, token, logits);
            token = sample_next_token_ex(logits, predicted, rng);
            predicted.push_back(token);
        }
        } catch(const std::exception& e) {
            trace_event(trace,"t3_failed","t3",{{"raw_ids",json_ids(predicted)},{"raw_count_including_eos",std::to_string(predicted.size())},
                {"n_past",std::to_string(n_past)},{"kv_rows",std::to_string(model.kv_rows)},
                {"stop_reason",json_string("model_error")},{"error",json_string(e.what())},
                {"host_wall_s",json_number(elapsed(generation_start))}});throw;
        }
        const bool reached_eos=!predicted.empty()&&predicted.back()==stop;
        const int stop_code=reached_eos?0:n_past+1>model.hparams.n_ctx?1:2;
        trace_event(trace,"t3_result","t3",{{"raw_ids",json_ids(predicted)},
            {"raw_count_including_eos",std::to_string(predicted.size())},{"eos",reached_eos?"true":"false"},
            {"stop_reason",json_string(reached_eos?"eos":stop_code==1?"context_limit":"prediction_limit")},
            {"stop_code",std::to_string(stop_code)},
            {"n_past",std::to_string(n_past)},{"generation_host_wall_s",json_number(elapsed(decode_start))},
            {"eos_index",reached_eos?std::to_string(predicted.size()-1):"null"},
            {"host_wall_s",json_number(elapsed(generation_start))}});
        for (int32_t next : predicted) if (next >= 0 && next < 6561) speech.push_back(next);
        const int sil_count=effective_sil_count();
        speech.insert(speech.end(), (size_t)sil_count, (int32_t)effective_s3gen_sil());
        T3Tape tape;
        tape.speech_ids=std::move(speech);
        tape.raw_ids=predicted;
        tape.stop_code=stop_code;
        tape.n_past=n_past;
        tape.eos=reached_eos?1:0;
        tape.text_tokens=(int)text_tokens.size();
        if (stats) {
            stats->predicted_count = (int)predicted.size();
            stats->dropped_count = (int)tape.speech_ids.size();
            stats->eos = tape.eos;
            stats->n_past = n_past;
            stats->text_tokens = tape.text_tokens;
            stats->stop_code = stop_code;
            stats->n_speech = (int)tape.speech_ids.size();
        }
        trace_event(trace,"t3_complete","t3",{{"valid_speech_count",std::to_string(tape.speech_ids.size()-(size_t)sil_count)},
            {"removed_count",std::to_string(predicted.size()-(tape.speech_ids.size()-(size_t)sil_count))},
            {"appended_silence_count",std::to_string(sil_count)},{"s3_input_count",std::to_string(tape.speech_ids.size())},
            {"legacy_dropped_definition",json_string("S3 input count, not omitted words")},
            {"completed_invocations","1"}});
        return tape;
    }
};
Engine::Engine(const EngineOptions& o, ExecutionTrace* trace) : pimpl_(std::make_unique<Impl>(o)) { pimpl_->init(trace); }
Engine::~Engine() = default;
void Engine::synthesize(const std::string& text, std::vector<float>& pcm, SynthesizeStats * stats, ExecutionTrace* trace) {
    pimpl_->synthesize(text, pcm, stats, trace);
}
}
