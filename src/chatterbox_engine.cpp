#include "tts-cpp/chatterbox/engine.h"
#include "tts-cpp/chatterbox/v3.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "chatterbox_t3_internal.h"
#include "mtl_bpe.h"
#include "s3gen_pipeline.h"
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
        if (opts.language_id.empty()) throw std::runtime_error("language");
        std::mt19937 rng(SEED);
        mtl_bpe bpe;
        if (!bpe.load_from_arrays(model.tok_tokens, model.tok_types, model.tok_merges))
            throw std::runtime_error("tokenizer");
        auto ids = bpe.encode(text, opts.language_id);
        std::vector<int32_t> text_tokens;
        text_tokens.reserve(ids.size() + 2);
        text_tokens.push_back(model.hparams.start_text_token);
        text_tokens.insert(text_tokens.end(), ids.begin(), ids.end());
        text_tokens.push_back(model.hparams.stop_text_token);
        int n_past = 0;
        const int32_t stop = model.hparams.stop_speech_token;
        const int32_t sos = model.hparams.start_speech_token;
        std::vector<float> logits;
        std::string dump_dir = opts.t3_gguf_path;
        auto slash = dump_dir.find_last_of("\\/");
        if (slash == std::string::npos) throw std::runtime_error("dump path");
        dump_dir = dump_dir.substr(0, slash);
        eval_prompt(model, allocr, text_tokens, logits, n_past, dump_dir);
        std::vector<int32_t> generated;
        generated.push_back(sos);
        std::vector<int32_t> predicted;
        predicted.reserve((size_t)N_PREDICT);
        for (int i = 0; i < N_PREDICT && n_past + 1 <= model.hparams.n_ctx; ++i) {
            int32_t token = sample_next_token_ex(logits, generated, rng);
            predicted.push_back(token);
            generated.push_back(token);
            if (token == stop) {
                if (!cfg_dump_step(i)) {
                    char label[64];
                    std::snprintf(label, sizeof(label), "gen_step=%d last n_past=%d", i, n_past);
                    dump_last_cfg_meta(dump_dir, label, i);
                }
                break;
            }
            eval_step(model, allocr, n_past++, token, i + 1, logits, dump_dir, i);
        }
        if (predicted.empty() || predicted.back() != stop) throw std::runtime_error("T3 stopped without EOS");
        auto dropped = drop_invalid_tokens(predicted, sos, stop);
        {
            std::string p = opts.t3_gguf_path;
            auto slash2 = p.find_last_of("\\/");
            if (slash2 == std::string::npos) throw std::runtime_error("dump path");
            std::ofstream f(p.substr(0, slash2) + "/v3_t3_dump.txt");
            if (!f) throw std::runtime_error("dump");
            f << "bpe";
            for (int32_t id : ids) f << " " << id;
            f << "\ntext";
            for (int32_t id : text_tokens) f << " " << id;
            f << "\npredicted";
            for (int32_t id : predicted) f << " " << id;
            f << "\ndropped";
            for (int32_t id : dropped) f << " " << id;
            f << "\npredicted_count " << predicted.size();
            f << "\ndropped_count " << dropped.size() << "\n";
            if (!f) throw std::runtime_error("dump write");
        }
        return dropped;
    }
};
Engine::Engine(const EngineOptions& o) : pimpl_(std::make_unique<Impl>(o)) { pimpl_->init(); }
Engine::~Engine() = default;
std::vector<float> Engine::synthesize(const std::string& text) {
    auto tokens = pimpl_->generate_t3(text);
    auto wav = s3gen_synthesize(tokens);
    const int n_tokens = (int)tokens.size();
    const int st_len = std::max(1, n_tokens - 1);
    wav.resize((size_t)st_len * (size_t)kSamplesPerToken);
    return wav;
}
}
