#include "bake_native.h"
#include "voice_encoder.h"
#include "voice_features.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
struct repl {
    std::string name;
    ggml_type type = GGML_TYPE_F32;
    std::vector<int64_t> ne;
    const void * data = nullptr;
    size_t bytes = 0;
};

uint32_t require_u32(const gguf_context * g, const char * key) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) throw std::runtime_error(std::string("missing GGUF key: ") + key);
    return gguf_get_val_u32(g, id);
}

void replace_file(const std::string & tmp, const std::string & dest) {
    std::remove(dest.c_str());
    if (std::rename(tmp.c_str(), dest.c_str()) != 0)
        throw std::runtime_error("GGUF replace failed: " + dest);
}

void rewrite_gguf(const std::string & path, const std::vector<repl> & reps,
                  const std::vector<std::pair<const char *, uint32_t>> & kv_u32) {
    ggml_context * src_ctx = nullptr;
    gguf_init_params gp = { false, &src_ctx };
    gguf_context * gin = gguf_init_from_file(path.c_str(), gp);
    if (!gin || !src_ctx) throw std::runtime_error("GGUF open failed: " + path);
    std::unordered_map<std::string, const repl *> by_name;
    for (const auto & r : reps) by_name[r.name] = &r;
    ggml_init_params rp = { ggml_tensor_overhead() * (reps.size() + 4), nullptr, true };
    ggml_context * rctx = ggml_init(rp);
    if (!rctx) throw std::runtime_error("ggml_init");
    gguf_context * gout = gguf_init_empty();
    gguf_set_kv(gout, gin);
    for (const auto & kv : kv_u32) gguf_set_val_u32(gout, kv.first, kv.second);
    const int64_t n = gguf_get_n_tensors(gin);
    for (int64_t i = 0; i < n; ++i) {
        const char * name = gguf_get_tensor_name(gin, i);
        auto it = by_name.find(name);
        if (it == by_name.end()) {
            ggml_tensor * src = ggml_get_tensor(src_ctx, name);
            if (!src) throw std::runtime_error(std::string("tensor missing in ctx: ") + name);
            gguf_add_tensor(gout, src);
            continue;
        }
        const repl & r = *it->second;
        int64_t ne[4] = { 1, 1, 1, 1 };
        for (size_t d = 0; d < r.ne.size() && d < 4; ++d) ne[d] = r.ne[d];
        ggml_tensor * t = ggml_new_tensor(rctx, r.type, (int)r.ne.size(), ne);
        ggml_set_name(t, name);
        gguf_add_tensor(gout, t);
        gguf_set_tensor_data(gout, name, r.data);
        if (ggml_nbytes(t) != r.bytes) throw std::runtime_error(std::string("tensor size mismatch: ") + name);
    }
    const std::string tmp = path + ".tmp";
    if (!gguf_write_to_file(gout, tmp.c_str(), false)) throw std::runtime_error("GGUF write failed: " + path);
    gguf_free(gout);
    gguf_free(gin);
    ggml_free(rctx);
    ggml_free(src_ctx);
    replace_file(tmp, path);
}
}

int main(int, char ** argv) {
    const char * t3 = argv[1];
    const char * s3 = argv[2];
    const char * ref = argv[3];
    ggml_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    ggml_backend_t backend = ggml_backend_vk_init(0);
    if (!backend) throw std::runtime_error("Vulkan backend init failed");

    gguf_init_params meta = { true, nullptr };
    gguf_context * t3meta = gguf_init_from_file(t3, meta);
    if (!t3meta) throw std::runtime_error("T3 GGUF open failed");
    uint32_t max_cond = 0;
    const int64_t max_id = gguf_find_key(t3meta, "chatterbox.cond_prompt_max");
    if (max_id >= 0) max_cond = gguf_get_val_u32(t3meta, max_id);
    else max_cond = require_u32(t3meta, "chatterbox.cond_prompt_length");
    gguf_free(t3meta);

    voice_encoder_weights ve;
    if (!voice_encoder_load(t3, ve)) throw std::runtime_error("VE load");
    std::vector<float> wav, speaker;
    int sr = 0;
    wav_load(ref, wav, sr);
    normalise_lufs(wav, sr, -27.0);
    if (sr != 16000) wav = resample_sinc(wav, sr, 16000);
    if (wav.empty()) throw std::runtime_error("reference wav");
    if (!voice_encoder_embed(wav, ve, backend, speaker)) throw std::runtime_error("VE embed");
    if (speaker.size() != 256) throw std::runtime_error("speaker_emb size");

    std::vector<int32_t> prompt_token, cond;
    tts_cpp::chatterbox::detail::compute_speech_tokens_native(ref, s3, (int)max_cond, prompt_token, cond, backend);
    std::vector<float> prompt_feat, embedding;
    int prompt_rows = 0;
    tts_cpp::chatterbox::detail::compute_prompt_feat_native(ref, s3, prompt_feat, prompt_rows, backend);
    tts_cpp::chatterbox::detail::compute_embedding_native(ref, s3, embedding, backend);
    if (prompt_rows <= 0 || (int)prompt_feat.size() != prompt_rows * 80) throw std::runtime_error("prompt_feat");
    if (embedding.size() != 192) throw std::runtime_error("embedding size");
    if (cond.empty() || prompt_token.empty()) throw std::runtime_error("speech tokens");

    std::vector<int64_t> speaker_ne = { 256, 1 };
    std::vector<int64_t> cond_ne = { (int64_t)cond.size() };
    rewrite_gguf(t3, {
        { "chatterbox/builtin/speaker_emb", GGML_TYPE_F32, speaker_ne, speaker.data(), speaker.size() * sizeof(float) },
        { "chatterbox/builtin/cond_prompt_speech_tokens", GGML_TYPE_I32, cond_ne, cond.data(), cond.size() * sizeof(int32_t) },
    }, {
        { "chatterbox.cond_prompt_max", max_cond },
        { "chatterbox.cond_prompt_length", (uint32_t)cond.size() },
    });

    std::vector<int64_t> tok_ne = { (int64_t)prompt_token.size() };
    std::vector<int64_t> feat_ne = { 80, (int64_t)prompt_rows };
    std::vector<int64_t> emb_ne = { (int64_t)embedding.size() };
    rewrite_gguf(s3, {
        { "s3gen/builtin/prompt_token", GGML_TYPE_I32, tok_ne, prompt_token.data(), prompt_token.size() * sizeof(int32_t) },
        { "s3gen/builtin/prompt_feat", GGML_TYPE_F32, feat_ne, prompt_feat.data(), prompt_feat.size() * sizeof(float) },
        { "s3gen/builtin/embedding", GGML_TYPE_F32, emb_ne, embedding.data(), embedding.size() * sizeof(float) },
    }, {
        { "s3gen.builtin.prompt_token_len", (uint32_t)prompt_token.size() },
        { "s3gen.builtin.prompt_feat_frames", (uint32_t)prompt_rows },
    });

    ggml_backend_free(backend);
}
