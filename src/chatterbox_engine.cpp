#include "tts-cpp/chatterbox/engine.h"
#include "tts-cpp/chatterbox/log.h"
#include "tts-cpp/chatterbox/nano.h"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "chatterbox_t3_internal.h"
#include "gpt2_bpe.h"
#include "s3gen_pipeline.h"
#include "voice_encoder.h"
#include "voice_features.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
namespace tts_cpp::chatterbox {
using namespace detail;
namespace {
std::uint64_t hash_bytes(const void* data, std::size_t size) {
    const auto* p = static_cast<const std::uint8_t*>(data);
    std::uint64_t h = 1469598103934665603ull;
    for (std::size_t i = 0; i < size; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}
std::string hash_hex(std::uint64_t value) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)value);
    return buf;
}
std::string token_hash(const std::vector<int32_t>& tokens) {
    return hash_hex(hash_bytes(tokens.data(), tokens.size() * sizeof(int32_t)));
}
std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += (char)c; }
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 0x20) {
            char b[8];
            std::snprintf(b, sizeof(b), "\\u%04x", c);
            out += b;
        } else out += (char)c;
    }
    return out;
}
bool consecutive_repeat(const std::vector<int32_t>& generated, int32_t token, int count) {
    if (count < 2 || (int)generated.size() < count - 1) return false;
    for (int i = 1; i < count; ++i) {
        if (generated[generated.size() - (size_t)i] != token) return false;
    }
    return true;
}
}
struct Engine::Impl {
    EngineOptions opts;
    chatterbox_model model{};
    ggml_gallocr_t allocr = nullptr;
    std::vector<float> prompt_feat;
    int prompt_rows = 0;
    std::vector<float> embedding;
    std::vector<int32_t> prompt_token;
    s3gen_piece_state acoustic;
    std::vector<int32_t> speech_history;
    std::string piece_text, piece_stop;
    std::vector<int32_t> piece_text_tokens, piece_speech;
    int piece_t3_ms = 0, piece_s3_ms = 0;
    explicit Impl(const EngineOptions& o) : opts(o) {}
    void reset_acoustics() { acoustic = {}; speech_history.clear(); }
    void init() {
        if (!std::filesystem::exists(opts.t3_gguf_path)) throw std::runtime_error("T3 GGUF missing");
        if (!std::filesystem::exists(opts.s3gen_gguf_path)) throw std::runtime_error("S3Gen GGUF missing");
        if (!validate_reference_audio(opts.reference_audio)) throw std::runtime_error("reference WAV invalid");
        ggml_time_init();
        g_log_verbose = 0;
        ggml_log_set(chatterbox_log_cb, nullptr);
        model.backend = init_backend();
        if (!load_model_gguf(opts.t3_gguf_path, model, N_CTX)) throw std::runtime_error("T3 load failed");
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!allocr) throw std::runtime_error("T3 allocator failed");
        bake_voice();
        s3gen_preload(opts.s3gen_gguf_path, model.backend);
    }
    ~Impl() {
        s3gen_unload();
        if (allocr) ggml_gallocr_free(allocr);
        free_model();
    }
    void free_model() {
        if (model.buffer_w) ggml_backend_buffer_free(model.buffer_w);
        if (model.buffer_kv) ggml_backend_buffer_free(model.buffer_kv);
        if (model.buffer_override) ggml_backend_buffer_free(model.buffer_override);
        if (model.backend) ggml_backend_free(model.backend);
        if (model.ctx_w) ggml_free(model.ctx_w);
        if (model.ctx_kv) ggml_free(model.ctx_kv);
        if (model.ctx_override) ggml_free(model.ctx_override);
        model = {};
    }
    void bake_voice() {
        voice_encoder_weights ve;
        if (!voice_encoder_load(opts.t3_gguf_path, ve)) throw std::runtime_error("VoiceEncoder weights missing");
        std::vector<float> wav, speaker;
        int sr = 0;
        if (!wav_load(opts.reference_audio, wav, sr)) throw std::runtime_error("reference WAV load failed");
        normalise_lufs(wav, sr, -27.0);
        if (sr != 16000) wav = resample_sinc(wav, sr, 16000);
        if (wav.size() > 30u * 16000u) wav.resize(30u * 16000u);
        if (!voice_encoder_embed(wav, ve, model.backend, speaker)) throw std::runtime_error("VoiceEncoder failed");
        if ((int64_t)speaker.size() != ggml_nelements(model.builtin_speaker_emb)) throw std::runtime_error("speaker embedding size mismatch");
        ggml_backend_tensor_set(model.builtin_speaker_emb, speaker.data(), 0, ggml_nbytes(model.builtin_speaker_emb));
        std::vector<int32_t> cond;
        if (!compute_speech_tokens_native(opts.reference_audio, opts.s3gen_gguf_path, model.hparams.cond_prompt_len,
                prompt_token, cond, model.backend)) throw std::runtime_error("S3Tokenizer failed");
        ggml_init_params p = {ggml_tensor_overhead() * 2, nullptr, true};
        model.ctx_override = ggml_init(p);
        auto* t = ggml_new_tensor_1d(model.ctx_override, GGML_TYPE_I32, (int64_t)cond.size());
        model.buffer_override = ggml_backend_alloc_ctx_tensors(model.ctx_override, model.backend);
        ggml_backend_tensor_set(t, cond.data(), 0, cond.size() * sizeof(int32_t));
        model.builtin_cond_prompt_tokens = t;
        model.hparams.cond_prompt_len = (int32_t)cond.size();
        if (!compute_prompt_feat_native(opts.reference_audio, opts.s3gen_gguf_path, prompt_feat, prompt_rows, model.backend)) throw std::runtime_error("prompt feature failed");
        if (!compute_embedding_native(opts.reference_audio, opts.s3gen_gguf_path, embedding, model.backend)) throw std::runtime_error("CAMPPlus failed");
        if (prompt_token.empty() || prompt_feat.empty() || embedding.empty()) throw std::runtime_error("voice conditioning empty");
    }
    std::vector<int32_t> generate_t3(const std::string& text, int session_index, std::uint32_t external_piece) {
        auto synthesis_context = tts_get_context();
        if (session_index >= 0) { synthesis_context.valid = true; synthesis_context.piece = external_piece; }
        tts_context_scope context_scope(synthesis_context);
        const auto started = std::chrono::steady_clock::now();
        std::mt19937 rng(SEED);
        chatterbox_sampling_params sp;
        sp.top_k = TOP_K;
        sp.top_p = TOP_P;
        sp.temp = TEMPERATURE;
        sp.repeat_penalty = REPEAT_PENALTY;

        gpt2_bpe bpe;
        bpe.load_from_arrays(model.tok_tokens, model.tok_merges);
        std::vector<int32_t> text_tokens = bpe.tokenize(gpt2_bpe::punc_norm(text));

        int n_past = 0;
        int32_t token = 0;
        bool repeat_stopped = false;
        std::vector<int32_t> out, tokens;
        out.reserve((size_t)N_PREDICT + 1);
        tokens.reserve((size_t)N_PREDICT);
        const int32_t stop = model.hparams.stop_speech_token;
        const int n_text = (int)text_tokens.size();
        auto hold_eos = [&](std::vector<float> & logits) {
            if ((int)out.size() < n_text * 4)
                logits[(size_t)stop] = -INFINITY;
        };

        std::vector<float> logits;
        if (!eval_prompt(model, allocr, text_tokens, logits, n_past))
            throw std::runtime_error("T3 prompt failed");
        hold_eos(logits);
        token = sample_next_token_ex(logits, out, sp, rng);
        out.push_back(token);
        if (token >= 0 && token < model.hparams.start_speech_token) tokens.push_back(token);

        for (int step = 1; step < N_PREDICT && token != stop && n_past + 1 <= model.hparams.n_ctx; ++step) {
            if (!eval_step(model, allocr, n_past++, token, logits))
                throw std::runtime_error("T3 step failed");
            hold_eos(logits);
            token = sample_next_token_ex(logits, out, sp, rng);
            if (REPEAT_STOP >= 2 && consecutive_repeat(out, token, REPEAT_STOP)) {
                repeat_stopped = true;
                token = stop;
            }
            out.push_back(token);
            if (token >= 0 && token < model.hparams.start_speech_token) tokens.push_back(token);
        }

        if (token != stop) throw std::runtime_error("T3 stopped without EOS");
        piece_text = text;
        piece_text_tokens = std::move(text_tokens);
        piece_speech = tokens;
        piece_stop = repeat_stopped ? "repeat" : "eos";
        piece_t3_ms = (int)(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count() + .5);
        return tokens;
    }
    void emit_piece_ledger(std::uint32_t response, std::uint32_t piece, const std::vector<int32_t>& neu) {
        const auto ctx = tts_get_context();
        tts_jsonl(std::string("{\"response\":") + std::to_string(ctx.valid ? ctx.response : response) +
            ",\"piece\":" + std::to_string(piece) +
            ",\"text\":\"" + json_escape(piece_text) + "\"" +
            ",\"text_sha\":\"" + hash_hex(hash_bytes(piece_text.data(), piece_text.size())) + "\"" +
            ",\"n_text_tok\":" + std::to_string(piece_text_tokens.size()) +
            ",\"n_speech_tok\":" + std::to_string(piece_speech.size()) +
            ",\"speech_hash\":\"" + token_hash(piece_speech) + "\"" +
            ",\"stop\":\"" + piece_stop + "\"" +
            ",\"t3_ms\":" + std::to_string(piece_t3_ms) +
            ",\"s3_ms\":" + std::to_string(piece_s3_ms) +
            ",\"ms\":" + std::to_string(piece_t3_ms + piece_s3_ms) +
            ",\"s3_history\":" + std::to_string(acoustic.history_tokens) +
            ",\"s3_new\":" + std::to_string(neu.size()) +
            ",\"emitted_samples\":" + std::to_string(acoustic.emitted) + "}");
    }
    void run_s3(const std::vector<int32_t>& tokens, int session_index, std::uint32_t external_piece, bool last_piece, const PieceCallback& cb) {
        auto synthesis_context = tts_get_context();
        if (session_index >= 0) { synthesis_context.valid = true; synthesis_context.piece = external_piece; }
        tts_context_scope context_scope(synthesis_context);
        if (tokens.empty()) throw std::runtime_error("S3Gen speech tokens empty");
        acoustic.encoder_ms = acoustic.cfm_ms = acoustic.f0_ms = acoustic.stft_ms = acoustic.hift_ms = acoustic.pipeline_ms = 0;
        acoustic.samples = acoustic.prompt_tokens = acoustic.speech_tokens = 0;
        acoustic.token_end = (int)speech_history.size();
        std::vector<int32_t> window;
        window.reserve(speech_history.size() + tokens.size());
        window.insert(window.end(), speech_history.begin(), speech_history.end());
        window.insert(window.end(), tokens.begin(), tokens.end());
        s3gen_synthesize_opts s;
        s.s3gen_gguf_path = opts.s3gen_gguf_path;
        s.prompt_feat = prompt_feat;
        s.prompt_rows = prompt_rows;
        s.embedding = embedding;
        s.prompt_token = prompt_token;
        s.state = &acoustic;
        s.token_start = 0;
        s.token_end = (int)window.size();
        s.last_piece = last_piece;
        s.first_piece = (session_index <= 0);
        s.chunk_id = 0;
        std::vector<float> pcm;
        s.pcm_out = &pcm;
        const auto s3_started = std::chrono::steady_clock::now();
        s3gen_synthesize(window, s);
        piece_s3_ms = (int)(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - s3_started).count() + .5);
        if (cb) cb(session_index, pcm.data(), pcm.size(), 0, true);
        if (session_index >= 0) {
            const auto ctx = tts_get_context();
            emit_piece_ledger(ctx.response, external_piece, tokens);
        }
        if ((int)window.size() > kSpeechHistoryTokens) {
            speech_history.assign(window.end() - kSpeechHistoryTokens, window.end());
        } else {
            speech_history = window;
        }
    }
};
Engine::Engine(const EngineOptions& o) : pimpl_(std::make_unique<Impl>(o)) { pimpl_->init(); }
Engine::~Engine() = default;
Engine::Engine(Engine&&) noexcept = default;
Engine& Engine::operator=(Engine&&) noexcept = default;
void Engine::synthesize_pieces_streaming(const std::vector<SynthesisPiece>& pieces, const PieceCallback& cb) {
    pimpl_->reset_acoustics();
    for (std::size_t index = 0; index < pieces.size(); ++index) {
        const auto& piece = pieces[index];
        if (piece.text.empty()) throw std::runtime_error("empty synthesis piece");
        auto tokens = pimpl_->generate_t3(piece.text, (int)index, piece.id);
        pimpl_->run_s3(tokens, (int)index, piece.id, index + 1 == pieces.size(),
            [&](int, const float* pcm, std::size_t n, int chunk, bool final) {
                if (cb) cb((int)index, pcm, n, chunk, final);
            });
    }
}
void Engine::warm_up() {
    pimpl_->reset_acoustics();
    std::size_t samples = 0;
    auto tokens = pimpl_->generate_t3("Warm up.", -1, 0);
    pimpl_->run_s3(tokens, -1, 0, true, [&](int, const float*, std::size_t n, int, bool) { samples += n; });
    if (!samples) throw std::runtime_error("warm-up produced no PCM");
    if (pimpl_->model.buffer_kv) ggml_backend_buffer_clear(pimpl_->model.buffer_kv, 0);
    pimpl_->reset_acoustics();
}
}
