#include "tts-cpp/chatterbox/engine.h"
#include "tts-cpp/chatterbox/log.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "chatterbox_t3_internal.h"
#include "gpt2_bpe.h"
#ifdef TTS_CPP_MTL
#include "mtl_tokenizer.h"
#endif
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
int threads(int n) {
    if (n > 0) return n;
    const int hw = (int)std::thread::hardware_concurrency();
    return hw > 0 ? std::min(hw, 4) : 4;
}
void join(std::thread& t) { if (t.joinable()) t.join(); }
void ledger_append(const std::string& dir, const char* name, const std::string& line) {
    if (dir.empty()) return;
    std::ofstream out(std::filesystem::path(dir) / name, std::ios::app);
    if (!out) throw std::runtime_error(std::string("cannot append ledger ") + name);
    out << line << '\n';
}
void ledger_bin_append(const std::string& dir, const char* name, const std::vector<int32_t>& tokens) {
    if (dir.empty()) return;
    std::ofstream out(std::filesystem::path(dir) / name, std::ios::binary | std::ios::app);
    if (!out) throw std::runtime_error(std::string("cannot append ledger ") + name);
    const std::uint32_t n = (std::uint32_t)tokens.size();
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    if (n) out.write(reinterpret_cast<const char*>(tokens.data()), n * sizeof(int32_t));
}
}
struct Engine::Impl {
    EngineOptions opts;
    chatterbox_model model{};
    ggml_gallocr_t allocr = nullptr;
    std::thread preload;
    std::vector<float> prompt_feat;
    int prompt_rows = 0;
    std::vector<float> embedding;
    std::vector<int32_t> prompt_token;
#ifdef TTS_CPP_MTL
    std::unique_ptr<mtl_tokenizer> mtl_tok;
#endif
    s3gen_piece_state acoustic;
    std::vector<int32_t> speech_history;
    std::string piece_text, piece_stop;
    std::vector<int32_t> piece_text_tokens, piece_speech;
    int piece_ms = 0;
    std::uint32_t speech_bin_offset = 0;
    explicit Impl(const EngineOptions& o) : opts(o) {}
    void reset_acoustics() { acoustic = {}; speech_history.clear(); }
    void write_meta() {
        if (opts.audit_dir.empty()) return;
        std::filesystem::create_directories(opts.audit_dir);
        std::ofstream out(std::filesystem::path(opts.audit_dir) / "00-meta.json", std::ios::trunc);
        if (!out) throw std::runtime_error("cannot write 00-meta.json");
        out << "{\"run\":\"" << json_escape(tts_run_identity())
            << "\",\"seed\":" << opts.seed
            << ",\"min_p\":" << opts.min_p
            << ",\"repeat_penalty\":" << opts.repeat_penalty
            << ",\"repeat_last_n\":" << REPEAT_PENALTY_LAST_N
            << ",\"repeat_stop\":" << REPEAT_STOP_CONSECUTIVE
            << ",\"cfm_steps\":" << opts.cfm_steps
            << ",\"n_ctx\":" << opts.n_ctx
            << ",\"max_tokens\":" << opts.n_predict
            << ",\"temperature\":" << opts.temperature
            << ",\"top_k\":" << opts.top_k
            << ",\"top_p\":" << opts.top_p
            << ",\"audit_tensors\":" << (opts.audit_tensors ? "true" : "false")
            << "}\n";
    }
    void init() {
        if (!std::filesystem::exists(opts.t3_gguf_path)) throw std::runtime_error("T3 GGUF missing");
        if (!std::filesystem::exists(opts.s3gen_gguf_path)) throw std::runtime_error("S3Gen GGUF missing");
        if (!validate_reference_audio(opts.reference_audio)) throw std::runtime_error("reference WAV invalid");
        ggml_time_init();
        g_log_verbose = 0;
        ggml_log_set(chatterbox_log_cb, nullptr);
        if (!load_model_gguf(opts.t3_gguf_path, model, opts.n_ctx, opts.n_gpu_layers)) throw std::runtime_error("T3 load failed");
        if (model.hparams.variant != CHBX_VARIANT_TURBO && model.hparams.variant != CHBX_VARIANT_MTL)
            throw std::runtime_error("unsupported T3 variant");
#ifndef TTS_CPP_MTL
        if (model.hparams.variant == CHBX_VARIANT_MTL) throw std::runtime_error("multilingual T3 was not compiled");
#else
        if (model.hparams.variant == CHBX_VARIANT_MTL) {
            mtl_tok = std::make_unique<mtl_tokenizer>();
            if (model.mtl_tokenizer_json.empty() || !mtl_tok->load_from_json(model.mtl_tokenizer_json))
                throw std::runtime_error("MTL tokenizer missing");
            if (opts.language == "zh" && (model.mtl_cangjie_json.empty() || !mtl_tok->load_cangjie_json(model.mtl_cangjie_json)))
                throw std::runtime_error("MTL Cangjie mapping missing or invalid");
        }
#endif
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!allocr) throw std::runtime_error("T3 allocator failed");
        preload = std::thread([this] { s3gen_preload(opts.s3gen_gguf_path, opts.n_gpu_layers, opts.fastconv); });
        bake_voice();
        join(preload);
        write_meta();
    }
    ~Impl() {
        join(preload);
        s3gen_unload();
        if (allocr) ggml_gallocr_free(allocr);
        free_model();
    }
    void free_model() {
        if (model.buffer_stack || model.ctx_stack) t3_stack_unregister(model.buffer_stack, model.ctx_stack);
        if (model.buffer_w) ggml_backend_buffer_free(model.buffer_w);
        if (model.buffer_kv) ggml_backend_buffer_free(model.buffer_kv);
        if (model.buffer_stack) ggml_backend_buffer_free(model.buffer_stack);
        if (model.buffer_override) ggml_backend_buffer_free(model.buffer_override);
        if (model.backend) ggml_backend_free(model.backend);
        if (model.ctx_w) ggml_free(model.ctx_w);
        if (model.ctx_kv) ggml_free(model.ctx_kv);
        if (model.ctx_stack) ggml_free(model.ctx_stack);
        if (model.ctx_override) ggml_free(model.ctx_override);
        model = {};
    }
    void bake_voice() {
        const int n_threads = threads(opts.n_threads);
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
                prompt_token, cond, n_threads, model.backend, false)) throw std::runtime_error("S3Tokenizer failed");
        if ((int64_t)cond.size() == ggml_nelements(model.builtin_cond_prompt_tokens)) {
            ggml_backend_tensor_set(model.builtin_cond_prompt_tokens, cond.data(), 0, ggml_nbytes(model.builtin_cond_prompt_tokens));
        } else {
            ggml_init_params p = {ggml_tensor_overhead() * 2, nullptr, true};
            model.ctx_override = ggml_init(p);
            if (!model.ctx_override) throw std::runtime_error("conditioning context failed");
            auto* t = ggml_new_tensor_1d(model.ctx_override, GGML_TYPE_I32, (int64_t)cond.size());
            model.buffer_override = ggml_backend_alloc_ctx_tensors(model.ctx_override, model.backend);
            if (!model.buffer_override) throw std::runtime_error("conditioning buffer failed");
            ggml_backend_tensor_set(t, cond.data(), 0, cond.size() * sizeof(int32_t));
            model.builtin_cond_prompt_tokens = t;
            model.hparams.cond_prompt_len = (int32_t)cond.size();
        }
        if (!compute_prompt_feat_native(opts.reference_audio, opts.s3gen_gguf_path, prompt_feat, prompt_rows, false)) throw std::runtime_error("prompt feature failed");
        if (!compute_embedding_native(opts.reference_audio, opts.s3gen_gguf_path, embedding, false)) throw std::runtime_error("CAMPPlus failed");
        if (prompt_token.empty() || prompt_feat.empty() || embedding.empty()) throw std::runtime_error("voice conditioning empty");
    }
    std::vector<int32_t> generate_t3(const std::string& text, int session_index, std::uint32_t external_piece) {
        auto synthesis_context = tts_get_context();
        if (session_index >= 0) { synthesis_context.valid = true; synthesis_context.piece = external_piece; }
        tts_context_scope context_scope(synthesis_context);
        const auto started = std::chrono::steady_clock::now();
        const int n_threads = threads(opts.n_threads);
        std::mt19937 rng(opts.seed);
        chatterbox_sampling_params sp;
        sp.top_k = opts.top_k;
        sp.top_p = opts.top_p;
        sp.min_p = opts.min_p;
        sp.temp = opts.temperature;
        sp.repeat_penalty = opts.repeat_penalty;
        sp.cfg_weight = opts.cfg_weight;

        std::vector<int32_t> text_tokens;
#ifdef TTS_CPP_MTL
        if (model.hparams.variant == CHBX_VARIANT_MTL) {
            if (!mtl_tok) throw std::runtime_error("MTL tokenizer missing");
            text_tokens = mtl_tok->encode(text, opts.language);
            text_tokens.insert(text_tokens.begin(), model.hparams.start_text_token);
            text_tokens.push_back(model.hparams.stop_text_token);
        } else
#endif
        {
            if (model.tok_tokens.empty()) throw std::runtime_error("Turbo tokenizer missing");
            gpt2_bpe bpe;
            bpe.load_from_arrays(model.tok_tokens, model.tok_merges);
            text_tokens = bpe.tokenize(gpt2_bpe::punc_norm(text));
        }
        if (text_tokens.empty()) throw std::runtime_error("empty T3 text tokens");

        int n_past = 0, speech_pos = 1;
        int32_t token = 0, pending_mtl = -1;
        bool repeat_stopped = false;
        std::vector<int32_t> out, tokens;
        out.reserve((size_t)opts.n_predict + 1);
        tokens.reserve((size_t)opts.n_predict);
        auto publish = [&](int32_t value) {
            if (value < 0 || value >= model.hparams.start_speech_token || value == model.hparams.stop_speech_token) return;
#ifdef TTS_CPP_MTL
            if (model.hparams.variant == CHBX_VARIANT_MTL) {
                if (pending_mtl >= 0) tokens.push_back(pending_mtl);
                pending_mtl = value;
                return;
            }
#endif
            tokens.push_back(value);
        };

#ifdef TTS_CPP_MTL
        if (model.hparams.variant == CHBX_VARIANT_MTL) {
            std::vector<float> logits_c, logits_u;
            if (!eval_prompt_mtl(model, allocr, n_threads, text_tokens, opts.exaggeration, logits_c, logits_u, n_past))
                throw std::runtime_error("MTL prompt failed");
            token = sample_next_token_mtl(logits_c, logits_u, out, sp, rng, model.hparams.stop_speech_token);
        } else
#endif
        {
            std::vector<float> logits;
            if (!eval_prompt(model, allocr, n_threads, text_tokens, logits, n_past)) throw std::runtime_error("Turbo prompt failed");
            token = sample_next_token_ex(logits, out, sp, rng);
        }
        out.push_back(token);
        publish(token);

        for (int i = 0; i < opts.n_predict && token != model.hparams.stop_speech_token && n_past + 1 <= model.hparams.n_ctx; ++i) {
#ifdef TTS_CPP_MTL
            if (model.hparams.variant == CHBX_VARIANT_MTL) {
                std::vector<float> logits_c, logits_u;
                if (!eval_step_mtl(model, allocr, n_threads, n_past++, speech_pos++, token, logits_c, logits_u))
                    throw std::runtime_error("MTL step failed");
                token = sample_next_token_mtl(logits_c, logits_u, out, sp, rng, model.hparams.stop_speech_token);
            } else
#endif
            {
                std::vector<float> logits;
                if (!eval_step(model, allocr, n_threads, n_past++, token, logits)) throw std::runtime_error("Turbo step failed");
                token = sample_next_token_ex(logits, out, sp, rng);
            }
            if (consecutive_repeat(out, token, REPEAT_STOP_CONSECUTIVE)) {
                repeat_stopped = true;
                token = model.hparams.stop_speech_token;
            }
            out.push_back(token);
            publish(token);
        }

        if (token != model.hparams.stop_speech_token) throw std::runtime_error("T3 stopped without EOS");
#ifdef TTS_CPP_MTL
        if (tokens.empty() && pending_mtl >= 0) tokens.push_back(pending_mtl);
#endif
        piece_text = text;
        piece_text_tokens = std::move(text_tokens);
        piece_speech = tokens;
        piece_stop = repeat_stopped ? "repeat" : "eos";
        piece_ms = (int)(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count() + .5);
        (void)speech_pos;
        (void)external_piece;
        return tokens;
    }
    void emit_piece_ledger(std::uint32_t response, std::uint32_t piece, const std::vector<int32_t>& window, const std::vector<int32_t>& neu) {
        const auto ctx = tts_get_context();
        const std::string line =
            std::string("{\"response\":") + std::to_string(ctx.valid ? ctx.response : response) +
            ",\"piece\":" + std::to_string(piece) +
            ",\"text\":\"" + json_escape(piece_text) + "\"" +
            ",\"text_sha\":\"" + hash_hex(hash_bytes(piece_text.data(), piece_text.size())) + "\"" +
            ",\"n_text_tok\":" + std::to_string(piece_text_tokens.size()) +
            ",\"n_speech_tok\":" + std::to_string(piece_speech.size()) +
            ",\"speech_hash\":\"" + token_hash(piece_speech) + "\"" +
            ",\"stop\":\"" + piece_stop + "\"" +
            ",\"ms\":" + std::to_string(piece_ms) +
            ",\"s3_history\":" + std::to_string(acoustic.history_tokens) +
            ",\"s3_new\":" + std::to_string(neu.size()) +
            ",\"emitted_samples\":" + std::to_string(acoustic.emitted) + "}";
        tts_jsonl(line);
        if (opts.audit_dir.empty()) return;
        ledger_append(opts.audit_dir, "01-pieces.jsonl",
            "{\"piece\":" + std::to_string(piece) + ",\"response\":" + std::to_string(response) +
            ",\"text\":\"" + json_escape(piece_text) + "\",\"chars\":" + std::to_string(piece_text.size()) + "}");
        {
            std::string ids = "{\"piece\":" + std::to_string(piece) + ",\"ids\":[";
            for (size_t i = 0; i < piece_text_tokens.size(); ++i) {
                if (i) ids += ',';
                ids += std::to_string(piece_text_tokens[i]);
            }
            ids += "]}";
            ledger_append(opts.audit_dir, "02-text-tokens.jsonl", ids);
        }
        const std::uint32_t before = speech_bin_offset;
        ledger_bin_append(opts.audit_dir, "03-speech-tokens.bin", piece_speech);
        speech_bin_offset += 1 + (std::uint32_t)piece_speech.size();
        ledger_append(opts.audit_dir, "03-index.jsonl",
            "{\"piece\":" + std::to_string(piece) + ",\"n\":" + std::to_string(piece_speech.size()) +
            ",\"u32_offset\":" + std::to_string(before) + "}");
        for (size_t i = 0; i < piece_speech.size(); ++i) {
            ledger_append(opts.audit_dir, "04-sample.jsonl",
                "{\"piece\":" + std::to_string(piece) + ",\"i\":" + std::to_string(i) +
                ",\"id\":" + std::to_string(piece_speech[i]) + "}");
        }
        ledger_append(opts.audit_dir, "05-s3-window.jsonl",
            "{\"piece\":" + std::to_string(piece) +
            ",\"history_tokens\":" + std::to_string(acoustic.history_tokens) +
            ",\"history_hash\":\"" + token_hash(std::vector<int32_t>(window.begin(), window.begin() + acoustic.history_tokens)) + "\"" +
            ",\"new_tokens\":" + std::to_string(neu.size()) +
            ",\"new_hash\":\"" + token_hash(neu) + "\"" +
            ",\"emit_begin\":" + std::to_string(acoustic.emit_begin) +
            ",\"emit_end\":" + std::to_string(acoustic.emit_end) +
            ",\"pending_in\":" + std::to_string(acoustic.pending_in) +
            ",\"emitted\":" + std::to_string(acoustic.emitted) + "}");
        for (size_t i = 0; i < neu.size(); ++i) {
            const int window_i = acoustic.history_tokens + (int)i;
            const int raw0 = window_i * kSamplesPerToken;
            const int local0 = raw0 - (int)acoustic.emit_begin;
            ledger_append(opts.audit_dir, "06-pcm-map.jsonl",
                "{\"piece\":" + std::to_string(piece) + ",\"token_i\":" + std::to_string(i) +
                ",\"window_i\":" + std::to_string(window_i) +
                ",\"sample_start\":" + std::to_string(local0) +
                ",\"sample_end\":" + std::to_string(local0 + kSamplesPerToken) + "}");
        }
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
        s.seed = opts.seed;
        s.n_threads = threads(opts.n_threads);
        s.n_gpu_layers = opts.n_gpu_layers;
        s.fastconv = opts.fastconv;
        s.cfm_steps = opts.cfm_steps;
        s.prompt_feat = prompt_feat;
        s.prompt_rows = prompt_rows;
        s.embedding = embedding;
        s.prompt_token = prompt_token;
        s.state = &acoustic;
        s.token_start = 0;
        s.token_end = (int)window.size();
        s.final = true;
        s.last_piece = last_piece;
        s.first_piece = (session_index <= 0);
        s.chunk_id = 0;
        s.audit_tensors = opts.audit_tensors;
        if (opts.audit_tensors && !opts.audit_dir.empty()) {
            const auto ctx = tts_get_context();
            s.audit_prefix = opts.audit_dir + "/r" + std::to_string(ctx.response) +
                "_p" + std::to_string(external_piece);
        }
        std::vector<float> pcm;
        s.pcm_out = &pcm;
        s3gen_synthesize(window, s);
        if (cb) cb(session_index, pcm.data(), pcm.size(), 0, true);
        if (session_index >= 0) {
            const auto ctx = tts_get_context();
            emit_piece_ledger(ctx.response, external_piece, window, tokens);
        }
        if ((int)window.size() > kSpeechHistoryTokens) {
            speech_history.assign(window.end() - kSpeechHistoryTokens, window.end());
        } else {
            speech_history = window;
        }
    }
    void piece_streaming(const std::string& text, int index, const PieceCallback& cb) {
        if (text.empty()) return;
        auto tokens = generate_t3(text, index, 0);
        run_s3(tokens, index, 0, true, [&](int, const float* pcm, std::size_t n, int chunk, bool final) {
            if (cb) cb(0, pcm, n, chunk, final);
        });
    }
};
Engine::Engine(const EngineOptions& o) : pimpl_(std::make_unique<Impl>(o)) { pimpl_->init(); }
Engine::~Engine() = default;
Engine::Engine(Engine&&) noexcept = default;
Engine& Engine::operator=(Engine&&) noexcept = default;
void Engine::synthesize_pieces_streaming(const std::vector<SynthesisPiece>& pieces, const PieceCallback& cb) {
    pimpl_->reset_acoustics();
    pimpl_->speech_bin_offset = 0;
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
    pimpl_->piece_streaming("Warm up.", -1, [&](int, const float*, std::size_t n, int, bool) { samples += n; });
    if (!samples) throw std::runtime_error("warm-up produced no PCM");
    if (pimpl_->model.buffer_kv) ggml_backend_buffer_clear(pimpl_->model.buffer_kv, 0);
    pimpl_->reset_acoustics();
}
}
