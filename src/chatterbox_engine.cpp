#include "tts-cpp/chatterbox/engine.h"
#include "tts-cpp/chatterbox/log.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "chatterbox_t3_internal.h"
#include "gpt2_bpe.h"
#include "mtl_tokenizer.h"
#include "t3_mtl.h"
#include "s3gen_pipeline.h"
#include "s3tokenizer.h"
#include "voice_encoder.h"
#include "voice_features.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#ifdef GGML_USE_VULKAN
#ifdef _WIN32
extern "C" __declspec(dllimport) void ggml_vk_overlap_counters(ggml_backend_t, unsigned long long *, unsigned long long *, unsigned long long *, int);
#else
extern "C" void ggml_vk_overlap_counters(ggml_backend_t, unsigned long long *, unsigned long long *, unsigned long long *, int);
#endif
#endif
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
std::string token_csv(const std::vector<int32_t>& tokens) {
    std::string out;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (i) out += ',';
        out += std::to_string(tokens[i]);
    }
    return out;
}
std::string string_csv(const std::vector<std::string>& values) {
    std::string out;
    for (const auto& value : values) {
        if (!out.empty()) out += ',';
        out += value;
    }
    return out.empty() ? "-" : out;
}
std::string float_hash(const std::vector<float>& values) {
    return hash_hex(hash_bytes(values.data(), values.size() * sizeof(float)));
}
std::size_t token_edit_distance(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    std::vector<std::size_t> row(b.size() + 1);
    for (std::size_t j = 0; j <= b.size(); ++j) row[j] = j;
    for (std::size_t i = 1; i <= a.size(); ++i) {
        std::size_t previous = row[0];
        row[0] = i;
        for (std::size_t j = 1; j <= b.size(); ++j) {
            const std::size_t old = row[j];
            row[j] = std::min({row[j] + 1, row[j - 1] + 1,
                               previous + (a[i - 1] == b[j - 1] ? 0u : 1u)});
            previous = old;
        }
    }
    return row.back();
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
#ifdef GGML_USE_VULKAN
void vk_overlap_reset(ggml_backend_t b) {
    if (b) ggml_vk_overlap_counters(b, nullptr, nullptr, nullptr, 1);
}
std::string vk_overlap_fields(ggml_backend_t b) {
    unsigned long long wait_us = 0, submit_n = 0, barrier_n = 0;
    if (b) ggml_vk_overlap_counters(b, &wait_us, &submit_n, &barrier_n, 0);
    return std::string(" wait_us=") + std::to_string(wait_us)
        + " submit_n=" + std::to_string(submit_n)
        + " barrier_n=" + std::to_string(barrier_n);
}
#else
void vk_overlap_reset(ggml_backend_t) {}
std::string vk_overlap_fields(ggml_backend_t) { return {}; }
#endif
std::string s3_overlap_fields() {
    unsigned long long wait_us = 0, submit_n = 0, barrier_n = 0;
    s3gen_vk_overlap_counters(&wait_us, &submit_n, &barrier_n, 0);
    return std::string(" wait_us=") + std::to_string(wait_us)
        + " submit_n=" + std::to_string(submit_n)
        + " barrier_n=" + std::to_string(barrier_n);
}
void s3_overlap_reset() {
    s3gen_vk_overlap_counters(nullptr, nullptr, nullptr, 1);
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
    std::unique_ptr<mtl_tokenizer> mtl_tok;
    std::unique_ptr<s3tokv2_weights> audit_tok;
    s3gen_piece_state acoustic;
    std::vector<int32_t> speech_history;
    explicit Impl(const EngineOptions& o) : opts(o) {}
    void reset_acoustics() { acoustic = {}; speech_history.clear(); tts_emit("s3.reset"); }
    void init() {
        if (!std::filesystem::exists(opts.t3_gguf_path)) throw std::runtime_error("T3 GGUF missing");
        if (!std::filesystem::exists(opts.s3gen_gguf_path)) throw std::runtime_error("S3Gen GGUF missing");
        if (!validate_reference_audio(opts.reference_audio)) throw std::runtime_error("reference WAV invalid");
        ggml_time_init();
        g_log_verbose = 0;
        ggml_log_set(chatterbox_log_cb, nullptr);
        tts_emit("t3.model.load.begin", " path=" + opts.t3_gguf_path);
        const auto t3_load_started = std::chrono::steady_clock::now();
        if (!load_model_gguf(opts.t3_gguf_path, model, opts.n_ctx, opts.n_gpu_layers)) throw std::runtime_error("T3 load failed");
        tts_emit("t3.model.load.completed", std::string(" ms=") + std::to_string((int)(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t3_load_started).count() + .5))
            + " weights_bytes=" + std::to_string(model.buffer_w ? ggml_backend_buffer_get_size(model.buffer_w) : 0)
            + " kv_bytes=" + std::to_string(model.buffer_kv ? ggml_backend_buffer_get_size(model.buffer_kv) : 0));
        if (model.hparams.variant != CHBX_VARIANT_TURBO && model.hparams.variant != CHBX_VARIANT_MTL) throw std::runtime_error("unsupported T3 variant");
        if (model.hparams.variant == CHBX_VARIANT_MTL) {
            mtl_tok = std::make_unique<mtl_tokenizer>();
            if (model.mtl_tokenizer_json.empty() || !mtl_tok->load_from_json(model.mtl_tokenizer_json)) throw std::runtime_error("MTL tokenizer missing");
            if (opts.language == "zh" && (model.mtl_cangjie_json.empty() || !mtl_tok->load_cangjie_json(model.mtl_cangjie_json))) throw std::runtime_error("MTL Cangjie mapping missing or invalid");
        }
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!allocr) throw std::runtime_error("T3 allocator failed");
        tts_emit("t3.workspace.ready", " ok");
        preload = std::thread([this] { s3gen_preload(opts.s3gen_gguf_path, opts.n_gpu_layers, opts.fastconv); });
        bake_voice();
        join(preload);
        if (!opts.audit_dir.empty()) {
            std::filesystem::create_directories(opts.audit_dir);
            audit_tok = std::make_unique<s3tokv2_weights>();
            if (!s3tokv2_load(opts.s3gen_gguf_path, *audit_tok))
                throw std::runtime_error("S3 audit tokenizer load failed");
            tts_emit("audit.ready", "dir=" + opts.audit_dir);
        }
    }
    ~Impl() {
        join(preload);
        tts_emit("t3.unload.begin", " start");
        s3gen_unload();
        if (allocr) ggml_gallocr_free(allocr);
        free_model();
        tts_emit("t3.unload.completed", " done");
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
        if (session_index >= 0) tts_session_begin_if_needed();
        const auto started = std::chrono::steady_clock::now();
        vk_overlap_reset(model.backend);
        const int n_threads = threads(opts.n_threads);
        std::mt19937 rng(opts.seed);
        chatterbox_sampling_params sp;
        sp.top_k = opts.top_k;
        sp.top_p = opts.top_p;
        sp.min_p = opts.min_p;
        sp.temp = opts.temperature;
        sp.repeat_penalty = opts.repeat_penalty;
        sp.cfg_weight = opts.cfg_weight;
        tts_emit_piece("t3.begin", std::string("session_piece=") + std::to_string(session_index)
            + " text_chars=" + std::to_string(text.size())
            + " text_hash=" + hash_hex(hash_bytes(text.data(), text.size()))
            + " max_tokens=" + std::to_string(opts.n_predict)
            + " temp=" + std::to_string(sp.temp)
            + " top_k=" + std::to_string(sp.top_k)
            + " top_p=" + std::to_string(sp.top_p)
            + " min_p=" + std::to_string(sp.min_p)
            + " repeat_penalty=" + std::to_string(sp.repeat_penalty)
            + " repeat_last_n=" + std::to_string(REPEAT_PENALTY_LAST_N)
            + " repeat_stop=" + std::to_string(REPEAT_STOP_CONSECUTIVE)
            + " cfg_weight=" + std::to_string(sp.cfg_weight));

        std::vector<int32_t> text_tokens;
        if (model.hparams.variant == CHBX_VARIANT_MTL) {
            if (!mtl_tok) throw std::runtime_error("MTL tokenizer missing");
            text_tokens = mtl_tok->encode(text, opts.language);
            text_tokens.insert(text_tokens.begin(), model.hparams.start_text_token);
            text_tokens.push_back(model.hparams.stop_text_token);
        } else {
            if (model.tok_tokens.empty()) throw std::runtime_error("Turbo tokenizer missing");
            gpt2_bpe bpe;
            bpe.load_from_arrays(model.tok_tokens, model.tok_merges);
            text_tokens = bpe.tokenize(gpt2_bpe::punc_norm(text));
        }
        if (text_tokens.empty()) throw std::runtime_error("empty T3 text tokens");
        tts_emit_piece("t3.text", std::string("session_piece=") + std::to_string(session_index)
            + " tokens=" + std::to_string(text_tokens.size()) + " token_hash=" + token_hash(text_tokens));
        if (!opts.audit_dir.empty())
            tts_emit_piece("t3.audit.text", "text_seq=" + token_csv(text_tokens));

        int n_past = 0, speech_pos = 1;
        int32_t token = 0, pending_mtl = -1;
        bool repeat_stopped = false;
        std::vector<int32_t> out, tokens;
        std::vector<std::string> logits_hashes;
        out.reserve((size_t)opts.n_predict + 1);
        tokens.reserve((size_t)opts.n_predict);
        auto publish = [&](int32_t value) {
            if (value < 0 || value >= model.hparams.start_speech_token || value == model.hparams.stop_speech_token) return;
            if (model.hparams.variant == CHBX_VARIANT_MTL) {
                if (pending_mtl >= 0) tokens.push_back(pending_mtl);
                pending_mtl = value;
            } else tokens.push_back(value);
        };

        if (model.hparams.variant == CHBX_VARIANT_MTL) {
            std::vector<float> logits_c, logits_u;
            if (!eval_prompt_mtl(model, allocr, n_threads, text_tokens, opts.exaggeration, logits_c, logits_u, n_past))
                throw std::runtime_error("MTL prompt failed");
            if (!opts.audit_dir.empty()) logits_hashes.push_back(float_hash(logits_c) + "/" + float_hash(logits_u));
            token = sample_next_token_mtl(logits_c, logits_u, out, sp, rng, model.hparams.stop_speech_token);
        } else {
            std::vector<float> logits;
            if (!eval_prompt(model, allocr, n_threads, text_tokens, logits, n_past)) throw std::runtime_error("Turbo prompt failed");
            if (!opts.audit_dir.empty()) logits_hashes.push_back(float_hash(logits));
            token = sample_next_token_ex(logits, out, sp, rng);
        }
        out.push_back(token);
        publish(token);

        for (int i = 0; i < opts.n_predict && token != model.hparams.stop_speech_token && n_past + 1 <= model.hparams.n_ctx; ++i) {
            if (model.hparams.variant == CHBX_VARIANT_MTL) {
                std::vector<float> logits_c, logits_u;
                if (!eval_step_mtl(model, allocr, n_threads, n_past++, speech_pos++, token, logits_c, logits_u))
                    throw std::runtime_error("MTL step failed");
                if (!opts.audit_dir.empty()) logits_hashes.push_back(float_hash(logits_c) + "/" + float_hash(logits_u));
                token = sample_next_token_mtl(logits_c, logits_u, out, sp, rng, model.hparams.stop_speech_token);
            } else {
                std::vector<float> logits;
                if (!eval_step(model, allocr, n_threads, n_past++, token, logits)) throw std::runtime_error("Turbo step failed");
                if (!opts.audit_dir.empty()) logits_hashes.push_back(float_hash(logits));
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
        if (tokens.empty() && pending_mtl >= 0) tokens.push_back(pending_mtl);
        const int tail_drop = model.hparams.variant == CHBX_VARIANT_MTL && pending_mtl >= 0 ? 1 : 0;
        const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        tts_emit_piece("t3.end", std::string("session_piece=") + std::to_string(session_index)
            + " speech_tokens=" + std::to_string(tokens.size())
            + " speech_hash=" + token_hash(tokens)
            + " speech_seq=" + token_csv(tokens)
            + " tail_drop=" + std::to_string(tail_drop)
            + " tail_token=" + std::to_string(tail_drop ? pending_mtl : -1)
            + " ms=" + std::to_string((int)(elapsed + .5))
            + " stop=" + (repeat_stopped ? "repeat" : "eos")
            + " kv_pos=" + std::to_string(n_past)
            + " speech_pos=" + std::to_string(speech_pos)
            + vk_overlap_fields(model.backend));
        if (!opts.audit_dir.empty())
            tts_emit_piece("t3.audit.logits", "steps=" + std::to_string(logits_hashes.size()) +
                " hashes=" + string_csv(logits_hashes));
        return tokens;
    }
    void emit_s3_line() {
        if (acoustic.samples <= 0) return;
        const double audio_ms = 1000.0 * acoustic.samples / 24000.0;
        char rtf[32];
        std::snprintf(rtf, sizeof(rtf), "%.3f", audio_ms > 0.0 ? acoustic.pipeline_ms / audio_ms : 0.0);
        tts_emit_piece("s3.end",
            std::string(" history_tokens=") + std::to_string(acoustic.history_tokens)
            + " speech_tokens=" + std::to_string(acoustic.speech_tokens)
            + " window_tokens=" + std::to_string(acoustic.window_tokens)
            + " cfm_steps=" + std::to_string(acoustic.cfm_steps_used)
            + " pending_in=" + std::to_string(acoustic.pending_in)
            + " pending_out=" + std::to_string(acoustic.pending_pcm.size())
            + " mel_cache_frames=" + std::to_string(acoustic.mel.size() / 80)
            + " source_cache=" + std::to_string(acoustic.source.size())
            + " phase=" + std::to_string(acoustic.phase.size())
            + " emit_begin=" + std::to_string(acoustic.emit_begin)
            + " emit_end=" + std::to_string(acoustic.emit_end)
            + " hold=" + std::to_string(acoustic.hold)
            + " emitted=" + std::to_string(acoustic.emitted)
            + " encoder_ms=" + std::to_string((int)(acoustic.encoder_ms + 0.5))
            + " cfm_ms=" + std::to_string((int)(acoustic.cfm_ms + 0.5))
            + " f0_ms=" + std::to_string((int)(acoustic.f0_ms + 0.5))
            + " stft_ms=" + std::to_string((int)(acoustic.stft_ms + 0.5))
            + " hift_ms=" + std::to_string((int)(acoustic.hift_ms + 0.5))
            + " audio_ms=" + std::to_string((int)(audio_ms + 0.5))
            + " rtf=" + rtf
            + " samples=" + std::to_string(acoustic.samples)
            + " prompt_tokens=" + std::to_string(acoustic.prompt_tokens)
            + s3_overlap_fields());
        if (!acoustic.audit_summary.empty()) tts_emit_piece("s3.audit", acoustic.audit_summary);
        if (!acoustic.audit_local.empty()) tts_emit_piece("s3.audit.local", acoustic.audit_local);
    }
    void run_s3(const std::vector<int32_t>& tokens, int session_index, std::uint32_t external_piece, bool last_piece, const PieceCallback& cb) {
        auto synthesis_context = tts_get_context();
        if (session_index >= 0) { synthesis_context.valid = true; synthesis_context.piece = external_piece; }
        tts_context_scope context_scope(synthesis_context);
        if (tokens.empty()) throw std::runtime_error("S3Gen speech tokens empty");
        acoustic.encoder_ms = acoustic.cfm_ms = acoustic.f0_ms = acoustic.stft_ms = acoustic.hift_ms = acoustic.pipeline_ms = 0;
        acoustic.samples = acoustic.prompt_tokens = acoustic.speech_tokens = 0;
        s3_overlap_reset();
        acoustic.token_end = (int)speech_history.size();
        std::vector<int32_t> window;
        window.reserve(speech_history.size() + tokens.size());
        window.insert(window.end(), speech_history.begin(), speech_history.end());
        window.insert(window.end(), tokens.begin(), tokens.end());
        tts_emit_piece("s3.begin", std::string("session_piece=") + std::to_string(session_index)
            + " last=" + (last_piece ? "1" : "0")
            + " history_tokens=" + std::to_string(speech_history.size())
            + " history_hash=" + token_hash(speech_history)
            + " new_tokens=" + std::to_string(tokens.size())
            + " new_hash=" + token_hash(tokens)
            + " window_hash=" + token_hash(window)
            + " token_start=0 internal_final=1 lookahead=" + std::to_string(kSpeechLookaheadTokens)
            + " audit=" + (opts.audit_dir.empty() ? "0" : "1"));
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
        if (!opts.audit_dir.empty()) {
            const auto ctx = tts_get_context();
            s.audit_prefix = opts.audit_dir + "/r" + std::to_string(ctx.response) +
                "_p" + std::to_string(external_piece);
        }
        std::vector<float> pcm;
        s.pcm_out = &pcm;
        s3gen_synthesize(window, s);
        if (audit_tok && !pcm.empty()) {
            std::vector<float> audit_pcm = pcm;
            normalise_lufs(audit_pcm, 24000, -27.0);
            audit_pcm = resample_sinc(audit_pcm, 24000, 16000);
            std::vector<int32_t> roundtrip;
            if (!s3tokv2_tokenize(audit_pcm, *audit_tok, -1, roundtrip, threads(opts.n_threads), model.backend))
                throw std::runtime_error("S3 audit round-trip tokenize failed");
            tts_emit_piece("s3.roundtrip", "expected_tokens=" + std::to_string(tokens.size()) +
                " expected_hash=" + token_hash(tokens) +
                " observed_tokens=" + std::to_string(roundtrip.size()) +
                " observed_hash=" + token_hash(roundtrip) +
                " edit_distance=" + std::to_string(token_edit_distance(tokens, roundtrip)) +
                " observed_seq=" + token_csv(roundtrip));
        }
        if (!pcm.empty()) tts_session_note_first_audio();
        if (cb) cb(session_index, pcm.data(), pcm.size(), 0, true);
        if ((int)window.size() > kSpeechHistoryTokens) {
            speech_history.assign(window.end() - kSpeechHistoryTokens, window.end());
        } else {
            speech_history = window;
        }
        emit_s3_line();
        if (session_index >= 0) tts_session_touch_end();
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
    tts_emit("warmup.start", " begin");
    pimpl_->reset_acoustics();
    std::size_t samples = 0;
    pimpl_->piece_streaming("Warm up.", -1, [&](int, const float*, std::size_t n, int, bool) { samples += n; });
    if (!samples) throw std::runtime_error("warm-up produced no PCM");
    if (pimpl_->model.buffer_kv) ggml_backend_buffer_clear(pimpl_->model.buffer_kv, 0);
    pimpl_->reset_acoustics();
    tts_emit("warmup.completed", std::string(" samples=") + std::to_string(samples));
}
}
