#include "tts-cpp/chatterbox/engine.h"
#include "tts-cpp/chatterbox/log.h"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <mutex>
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
#include "voice_encoder.h"
#include "voice_features.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
namespace tts_cpp::chatterbox {
using namespace detail;
namespace {
bool third_consecutive(const std::vector<int32_t>& generated, int32_t token) {
    return generated.size() >= 2 && generated[generated.size() - 1] == token && generated[generated.size() - 2] == token;
}
int threads(int n) {
    if (n > 0) return n;
    const int hw = (int)std::thread::hardware_concurrency();
    return hw > 0 ? std::min(hw, 4) : 4;
}
void join(std::thread& t) { if (t.joinable()) t.join(); }
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
    std::atomic<bool> cancelled{false};
    std::uint32_t last_epoch = 0;
    int pieces_in_session = 0;
    s3gen_piece_state acoustic;
    std::vector<int32_t> speech_history;
    explicit Impl(const EngineOptions& o) : opts(o) {}
    void reset_acoustics() { acoustic = {}; speech_history.clear(); }
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
    void check() const {
        if (cancelled.load(std::memory_order_relaxed)) throw std::runtime_error("synthesis cancelled");
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
    struct T3Job {
        std::mutex mu;
        std::condition_variable cv;
        std::vector<int32_t> tokens;
        bool done = false;
        std::atomic<bool> failed{false};
        std::string error;
        std::thread thread;
        std::atomic<long long> start_us{0};
        std::atomic<long long> end_us{0};
        void join_thread() { join(thread); }
    };
    void launch_t3(T3Job& job, const std::string& text, int index) {
        auto synthesis_context = tts_get_context();
        if (index >= 0) { synthesis_context.valid = true; synthesis_context.piece_id = (std::uint32_t)index; }
        if (index >= 0) tts_session_begin_if_needed();
        job.thread = std::thread([this, &job, text, synthesis_context] {
            tts_context_scope context_scope(synthesis_context);
            job.start_us.store(tts_mono_us(), std::memory_order_relaxed);
            const auto started = std::chrono::steady_clock::now();
            std::vector<int32_t> text_tokens;
            try {
                const int n_threads = threads(opts.n_threads);
                std::mt19937 rng(opts.seed);
                chatterbox_sampling_params sp;
                sp.top_k = opts.top_k;
                sp.top_p = opts.top_p;
                sp.min_p = opts.min_p;
                sp.temp = opts.temperature;
                sp.repeat_penalty = opts.repeat_penalty;
                sp.cfg_weight = opts.cfg_weight;

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

                int n_past = 0;
                int32_t token = 0;
                int32_t repeat_token = -1;
                bool repeat_stopped = false;
                std::vector<int32_t> out;
                out.reserve((size_t)opts.n_predict + 1);
                int32_t pending_mtl = -1;
                auto publish = [&](int32_t value) {
                    if (value < 0 || value >= model.hparams.start_speech_token ||
                        value == model.hparams.stop_speech_token) return;
                    std::lock_guard lock(job.mu);
                    if (model.hparams.variant == CHBX_VARIANT_MTL) {
                        if (pending_mtl >= 0) job.tokens.push_back(pending_mtl);
                        pending_mtl = value;
                    } else job.tokens.push_back(value);
                    job.cv.notify_all();
                };
                if (model.hparams.variant == CHBX_VARIANT_MTL) {
                    std::vector<float> logits_c, logits_u;
                    if (!eval_prompt_mtl(model, allocr, n_threads, text_tokens, opts.exaggeration, logits_c, logits_u, n_past)) throw std::runtime_error("MTL prompt failed");
                    token = sample_next_token_mtl(logits_c, logits_u, out, sp, rng, model.hparams.stop_speech_token);
                } else {
                    std::vector<float> logits;
                    if (!eval_prompt(model, allocr, n_threads, text_tokens, logits, n_past)) throw std::runtime_error("Turbo prompt failed");
                    token = sample_next_token_ex(logits, out, sp, rng);
                }
                out.push_back(token);
                publish(token);
                job.cv.notify_all();

                for (int i = 0; i < opts.n_predict && token != model.hparams.stop_speech_token && n_past + 1 <= model.hparams.n_ctx; ++i) {
                    check();
                    if (model.hparams.variant == CHBX_VARIANT_MTL) {
                        std::vector<float> logits_c, logits_u;
                        if (!eval_step_mtl(model, allocr, n_threads, n_past++, token, logits_c, logits_u)) throw std::runtime_error("MTL step failed");
                        token = sample_next_token_mtl(logits_c, logits_u, out, sp, rng, model.hparams.stop_speech_token);
                        if (third_consecutive(out, token)) {
                            repeat_token = token; repeat_stopped = true; token = model.hparams.stop_speech_token;
                        }
                    } else {
                        std::vector<float> logits;
                        if (!eval_step(model, allocr, n_threads, n_past++, token, logits)) throw std::runtime_error("Turbo step failed");
                        token = sample_next_token_ex(logits, out, sp, rng);
                    }
                    out.push_back(token);
                    publish(token);
                    job.cv.notify_all();
                }

                const bool eos = token == model.hparams.stop_speech_token;
                if (!eos) throw std::runtime_error("streaming T3 stopped without EOS");
                std::vector<int32_t> logged_tokens;
                {
                    std::unique_lock lock(job.mu);
                    if (job.tokens.empty() && pending_mtl >= 0) job.tokens.push_back(pending_mtl);
                    logged_tokens = job.tokens;
                    job.done = true;
                }
                job.cv.notify_all();
                job.end_us.store(tts_mono_us(), std::memory_order_relaxed);
                const double elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
                const std::string stop_reason = repeat_stopped ? "repeat" : (eos ? "eos" : "window");
                tts_emit_piece("t3", std::string(" tokens=") + std::to_string(logged_tokens.size())
                    + " ms=" + std::to_string((int)(elapsed_ms + 0.5))
                    + " stop=" + stop_reason);
            } catch (const std::exception& e) {
                const bool was_cancelled = cancelled.load(std::memory_order_relaxed);
                std::vector<int32_t> partial;
                { std::unique_lock lock(job.mu); partial = job.tokens; }
                job.end_us.store(tts_mono_us(), std::memory_order_relaxed);
                tts_emit_piece("t3", std::string(" tokens=") + std::to_string(partial.size())
                    + " ms=" + std::to_string((int)(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count() + 0.5))
                    + " stop=" + (was_cancelled ? "cancelled" : "error"));
                { std::unique_lock lock(job.mu); job.error = e.what(); job.failed.store(true); job.done = true; }
                cancelled.store(true, std::memory_order_relaxed);
                job.cv.notify_all();
            }
        });
    }
    void finish_t3(T3Job& job) {
        job.join_thread();
        if (job.failed.load()) throw std::runtime_error(job.error);
    }
    void emit_s3_line() {
        if (acoustic.samples <= 0) return;
        const double audio_ms = 1000.0 * acoustic.samples / 24000.0;
        char rtf[32];
        std::snprintf(rtf, sizeof(rtf), "%.3f", audio_ms > 0.0 ? acoustic.pipeline_ms / audio_ms : 0.0);
        tts_emit_piece("s3",
            std::string(" encoder_ms=") + std::to_string((int)(acoustic.encoder_ms + 0.5))
            + " cfm_ms=" + std::to_string((int)(acoustic.cfm_ms + 0.5))
            + " f0_ms=" + std::to_string((int)(acoustic.f0_ms + 0.5))
            + " stft_ms=" + std::to_string((int)(acoustic.stft_ms + 0.5))
            + " hift_ms=" + std::to_string((int)(acoustic.hift_ms + 0.5))
            + " audio_ms=" + std::to_string((int)(audio_ms + 0.5))
            + " rtf=" + rtf
            + " samples=" + std::to_string(acoustic.samples)
            + " prompt_tokens=" + std::to_string(acoustic.prompt_tokens)
            + " speech_tokens=" + std::to_string(acoustic.speech_tokens));
    }
    void run_s3(const std::vector<int32_t>& tokens, int index, const PieceCallback& cb) {
        auto synthesis_context = tts_get_context();
        if (index >= 0) { synthesis_context.valid = true; synthesis_context.piece_id = (std::uint32_t)index; }
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
        s.cancel = &cancelled;
        s.state = &acoustic;
        s.token_start = 0;
        s.token_end = (int)window.size();
        s.final = true;
        s.first_piece = (index <= 0);
        s.chunk_id = 0;
        std::vector<float> pcm;
        s.pcm_out = &pcm;
        s3gen_synthesize(window, s);
        check();
        if (!pcm.empty()) tts_session_note_first_audio();
        if (cb) cb(index, pcm.data(), pcm.size(), 0, true);
        if ((int)window.size() > kSpeechHistoryTokens)
            speech_history.assign(window.end() - kSpeechHistoryTokens, window.end());
        else
            speech_history = window;
        emit_s3_line();
        if (index >= 0) tts_session_touch_end();
    }
    struct PieceWork {
        int session_index = 0;
        int cb_index = 0;
        std::string text;
    };
    void pipeline_pieces(const std::vector<PieceWork>& work, const PieceCallback& cb) {
        if (work.empty()) return;
        auto job = std::make_unique<T3Job>();
        launch_t3(*job, work[0].text, work[0].session_index);
        for (size_t i = 0; i < work.size(); ++i) {
            finish_t3(*job);
            std::vector<int32_t> tokens;
            { std::lock_guard lock(job->mu); tokens = std::move(job->tokens); }
            std::unique_ptr<T3Job> next;
            if (i + 1 < work.size()) {
                next = std::make_unique<T3Job>();
                launch_t3(*next, work[i + 1].text, work[i + 1].session_index);
            }
            const long long s3_t0 = tts_mono_us();
            try {
                run_s3(tokens, work[i].session_index, [&](int, const float* pcm, std::size_t n, int chunk, bool final) {
                    if (cb) cb(work[i].cb_index, pcm, n, chunk, final);
                });
            } catch (...) {
                cancelled.store(true, std::memory_order_relaxed);
                if (next) {
                    next->cv.notify_all();
                    next->join_thread();
                    if (next->failed.load()) throw std::runtime_error(next->error);
                }
                throw;
            }
            const long long s3_t1 = tts_mono_us();
            if (next) {
                const long long t3s = next->start_us.load(std::memory_order_relaxed);
                long long t3e = next->end_us.load(std::memory_order_relaxed);
                if (!t3e) t3e = s3_t1;
                const long long lo = std::max(s3_t0, t3s);
                const long long hi = std::min(s3_t1, t3e);
                if (hi > lo) tts_session_add_overlap((double)(hi - lo) / 1000.0);
                job = std::move(next);
            }
        }
    }
    void piece_streaming(const std::string& text, int index, const PieceCallback& cb) {
        if (text.empty()) return;
        pipeline_pieces({{index, 0, text}}, cb);
    }
};
Engine::Engine(const EngineOptions& o) : pimpl_(std::make_unique<Impl>(o)) { pimpl_->init(); }
Engine::~Engine() = default;
Engine::Engine(Engine&&) noexcept = default;
Engine& Engine::operator=(Engine&&) noexcept = default;
void Engine::begin_synthesis() {
    pimpl_->cancelled.store(false, std::memory_order_release);
    pimpl_->pieces_in_session = 0;
    pimpl_->reset_acoustics();
}
void Engine::synthesize_pieces_streaming(const std::vector<std::string>& texts, const PieceCallback& cb) {
    const auto ctx = tts_get_context();
    const std::uint32_t epoch = ctx.valid ? ctx.epoch : 0;
    if (epoch != pimpl_->last_epoch) {
        pimpl_->pieces_in_session = 0;
        pimpl_->last_epoch = epoch;
        pimpl_->reset_acoustics();
    }
    std::vector<Impl::PieceWork> work;
    int cb_index = 0;
    for (const auto& text : texts) {
        if (text.empty()) continue;
        pimpl_->check();
        work.push_back({pimpl_->pieces_in_session, cb_index++, text});
        ++pimpl_->pieces_in_session;
    }
    pimpl_->pipeline_pieces(work, cb);
}
void Engine::warm_up() {
    tts_emit("warmup.start", " begin");
    begin_synthesis();
    pimpl_->reset_acoustics();
    std::size_t samples = 0;
    pimpl_->piece_streaming("Warm up.", -1, [&](int, const float*, std::size_t n, int, bool) { samples += n; });
    if (!samples) throw std::runtime_error("warm-up produced no PCM");
    if (pimpl_->model.buffer_kv) ggml_backend_buffer_clear(pimpl_->model.buffer_kv, 0);
    pimpl_->pieces_in_session = 0;
    pimpl_->reset_acoustics();
    begin_synthesis();
    tts_emit("warmup.completed", std::string(" samples=") + std::to_string(samples));
}
void Engine::cancel() {
    pimpl_->cancelled.store(true, std::memory_order_relaxed);
    tts_emit("synthesis.cancel.requested", " ok");
}
}
