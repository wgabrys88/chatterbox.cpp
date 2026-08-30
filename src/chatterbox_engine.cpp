#include "tts-cpp/chatterbox/engine.h"
#include "tts-cpp/chatterbox/log.h"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <random>
#include <regex>
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "chatterbox_t3_internal.h"
#include "mtl_tokenizer.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
namespace tts_cpp::chatterbox {
using namespace detail;
namespace {
struct gpt2_bpe {
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int> bpe_ranks;
    bool load_from_arrays(const std::vector<std::string>& tokens, const std::vector<std::string>& merges);
    std::vector<int32_t> tokenize(const std::string& text) const;
    static std::string punc_norm(const std::string& text);
};

static std::unordered_map<uint8_t, std::string> build_byte_to_unicode() {
    std::unordered_map<uint8_t, std::string> m;
    auto cpt_to_utf8 = [](uint32_t cp) -> std::string {
        std::string s;
        if (cp < 0x80) { s += (char)cp; }
        else if (cp < 0x800) { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
        else { s += (char)(0xF0 | (cp >> 18)); s += (char)(0x80 | ((cp >> 12) & 0x3F)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
        return s;
    };
    for (int c = 0x21; c <= 0x7E; ++c) m[(uint8_t)c] = cpt_to_utf8((uint32_t)c);
    for (int c = 0xA1; c <= 0xAC; ++c) m[(uint8_t)c] = cpt_to_utf8((uint32_t)c);
    for (int c = 0xAE; c <= 0xFF; ++c) m[(uint8_t)c] = cpt_to_utf8((uint32_t)c);
    int n = 0;
    for (int c = 0; c < 256; ++c) {
        if (m.find((uint8_t)c) == m.end()) {
            m[(uint8_t)c] = cpt_to_utf8(256 + n);
            ++n;
        }
    }
    return m;
}
static const std::unordered_map<uint8_t, std::string> & byte_to_unicode() {
    static auto m = build_byte_to_unicode();
    return m;
}
static std::string bytes_to_unicode_str(const std::string & raw) {
    std::string out;
    auto & b2u = byte_to_unicode();
    for (unsigned char c : raw) out += b2u.at(c);
    return out;
}
bool gpt2_bpe::load_from_arrays(const std::vector<std::string> & tokens,
                                const std::vector<std::string> & merges) {
    if (tokens.empty()) return false;
    token_to_id.clear();
    token_to_id.reserve(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        token_to_id[tokens[i]] = (int32_t) i;
    }
    bpe_ranks.clear();
    bpe_ranks.reserve(merges.size());
    for (size_t i = 0; i < merges.size(); ++i) {
        bpe_ranks[merges[i]] = (int) i;
    }
    return true;
}
static std::vector<std::string> gpt2_regex_split(const std::string & text) {
    static const std::regex re(
        R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)",
        std::regex::optimize);
    std::vector<std::string> words;
    auto begin = std::sregex_iterator(text.begin(), text.end(), re);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        words.push_back(it->str());
    }
    return words;
}
static int find_rank(const std::unordered_map<std::string, int> & ranks,
                     const std::string & left, const std::string & right) {
    auto it = ranks.find(left + " " + right);
    return it != ranks.end() ? it->second : -1;
}
static std::vector<std::string> bpe_merge(const std::string & token,
                                          const std::unordered_map<std::string, int> & ranks) {
    std::vector<std::string> parts;
    for (size_t i = 0; i < token.size(); ) {
        size_t len = 1;
        unsigned char c = (unsigned char)token[i];
        if      ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        parts.push_back(token.substr(i, len));
        i += len;
    }
    while (parts.size() >= 2) {
        int best_rank = INT32_MAX;
        size_t best_i = 0;
        for (size_t i = 0; i + 1 < parts.size(); ++i) {
            int r = find_rank(ranks, parts[i], parts[i+1]);
            if (r >= 0 && r < best_rank) { best_rank = r; best_i = i; }
        }
        if (best_rank == INT32_MAX) break;
        parts[best_i] = parts[best_i] + parts[best_i + 1];
        parts.erase(parts.begin() + (int)best_i + 1);
    }
    return parts;
}
std::vector<int32_t> gpt2_bpe::tokenize(const std::string & text) const {
    std::vector<int32_t> ids;
    if (text.empty()) return ids;
    struct added_span { size_t start; size_t len; int32_t id; };
    std::vector<added_span> spans;
    for (auto & [tok, id] : token_to_id) {
        if (id < 50257) continue;
        size_t pos = 0;
        while ((pos = text.find(tok, pos)) != std::string::npos) {
            spans.push_back({pos, tok.size(), id});
            pos += tok.size();
        }
    }
    std::sort(spans.begin(), spans.end(), [](const added_span & a, const added_span & b) { return a.start < b.start; });
    std::vector<added_span> clean;
    size_t last_end = 0;
    for (auto & sp : spans) {
        if (sp.start >= last_end) { clean.push_back(sp); last_end = sp.start + sp.len; }
    }
    auto tokenize_fragment = [&](const std::string & frag) {
        auto words = gpt2_regex_split(frag);
        for (auto & word : words) {
            std::string bpe_input = bytes_to_unicode_str(word);
            auto parts = bpe_merge(bpe_input, bpe_ranks);
            for (auto & part : parts) {
                auto it = token_to_id.find(part);
                if (it != token_to_id.end()) {
                    ids.push_back(it->second);
                } else {
                    for (unsigned char c : word) {
                        auto & b2u = byte_to_unicode();
                        auto jt = token_to_id.find(b2u.at(c));
                        if (jt != token_to_id.end()) ids.push_back(jt->second);
                    }
                }
            }
        }
    };
    size_t cursor = 0;
    for (auto & sp : clean) {
        if (sp.start > cursor) tokenize_fragment(text.substr(cursor, sp.start - cursor));
        ids.push_back(sp.id);
        cursor = sp.start + sp.len;
    }
    if (cursor < text.size()) tokenize_fragment(text.substr(cursor));
    return ids;
}
std::string gpt2_bpe::punc_norm(const std::string & text) {
    if (text.empty()) return "You need to add some text for me to talk.";
    std::string t = text;
    if (t[0] >= 'a' && t[0] <= 'z') t[0] = t[0] - 'a' + 'A';
    {
        std::string r;
        bool prev_space = false;
        for (char c : t) {
            if (c == ' ') { if (!prev_space) r += c; prev_space = true; }
            else { r += c; prev_space = false; }
        }
        t = r;
    }
    auto replace_all = [](std::string & s, const std::string & from, const std::string & to) {
        size_t pos = 0;
        while ((pos = s.find(from, pos)) != std::string::npos) {
            s.replace(pos, from.size(), to);
            pos += to.size();
        }
    };
    replace_all(t, "\xe2\x80\xa6", ", ");
    replace_all(t, ":", ",");
    replace_all(t, "\xe2\x80\x94", "-");
    replace_all(t, "\xe2\x80\x93", "-");
    replace_all(t, " ,", ",");
    replace_all(t, "\xe2\x80\x9c", "\"");
    replace_all(t, "\xe2\x80\x9d", "\"");
    replace_all(t, "\xe2\x80\x98", "'");
    replace_all(t, "\xe2\x80\x99", "'");
    while (!t.empty()) {
        char b = t.back();
        if (b == ' ' || b == '\t' || b == '\n' || b == '\r') t.pop_back();
        else break;
    }
    if (!t.empty()) {
        char last = t.back();
        if (last != '.' && last != '!' && last != '?' && last != '-' && last != ',')
            t += '.';
    }
    return t;
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
    gpt2_bpe turbo_tok;
    std::atomic<bool> cancelled{false};
    explicit Impl(const EngineOptions& o) : opts(o) {}
    void init() {
        if (!std::filesystem::exists(opts.t3_gguf_path)) throw std::runtime_error("T3 GGUF missing");
        if (!std::filesystem::exists(opts.s3gen_gguf_path)) throw std::runtime_error("S3Gen GGUF missing");
        if (!validate_reference_audio(opts.reference_audio)) throw std::runtime_error("reference WAV invalid");
        ggml_time_init();
        ggml_log_set(chatterbox_log_cb, nullptr);
        if (!load_model_gguf(opts.t3_gguf_path, model, opts.n_ctx, opts.n_gpu_layers)) throw std::runtime_error("T3 load failed");
        if (model.hparams.variant != CHBX_VARIANT_TURBO && model.hparams.variant != CHBX_VARIANT_MTL) throw std::runtime_error("unsupported T3 variant");
        if (model.hparams.variant == CHBX_VARIANT_MTL) {
            mtl_tok = std::make_unique<mtl_tokenizer>();
            if (model.mtl_tokenizer_json.empty() || !mtl_tok->load_from_json(model.mtl_tokenizer_json)) throw std::runtime_error("MTL tokenizer missing");
            if (opts.language == "zh" && (model.mtl_cangjie_json.empty() || !mtl_tok->load_cangjie_json(model.mtl_cangjie_json))) throw std::runtime_error("MTL Cangjie mapping missing or invalid");
        } else if (!turbo_tok.load_from_arrays(model.tok_tokens, model.tok_merges)) {
            throw std::runtime_error("Turbo tokenizer missing");
        }
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!allocr) throw std::runtime_error("T3 allocator failed");
        preload = std::thread([this] { s3gen_preload(opts.s3gen_gguf_path, opts.n_gpu_layers, opts.fastconv); });
        bake_voice();
        join(preload);
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
        normalise_lufs(wav, sr);
        if (sr != 16000) wav = resample_sinc(wav, sr, 16000);
        if (wav.size() > 30u * 16000u) wav.resize(30u * 16000u);
        if (!voice_encoder_embed(wav, ve, model.backend, speaker)) throw std::runtime_error("VoiceEncoder failed");
        if ((int64_t)speaker.size() != ggml_nelements(model.builtin_speaker_emb)) throw std::runtime_error("speaker embedding size mismatch");
        ggml_backend_tensor_set(model.builtin_speaker_emb, speaker.data(), 0, ggml_nbytes(model.builtin_speaker_emb));
        std::vector<int32_t> cond;
        if (!compute_speech_tokens_native(opts.reference_audio, opts.s3gen_gguf_path, model.hparams.cond_prompt_len,
                prompt_token, cond, n_threads, model.backend)) throw std::runtime_error("S3Tokenizer failed");
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
        if (!compute_prompt_feat_native(opts.reference_audio, opts.s3gen_gguf_path, prompt_feat, prompt_rows)) throw std::runtime_error("prompt feature failed");
        if (!compute_embedding_native(opts.reference_audio, opts.s3gen_gguf_path, embedding)) throw std::runtime_error("CAMPPlus failed");
        if (prompt_token.empty() || prompt_feat.empty() || embedding.empty()) throw std::runtime_error("voice conditioning empty");
    }
    // True streaming T3->s3gen pipeline. T3 runs on a worker thread, emitting
    // each new speech token to a shared buffer. The main thread consumes tokens
    // in 12 / +25 chunks and runs s3gen on the prefix, hiding T3 latency
    // behind s3gen inference.
    void piece_streaming(const std::string& text, int index, const PieceCallback& cb) {
        if (text.empty()) return;
        std::mutex mu;
        std::condition_variable cv;
        std::vector<int32_t> tokens;
        bool t3_done = false;
        std::atomic<bool> t3_failed{false};
        std::string t3_error;
        const int chunk_first = 12;
        const int chunk_next = 25;

        std::thread t3_thread([&] {
            const auto started = std::chrono::steady_clock::now();
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

                std::vector<int32_t> text_tokens;
                if (model.hparams.variant == CHBX_VARIANT_MTL) {
                    if (!mtl_tok) throw std::runtime_error("MTL tokenizer missing");
                    text_tokens = mtl_tok->encode(text, opts.language);
                    text_tokens.insert(text_tokens.begin(), model.hparams.start_text_token);
                    text_tokens.push_back(model.hparams.stop_text_token);
                } else {
                    text_tokens = turbo_tok.tokenize(gpt2_bpe::punc_norm(text));
                }
                if (text_tokens.empty()) throw std::runtime_error("empty T3 text tokens");

                int n_past = 0;
                int32_t token = 0;
                bool ready_emitted = false;
                std::vector<int32_t> out;
                out.reserve((size_t)opts.n_predict + 1);
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
                {
                    std::lock_guard lock(mu);
                    tokens = out;
                }
                cv.notify_all();

                for (int i = 0; i < opts.n_predict && token != model.hparams.stop_speech_token && n_past + 1 <= model.hparams.n_ctx; ++i) {
                    check();
                    if (!ready_emitted && (int)out.size() >= chunk_first) {
                        const double ready_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
                        tts_emit("t3.ready", ",\"index\":" + std::to_string(index)
                            + ",\"tokens\":" + std::to_string(out.size())
                            + ",\"ms\":" + std::to_string((int)(ready_ms + 0.5)));
                        ready_emitted = true;
                    }
                    if (model.hparams.variant == CHBX_VARIANT_MTL) {
                        std::vector<float> logits_c, logits_u;
                        if (!eval_step_mtl(model, allocr, n_threads, n_past++, token, logits_c, logits_u)) throw std::runtime_error("MTL step failed");
                        token = sample_next_token_mtl(logits_c, logits_u, out, sp, rng, model.hparams.stop_speech_token);
                    } else {
                        std::vector<float> logits;
                        if (!eval_step(model, allocr, n_threads, n_past++, token, logits)) throw std::runtime_error("Turbo step failed");
                        token = sample_next_token_ex(logits, out, sp, rng);
                    }
                    out.push_back(token);
                    {
                        std::lock_guard lock(mu);
                        tokens.push_back(token);
                    }
                    cv.notify_all();
                }
                if (!ready_emitted && (int)out.size() >= chunk_first) {
                    const double ready_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
                    tts_emit("t3.ready", ",\"index\":" + std::to_string(index)
                        + ",\"tokens\":" + std::to_string(out.size())
                        + ",\"ms\":" + std::to_string((int)(ready_ms + 0.5)));
                }

                std::size_t token_count = 0;
                {
                    std::unique_lock lock(mu);
                    if (!tokens.empty() && tokens.back() == model.hparams.stop_speech_token) tokens.pop_back();
                    if (model.hparams.variant == CHBX_VARIANT_MTL && tokens.size() > 1) tokens.pop_back();
                    if (model.hparams.variant != CHBX_VARIANT_MTL) {
                        const int32_t vocab = model.hparams.start_speech_token;
                        tokens.erase(std::remove_if(tokens.begin(), tokens.end(), [vocab](int32_t v) { return v < 0 || v >= vocab; }), tokens.end());
                    }
                    token_count = tokens.size();
                    t3_done = true;
                }
                const double elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
                tts_emit("t3", ",\"index\":" + std::to_string(index)
                    + ",\"chars\":" + std::to_string(text.size())
                    + ",\"tokens\":" + std::to_string(token_count)
                    + ",\"ms\":" + std::to_string((int)(elapsed_ms + 0.5))
                    + ",\"stream\":true");
                cv.notify_all();
            } catch (const std::exception& e) {
                t3_failed.store(true, std::memory_order_relaxed);
                { std::unique_lock lock(mu); t3_error = e.what(); t3_done = true; }
                cv.notify_all();
            }
        });

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

        std::vector<float> cache;
        int emitted = 0;
        int chunk_id = 0;
        int end = 0;
        try {
            while (true) {
                const int threshold = (chunk_id == 0) ? chunk_first : (end + chunk_next);
                std::vector<int32_t> prefix;
                bool final = false;
                {
                    std::unique_lock lock(mu);
                    cv.wait(lock, [&] { return t3_failed.load(std::memory_order_relaxed) || (int)tokens.size() >= threshold || t3_done; });
                    if (t3_failed.load(std::memory_order_relaxed)) throw std::runtime_error(t3_error);
                    if ((int)tokens.size() < threshold && !t3_done) continue;
                    if (t3_done && (int)tokens.size() <= end) break;
                    end = std::min((int)tokens.size(), threshold);
                    prefix.assign(tokens.begin(), tokens.begin() + end);
                    final = t3_done && end == (int)tokens.size();
                }
                std::vector<float> pcm, tail;
                s.pcm_out = &pcm;
                s.final = final;
                s.skip_mel_frames = emitted;
                s.chunk_id = chunk_id++;
                s.hift_cache_source = std::move(cache);
                s.hift_source_tail = &tail;
                tts_emit("s3gen.begin", ",\"index\":" + std::to_string(index)
                    + ",\"chunk_id\":" + std::to_string(s.chunk_id)
                    + ",\"tokens\":" + std::to_string(prefix.size())
                    + ",\"final\":" + (s.final ? "true" : "false"));
                s3gen_synthesize(prefix, s);
                check();
                if (cb) cb(index, pcm.data(), pcm.size(), chunk_id - 1, s.final);
                emitted += (int)pcm.size() / 480;
                cache = std::move(tail);
                if (s.final) break;
            }
        } catch (...) {
            cancelled.store(true, std::memory_order_relaxed);
            cv.notify_all();
            join(t3_thread);
            throw;
        }
        join(t3_thread);
    }
};
Engine::Engine(const EngineOptions& o) : pimpl_(std::make_unique<Impl>(o)) { pimpl_->init(); }
Engine::~Engine() = default;
Engine::Engine(Engine&&) noexcept = default;
Engine& Engine::operator=(Engine&&) noexcept = default;
void Engine::synthesize_pieces_streaming(const std::vector<std::string>& texts, const PieceCallback& cb) {
    pimpl_->cancelled.store(false, std::memory_order_relaxed);
    if (texts.empty()) return;
    if (texts.size() == 1) {
        pimpl_->check();
        pimpl_->piece_streaming(texts[0], 0, cb);
        return;
    }
    std::string joined;
    for (const auto& text : texts) {
        if (text.empty()) continue;
        if (!joined.empty() && !std::isspace(static_cast<unsigned char>(joined.back()))) joined.push_back(' ');
        joined += text;
    }
    if (joined.empty()) return;
    pimpl_->check();
    pimpl_->piece_streaming(joined, 0, cb);
}
void Engine::cancel() { pimpl_->cancelled.store(true, std::memory_order_relaxed); }
}
