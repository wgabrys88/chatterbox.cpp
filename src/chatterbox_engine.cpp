#include "tts-cpp/chatterbox/engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "chatterbox_t3_internal.h"
#include "gpt2_bpe.h"
#include "mtl_tokenizer.h"
#include "npy.h"
#include "t3_mtl.h"
#include "tts-cpp/chatterbox/s3gen_pipeline.h"
#include "voice_encoder.h"
#include "voice_features.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

namespace tts_cpp::chatterbox {

using namespace detail;

namespace {

int resolve_thread_count(int requested) {
    if (requested > 0) return requested;
    const int hw = (int) std::thread::hardware_concurrency();
    return hw > 0 ? std::min(hw, 4) : 4;
}

void wait_for_preload(std::thread & t) {
    if (t.joinable()) t.join();
}



}

struct Engine::Impl {
    EngineOptions opts;

    chatterbox_model     model{};
    ggml_gallocr_t       allocr = nullptr;
    std::thread          s3gen_preload_thread;


    std::vector<float>   s3gen_prompt_feat;
    int                  s3gen_prompt_feat_rows = 0;
    std::vector<float>   s3gen_embedding;
    std::vector<int32_t> s3gen_prompt_token;

    std::atomic<bool>    cancel_flag{false};
    double last_t3_ms = 0.0;
    int    last_t3_tokens = 0;

    bool cancelled() const {
        return cancel_flag.load(std::memory_order_relaxed);
    }

    void ensure_not_cancelled(const char * where) const {
        if (cancelled()) {
            throw std::runtime_error(std::string("Engine: synthesis cancelled ") + where);
        }
    }

    explicit Impl(const EngineOptions & o)
        : opts(o) {
        if (opts.t3_gguf_path.empty()) {
            throw std::runtime_error("Engine: t3_gguf_path is required");
        }
        if (opts.s3gen_gguf_path.empty()) {
            throw std::runtime_error("Engine: s3gen_gguf_path is required");
        }
        if (!std::filesystem::exists(opts.t3_gguf_path)) {
            throw std::runtime_error("Engine: T3 GGUF not found: " + opts.t3_gguf_path);
        }
        if (!std::filesystem::exists(opts.s3gen_gguf_path)) {
            throw std::runtime_error("Engine: S3Gen GGUF not found: " + opts.s3gen_gguf_path);
        }
        if (!opts.reference_audio.empty() &&
            !std::filesystem::exists(opts.reference_audio)) {
            throw std::runtime_error("Engine: reference_audio not found: " + opts.reference_audio);
        }
        if (!opts.voice_dir.empty() &&
            !std::filesystem::is_directory(opts.voice_dir)) {
            throw std::runtime_error("Engine: voice_dir not found: " + opts.voice_dir);
        }

        ggml_time_init();
        g_log_verbose = opts.verbose ? 1 : 0;
        ggml_log_set(chatterbox_log_cb, nullptr);

        if (!opts.reference_audio.empty() &&
            !validate_reference_audio(opts.reference_audio)) {
            throw std::runtime_error("Engine: reference_audio failed validation: " + opts.reference_audio);
        }

        if (!load_model_gguf(opts.t3_gguf_path, model, opts.n_ctx, opts.n_gpu_layers)) {
            throw std::runtime_error("Engine: failed to load T3 GGUF: " + opts.t3_gguf_path);
        }

        if (model.hparams.variant != CHBX_VARIANT_TURBO &&
            model.hparams.variant != CHBX_VARIANT_MTL) {
            free_model();
            throw std::runtime_error(
                "Engine: unknown chatterbox.variant in " + opts.t3_gguf_path);
        }

        s3gen_preload_thread = std::thread([path = opts.s3gen_gguf_path, ngpu = opts.n_gpu_layers, fastconv = opts.fastconv]() {
            s3gen_preload(path, ngpu, fastconv);
        });

        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!allocr) {
            wait_for_preload(s3gen_preload_thread);
            s3gen_unload();
            free_model();
            throw std::runtime_error("Engine: ggml_gallocr_new failed");
        }

        try {
            bake_voice_conditioning();
        } catch (...) {
            wait_for_preload(s3gen_preload_thread);
            s3gen_unload();
            if (allocr) { ggml_gallocr_free(allocr); allocr = nullptr; }
            free_model();
            throw;
        }
        wait_for_preload(s3gen_preload_thread);
    }

    ~Impl() {
        wait_for_preload(s3gen_preload_thread);


        s3gen_unload();
        if (allocr) {
            ggml_gallocr_free(allocr);
            allocr = nullptr;
        }
        free_model();
    }

    Impl(const Impl &)             = delete;
    Impl & operator=(const Impl &) = delete;

    void free_model() {


        if (model.buffer_stack || model.ctx_stack) {
            t3_stack_unregister(model.buffer_stack, model.ctx_stack);
        }
        if (model.buffer_w)        { ggml_backend_buffer_free(model.buffer_w);        model.buffer_w        = nullptr; }
        if (model.buffer_kv)       { ggml_backend_buffer_free(model.buffer_kv);       model.buffer_kv       = nullptr; }
        if (model.buffer_stack)    { ggml_backend_buffer_free(model.buffer_stack);    model.buffer_stack    = nullptr; }
        if (model.buffer_override) { ggml_backend_buffer_free(model.buffer_override); model.buffer_override = nullptr; }
        if (model.backend)         { ggml_backend_free(model.backend);                model.backend         = nullptr; }
        if (model.ctx_w)           { ggml_free(model.ctx_w);                          model.ctx_w           = nullptr; }
        if (model.ctx_kv)          { ggml_free(model.ctx_kv);                         model.ctx_kv          = nullptr; }
        if (model.ctx_stack)       { ggml_free(model.ctx_stack);                      model.ctx_stack       = nullptr; }
        if (model.ctx_override)    { ggml_free(model.ctx_override);                   model.ctx_override    = nullptr; }
    }


    void bake_voice_conditioning() {
        if (opts.reference_audio.empty() && opts.voice_dir.empty()) {
            return;
        }

        const int n_threads = resolve_thread_count(opts.n_threads);

        bool have_se = false;
        bool have_ct = false;
        std::vector<float>   se_data;
        std::vector<int32_t> ct_data;

        if (!opts.voice_dir.empty()) {
            const std::string se_path = opts.voice_dir + "/speaker_emb.npy";
            const std::string ct_path = opts.voice_dir + "/cond_prompt_speech_tokens.npy";
            const std::string emb_path = opts.voice_dir + "/embedding.npy";
            const std::string pt_path  = opts.voice_dir + "/prompt_token.npy";
            const std::string pf_path  = opts.voice_dir + "/prompt_feat.npy";

            if (std::filesystem::exists(se_path)) {
                npy_array a = npy_load(se_path);
                se_data.assign((const float *) a.data.data(),
                               (const float *) a.data.data() + a.n_elements());
                have_se = true;
            }
            if (std::filesystem::exists(ct_path)) {
                npy_array a = npy_load(ct_path);
                ct_data.assign((const int32_t *) a.data.data(),
                               (const int32_t *) a.data.data() + a.n_elements());
                have_ct = true;
            }
            if (std::filesystem::exists(emb_path)) {
                npy_array a = npy_load(emb_path);
                s3gen_embedding.assign((const float *) a.data.data(),
                                      (const float *) a.data.data() + a.n_elements());
            }
            if (std::filesystem::exists(pt_path)) {
                npy_array a = npy_load(pt_path);
                s3gen_prompt_token.assign((const int32_t *) a.data.data(),
                                          (const int32_t *) a.data.data() + a.n_elements());
            }
            if (std::filesystem::exists(pf_path)) {
                npy_array a = npy_load(pf_path);
                s3gen_prompt_feat.assign((const float *) a.data.data(),
                                         (const float *) a.data.data() + a.n_elements());

                if (a.shape.size() >= 1) {
                    s3gen_prompt_feat_rows = (int) a.shape[0];
                }
            }
        }

        if (!have_se && !opts.reference_audio.empty()) {
            voice_encoder_weights vew;
            if (!voice_encoder_load(opts.t3_gguf_path, vew)) {
                throw std::runtime_error("Engine: VoiceEncoder weights unavailable");
            }
            std::vector<float> wav;
            int sr = 0;
            if (!wav_load(opts.reference_audio, wav, sr)) {
                throw std::runtime_error("Engine: failed to load reference_audio");
            }
            normalise_lufs(wav, sr, -27.0);
            if (sr != 16000) wav = resample_sinc(wav, sr, 16000);
            const size_t kVeMaxSamples = (size_t) 30 * 16000;
            if (wav.size() > kVeMaxSamples) wav.resize(kVeMaxSamples);
            if (!voice_encoder_embed(wav, vew, model.backend, se_data)) {
                throw std::runtime_error("Engine: VoiceEncoder forward failed");
            }
            have_se = true;
        }

        std::vector<int32_t> prompt_token_from_ref;
        if (!have_ct && !opts.reference_audio.empty()) {
            std::vector<int32_t> cond_tokens;
            if (!compute_speech_tokens_native(
                    opts.reference_audio, opts.s3gen_gguf_path,
                     model.hparams.cond_prompt_len,
                    prompt_token_from_ref, cond_tokens,
                    n_threads,  model.backend, opts.verbose)) {
                throw std::runtime_error("Engine: S3TokenizerV2 reference tokens failed");
            }
            ct_data = std::move(cond_tokens);
            have_ct = true;
            if (prompt_token_from_ref.empty() || ct_data.empty()) {
                throw std::runtime_error("Engine: S3TokenizerV2 returned empty conditioning");
            }
        }

        if (have_se) {
            if ((int64_t) se_data.size() != ggml_nelements(model.builtin_speaker_emb)) {
                throw std::runtime_error(
                    "Engine: speaker_emb size mismatch with builtin tensor");
            }
            ggml_backend_tensor_set(
                model.builtin_speaker_emb, se_data.data(), 0,
                ggml_nbytes(model.builtin_speaker_emb));
        }

        if (have_ct) {
            if ((int64_t) ct_data.size() == ggml_nelements(model.builtin_cond_prompt_tokens)) {
                ggml_backend_tensor_set(
                    model.builtin_cond_prompt_tokens, ct_data.data(), 0,
                    ggml_nbytes(model.builtin_cond_prompt_tokens));
            } else {
                ggml_init_params op = { ggml_tensor_overhead() * 2, nullptr, true };
                model.ctx_override = ggml_init(op);
                if (!model.ctx_override) {
                    throw std::runtime_error("Engine: ggml_init(ctx_override) failed");
                }
                ggml_tensor * new_ct = ggml_new_tensor_1d(
                    model.ctx_override, GGML_TYPE_I32, (int64_t) ct_data.size());
                ggml_set_name(new_ct,
                              "chatterbox/builtin/cond_prompt_speech_tokens_override");
                model.buffer_override = ggml_backend_alloc_ctx_tensors(
                    model.ctx_override, model.backend);
                if (!model.buffer_override) {
                    throw std::runtime_error("Engine: alloc override buffer failed");
                }
                ggml_backend_tensor_set(
                    new_ct, ct_data.data(), 0, ct_data.size() * sizeof(int32_t));
                model.builtin_cond_prompt_tokens = new_ct;
                model.hparams.cond_prompt_len = (int32_t) ct_data.size();
            }
        }

        if (!opts.reference_audio.empty()) {
            if (s3gen_prompt_feat.empty()) {
                int rows = 0;
                if (!compute_prompt_feat_native(
                        opts.reference_audio, opts.s3gen_gguf_path,
                        s3gen_prompt_feat, rows, opts.verbose)) {
                    throw std::runtime_error("Engine: prompt_feat from reference_audio failed");
                }
                if (rows < 1 || s3gen_prompt_feat.empty()) {
                    throw std::runtime_error("Engine: prompt_feat is empty");
                }
                s3gen_prompt_feat_rows = rows;
            }
            if (s3gen_embedding.empty()) {
                if (!compute_embedding_native(
                        opts.reference_audio, opts.s3gen_gguf_path,
                        s3gen_embedding,
                         model.backend, opts.verbose)) {
                    throw std::runtime_error("Engine: CAMPPlus embedding failed");
                }
                if (s3gen_embedding.empty()) {
                    throw std::runtime_error("Engine: CAMPPlus embedding is empty");
                }
            }
            if (s3gen_prompt_token.empty() && !prompt_token_from_ref.empty()) {
                s3gen_prompt_token = std::move(prompt_token_from_ref);
            }
            if (s3gen_prompt_token.empty()) {
                throw std::runtime_error("Engine: prompt_token unavailable");
            }
            if (s3gen_prompt_feat.empty()) {
                throw std::runtime_error("Engine: prompt_feat unavailable");
            }
            if (s3gen_embedding.empty()) {
                throw std::runtime_error("Engine: embedding unavailable");
            }
        }
    }

    std::vector<int32_t> run_t3(const std::string & text) {
        ensure_not_cancelled("before T3");
        const int n_threads = resolve_thread_count(opts.n_threads);
        std::mt19937 rng(opts.seed);
        chatterbox_sampling_params sp;
        sp.top_k          = opts.top_k;
        sp.top_p          = opts.top_p;
        sp.temp           = opts.temperature;
        sp.repeat_penalty = opts.repeat_penalty;
        sp.min_p          = opts.min_p;
        sp.cfg_weight     = opts.cfg_weight;

        std::vector<int32_t> text_tokens;
        if (model.hparams.variant == CHBX_VARIANT_MTL) {
            if (model.mtl_tokenizer_json.empty()) {
                throw std::runtime_error("Engine: MTL GGUF has no tokenizer json");
            }
            mtl_tokenizer tok;
            if (!tok.load_from_json(model.mtl_tokenizer_json)) {
                throw std::runtime_error("Engine: MTL tokenizer json failed to load");
            }
            text_tokens = tok.encode(text, opts.language);


            std::vector<int32_t> padded;
            padded.reserve(text_tokens.size() + 2);
            padded.push_back(model.hparams.start_text_token);
            padded.insert(padded.end(), text_tokens.begin(), text_tokens.end());
            padded.push_back(model.hparams.stop_text_token);
            text_tokens = std::move(padded);
        } else {
            if (model.tok_tokens.empty()) {
                throw std::runtime_error(
                    "Engine: T3 GGUF has no embedded tokenizer; "
                    "re-run scripts/convert-t3-turbo-to-gguf.py");
            }
            gpt2_bpe bpe;
            bpe.load_from_arrays(model.tok_tokens, model.tok_merges);
            text_tokens = bpe.tokenize(gpt2_bpe::punc_norm(text));
        }
        if (text_tokens.empty()) {
            throw std::runtime_error("Engine: text tokenised to empty sequence");
        }

        if (model.hparams.variant == CHBX_VARIANT_MTL) {
            std::vector<float> logits_c, logits_u;
            int prompt_len = 0;
            if (!eval_prompt_mtl(model, allocr, n_threads, text_tokens,
                                 opts.exaggeration, logits_c, logits_u, prompt_len)) {
                throw std::runtime_error("Engine: T3 MTL prompt eval failed");
            }
            int n_past = prompt_len;
            std::vector<int32_t> generated;
            generated.reserve((size_t) opts.n_predict + 1);
            int32_t current = sample_next_token_mtl(
                logits_c, logits_u, generated, sp, rng, model.hparams.stop_speech_token);
            generated.push_back(current);
            for (int i = 0; i < opts.n_predict; ++i) {
                if (cancel_flag.load(std::memory_order_relaxed)) {
                    throw std::runtime_error("Engine: synthesis cancelled during T3 decode");
                }
                if (current == model.hparams.stop_speech_token) break;
                if (n_past + 1 > model.hparams.n_ctx) break;
                if (!eval_step_mtl(model, allocr, n_threads, n_past, current, logits_c, logits_u)) {
                    throw std::runtime_error("Engine: T3 MTL step eval failed");
                }
                ++n_past;
                current = sample_next_token_mtl(
                    logits_c, logits_u, generated, sp, rng, model.hparams.stop_speech_token);
                generated.push_back(current);
            }
            const bool eos = !generated.empty() && generated.back() == model.hparams.stop_speech_token;
            if (!eos) throw std::runtime_error("Engine: T3 speech generation stopped without EOS");
            generated.pop_back();

            if (generated.size() > 1) generated.pop_back();
            return generated;
        }

        std::vector<float> logits;
        int prompt_len = 0;
        if (!eval_prompt(model, allocr, n_threads, text_tokens, logits, prompt_len)) {
            throw std::runtime_error("Engine: T3 prompt eval failed");
        }

        int n_past = prompt_len;
        std::vector<int32_t> generated;
        generated.reserve((size_t) opts.n_predict + 1);

        int32_t current = sample_next_token_ex(logits, generated, sp, rng);
        generated.push_back(current);

        for (int i = 0; i < opts.n_predict; ++i) {
            if (cancel_flag.load(std::memory_order_relaxed)) {
                throw std::runtime_error("Engine: synthesis cancelled during T3 decode");
            }
            if (current == model.hparams.stop_speech_token) break;
            if (n_past + 1 > model.hparams.n_ctx) break;
            if (!eval_step(model, allocr, n_threads, n_past, current, logits)) {
                throw std::runtime_error("Engine: T3 step eval failed");
            }
            ++n_past;
            current = sample_next_token_ex(logits, generated, sp, rng);
            generated.push_back(current);
        }

        {
            const bool eos = !generated.empty() && generated.back() == model.hparams.stop_speech_token;
            if (!eos) throw std::runtime_error("Engine: T3 speech generation stopped without EOS");
            generated.pop_back();
        }
        {
            const int32_t oov = model.hparams.start_speech_token;
            size_t w = 0;
            for (size_t r = 0; r < generated.size(); ++r)
                if (generated[r] >= 0 && generated[r] < oov) generated[w++] = generated[r];
            generated.resize(w);
        }
        return generated;
    }


    void fill_common_s3gen_opts(s3gen_synthesize_opts & sopts) {
        sopts.s3gen_gguf_path = opts.s3gen_gguf_path;
        sopts.out_wav_path    = "";
        sopts.seed            = opts.seed;
        sopts.n_threads       = resolve_thread_count(opts.n_threads);
        sopts.verbose         = opts.verbose;
        sopts.n_gpu_layers    = opts.n_gpu_layers;
        sopts.fastconv        = opts.fastconv;

        if (!s3gen_prompt_feat.empty()) {
            sopts.prompt_feat_override      = s3gen_prompt_feat;
            sopts.prompt_feat_rows_override = s3gen_prompt_feat_rows;
        }
        if (!s3gen_embedding.empty()) {
            sopts.embedding_override = s3gen_embedding;
        }
        if (!s3gen_prompt_token.empty()) {
            sopts.prompt_token_override = s3gen_prompt_token;
        }
    }

    SynthesisResult synthesize_batch(const std::vector<int32_t> & speech_tokens,
                                     SynthesisResult && partial) {
        ensure_not_cancelled("before S3Gen");
        s3gen_synthesize_opts sopts;
        fill_common_s3gen_opts(sopts);
        sopts.cfm_steps = opts.cfm_steps;

        SynthesisResult result = std::move(partial);
        sopts.pcm_out = &result.pcm;

        const auto s3_t0 = std::chrono::steady_clock::now();
        const int rc = s3gen_synthesize_to_wav(speech_tokens, sopts);
        const auto s3_t1 = std::chrono::steady_clock::now();
        if (rc != 0) {
            throw std::runtime_error("Engine: s3gen_synthesize_to_wav failed with code "
                                     + std::to_string(rc));
        }
        ensure_not_cancelled("after S3Gen");

        result.sample_rate   = 24000;
        result.t3_tokens     = (int) speech_tokens.size();
        result.audio_samples = (int) result.pcm.size();
        result.s3gen_ms      = std::chrono::duration<double, std::milli>(s3_t1 - s3_t0).count();
        return result;
    }


    SynthesisResult synthesize_streaming(
        const std::vector<int32_t> & speech_tokens,
        const StreamCallback & on_chunk,
        SynthesisResult && partial) {

        std::vector<int32_t> seg_toks = speech_tokens;
        for (int i = 0; i < kS3GenLookaheadTokens; ++i) {
            seg_toks.push_back(kS3GenSilenceToken);
        }
        const int total_n = (int) seg_toks.size();

        const int chunk_n       = opts.stream_chunk_tokens;
        const int first_chunk_n = opts.stream_first_chunk_tokens > 0
                                    ? opts.stream_first_chunk_tokens
                                    : chunk_n;

        std::vector<int> boundaries = {0};
        int cursor = std::min(first_chunk_n, total_n);
        boundaries.push_back(cursor);
        int adaptive_chunk = chunk_n;
        while (cursor < total_n) {
            int next = cursor + adaptive_chunk;
            if (next > total_n) next = total_n;
            int remaining = total_n - next;
            if (remaining > 0 && remaining < adaptive_chunk / 2) {
                next = total_n;
            } else if (adaptive_chunk < chunk_n * 2) {
                adaptive_chunk = std::min(adaptive_chunk + std::max(1, chunk_n / 4), chunk_n * 2);
            }
            cursor = next;
            boundaries.push_back(cursor);
        }


        const int min_tail = std::max(6, chunk_n / 3);
        if (boundaries.size() >= 3) {
            const int tail_len = boundaries.back() - boundaries[boundaries.size() - 2];
            if (tail_len < min_tail) boundaries.erase(boundaries.end() - 2);
        }

        std::vector<float> hift_cache_source;
        int prev_mels_emitted = 0;

        SynthesisResult result = std::move(partial);
        result.pcm.clear();

        const int n_chunks = (int) boundaries.size() - 1;
        double s3gen_ms_total = 0.0;

        for (int k = 1; k <= n_chunks; ++k) {
            if (cancel_flag.load(std::memory_order_relaxed)) {
                throw std::runtime_error("Engine: synthesis cancelled during streaming");
            }
            const int end              = boundaries[k];
            const bool is_last_in_seg  = (end == total_n);
            std::vector<int32_t> toks(seg_toks.begin(), seg_toks.begin() + end);

            s3gen_synthesize_opts copts;
            fill_common_s3gen_opts(copts);
            std::vector<float> chunk_pcm;
            copts.pcm_out                   = &chunk_pcm;
            copts.append_lookahead_silence  = false;
            copts.finalize                  = is_last_in_seg;
            copts.skip_mel_frames           = prev_mels_emitted;
            copts.apply_trim_fade           = true;
            copts.hift_cache_source         = hift_cache_source;
            std::vector<float> tail_out;
            copts.hift_source_tail_out      = &tail_out;
            copts.source_tail_samples       = 480;
            copts.cfm_steps                 = opts.stream_cfm_steps;

            const auto s3_t0 = std::chrono::steady_clock::now();
            const int rc = s3gen_synthesize_to_wav(toks, copts);
            const auto s3_t1 = std::chrono::steady_clock::now();
            if (rc != 0) {
                throw std::runtime_error(
                    "Engine: streaming chunk " + std::to_string(k) +
                    " failed with code " + std::to_string(rc));
            }
            s3gen_ms_total += std::chrono::duration<double, std::milli>(s3_t1 - s3_t0).count();

            on_chunk(chunk_pcm.data(), chunk_pcm.size(), k - 1, is_last_in_seg);

            result.pcm.insert(result.pcm.end(), chunk_pcm.begin(), chunk_pcm.end());
            hift_cache_source = std::move(tail_out);
            const size_t chunk_samples = chunk_pcm.size();
            prev_mels_emitted += (int)(chunk_samples / 480);
        }

        result.sample_rate   = 24000;
        result.t3_tokens     = (int) speech_tokens.size();
        result.audio_samples = (int) result.pcm.size();
        result.s3gen_ms      = s3gen_ms_total;
        return result;
    }

    SynthesisResult synthesize(const std::string & text,
                               const StreamCallback & on_chunk) {
        if (text.empty()) {
            throw std::runtime_error("Engine: text is empty");
        }
        cancel_flag.store(false, std::memory_order_relaxed);

        const auto t3_t0 = std::chrono::steady_clock::now();
        std::vector<int32_t> speech_tokens = run_t3(text);
        const auto t3_t1 = std::chrono::steady_clock::now();

        wait_for_preload(s3gen_preload_thread);

        SynthesisResult partial;
        partial.t3_ms = std::chrono::duration<double, std::milli>(t3_t1 - t3_t0).count();

        const bool use_streaming = on_chunk && opts.stream_chunk_tokens > 0;
        return use_streaming
            ? synthesize_streaming(speech_tokens, on_chunk, std::move(partial))
            : synthesize_batch(speech_tokens, std::move(partial));
    }

    SynthesisResult synthesize(const std::string & text) {
        return synthesize(text, StreamCallback{});
    }

    std::vector<SynthesisResult> do_synthesize_pieces(
        const std::vector<std::string> & texts,
        const PieceCallback & on_piece_chunk) {
        cancel_flag.store(false, std::memory_order_relaxed);
        std::vector<SynthesisResult> results;
        results.reserve(texts.size());
        for (size_t i = 0; i < texts.size(); ++i) {
            ensure_not_cancelled("before piece");
            if (texts[i].empty()) {
                SynthesisResult r;
                r.sample_rate = 24000;
                results.push_back(r);
                if (on_piece_chunk) on_piece_chunk((int) i, nullptr, 0, 0, true);
                continue;
            }
            const auto t3_t0 = std::chrono::steady_clock::now();
            std::vector<int32_t> speech_tokens = run_t3(texts[i]);
            const auto t3_t1 = std::chrono::steady_clock::now();
            ensure_not_cancelled("after T3");
            double t3_ms = std::chrono::duration<double, std::milli>(t3_t1 - t3_t0).count();
            last_t3_ms = t3_ms;
            last_t3_tokens = (int) speech_tokens.size();
            wait_for_preload(s3gen_preload_thread);
            ensure_not_cancelled("before vocoder");
            SynthesisResult partial;
            partial.t3_ms = t3_ms;
            const bool use_streaming = on_piece_chunk && opts.stream_chunk_tokens > 0;
            SynthesisResult r = use_streaming
                ? synthesize_streaming(speech_tokens,
                    [i, on_piece_chunk](const float * pcm, std::size_t n, int ci, bool last) {
                        if (on_piece_chunk) on_piece_chunk((int) i, pcm, n, ci, last);
                    },
                    std::move(partial))
                : synthesize_batch(speech_tokens, std::move(partial));
            ensure_not_cancelled("before audio emit");
            if (!on_piece_chunk || opts.stream_chunk_tokens <= 0) {
                if (on_piece_chunk) on_piece_chunk((int) i, r.pcm.data(), r.pcm.size(), 0, true);
            }
            results.push_back(std::move(r));
        }
        return results;
    }

    bool rebake_voice_from_reference(const std::string & wav_path) {
        if (wav_path.empty() || !std::filesystem::exists(wav_path)) return false;
        if (!validate_reference_audio(wav_path)) return false;
        std::vector<float>   se_data;
        std::vector<int32_t> ct_data;
        std::vector<int32_t> prompt_token_from_ref;
        std::vector<float>   pf_data;
        int                  pf_rows = 0;
        std::vector<float>   emb_data;
        const int n_threads = resolve_thread_count(opts.n_threads);
        voice_encoder_weights vew;
        if (!voice_encoder_load(opts.t3_gguf_path, vew)) return false;
        std::vector<float> wav;
        int sr = 0;
        if (!wav_load(wav_path, wav, sr)) return false;
        normalise_lufs(wav, sr, -27.0);
        if (sr != 16000) wav = resample_sinc(wav, sr, 16000);
        const size_t kVeMaxSamples = (size_t) 30 * 16000;
        if (wav.size() > kVeMaxSamples) wav.resize(kVeMaxSamples);
        if (!voice_encoder_embed(wav, vew, model.backend, se_data)) return false;
        if (!compute_speech_tokens_native(
                wav_path, opts.s3gen_gguf_path,
                 model.hparams.cond_prompt_len,
                prompt_token_from_ref, ct_data,
                n_threads,  model.backend, opts.verbose)) return false;
        if (!compute_prompt_feat_native(
                wav_path, opts.s3gen_gguf_path,
                pf_data, pf_rows, opts.verbose)) return false;
        if (!compute_embedding_native(
                wav_path, opts.s3gen_gguf_path,
                emb_data,
                 model.backend, opts.verbose)) return false;
        if ((int64_t) se_data.size() != ggml_nelements(model.builtin_speaker_emb)) return false;
        ggml_backend_tensor_set(model.builtin_speaker_emb, se_data.data(), 0,
            ggml_nbytes(model.builtin_speaker_emb));
        if (!ct_data.empty()) {
            if ((int64_t) ct_data.size() == ggml_nelements(model.builtin_cond_prompt_tokens)) {
                ggml_backend_tensor_set(model.builtin_cond_prompt_tokens, ct_data.data(), 0,
                    ggml_nbytes(model.builtin_cond_prompt_tokens));
            } else {
                if (model.ctx_override) { ggml_free(model.ctx_override); model.ctx_override = nullptr; }
                if (model.buffer_override) { ggml_backend_buffer_free(model.buffer_override); model.buffer_override = nullptr; }
                ggml_init_params op = { ggml_tensor_overhead() * 2, nullptr, true };
                model.ctx_override = ggml_init(op);
                if (!model.ctx_override) return false;
                ggml_tensor * new_ct = ggml_new_tensor_1d(
                    model.ctx_override, GGML_TYPE_I32, (int64_t) ct_data.size());
                ggml_set_name(new_ct, "chatterbox/builtin/cond_prompt_speech_tokens_override");
                model.buffer_override = ggml_backend_alloc_ctx_tensors(
                    model.ctx_override, model.backend);
                if (!model.buffer_override) return false;
                ggml_backend_tensor_set(new_ct, ct_data.data(), 0, ct_data.size() * sizeof(int32_t));
                model.builtin_cond_prompt_tokens = new_ct;
                model.hparams.cond_prompt_len = (int32_t) ct_data.size();
            }
        }
        s3gen_prompt_feat = std::move(pf_data);
        s3gen_prompt_feat_rows = pf_rows;
        s3gen_embedding = std::move(emb_data);
        s3gen_prompt_token = std::move(prompt_token_from_ref);
        opts.reference_audio = wav_path;
        return true;
    }

    bool rebake_voice_from_dir(const std::string & dir) {
        if (dir.empty() || !std::filesystem::is_directory(dir)) return false;
        const std::string se_path = dir + "/speaker_emb.npy";
        const std::string ct_path = dir + "/cond_prompt_speech_tokens.npy";
        const std::string emb_path = dir + "/embedding.npy";
        const std::string pt_path  = dir + "/prompt_token.npy";
        const std::string pf_path  = dir + "/prompt_feat.npy";
        std::vector<float>   se_data, pf_data, emb_data;
        std::vector<int32_t> ct_data, pt_data;
        int pf_rows = 0;
        npy_array a;
        if (std::filesystem::exists(se_path)) { a = npy_load(se_path); se_data.assign((const float*) a.data.data(), (const float*) a.data.data() + a.n_elements()); }
        if (std::filesystem::exists(ct_path)) { a = npy_load(ct_path); ct_data.assign((const int32_t*) a.data.data(), (const int32_t*) a.data.data() + a.n_elements()); }
        if (std::filesystem::exists(emb_path)) { a = npy_load(emb_path); emb_data.assign((const float*) a.data.data(), (const float*) a.data.data() + a.n_elements()); }
        if (std::filesystem::exists(pt_path)) { a = npy_load(pt_path); pt_data.assign((const int32_t*) a.data.data(), (const int32_t*) a.data.data() + a.n_elements()); }
        if (std::filesystem::exists(pf_path)) { a = npy_load(pf_path); pf_data.assign((const float*) a.data.data(), (const float*) a.data.data() + a.n_elements()); if (a.shape.size() >= 1) pf_rows = (int) a.shape[0]; }
        if (!se_data.empty() && (int64_t) se_data.size() == ggml_nelements(model.builtin_speaker_emb)) {
            ggml_backend_tensor_set(model.builtin_speaker_emb, se_data.data(), 0, ggml_nbytes(model.builtin_speaker_emb));
        }
        if (!ct_data.empty() && (int64_t) ct_data.size() == ggml_nelements(model.builtin_cond_prompt_tokens)) {
            ggml_backend_tensor_set(model.builtin_cond_prompt_tokens, ct_data.data(), 0, ggml_nbytes(model.builtin_cond_prompt_tokens));
        } else if (!ct_data.empty()) {
            if (model.ctx_override) { ggml_free(model.ctx_override); model.ctx_override = nullptr; }
            if (model.buffer_override) { ggml_backend_buffer_free(model.buffer_override); model.buffer_override = nullptr; }
            ggml_init_params op = { ggml_tensor_overhead() * 2, nullptr, true };
            model.ctx_override = ggml_init(op);
            if (!model.ctx_override) return false;
            ggml_tensor * new_ct = ggml_new_tensor_1d(model.ctx_override, GGML_TYPE_I32, (int64_t) ct_data.size());
            ggml_set_name(new_ct, "chatterbox/builtin/cond_prompt_speech_tokens_override");
            model.buffer_override = ggml_backend_alloc_ctx_tensors(model.ctx_override, model.backend);
            if (!model.buffer_override) return false;
            ggml_backend_tensor_set(new_ct, ct_data.data(), 0, ct_data.size() * sizeof(int32_t));
            model.builtin_cond_prompt_tokens = new_ct;
            model.hparams.cond_prompt_len = (int32_t) ct_data.size();
        }
        s3gen_prompt_feat = std::move(pf_data);
        s3gen_prompt_feat_rows = pf_rows;
        s3gen_embedding = std::move(emb_data);
        s3gen_prompt_token = std::move(pt_data);
        opts.voice_dir = dir;
        return true;
    }
};

Engine::Engine(const EngineOptions & opts)
    : pimpl_(std::make_unique<Impl>(opts)) {}

Engine::~Engine() = default;
Engine::Engine(Engine &&) noexcept            = default;
Engine & Engine::operator=(Engine &&) noexcept = default;

SynthesisResult Engine::synthesize(const std::string & text) {
    return pimpl_->synthesize(text);
}

SynthesisResult Engine::synthesize(const std::string & text,
                                   const StreamCallback & on_chunk) {
    return pimpl_->synthesize(text, on_chunk);
}

std::vector<Engine::PieceResult> Engine::synthesize_pieces(
    const std::vector<std::string> & texts,
    const PieceCallback & on_piece_chunk) {
    if (!pimpl_) return {};
    auto raw = pimpl_->do_synthesize_pieces(texts, on_piece_chunk);
    std::vector<PieceResult> out;
    out.reserve(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        PieceResult pr;
        pr.piece_index = (int) i;
        pr.pcm = std::move(raw[i].pcm);
        pr.sample_rate = raw[i].sample_rate;
        pr.t3_ms = raw[i].t3_ms;
        pr.s3gen_ms = raw[i].s3gen_ms;
        pr.t3_tokens = raw[i].t3_tokens;
        pr.audio_samples = raw[i].audio_samples;
        out.push_back(std::move(pr));
    }
    return out;
}

bool Engine::set_reference_audio(const std::string & path) {
    if (!pimpl_) return false;
    return pimpl_->rebake_voice_from_reference(path);
}

bool Engine::set_voice_dir(const std::string & dir) {
    if (!pimpl_) return false;
    return pimpl_->rebake_voice_from_dir(dir);
}

Engine::TokenTiming Engine::timing_hint() const {
    TokenTiming t;
    if (pimpl_) {
        t.total_speech_tokens = pimpl_->last_t3_tokens;
        t.ms_per_token_estimate = pimpl_->last_t3_tokens > 0 ? pimpl_->last_t3_ms / pimpl_->last_t3_tokens : 0.0;
    }
    return t;
}

void Engine::cancel() {
    if (pimpl_) pimpl_->cancel_flag.store(true, std::memory_order_relaxed);
}

const EngineOptions & Engine::options() const {
    return pimpl_->opts;
}

}
