#pragma once
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>
constexpr int kSpeechHistoryTokens = 25;
constexpr int kSpeechLookaheadTokens = 3;
constexpr int kSamplesPerToken = 960;
struct s3gen_piece_state {
    int token_end = 0;
    // Channel-major mel and sample-major excitation for the preceding history tokens.
    std::vector<float> mel, source, pending_pcm;
    std::vector<double> phase;
    double encoder_ms = 0, cfm_ms = 0, f0_ms = 0, stft_ms = 0, hift_ms = 0, pipeline_ms = 0;
    int samples = 0, prompt_tokens = 0, speech_tokens = 0;
};
struct s3gen_synthesize_opts {
    std::string s3gen_gguf_path;
    std::vector<float>* pcm_out = nullptr;
    std::vector<float> prompt_feat;
    int prompt_rows = 0;
    std::vector<float> embedding;
    std::vector<int32_t> prompt_token;
    int seed = 42;
    int n_threads = 4;
    int n_gpu_layers = 99;
    int cfm_steps = 2;
    bool fastconv = true;
    const std::atomic<bool>* cancel = nullptr;
    bool final = true;
    bool first_piece = true;
    int chunk_id = 0;
    int token_start = 0;
    int token_end = 0;
    s3gen_piece_state* state = nullptr;
};
void s3gen_synthesize(const std::vector<int32_t>&, const s3gen_synthesize_opts&);
void s3gen_preload(const std::string&, int, bool);
void s3gen_unload();
void s3gen_vk_overlap_counters(unsigned long long * wait_us, unsigned long long * submit_n,
                               unsigned long long * barrier_n, int reset);
