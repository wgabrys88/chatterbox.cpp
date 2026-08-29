#pragma once
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>
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
    int skip_mel_frames = 0;
    int chunk_id = 0;
    std::vector<float> hift_cache_source;
    std::vector<float>* hift_source_tail = nullptr;
};
void s3gen_synthesize(const std::vector<int32_t>&, const s3gen_synthesize_opts&);
void s3gen_preload(const std::string&, int, bool);
void s3gen_unload();
