#pragma once
#include <string>
#include <vector>
struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;
constexpr int kSamplesPerToken = 960;
struct s3gen_synthesize_opts {
    std::string s3gen_gguf_path;
    std::vector<float> prompt_feat;
    int prompt_rows = 0;
    std::vector<float> embedding;
    std::vector<int32_t> prompt_token;
};
std::vector<float> s3gen_synthesize(const std::vector<int32_t>&, const s3gen_synthesize_opts&);
void s3gen_preload(const std::string&, ggml_backend_t);
void s3gen_unload();
