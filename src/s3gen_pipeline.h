#pragma once
#include <string>
#include <vector>
struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;
constexpr int kSamplesPerToken = 960;
std::vector<float> s3gen_synthesize(const std::vector<int32_t>&);
void s3gen_preload(const std::string&, ggml_backend_t);
void s3gen_unload();
