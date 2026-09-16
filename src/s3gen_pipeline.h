#pragma once
#include <cstdint>
#include <string>
#include <vector>
struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;
constexpr int kSamplesPerToken = 960;
constexpr float kVoicedThreshold = 10.0f;
// One adult speech breath in codec time. Tank C is this many tokens of
// THIS voice's prompt effort. Drain per token is fuel * pitch/prompt_f0.
constexpr float kBreathSeconds = 4.0f;
constexpr int kBreathTokens = 100; // 4.0 / 0.040
struct S3Gauge {
    std::vector<float> codebook_norm;
    std::vector<float> fuel;
    std::vector<float> f0;
    std::vector<int32_t> voiced;
    int n_prompt = 0;
    float prompt_fuel_mean = 0.0f;
    float prompt_f0_mean = 0.0f;
    float breath_capacity = 0.0f;
};
S3Gauge s3gen_gauge(const std::vector<int32_t>&);
std::vector<float> s3gen_synthesize(const std::vector<int32_t>&);
void s3gen_preload(const std::string&, ggml_backend_t);
void s3gen_unload();
