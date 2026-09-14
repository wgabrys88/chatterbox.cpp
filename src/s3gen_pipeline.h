#pragma once
#include <cstdint>
#include <string>
#include <vector>
struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;
constexpr int kSamplesPerToken = 960;
#if defined(TTS_FAMILY_NANO)
void s3gen_synthesize_stream(
    const std::vector<int32_t>& speech_tokens,
    bool finalize,
    const std::vector<float>& source_cache,
    std::vector<float>& wav,
    std::vector<float>& source);
#else
std::vector<float> s3gen_synthesize(const std::vector<int32_t>&);
#endif
void s3gen_preload(const std::string&, ggml_backend_t);
void s3gen_unload();
