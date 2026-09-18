#pragma once
#include <cstdint>
#include <string>
#include <vector>
struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;
namespace tts_cpp::chatterbox { class ExecutionTrace; }
constexpr int kSamplesPerToken = 960;
std::vector<float> s3gen_synthesize(const std::vector<int32_t>&, tts_cpp::chatterbox::ExecutionTrace* = nullptr);
void s3gen_preload(const std::string&, ggml_backend_t);
void s3gen_unload();
