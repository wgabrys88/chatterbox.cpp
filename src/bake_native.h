#pragma once
#include <string>
#include <vector>
typedef struct ggml_backend * ggml_backend_t;
namespace tts_cpp::chatterbox::detail {
void compute_prompt_feat_native(const std::string &, const std::string &, std::vector<float> &, int &, ggml_backend_t);
void compute_embedding_native(const std::string &, const std::string &, std::vector<float> &, ggml_backend_t);
void compute_speech_tokens_native(const std::string &, const std::string &, int, std::vector<int32_t> &, std::vector<int32_t> &, ggml_backend_t);
}
