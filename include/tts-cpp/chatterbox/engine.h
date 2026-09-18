#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
namespace tts_cpp::chatterbox {
class ExecutionTrace;
struct EngineOptions {
    std::string t3_gguf_path, s3gen_gguf_path, language_id;
    std::string tokenizer_python, tokenizer_script, tokenizer_source, tokenizer_tts_source, tokenizer_json, cangjie_json, dicta_model;
};
struct SynthesizeStats {
    int predicted_count = 0;
    int dropped_count = 0;
    int eos = 0;
    int n_past = 0;
    int units = 0;
    int text_tokens = 0;
    int max_unit_predicted = 0;
};
class Engine {
public:
    explicit Engine(const EngineOptions&, ExecutionTrace* = nullptr);
    ~Engine();
    void synthesize(const std::string&, std::vector<float>& pcm, SynthesizeStats* = nullptr, ExecutionTrace* = nullptr);
private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};
}
