#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
namespace tts_cpp::chatterbox {
struct EngineOptions {
    std::string t3_gguf_path, s3gen_gguf_path, language_id;
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
    explicit Engine(const EngineOptions&);
    ~Engine();
    // Every family: text is split into utterances (see --split-tokens), each
    // synthesized on a fresh T3 KV, PCM delivered through the callback in
    // order. Nano flushes every STREAM_TOKENS; Turbo/V3 once per utterance.
    using AudioCallback = void (*)(const float *, std::size_t, void *);
    void synthesize(const std::string&, AudioCallback, void *, SynthesizeStats * = nullptr);
private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};
}
