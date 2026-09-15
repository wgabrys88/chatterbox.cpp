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
};
class Engine {
public:
    explicit Engine(const EngineOptions&);
    ~Engine();
#if defined(TTS_FAMILY_NANO)
    using AudioCallback = void (*)(const float *, std::size_t, void *);
    void synthesize(const std::string&, AudioCallback, void *, SynthesizeStats * = nullptr);
#else
    std::vector<float> synthesize(const std::string&, SynthesizeStats * = nullptr);
#endif
private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};
}
