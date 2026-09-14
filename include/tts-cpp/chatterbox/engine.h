#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
namespace tts_cpp::chatterbox {
struct EngineOptions {
    std::string t3_gguf_path, s3gen_gguf_path, language_id;
};
class Engine {
public:
    explicit Engine(const EngineOptions&);
    ~Engine();
#if defined(TTS_FAMILY_NANO)
    using AudioCallback = void (*)(const float *, std::size_t, void *);
    void synthesize(const std::string&, AudioCallback, void *);
#else
    std::vector<float> synthesize(const std::string&);
#endif
private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};
}
