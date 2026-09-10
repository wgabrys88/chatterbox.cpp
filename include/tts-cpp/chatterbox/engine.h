#pragma once
#include <memory>
#include <string>
#include <vector>
namespace tts_cpp::chatterbox {
struct EngineOptions {
    std::string t3_gguf_path, s3gen_gguf_path, reference_audio;
};
class Engine {
public:
    explicit Engine(const EngineOptions&);
    ~Engine();
    std::vector<float> synthesize(const std::string&);
private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};
}
