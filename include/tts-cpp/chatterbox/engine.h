#pragma once
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
namespace tts_cpp::chatterbox {
struct EngineOptions {
    std::string t3_gguf_path;
    std::string s3gen_gguf_path;
    std::string reference_audio;
};
using PcmCallback = std::function<void(const float*, std::size_t)>;
class Engine {
public:
    explicit Engine(const EngineOptions&);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;
    void synthesize(const std::string&, const PcmCallback&);
private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};
}
