#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>
namespace tts_cpp::chatterbox {
struct EngineOptions {
    std::string t3_gguf_path;
    std::string s3gen_gguf_path;
    std::string reference_audio;
};
struct SynthesisPiece {
    std::uint32_t id = 0;
    std::string text;
};
using PieceCallback = std::function<void(int, const float*, std::size_t, int, bool)>;
class Engine {
public:
    explicit Engine(const EngineOptions&);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;
    void synthesize_pieces_streaming(const std::vector<SynthesisPiece>&, const PieceCallback&);
    void warm_up();
private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};
}
