#pragma once
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>
namespace tts_cpp::chatterbox {
struct EngineOptions {
    std::string t3_gguf_path;
    std::string s3gen_gguf_path;
    std::string reference_audio;
    std::string language;
    int n_gpu_layers = 0;
    int n_threads = 0;
    int seed = 42;
    int n_predict = 1000;
    int n_ctx = 0;
    int top_k = 1000;
    float top_p = .95f;
    float min_p = .05f;
    float temperature = .8f;
    float repeat_penalty = 1.2f;
    float cfg_weight = .5f;
    float exaggeration = .5f;
    int cfm_steps = 0;
    bool fastconv = false;
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
    void begin_synthesis();
    void synthesize_pieces(const std::vector<std::string>&, const PieceCallback&);
    void synthesize_pieces_streaming(const std::vector<std::string>&, const PieceCallback&);
    void warm_up();
    void cancel();
private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};
}
