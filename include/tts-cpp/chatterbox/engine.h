#pragma once


#include <cstdint>
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
    std::string voice_dir;


    int n_gpu_layers = 0;


    int n_threads = 0;


    int seed = 42;


    int n_predict = 1000;


    int n_ctx = 0;


    int   top_k          = 1000;
    float top_p          = 0.95f;
    float temperature    = 0.8f;
    float repeat_penalty = 1.2f;

    std::string language;
    float exaggeration   = 0.5f;
    float cfg_weight     = 0.5f;
    float min_p          = 0.05f;


    int cfm_steps = 0;


    int stream_chunk_tokens       = 0;
    int stream_first_chunk_tokens = 0;
    int stream_cfm_steps          = 0;


    bool fastconv = false;
    bool verbose = false;
};


using StreamCallback = std::function<void(
    const float * pcm, std::size_t samples, int chunk_index, bool is_last)>;

using PieceCallback = std::function<void(
    int piece_index, const float * pcm, std::size_t samples, int chunk_index, bool is_last)>;

struct SynthesisResult {

    std::vector<float> pcm;
    int sample_rate = 24000;


    double t3_ms         = 0.0;
    double s3gen_ms      = 0.0;
    int    t3_tokens     = 0;
    int    audio_samples = 0;
};

class Engine {
public:


    explicit Engine(const EngineOptions & opts);


    ~Engine();

    Engine(const Engine &)            = delete;
    Engine & operator=(const Engine &) = delete;

    Engine(Engine &&) noexcept;
    Engine & operator=(Engine &&) noexcept;


    SynthesisResult synthesize(const std::string & text);


    SynthesisResult synthesize(const std::string & text,
                               const StreamCallback & on_chunk);


    struct PieceResult {
        int piece_index = -1;
        std::vector<float> pcm;
        int sample_rate = 24000;
        double t3_ms = 0.0;
        double s3gen_ms = 0.0;
        int    t3_tokens = 0;
        int    audio_samples = 0;
    };

    std::vector<PieceResult> synthesize_pieces(
        const std::vector<std::string> & texts,
        const PieceCallback & on_piece_chunk);

    bool set_reference_audio(const std::string & path);
    bool set_voice_dir(const std::string & dir);

    struct TokenTiming {
        int    total_speech_tokens = 0;
        double ms_per_token_estimate = 0.0;
    };
    TokenTiming timing_hint() const;

    void cancel();


    const EngineOptions & options() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

}
