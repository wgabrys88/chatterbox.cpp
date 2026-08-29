#pragma once
#include <cstdint>
#include <string>
#include <vector>
struct ggml_context;
struct ggml_tensor;
typedef struct ggml_backend * ggml_backend_t;
struct voice_encoder_lstm_layer {
    std::vector<float> w_ih;
    std::vector<float> w_hh;
    std::vector<float> b_ih;
    std::vector<float> b_hh;
    int H = 0;
    int I = 0;
};
struct voice_encoder_weights {
    int n_layers  = 3;
    int n_mels    = 40;
    int hidden    = 256;
    int embedding = 256;
    std::vector<voice_encoder_lstm_layer> lstm;
    std::vector<float> proj_w;
    std::vector<float> proj_b;
    std::vector<float> mel_fb;
    int   partial_frames = 160;
    float overlap        = 0.5f;
    float rate           = 1.3f;
    float min_coverage   = 0.8f;
};
bool voice_encoder_load(const std::string & t3_gguf_path,
                        voice_encoder_weights & out);
bool voice_encoder_embed(const std::vector<float> & wav_16k,
                         const voice_encoder_weights & w,
                         ggml_backend_t backend,
                         std::vector<float> & out);
