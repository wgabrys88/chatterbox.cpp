#pragma once
#include <cstdint>
#include <string>
#include <vector>
struct ggml_tensor;
typedef struct ggml_backend * ggml_backend_t;
struct s3tokv2_block {
    std::vector<float> attn_ln_w;
    std::vector<float> attn_ln_b;
    std::vector<float> q_w, q_b;
    std::vector<float> k_w;
    std::vector<float> v_w, v_b;
    std::vector<float> out_w, out_b;
    std::vector<float> fsmn_w;
    std::vector<float> mlp_ln_w;
    std::vector<float> mlp_ln_b;
    std::vector<float> mlp0_w, mlp0_b;
    std::vector<float> mlp2_w, mlp2_b;
};
struct s3tokv2_weights {
    int n_mels       = 128;
    int n_state      = 1280;
    int n_head       = 20;
    int n_layer      = 6;
    int head_dim     = 64;
    int mlp_ratio    = 4;
    int fsmn_kernel  = 31;
    int fsq_levels   = 3;
    int fsq_dim      = 8;
    int codebook_size= 6561;
    int conv_stride  = 2;
    int n_fft        = 400;
    int hop          = 160;
    int sample_rate  = 16000;
    float rope_theta = 10000.0f;
    int rope_max_pos = 2048;
    std::vector<float> mel_fb;
    std::vector<float> conv1_w;
    std::vector<float> conv1_b;
    std::vector<float> conv2_w;
    std::vector<float> conv2_b;
    std::vector<s3tokv2_block> blocks;
    std::vector<float> fsq_w;
    std::vector<float> fsq_b;
};
bool s3tokv2_load(const std::string & s3gen_gguf_path,
                  s3tokv2_weights & out);
std::vector<float> s3tokv2_log_mel(const std::vector<float> & wav_16k,
                                   const s3tokv2_weights & w,
                                   int & out_T);
bool s3tokv2_tokenize(const std::vector<float> & wav_16k,
                      const s3tokv2_weights & w,
                      int max_tokens,
                      std::vector<int32_t> & out_tokens,
                      int n_threads = 0,
                      ggml_backend_t backend = nullptr);
