#pragma once


#include <cstdint>
#include <string>
#include <vector>

namespace tts_cpp::chatterbox {

inline constexpr int kS3GenSilenceToken    = 4299;
inline constexpr int kS3GenLookaheadTokens = 3;

}

struct s3gen_synthesize_opts {
    std::string s3gen_gguf_path;


    std::string out_wav_path;


    std::vector<float> * pcm_out = nullptr;


    std::string ref_dir;


    std::vector<float> prompt_feat_override;
    int prompt_feat_rows_override = 0;


    std::vector<float> embedding_override;


    std::vector<int32_t> prompt_token_override;

    int  seed      = 42;
    int  n_threads = 0;
    int  sr        = 24000;
    bool debug     = false;
    bool verbose   = false;


    int  n_gpu_layers = 0;


    bool finalize                  = true;
    bool append_lookahead_silence  = true;
    int  skip_mel_frames           = 0;


    std::string dump_mel_path;


    std::vector<float> cfm_z0_override;


    std::vector<float>   hift_cache_source;
    bool                 apply_trim_fade       = true;
    std::vector<float> * hift_source_tail_out  = nullptr;
    int                  source_tail_samples   = 480;


    int                  cfm_steps             = 0;


    bool                 cfm_f16_kv_attn       = false;
};


int s3gen_synthesize_to_wav(
    const std::vector<int32_t> & speech_tokens,
    const s3gen_synthesize_opts & opts);


int s3gen_preload(const std::string & s3gen_gguf_path, int n_gpu_layers);


void s3gen_unload();
