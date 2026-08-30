#pragma once
#include <atomic>
#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <vector>
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
bool wav_load(const std::string & path,
              std::vector<float> & out_samples,
              int & out_sr);
std::vector<float> resample_sinc(const std::vector<float> & in, int sr_in, int sr_out);
double measure_lufs(const std::vector<float> & wav, int sr);
void normalise_lufs(std::vector<float> & wav, int sr);
std::vector<float> mel_extract_24k_80(const std::vector<float> & wav_24k,
                                      const std::vector<float> & mel_filterbank);
std::vector<float> mel_extract_16k_40(const std::vector<float> & wav_16k,
                                      const std::vector<float> & mel_filterbank);
std::vector<float> fbank_kaldi_80(const std::vector<float> & wav_16k,
                                  const std::vector<float> & mel_filterbank);

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

struct campplus_conv {
    std::vector<float> w;
    std::vector<float> b;
    int C_out = 0, C_in = 0, k = 0;
    int kH = 0, kW = 0;
    int stride_h = 1, stride_w = 1;
    int pad_h = 0, pad_w = 0;
    int dilation_h = 1, dilation_w = 1;
    bool is_2d = false;
};
struct campplus_bn {
    std::vector<float> scale;
    std::vector<float> shift;
};
struct campplus_res_block {
    campplus_conv conv1;
    campplus_bn   bn1;
    campplus_conv conv2;
    campplus_bn   bn2;
    campplus_conv shortcut_conv;
    campplus_bn   shortcut_bn;
    int stride_h = 1;
};
struct campplus_fcm {
    campplus_conv conv1;
    campplus_bn   bn1;
    std::vector<campplus_res_block> layer1;
    std::vector<campplus_res_block> layer2;
    campplus_conv conv2;
    campplus_bn   bn2;
};
struct campplus_cam_dense_tdnn_layer {
    campplus_bn   bn1;
    campplus_conv linear1;
    campplus_bn   bn2;
    campplus_conv cam_linear_local;
    campplus_conv cam_linear1;
    campplus_conv cam_linear2;
};
struct campplus_cam_block {
    int num_layers = 0;
    int kernel_size = 3;
    int dilation = 1;
    std::vector<campplus_cam_dense_tdnn_layer> layers;
};
struct campplus_transit {
    campplus_bn   bn;
    campplus_conv linear;
};
struct campplus_weights {
    int feat_dim      = 80;
    int embedding_size= 192;
    int seg_pool_len  = 100;
    int sample_rate   = 16000;
    campplus_fcm head;
    campplus_conv tdnn_linear;
    campplus_bn   tdnn_bn;
    campplus_cam_block block1;
    campplus_transit   transit1;
    campplus_cam_block block2;
    campplus_transit   transit2;
    campplus_cam_block block3;
    campplus_transit   transit3;
    campplus_bn   out_nonlinear_bn;
    campplus_conv dense_linear;
    campplus_bn   dense_bn;
};
bool campplus_load(const std::string & s3gen_gguf_path,
                   campplus_weights & out);
bool campplus_embed(const std::vector<float>& fbank_t_by_c, int T,
                    const campplus_weights& w, std::vector<float>& out);

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
                      int n_threads,
                      ggml_backend_t backend);

struct s3gen_synthesize_opts {
    std::string s3gen_gguf_path;
    std::vector<float>* pcm_out = nullptr;
    std::vector<float> prompt_feat;
    int prompt_rows = 0;
    std::vector<float> embedding;
    std::vector<int32_t> prompt_token;
    int seed = 42;
    int n_threads = 4;
    int n_gpu_layers = 99;
    int cfm_steps = 2;
    bool fastconv = true;
    const std::atomic<bool>* cancel = nullptr;
    bool final = true;
    int skip_mel_frames = 0;
    int chunk_id = 0;
    std::vector<float> hift_cache_source;
    std::vector<float>* hift_source_tail = nullptr;
};
void s3gen_synthesize(const std::vector<int32_t>&, const s3gen_synthesize_opts&);
void s3gen_preload(const std::string&, int, bool);
void s3gen_unload();

namespace tts_cpp::chatterbox::detail {
constexpr int CHBX_MAX_NODES = 8192;
constexpr const char * KEY_VARIANT           = "chatterbox.variant";
constexpr const char * KEY_TEXT_VOCAB_SIZE   = "chatterbox.text_vocab_size";
constexpr const char * KEY_SPEECH_VOCAB_SIZE = "chatterbox.speech_vocab_size";
constexpr const char * KEY_START_SPEECH      = "chatterbox.start_speech_token";
constexpr const char * KEY_STOP_SPEECH       = "chatterbox.stop_speech_token";
constexpr const char * KEY_SPEAKER_EMBED     = "chatterbox.speaker_embed_size";
constexpr const char * KEY_LAYER_NORM_EPS    = "chatterbox.layer_norm_eps";
constexpr const char * KEY_COND_PROMPT_LEN   = "chatterbox.cond_prompt_length";
constexpr const char * KEY_N_CTX             = "chatterbox.n_ctx";
constexpr const char * KEY_N_EMBD            = "chatterbox.n_embd";
constexpr const char * KEY_N_HEAD            = "chatterbox.n_head";
constexpr const char * KEY_N_LAYER           = "chatterbox.n_layer";
constexpr const char * KEY_N_KV_HEAD         = "chatterbox.n_kv_head";
constexpr const char * KEY_HEAD_DIM          = "chatterbox.head_dim";
constexpr const char * KEY_INTERMEDIATE_SIZE = "chatterbox.intermediate_size";
constexpr const char * KEY_RMS_EPS           = "chatterbox.rms_norm_eps";
constexpr const char * KEY_ROPE_THETA        = "chatterbox.rope_theta";
constexpr const char * KEY_ROPE_SCALING_TYPE = "chatterbox.rope.scaling_type";
constexpr const char * KEY_ROPE_SCALING_FACTOR = "chatterbox.rope.scaling_factor";
constexpr const char * KEY_ROPE_LOW_FREQ     = "chatterbox.rope.low_freq_factor";
constexpr const char * KEY_ROPE_HIGH_FREQ    = "chatterbox.rope.high_freq_factor";
constexpr const char * KEY_ROPE_ORIG_MAX_POS = "chatterbox.rope.original_max_position";
constexpr const char * KEY_MAX_TEXT_TOKENS   = "chatterbox.max_text_tokens";
constexpr const char * KEY_MAX_SPEECH_TOKENS = "chatterbox.max_speech_tokens";
constexpr const char * KEY_SPEECH_COND_LEN   = "chatterbox.speech_cond_prompt_len";
constexpr const char * KEY_PERCEIVER_QUERIES = "chatterbox.perceiver_query_tokens";
constexpr const char * KEY_PERCEIVER_HEADS   = "chatterbox.perceiver_num_heads";
constexpr const char * KEY_EMOTION_ADV       = "chatterbox.emotion_adv";
constexpr const char * KEY_START_TEXT        = "chatterbox.start_text_token";
constexpr const char * KEY_STOP_TEXT         = "chatterbox.stop_text_token";
enum chatterbox_variant {
    CHBX_VARIANT_TURBO = 0,
    CHBX_VARIANT_MTL   = 1,
};
struct chatterbox_hparams {
    chatterbox_variant variant = CHBX_VARIANT_TURBO;
    int32_t n_text_vocab       = 0;
    int32_t n_speech_vocab     = 0;
    int32_t start_speech_token = 0;
    int32_t stop_speech_token  = 0;
    int32_t start_text_token   = 0;
    int32_t stop_text_token    = 0;
    int32_t n_ctx              = 0;
    int32_t n_embd             = 0;
    int32_t n_head             = 0;
    int32_t n_kv_head          = 0;
    int32_t head_dim           = 0;
    int32_t intermediate_size  = 0;
    int32_t n_layer            = 0;
    int32_t speaker_embed_size = 0;
    int32_t cond_prompt_len    = 0;
    int32_t max_text_tokens    = 0;
    int32_t max_speech_tokens  = 0;
    int32_t speech_cond_prompt_len = 0;
    int32_t perceiver_queries  = 0;
    int32_t perceiver_heads    = 0;
    bool    emotion_adv        = false;
    float   eps                = 1e-5f;
    float   rope_theta         = 10000.0f;
    float   rope_scale_factor  = 1.0f;
    float   rope_low_freq      = 1.0f;
    float   rope_high_freq     = 4.0f;
    int32_t rope_orig_max_pos  = 8192;
};
struct gpt2_layer {
    ggml_tensor * ln_1_g = nullptr;
    ggml_tensor * ln_1_b = nullptr;
    ggml_tensor * ln_2_g = nullptr;
    ggml_tensor * ln_2_b = nullptr;
    ggml_tensor * c_attn_attn_w = nullptr;
    ggml_tensor * c_attn_attn_b = nullptr;
    ggml_tensor * c_attn_proj_w = nullptr;
    ggml_tensor * c_attn_proj_b = nullptr;
    ggml_tensor * c_mlp_fc_w   = nullptr;
    ggml_tensor * c_mlp_fc_b   = nullptr;
    ggml_tensor * c_mlp_proj_w = nullptr;
    ggml_tensor * c_mlp_proj_b = nullptr;
};
struct llama_layer {
    ggml_tensor * ln_attn_g = nullptr;
    ggml_tensor * ln_mlp_g  = nullptr;
    ggml_tensor * wq = nullptr;
    ggml_tensor * wk = nullptr;
    ggml_tensor * wv = nullptr;
    ggml_tensor * wo = nullptr;
    ggml_tensor * mlp_gate = nullptr;
    ggml_tensor * mlp_up   = nullptr;
    ggml_tensor * mlp_down = nullptr;
    ggml_tensor * wqkv = nullptr;
};
struct perceiver_weights {
    ggml_tensor * pre_attention_query = nullptr;
    ggml_tensor * norm_g = nullptr;
    ggml_tensor * norm_b = nullptr;
    ggml_tensor * to_q_w = nullptr;
    ggml_tensor * to_q_b = nullptr;
    ggml_tensor * to_k_w = nullptr;
    ggml_tensor * to_k_b = nullptr;
    ggml_tensor * to_v_w = nullptr;
    ggml_tensor * to_v_b = nullptr;
    ggml_tensor * proj_out_w = nullptr;
    ggml_tensor * proj_out_b = nullptr;
};
struct chatterbox_model {
    chatterbox_hparams hparams;
    ggml_tensor * wpe              = nullptr;
    ggml_tensor * ln_f_g           = nullptr;
    ggml_tensor * ln_f_b           = nullptr;
    ggml_tensor * text_emb         = nullptr;
    ggml_tensor * speech_emb       = nullptr;
    ggml_tensor * speech_head      = nullptr;
    ggml_tensor * speech_head_bias = nullptr;
    ggml_tensor * cond_spkr_w      = nullptr;
    ggml_tensor * cond_spkr_b      = nullptr;
    ggml_tensor * text_pos_emb     = nullptr;
    ggml_tensor * speech_pos_emb   = nullptr;
    ggml_tensor * text_head        = nullptr;
    ggml_tensor * norm_g           = nullptr;
    ggml_tensor * emotion_adv_w    = nullptr;
    ggml_tensor * rope_freq_factors = nullptr;
    perceiver_weights perceiver;
    ggml_tensor * builtin_speaker_emb        = nullptr;
    ggml_tensor * builtin_cond_prompt_tokens = nullptr;
    std::vector<gpt2_layer>  layers;
    std::vector<llama_layer> layers_mtl;
    ggml_tensor * memory_k = nullptr;
    ggml_tensor * memory_v = nullptr;
    ggml_tensor * memory_k_uncond = nullptr;
    ggml_tensor * memory_v_uncond = nullptr;
    ggml_context * ctx_w  = nullptr;
    ggml_context * ctx_kv = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer_w  = nullptr;
    ggml_backend_buffer_t buffer_kv = nullptr;
    ggml_context *        ctx_stack    = nullptr;
    ggml_backend_buffer_t buffer_stack = nullptr;
    ggml_context *        ctx_override    = nullptr;
    ggml_backend_buffer_t buffer_override = nullptr;
    std::map<std::string, ggml_tensor *> tensors;
    std::vector<std::string> tok_tokens;
    std::vector<std::string> tok_merges;
    std::string mtl_tokenizer_json;
    std::string mtl_cangjie_json;
};
struct chatterbox_sampling_params {
    int32_t top_k          = 1000;
    float   top_p          = 0.95f;
    float   temp           = 0.8f;
    float   repeat_penalty = 1.2f;
    float   min_p          = 0.0f;
    float   cfg_weight     = 0.0f;
};
ggml_backend_t init_backend(int n_gpu_layers);
bool load_model_gguf(
    const std::string & path,
    chatterbox_model &  model,
    int                 requested_ctx,
    int                 n_gpu_layers);
bool eval_prompt(
    const chatterbox_model &     model,
    ggml_gallocr_t               allocr,
    int                          n_threads,
    const std::vector<int32_t> & text_tokens,
    std::vector<float> &         logits_out,
    int &                        prompt_len);
bool eval_step(
    const chatterbox_model & model,
    ggml_gallocr_t           allocr,
    int                      n_threads,
    int                      n_past,
    int32_t                  token,
    std::vector<float> &     logits_out);
int32_t sample_next_token_ex(
    const std::vector<float> &          logits,
    const std::vector<int32_t> &        generated,
    const chatterbox_sampling_params &  params,
    std::mt19937 &                      rng);
void chatterbox_log_cb(ggml_log_level level, const char * text, void * ud);
bool compute_prompt_feat_native(
    const std::string &  wav_path,
    const std::string &  s3gen_gguf,
    std::vector<float> & prompt_feat,
    int &                prompt_feat_rows);
bool compute_embedding_native(
    const std::string &  wav_path,
    const std::string &  s3gen_gguf,
    std::vector<float> & embedding);
bool compute_speech_tokens_native(
    const std::string &    wav_path,
    const std::string &    s3gen_gguf,
    int                    max_cond_tokens,
    std::vector<int32_t> & prompt_token,
    std::vector<int32_t> & cond_prompt_tokens,
    int                    n_threads,
    ggml_backend_t         backend);
bool validate_reference_audio(const std::string & path);
void t3_stack_unregister(ggml_backend_buffer_t buf, ggml_context * ctx);
ggml_cgraph * build_stage_cond_emb_graph(const chatterbox_model & m);
ggml_cgraph * build_stage_text_emb_graph(const chatterbox_model & m, int T_text);
ggml_cgraph * build_stage_inputs_graph(const chatterbox_model & m, int T_text,
                                       bool is_uncond);
ggml_cgraph * build_stage_layers_graph(const chatterbox_model & m, int N,
                                       int n_layers, bool is_uncond);
ggml_cgraph * build_stage_head_graph(const chatterbox_model & m, int N);
bool load_model_gguf_mtl(
    const std::string & path,
    chatterbox_model &  model,
    int                 requested_ctx,
    int                 n_gpu_layers);
bool eval_prompt_mtl(
    const chatterbox_model &     model,
    ggml_gallocr_t               allocr,
    int                          n_threads,
    const std::vector<int32_t> & text_tokens,
    float                        exaggeration,
    std::vector<float> &         logits_cond_out,
    std::vector<float> &         logits_uncond_out,
    int &                        prompt_len);
bool eval_step_mtl(
    const chatterbox_model & model,
    ggml_gallocr_t           allocr,
    int                      n_threads,
    int                      n_past,
    int32_t                  token,
    std::vector<float> &     logits_cond_out,
    std::vector<float> &     logits_uncond_out);
int32_t sample_next_token_mtl(
    const std::vector<float> &         logits_cond,
    const std::vector<float> &         logits_uncond,
    const std::vector<int32_t> &       generated,
    const chatterbox_sampling_params & params,
    std::mt19937 &                     rng,
    int32_t                            stop_token);
}
