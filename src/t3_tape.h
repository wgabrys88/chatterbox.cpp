#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace tts_cpp::chatterbox {
// Clock: one speech token = 40 ms. Model envelope X_MODEL is already known.
// The architecture is the lung inequality, not those ceilings:
//   C  = prompt_fuel_mean * kBreathTokens     static tank for this voice
//   e  = fuel[t] * (voiced ? f0[t]/prompt_f0 : 1)
//   one breath iff sum(e) <= C
inline constexpr int X_BREATH = 100; // 4 s biology, used only to scale C
inline constexpr int X_MODEL = 750;  // ~30 s native envelope, packer/call cap
inline constexpr int VCHUNK_MIN = 2; // vocoder needs two speech tokens
struct T3Tape {
    std::vector<int32_t> speech_ids;
    std::vector<int32_t> raw_ids;
    int stop_code = 0; // 0 eos, 1 context_limit, 2 prediction_limit
    int n_past = 0;
    int eos = 0;
    int text_tokens = 0;
    int s3gen_sil = 4299;
    int appended_silence_count = 0;
    int n_predict = 0;
    int seed = 0;
    int stage = 0;
    int cut_x = 0;
};
struct S3GaugeTensors {
    std::vector<float> codebook_norm;
    std::vector<float> fuel;
    std::vector<float> f0;
    std::vector<int32_t> voiced;
    std::vector<int32_t> sil_index;
    int n_prompt = 0;
    int n_frames = 0;
    int loop_start = -1;
    float reuse_score = 0.0f;
    float voiced_threshold = 10.0f;
    float prompt_fuel_mean = 0.0f;
    float prompt_f0_mean = 0.0f;
    float breath_capacity = 0.0f;
    float effort_sum = 0.0f;
};
struct VChunk {
    int begin = 0;
    int end = 0;
    int reason = 0; // 0 end, 1 x, 3 speaking_loop, 5 tank
};
struct VChunkPlan {
    std::vector<VChunk> chunks;
};
int first_speaking_loop(const std::vector<int32_t>& ids, const std::vector<int>& voiced_t,
                        int n, int window = 8);
void analyze_tape_ids(const T3Tape& tape, S3GaugeTensors& g);
VChunkPlan vchunker(const T3Tape& tape, S3GaugeTensors& g);
std::vector<int32_t> chunk_ids(const T3Tape& tape, const VChunk& c);
void write_utterance_gguf(const std::string& path, const T3Tape& tape,
                          const S3GaugeTensors* gauge, const VChunkPlan& plan);
}
