#include "t3_tape.h"
#include "ggml.h"
#include "gguf.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <windows.h>
namespace tts_cpp::chatterbox {
static void replace_file(const std::string& tmp, const std::string& dest) {
    if (!MoveFileExW(std::filesystem::u8path(tmp).c_str(), std::filesystem::u8path(dest).c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("utterance GGUF replace failed: " + dest);
}
int first_reuse_loop(const std::vector<int32_t>& ids, int n, int window) {
    if (window < 2 || n < window * 2) return -1;
    std::map<std::vector<int32_t>, int> first;
    for (int i = 0; i + window <= n; ++i) {
        std::vector<int32_t> g(ids.begin() + i, ids.begin() + i + window);
        auto it = first.find(g);
        if (it != first.end()) return i;
        first.emplace(std::move(g), i);
    }
    return -1;
}
void analyze_tape_ids(const T3Tape& tape, S3GaugeTensors& g) {
    const int n = (int)tape.speech_ids.size();
    const int body = std::max(0, n - std::max(tape.appended_silence_count, 0));
    g.sil_index.clear();
    for (int i = 0; i < body; ++i)
        if (tape.speech_ids[(size_t)i] == tape.s3gen_sil) g.sil_index.push_back(i);
    g.loop_start = first_reuse_loop(tape.speech_ids, body, 8);
    int uniq = 0;
    std::map<int32_t, int> seen;
    for (int i = 0; i < n; ++i)
        if (seen[tape.speech_ids[(size_t)i]]++ == 0) ++uniq;
    g.reuse_score = n > 0 ? 1.0f - (float)uniq / (float)n : 0.0f;
}
static void token_signals(const T3Tape& tape, const S3GaugeTensors& g, int body,
                          std::vector<float>& fuel_t, std::vector<int>& voiced_t) {
    fuel_t.assign((size_t)body, 0.0f);
    voiced_t.assign((size_t)body, 1);
    const int F = (int)g.fuel.size();
    if (F < 1 || body < 1) return;
    const int n_all = (int)tape.speech_ids.size();
    const bool two = (F == 2 * n_all) || (F == 2 * body);
    for (int i = 0; i < body; ++i) {
        int a, b;
        if (two) {
            a = 2 * i;
            b = a + 2;
        } else {
            a = (int)((long long)i * F / body);
            b = (int)((long long)(i + 1) * F / body);
            if (b <= a) b = a + 1;
        }
        if (a < 0) a = 0;
        if (b > F) b = F;
        float s = 0.0f;
        int v = 0, c = 0;
        for (int f = a; f < b; ++f) {
            s += g.fuel[(size_t)f];
            if (f < (int)g.voiced.size() && g.voiced[(size_t)f]) ++v;
            ++c;
        }
        if (c) {
            fuel_t[(size_t)i] = s / (float)c;
            voiced_t[(size_t)i] = (v * 2 >= c) ? 1 : 0;
        }
    }
}
static int pick_natural_end(int start, int hard, int lo,
                            const std::vector<int32_t>& sil,
                            const std::vector<float>& fuel_t,
                            const std::vector<int>& voiced_t) {
    int best_sil = -1;
    for (int s : sil) {
        if (s + 1 > lo && s + 1 <= hard && s >= start)
            best_sil = s + 1;
    }
    if (best_sil > start) return best_sil;
    const int win = std::max(16, (hard - start) / 4);
    const int from = std::max(lo, hard - win);
    if (!voiced_t.empty()) {
        for (int i = hard; i > from; --i) {
            if (i - 1 >= start && i - 1 < (int)voiced_t.size() && voiced_t[(size_t)(i - 1)] == 0)
                return i;
        }
    }
    if (fuel_t.size() >= 3) {
        for (int i = hard - 1; i > from; --i) {
            if (i <= start || i + 1 >= (int)fuel_t.size()) continue;
            const float x = fuel_t[(size_t)i];
            if (x <= fuel_t[(size_t)(i - 1)] && x <= fuel_t[(size_t)(i + 1)])
                return i + 1;
        }
    }
    return hard;
}
VChunkPlan vchunker(const T3Tape& tape, const S3GaugeTensors& g) {
    VChunkPlan plan;
    const int n = (int)tape.speech_ids.size();
    if (n < 1) throw std::runtime_error("empty T3 tape");
    int body = std::max(1, n - std::max(tape.appended_silence_count, 0));
    int reason_stop = 0;
    if (g.loop_start >= VCHUNK_MIN && g.loop_start < body) {
        body = g.loop_start;
        reason_stop = 3;
    }
    const int x = tape.cut_x > 0 ? tape.cut_x : body;
    std::vector<float> fuel_t;
    std::vector<int> voiced_t;
    token_signals(tape, g, body, fuel_t, voiced_t);
    int start = 0;
    while (start < body) {
        int hard = start + x;
        if (hard > body) hard = body;
        VChunk c;
        c.begin = start;
        if (hard == body) {
            c.end = body;
            c.reason = reason_stop ? reason_stop : 0;
            plan.chunks.push_back(c);
            break;
        }
        const int lo = start + VCHUNK_MIN;
        const int nat = pick_natural_end(start, hard, lo, g.sil_index, fuel_t, voiced_t);
        c.end = nat;
        if (c.end <= start) c.end = hard;
        if (nat < hard && nat > start) {
            bool sil = false;
            for (int s : g.sil_index) if (s + 1 == nat) { sil = true; break; }
            if (sil) c.reason = 2;
            else if (!voiced_t.empty() && nat - 1 < (int)voiced_t.size() && voiced_t[(size_t)(nat - 1)] == 0) c.reason = 4;
            else c.reason = 5;
        } else c.reason = 1;
        if (c.end - c.begin < 1) c.end = std::min(start + 1, body);
        plan.chunks.push_back(c);
        start = c.end;
    }
    if (plan.chunks.empty()) {
        VChunk c;
        c.begin = 0;
        c.end = n;
        plan.chunks.push_back(c);
    }
    if (plan.chunks.size() >= 2) {
        VChunk& last = plan.chunks.back();
        if (last.end - last.begin < VCHUNK_MIN) {
            plan.chunks[plan.chunks.size() - 2].end = last.end;
            plan.chunks.pop_back();
        }
    }
    return plan;
}
std::vector<int32_t> chunk_ids(const T3Tape& tape, const VChunk& c) {
    if (c.begin < 0 || c.end < c.begin || c.end > (int)tape.speech_ids.size())
        throw std::runtime_error("vchunker range");
    return std::vector<int32_t>(tape.speech_ids.begin() + c.begin, tape.speech_ids.begin() + c.end);
}
struct packed {
    std::string name;
    ggml_type type = GGML_TYPE_F32;
    std::vector<int64_t> ne;
    const void* data = nullptr;
    size_t bytes = 0;
};
static packed i32_tensor(const char* name, const std::vector<int32_t>& v) {
    packed p;
    p.name = name;
    p.type = GGML_TYPE_I32;
    p.ne = { (int64_t)std::max<size_t>(v.size(), 1) };
    p.data = v.empty() ? nullptr : v.data();
    p.bytes = v.size() * sizeof(int32_t);
    return p;
}
static packed f32_tensor(const char* name, const std::vector<float>& v) {
    packed p;
    p.name = name;
    p.type = GGML_TYPE_F32;
    p.ne = { (int64_t)std::max<size_t>(v.size(), 1) };
    p.data = v.empty() ? nullptr : v.data();
    p.bytes = v.size() * sizeof(float);
    return p;
}
void write_utterance_gguf(const std::string& path, const T3Tape& tape,
                          const S3GaugeTensors* gauge, const VChunkPlan& plan) {
    if (tape.speech_ids.empty() || plan.chunks.empty())
        throw std::runtime_error("utterance GGUF empty tape");
    const int32_t pad_i = 0;
    const float pad_f = 0.0f;
    std::vector<int32_t> ends, reasons;
    ends.reserve(plan.chunks.size());
    reasons.reserve(plan.chunks.size());
    for (const auto& c : plan.chunks) {
        ends.push_back(c.end);
        reasons.push_back(c.reason);
    }
    std::vector<packed> tensors;
    tensors.push_back(i32_tensor("t3/tape/speech_ids", tape.speech_ids));
    tensors.push_back(i32_tensor("t3/tape/raw_ids", tape.raw_ids.empty() ? tape.speech_ids : tape.raw_ids));
    tensors.push_back(i32_tensor("t3/vchunker/ends", ends));
    tensors.push_back(i32_tensor("t3/vchunker/reason", reasons));
    if (gauge) {
        tensors.push_back(f32_tensor("s3/gauge/codebook_norm", gauge->codebook_norm));
        tensors.push_back(f32_tensor("s3/gauge/fuel", gauge->fuel));
        tensors.push_back(f32_tensor("s3/gauge/f0", gauge->f0));
        tensors.push_back(i32_tensor("s3/gauge/voiced", gauge->voiced));
        if (!gauge->sil_index.empty())
            tensors.push_back(i32_tensor("s3/gauge/sil_index", gauge->sil_index));
    }
    ggml_init_params rp = { ggml_tensor_overhead() * (tensors.size() + 4), nullptr, true };
    ggml_context* rctx = ggml_init(rp);
    if (!rctx) throw std::runtime_error("utterance ggml_init");
    gguf_context* gout = gguf_init_empty();
    gguf_set_val_str(gout, "t3.tape.arch", "t3-s3-vchunker");
    gguf_set_val_u32(gout, "t3.tape.schema", 1);
    gguf_set_val_u32(gout, "t3.tape.stop_code", (uint32_t)tape.stop_code);
    gguf_set_val_u32(gout, "t3.tape.n_past", (uint32_t)tape.n_past);
    gguf_set_val_u32(gout, "t3.tape.eos", (uint32_t)tape.eos);
    gguf_set_val_u32(gout, "t3.tape.text_tokens", (uint32_t)tape.text_tokens);
    gguf_set_val_u32(gout, "t3.tape.samples_per_token", 960);
    gguf_set_val_u32(gout, "t3.tape.sample_rate", 24000);
    gguf_set_val_u32(gout, "t3.tape.token_ms", 40);
    gguf_set_val_u32(gout, "t3.tape.s3gen_sil", (uint32_t)tape.s3gen_sil);
    gguf_set_val_u32(gout, "t3.tape.appended_silence", (uint32_t)tape.appended_silence_count);
    gguf_set_val_u32(gout, "t3.tape.n_speech", (uint32_t)tape.speech_ids.size());
    gguf_set_val_u32(gout, "t3.tape.n_raw", (uint32_t)(tape.raw_ids.empty() ? tape.speech_ids.size() : tape.raw_ids.size()));
    gguf_set_val_u32(gout, "t3.tape.n_predict", (uint32_t)tape.n_predict);
    gguf_set_val_u32(gout, "t3.tape.seed", (uint32_t)tape.seed);
    gguf_set_val_u32(gout, "t3.tape.stage", (uint32_t)tape.stage);
    gguf_set_val_u32(gout, "t3.tape.cut_x", (uint32_t)tape.cut_x);
    gguf_set_val_u32(gout, "t3.vchunker.n_chunks", (uint32_t)plan.chunks.size());
    gguf_set_val_u32(gout, "t3.vchunker.first_end", (uint32_t)plan.chunks.front().end);
    gguf_set_val_u32(gout, "t3.vchunker.first_reason", (uint32_t)plan.chunks.front().reason);
    gguf_set_val_u32(gout, "t3.tape.duration_ms", (uint32_t)tape.speech_ids.size() * 40u);
    if (gauge) {
        gguf_set_val_u32(gout, "s3.gauge.schema", 1);
        gguf_set_val_u32(gout, "s3.gauge.n_prompt", (uint32_t)gauge->n_prompt);
        gguf_set_val_u32(gout, "s3.gauge.n_frames", (uint32_t)gauge->n_frames);
        gguf_set_val_u32(gout, "s3.gauge.n_sil", (uint32_t)gauge->sil_index.size());
        gguf_set_val_u32(gout, "s3.gauge.loop_start", gauge->loop_start < 0 ? 0xffffffffu : (uint32_t)gauge->loop_start);
        gguf_set_val_f32(gout, "s3.gauge.reuse_score", gauge->reuse_score);
        gguf_set_val_f32(gout, "s3.gauge.voiced_threshold", gauge->voiced_threshold);
    }
    for (auto& t : tensors) {
        int64_t ne[4] = { 1, 1, 1, 1 };
        for (size_t d = 0; d < t.ne.size() && d < 4; ++d) ne[d] = t.ne[d];
        ggml_tensor* gt = ggml_new_tensor(rctx, t.type, (int)t.ne.size(), ne);
        ggml_set_name(gt, t.name.c_str());
        gguf_add_tensor(gout, gt);
        const void* data = t.data;
        if (!data) {
            if (t.type == GGML_TYPE_I32) data = &pad_i;
            else data = &pad_f;
        }
        if (t.bytes) {
            if (ggml_nbytes(gt) != t.bytes) throw std::runtime_error("utterance tensor size: " + t.name);
            gguf_set_tensor_data(gout, t.name.c_str(), data);
        }
    }
    const std::string tmp = path + ".tmp";
    try {
        if (!gguf_write_to_file(gout, tmp.c_str(), false))
            throw std::runtime_error("utterance GGUF write failed: " + path);
        gguf_free(gout);
        ggml_free(rctx);
        gout = nullptr;
        rctx = nullptr;
        replace_file(tmp, path);
    } catch (...) {
        if (gout) gguf_free(gout);
        if (rctx) ggml_free(rctx);
        std::error_code ec;
        std::filesystem::remove(std::filesystem::u8path(tmp), ec);
        throw;
    }
}
}
