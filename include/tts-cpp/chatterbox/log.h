#pragma once
#include <atomic>
#include <cstdio>
#include <cstdint>
#include <string>

struct tts_synthesis_context {
    std::uint32_t response = 0;
    std::uint32_t piece = 0;
    bool valid = false;
};

inline std::string& tts_run_identity() { static std::string id; return id; }
inline std::atomic<unsigned long long>& tts_log_sequence() { static std::atomic<unsigned long long> seq{0}; return seq; }
inline std::atomic<unsigned long long>& tts_connection() { static std::atomic<unsigned long long> id{0}; return id; }
inline tts_synthesis_context& tts_context() { thread_local tts_synthesis_context c; return c; }
inline void tts_set_run_identity(std::string id) { tts_run_identity() = std::move(id); }
inline void tts_set_connection(unsigned long long id) { tts_connection().store(id, std::memory_order_release); }
inline tts_synthesis_context tts_get_context() { return tts_context(); }

class tts_context_scope {
    tts_synthesis_context previous_;
public:
    explicit tts_context_scope(tts_synthesis_context next) : previous_(tts_get_context()) { tts_context() = next; }
    tts_context_scope(std::uint32_t response, std::uint32_t piece) : tts_context_scope(tts_synthesis_context{response, piece, true}) {}
    ~tts_context_scope() { tts_context() = previous_; }
};

inline void tts_log_line(const std::string& line) {
    std::fputs(line.c_str(), stderr);
    std::fputc('\n', stderr);
    std::fflush(stderr);
}

inline void tts_emit(const char* event, const char* extra = nullptr) {
    const auto ctx = tts_get_context();
    const auto seq = tts_log_sequence().fetch_add(1, std::memory_order_relaxed) + 1;
    const auto conn = tts_connection().load(std::memory_order_acquire);
    std::string line = std::string("{\"event\":\"") + event + "\",\"seq\":" + std::to_string(seq) +
        ",\"conn\":" + std::to_string(conn) + ",\"run\":\"" + tts_run_identity() + "\"";
    if (ctx.valid) line += ",\"response\":" + std::to_string(ctx.response) + ",\"piece\":" + std::to_string(ctx.piece);
    if (extra && extra[0]) line += std::string(",\"detail\":\"") + extra + "\"";
    line += "}";
    tts_log_line(line);
}

inline void tts_emit(const char* event, const std::string& extra) { tts_emit(event, extra.c_str()); }
inline void tts_jsonl(const std::string& line) { tts_log_line(line); }
