#pragma once
#include <atomic>
#include <cstdio>
#include <cstdint>
#include <ctime>
#include <mutex>
#include <string>

struct tts_synthesis_context {
    std::uint32_t epoch = 0;
    std::uint32_t response_id = 0;
    std::uint32_t piece_id = 0;
    bool valid = false;
};

inline std::string& tts_run_identity() { static std::string id; return id; }
inline std::atomic<std::uint32_t>& tts_live_epoch() { static std::atomic<std::uint32_t> epoch{0}; return epoch; }
inline tts_synthesis_context& tts_context() { thread_local tts_synthesis_context c; return c; }
inline void tts_set_run_identity(std::string id) { tts_run_identity() = std::move(id); }
inline void tts_set_live_epoch(std::uint32_t epoch) { tts_live_epoch().store(epoch, std::memory_order_release); }
inline void tts_set_context(std::uint32_t epoch, std::uint32_t response, std::uint32_t piece) { tts_context() = {epoch, response, piece, true}; }
inline void tts_clear_context() { tts_context() = {}; }
inline tts_synthesis_context tts_get_context() { return tts_context(); }

class tts_context_scope {
    tts_synthesis_context previous_;
public:
    explicit tts_context_scope(tts_synthesis_context next) : previous_(tts_get_context()) { tts_context() = next; }
    tts_context_scope(std::uint32_t epoch, std::uint32_t response, std::uint32_t piece) : tts_context_scope(tts_synthesis_context{epoch, response, piece, true}) {}
    ~tts_context_scope() { tts_context() = previous_; }
};

inline void tts_emit(const char* event, const char* extra = nullptr) {
    const auto now = std::time(nullptr);
    std::tm tm;
#if defined(_WIN32)
    localtime_s(&tm, &now);
#else
    localtime_r(&now, &tm);
#endif
    char ts[32];
    std::strftime(ts, sizeof(ts), "%H:%M:%S", &tm);

    const auto ctx = tts_get_context();
    std::string ctx_str;
    if (ctx.valid) {
        ctx_str = " epoch=" + std::to_string(ctx.epoch) + " piece=" + std::to_string(ctx.piece_id);
    }

    std::fprintf(stderr, "[%s] %s%s | %s%s%s\n",
        ts, event, ctx_str.c_str(), tts_run_identity().c_str(),
        extra ? " " : "", extra ? extra : "");
    std::fflush(stderr);
}

inline void tts_emit(const char* event, const std::string& extra) {
    tts_emit(event, extra.c_str());
}
