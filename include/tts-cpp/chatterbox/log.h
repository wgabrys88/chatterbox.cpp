#pragma once
#include <atomic>
#include <chrono>
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

inline long long tts_mono_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline void tts_emit(const char* event, const char* extra, bool with_mono) {
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

    if (with_mono) {
        std::fprintf(stderr, "[%s] %s%s mono_us=%lld | %s%s%s\n",
            ts, event, ctx_str.c_str(), tts_mono_us(), tts_run_identity().c_str(),
            extra ? " " : "", extra ? extra : "");
        std::fflush(stderr);
    } else {
        std::fprintf(stderr, "[%s] %s%s | %s%s%s\n",
            ts, event, ctx_str.c_str(), tts_run_identity().c_str(),
            extra ? " " : "", extra ? extra : "");
    }
}

inline void tts_emit(const char* event, const char* extra = nullptr) {
    tts_emit(event, extra, false);
}

inline void tts_emit(const char* event, const std::string& extra) {
    tts_emit(event, extra.c_str(), false);
}

inline void tts_emit_piece(const char* event, const std::string& extra) {
    tts_emit(event, extra.c_str(), true);
}

struct tts_session_acc {
    std::chrono::steady_clock::time_point t0{};
    std::chrono::steady_clock::time_point t1{};
    double first_audio_ms = -1;
    double overlap_ms = 0;
    bool open = false;
};

inline tts_session_acc& tts_session() { static tts_session_acc s; return s; }
inline std::mutex& tts_session_mu() { static std::mutex m; return m; }

inline void tts_session_begin_if_needed() {
    std::lock_guard<std::mutex> lock(tts_session_mu());
    auto& s = tts_session();
    if (s.open) return;
    s.t0 = s.t1 = std::chrono::steady_clock::now();
    s.first_audio_ms = -1;
    s.overlap_ms = 0;
    s.open = true;
}

inline void tts_session_note_first_audio() {
    std::lock_guard<std::mutex> lock(tts_session_mu());
    auto& s = tts_session();
    if (!s.open || s.first_audio_ms >= 0) return;
    s.first_audio_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - s.t0).count();
}

inline void tts_session_add_overlap(double ms) {
    std::lock_guard<std::mutex> lock(tts_session_mu());
    if (tts_session().open) tts_session().overlap_ms += ms;
}

inline void tts_session_touch_end() {
    std::lock_guard<std::mutex> lock(tts_session_mu());
    if (tts_session().open) tts_session().t1 = std::chrono::steady_clock::now();
}

inline void tts_session_emit() {
    int wall = 0, fa = -1, ov = 0;
    {
        std::lock_guard<std::mutex> lock(tts_session_mu());
        auto& s = tts_session();
        if (!s.open) return;
        wall = (int)(std::chrono::duration<double, std::milli>(s.t1 - s.t0).count() + 0.5);
        fa = s.first_audio_ms < 0 ? -1 : (int)(s.first_audio_ms + 0.5);
        ov = (int)(s.overlap_ms + 0.5);
        s.open = false;
    }
    tts_emit_piece("session", std::string(" tts_synth=") + std::to_string(wall)
        + " first_audio_ms=" + std::to_string(fa)
        + " overlap_ms=" + std::to_string(ov));
}
