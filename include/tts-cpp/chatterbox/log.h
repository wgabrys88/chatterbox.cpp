#pragma once
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <ctime>
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
    std::string ids;
    if (ctx.valid) ids = " response=" + std::to_string(ctx.response) + " piece=" + std::to_string(ctx.piece);
    const auto seq = tts_log_sequence().fetch_add(1, std::memory_order_relaxed) + 1;
    const auto conn = tts_connection().load(std::memory_order_acquire);
    if (with_mono) {
        std::fprintf(stderr, "[%s] seq=%llu conn=%llu event=%s%s mono_us=%lld | %s%s%s\n",
            ts, seq, conn, event, ids.c_str(), tts_mono_us(), tts_run_identity().c_str(),
            extra ? " " : "", extra ? extra : "");
    } else {
        std::fprintf(stderr, "[%s] seq=%llu conn=%llu event=%s%s | %s%s%s\n",
            ts, seq, conn, event, ids.c_str(), tts_run_identity().c_str(),
            extra ? " " : "", extra ? extra : "");
    }
    std::fflush(stderr);
}

inline void tts_emit(const char* event, const char* extra = nullptr) { tts_emit(event, extra, false); }
inline void tts_emit(const char* event, const std::string& extra) { tts_emit(event, extra.c_str(), false); }
inline void tts_emit_piece(const char* event, const std::string& extra) { tts_emit(event, extra.c_str(), true); }

struct tts_session_acc {
    std::chrono::steady_clock::time_point t0{}, t1{};
    double first_audio_ms = -1;
    bool open = false;
};
inline tts_session_acc& tts_session() { static tts_session_acc s; return s; }
inline void tts_session_begin_if_needed() {
    auto& s = tts_session();
    if (s.open) return;
    s.t0 = s.t1 = std::chrono::steady_clock::now();
    s.first_audio_ms = -1;
    s.open = true;
}
inline void tts_session_note_first_audio() {
    auto& s = tts_session();
    if (!s.open || s.first_audio_ms >= 0) return;
    s.first_audio_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s.t0).count();
}
inline void tts_session_touch_end() {
    if (tts_session().open) tts_session().t1 = std::chrono::steady_clock::now();
}
inline void tts_session_emit() {
    int wall = 0, first = -1;
    auto& s = tts_session();
    if (!s.open) return;
    wall = (int)(std::chrono::duration<double, std::milli>(s.t1 - s.t0).count() + .5);
    first = s.first_audio_ms < 0 ? -1 : (int)(s.first_audio_ms + .5);
    s.open = false;
    tts_emit_piece("session", "tts_synth=" + std::to_string(wall) + " first_audio_ms=" + std::to_string(first));
}
