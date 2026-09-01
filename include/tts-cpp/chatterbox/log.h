#pragma once
#include <chrono>
#include <atomic>
#include <cstdio>
#include <cstdint>
#include <ctime>
#include <mutex>
#include <string>
#include <utility>

struct tts_synthesis_context {
    std::uint32_t epoch = 0;
    std::uint32_t response_id = 0;
    std::uint32_t piece_id = 0;
    bool valid = false;
};

inline std::string tts_json_escape(const std::string& s) {
    std::string o; o.reserve(s.size() + 2); o.push_back('"');
    for (unsigned char c : s) {
        switch (c) {
        case '"': o += "\\\""; break; case '\\': o += "\\\\"; break;
        case '\n': o += "\\n"; break; case '\r': o += "\\r"; break; case '\t': o += "\\t"; break;
        default:
            if (c < 0x20) { char b[8]; std::snprintf(b, sizeof(b), "\\u%04x", c); o += b; }
            else o.push_back(char(c));
        }
    }
    o.push_back('"'); return o;
}

inline std::mutex& tts_log_mutex() { static std::mutex m; return m; }
inline std::string& tts_run_identity() { static std::string id; return id; }
inline std::uint64_t& tts_log_sequence() { static std::uint64_t n = 0; return n; }
inline std::atomic<std::uint32_t>& tts_live_epoch() { static std::atomic<std::uint32_t> epoch{0}; return epoch; }
inline tts_synthesis_context& tts_context() { thread_local tts_synthesis_context c; return c; }
inline void tts_set_run_identity(std::string id) { std::lock_guard<std::mutex> lock(tts_log_mutex()); tts_run_identity() = std::move(id); }
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

inline void tts_emit(const char* event, const std::string& extra = {}, bool stale_is_violation = true) {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto mono = duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
    const auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    const std::time_t t = system_clock::to_time_t(now); std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char ts[40], tz[8]; std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", &tm); std::strftime(tz, sizeof(tz), "%z", &tm);
    const auto ctx = tts_get_context();
    std::lock_guard<std::mutex> lock(tts_log_mutex());
    std::string ids;
    if (ctx.valid) {
        const auto live_epoch = tts_live_epoch().load(std::memory_order_acquire);
        ids = ",\"epoch\":" + std::to_string(ctx.epoch)
            + ",\"response_id\":" + std::to_string(ctx.response_id)
            + ",\"piece_id\":" + std::to_string(ctx.piece_id)
            + ",\"live_epoch\":" + std::to_string(live_epoch)
            + ",\"epoch_violation\":" + (ctx.epoch != live_epoch && stale_is_violation ? "true" : "false");
    }
    std::fprintf(stderr, "{\"schema_version\":2,\"run_id\":%s,\"sequence\":%llu,\"wall_timestamp\":\"%s.%03d%s\",\"monotonic_ns\":%lld,\"component\":\"chatterbox\",\"event\":%s%s%s}\n",
        tts_json_escape(tts_run_identity()).c_str(), (unsigned long long)++tts_log_sequence(), ts, (int)ms.count(), tz,
        (long long)mono, tts_json_escape(event).c_str(), ids.c_str(), extra.c_str());
    std::fflush(stderr);
}
