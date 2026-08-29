#pragma once
#include <chrono>
#include <cstdio>
#include <ctime>
#include <string>

inline std::string tts_json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 2);
    o.push_back('"');
    for (unsigned char c : s) {
        switch (c) {
        case '"': o += "\\\""; break;
        case '\\': o += "\\\\"; break;
        case '\n': o += "\\n"; break;
        case '\r': o += "\\r"; break;
        case '\t': o += "\\t"; break;
        default:
            if (c < 0x20) {
                char b[8];
                std::snprintf(b, sizeof(b), "\\u%04x", c);
                o += b;
            } else o.push_back(char(c));
        }
    }
    o.push_back('"');
    return o;
}

inline void tts_emit(const char* event, const std::string& extra = {}) {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    const std::time_t t = system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char ts[40], tz[8];
    std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", &tm);
    std::strftime(tz, sizeof(tz), "%z", &tm);
    fprintf(stderr, "{\"ts\":\"%s.%03d%s\",\"event\":%s%s}\n", ts, (int)ms.count(), tz, tts_json_escape(event).c_str(), extra.c_str());
    fflush(stderr);
}
