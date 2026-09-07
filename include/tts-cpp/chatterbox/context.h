#pragma once
#include <cstdint>

struct tts_synthesis_context {
    std::uint32_t response = 0;
    std::uint32_t piece = 0;
    bool valid = false;
};

inline tts_synthesis_context& tts_context() { thread_local tts_synthesis_context c; return c; }
inline tts_synthesis_context tts_get_context() { return tts_context(); }

class tts_context_scope {
    tts_synthesis_context previous_;
public:
    explicit tts_context_scope(tts_synthesis_context next) : previous_(tts_get_context()) { tts_context() = next; }
    tts_context_scope(std::uint32_t response, std::uint32_t piece) : tts_context_scope(tts_synthesis_context{response, piece, true}) {}
    ~tts_context_scope() { tts_context() = previous_; }
};

