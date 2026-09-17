#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include "tts-cpp/chatterbox/runtime_knobs.h"
namespace tts_cpp::chatterbox::detail {
inline void apply_speech_repeat_penalty(float * scores, int vocab,
                                        const std::vector<int32_t> & generated) {
    if (generated.empty() || vocab <= 0) return;
    const float penalty = effective_repeat_penalty();
    thread_local std::vector<uint32_t> marks;
    thread_local uint32_t epoch = 0;
    if (marks.size() < (size_t)vocab) marks.resize((size_t)vocab, 0);
    if (++epoch == 0) {
        std::fill(marks.begin(), marks.end(), 0);
        epoch = 1;
    }
    for (int32_t t : generated) {
        if (t < 0 || t >= vocab || marks[(size_t)t] == epoch) continue;
        marks[(size_t)t] = epoch;
        float & s = scores[t];
        if (s == -INFINITY) continue;
        s = s > 0.0f ? s / penalty : s * penalty;
    }
}
}
