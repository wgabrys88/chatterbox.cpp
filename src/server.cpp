#include "tts-cpp/chatterbox/engine.h"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

int main(int, char** argv) {
    tts_cpp::chatterbox::Engine tts({argv[1], argv[2], argv[3]});
    auto pcm = tts.synthesize(argv[5]);
    const uint32_t data = (uint32_t)(pcm.size() * 2), riff = 36 + data, sr = 24000;
    std::ofstream f(argv[4], std::ios::binary);
    auto w16 = [&](uint16_t v) { f.write((char*)&v, 2); };
    auto w32 = [&](uint32_t v) { f.write((char*)&v, 4); };
    f.write("RIFF", 4); w32(riff); f.write("WAVEfmt ", 8);
    w32(16); w16(1); w16(1); w32(sr); w32(sr * 2); w16(2); w16(16);
    f.write("data", 4); w32(data);
    for (float x : pcm) {
        int16_t s = (int16_t)(std::clamp(x, -1.f, 1.f) * 32767.f);
        f.write((char*)&s, 2);
    }
}
