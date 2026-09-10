#include "tts-cpp/chatterbox/engine.h"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <windows.h>

static void write_wav(const char* path, const std::vector<float>& pcm) {
    const std::string tmp = std::string(path) + ".tmp";
    const uint32_t data = (uint32_t)(pcm.size() * 2), riff = 36 + data, sr = 24000;
    std::ofstream f(tmp, std::ios::binary);
    auto w16 = [&](uint16_t v) { f.write((char*)&v, 2); };
    auto w32 = [&](uint32_t v) { f.write((char*)&v, 4); };
    f.write("RIFF", 4); w32(riff); f.write("WAVEfmt ", 8);
    w32(16); w16(1); w16(1); w32(sr); w32(sr * 2); w16(2); w16(16);
    f.write("data", 4); w32(data);
    for (float x : pcm) {
        int16_t s = (int16_t)(std::clamp(x, -1.f, 1.f) * 32767.f);
        f.write((char*)&s, 2);
    }
    f.close();
    if (!MoveFileExA(tmp.c_str(), path, MOVEFILE_REPLACE_EXISTING))
        throw std::runtime_error("WAV replace");
}

static std::string read_line(HANDLE h) {
    std::string line;
    char c;
    DWORD n = 0;
    while (ReadFile(h, &c, 1, &n, nullptr) && n) {
        if (c == '\n') return line;
        if (c != '\r') line += c;
    }
    return {};
}

int main(int argc, char** argv) {
    if (argc < 6) throw std::runtime_error("argv");
    tts_cpp::chatterbox::Engine tts({argv[1], argv[2], argv[5]});
    HANDLE h = CreateNamedPipeA(argv[4], PIPE_ACCESS_DUPLEX,
        PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
        1, 4096, 4096, 0, nullptr);
    if (h == INVALID_HANDLE_VALUE) throw std::runtime_error("pipe");
    for (;;) {
        if (!ConnectNamedPipe(h, nullptr) && GetLastError() != ERROR_PIPE_CONNECTED)
            throw std::runtime_error("pipe connect");
        std::string line = read_line(h);
        if (!line.empty()) {
            write_wav(argv[3], tts.synthesize(line));
            DWORD n = 0;
            WriteFile(h, "ok\n", 3, &n, nullptr);
            FlushFileBuffers(h);
        }
        DisconnectNamedPipe(h);
    }
}
