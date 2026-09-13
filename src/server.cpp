#include "tts-cpp/chatterbox/engine.h"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <cstdlib>
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

static bool read_exact(HANDLE h, char* buf, DWORD need) {
    DWORD got = 0;
    while (got < need) {
        DWORD n = 0;
        if (!ReadFile(h, buf + got, need - got, &n, nullptr) || n == 0) return false;
        got += n;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 4) throw std::runtime_error("argv");
    HANDLE h = CreateNamedPipeA(argv[3], PIPE_ACCESS_DUPLEX,
        PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
        1, 4096, 4096, 0, nullptr);
    if (h == INVALID_HANDLE_VALUE) throw std::runtime_error("pipe");
    if (!ConnectNamedPipe(h, nullptr) && GetLastError() != ERROR_PIPE_CONNECTED)
        throw std::runtime_error("pipe connect");
    tts_cpp::chatterbox::Engine tts({argv[1], argv[2]});
    for (;;) {
        std::string path = read_line(h);
        std::string len_s = read_line(h);
        char* end = nullptr;
        unsigned long nbytes = std::strtoul(len_s.c_str(), &end, 10);
        if (!path.empty() && end != len_s.c_str() && nbytes > 0 && nbytes <= 1u << 20) {
            std::string text(nbytes, '\0');
            if (read_exact(h, text.data(), (DWORD)nbytes)) {
                write_wav(path.c_str(), tts.synthesize(text));
                DWORD n = 0;
                WriteFile(h, "ok\n", 3, &n, nullptr);
                FlushFileBuffers(h);
            }
        }
        DisconnectNamedPipe(h);
        if (!ConnectNamedPipe(h, nullptr) && GetLastError() != ERROR_PIPE_CONNECTED)
            throw std::runtime_error("pipe connect");
    }
}
