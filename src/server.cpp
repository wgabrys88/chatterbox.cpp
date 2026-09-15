#include "tts-cpp/chatterbox/engine.h"
#include "tts-cpp/chatterbox/runtime_knobs.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <windows.h>

using tts_cpp::chatterbox::detail::Mode;
using tts_cpp::chatterbox::detail::mode_name;

static void write_wav(const char* path, const std::vector<int16_t>& pcm) {
    const std::string tmp = std::string(path) + ".tmp";
    const uint32_t data = (uint32_t)(pcm.size() * 2), riff = 36 + data, sr = 24000;
    std::ofstream f(tmp, std::ios::binary);
    auto w16 = [&](uint16_t v) { f.write((char*)&v, 2); };
    auto w32 = [&](uint32_t v) { f.write((char*)&v, 4); };
    f.write("RIFF", 4); w32(riff); f.write("WAVEfmt ", 8);
    w32(16); w16(1); w16(1); w32(sr); w32(sr * 2); w16(2); w16(16);
    f.write("data", 4); w32(data);
    f.write((const char*)pcm.data(), (std::streamsize)data);
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

static void write_exact(HANDLE h, const void* src, DWORD need) {
    const char* p = (const char*)src;
    DWORD sent = 0;
    while (sent < need) {
        DWORD n = 0;
        if (!WriteFile(h, p + sent, need - sent, &n, nullptr) || n == 0)
            throw std::runtime_error("pipe write");
        sent += n;
    }
}

static std::string one_line(const char* s) {
    std::string o = s && *s ? s : "error";
    for (char& c : o) {
        if (c == '\n' || c == '\r') c = ' ';
    }
    if (o.size() > 4096) o.resize(4096);
    return o;
}

static int parse_int(const char* s) {
    char* end = nullptr;
    const long x = std::strtol(s, &end, 10);
    if (!s || end == s || *end) throw std::runtime_error("knob int");
    return (int)x;
}

static float parse_float(const char* s) {
    char* end = nullptr;
    const float x = std::strtof(s, &end);
    if (!s || end == s || *end) throw std::runtime_error("knob float");
    return x;
}

static Mode parse_mode(const char* s) {
    if (std::strcmp(s, "streaming") == 0) return Mode::Streaming;
    if (std::strcmp(s, "batching") == 0) return Mode::Batching;
    throw std::runtime_error("mode");
}

static std::string parse_flags(int argc, char** argv) {
    auto& k = tts_cpp::chatterbox::detail::runtime_knobs();
    std::string language;
    int i = 4;
    while (i < argc) {
        const char* a = argv[i++];
        if (i >= argc) throw std::runtime_error(a);
        const char* v = argv[i++];
        if (std::strcmp(a, "--language") == 0) {
#if defined(TTS_FAMILY_V3)
            language = v;
            continue;
#else
            throw std::runtime_error(a);
#endif
        }
        if (std::strcmp(a, "--mode") == 0) { k.mode = parse_mode(v); continue; }
        if (std::strcmp(a, "--split-tokens") == 0) { k.split_tokens = parse_int(v); continue; }
        if (std::strcmp(a, "--n-ctx") == 0) { k.n_ctx = parse_int(v); continue; }
        if (std::strcmp(a, "--seed") == 0) { k.seed = parse_int(v); continue; }
        if (std::strcmp(a, "--temperature") == 0) { k.temperature = parse_float(v); continue; }
        if (std::strcmp(a, "--top-k") == 0) { k.top_k = parse_int(v); continue; }
        if (std::strcmp(a, "--top-p") == 0) { k.top_p = parse_float(v); continue; }
        if (std::strcmp(a, "--repeat-penalty") == 0) { k.repeat_penalty = parse_float(v); continue; }
        if (std::strcmp(a, "--n-predict") == 0) { k.n_predict = parse_int(v); continue; }
        if (std::strcmp(a, "--cfm-steps") == 0) { k.cfm_steps = parse_int(v); continue; }
        if (std::strcmp(a, "--silence-token") == 0) { k.silence_token = parse_int(v); continue; }
#if defined(TTS_FAMILY_V3)
        if (std::strcmp(a, "--min-p") == 0) { k.min_p = parse_float(v); continue; }
        if (std::strcmp(a, "--cfg-weight") == 0) { k.cfg_weight = parse_float(v); continue; }
        if (std::strcmp(a, "--cfm-cfg") == 0) { k.cfm_cfg = parse_float(v); continue; }
#else
        if (std::strcmp(a, "--silence-count") == 0) { k.silence_count = parse_int(v); continue; }
#endif
        throw std::runtime_error(a);
    }
#if defined(TTS_FAMILY_V3)
    if (language.empty()) throw std::runtime_error("language");
#endif
    return language;
}

// Resolved knobs, one key=value list shared by the stderr banner and the
// stats trailer so the client sees exactly what the engine ran with.
static std::string knob_list() {
    const auto& k = tts_cpp::chatterbox::detail::runtime_knobs();
    char buf[768];
#if defined(TTS_FAMILY_V3)
    const int n = std::snprintf(buf, sizeof(buf),
        "mode=%s split_tokens=%d n_ctx=%d seed=%d temperature=%g top_k=%d top_p=%g repeat_penalty=%g n_predict=%d cfm_steps=%d silence_token=%d min_p=%g cfg_weight=%g cfm_cfg=%g",
        mode_name(k.mode), k.split_tokens, k.n_ctx,
        k.seed, k.temperature, k.top_k, k.top_p, k.repeat_penalty, k.n_predict, k.cfm_steps, k.silence_token,
        k.min_p, k.cfg_weight, k.cfm_cfg);
#else
    const int n = std::snprintf(buf, sizeof(buf),
        "mode=%s split_tokens=%d n_ctx=%d seed=%d temperature=%g top_k=%d top_p=%g repeat_penalty=%g n_predict=%d cfm_steps=%d silence_token=%d silence_count=%d",
        mode_name(k.mode), k.split_tokens, k.n_ctx,
        k.seed, k.temperature, k.top_k, k.top_p, k.repeat_penalty, k.n_predict, k.cfm_steps, k.silence_token,
        k.silence_count);
#endif
    if (n <= 0 || n >= (int)sizeof(buf)) throw std::runtime_error("knobs");
    return std::string(buf, (size_t)n);
}

static std::string stats_line(const tts_cpp::chatterbox::SynthesizeStats& s) {
    char buf[256];
    const int n = std::snprintf(buf, sizeof(buf),
        "predicted=%d dropped=%d eos=%d n_past=%d units=%d text_tokens=%d max_unit_predicted=%d ",
        s.predicted_count, s.dropped_count, s.eos, s.n_past, s.units, s.text_tokens, s.max_unit_predicted);
    if (n <= 0 || n >= (int)sizeof(buf)) throw std::runtime_error("stats");
    return std::string(buf, (size_t)n) + knob_list() + "\n";
}

struct Sink {
    Mode mode;
    HANDLE handle;
    std::vector<int16_t> frame;  // streaming scratch
    std::vector<int16_t> all;    // batching accumulator
};

static void on_pcm(const float* pcm, std::size_t samples, void* user) {
    auto& sink = *(Sink*)user;
    if (sink.mode == Mode::Batching) {
        const std::size_t off = sink.all.size();
        sink.all.resize(off + samples);
        for (std::size_t i = 0; i < samples; ++i)
            sink.all[off + i] = (int16_t)(std::clamp(pcm[i], -1.f, 1.f) * 32767.f);
        return;
    }
    sink.frame.resize(samples);
    for (std::size_t i = 0; i < samples; ++i)
        sink.frame[i] = (int16_t)(std::clamp(pcm[i], -1.f, 1.f) * 32767.f);
    const uint32_t bytes = (uint32_t)(sink.frame.size() * sizeof(int16_t));
    write_exact(sink.handle, &bytes, sizeof(bytes));
    if (bytes) write_exact(sink.handle, sink.frame.data(), bytes);
}

static void write_stream_error(HANDLE h, const char* msg) {
    const uint32_t err = 0xFFFFFFFFu;
    write_exact(h, &err, sizeof(err));
    const std::string m = one_line(msg);
    const uint32_t n = (uint32_t)m.size();
    write_exact(h, &n, sizeof(n));
    if (n) write_exact(h, m.data(), n);
}

// streaming request:  "<nbytes>\n" text            reply: [u32 len][pcm]... [u32 0] stats\n  |  [u32 0xFFFFFFFF][u32 len][msg]
// batching request:   "<path>\n<nbytes>\n" text     reply: "ok stats\n"  |  "err msg\n"
static void serve_one(HANDLE h, tts_cpp::chatterbox::Engine& tts, Mode mode) {
    std::string path;
    if (mode == Mode::Batching) path = read_line(h);
    const std::string len_s = read_line(h);
    char* end = nullptr;
    const unsigned long nbytes = std::strtoul(len_s.c_str(), &end, 10);
    if (end == len_s.c_str() || nbytes == 0) return;
    if (mode == Mode::Batching && path.empty()) return;
    std::string text(nbytes, '\0');
    if (!read_exact(h, text.data(), (DWORD)nbytes)) return;
    Sink sink{mode, h, {}, {}};
    try {
        tts_cpp::chatterbox::SynthesizeStats stats;
        tts.synthesize(text, on_pcm, &sink, &stats);
        if (mode == Mode::Batching) {
            write_wav(path.c_str(), sink.all);
            const std::string line = "ok " + stats_line(stats);
            write_exact(h, line.data(), (DWORD)line.size());
        } else {
            const uint32_t done = 0;
            write_exact(h, &done, sizeof(done));
            const std::string line = stats_line(stats);
            write_exact(h, line.data(), (DWORD)line.size());
        }
        FlushFileBuffers(h);
    } catch (const std::exception& e) {
        fprintf(stderr, "synthesize error: %s\n", e.what());
        fflush(stderr);
        try {
            if (mode == Mode::Batching) {
                const std::string line = std::string("err ") + one_line(e.what()) + "\n";
                write_exact(h, line.data(), (DWORD)line.size());
            } else {
                write_stream_error(h, e.what());
            }
            FlushFileBuffers(h);
        } catch (...) {
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 4) throw std::runtime_error("argv");
    const std::string language = parse_flags(argc, argv);
    std::fprintf(stderr, "knobs %s\n", knob_list().c_str());
    std::fflush(stderr);
    const Mode mode = tts_cpp::chatterbox::detail::effective_mode();
    const DWORD pipe_out_bytes = mode == Mode::Streaming ? 65536 : 4096;
    HANDLE h = CreateNamedPipeA(argv[3], PIPE_ACCESS_DUPLEX,
        PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
        1, pipe_out_bytes, 4096, 0, nullptr);
    if (h == INVALID_HANDLE_VALUE) throw std::runtime_error("pipe");
    if (!ConnectNamedPipe(h, nullptr) && GetLastError() != ERROR_PIPE_CONNECTED)
        throw std::runtime_error("pipe connect");
    tts_cpp::chatterbox::Engine tts({argv[1], argv[2], language});
    for (;;) {
        serve_one(h, tts, mode);
        DisconnectNamedPipe(h);
        if (!ConnectNamedPipe(h, nullptr) && GetLastError() != ERROR_PIPE_CONNECTED)
            throw std::runtime_error("pipe connect");
    }
}
