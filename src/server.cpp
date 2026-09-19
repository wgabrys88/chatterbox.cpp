#include "tts-cpp/chatterbox/engine.h"
#include "tts-cpp/chatterbox/runtime_knobs.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <windows.h>
#include <filesystem>
#include <memory>
#include <map>
using namespace tts_cpp::chatterbox;

static void write_wav(const std::string& path, const std::vector<float>& pcm) {
    std::vector<int16_t> samples(pcm.size());
    for (size_t i = 0; i < pcm.size(); ++i)
        samples[i] = int16_t(std::clamp(pcm[i], -1.f, 1.f) * 32767.f);
    std::ofstream f(std::filesystem::u8path(path), std::ios::binary);
    const uint32_t data = uint32_t(samples.size() * 2);
    auto w16 = [&](uint16_t v) { f.write(reinterpret_cast<const char*>(&v), 2); };
    auto w32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
    f.write("RIFF", 4); w32(36 + data); f.write("WAVEfmt ", 8);
    w32(16); w16(1); w16(1); w32(24000); w32(48000); w16(2); w16(16);
    f.write("data", 4); w32(data);
    f.write(reinterpret_cast<const char*>(samples.data()), data);
}

static std::string read_line(HANDLE h) {
    std::string line; char c; DWORD n=0;
    while(ReadFile(h,&c,1,&n,nullptr)&&n) {
        if(c=='\n')return line;
        line+=c;
    }
    throw std::runtime_error("truncated protocol line");
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

static EngineOptions parse_flags(int argc, char** argv) {
    auto& k = tts_cpp::chatterbox::detail::runtime_knobs();
    EngineOptions options{argv[1], argv[2]};
    std::map<std::string, int*> integers = {
        {"--seed", &k.seed}, {"--n-predict", &k.n_predict},
        {"--cfm-steps", &k.cfm_steps}, {"--trim-fade-samples", &k.trim_fade},
#if defined(TTS_FAMILY_GPT2)
        {"--top-k", &k.top_k},
#endif
    };
    std::map<std::string, float*> floats = {
        {"--temperature", &k.temperature}, {"--top-p", &k.top_p}, {"--repeat-penalty", &k.repeat_penalty},
#if defined(TTS_FAMILY_V3)
        {"--min-p", &k.min_p}, {"--cfg-weight", &k.cfg_weight},
        {"--exaggeration", &k.exaggeration}, {"--cfm-cfg", &k.cfm_cfg},
#endif
    };
    std::map<std::string, std::string*> strings = {
        {"--language", &options.language_id}, {"--tokenizer-python", &options.tokenizer_python},
        {"--tokenizer-script", &options.tokenizer_script}, {"--tokenizer-source", &options.tokenizer_source},
        {"--tokenizer-tts-source", &options.tokenizer_tts_source}, {"--tokenizer-json", &options.tokenizer_json},
        {"--cangjie-json", &options.cangjie_json}, {"--dicta-model", &options.dicta_model},
    };
    for (int i = 4; i < argc; i += 2) {
        const std::string flag = argv[i];
        if (integers.count(flag)) *integers[flag] = std::stoi(argv[i + 1]);
        else if (floats.count(flag)) *floats[flag] = std::stof(argv[i + 1]);
        else *strings.at(flag) = argv[i + 1];
    }
    for (char& c : options.language_id) if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
    return options;
}

static std::string stats_line(const tts_cpp::chatterbox::SynthesizeStats& s) {
    char buf[256];
    const int n = std::snprintf(buf, sizeof(buf),
        "predicted=%d dropped=%d eos=%d n_past=%d units=%d text_tokens=%d max_unit_predicted=%d ",
        s.predicted_count, s.dropped_count, s.eos, s.n_past, s.units, s.text_tokens, s.max_unit_predicted);
    return std::string(buf, (size_t)n) + "\n";
}

static void serve_one(HANDLE h, Engine& tts) {
    const auto path = read_line(h);
    std::string text(std::stoul(read_line(h)), '\0');
    if (!read_exact(h, text.data(), DWORD(text.size()))) throw std::runtime_error("pipe read");
    SynthesizeStats stats;
    std::vector<float> pcm;
    tts.synthesize(text, pcm, &stats);
    write_wav(path, pcm);
    const auto line = "ok " + stats_line(stats);
    write_exact(h, line.data(), DWORD(line.size()));
    FlushFileBuffers(h);
}
int main(int argc,char** argv) {
    try {
        const auto options=parse_flags(argc,argv);
        for (int i = 4; i < argc; i += 2) std::fprintf(stderr, "%s=%s ", argv[i], argv[i + 1]);
        std::fputc('\n', stderr); std::fflush(stderr);
        HANDLE h=CreateNamedPipeA(argv[3],PIPE_ACCESS_DUPLEX,
            PIPE_TYPE_BYTE|PIPE_READMODE_BYTE|PIPE_WAIT|PIPE_REJECT_REMOTE_CLIENTS,1,4096,4096,0,nullptr);
        if(h==INVALID_HANDLE_VALUE)throw std::runtime_error("pipe create");
        std::unique_ptr<Engine> tts;
        try {
            for(;;){
                if(!ConnectNamedPipe(h,nullptr)&&GetLastError()!=ERROR_PIPE_CONNECTED)throw std::runtime_error("pipe connect");
                if (!tts) tts = std::make_unique<Engine>(options);
                serve_one(h, *tts);
                if(!DisconnectNamedPipe(h))throw std::runtime_error("pipe disconnect");
            }
        }catch(...){CloseHandle(h);throw;}
    }catch(const std::exception& e){std::fprintf(stderr,"fatal: %s\n",e.what());return 1;}
}
