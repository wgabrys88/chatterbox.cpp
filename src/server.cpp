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
#include <filesystem>
#include <limits>
#include <memory>
#include <set>
#include <cerrno>
#include <cmath>
#include "execution_trace.h"
#include "text_prepare.h"
using namespace tts_cpp::chatterbox;

static void write_wav(const std::string& path, const std::vector<int16_t>& pcm) {
    if(pcm.empty() || pcm.size()>(size_t(UINT32_MAX)-36)/2) throw std::runtime_error("WAV size limit");
    const auto dest=std::filesystem::u8path(path), tmp=std::filesystem::u8path(path+".tmp");
    const uint32_t data=uint32_t(pcm.size()*2),riff=36+data,sr=24000;
    try {
        std::ofstream f; f.exceptions(std::ios::badbit|std::ios::failbit); f.open(tmp,std::ios::binary|std::ios::trunc);
        auto w16=[&](uint16_t v){f.write(reinterpret_cast<const char*>(&v),2);};
        auto w32=[&](uint32_t v){f.write(reinterpret_cast<const char*>(&v),4);};
        f.write("RIFF",4);w32(riff);f.write("WAVEfmt ",8);w32(16);w16(1);w16(1);w32(sr);w32(sr*2);w16(2);w16(16);
        f.write("data",4);w32(data);f.write(reinterpret_cast<const char*>(pcm.data()),std::streamsize(data));f.flush();f.close();
        if(!MoveFileExW(tmp.c_str(),dest.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))
            throw std::runtime_error("WAV replace Windows error "+std::to_string(GetLastError()));
    } catch(...) { std::error_code ec;std::filesystem::remove(tmp,ec);throw; }
}

static std::string read_line(HANDLE h) {
    std::string line; char c; DWORD n=0;
    while(ReadFile(h,&c,1,&n,nullptr)&&n) {
        if(c=='\n')return line;
        if(c=='\r'||c=='\0')throw std::runtime_error("invalid protocol line control");
        if(line.size()>=32768)throw std::runtime_error("protocol line too long");
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
    errno=0;
    const long x = std::strtol(s, &end, 10);
    if (!s || end == s || *end || errno==ERANGE || x<INT_MIN || x>INT_MAX) throw std::runtime_error("knob int");
    return (int)x;
}

static float parse_float(const char* s) {
    char* end = nullptr;
    errno=0;
    const float x = std::strtof(s, &end);
    if (!s || end == s || *end || errno==ERANGE || !std::isfinite(x)) throw std::runtime_error("knob float");
    return x;
}

struct ServerFlags {
    std::string language, tokenizer_python, tokenizer_script, tokenizer_source, tokenizer_tts_source, tokenizer_json, cangjie_json, dicta_model;
};
static ServerFlags parse_flags(int argc, char** argv) {
    auto& k = tts_cpp::chatterbox::detail::runtime_knobs();
    ServerFlags flags;
    std::set<std::string> seen;
    int i = 4;
    while (i < argc) {
        const char* a = argv[i++];
        if(!seen.insert(a).second)throw std::runtime_error("duplicate flag");
        if (i >= argc) throw std::runtime_error(a);
        const char* v = argv[i++];
        if (std::strcmp(a, "--language") == 0) {
#if defined(TTS_FAMILY_V3)
            flags.language = v;
            continue;
#else
            throw std::runtime_error(a);
#endif
        }
#if defined(TTS_FAMILY_V3)
        if (std::strcmp(a, "--tokenizer-python") == 0) { flags.tokenizer_python = v; continue; }
        if (std::strcmp(a, "--tokenizer-script") == 0) { flags.tokenizer_script = v; continue; }
        if (std::strcmp(a, "--tokenizer-source") == 0) { flags.tokenizer_source = v; continue; }
        if (std::strcmp(a, "--tokenizer-tts-source") == 0) { flags.tokenizer_tts_source = v; continue; }
        if (std::strcmp(a, "--tokenizer-json") == 0) { flags.tokenizer_json = v; continue; }
        if (std::strcmp(a, "--cangjie-json") == 0) { flags.cangjie_json = v; continue; }
        if (std::strcmp(a, "--dicta-model") == 0) { flags.dicta_model = v; continue; }
#endif
        if (std::strcmp(a, "--seed") == 0) { k.seed = parse_int(v); continue; }
        if (std::strcmp(a, "--temperature") == 0) { k.temperature = parse_float(v); continue; }
        if (std::strcmp(a, "--top-p") == 0) { k.top_p = parse_float(v); continue; }
        if (std::strcmp(a, "--repeat-penalty") == 0) { k.repeat_penalty = parse_float(v); continue; }
        if (std::strcmp(a, "--n-predict") == 0) { k.n_predict = parse_int(v); continue; }
        if (std::strcmp(a, "--cfm-steps") == 0) { k.cfm_steps = parse_int(v); continue; }
        if (std::strcmp(a, "--trim-fade-samples") == 0) { k.trim_fade = parse_int(v); continue; }
#if defined(TTS_FAMILY_V3)
        if (std::strcmp(a, "--min-p") == 0) { k.min_p = parse_float(v); continue; }
        if (std::strcmp(a, "--cfg-weight") == 0) { k.cfg_weight = parse_float(v); continue; }
        if (std::strcmp(a, "--exaggeration") == 0) { k.exaggeration = parse_float(v); continue; }
        if (std::strcmp(a, "--cfm-cfg") == 0) { k.cfm_cfg = parse_float(v); continue; }
#elif defined(TTS_FAMILY_GPT2)
        if (std::strcmp(a, "--top-k") == 0) { k.top_k = parse_int(v); continue; }
#endif
        throw std::runtime_error(a);
    }
    auto require_flag = [&](const char* name) {
        if (!seen.count(name)) throw std::runtime_error(std::string("missing required flag ") + name);
    };
    require_flag("--seed");
    require_flag("--temperature");
    require_flag("--top-p");
    require_flag("--repeat-penalty");
    require_flag("--n-predict");
    require_flag("--cfm-steps");
    require_flag("--trim-fade-samples");
#if defined(TTS_FAMILY_V3)
    require_flag("--min-p");
    require_flag("--cfg-weight");
    require_flag("--exaggeration");
    require_flag("--cfm-cfg");
    for (char & c : flags.language) if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + ('a' - 'A'));
    if (flags.language.empty()) throw std::runtime_error("language");
    if (flags.tokenizer_python.empty() || flags.tokenizer_script.empty() || flags.tokenizer_source.empty() || flags.tokenizer_tts_source.empty() || flags.tokenizer_json.empty() || flags.cangjie_json.empty() || flags.dicta_model.empty())
        throw std::runtime_error("official tokenizer configuration");
#endif
    if(k.n_predict<1 || k.cfm_steps<1 || k.trim_fade<0 || k.temperature<0 || k.repeat_penalty<=0 || k.top_p<=0 || k.top_p>1)
        throw std::runtime_error("generation argument out of range");
#if defined(TTS_FAMILY_V3)
    if(k.min_p<0 || k.min_p>1)throw std::runtime_error("V3 argument out of range");
#else
    require_flag("--top-k");
    if(k.top_k<0)throw std::runtime_error("top-k out of range");
#endif
    return flags;
}

static std::string knobs_json() {
    const auto& k = tts_cpp::chatterbox::detail::runtime_knobs();
    char buf[768];
#if defined(TTS_FAMILY_V3)
    const int n = std::snprintf(buf, sizeof(buf),
        "{\"seed\":%d,\"temperature\":%.17g,\"top-p\":%.17g,\"repeat-penalty\":%.17g,"
        "\"n-predict\":%d,\"min-p\":%.17g,\"cfg-weight\":%.17g,\"exaggeration\":%.17g,"
        "\"cfm-steps\":%d,\"cfm-cfg\":%.17g,\"trim-fade-samples\":%d}",
        k.seed, k.temperature, k.top_p, k.repeat_penalty, k.n_predict,
        k.min_p, k.cfg_weight, k.exaggeration, k.cfm_steps, k.cfm_cfg, k.trim_fade);
#else
    const int n = std::snprintf(buf, sizeof(buf),
        "{\"seed\":%d,\"temperature\":%.17g,\"top-k\":%d,\"top-p\":%.17g,"
        "\"repeat-penalty\":%.17g,\"n-predict\":%d,\"cfm-steps\":%d,\"trim-fade-samples\":%d}",
        k.seed, k.temperature, k.top_k, k.top_p, k.repeat_penalty, k.n_predict, k.cfm_steps, k.trim_fade);
#endif
    if (n <= 0 || n >= (int)sizeof(buf)) throw std::runtime_error("knobs json");
    return std::string(buf, (size_t)n);
}

static std::string language_json(const std::string& language) {
    return language.empty() ? std::string("null") : json_string(language);
}

static std::string stats_line(const tts_cpp::chatterbox::SynthesizeStats& s) {
    char buf[256];
    const int n = std::snprintf(buf, sizeof(buf),
        "predicted=%d dropped=%d eos=%d n_past=%d units=%d text_tokens=%d max_unit_predicted=%d ",
        s.predicted_count, s.dropped_count, s.eos, s.n_past, s.units, s.text_tokens, s.max_unit_predicted);
    if (n <= 0 || n >= (int)sizeof(buf)) throw std::runtime_error("stats");
    return std::string(buf, (size_t)n) + "\n";
}

static void serve_one(HANDLE h, std::unique_ptr<Engine>& tts, const EngineOptions& options) {
    std::unique_ptr<ExecutionTrace> trace;
    std::string stage="transport",path;
    bool published=false;
    try {
        path=read_line(h);validate_utf8(path);
        if(path.empty())throw std::runtime_error("empty output path");
        const auto dest=std::filesystem::u8path(path);
        if(!dest.is_absolute() || std::filesystem::exists(dest))throw std::runtime_error("output path must be absolute and unused");
        trace=std::make_unique<ExecutionTrace>(path);
        const std::string length=read_line(h);
        if(length.empty()||length.find_first_not_of("0123456789")!=std::string::npos)throw std::runtime_error("invalid payload length");
        size_t size=0;
        for(char c:length){if(size>(size_t(INT_MAX)-unsigned(c-'0'))/10)throw std::runtime_error("payload length overflow");size=size*10+unsigned(c-'0');}
        if(!size)throw std::runtime_error("empty payload");
        std::string text(size,'\0');
        if(!read_exact(h,text.data(),DWORD(size)))throw std::runtime_error("truncated text payload");
        validate_utf8(text);
        trace->event("request_start","request",{{"original_text",json_string(text)},{"input_sha256",json_string(sha256_text(text))},
            {"utf8_bytes",std::to_string(text.size())},{"effective_knobs",knobs_json()},
            {"language",json_string(options.language_id)},{"output",json_string(path)}});
        stage="model_load";
        if(!tts){auto start=TraceClock::now();trace->event("model_load_start",stage);
            tts=std::make_unique<Engine>(options,trace.get());
            record_runtime_identity(trace.get());
            trace->event("model_load_end",stage,{{"host_wall_s",json_number(elapsed(start))},{"reused","false"}});
        }else trace->event("model_load_end",stage,{{"reused","true"}});
        stage="synthesis";SynthesizeStats stats;std::vector<float> pcm;
        tts->synthesize(text,pcm,&stats,trace.get());
        stage="output";trace->event("output_write_start",stage);
        const auto start=TraceClock::now();size_t clipped=0;std::vector<int16_t> samples(pcm.size());
        for(size_t i=0;i<pcm.size();++i){
            if(!std::isfinite(pcm[i]))throw std::runtime_error("non-finite output sample");
            if(pcm[i]<-1.f||pcm[i]>1.f)++clipped;
            samples[i]=int16_t(std::clamp(pcm[i],-1.f,1.f)*32767.f);
        }
        write_wav(path,samples);published=true;
        const std::string wav_sha=sha256_file(path);
        const std::string duration=json_number(double(samples.size())/24000);
        const auto& rk=tts_cpp::chatterbox::detail::runtime_knobs();
        trace->event("output_write_end",stage,{{"samples",std::to_string(samples.size())},{"sample_rate","24000"},
            {"channels","1"},{"clipping_count",std::to_string(clipped)},{"finite_samples","true"},
            {"duration_s",duration},{"wav_sha256",json_string(wav_sha)},
            {"host_wall_s",json_number(elapsed(start))},{"published","true"}});
        trace->event("request_complete","request",{{"stats",json_string(stats_line(stats))}});
        trace->write_meta("execution_complete",{
            {"seed",json_string(std::to_string(rk.seed))},
            {"knobs",knobs_json()},
            {"language_id",language_json(options.language_id)},
            {"wav_sha256",json_string(wav_sha)},
            {"duration_s",duration},
            {"sr","24000"}});
        stage="acknowledgement";
        const std::string line="ok "+stats_line(stats);write_exact(h,line.data(),DWORD(line.size()));
        if(!FlushFileBuffers(h))throw std::runtime_error("acknowledgement flush failed");
    } catch(const std::exception& e) {
        std::fprintf(stderr,"request failed stage=%s error=%s\n",stage.c_str(),e.what());std::fflush(stderr);

        if(published&&stage!="acknowledgement") {std::error_code ec;std::filesystem::remove(std::filesystem::u8path(path),ec);}
        if(trace){
            const auto& rk=tts_cpp::chatterbox::detail::runtime_knobs();
            trace->event("request_failed",stage,{{"error",json_string(e.what())},{"completed_wav_preserved",published&&stage=="acknowledgement"?"true":"false"}});
            trace->write_meta("failed",{
                {"seed",json_string(std::to_string(rk.seed))},
                {"knobs",knobs_json()},
                {"language_id",language_json(options.language_id)},
                {"sr","24000"}});
        }
        const std::string line="err "+one_line(e.what())+"\n";
        write_exact(h,line.data(),DWORD(line.size()));
        FlushFileBuffers(h);
    }
}
int main(int argc,char** argv) {
    try {
        if(argc<4)throw std::runtime_error("usage: chatterbox-server T3 S3 PIPE [flags]");
        const auto flags=parse_flags(argc,argv);
        std::fprintf(stderr,"knobs %s\n",knobs_json().c_str());std::fflush(stderr);
        HANDLE h=CreateNamedPipeA(argv[3],PIPE_ACCESS_DUPLEX,
            PIPE_TYPE_BYTE|PIPE_READMODE_BYTE|PIPE_WAIT|PIPE_REJECT_REMOTE_CLIENTS,1,4096,4096,0,nullptr);
        if(h==INVALID_HANDLE_VALUE)throw std::runtime_error("pipe create");
        std::unique_ptr<Engine> tts;const EngineOptions options{argv[1],argv[2],flags.language,flags.tokenizer_python,flags.tokenizer_script,flags.tokenizer_source,flags.tokenizer_tts_source,flags.tokenizer_json,flags.cangjie_json,flags.dicta_model};
        try {
            for(;;){
                if(!ConnectNamedPipe(h,nullptr)&&GetLastError()!=ERROR_PIPE_CONNECTED)throw std::runtime_error("pipe connect");
                serve_one(h,tts,options);
                if(!DisconnectNamedPipe(h))throw std::runtime_error("pipe disconnect");
            }
        }catch(...){CloseHandle(h);throw;}
    }catch(const std::exception& e){std::fprintf(stderr,"fatal: %s\n",e.what());return 1;}
}
