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

static std::string parse_flags(int argc, char** argv) {
    auto& k = tts_cpp::chatterbox::detail::runtime_knobs();
    std::string language;
    std::set<std::string> seen;
    int i = 4;
    while (i < argc) {
        const char* a = argv[i++];
        if(!seen.insert(a).second)throw std::runtime_error("duplicate flag");
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
        if (std::strcmp(a, "--seed") == 0) { k.seed = parse_int(v); continue; }
        if (std::strcmp(a, "--temperature") == 0) { k.temperature = parse_float(v); continue; }
        if (std::strcmp(a, "--top-p") == 0) { k.top_p = parse_float(v); continue; }
        if (std::strcmp(a, "--min-p") == 0) { k.min_p = parse_float(v); continue; }
        if (std::strcmp(a, "--repeat-penalty") == 0) { k.repeat_penalty = parse_float(v); continue; }
        if (std::strcmp(a, "--n-predict") == 0) { k.n_predict = parse_int(v); continue; }
        if (std::strcmp(a, "--cfm-steps") == 0) { k.cfm_steps = parse_int(v); continue; }
        if (std::strcmp(a, "--trim-fade") == 0) { k.trim_fade = parse_int(v); continue; }
        if (std::strcmp(a, "--stage") == 0) { k.stage = parse_int(v); continue; }
        if (std::strcmp(a, "--cut-x") == 0) { k.cut_x = parse_int(v); continue; }
#if defined(TTS_FAMILY_V3)
        if (std::strcmp(a, "--cfg-weight") == 0) { k.cfg_weight = parse_float(v); continue; }
        if (std::strcmp(a, "--cfm-cfg") == 0) { k.cfm_cfg = parse_float(v); continue; }
        if (std::strcmp(a, "--exaggeration") == 0) { k.exaggeration = parse_float(v); continue; }
#elif defined(TTS_FAMILY_GPT2)
        if (std::strcmp(a, "--top-k") == 0) { k.top_k = parse_int(v); continue; }
        if (std::strcmp(a, "--sil-count") == 0) { k.sil_count = parse_int(v); continue; }
        if (std::strcmp(a, "--s3gen-sil") == 0) { k.s3gen_sil = parse_int(v); continue; }
#endif
        throw std::runtime_error(a);
    }
#if defined(TTS_FAMILY_V3)
    if (language.empty()) throw std::runtime_error("language");
#endif
    if(k.n_predict<1 || k.temperature<0 || k.repeat_penalty<=0 || k.top_p<=0 || k.top_p>1
        || k.min_p<0 || k.min_p>1 || k.cfm_steps<1 || k.trim_fade<0
        || k.stage<0 || k.stage>3 || k.cut_x<0)
        throw std::runtime_error("generation argument out of range");
#if defined(TTS_FAMILY_V3)
    if(k.cfg_weight<0 || k.cfm_cfg<0 || k.exaggeration<0)throw std::runtime_error("V3 argument out of range");
#else
    if(k.top_k<0 || k.sil_count<0 || k.s3gen_sil<0)throw std::runtime_error("GPT2 argument out of range");
#endif
    return language;
}

// Resolved knobs, one key=value list shared by the stderr banner and the
// stats trailer so the client sees exactly what the engine ran with.
static std::string knob_list() {
    const auto& k = tts_cpp::chatterbox::detail::runtime_knobs();
    char buf[1024];
#if defined(TTS_FAMILY_V3)
    const int n = std::snprintf(buf, sizeof(buf),
        "seed=%d temperature=%g top_p=%g min_p=%g repeat_penalty=%g n_predict=%d cfg_weight=%g exaggeration=%g cfm_steps=%d cfm_cfg=%g trim_fade=%d stage=%d cut_x=%d",
        k.seed, k.temperature, k.top_p, k.min_p, k.repeat_penalty, k.n_predict,
        k.cfg_weight, k.exaggeration, k.cfm_steps, k.cfm_cfg, k.trim_fade, k.stage, k.cut_x);
#else
    const int n = std::snprintf(buf, sizeof(buf),
        "seed=%d temperature=%g top_k=%d top_p=%g min_p=%g repeat_penalty=%g n_predict=%d cfm_steps=%d trim_fade=%d sil_count=%d s3gen_sil=%d stage=%d cut_x=%d",
        k.seed, k.temperature, k.top_k, k.top_p, k.min_p, k.repeat_penalty, k.n_predict,
        k.cfm_steps, k.trim_fade, k.sil_count, k.s3gen_sil, k.stage, k.cut_x);
#endif
    if (n <= 0 || n >= (int)sizeof(buf)) throw std::runtime_error("knobs");
    return std::string(buf, (size_t)n);
}

static std::string stats_line(const tts_cpp::chatterbox::SynthesizeStats& s) {
    char buf[512];
    const int n = std::snprintf(buf, sizeof(buf),
        "predicted=%d dropped=%d eos=%d n_past=%d units=%d text_tokens=%d max_unit_predicted=%d stop_code=%d n_speech=%d cut_at=%d cut_reason=%d n_chunks=%d stage=%d ",
        s.predicted_count, s.dropped_count, s.eos, s.n_past, s.units, s.text_tokens, s.max_unit_predicted,
        s.stop_code, s.n_speech, s.cut_at, s.cut_reason, s.n_chunks, s.stage);
    if (n <= 0 || n >= (int)sizeof(buf)) throw std::runtime_error("stats");
    return std::string(buf, (size_t)n) + knob_list() + "\n";
}

// request: "<path>\n<nbytes>\n" + UTF-8 text. No mode or streaming fields.
static void serve_one(HANDLE h, std::unique_ptr<Engine>& tts, const EngineOptions& options) {
    std::unique_ptr<ExecutionTrace> trace;
    std::string stage="transport",path;
    bool published=false;
    try {
        path=read_line(h);validate_utf8(path);
        if(path.empty())throw std::runtime_error("empty output path");
        const auto dest=std::filesystem::u8path(path);
        if(!dest.is_absolute() || std::filesystem::exists(dest))throw std::runtime_error("output path must be absolute and unused");
        tts_cpp::chatterbox::detail::runtime_knobs().artifact_path=path;
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
            {"utf8_bytes",std::to_string(text.size())},{"effective_knobs",json_string(knob_list())},
            {"language",json_string(options.language_id)},{"output",json_string(path)}});
        stage="model_load";
        if(!tts){auto start=TraceClock::now();trace->event("model_load_start",stage);
            tts=std::make_unique<Engine>(options,trace.get());
            record_runtime_identity(trace.get());
            trace->event("model_load_end",stage,{{"host_wall_s",json_number(elapsed(start))},{"reused","false"}});
        }else trace->event("model_load_end",stage,{{"reused","true"}});
        stage="synthesis";SynthesizeStats stats;std::vector<float> pcm;
        tts->synthesize(text,pcm,&stats,trace.get());
        const int stage_n=tts_cpp::chatterbox::detail::effective_stage();
        if(stage_n==1||stage_n==2){
            const std::string tape=path+".tape.gguf";
            if(!std::filesystem::exists(std::filesystem::u8path(tape)))throw std::runtime_error("missing utterance GGUF");
            trace->event("output_write_end","output",{{"samples","0"},{"published","false"},
                {"tape",json_string(tape)},{"tape_sha256",json_string(sha256_file(tape))},
                {"stage",std::to_string(stage_n)}});
            trace->event("request_complete","request",{{"stats",json_string(stats_line(stats))}});
            stage="acknowledgement";
            const std::string line="ok "+stats_line(stats);write_exact(h,line.data(),DWORD(line.size()));
            if(!FlushFileBuffers(h))throw std::runtime_error("acknowledgement flush failed");
            return;
        }
        stage="output";trace->event("output_write_start",stage);
        const auto start=TraceClock::now();size_t clipped=0;std::vector<int16_t> samples(pcm.size());
        for(size_t i=0;i<pcm.size();++i){
            if(!std::isfinite(pcm[i]))throw std::runtime_error("non-finite output sample");
            if(pcm[i]<-1.f||pcm[i]>1.f)++clipped;
            samples[i]=int16_t(std::clamp(pcm[i],-1.f,1.f)*32767.f);
        }
        write_wav(path,samples);published=true;
        trace->event("output_write_end",stage,{{"samples",std::to_string(samples.size())},{"sample_rate","24000"},
            {"channels","1"},{"clipping_count",std::to_string(clipped)},{"finite_samples","true"},
            {"duration_s",json_number(double(samples.size())/24000)},{"wav_sha256",json_string(sha256_file(path))},
            {"host_wall_s",json_number(elapsed(start))},{"published","true"}});
        trace->event("request_complete","request",{{"stats",json_string(stats_line(stats))}});
        stage="acknowledgement";
        const std::string line="ok "+stats_line(stats);write_exact(h,line.data(),DWORD(line.size()));
        if(!FlushFileBuffers(h))throw std::runtime_error("acknowledgement flush failed");
    } catch(const std::exception& e) {
        std::fprintf(stderr,"request failed stage=%s error=%s\n",stage.c_str(),e.what());std::fflush(stderr);
        // Preserve a completed WAV on transport-only failure; the client marks it unknown.
        if(published&&stage!="acknowledgement") {std::error_code ec;std::filesystem::remove(std::filesystem::u8path(path),ec);}
        if(trace)trace->event("request_failed",stage,{{"error",json_string(e.what())},{"completed_wav_preserved",published&&stage=="acknowledgement"?"true":"false"}});
        const std::string line="err "+one_line(e.what())+"\n";
        write_exact(h,line.data(),DWORD(line.size()));
        FlushFileBuffers(h);
    }
}
int main(int argc,char** argv) {
    try {
        if(argc<4)throw std::runtime_error("usage: chatterbox-server T3 S3 PIPE [flags]");
        const std::string language=parse_flags(argc,argv);
        std::fprintf(stderr,"knobs %s\n",knob_list().c_str());std::fflush(stderr);
        HANDLE h=CreateNamedPipeA(argv[3],PIPE_ACCESS_DUPLEX,
            PIPE_TYPE_BYTE|PIPE_READMODE_BYTE|PIPE_WAIT|PIPE_REJECT_REMOTE_CLIENTS,1,4096,4096,0,nullptr);
        if(h==INVALID_HANDLE_VALUE)throw std::runtime_error("pipe create");
        std::unique_ptr<Engine> tts;const EngineOptions options{argv[1],argv[2],language};
        try {
            for(;;){
                if(!ConnectNamedPipe(h,nullptr)&&GetLastError()!=ERROR_PIPE_CONNECTED)throw std::runtime_error("pipe connect");
                serve_one(h,tts,options);
                if(!DisconnectNamedPipe(h))throw std::runtime_error("pipe disconnect");
            }
        }catch(...){CloseHandle(h);throw;}
    }catch(const std::exception& e){std::fprintf(stderr,"fatal: %s\n",e.what());return 1;}
}
