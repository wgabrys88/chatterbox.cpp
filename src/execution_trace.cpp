#include "execution_trace.h"
#include <windows.h>
#include <bcrypt.h>
#include <psapi.h>
#include <vulkan/vulkan.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
namespace tts_cpp::chatterbox {
std::string json_string(const std::string& v) {
    std::string s = "\"";
    const char* hex = "0123456789abcdef";
    for (unsigned char c : v) {
        if (c == '"' || c == '\\') { s += '\\'; s += char(c); }
        else if (c < 32) { s += "\\u00"; s += hex[c >> 4]; s += hex[c & 15]; }
        else s += char(c);
    }
    return s + '"';
}
std::string json_number(double v) {
    if (!std::isfinite(v)) throw std::runtime_error("non-finite diagnostic value");
    std::ostringstream s; s.imbue(std::locale::classic()); s << std::setprecision(17) << v;
    return s.str();
}
std::string json_ids(const std::vector<int32_t>& v) {
    std::string s = "[";
    for (size_t i=0;i<v.size();++i) { if (i) s+=','; s+=std::to_string(v[i]); }
    return s + ']';
}
namespace {
struct Hash {
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    Hash() {
        if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0)
            throw std::runtime_error("SHA256 provider");
        if (BCryptCreateHash(algorithm, &hash, nullptr, 0, nullptr, 0, 0) < 0) {
            BCryptCloseAlgorithmProvider(algorithm, 0); algorithm=nullptr;
            throw std::runtime_error("SHA256 create");
        }
    }
    ~Hash() { if(hash) BCryptDestroyHash(hash); if(algorithm) BCryptCloseAlgorithmProvider(algorithm,0); }
    void add(const char* data, size_t n) {
        while(n) {
            const ULONG k=static_cast<ULONG>((std::min)(n,size_t(1048576)));
            if (BCryptHashData(hash, reinterpret_cast<PUCHAR>(const_cast<char*>(data)),k,0)<0)
                throw std::runtime_error("SHA256 update");
            data+=k; n-=k;
        }
    }
    std::string finish() {
        std::array<unsigned char,32> digest{};
        if (BCryptFinishHash(hash,digest.data(),ULONG(digest.size()),0)<0)
            throw std::runtime_error("SHA256 finish");
        std::ostringstream s; s<<std::hex<<std::setfill('0');
        for(auto x:digest) s<<std::setw(2)<<unsigned(x);
        return s.str();
    }
};
std::string lookup_field(Fields fields, const char* key, const std::string& fallback) {
    for (const auto& f : fields) if (f.first == key) return f.second;
    return fallback;
}
}
std::string sha256_text(const std::string& value) { Hash h; h.add(value.data(),value.size()); return h.finish(); }
std::string sha256_file(const std::string& path) {
    std::ifstream f(std::filesystem::u8path(path),std::ios::binary);
    if(!f) throw std::runtime_error("hash open: "+path);
    Hash h; std::array<char,65536> b;
    while(f.read(b.data(),b.size()) || f.gcount()) h.add(b.data(),size_t(f.gcount()));
    if(!f.eof()) throw std::runtime_error("hash read: "+path);
    return h.finish();
}
ExecutionTrace::ExecutionTrace(const std::string& wav) {
    const auto wav_path=std::filesystem::u8path(wav);
    const auto dir=wav_path.parent_path();
    dir_=dir.u8string();
    id_=dir.filename().u8string();
    bool valid=id_.size()==32 && id_.find_first_not_of("0123456789abcdef")==std::string::npos;
    if(!valid) id_="engine-"+std::to_string(GetCurrentProcessId())+"-"+std::to_string(GetTickCount64());
    auto open_table=[&](std::ofstream& f, const char* name){
        f.exceptions(std::ios::badbit|std::ios::failbit);
        f.open(dir / name, std::ios::binary|std::ios::app);
    };
    open_table(stream_, "events.jsonl");
    open_table(text_tokens_, "text_tokens.jsonl");
    open_table(t3_tokens_, "t3_tokens.jsonl");
    open_table(s3_tokens_, "s3_tokens.jsonl");
    event("trace_open","diagnostics",{{"caller_run_id",valid?json_string(id_):"null"},{"schema",json_string("tables_v3")}});
    std::fprintf(stderr,"tables_v3 run_id=%s\n",id_.c_str());
    std::fflush(stderr);
}
void ExecutionTrace::event(const std::string& name,const std::string& stage,Fields fields) {
    if(name=="unit_start")for(const auto& f:fields)if(f.first=="index")unit_=f.second;
    stream_<<"{\"schema_version\":3,\"component\":\"engine\",\"run_id\":"<<json_string(id_)
      <<",\"seq\":"<<seq_++<<",\"event\":"<<json_string(name)<<",\"stage\":"<<json_string(stage)
      <<",\"unit_index\":"<<unit_<<",\"monotonic_elapsed_s\":"<<json_number(elapsed(start_));
    for(const auto& f:fields) stream_<<','<<json_string(f.first)<<':'<<f.second;
    stream_<<"}\n"; stream_.flush();
}
void ExecutionTrace::token_rows(const std::string& name, const std::vector<int32_t>& ids) {
    std::ofstream* out=nullptr;
    const bool timed = (name=="t3" || name=="s3");
    if(name=="text") out=&text_tokens_;
    else if(name=="t3") out=&t3_tokens_;
    else if(name=="s3") out=&s3_tokens_;
    else throw std::runtime_error("unknown token table");
    for(size_t i=0;i<ids.size();++i){
        *out<<"{\"run_id\":"<<json_string(id_)<<",\"table\":"<<json_string(name)
            <<",\"i\":"<<i<<",\"id\":"<<ids[i]
            <<",\"t_s\":"<<json_number(timed ? double(i)*0.04 : 0.0)<<"}\n";
    }
    out->flush();
}
void ExecutionTrace::write_meta(const std::string& status, Fields fields) {
    const auto path=std::filesystem::u8path(dir_) / "meta.json";
    std::ofstream f;
    f.exceptions(std::ios::badbit|std::ios::failbit);
    f.open(path, std::ios::binary|std::ios::trunc);
    f<<"{\"schema_version\":3"
     <<",\"run_id\":"<<json_string(id_)
     <<",\"seed\":"<<lookup_field(fields,"seed",json_string("42"))
     <<",\"knobs\":"<<lookup_field(fields,"knobs","{}")
     <<",\"language_id\":"<<lookup_field(fields,"language_id","null")
     <<",\"git_heads\":{\"trident\":null,\"engine\":null,\"ggml\":null}"
     <<",\"ENGINE_REV\":null"
     <<",\"binary_sha256\":null"
     <<",\"gguf_sha256\":{\"t3\":null,\"s3\":null}"
     <<",\"wav_sha256\":"<<lookup_field(fields,"wav_sha256","null")
     <<",\"duration_s\":"<<lookup_field(fields,"duration_s","null")
     <<",\"sr\":"<<lookup_field(fields,"sr","24000")
     <<",\"status\":"<<json_string(status)
     <<"}";
    f<<"\n";
    f.flush();
}
void record_runtime_identity(ExecutionTrace* trace) {
    if(!trace)return;
    DWORD needed=0;
    if(!EnumProcessModules(GetCurrentProcess(),nullptr,0,&needed))throw std::runtime_error("module enumeration");
    std::vector<HMODULE> modules(needed/sizeof(HMODULE));
    if(!EnumProcessModules(GetCurrentProcess(),modules.data(),needed,&needed))throw std::runtime_error("module enumeration");
    std::string list="[";
    for(size_t i=0;i<modules.size();++i){
        std::vector<wchar_t> path(32768);
        DWORD n=GetModuleFileNameExW(GetCurrentProcess(),modules[i],path.data(),DWORD(path.size()));
        if(!n||n>=path.size())throw std::runtime_error("module filename");
        std::string name=std::filesystem::path(std::wstring(path.data(),n)).u8string();
        if(i)list+=',';list+="{\"path\":"+json_string(name)+",\"sha256\":"+json_string(sha256_file(name))+"}";
    }
    trace->event("loaded_modules","identity",{{"files",list+"]"}});
    VkApplicationInfo app{};app.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO;app.pApplicationName="chatterbox-diagnostics";app.apiVersion=VK_API_VERSION_1_0;
    VkInstanceCreateInfo create{};create.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;create.pApplicationInfo=&app;
    VkInstance instance=VK_NULL_HANDLE;
    VkResult status=vkCreateInstance(&create,nullptr,&instance);
    if(status!=VK_SUCCESS)throw std::runtime_error("Vulkan identity instance status "+std::to_string(status));
    try {
        uint32_t count=0;
        if(vkEnumeratePhysicalDevices(instance,&count,nullptr)!=VK_SUCCESS)throw std::runtime_error("Vulkan identity count");
        std::vector<VkPhysicalDevice> devices(count);
        if(vkEnumeratePhysicalDevices(instance,&count,devices.data())!=VK_SUCCESS)throw std::runtime_error("Vulkan identity devices");
        std::string inventory="[";
        for(uint32_t i=0;i<count;++i){
            VkPhysicalDeviceProperties props{};vkGetPhysicalDeviceProperties(devices[i],&props);
            VkPhysicalDeviceMemoryProperties mem{};vkGetPhysicalDeviceMemoryProperties(devices[i],&mem);
            uint64_t bytes=0;for(uint32_t j=0;j<mem.memoryHeapCount;++j)if(mem.memoryHeaps[j].flags&VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)bytes+=mem.memoryHeaps[j].size;
            if(i)inventory+=',';
            inventory+="{\"device\":"+json_string(props.deviceName)+",\"vendor_id\":"+std::to_string(props.vendorID)+",\"device_id\":"+std::to_string(props.deviceID)+",\"driver_version_raw\":"+std::to_string(props.driverVersion)+",\"api_version_raw\":"+std::to_string(props.apiVersion)+",\"device_local_bytes\":"+std::to_string(bytes)+"}";
        }
        trace->event("vulkan_driver_inventory","identity",{{"devices",inventory+"]"},{"selection_rule",json_string("match backend_identity.device; physical enumeration index is not the GGML index")}});
    } catch(...){vkDestroyInstance(instance,nullptr);throw;}
    vkDestroyInstance(instance,nullptr);
}

}
