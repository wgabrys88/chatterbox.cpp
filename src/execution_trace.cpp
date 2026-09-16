#include "execution_trace.h"
#include <windows.h>
#include <bcrypt.h>
#include <psapi.h>
#include <vulkan/vulkan.h>
#include <algorithm>
#include <array>
#include <cmath>
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
    id_=std::filesystem::u8path(wav).parent_path().filename().u8string();
    bool valid=id_.size()==32 && id_.find_first_not_of("0123456789abcdef")==std::string::npos;
    if(!valid) id_="engine-"+std::to_string(GetCurrentProcessId())+"-"+std::to_string(GetTickCount64());
    stream_.exceptions(std::ios::badbit|std::ios::failbit);
    stream_.open(std::filesystem::u8path(wav+".engine.jsonl"),std::ios::binary|std::ios::trunc);
    event("trace_open","diagnostics",{{"caller_run_id",valid?json_string(id_):"null"}});
}
void ExecutionTrace::event(const std::string& name,const std::string& stage,Fields fields) {
    if(name=="unit_start")for(const auto& f:fields)if(f.first=="index")unit_=f.second;
    stream_<<"{\"schema_version\":1,\"component\":\"engine\",\"run_id\":"<<json_string(id_)
      <<",\"seq\":"<<seq_++<<",\"event\":"<<json_string(name)<<",\"stage\":"<<json_string(stage)
      <<",\"unit_index\":"<<unit_<<",\"monotonic_elapsed_s\":"<<json_number(elapsed(start_));
    for(const auto& f:fields) stream_<<','<<json_string(f.first)<<':'<<f.second;
    stream_<<"}\n"; stream_.flush();
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
