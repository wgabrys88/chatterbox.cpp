#pragma once
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// Audit IO is deliberately outside the numerical path. All failures are fatal.
namespace diagnostic {
inline std::string quote(const std::string& value) {
    std::ostringstream out; out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
        else out << c;
    }
    out << '"'; return out.str();
}
inline void event(const std::string& prefix, const std::string& kind, const std::string& fields) {
    if (prefix.empty()) return;
    std::ofstream out(prefix + ".audit.jsonl", std::ios::app | std::ios::binary);
    out << "{\"schema\":1,\"event\":" << quote(kind) << ',' << fields << "}\n";
    out.flush();
    if (!out) throw std::runtime_error("audit event write failed: " + prefix);
}
template<class T> inline void tensor(const std::string& prefix, const std::string& name,
                                     const std::vector<T>& values,
                                     std::vector<int64_t> shape = {}) {
    if (prefix.empty()) return;
    const std::string dtype = std::is_same_v<T,float> ? "f32" : std::is_same_v<T,double> ? "f64" : "i32";
    const auto path = prefix + "." + name + "." + dtype;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(values.data()), values.size()*sizeof(T));
    out.flush();
    if (!out) throw std::runtime_error("audit tensor write failed: " + path);
    if (shape.empty()) shape.push_back(static_cast<int64_t>(values.size()));
    size_t count = 1; std::ostringstream dims;
    for (size_t i=0; i<shape.size(); ++i) { count *= shape[i]; if(i) dims << ','; dims << shape[i]; }
    if (count != values.size()) throw std::runtime_error("audit tensor shape mismatch: " + path);
    uint64_t hash=1469598103934665603ull;
    const auto* bytes=reinterpret_cast<const uint8_t*>(values.data());
    for(size_t i=0;i<values.size()*sizeof(T);++i) { hash ^= bytes[i]; hash *= 1099511628211ull; }
    std::ostringstream hex; hex << std::hex << std::setw(16) << std::setfill('0') << hash;
    event(prefix,"tensor","\"name\":"+quote(name)+",\"file\":"+quote(std::filesystem::path(path).filename().string())+
        ",\"dtype\":"+quote(dtype)+",\"endian\":\"little\",\"layout\":\"C\",\"shape\":["+dims.str()+
        "],\"bytes\":"+std::to_string(values.size()*sizeof(T))+",\"fnv64\":"+quote(hex.str()));
}
}
