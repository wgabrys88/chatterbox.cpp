#pragma once
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// One append-only stream: LE u32 JSON length, LE u64 payload length, JSON, raw bytes.
// No tensor formatting, duplicate hashes or inference during capture.
namespace diagnostic {
inline std::ofstream stream;
inline std::string root;
inline std::mutex mutex;
inline uint64_t sequence = 0;
inline std::string quote(const std::string& value) {
    std::ostringstream out; out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
        else out << c;
    }
    out << '"'; return out.str();
}
inline void open(const std::string& prefix) {
    std::lock_guard<std::mutex> lock(mutex);
    if (stream.is_open()) stream.close();
    root = prefix; sequence = 0;
    if (root.empty()) return;
    stream.open(root + ".capture", std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("cannot open replay capture");
}
inline void record(const std::string& prefix, const std::string& kind,
                   const std::string& fields, const void* data = nullptr, uint64_t bytes = 0) {
    if (prefix.empty()) return;
    std::lock_guard<std::mutex> lock(mutex);
    const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const std::string header = "{\"schema\":2,\"sequence\":" + std::to_string(sequence++) +
        ",\"utc_ns\":" + std::to_string(now) + ",\"scope\":" + quote(prefix.substr(root.size())) +
        ",\"event\":" + quote(kind) + (fields.empty() ? "" : "," + fields) + "}";
    const uint32_t length = static_cast<uint32_t>(header.size());
    stream.write(reinterpret_cast<const char*>(&length), sizeof(length));
    stream.write(reinterpret_cast<const char*>(&bytes), sizeof(bytes));
    stream.write(header.data(), header.size());
    if (bytes) stream.write(static_cast<const char*>(data), bytes);
    stream.flush();
    if (!stream) throw std::runtime_error("replay capture write failed");
}
inline void event(const std::string& prefix, const std::string& kind, const std::string& fields) {
    record(prefix, kind, fields);
}
inline void raw(const std::string& prefix, const std::string& name, const void* data,
                uint64_t bytes, const std::string& dtype, const std::vector<int64_t>& shape) {
    if (prefix.empty()) return;
    std::ostringstream dims;
    for (size_t i = 0; i < shape.size(); ++i) { if (i) dims << ','; dims << shape[i]; }
    record(prefix, "tensor", "\"name\":" + quote(name) + ",\"dtype\":" + quote(dtype) +
        ",\"endian\":\"little\",\"layout\":\"C\",\"shape\":[" + dims.str() + "]", data, bytes);
}
template<class T> inline void tensor(const std::string& prefix, const std::string& name,
                                    const std::vector<T>& values, std::vector<int64_t> shape = {}) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, int32_t>);
    if (shape.empty()) shape = {static_cast<int64_t>(values.size())};
    raw(prefix, name, values.data(), values.size() * sizeof(T),
        std::is_same_v<T,float> ? "f32" : std::is_same_v<T,double> ? "f64" : "i32", shape);
}
}
