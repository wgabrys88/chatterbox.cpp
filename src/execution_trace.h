#pragma once
#include <chrono>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>
namespace tts_cpp::chatterbox {
class ExecutionTrace;
using Fields = std::initializer_list<std::pair<std::string, std::string>>;
std::string json_string(const std::string& value);
std::string json_number(double value);
std::string json_ids(const std::vector<int32_t>& values);
void record_runtime_identity(ExecutionTrace* trace);
std::string sha256_text(const std::string& value);
std::string sha256_file(const std::string& path);
using TraceClock = std::chrono::steady_clock;
inline double elapsed(TraceClock::time_point start) {
    return std::chrono::duration<double>(TraceClock::now() - start).count();
}
class ExecutionTrace {
    std::ofstream stream_;
    std::ofstream text_tokens_;
    std::ofstream t3_tokens_;
    std::ofstream s3_tokens_;
    std::string id_;
    std::string dir_;
    size_t seq_ = 0;
    std::string unit_ = "null";
    TraceClock::time_point start_ = TraceClock::now();
public:
    explicit ExecutionTrace(const std::string& wav_path);
    void event(const std::string& event, const std::string& stage, Fields fields = {});
    void token_rows(const std::string& table, const std::vector<int32_t>& ids);
    void write_meta(const std::string& status, Fields fields = {});
    const std::string& run_id() const { return id_; }
};
inline void trace_event(ExecutionTrace* trace, const std::string& name,
                        const std::string& stage, Fields fields = {}) {
    if (trace) trace->event(name, stage, fields);
}
inline void trace_tokens(ExecutionTrace* trace, const std::string& table,
                         const std::vector<int32_t>& ids) {
    if (trace) trace->token_rows(table, ids);
}
}
