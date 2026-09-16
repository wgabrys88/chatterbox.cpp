#pragma once
#include <cstddef>
#include <string>
#include <vector>
namespace tts_cpp::chatterbox {
class ExecutionTrace;
struct TextEdit {
    size_t original_begin, original_end, prepared_begin, prepared_end;
    std::string replacement, rule;
};
struct TextSpan { size_t begin, end; };
struct PreparedText {
    std::string text;
    std::vector<TextEdit> edits;
    std::vector<TextSpan> atoms;
    std::vector<TextSpan> unhandled;
};
void validate_utf8(const std::string& text);
PreparedText prepare_text(const std::string& text, bool english, ExecutionTrace* trace);
}
