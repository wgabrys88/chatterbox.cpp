#pragma once
#include <cstddef>
#include <string>
#include <vector>
namespace tts_cpp::chatterbox {
struct PreparedText { std::string text; };
PreparedText prepare_text(const std::string& text);
}
