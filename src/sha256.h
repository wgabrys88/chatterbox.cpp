#pragma once
#include <string>
namespace tts_cpp::chatterbox {
std::string sha256_text(const std::string& value);
std::string sha256_file(const std::string& path);
}
