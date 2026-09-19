#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct mtl_external_tokenizer_options {
    std::string python;
    std::string script;
    std::string source;
    std::string tts_source;
    std::string tokenizer_json;
    std::string cangjie_json;
    std::string dicta_model;
};

class mtl_external_tokenizer {
public:
    mtl_external_tokenizer(const mtl_external_tokenizer_options &, const std::string & language);
    ~mtl_external_tokenizer();
    mtl_external_tokenizer(const mtl_external_tokenizer &) = delete;
    mtl_external_tokenizer & operator=(const mtl_external_tokenizer &) = delete;
    std::string punctuation(const std::string & text);
    std::vector<int32_t> tokenize(const std::string & text);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
