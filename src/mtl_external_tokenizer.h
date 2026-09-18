#pragma once
#include <cstdint>
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
struct mtl_external_tokenizer_result {
    std::string tokenizer_input;
    std::vector<int32_t> ids;
};
std::string mtl_external_punc_norm(const mtl_external_tokenizer_options & options, const std::string & text);
mtl_external_tokenizer_result mtl_external_tokenize(const mtl_external_tokenizer_options &, const std::string &, const std::string &);
