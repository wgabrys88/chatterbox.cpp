#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
struct mtl_number_rewrite {
    std::string source;
    std::string target;
};
struct mtl_number_result {
    std::string text;
    std::vector<mtl_number_rewrite> rewrites;
    std::string provider;
};
struct mtl_bpe {
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int> bpe_ranks;
    std::vector<std::string> added;
    bool load_from_arrays(const std::vector<std::string> & tokens,
                          const std::vector<int> & types,
                          const std::vector<std::string> & merges);
    std::vector<int32_t> encode(const std::string & text, const std::string & language_id) const;
    mtl_number_result verbalize_numbers(const std::string & text, const std::string & language_id) const;
    static std::string prepare_input(const std::string & text, const std::string & language_id);
    static std::string punc_norm(const std::string & text);
};
