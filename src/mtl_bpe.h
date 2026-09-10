#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
struct mtl_bpe {
    std::unordered_map<std::string, int32_t> token_to_id;
    std::vector<std::string>                 id_to_token;
    std::unordered_map<std::string, int>     bpe_ranks;
    std::vector<std::string>                 added;
    bool load_from_arrays(const std::vector<std::string> & tokens,
                          const std::vector<int> & types,
                          const std::vector<std::string> & merges);
    std::vector<int32_t> encode(const std::string & text, const std::string & language_id) const;
    static std::string punc_norm(const std::string & text);
};
