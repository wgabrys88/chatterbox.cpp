#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
namespace tts_cpp::chatterbox::detail {
class mtl_tokenizer {
public:
    bool load_from_json(const std::string & json_blob);
    bool load_cangjie_json(const std::string & json_blob);
    std::vector<int32_t> encode(const std::string & text, const std::string & language_id) const;
    bool is_language_supported(const std::string & lang) const;
    static const std::vector<std::string> & supported_languages();
private:
    struct added_token {
        std::string content;
        int32_t     id;
    };
    std::unordered_map<std::string, int32_t> m_vocab;
    std::vector<std::string>                 m_id_to_token;
    std::unordered_map<std::string, int32_t> m_bpe_ranks;
    std::unordered_map<std::string, std::string> m_cangjie_word_to_code;
    std::unordered_map<std::string, std::vector<std::string>> m_cangjie_code_to_words;
    std::vector<added_token> m_added_tokens;
    std::string m_unk_token = "[UNK]";
    int32_t m_sot_id = -1;
    int32_t m_eot_id = -1;
    int32_t m_unk_id = -1;
    int32_t m_space_id = -1;
    void index_vocab();
    void bpe_word(const std::string & word, std::vector<int32_t> & out) const;
    std::string cangjie_normalize(const std::string & text) const;
};
}
