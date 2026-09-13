#include "mtl_bpe.h"
#include <algorithm>
#include <cstdint>
#include <regex>
#include <stdexcept>
#include <windows.h>
#pragma comment(lib, "Normaliz.lib")
static std::string to_lower_ascii(const std::string & s) {
    std::string o = s;
    for (char & c : o) if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
    return o;
}
static std::string nfkd_utf8(const std::string & s) {
    bool ascii = true;
    for (unsigned char c : s) if (c >= 128) { ascii = false; break; }
    if (ascii) return s;
    int wlen = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    if (wlen <= 0) throw std::runtime_error("utf8");
    std::wstring w((size_t)wlen, 0);
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), w.data(), wlen);
    int nlen = NormalizeString(NormalizationKD, w.c_str(), wlen, nullptr, 0);
    if (nlen <= 0) throw std::runtime_error("nfkd");
    std::wstring n((size_t)nlen, 0);
    nlen = NormalizeString(NormalizationKD, w.c_str(), wlen, n.data(), nlen);
    if (nlen <= 0) throw std::runtime_error("nfkd");
    n.resize((size_t)nlen);
    int ulen = WideCharToMultiByte(CP_UTF8, 0, n.c_str(), (int)n.size(), nullptr, 0, nullptr, nullptr);
    if (ulen <= 0) throw std::runtime_error("utf8");
    std::string o((size_t)ulen, 0);
    WideCharToMultiByte(CP_UTF8, 0, n.c_str(), (int)n.size(), o.data(), ulen, nullptr, nullptr);
    return o;
}
static std::vector<std::string> utf8_chars(const std::string & s) {
    std::vector<std::string> parts;
    for (size_t i = 0; i < s.size(); ) {
        size_t len = 1;
        unsigned char c = (unsigned char)s[i];
        if      ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        if (i + len > s.size()) throw std::runtime_error("utf8");
        parts.push_back(s.substr(i, len));
        i += len;
    }
    return parts;
}
static int find_rank(const std::unordered_map<std::string, int> & ranks,
                     const std::string & left, const std::string & right) {
    auto it = ranks.find(left + " " + right);
    return it != ranks.end() ? it->second : -1;
}
static std::vector<std::string> bpe_merge(const std::string & token,
                                          const std::unordered_map<std::string, int> & ranks) {
    auto parts = utf8_chars(token);
    while (parts.size() >= 2) {
        int best_rank = INT32_MAX;
        size_t best_i = 0;
        for (size_t i = 0; i + 1 < parts.size(); ++i) {
            int r = find_rank(ranks, parts[i], parts[i + 1]);
            if (r >= 0 && r < best_rank) { best_rank = r; best_i = i; }
        }
        if (best_rank == INT32_MAX) break;
        parts[best_i] = parts[best_i] + parts[best_i + 1];
        parts.erase(parts.begin() + (int)best_i + 1);
    }
    return parts;
}
static std::vector<std::string> whitespace_split(const std::string & text) {
    static const std::regex re(R"(\w+|[^\w\s]+)", std::regex::optimize);
    std::vector<std::string> words;
    auto begin = std::sregex_iterator(text.begin(), text.end(), re);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) words.push_back(it->str());
    return words;
}
bool mtl_bpe::load_from_arrays(const std::vector<std::string> & tokens,
                               const std::vector<int> & types,
                               const std::vector<std::string> & merges) {
    if (tokens.empty() || types.size() != tokens.size()) return false;
    token_to_id.clear();
    token_to_id.reserve(tokens.size());
    added.clear();
    for (size_t i = 0; i < tokens.size(); ++i) {
        token_to_id[tokens[i]] = (int32_t)i;
        if (types[i] == 4) added.push_back(tokens[i]);
    }
    std::sort(added.begin(), added.end(), [](const std::string & a, const std::string & b) {
        if (a.size() != b.size()) return a.size() > b.size();
        return a < b;
    });
    bpe_ranks.clear();
    bpe_ranks.reserve(merges.size());
    for (size_t i = 0; i < merges.size(); ++i) bpe_ranks[merges[i]] = (int)i;
    return true;
}
std::string mtl_bpe::punc_norm(const std::string & text) {
    if (text.empty()) throw std::runtime_error("empty text");
    std::string t = text;
    if (t[0] >= 'a' && t[0] <= 'z') t[0] = (char)(t[0] - 'a' + 'A');
    {
        std::string r;
        bool prev_space = false;
        for (char c : t) {
            if (c == ' ') { if (!prev_space) r += c; prev_space = true; }
            else { r += c; prev_space = false; }
        }
        t = r;
    }
    auto replace_all = [](std::string & s, const std::string & from, const std::string & to) {
        size_t pos = 0;
        while ((pos = s.find(from, pos)) != std::string::npos) {
            s.replace(pos, from.size(), to);
            pos += to.size();
        }
    };
    replace_all(t, "...", ", ");
    replace_all(t, "\xe2\x80\xa6", ", ");
    replace_all(t, ":", ",");
    replace_all(t, " - ", ", ");
    replace_all(t, ";", ",");
    replace_all(t, "\xe2\x80\x94", "-");
    replace_all(t, "\xe2\x80\x93", "-");
    replace_all(t, " ,", ",");
    replace_all(t, "\xe2\x80\x9c", "\"");
    replace_all(t, "\xe2\x80\x9d", "\"");
    replace_all(t, "\xe2\x80\x98", "'");
    replace_all(t, "\xe2\x80\x99", "'");
    while (!t.empty()) {
        char b = t.back();
        if (b == ' ' || b == '\t' || b == '\n' || b == '\r') t.pop_back();
        else break;
    }
    if (t.empty()) throw std::runtime_error("empty text");
    auto ends = [&](const std::string & s) {
        return t.size() >= s.size() && t.compare(t.size() - s.size(), s.size(), s) == 0;
    };
    const char last = t.back();
    if (last != '.' && last != '!' && last != '?' && last != '-' && last != ','
        && !ends("\xe3\x80\x81") && !ends("\xef\xbc\x8c") && !ends("\xe3\x80\x82")
        && !ends("\xef\xbc\x9f") && !ends("\xef\xbc\x81"))
        t += '.';
    return t;
}
std::vector<int32_t> mtl_bpe::encode(const std::string & text, const std::string & language_id) const {
    if (language_id.empty()) throw std::runtime_error("language");
    std::string t = punc_norm(text);
    t = to_lower_ascii(t);
    t = nfkd_utf8(t);
    if (language_id != "en" && language_id != "ar" && language_id != "da" && language_id != "de"
        && language_id != "el" && language_id != "es" && language_id != "fi" && language_id != "fr"
        && language_id != "hi" && language_id != "it" && language_id != "ms" && language_id != "nl"
        && language_id != "no" && language_id != "pl" && language_id != "pt" && language_id != "sv"
        && language_id != "sw" && language_id != "tr")
        throw std::runtime_error("language extras unread");
    t = "[" + language_id + "]" + t;
    {
        std::string r;
        for (char c : t) r += (c == ' ') ? std::string("[SPACE]") : std::string(1, c);
        t = r;
    }
    std::vector<int32_t> ids;
    auto emit_bpe = [&](const std::string & frag) {
        if (frag.empty()) return;
        for (const auto & word : whitespace_split(frag)) {
            for (const auto & part : bpe_merge(word, bpe_ranks)) {
                auto it = token_to_id.find(part);
                if (it == token_to_id.end()) {
                    auto u = token_to_id.find("[UNK]");
                    if (u == token_to_id.end()) throw std::runtime_error("unk");
                    ids.push_back(u->second);
                } else ids.push_back(it->second);
            }
        }
    };
    size_t i = 0;
    while (i < t.size()) {
        size_t best = 0;
        int32_t best_id = -1;
        for (const auto & a : added) {
            if (a.size() > best && i + a.size() <= t.size() && t.compare(i, a.size(), a) == 0) {
                auto it = token_to_id.find(a);
                if (it == token_to_id.end()) throw std::runtime_error("added");
                best = a.size();
                best_id = it->second;
            }
        }
        if (best) {
            ids.push_back(best_id);
            i += best;
        } else {
            size_t j = i + 1;
            while (j < t.size()) {
                bool hit = false;
                for (const auto & a : added) {
                    if (j + a.size() <= t.size() && t.compare(j, a.size(), a) == 0) { hit = true; break; }
                }
                if (hit) break;
                ++j;
            }
            emit_bpe(t.substr(i, j - i));
            i = j;
        }
    }
    return ids;
}
