#include "mtl_bpe.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <icu.h>
namespace {
using UChars = std::vector<UChar>;
UChars utf8_to_u16(const std::string & s) {
    UErrorCode status = U_ZERO_ERROR;
    int32_t n = 0;
    u_strFromUTF8(nullptr, 0, &n, s.data(), static_cast<int32_t>(s.size()), &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) throw std::runtime_error("ICU UTF-8 decode");
    status = U_ZERO_ERROR;
    UChars out(static_cast<size_t>(n));
    u_strFromUTF8(out.data(), n, nullptr, s.data(), static_cast<int32_t>(s.size()), &status);
    if (U_FAILURE(status)) throw std::runtime_error("ICU UTF-8 decode");
    return out;
}
std::string u16_to_utf8(const UChar * s, int32_t n) {
    UErrorCode status = U_ZERO_ERROR;
    int32_t bytes = 0;
    u_strToUTF8(nullptr, 0, &bytes, s, n, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) throw std::runtime_error("ICU UTF-8 encode");
    status = U_ZERO_ERROR;
    std::string out(static_cast<size_t>(bytes), '\0');
    u_strToUTF8(out.data(), bytes, nullptr, s, n, &status);
    if (U_FAILURE(status)) throw std::runtime_error("ICU UTF-8 encode");
    return out;
}
std::string u16_to_utf8(const UChars & s) { return u16_to_utf8(s.data(), static_cast<int32_t>(s.size())); }
UChars unicode_lower(const UChars & input) {
    UErrorCode status = U_ZERO_ERROR;
    int32_t n = u_strToLower(nullptr, 0, input.data(), static_cast<int32_t>(input.size()), "", &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) throw std::runtime_error("ICU lowercase");
    status = U_ZERO_ERROR;
    UChars out(static_cast<size_t>(n));
    u_strToLower(out.data(), n, input.data(), static_cast<int32_t>(input.size()), "", &status);
    if (U_FAILURE(status)) throw std::runtime_error("ICU lowercase");
    return out;
}
UChars nfkd(const UChars & input) {
    UErrorCode status = U_ZERO_ERROR;
    const UNormalizer2 * normalizer = unorm2_getNFKDInstance(&status);
    if (U_FAILURE(status)) throw std::runtime_error("ICU NFKD instance");
    int32_t n = unorm2_normalize(normalizer, input.data(), static_cast<int32_t>(input.size()), nullptr, 0, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) throw std::runtime_error("ICU NFKD");
    status = U_ZERO_ERROR;
    UChars out(static_cast<size_t>(n));
    unorm2_normalize(normalizer, input.data(), static_cast<int32_t>(input.size()), out.data(), n, &status);
    if (U_FAILURE(status)) throw std::runtime_error("ICU NFKD");
    return out;
}
bool word_codepoint(UChar32 c) {
    const int8_t type = u_charType(c);
    return u_hasBinaryProperty(c, UCHAR_ALPHABETIC)
        || type == U_NON_SPACING_MARK || type == U_COMBINING_SPACING_MARK || type == U_ENCLOSING_MARK
        || type == U_DECIMAL_DIGIT_NUMBER || type == U_CONNECTOR_PUNCTUATION || c == 0x200c || c == 0x200d;
}
std::vector<std::string> whitespace_split(const std::string & text) {
    const auto s = utf8_to_u16(text);
    std::vector<std::string> out;
    int32_t i = 0;
    while (i < static_cast<int32_t>(s.size())) {
        int32_t p = i;
        UChar32 c;
        U16_NEXT(s.data(), p, static_cast<int32_t>(s.size()), c);
        if (u_isUWhiteSpace(c)) { i = p; continue; }
        const bool word = word_codepoint(c);
        const int32_t begin = i;
        i = p;
        while (i < static_cast<int32_t>(s.size())) {
            p = i;
            U16_NEXT(s.data(), p, static_cast<int32_t>(s.size()), c);
            if (u_isUWhiteSpace(c) || word_codepoint(c) != word) break;
            i = p;
        }
        out.push_back(u16_to_utf8(s.data() + begin, i - begin));
    }
    return out;
}
int find_rank(const std::unordered_map<std::string, int> & ranks, const std::string & left, const std::string & right) {
    auto it = ranks.find(left + " " + right);
    return it != ranks.end() ? it->second : -1;
}
std::vector<std::string> utf8_chars(const std::string & s) {
    const auto u = utf8_to_u16(s);
    std::vector<std::string> out;
    int32_t i = 0;
    while (i < static_cast<int32_t>(u.size())) {
        const int32_t begin = i;
        UChar32 c;
        U16_NEXT(u.data(), i, static_cast<int32_t>(u.size()), c);
        out.push_back(u16_to_utf8(u.data() + begin, i - begin));
    }
    return out;
}
std::vector<std::string> bpe_merge(const std::string & token, const std::unordered_map<std::string, int> & ranks) {
    auto parts = utf8_chars(token);
    while (parts.size() >= 2) {
        int best_rank = INT32_MAX;
        size_t best_i = 0;
        for (size_t i = 0; i + 1 < parts.size(); ++i) {
            int r = find_rank(ranks, parts[i], parts[i + 1]);
            if (r >= 0 && r < best_rank) { best_rank = r; best_i = i; }
        }
        if (best_rank == INT32_MAX) break;
        parts[best_i] += parts[best_i + 1];
        parts.erase(parts.begin() + static_cast<std::ptrdiff_t>(best_i + 1));
    }
    return parts;
}
struct NumberFormatCloser { void operator()(UNumberFormat * p) const { if (p) unum_close(p); } };
using NumberFormatPtr = std::unique_ptr<UNumberFormat, NumberFormatCloser>;
std::string icu_version() {
    UVersionInfo v{};
    char text[U_MAX_VERSION_STRING_LENGTH]{};
    u_getVersion(v);
    u_versionToString(v, text);
    return text;
}
UChars remove_format_chars(const UChars & input) {
    UChars out;
    out.reserve(input.size());
    int32_t i = 0;
    while (i < static_cast<int32_t>(input.size())) {
        const int32_t begin = i;
        UChar32 c;
        U16_NEXT(input.data(), i, static_cast<int32_t>(input.size()), c);
        if (u_charType(c) != U_FORMAT_CHAR) out.insert(out.end(), input.begin() + begin, input.begin() + i);
    }
    return out;
}
bool contains_decimal_digit(const UChars & s) {
    int32_t i = 0;
    while (i < static_cast<int32_t>(s.size())) {
        UChar32 c;
        U16_NEXT(s.data(), i, static_cast<int32_t>(s.size()), c);
        if (u_charDigitValue(c) >= 0) return true;
    }
    return false;
}
UChars format_decimal(const UNumberFormat * format, const std::string & value) {
    UErrorCode status = U_ZERO_ERROR;
    int32_t n = unum_formatDecimal(format, value.data(), static_cast<int32_t>(value.size()), nullptr, 0, nullptr, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) throw std::runtime_error("ICU spellout");
    status = U_ZERO_ERROR;
    UChars out(static_cast<size_t>(n));
    unum_formatDecimal(format, value.data(), static_cast<int32_t>(value.size()), out.data(), n, nullptr, &status);
    if (U_FAILURE(status)) throw std::runtime_error("ICU spellout");
    return remove_format_chars(out);
}
UChars format_digits(const UNumberFormat * format, const std::string & digits) {
    UChars out;
    for (char digit : digits) {
        auto word = format_decimal(format, std::string(1, digit));
        if (contains_decimal_digit(word)) throw std::runtime_error("ICU locale has no number spellout data");
        if (!out.empty()) out.push_back(static_cast<UChar>(' '));
        out.insert(out.end(), word.begin(), word.end());
    }
    return out;
}
bool edge_word(const UChars & s, bool first) {
    if (s.empty()) return false;
    int32_t i = first ? 0 : static_cast<int32_t>(s.size());
    UChar32 c;
    if (first) U16_NEXT(s.data(), i, static_cast<int32_t>(s.size()), c);
    else U16_PREV(s.data(), 0, i, c);
    return word_codepoint(c);
}
}
bool mtl_bpe::load_from_arrays(const std::vector<std::string> & tokens,
                               const std::vector<int> & types,
                               const std::vector<std::string> & merges) {
    if (tokens.empty() || types.size() != tokens.size()) return false;
    token_to_id.clear();
    token_to_id.reserve(tokens.size());
    added.clear();
    for (size_t i = 0; i < tokens.size(); ++i) {
        token_to_id[tokens[i]] = static_cast<int32_t>(i);
        if (types[i] == 4) added.push_back(tokens[i]);
    }
    std::sort(added.begin(), added.end(), [](const std::string & a, const std::string & b) {
        if (a.size() != b.size()) return a.size() > b.size();
        return a < b;
    });
    bpe_ranks.clear();
    bpe_ranks.reserve(merges.size());
    for (size_t i = 0; i < merges.size(); ++i) bpe_ranks[merges[i]] = static_cast<int>(i);
    return true;
}
std::string mtl_bpe::punc_norm(const std::string & text) {
    if (text.empty()) throw std::runtime_error("empty text");
    std::string t = text;
    if (t[0] >= 'a' && t[0] <= 'z') t[0] = static_cast<char>(t[0] - 'a' + 'A');
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
    auto ends = [&](const std::string & s) { return t.size() >= s.size() && t.compare(t.size() - s.size(), s.size(), s) == 0; };
    const char last = t.back();
    if (last != '.' && last != '!' && last != '?' && last != '-' && last != ','
        && !ends("\xe3\x80\x81") && !ends("\xef\xbc\x8c") && !ends("\xe3\x80\x82")
        && !ends("\xef\xbc\x9f") && !ends("\xef\xbc\x81")) t += '.';
    return t;
}
std::string mtl_bpe::prepare_input(const std::string & text, const std::string & language_id) {
    if (language_id.empty()) throw std::runtime_error("language");
    auto unicode = nfkd(unicode_lower(utf8_to_u16(punc_norm(text))));
    std::string t = u16_to_utf8(unicode);
    t = "[" + language_id + "]" + t;
    std::string r;
    r.reserve(t.size());
    for (char c : t) r += c == ' ' ? "[SPACE]" : std::string(1, c);
    return r;
}
mtl_number_result mtl_bpe::verbalize_numbers(const std::string & text, const std::string & language_id) const {
    if (token_to_id.find("[" + language_id + "]") == token_to_id.end()) throw std::runtime_error("language token not present in tokenizer");
    const auto input = utf8_to_u16(text);
    bool has_digit = false;
    for (int32_t i = 0; i < static_cast<int32_t>(input.size());) {
        UChar32 c;
        U16_NEXT(input.data(), i, static_cast<int32_t>(input.size()), c);
        if (u_charDigitValue(c) >= 0) { has_digit = true; break; }
    }
    mtl_number_result result{text, {}, "Windows ICU " + icu_version() + " / CLDR RBNF"};
    if (!has_digit) return result;
    UErrorCode status = U_ZERO_ERROR;
    NumberFormatPtr spell(unum_open(UNUM_SPELLOUT, nullptr, 0, language_id.c_str(), nullptr, &status));
    if (U_FAILURE(status) || !spell) throw std::runtime_error("ICU spellout locale");
    UChars output;
    int32_t cursor = 0;
    int32_t i = 0;
    while (i < static_cast<int32_t>(input.size())) {
        int32_t p = i;
        UChar32 c;
        U16_NEXT(input.data(), p, static_cast<int32_t>(input.size()), c);
        if (u_charDigitValue(c) < 0) { i = p; continue; }
        const int32_t begin = i;
        std::string digits;
        while (i < static_cast<int32_t>(input.size())) {
            p = i;
            U16_NEXT(input.data(), p, static_cast<int32_t>(input.size()), c);
            const int digit = u_charDigitValue(c);
            if (digit < 0) break;
            digits.push_back(static_cast<char>('0' + digit));
            i = p;
        }
        UChars spoken = digits.size() > 1 && digits.front() == '0' ? format_digits(spell.get(), digits) : format_decimal(spell.get(), digits);
        if (contains_decimal_digit(spoken)) spoken = format_digits(spell.get(), digits);
        output.insert(output.end(), input.begin() + cursor, input.begin() + begin);
        if (!output.empty() && edge_word(output, false) && edge_word(spoken, true)) output.push_back(static_cast<UChar>(' '));
        output.insert(output.end(), spoken.begin(), spoken.end());
        if (i < static_cast<int32_t>(input.size())) {
            int32_t q = i;
            UChar32 next;
            U16_NEXT(input.data(), q, static_cast<int32_t>(input.size()), next);
            if (word_codepoint(next) && edge_word(spoken, false)) output.push_back(static_cast<UChar>(' '));
        }
        result.rewrites.push_back({u16_to_utf8(input.data() + begin, i - begin), u16_to_utf8(spoken)});
        cursor = i;
    }
    output.insert(output.end(), input.begin() + cursor, input.end());
    result.text = u16_to_utf8(output);
    return result;
}
std::vector<int32_t> mtl_bpe::encode(const std::string & text, const std::string & language_id) const {
    if (token_to_id.find("[" + language_id + "]") == token_to_id.end()) throw std::runtime_error("language token not present in tokenizer");
    const std::string t = prepare_input(text, language_id);
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
