#include "mtl_numbers.h"
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
bool word_codepoint(UChar32 c) {
    const int8_t type = u_charType(c);
    return u_hasBinaryProperty(c, UCHAR_ALPHABETIC)
        || type == U_NON_SPACING_MARK || type == U_COMBINING_SPACING_MARK || type == U_ENCLOSING_MARK
        || type == U_DECIMAL_DIGIT_NUMBER || type == U_CONNECTOR_PUNCTUATION || c == 0x200c || c == 0x200d;
}
bool edge_word(const UChars & s, bool first) {
    if (s.empty()) return false;
    int32_t i = first ? 0 : static_cast<int32_t>(s.size());
    UChar32 c;
    if (first) U16_NEXT(s.data(), i, static_cast<int32_t>(s.size()), c);
    else U16_PREV(s.data(), 0, i, c);
    return word_codepoint(c);
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
}
mtl_number_result mtl_numbers::verbalize_numbers(const std::string & text, const std::string & language_id) const {
    if (language_id.empty()) throw std::runtime_error("language");
    const auto input = utf8_to_u16(text);
    bool has_digit = false;
    for (int32_t i = 0; i < static_cast<int32_t>(input.size());) {
        UChar32 c;
        U16_NEXT(input.data(), i, static_cast<int32_t>(input.size()), c);
        if (u_charDigitValue(c) >= 0) { has_digit = true; break; }
    }
    mtl_number_result result{text, {}, "ICU " + icu_version() + " / CLDR RBNF"};
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
