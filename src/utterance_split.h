#pragma once
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
#include "tts-cpp/chatterbox/engine.h"
namespace tts_cpp::chatterbox::detail {
using TokenCount = std::function<int(const std::string&)>;

inline std::string collapse_ws(const std::string& s) {
    std::string o;
    bool sp = false;
    for (char c : s) {
        const bool w = c == ' ' || c == '\t' || c == '\n' || c == '\r';
        if (w) { if (!sp && !o.empty()) o += ' '; sp = true; }
        else { o += c; sp = false; }
    }
    while (!o.empty() && o.back() == ' ') o.pop_back();
    return o;
}

// Sentence boundary: one of .!? followed by optional closers, then a space or end.
inline std::vector<std::string> split_sentences(const std::string& t) {
    std::vector<std::string> out;
    std::string cur;
    for (size_t i = 0; i < t.size(); ++i) {
        cur += t[i];
        if (t[i] != '.' && t[i] != '!' && t[i] != '?') continue;
        size_t j = i + 1;
        while (j < t.size() && (t[j] == '.' || t[j] == '!' || t[j] == '?' || t[j] == '"' || t[j] == '\'' || t[j] == ')')) cur += t[j++];
        if (j < t.size() && t[j] != ' ') { i = j - 1; continue; }
        const std::string s = collapse_ws(cur);
        if (!s.empty()) out.push_back(s);
        cur.clear();
        i = j;  // skip the space
    }
    const std::string s = collapse_ws(cur);
    if (!s.empty()) out.push_back(s);
    return out;
}

// Split one over-budget sentence at clause marks, then at spaces.
inline std::vector<std::string> split_long(const std::string& s, int budget, const TokenCount& count) {
    std::vector<std::string> words, out;
    {
        std::string w;
        for (char c : s) {
            if (c == ' ') { if (!w.empty()) words.push_back(w); w.clear(); }
            else w += c;
        }
        if (!w.empty()) words.push_back(w);
    }
    std::string cur;
    for (const auto& w : words) {
        const std::string cand = cur.empty() ? w : cur + " " + w;
        if (!cur.empty() && count(cand) > budget) {
            out.push_back(cur);
            cur = w;
        } else {
            cur = cand;
        }
        const char b = cur.back();
        if ((b == ',' || b == ';' || b == ':') && count(cur) * 2 > budget) {
            out.push_back(cur);
            cur.clear();
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// budget <= 0: one utterance (unsplit), so the roof can be measured as-is.
inline std::vector<std::string> split_utterances(const std::string& text, int budget, const TokenCount& count) {
    const std::string t = collapse_ws(text);
    if (t.empty()) throw std::runtime_error("empty text");
    if (budget <= 0) return { t };
    std::vector<std::string> units;
    std::string cur;
    for (const auto& s : split_sentences(t)) {
        if (count(s) > budget) {
            if (!cur.empty()) { units.push_back(cur); cur.clear(); }
            for (auto& piece : split_long(s, budget, count)) units.push_back(piece);
            continue;
        }
        const std::string cand = cur.empty() ? s : cur + " " + s;
        if (!cur.empty() && count(cand) > budget) {
            units.push_back(cur);
            cur = s;
        } else {
            cur = cand;
        }
    }
    if (!cur.empty()) units.push_back(cur);
    if (units.empty()) throw std::runtime_error("empty text");
    return units;
}

inline void accumulate_unit(SynthesizeStats* total, const SynthesizeStats& unit, int index, int n, const std::string& text) {
    std::fprintf(stderr, "unit %d/%d text_tokens=%d predicted=%d dropped=%d eos=%d n_past=%d text=\"%s\"\n",
        index + 1, n, unit.text_tokens, unit.predicted_count, unit.dropped_count, unit.eos, unit.n_past, text.c_str());
    std::fflush(stderr);
    if (!total) return;
    total->predicted_count += unit.predicted_count;
    total->dropped_count += unit.dropped_count;
    total->text_tokens += unit.text_tokens;
    total->units += 1;
    if (unit.n_past > total->n_past) total->n_past = unit.n_past;
    if (unit.predicted_count > total->max_unit_predicted) total->max_unit_predicted = unit.predicted_count;
    total->eos = (total->units == 1) ? unit.eos : (total->eos && unit.eos ? 1 : 0);
}
}
