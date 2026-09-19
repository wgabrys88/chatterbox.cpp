#pragma once
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
#include "tts-cpp/chatterbox/engine.h"
#include "text_prepare.h"
namespace tts_cpp::chatterbox::detail {
struct UtteranceUnit { size_t begin,end; std::string text; std::vector<int32_t> ids; };
using EncodeText = std::function<std::vector<int32_t>(const std::string&)>;
inline UtteranceUnit encode_utterance(const PreparedText& p,const EncodeText& encode) {
    if(p.text.find_first_not_of(' ')==std::string::npos)throw std::runtime_error("empty text");
    UtteranceUnit u{0,p.text.size(),p.text,encode(p.text)};
    if(u.ids.empty())throw std::runtime_error("empty tokenizer output");
    return u;
}
inline void accumulate_unit(SynthesizeStats* total, const SynthesizeStats& unit) {
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
