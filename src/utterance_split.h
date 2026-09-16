#pragma once
#include <algorithm>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
#include "tts-cpp/chatterbox/engine.h"
#include "text_prepare.h"
#include "execution_trace.h"
namespace tts_cpp::chatterbox::detail {
struct UtteranceUnit { size_t begin,end; std::string text; std::vector<int32_t> ids; };
using EncodeText = std::function<std::vector<int32_t>(const std::string&)>;
inline std::vector<UtteranceUnit> split_utterances(const PreparedText& p,int budget,const EncodeText& encode,ExecutionTrace* trace) {
    if(budget<0)throw std::runtime_error("negative split budget");
    std::vector<UtteranceUnit> out;
    auto protected_boundary=[&](size_t pos){for(const auto&a:p.atoms)if(pos>a.begin&&pos<a.end)return true;return false;};
    auto make=[&](size_t b,size_t e){auto t=p.text.substr(b,e-b);return UtteranceUnit{b,e,t,encode(t)};};
    std::vector<size_t> stops=p.boundaries;stops.push_back(p.text.size());size_t segment=0;
    for(size_t end:stops){
        size_t b=segment;segment=end;
        if(p.text.substr(b,end-b).find_first_not_of(' ')==std::string::npos)continue;
        if(budget==0){out.push_back(make(b,end));continue;}
        while(b<end){
            if(p.text.substr(b,end-b).find_first_not_of(' ')==std::string::npos)break;
            auto whole=make(b,end);if(whole.ids.size()<=size_t(budget)){out.push_back(std::move(whole));break;}
            size_t chosen=b;
            // Prefer the furthest fitting sentence, then clause, then word.
            for(int level=0;level<3&&chosen==b;++level){
                for(size_t j=b+1;j<end;++j){
                    if(p.text[j]!=' '||protected_boundary(j+1))continue;
                    char c=p.text[j-1];
                    if(level==0&&c!='.'&&c!='!'&&c!='?')continue;
                    if(level==1&&c!=','&&c!=';'&&c!=':')continue;
                    auto u=make(b,j+1);if(u.ids.size()<=size_t(budget))chosen=j+1;
                }
            }
            if(chosen==b)throw std::runtime_error("indivisible text atom exceeds split budget at prepared byte "+std::to_string(b));
            out.push_back(make(b,chosen));b=chosen;
        }
    }
    if(out.empty())throw std::runtime_error("empty text");
    size_t cursor=0;std::string units="[";
    for(size_t i=0;i<out.size();++i){const auto&u=out[i];
        if(u.begin<cursor||p.text.substr(cursor,u.begin-cursor).find_first_not_of(' ')!=std::string::npos)throw std::runtime_error("split coverage gap");
        if(u.ids.empty()||(budget>0&&u.ids.size()>size_t(budget)))throw std::runtime_error("invalid unit token budget");
        cursor=u.end;if(i)units+=',';
        units+="{\"index\":"+std::to_string(i)+",\"begin\":"+std::to_string(u.begin)+",\"end\":"+std::to_string(u.end)+",\"tokenizer_input\":"+json_string(u.text)+",\"text_tokens\":"+std::to_string(u.ids.size())+"}";
    }
    if(p.text.substr(cursor).find_first_not_of(' ')!=std::string::npos)throw std::runtime_error("split trailing gap");
    trace_event(trace,"split_complete","split",{{"units",units+"]"},{"budget",std::to_string(budget)},{"coverage","true"},{"offset_units",json_string("prepared_utf8_bytes")}});
    return out;
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
