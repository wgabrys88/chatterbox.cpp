#include "text_prepare.h"
#include "execution_trace.h"
#include <algorithm>
#include <cctype>
#include <regex>
#include <stdexcept>
namespace tts_cpp::chatterbox {
void validate_utf8(const std::string& s) {
    for(size_t i=0;i<s.size();) {
        const unsigned char c=s[i++];
        if(c==0 || (c<32 && c!='\t' && c!='\r' && c!='\n')) throw std::runtime_error("invalid text control");
        if(c<128) continue;
        int n; uint32_t cp, minimum;
        if(c>=0xc2 && c<=0xdf) {n=1;cp=c&31;minimum=0x80;}
        else if(c>=0xe0 && c<=0xef) {n=2;cp=c&15;minimum=0x800;}
        else if(c>=0xf0 && c<=0xf4) {n=3;cp=c&7;minimum=0x10000;}
        else throw std::runtime_error("invalid UTF-8 lead");
        while(n--) { if(i==s.size() || (static_cast<unsigned char>(s[i])&0xc0)!=0x80) throw std::runtime_error("invalid UTF-8 continuation"); cp=(cp<<6)|(s[i++]&63); }
        if(cp<minimum || cp>0x10ffff || (cp>=0xd800 && cp<=0xdfff)) throw std::runtime_error("invalid UTF-8 scalar");
    }
}
namespace {
const char* small[]={"zero","one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen"};
std::string cardinal(unsigned n) {
    const char* tens[]={"","","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"};
    if(n<20) return small[n];
    if(n<100) return std::string(tens[n/10])+(n%10?"-"+cardinal(n%10):"");
    if(n<1000) return cardinal(n/100)+" hundred"+(n%100?" "+cardinal(n%100):"");
    return cardinal(n/1000)+" thousand"+(n%1000?" "+cardinal(n%1000):"");
}
std::string ordinal(unsigned n) {
    const char* o[]={"","first","second","third","fourth","fifth","sixth","seventh","eighth","ninth","tenth","eleventh","twelfth","thirteenth","fourteenth","fifteenth","sixteenth","seventeenth","eighteenth","nineteenth","twentieth"};
    if(n<=20) return o[n];
    if(n==30) return "thirtieth";
    return std::string(n<30?"twenty-":"thirty-")+o[n%10];
}
std::string digits(const std::string& s) {
    std::string r; for(char c:s) {if(!r.empty())r+=' ';r+=small[c-'0'];} return r;
}
bool space(char c) {return c==' '||c=='\t'||c=='\n'||c=='\r';}
bool word(unsigned char c) {return c>=128||std::isalnum(c)||c=='_';}
std::string lower(std::string s) {for(char&c:s)if(c>='A'&&c<='Z')c+=32;return s;}
bool match(const std::string& s,size_t i,const std::regex& re,std::smatch& m) {
    return std::regex_search(s.begin()+std::ptrdiff_t(i),s.end(),m,re,std::regex_constants::match_continuous);
}
}
PreparedText prepare_text(const std::string& s,bool english,ExecutionTrace* trace) {
    validate_utf8(s); PreparedText p;
    static const std::regex date(R"((January|February|March|April|May|June|July|August|September|October|November|December) ([0-9]{1,2}), ([0-9]{4}))",std::regex::icase);
    static const std::regex currency(R"(\$([0-9]{1,6})(\.([0-9]{2}))?)");
    static const std::regex clock(R"(([0-9]{1,2}):([0-9]{2}) ([ap])\.m\.)",std::regex::icase);
    static const std::regex phone(R"(([0-9]{3})-([0-9]{4}))");
    static const std::regex ord(R"(([0-9]{1,2})(st|nd|rd|th))",std::regex::icase);
    static const std::regex decimal(R"(([0-9]{0,6})\.([0-9]+))");
    static const std::regex integer(R"([0-9]{1,6})");
    static const std::regex seat(R"(([0-9]{1,6})([A-Za-z]))");
    static const std::regex phone_context(R"((call|phone|telephone)\b[^.!?\n]*$)",std::regex::icase);
    for(size_t i=0;i<s.size();) {
        if(s.compare(i,3,"|||")==0) {
            size_t b=p.text.size(); p.text+=' ';
            p.edits.push_back({i,i+3,b,p.text.size()," ","explicit_boundary"});
            p.boundaries.push_back(p.text.size()); i+=3;continue;
        }
        if(space(s[i])) {size_t j=i+1;while(j<s.size()&&space(s[j]))++j;
            size_t b=p.text.size();p.text+=' '; if(s.substr(i,j-i)!=" ")p.edits.push_back({i,j,b,b+1," ","whitespace"});i=j;continue;}
        size_t used=0;std::string replacement,rule;std::smatch m;
        const bool start=i==0 || !word(static_cast<unsigned char>(s[i-1]));
        auto end_ok=[&](size_t n){return i+n==s.size() || (!word(static_cast<unsigned char>(s[i+n])) && s[i+n]!='.' && s[i+n]!='-' && s[i+n]!=':') || (s[i+n]=='.' && (i+n+1==s.size()||space(s[i+n+1])));};
        // Protect the whole whitespace-delimited URL/email/mixed identifier.
        if(start) {
            size_t j=i;while(j<s.size()&&!space(s[j])&&s.compare(j,3,"|||")!=0)++j;
            std::string atom=s.substr(i,j-i); bool alpha=false,num=false;
            for(unsigned char c:atom){alpha|=std::isalpha(c)!=0;num|=std::isdigit(c)!=0;}
            const bool seat_context=i>=5 && lower(s.substr(i-5,5))=="seat ";
            const bool suffix_ordinal=match(s,i,ord,m) && end_ok(size_t(m.length()));
            bool protected_atom=atom.find("://")!=std::string::npos || atom.find('@')!=std::string::npos || atom.find('_')!=std::string::npos || (alpha&&num&&!seat_context&&!suffix_ordinal&&atom.find(':')==std::string::npos);
            // Month names are not mixed identifiers; clock tokens are parsed below.
            if(protected_atom) {size_t b=p.text.size();p.text+=atom;p.atoms.push_back({b,p.text.size()});if(num)p.unhandled.push_back({b,p.text.size()});i=j;continue;}
        }
        if(english&&start) {
            if(match(s,i,date,m)&&end_ok(m.length())) {
                unsigned day=std::stoul(m[2]),year=std::stoul(m[3]);
                std::string month=lower(m[1]);
                const std::string months[]={"january","february","march","april","may","june","july","august","september","october","november","december"};
                size_t mi=std::find(std::begin(months),std::end(months),month)-std::begin(months);
                const int lengths[]={31,28,31,30,31,30,31,31,30,31,30,31};
                int maxday=lengths[mi]+(mi==1&&(year%400==0||(year%4==0&&year%100!=0)));
                if(day>=1&&day<=unsigned(maxday)) {used=m.length();replacement=m[1].str()+" "+ordinal(day)+", "+cardinal(year);rule="en_date";}
            }
            if(!used&&match(s,i,currency,m)&&end_ok(m.length())) {
                unsigned dollars=std::stoul(m[1]),cents=m[3].matched?std::stoul(m[3]):0;
                used=m.length();replacement=cardinal(dollars)+(dollars==1?" dollar":" dollars");
                if(cents)replacement+=" and "+cardinal(cents)+(cents==1?" cent":" cents");rule="en_usd";
            }
            if(!used&&match(s,i,clock,m)&&end_ok(m.length())) {
                unsigned h=std::stoul(m[1]),minute=std::stoul(m[2]);
                if(h>=1&&h<=12&&minute<60){used=m.length();replacement=cardinal(h)+(minute==0?" o clock":minute<10?" oh "+cardinal(minute):" "+cardinal(minute))+" "+lower(m[3])+" m";rule="en_clock";}
            }
            if(!used&&match(s,i,phone,m)&&end_ok(m.length())&&std::regex_search(s.substr(0,i),phone_context)) {
                used=m.length();replacement=digits(m[1])+", "+digits(m[2]);rule="en_phone";
            }
            if(!used&&match(s,i,ord,m)&&end_ok(m.length())) {
                unsigned n=std::stoul(m[1]);std::string suffix=(n%100>=11&&n%100<=13)?"th":n%10==1?"st":n%10==2?"nd":n%10==3?"rd":"th";
                if(n>=1&&n<=31&&lower(m[2])==suffix){used=m.length();replacement=ordinal(n);rule="en_ordinal";}
            }
            if(!used&&match(s,i,seat,m)&&end_ok(m.length())&&i>=5&&lower(s.substr(i-5,5))=="seat ") {
                used=m.length();replacement=cardinal(std::stoul(m[1]))+" "+m[2].str();rule="en_seat";
            }
            if(!used&&match(s,i,decimal,m)&&end_ok(m.length())) {
                used=m.length();replacement=cardinal(m[1].length()?std::stoul(m[1]):0)+" point "+digits(m[2]);rule="en_decimal";
            }
            if(!used&&match(s,i,integer,m)&&end_ok(m.length())&&(i==0||(s[i-1]!='-'&&s[i-1]!='.'&&s[i-1]!='$'&&s[i-1]!=':'))) {
                used=m.length();replacement=cardinal(std::stoul(m[0]));rule="en_integer";
            }
        }
        if(used) {size_t b=p.text.size();p.text+=replacement;p.edits.push_back({i,i+used,b,p.text.size(),replacement,rule});p.atoms.push_back({b,p.text.size()});i+=used;}
        else {size_t b=p.text.size();p.text+=s[i];if(s[i]>='0'&&s[i]<='9')p.unhandled.push_back({b,b+1});++i;}
    }
    if(p.text.find_first_not_of(' ')==std::string::npos)throw std::runtime_error("empty text");
    std::string edits="[",unhandled="[",boundaries="[";
    for(size_t i=0;i<p.edits.size();++i){const auto&e=p.edits[i];if(i)edits+=',';edits+="{\"original_begin\":"+std::to_string(e.original_begin)+",\"original_end\":"+std::to_string(e.original_end)+",\"prepared_begin\":"+std::to_string(e.prepared_begin)+",\"prepared_end\":"+std::to_string(e.prepared_end)+",\"replacement\":"+json_string(e.replacement)+",\"rule\":"+json_string(e.rule)+"}";}
    for(size_t i=0;i<p.unhandled.size();++i){if(i)unhandled+=',';unhandled+="["+std::to_string(p.unhandled[i].begin)+","+std::to_string(p.unhandled[i].end)+"]";}
    for(size_t i=0;i<p.boundaries.size();++i){if(i)boundaries+=',';boundaries+=std::to_string(p.boundaries[i]);}
    trace_event(trace,"text_prepared","prepare",{{"original_sha256",json_string(sha256_text(s))},{"prepared_sha256",json_string(sha256_text(p.text))},{"text",json_string(p.text)},{"edits",edits+"]"},{"unhandled_spans",unhandled+"]"},{"explicit_boundaries",boundaries+"]"},{"policy",json_string(english?"english_bounded_v1":"language_tokenizer_only")}});
    return p;
}
}
