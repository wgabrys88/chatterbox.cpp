#include "sha256.h"
#include <windows.h>
#include <bcrypt.h>
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
namespace tts_cpp::chatterbox {
namespace {
struct Hash {
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    Hash() {
        if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0)
            throw std::runtime_error("SHA256 provider");
        if (BCryptCreateHash(algorithm, &hash, nullptr, 0, nullptr, 0, 0) < 0) {
            BCryptCloseAlgorithmProvider(algorithm, 0); algorithm=nullptr;
            throw std::runtime_error("SHA256 create");
        }
    }
    ~Hash() { if(hash) BCryptDestroyHash(hash); if(algorithm) BCryptCloseAlgorithmProvider(algorithm,0); }
    void add(const char* data, size_t n) {
        while(n) {
            const ULONG k=static_cast<ULONG>((std::min)(n,size_t(1048576)));
            if (BCryptHashData(hash, reinterpret_cast<PUCHAR>(const_cast<char*>(data)),k,0)<0)
                throw std::runtime_error("SHA256 update");
            data+=k; n-=k;
        }
    }
    std::string finish() {
        std::array<unsigned char,32> digest{};
        if (BCryptFinishHash(hash,digest.data(),ULONG(digest.size()),0)<0)
            throw std::runtime_error("SHA256 finish");
        std::ostringstream s; s<<std::hex<<std::setfill('0');
        for(auto x:digest) s<<std::setw(2)<<unsigned(x);
        return s.str();
    }
};
}
std::string sha256_text(const std::string& value) { Hash h; h.add(value.data(),value.size()); return h.finish(); }
std::string sha256_file(const std::string& path) {
    std::ifstream f(std::filesystem::u8path(path),std::ios::binary);
    if(!f) throw std::runtime_error("hash open: "+path);
    Hash h; std::array<char,65536> b;
    while(f.read(b.data(),b.size()) || f.gcount()) h.add(b.data(),size_t(f.gcount()));
    if(!f.eof()) throw std::runtime_error("hash read: "+path);
    return h.finish();
}
}
