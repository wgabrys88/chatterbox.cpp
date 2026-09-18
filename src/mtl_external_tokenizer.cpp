#include "mtl_external_tokenizer.h"
#include <windows.h>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
std::wstring wide(const std::string & s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), static_cast<int>(s.size()), nullptr, 0);
    if (!n) throw std::runtime_error("UTF-8 path conversion");
    std::wstring out(static_cast<size_t>(n), L'\0');
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), static_cast<int>(s.size()), out.data(), n) != n)
        throw std::runtime_error("UTF-8 path conversion");
    return out;
}
std::wstring quote(const std::wstring & arg) {
    std::wstring out = L"\"";
    size_t slashes = 0;
    for (wchar_t c : arg) {
        if (c == L'\\') { ++slashes; continue; }
        if (c == L'\"') {
            out.append(slashes * 2 + 1, L'\\');
            out.push_back(L'\"');
            slashes = 0;
            continue;
        }
        out.append(slashes, L'\\');
        slashes = 0;
        out.push_back(c);
    }
    out.append(slashes * 2, L'\\');
    out.push_back(L'\"');
    return out;
}
std::filesystem::path temp_file(const wchar_t * prefix) {
    wchar_t dir[MAX_PATH + 1]{};
    const DWORD n = GetTempPathW(MAX_PATH, dir);
    if (!n || n > MAX_PATH) throw std::runtime_error("GetTempPathW");
    wchar_t file[MAX_PATH + 1]{};
    if (!GetTempFileNameW(dir, prefix, 0, file)) throw std::runtime_error("GetTempFileNameW");
    return std::filesystem::path(file);
}
struct TempPair {
    std::filesystem::path input = temp_file(L"mtl"), output = temp_file(L"mto");
    ~TempPair() { std::error_code ec; std::filesystem::remove(input, ec); std::filesystem::remove(output, ec); }
};
void run(const mtl_external_tokenizer_options & o, const std::string & mode, const std::string & language, const std::filesystem::path & input, const std::filesystem::path & output) {
    std::vector<std::wstring> args = {
        wide(o.python), wide(o.script), L"--mode", wide(mode), L"--source", wide(o.source), L"--tts-source", wide(o.tts_source),
        L"--tokenizer", wide(o.tokenizer_json), L"--cangjie", wide(o.cangjie_json), L"--dicta-model", wide(o.dicta_model),
        L"--language", wide(language), L"--input", input.wstring(), L"--output", output.wstring()
    };
    std::wstring cmd;
    for (const auto & arg : args) { if (!cmd.empty()) cmd.push_back(L' '); cmd += quote(arg); }
    STARTUPINFOW si{}; si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    std::vector<wchar_t> mutable_cmd(cmd.begin(), cmd.end()); mutable_cmd.push_back(L'\0');
    if (!CreateProcessW(nullptr, mutable_cmd.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi))
        throw std::runtime_error("official tokenizer process start");
    CloseHandle(pi.hThread);
    const DWORD wait = WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    if (wait != WAIT_OBJECT_0 || !GetExitCodeProcess(pi.hProcess, &code)) { CloseHandle(pi.hProcess); throw std::runtime_error("official tokenizer process wait"); }
    CloseHandle(pi.hProcess);
    if (code != 0) throw std::runtime_error("official tokenizer process failed");
}
template <typename T> T read_scalar(std::ifstream & f) {
    T value{};
    f.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!f) throw std::runtime_error("official tokenizer output truncated");
    return value;
}
}

std::string mtl_external_punc_norm(const mtl_external_tokenizer_options & o, const std::string & text) {
    TempPair files;
    { std::ofstream f(files.input, std::ios::binary | std::ios::trunc); f.write(text.data(), static_cast<std::streamsize>(text.size())); if (!f) throw std::runtime_error("official punctuation input write"); }
    run(o, "punc", "", files.input, files.output);
    std::ifstream f(files.output, std::ios::binary | std::ios::ate);
    const auto size = f.tellg();
    if (size < 0) throw std::runtime_error("official punctuation output");
    std::string result(static_cast<size_t>(size), '\0');
    f.seekg(0);
    if (size && !f.read(result.data(), static_cast<std::streamsize>(size))) throw std::runtime_error("official punctuation output");
    return result;
}
mtl_external_tokenizer_result mtl_external_tokenize(const mtl_external_tokenizer_options & o, const std::string & text, const std::string & language) {
    TempPair files;
    { std::ofstream f(files.input, std::ios::binary | std::ios::trunc); f.write(text.data(), static_cast<std::streamsize>(text.size())); if (!f) throw std::runtime_error("official tokenizer input write"); }
    run(o, "tokenize", language, files.input, files.output);
    std::ifstream f(files.output, std::ios::binary);
    const uint32_t magic = read_scalar<uint32_t>(f);
    if (magic != 0x344c544dU) throw std::runtime_error("official tokenizer output magic");
    const uint32_t text_size = read_scalar<uint32_t>(f);
    const uint32_t count = read_scalar<uint32_t>(f);
    mtl_external_tokenizer_result result;
    result.tokenizer_input.resize(text_size);
    f.read(result.tokenizer_input.data(), static_cast<std::streamsize>(text_size));
    if (!f) throw std::runtime_error("official tokenizer output text");
    result.ids.resize(count);
    if (count) f.read(reinterpret_cast<char *>(result.ids.data()), static_cast<std::streamsize>(count * sizeof(int32_t)));
    if (!f) throw std::runtime_error("official tokenizer output ids");
    char extra;
    if (f.read(&extra, 1)) throw std::runtime_error("official tokenizer output trailing data");
    return result;
}
