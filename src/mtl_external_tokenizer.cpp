#include "mtl_external_tokenizer.h"
#include <windows.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::wstring wide(const std::string & s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), static_cast<int>(s.size()), nullptr, 0);
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

template <typename T> T take_scalar(const std::vector<uint8_t> & data, size_t & offset) {
    T value{};
    std::memcpy(&value, data.data() + offset, sizeof(T));
    offset += sizeof(T);
    return value;
}

void write_all(HANDLE h, const void * data, size_t size) {
    const auto * p = static_cast<const uint8_t *>(data);
    while (size) {
        const DWORD chunk = static_cast<DWORD>(std::min<size_t>(size, std::numeric_limits<DWORD>::max()));
        DWORD written = 0;
        if (!WriteFile(h, p, chunk, &written, nullptr) || !written) throw std::runtime_error("official tokenizer pipe write");
        p += written;
        size -= written;
    }
}

void read_all(HANDLE h, void * data, size_t size) {
    auto * p = static_cast<uint8_t *>(data);
    while (size) {
        const DWORD chunk = static_cast<DWORD>(std::min<size_t>(size, std::numeric_limits<DWORD>::max()));
        DWORD got = 0;
        if (!ReadFile(h, p, chunk, &got, nullptr) || !got) throw std::runtime_error("official tokenizer pipe read");
        p += got;
        size -= got;
    }
}

struct PipePair {
    HANDLE parent_write = nullptr;
    HANDLE parent_read = nullptr;
    HANDLE child_read = nullptr;
    HANDLE child_write = nullptr;
    ~PipePair() {
        for (HANDLE h : {parent_write, parent_read, child_read, child_write}) if (h) CloseHandle(h);
    }
};
}

struct mtl_external_tokenizer::Impl {
    HANDLE input = nullptr;
    HANDLE output = nullptr;
    HANDLE process = nullptr;

    Impl(const mtl_external_tokenizer_options & o, const std::string & language) {
        PipePair pipes;
        SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
        if (!CreatePipe(&pipes.child_read, &pipes.parent_write, &sa, 0)) throw std::runtime_error("official tokenizer stdin pipe");
        if (!CreatePipe(&pipes.parent_read, &pipes.child_write, &sa, 0)) throw std::runtime_error("official tokenizer stdout pipe");
        if (!SetHandleInformation(pipes.parent_write, HANDLE_FLAG_INHERIT, 0) || !SetHandleInformation(pipes.parent_read, HANDLE_FLAG_INHERIT, 0))
            throw std::runtime_error("official tokenizer pipe inheritance");

        std::vector<std::wstring> args = {
            wide(o.python), wide(o.script), L"--source", wide(o.source), L"--tts-source", wide(o.tts_source),
            L"--tokenizer", wide(o.tokenizer_json), L"--cangjie", wide(o.cangjie_json), L"--dicta-model", wide(o.dicta_model),
            L"--language", wide(language),
        };
        std::wstring command;
        for (const auto & arg : args) { if (!command.empty()) command.push_back(L' '); command += quote(arg); }
        std::vector<wchar_t> mutable_command(command.begin(), command.end());
        mutable_command.push_back(L'\0');
        STARTUPINFOW si{};
        si.cb = sizeof(si);
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdInput = pipes.child_read;
        si.hStdOutput = pipes.child_write;
        si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
        PROCESS_INFORMATION pi{};
        if (!CreateProcessW(nullptr, mutable_command.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi))
            throw std::runtime_error("official tokenizer process start");
        CloseHandle(pi.hThread);
        process = pi.hProcess;
        input = pipes.parent_write; pipes.parent_write = nullptr;
        output = pipes.parent_read; pipes.parent_read = nullptr;
        CloseHandle(pipes.child_read); pipes.child_read = nullptr;
        CloseHandle(pipes.child_write); pipes.child_write = nullptr;
        const auto ready = request('R', "");
    }

    ~Impl() {
        if (input) { CloseHandle(input); input = nullptr; }
        if (process) {
            const DWORD wait = WaitForSingleObject(process, 3000);
            if (wait == WAIT_TIMEOUT) {
                TerminateProcess(process, 1);
                WaitForSingleObject(process, 3000);
            }
            CloseHandle(process);
            process = nullptr;
        }
        if (output) { CloseHandle(output); output = nullptr; }
    }

    std::vector<uint8_t> request(char mode, const std::string & text) {
        const uint32_t size = static_cast<uint32_t>(text.size());
        write_all(input, &mode, 1);
        write_all(input, &size, sizeof(size));
        if (size) write_all(input, text.data(), size);
        uint32_t status = 0, response_size = 0;
        read_all(output, &status, sizeof(status));
        read_all(output, &response_size, sizeof(response_size));
        std::vector<uint8_t> payload(response_size);
        if (response_size) read_all(output, payload.data(), payload.size());
        return payload;
    }
};

mtl_external_tokenizer::mtl_external_tokenizer(const mtl_external_tokenizer_options & options, const std::string & language)
    : impl_(std::make_unique<Impl>(options, language)) {}
mtl_external_tokenizer::~mtl_external_tokenizer() = default;

std::string mtl_external_tokenizer::punctuation(const std::string & text) {
    auto payload = impl_->request('P', text);
    return std::string(payload.begin(), payload.end());
}

std::vector<int32_t> mtl_external_tokenizer::tokenize(const std::string & text) {
    const auto payload = impl_->request('T', text);
    size_t offset = 0;
    const auto count = take_scalar<uint32_t>(payload, offset);
    std::vector<int32_t> ids(count);
    std::memcpy(ids.data(), payload.data() + offset, count * sizeof(int32_t));
    return ids;
}
