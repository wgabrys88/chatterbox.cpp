#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <winsock2.h>
#include <ws2tcpip.h>
#include "tts-cpp/chatterbox/engine.h"
#include "tts-cpp/chatterbox/log.h"
#include "tts-cpp/chatterbox/nano.h"

using args_t = std::unordered_map<std::string, std::string>;

namespace {
constexpr std::uint32_t PROTOCOL_MAGIC = 0x32525454u;
constexpr std::uint32_t PROTOCOL_VERSION = 4;
constexpr std::uint32_t MAX_TEXT_BYTES = 1u << 20;

enum class request_kind : std::uint32_t { synthesize = 1, close = 3 };
enum class response_kind : std::uint32_t { pcm = 1, done = 2, error = 4, closed = 5 };

struct request_t {
    request_kind kind = request_kind::synthesize;
    std::uint32_t response = 0, piece = 0, total = 0;
    std::string text;
};

bool io_all(SOCKET socket, char* data, std::size_t size, bool sending) {
    for (std::size_t done = 0; done < size;) {
        const int n = sending
            ? send(socket, data + done, static_cast<int>(std::min<std::size_t>(size - done, 1 << 20)), 0)
            : recv(socket, data + done, static_cast<int>(std::min<std::size_t>(size - done, 1 << 20)), 0);
        if (n <= 0) return false;
        done += static_cast<std::size_t>(n);
    }
    return true;
}
bool recv_all(SOCKET socket, void* dst, std::size_t size) { return io_all(socket, static_cast<char*>(dst), size, false); }
void send_all(SOCKET socket, const void* src, std::size_t size) {
    if (!io_all(socket, const_cast<char*>(static_cast<const char*>(src)), size, true))
        throw std::runtime_error("TTS send failed");
}

struct wire_writer {
    SOCKET socket;
    std::vector<std::int16_t> pcm_buffer;

    void frame(response_kind kind, const request_t& request, std::uint32_t chunk,
               const void* data = nullptr, std::size_t size = 0) {
        if (size > UINT32_MAX) throw std::runtime_error("TTS frame too large");
        const std::uint32_t header[] = {
            PROTOCOL_MAGIC, PROTOCOL_VERSION, static_cast<std::uint32_t>(kind),
            request.response, request.piece, chunk, static_cast<std::uint32_t>(size)
        };
        send_all(socket, header, sizeof(header));
        if (size) send_all(socket, data, size);
    }
    void pcm(const request_t& request, std::uint32_t chunk, const float* samples, std::size_t count) {
        pcm_buffer.resize(count);
        for (std::size_t i = 0; i < count; ++i)
            pcm_buffer[i] = static_cast<std::int16_t>(std::clamp(samples[i], -1.0f, 1.0f) * 32767.0f);
        frame(response_kind::pcm, request, chunk, pcm_buffer.data(), pcm_buffer.size() * sizeof(std::int16_t));
    }
    void terminal(response_kind kind, const request_t& request, const std::string& message = {}) {
        frame(kind, request, 0, message.data(), message.size());
    }
};

tts_cpp::chatterbox::Engine make_engine(const args_t& args) {
    tts_cpp::chatterbox::EngineOptions o;
    o.t3_gguf_path = args.at("--model");
    o.s3gen_gguf_path = args.at("--s3gen-gguf");
    o.reference_audio = args.at("--reference");
    return tts_cpp::chatterbox::Engine(o);
}

void emit_server_config() {
    using namespace tts_cpp::chatterbox;
    tts_jsonl(std::string("{\"event\":\"server.config\",\"family\":\"nano\"")
        + ",\"seed\":" + std::to_string(SEED)
        + ",\"temperature\":" + std::to_string(TEMPERATURE)
        + ",\"top_p\":" + std::to_string(TOP_P)
        + ",\"top_k\":" + std::to_string(TOP_K)
        + ",\"repeat_penalty\":" + std::to_string(REPEAT_PENALTY)
        + ",\"repeat_last_n\":" + std::to_string(REPEAT_LAST_N)
        + ",\"max_tokens\":" + std::to_string(N_PREDICT)
        + ",\"cfm_steps\":" + std::to_string(CFM_STEPS)
        + ",\"silence_token\":" + std::to_string(SILENCE_TOKEN)
        + ",\"silence_count\":" + std::to_string(SILENCE_COUNT)
        + "}");
}

void emit_synthesis_begin(const request_t& request) {
    tts_jsonl(std::string("{\"event\":\"synthesis.begin\"")
        + ",\"response\":" + std::to_string(request.response)
        + ",\"pieces\":1"
        + ",\"total_chars\":" + std::to_string(request.text.size())
        + "}");
}

bool receive(SOCKET socket, request_t& request) {
    std::uint32_t header[7];
    if (!recv_all(socket, header, sizeof(header))) return false;
    if (header[0] != PROTOCOL_MAGIC || header[1] != PROTOCOL_VERSION)
        throw std::runtime_error("unsupported TTS protocol");
    if (header[2] != static_cast<std::uint32_t>(request_kind::synthesize) &&
        header[2] != static_cast<std::uint32_t>(request_kind::close))
        throw std::runtime_error("invalid TTS request kind");
    request = {};
    request.kind = static_cast<request_kind>(header[2]);
    request.response = header[3]; request.piece = header[4]; request.total = header[5];
    if (header[6] > MAX_TEXT_BYTES) throw std::runtime_error("TTS request too large");
    if (request.kind == request_kind::close && (header[3] || header[4] || header[5] || header[6]))
        throw std::runtime_error("invalid TTS close frame");
    request.text.resize(header[6]);
    if (header[6] && !recv_all(socket, request.text.data(), request.text.size()))
        throw std::runtime_error("truncated TTS request");
    if (request.kind == request_kind::synthesize &&
        (request.text.empty() || !request.total || request.piece >= request.total))
        throw std::runtime_error("invalid TTS synthesis frame");
    return true;
}

void serve(SOCKET client, tts_cpp::chatterbox::Engine& tts) {
    wire_writer writer{client};
    request_t request;
    if (!receive(client, request)) return;
    if (request.kind == request_kind::close) {
        writer.terminal(response_kind::closed, request);
        return;
    }
    emit_synthesis_begin(request);
    tts_context_scope synthesis_context(request.response, 0);
    try {
        tts.synthesize(request.text, [&](const float* data, std::size_t size) {
            if (size) writer.pcm(request, 0, data, size);
            writer.terminal(response_kind::done, request);
        });
    } catch (const std::exception& error) {
        writer.terminal(response_kind::error, request, error.what());
        tts_emit("synthesis.failed", "error=" + std::string(error.what()));
        return;
    }
    request_t close;
    if (!receive(client, close)) {
        tts_emit("client.disconnected", "after=synthesis");
        return;
    }
    writer.terminal(response_kind::closed, close);
}
} // namespace

int main(int argc, char** argv) {
    SOCKET listener = INVALID_SOCKET, client = INVALID_SOCKET;
    bool wsa_started = false;
    try {
        args_t args;
        for (int i = 1; i + 1 < argc; i += 2) args[argv[i]] = argv[i + 1];
        tts_set_run_identity(args.at("--run-id"));
        tts_emit("server.start", "port=" + args.at("--port"));
        auto tts = make_engine(args);
        emit_server_config();

        WSADATA wsa{};
        if (WSAStartup(MAKEWORD(2, 2), &wsa)) throw std::runtime_error("WSAStartup failed");
        wsa_started = true;
        listener = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (listener == INVALID_SOCKET) throw std::runtime_error("socket failed");
        int exclusive = 1;
        if (setsockopt(listener, SOL_SOCKET, SO_EXCLUSIVEADDRUSE,
                       reinterpret_cast<const char*>(&exclusive), sizeof(exclusive)))
            throw std::runtime_error("SO_EXCLUSIVEADDRUSE failed");
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(static_cast<unsigned short>(std::stoi(args.at("--port"))));
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address))) throw std::runtime_error("bind failed");
        if (listen(listener, 1)) throw std::runtime_error("listen failed");
        tts_emit("server.ready", "port=" + std::to_string(ntohs(address.sin_port)));

        unsigned long long connection = 0;
        for (;;) {
            client = accept(listener, nullptr, nullptr);
            if (client == INVALID_SOCKET) break;
            tts_set_connection(++connection);
            try { serve(client, tts); }
            catch (const std::exception& error) { tts_emit("serve.failed", "error=" + std::string(error.what())); }
            closesocket(client); client = INVALID_SOCKET;
        }
        closesocket(listener); listener = INVALID_SOCKET;
        WSACleanup(); wsa_started = false;
        tts_emit("server.stopped", "clean=true");
        return 0;
    } catch (const std::exception& error) {
        tts_emit("server.failed", "error=" + std::string(error.what()));
        if (client != INVALID_SOCKET) closesocket(client);
        if (listener != INVALID_SOCKET) closesocket(listener);
        if (wsa_started) WSACleanup();
        return 1;
    }
}
