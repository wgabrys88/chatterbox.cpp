#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <winsock2.h>
#include <ws2tcpip.h>
#include "tts-cpp/chatterbox/engine.h"
#include "tts-cpp/chatterbox/log.h"
#include "chatterbox_t3_internal.h"

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
void audit_write(const std::string& dir, const std::string& name, const void* data, std::size_t size) {
    if (dir.empty()) return;
    std::filesystem::create_directories(dir);
    std::ofstream out(std::filesystem::path(dir) / name, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot create wire audit artifact");
    if (size) out.write(static_cast<const char*>(data), (std::streamsize)size);
    if (!out) throw std::runtime_error("cannot write wire audit artifact");
}

struct wire_writer {
    SOCKET socket;
    std::string audit_dir;
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
        const auto bytes = pcm_buffer.size() * sizeof(std::int16_t);
        const char* tensors = std::getenv("TTS_AUDIT_TENSORS");
        if (tensors && tensors[0] == '1' && tensors[1] == 0) {
            const std::string name = "native-r" + std::to_string(request.response) + "_p" +
                std::to_string(request.piece) + "_c" + std::to_string(chunk) + ".pcm16";
            audit_write(audit_dir, name, pcm_buffer.data(), bytes);
        }
        frame(response_kind::pcm, request, chunk, pcm_buffer.data(), bytes);
    }
    void terminal(response_kind kind, const request_t& request, const std::string& message = {}) {
        frame(kind, request, 0, message.data(), message.size());
    }
};

tts_cpp::chatterbox::Engine make_engine(const args_t& args) {
    tts_cpp::chatterbox::EngineOptions o;
    auto s = [&](const char* k) -> const std::string& { return args.at(k); };
    auto i = [&](const char* k) { return std::stoi(s(k)); };
    auto f = [&](const char* k) { return std::stof(s(k)); };
    o.t3_gguf_path = s("--model"); o.s3gen_gguf_path = s("--s3gen-gguf");
    o.reference_audio = s("--reference"); o.language = s("--language");
    o.n_gpu_layers = i("--n-gpu-layers"); o.n_threads = i("--threads"); o.seed = i("--seed");
    o.n_predict = i("--max-tokens"); o.n_ctx = i("--context"); o.top_k = i("--top-k");
    o.top_p = f("--top-p"); o.min_p = f("--min-p"); o.temperature = f("--temperature");
    o.repeat_penalty = f("--repeat-penalty");
    if (args.count("--repeat-stop")) o.repeat_stop_consecutive = i("--repeat-stop");
    o.cfg_weight = f("--cfg-weight");
    o.exaggeration = f("--exaggeration"); o.cfm_steps = i("--cfm-steps");
    o.fastconv = i("--fastconv") != 0; o.audit_dir = s("--audit-dir");
    if (args.count("--forensics")) o.forensics = i("--forensics") != 0;
    if (args.count("--text-aligned")) o.text_aligned_decode = i("--text-aligned") != 0;
    if (const char* tensors = std::getenv("TTS_AUDIT_TENSORS"))
        o.audit_tensors = tensors[0] == '1' && tensors[1] == 0;
    return tts_cpp::chatterbox::Engine(o);
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += (char)c; }
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 0x20) {
            char b[8];
            std::snprintf(b, sizeof(b), "\\u%04x", c);
            out += b;
        } else out += (char)c;
    }
    return out;
}

void emit_server_config(const args_t& args) {
    auto s = [&](const char* k) -> const std::string& { return args.at(k); };
    const std::string json =
        std::string("{\"event\":\"server.config\"")
        + ",\"family\":\"" + json_escape(s("--family")) + "\""
        + ",\"seed\":" + s("--seed")
        + ",\"temperature\":" + s("--temperature")
        + ",\"min_p\":" + s("--min-p")
        + ",\"top_p\":" + s("--top-p")
        + ",\"top_k\":" + s("--top-k")
        + ",\"repeat_penalty\":" + s("--repeat-penalty")
        + ",\"repeat_last_n\":" + std::to_string(tts_cpp::chatterbox::detail::REPEAT_PENALTY_LAST_N)
        + ",\"repeat_stop_consecutive\":" + (args.count("--repeat-stop") ? s("--repeat-stop")
            : std::to_string(tts_cpp::chatterbox::detail::REPEAT_STOP_CONSECUTIVE))
        + ",\"max_tokens\":" + s("--max-tokens")
        + ",\"context\":" + s("--context")
        + ",\"cfm_steps\":" + s("--cfm-steps")
        + ",\"cfg_weight\":" + s("--cfg-weight")
        + ",\"exaggeration\":" + s("--exaggeration")
        + ",\"text_aligned_decode\":" + (args.count("--text-aligned") ? s("--text-aligned") : "1")
        + "}";
    tts_jsonl(json);
}

void emit_synthesis_begin(const std::vector<request_t>& requests) {
    std::size_t total_chars = 0;
    for (const auto& request : requests) total_chars += request.text.size();
    std::string json =
        std::string("{\"event\":\"synthesis.begin\"")
        + ",\"response\":" + std::to_string(requests[0].response)
        + ",\"pieces\":" + std::to_string(requests.size())
        + ",\"total_chars\":" + std::to_string(total_chars)
        + ",\"piece_chars\":[";
    for (std::size_t i = 0; i < requests.size(); ++i) {
        if (i) json += ',';
        json += std::to_string(requests[i].text.size());
    }
    json += "]}";
    tts_jsonl(json);
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

void serve(SOCKET client, tts_cpp::chatterbox::Engine& tts, const std::string& audit_dir) {
    wire_writer writer{client, audit_dir};
    request_t first;
    if (!receive(client, first)) return;
    if (first.kind == request_kind::close) {
        writer.terminal(response_kind::closed, first);
        return;
    }
    if (first.piece != 0) throw std::runtime_error("first TTS piece must be zero");

    std::vector<request_t> requests;
    requests.reserve(first.total);
    requests.push_back(std::move(first));
    for (std::uint32_t expected = 1; expected < requests[0].total; ++expected) {
        request_t request;
        if (!receive(client, request)) throw std::runtime_error("TTS request ended before all pieces arrived");
        if (request.kind != request_kind::synthesize || request.response != requests[0].response ||
            request.total != requests[0].total || request.piece != expected)
            throw std::runtime_error("non-contiguous TTS request");
        requests.push_back(std::move(request));
    }

    std::vector<tts_cpp::chatterbox::SynthesisPiece> pieces;
    pieces.reserve(requests.size());
    for (const auto& request : requests) {
        pieces.push_back({request.piece, request.text});
    }

    emit_synthesis_begin(requests);

    std::vector<bool> done(requests.size(), false);
    tts_context_scope synthesis_context(requests[0].response, 0);
    try {
        tts.synthesize_pieces_streaming(pieces, [&](int index, const float* data, std::size_t size, int chunk, bool final) {
            if (index < 0 || static_cast<std::size_t>(index) >= requests.size())
                throw std::runtime_error("invalid TTS callback index");
            const auto& request = requests[static_cast<std::size_t>(index)];
            tts_context_scope context(request.response, request.piece);
            if (size) {
                writer.pcm(request, static_cast<std::uint32_t>(chunk), data, size);
            }
            if (final) {
                done[static_cast<std::size_t>(index)] = true;
                writer.terminal(response_kind::done, request);
            }
        });
    } catch (const std::exception& error) {
        auto it = std::find(done.begin(), done.end(), false);
        const std::size_t index = it == done.end() ? requests.size() - 1 : static_cast<std::size_t>(it - done.begin());
        tts_context_scope context(requests[index].response, requests[index].piece);
        writer.terminal(response_kind::error, requests[index], error.what());
        tts_emit("synthesis.failed", "error=" + std::string(error.what()));
        return;
    }

    request_t close;
    if (!receive(client, close)) {
        tts_emit("client.disconnected", "after=synthesis");
        return;
    }
    if (close.kind != request_kind::close) throw std::runtime_error("expected TTS close frame");
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
        tts.warm_up();
        emit_server_config(args);

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
        tts_emit("server.ready", "port=" + std::to_string(ntohs(address.sin_port)) +
            " family=" + args.at("--family") + " language=" + args.at("--language"));

        unsigned long long connection = 0;
        for (;;) {
            client = accept(listener, nullptr, nullptr);
            if (client == INVALID_SOCKET) break;
            tts_set_connection(++connection);
            try { serve(client, tts, args.at("--audit-dir")); }
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
