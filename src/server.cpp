#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <winsock2.h>
#include <ws2tcpip.h>
#include "tts-cpp/chatterbox/engine.h"
#include "tts-cpp/chatterbox/log.h"

using mono_clock = std::chrono::steady_clock;
using args_t = std::unordered_map<std::string, std::string>;

namespace {
constexpr std::uint32_t PROTOCOL_MAGIC = 0x32525454u; // "TTR2" little-endian
constexpr std::uint32_t PROTOCOL_VERSION = 2;
constexpr std::uint32_t MAX_TEXT_BYTES = 1u << 20;

enum class request_kind : std::uint32_t { synthesize = 1, advance_epoch = 2, close = 3 };
enum class response_kind : std::uint32_t { pcm = 1, done = 2, cancelled = 3, error = 4, closed = 5 };

struct request_t {
    request_kind kind = request_kind::synthesize;
    std::uint32_t epoch = 0;
    std::uint32_t response = 0;
    std::uint32_t piece = 0;
    std::string text;
    mono_clock::time_point queued{};
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

bool recv_all(SOCKET socket, void* dst, std::size_t size) {
    return io_all(socket, static_cast<char*>(dst), size, false);
}

void send_all(SOCKET socket, const void* src, std::size_t size) {
    if (!io_all(socket, const_cast<char*>(static_cast<const char*>(src)), size, true))
        throw std::runtime_error("TTS send failed");
}

std::string elapsed_ms(mono_clock::time_point a, mono_clock::time_point b) {
    return std::to_string((int)std::chrono::duration<double, std::milli>(a - b).count());
}

struct wire_writer {
    SOCKET socket;
    std::mutex mutex;

    void frame(response_kind kind, const request_t& request, std::uint32_t chunk, const void* data = nullptr, std::size_t size = 0) {
        if (size > UINT32_MAX) throw std::runtime_error("TTS frame too large");
        const std::uint32_t header[] = {
            PROTOCOL_MAGIC, PROTOCOL_VERSION, static_cast<std::uint32_t>(kind),
            request.epoch, request.response, request.piece, chunk, static_cast<std::uint32_t>(size)
        };
        std::lock_guard lock(mutex);
        send_all(socket, header, sizeof(header));
        if (size) send_all(socket, data, size);
    }

    void pcm(const request_t& request, std::uint32_t chunk, const float* samples, std::size_t count) {
        thread_local std::vector<std::int16_t> out; out.resize(count);
        for (std::size_t i = 0; i < count; ++i) out[i] = static_cast<std::int16_t>(std::clamp(samples[i], -1.0f, 1.0f) * 32767.0f);
        frame(response_kind::pcm, request, chunk, out.data(), out.size() * sizeof(std::int16_t));
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
    o.repeat_penalty = f("--repeat-penalty"); o.cfg_weight = f("--cfg-weight");
    o.exaggeration = f("--exaggeration"); o.cfm_steps = i("--cfm-steps");
    o.fastconv = i("--fastconv") != 0;
    return tts_cpp::chatterbox::Engine(o);
}

bool receive(SOCKET socket, request_t& request) {
    std::uint32_t header[7];
    if (!recv_all(socket, header, sizeof(header))) return false;
    if (header[0] != PROTOCOL_MAGIC || header[1] != PROTOCOL_VERSION) throw std::runtime_error("unsupported TTS protocol");
    if (header[2] < static_cast<std::uint32_t>(request_kind::synthesize) || header[2] > static_cast<std::uint32_t>(request_kind::close))
        throw std::runtime_error("invalid TTS request kind");
    request = {};
    request.kind = static_cast<request_kind>(header[2]);
    request.epoch = header[3]; request.response = header[4]; request.piece = header[5];
    if (header[6] > MAX_TEXT_BYTES) throw std::runtime_error("TTS request too large");
    if (request.kind != request_kind::synthesize && (header[6] || request.response || request.piece))
        throw std::runtime_error("invalid TTS control frame");
    request.text.resize(header[6]);
    if (header[6] && !recv_all(socket, request.text.data(), request.text.size())) throw std::runtime_error("truncated TTS request");
    if (request.kind == request_kind::synthesize && request.text.empty()) throw std::runtime_error("empty TTS sentence");
    request.queued = mono_clock::now();
    return true;
}

void serve(SOCKET client, tts_cpp::chatterbox::Engine& tts) {
    wire_writer writer{client};
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<request_t> pending;
    std::optional<request_t> active;
    std::set<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>> accepted;
    std::atomic<std::uint32_t> live_epoch{0};
    std::atomic<bool> stop{false};
    bool failed = false;
    auto cancel_queued = [&](std::uint32_t new_epoch) {
        std::vector<request_t> cancelled;
        for (auto it = pending.begin(); it != pending.end();) {
            if (it->epoch != new_epoch) { cancelled.push_back(std::move(*it)); it = pending.erase(it); }
            else ++it;
        }
        return cancelled;
    };
    auto acknowledge_cancelled = [&](const std::vector<request_t>& cancelled) {
        for (const auto& request : cancelled) {
            tts_context_scope context(request.epoch, request.response, request.piece);
            writer.terminal(response_kind::cancelled, request);
            tts_emit("synthesis.cancelled", " state=queued");
        }
    };
    auto finish_piece = [&](const request_t& request, response_kind kind, const std::string& message = {}) {
        writer.terminal(kind, request, message);
        active.reset();
    };

    std::thread synth([&] {
        std::optional<request_t> previous;
        mono_clock::time_point previous_ended{};
        try {
        for (;;) {
            request_t request;
            {
                std::unique_lock lock(mutex);
                changed.wait(lock, [&] { return stop.load(std::memory_order_acquire) || !pending.empty(); });
                if (stop.load(std::memory_order_acquire) && pending.empty()) break;
                request = std::move(pending.front()); pending.pop_front(); active = request;
                tts.begin_synthesis();
            }
            tts_context_scope context(request.epoch, request.response, request.piece);
            const auto started = mono_clock::now();
            tts_emit("synthesis.start", std::string(" chars=") + std::to_string(request.text.size()));
            try {
                std::uint32_t chunks = 0; bool first_pcm = true;
                tts.synthesize_pieces_streaming({request.text}, [&](int, const float* data, std::size_t size, int chunk, bool) {
                    if (live_epoch.load(std::memory_order_acquire) != request.epoch) {
                        if (size) tts_emit("synthesis.cancelled", std::string(" state=pcm-after-advance bytes=") + std::to_string(size * sizeof(std::int16_t)));
                        return;
                    }
                    chunks = std::max(chunks, static_cast<std::uint32_t>(chunk + 1));
                    if (!size) return;
                    if (first_pcm) { first_pcm = false; tts_emit("synthesis.first_result", std::string(" bytes=") + std::to_string(size * sizeof(std::int16_t)) + " ms=" + elapsed_ms(mono_clock::now(), started)); }
                    writer.pcm(request, static_cast<std::uint32_t>(chunk), data, size);
                });
                bool completed = false;
                {
                    std::lock_guard lock(mutex);
                    completed = live_epoch.load(std::memory_order_acquire) == request.epoch;
                    finish_piece(request, completed ? response_kind::done : response_kind::cancelled);
                }
                previous_ended = mono_clock::now(); previous = request;
                if (completed) tts_emit("synthesis.completed", std::string(" ms=") + elapsed_ms(previous_ended, started) + " terminal=done");
                else tts_emit("synthesis.cancelled", " state=active");
            } catch (const std::exception& error) {
                bool cancelled = false;
                {
                    std::lock_guard lock(mutex);
                    cancelled = live_epoch.load(std::memory_order_acquire) != request.epoch || stop.load(std::memory_order_acquire);
                    finish_piece(request, cancelled ? response_kind::cancelled : response_kind::error, cancelled ? std::string{} : error.what());
                    if (!cancelled) { failed = true; stop.store(true, std::memory_order_release); shutdown(client, SD_RECEIVE); }
                }
                previous_ended = mono_clock::now(); previous = request;
                if (cancelled) tts_emit("synthesis.cancelled", " state=active");
                else tts_emit("synthesis.failed", " error=" + std::string(error.what()));
            }
            {
                std::lock_guard lock(mutex);
                if (active && active->epoch == request.epoch && active->response == request.response && active->piece == request.piece) active.reset();
            }
            changed.notify_all();
        }
        } catch (const std::exception& error) {
            {
                std::lock_guard lock(mutex);
                failed = true; stop.store(true, std::memory_order_release); active.reset();
            }
            tts.cancel(); shutdown(client, SD_BOTH); changed.notify_all();
            tts_emit("synthesis.worker.failed", " error=" + std::string(error.what()));
        }
    });

    bool close_requested = false;
    try {
        request_t request;
        while (!close_requested) {
            tts_emit("serve.recv-wait", " entering");
            if (!receive(client, request)) {
                tts_emit("serve.recv-false", " reason=client_closed");
                break;
            }
            if (request.kind == request_kind::synthesize) {
                if (request.epoch != live_epoch.load(std::memory_order_acquire)) {
                    writer.terminal(response_kind::error, request, "sentence epoch is not live"); continue;
                }
                {
                    std::lock_guard lock(mutex);
                    if (!accepted.insert(std::make_tuple(request.epoch, request.response, request.piece)).second)
                        throw std::runtime_error("duplicate synthesis identity");
                    pending.push_back(request);
                }
                { tts_context_scope context(request.epoch, request.response, request.piece); tts_emit("synthesis.queued"); }
                changed.notify_one();
                continue;
            }

            if (request.kind == request_kind::advance_epoch) {
                const auto old_epoch = live_epoch.load(std::memory_order_acquire);
                if (request.epoch <= old_epoch) throw std::runtime_error("epoch must advance monotonically");
                bool cancel_active = false;
                std::vector<request_t> queued_cancelled;
                std::optional<request_t> in_flight;
                {
                    std::lock_guard lock(mutex);
                    live_epoch.store(request.epoch, std::memory_order_release);
                    tts_set_live_epoch(request.epoch);
                    cancel_active = active.has_value() && active->epoch != request.epoch;
                    if (cancel_active) in_flight = active;
                    queued_cancelled = cancel_queued(request.epoch);
                }
                acknowledge_cancelled(queued_cancelled);
                if (cancel_active) tts.cancel();
                tts_emit("epoch.advanced", " from=" + std::to_string(old_epoch) + " to=" + std::to_string(request.epoch));
                changed.notify_all();
                continue;
            }

            if (request.kind == request_kind::close) {
                close_requested = true;
                std::vector<request_t> queued_cancelled;
                bool cancel_active = false;
                {
                    std::lock_guard lock(mutex);
                    cancel_active = active.has_value();
                    queued_cancelled = cancel_queued(UINT32_MAX);
                    stop.store(true, std::memory_order_release);
                    const auto closing_epoch = live_epoch.fetch_add(1, std::memory_order_acq_rel) + 1;
                    tts_set_live_epoch(closing_epoch);
                }
                acknowledge_cancelled(queued_cancelled);
                if (cancel_active) tts.cancel();
                tts_emit("server.close.requested", " ok");
                changed.notify_all();
            }
        }
    } catch (...) {
        tts_emit("serve.catch-all", " entering");
        {
            std::lock_guard lock(mutex);
            stop.store(true, std::memory_order_release);
            const auto failed_epoch = live_epoch.fetch_add(1, std::memory_order_acq_rel) + 1;
            tts_set_live_epoch(failed_epoch);
        }
        tts.cancel(); changed.notify_all();
        if (synth.joinable()) synth.join();
        tts_session_emit();
        tts_emit("serve.catch-all", " done");
        return;
    }

    const bool unexpected_disconnect = !close_requested;
    if (unexpected_disconnect) {
        std::lock_guard lock(mutex);
        stop.store(true, std::memory_order_release);
        const auto disconnected_epoch = live_epoch.fetch_add(1, std::memory_order_acq_rel) + 1;
        tts_set_live_epoch(disconnected_epoch);
        tts.cancel();
        changed.notify_all();
    }
    if (synth.joinable()) synth.join();
    tts_session_emit();
    if (failed) { tts_emit("serve.failed", " ok"); }
    if (unexpected_disconnect) { tts_emit("client.disconnected", " ok"); }
    if (close_requested) {
        request_t close_frame{request_kind::close, live_epoch.load(), 0, 0, {}};
        writer.terminal(response_kind::closed, close_frame);
        tts_emit("server.closed", " ok");
    }
}
} // namespace

int main(int argc, char** argv) {
    SOCKET listener = INVALID_SOCKET, client = INVALID_SOCKET;
    bool wsa_started = false;
    try {
        args_t args;
        for (int i = 1; i + 1 < argc; i += 2) args[argv[i]] = argv[i + 1];
        tts_set_run_identity(args.at("--run-id"));
        tts_emit("server.start", std::string(" port=") + args.at("--port"));
        auto tts = make_engine(args);
        tts.warm_up();

        WSADATA wsa{};
        if (WSAStartup(MAKEWORD(2, 2), &wsa)) throw std::runtime_error("WSAStartup failed");
        wsa_started = true;
        listener = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (listener == INVALID_SOCKET) throw std::runtime_error("socket failed");
        int exclusive = 1;
        if (setsockopt(listener, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, reinterpret_cast<const char*>(&exclusive), sizeof(exclusive)))
            throw std::runtime_error("SO_EXCLUSIVEADDRUSE failed");
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(static_cast<unsigned short>(std::stoi(args.at("--port"))));
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address))) throw std::runtime_error("bind failed");
        if (listen(listener, 1)) throw std::runtime_error("listen failed");
        tts_emit("server.ready", std::string(" port=") + std::to_string(ntohs(address.sin_port))
            + " family=" + args.at("--family") + " language=" + args.at("--language"));

        static int iter = 0;
        for (;;) {
            tts_emit("accept-loop.iteration", std::string(" iter=") + std::to_string(++iter));
            client = accept(listener, nullptr, nullptr);
            tts_emit("accept-loop.accepted", std::string(" client=") + std::to_string(client));
            if (client == INVALID_SOCKET) {
                tts_emit("accept-loop.broken", std::string(" error=") + std::to_string(WSAGetLastError()));
                break;
            }
            tts_emit("client.accepted", " ok");
            try {
                tts_emit("serve.begin", " ok");
                serve(client, tts);
                tts_emit("serve.end", " ok");
            }
            catch (const std::exception& e) {
                tts_emit("serve.catch", std::string(" error=") + e.what());
                closesocket(client); client = INVALID_SOCKET;
            }
            closesocket(client); client = INVALID_SOCKET;
            tts_emit("client.done", std::string(" iter=") + std::to_string(iter));
        }
        closesocket(listener); listener = INVALID_SOCKET;
        WSACleanup(); wsa_started = false;
        tts_emit("server.stopped", " clean=true");
        return 0;
    } catch (const std::exception& error) {
        tts_emit("server.failed", " error=" + std::string(error.what()));
        if (client != INVALID_SOCKET) closesocket(client);
        if (listener != INVALID_SOCKET) closesocket(listener);
        if (wsa_started) WSACleanup();
        return 1;
    }
}
