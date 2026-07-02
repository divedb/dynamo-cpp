#include <dynamo/transport/tcp.h>

#include <spdlog/spdlog.h>

#include <asio.hpp>
#include <map>
#include <mutex>
#include <thread>
#include <atomic>

namespace dynamo::transport {

// ---------------------------------------------------------------------------
// TcpServer — TCP listener for call-home streaming responses
// ---------------------------------------------------------------------------

struct TcpServer::Impl {
    asio::io_context io_ctx;
    std::unique_ptr<asio::ip::tcp::acceptor> acceptor;
    std::thread io_thread;
    std::atomic<bool> running{false};

    // Pending registrations: subject -> callback
    std::mutex reg_mutex;
    std::map<std::string,
             std::function<void(asio::ip::tcp::socket)>> pending;
};

TcpServer::TcpServer(std::shared_ptr<Runtime> runtime, int port)
    : runtime_(std::move(runtime)), port_(port)
    , impl_(std::make_unique<Impl>()) {}

TcpServer::~TcpServer() {
    shutdown();
}

void TcpServer::start() {
    try {
        auto endpoint = asio::ip::tcp::endpoint(
            asio::ip::tcp::v4(), port_);
        impl_->acceptor = std::make_unique<asio::ip::tcp::acceptor>(
            impl_->io_ctx, endpoint);
        port_ = impl_->acceptor->local_endpoint().port();
        impl_->running = true;

        spdlog::info("TCP server listening on port {}", port_);

        auto accept_loop = [this]() {
            while (impl_->running) {
                try {
                    auto socket = impl_->acceptor->accept();
                    spdlog::debug("TCP server accepted connection");
                } catch (const std::exception& e) {
                    if (impl_->running) {
                        spdlog::warn("TCP accept error: {}", e.what());
                    }
                }
            }
        };

        impl_->io_thread = std::thread(accept_loop);
    } catch (const std::exception& e) {
        spdlog::error("TCP server start failed: {}", e.what());
    }
}

void TcpServer::shutdown() {
    if (impl_->running.exchange(false)) {
        if (impl_->acceptor) {
            impl_->acceptor->close();
        }
        if (impl_->io_thread.joinable()) {
            impl_->io_thread.join();
        }
        spdlog::debug("TCP server shut down");
    }
}

std::string TcpServer::register_pending(const std::string& subject) {
    // Atomically increment port to create unique connection info
    auto info = addr_ + ":" + std::to_string(port_);
    spdlog::debug("TCP registered pending connection for '{}'", subject);
    return info;
}

// ---------------------------------------------------------------------------
// TcpClient — caller side of TCP call-home pattern
// ---------------------------------------------------------------------------

struct TcpClient::Impl {
    asio::io_context io_ctx;
    std::thread io_thread;
    std::atomic<bool> running{false};
};

TcpClient::TcpClient(std::shared_ptr<Runtime> runtime)
    : runtime_(std::move(runtime))
    , impl_(std::make_unique<Impl>()) {}

TcpClient::~TcpClient() {
    disconnect();
}

bool TcpClient::connect(const std::string& host, int port) {
    try {
        asio::ip::tcp::resolver resolver(impl_->io_ctx);
        auto endpoints = resolver.resolve(host, std::to_string(port));
        socket_ = std::make_unique<asio::ip::tcp::socket>(impl_->io_ctx);
        asio::connect(*socket_, endpoints);
        connected_ = true;
        spdlog::debug("TCP client connected to {}:{}", host, port);
        return true;
    } catch (const std::exception& e) {
        spdlog::error("TCP client connect failed: {}", e.what());
        return false;
    }
}

void TcpClient::disconnect() {
    if (socket_ && socket_->is_open()) {
        try {
            socket_->close();
        } catch (...) {}
    }
    connected_ = false;
}

bool TcpClient::send(std::span<const char> data) {
    if (!socket_ || !socket_->is_open()) return false;
    try {
        asio::write(*socket_, asio::buffer(data.data(), data.size()));
        return true;
    } catch (const std::exception& e) {
        spdlog::error("TCP send error: {}", e.what());
        return false;
    }
}

std::vector<char> TcpClient::receive(size_t max_size) {
    if (!socket_ || !socket_->is_open()) return {};
    try {
        std::vector<char> buf(max_size);
        auto len = socket_->read_some(asio::buffer(buf));
        buf.resize(len);
        return buf;
    } catch (const std::exception& e) {
        spdlog::error("TCP receive error: {}", e.what());
        return {};
    }
}

} // namespace dynamo::transport
