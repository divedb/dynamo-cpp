#pragma once

#include <dynamo/runtime.h>

#include <memory>
#include <string>
#include <vector>
#include <span>
#include <functional>
#include <asio.hpp>

namespace dynamo::transport {

class TcpServer {
public:
    explicit TcpServer(std::shared_ptr<Runtime> runtime, int port = 0);
    ~TcpServer();

    void start();
    void shutdown();

    int port() const noexcept { return port_; }
    bool is_running() const noexcept { return impl_ && impl_->running.load(); }

    std::string register_pending(const std::string& subject);

private:
    std::shared_ptr<Runtime> runtime_;
    std::string addr_ = "127.0.0.1";
    int port_ = 0;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class TcpClient {
public:
    explicit TcpClient(std::shared_ptr<Runtime> runtime);
    ~TcpClient();

    bool connect(const std::string& host, int port);
    void disconnect();

    bool send(std::span<const char> data);
    std::vector<char> receive(size_t max_size = 65536);

    bool is_connected() const noexcept { return connected_; }

private:
    std::shared_ptr<Runtime> runtime_;
    std::unique_ptr<asio::ip::tcp::socket> socket_;
    bool connected_ = false;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace dynamo::transport
