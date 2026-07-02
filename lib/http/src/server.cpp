#include <dynamo/http/server.h>

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include <asio.hpp>
#include <sstream>
#include <thread>

namespace dynamo::http {

struct HttpServer::Impl {
    asio::io_context io_ctx;
    std::unique_ptr<asio::ip::tcp::acceptor> acceptor;
    std::thread io_thread;
};

HttpServer::HttpServer(std::shared_ptr<Runtime> runtime,
                       std::string addr, int port)
    : runtime_(std::move(runtime))
    , addr_(std::move(addr))
    , port_(port)
    , impl_(std::make_unique<Impl>()) {}

HttpServer::~HttpServer() {
    shutdown();
}

static std::string build_response(int status_code,
                                  const std::string& content_type,
                                  const std::string& body) {
    std::ostringstream resp;
    resp << "HTTP/1.1 " << status_code << " OK\r\n"
         << "Content-Type: " << content_type << "\r\n"
         << "Content-Length: " << body.size() << "\r\n"
         << "Connection: close\r\n"
         << "\r\n"
         << body;
    return resp.str();
}

static void handle_client(
    asio::ip::tcp::socket socket,
    CompletionHandler handler) {
    // Simple single-request handler (not production-grade)
    try {
        asio::streambuf buf;
        asio::read_until(socket, buf, "\r\n\r\n");
        std::istream stream(&buf);
        std::string method, path, version;
        stream >> method >> path >> version;

        // Read Content-Length
        std::string header;
        int content_length = 0;
        while (std::getline(stream, header) && header != "\r") {
            if (header.find("Content-Length:") == 0) {
                content_length = std::stoi(header.substr(15));
            }
        }

        // Read body
        std::string body;
        if (content_length > 0) {
            body.resize(content_length);
            asio::read(socket, asio::buffer(body));
        }

        if (path == "/v1/completions" && method == "POST") {
            try {
                auto json = nlohmann::json::parse(body);
                auto req = json.get<dynamo::llm::CompletionRequest>();

                auto stream = std::make_shared<ResponseStream<dynamo::llm::CompletionResponse>>(
                    std::make_shared<AsyncEngineContext>(),
                    [&socket](dynamo::llm::CompletionResponse resp)
                        -> folly::coro::Task<void> {
                        auto json = nlohmann::json(resp).dump();
                        auto sse = "data: " + json + "\n\n";
                        asio::write(socket, asio::buffer(sse));
                        co_return;
                    },
                    [&socket]() {
                        asio::write(socket, asio::buffer("data: [DONE]\n\n"));
                        socket.close();
                    },
                    [&socket](std::exception_ptr) {
                        socket.close();
                    });

                if (handler) {
                    handler(req, std::move(stream));
                }
            } catch (const std::exception& e) {
                auto err = build_response(400, "application/json",
                    nlohmann::json{{"error", e.what()}}.dump());
                asio::write(socket, asio::buffer(err));
            }
        } else {
            auto not_found = build_response(404, "text/plain", "Not found");
            asio::write(socket, asio::buffer(not_found));
        }

        socket.close();
    } catch (const std::exception& e) {
        spdlog::warn("HTTP handler error: {}", e.what());
    }
}

void HttpServer::start() {
    try {
        auto endpoint = asio::ip::tcp::endpoint(
            asio::ip::make_address(addr_), port_);
        impl_->acceptor = std::make_unique<asio::ip::tcp::acceptor>(
            impl_->io_ctx, endpoint);
        port_ = impl_->acceptor->local_endpoint().port();

        running_ = true;
        spdlog::info("HTTP server listening on {}:{}", addr_, port_);

        // Accept loop
        auto accept_loop = [this]() {
            while (running_) {
                try {
                    auto socket = impl_->acceptor->accept();
                    // Spawn a handler per connection
                    std::thread(&handle_client, std::move(socket),
                                complete_handler_).detach();
                } catch (const std::exception& e) {
                    if (running_) {
                        spdlog::warn("HTTP accept error: {}", e.what());
                    }
                }
            }
        };

        impl_->io_thread = std::thread(accept_loop);

    } catch (const std::exception& e) {
        spdlog::error("HTTP server start failed: {}", e.what());
        running_ = false;
    }
}

void HttpServer::shutdown() {
    if (running_) {
        running_ = false;
        if (impl_->acceptor) {
            impl_->acceptor->close();
        }
        if (impl_->io_thread.joinable()) {
            impl_->io_thread.join();
        }
        spdlog::info("HTTP server stopped");
    }
}

} // namespace dynamo::http
