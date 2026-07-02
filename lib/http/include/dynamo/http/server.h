#pragma once

#include <dynamo/runtime.h>
#include <dynamo/engine.h>
#include <dynamo/llm/protocols.h>

#include <memory>
#include <string>
#include <functional>
#include <folly/coro/Task.h>

namespace dynamo::http {

class HttpServer {
public:
    HttpServer(std::shared_ptr<Runtime> runtime,
               std::string addr = "0.0.0.0",
               int port = 8080);

    ~HttpServer();

    void start();
    void shutdown();

    int port() const noexcept { return port_; }
    bool is_running() const noexcept { return running_; }

    using CompletionHandler = std::function<
        folly::coro::Task<void>(
            const dynamo::llm::CompletionRequest&,
            std::shared_ptr<ResponseStream<dynamo::llm::CompletionResponse>>)>;

    void on_complete(CompletionHandler handler) {
        complete_handler_ = std::move(handler);
    }

private:
    std::shared_ptr<Runtime> runtime_;
    std::string addr_;
    int port_ = 8080;
    bool running_ = false;
    CompletionHandler complete_handler_;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace dynamo::http
