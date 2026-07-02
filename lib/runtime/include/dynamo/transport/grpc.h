#pragma once

#include <dynamo/runtime.h>
#include <dynamo/engine.h>

#include <memory>
#include <string>
#include <grpcpp/server.h>
#include <grpcpp/channel.h>
#include <folly/futures/Future.h>

namespace dynamo::transport {

class GrpcServer {
public:
    explicit GrpcServer(std::shared_ptr<Runtime> runtime,
                        std::string addr = "0.0.0.0:0");
    ~GrpcServer();

    void start();
    void shutdown();

    int port() const noexcept;
    std::string addr() const noexcept;

    grpc::Server& server() noexcept { return *server_; }
    grpc::ServerBuilder& builder() noexcept { return *builder_; }

private:
    std::shared_ptr<Runtime> runtime_;
    std::string addr_;
    int port_ = 0;
    std::unique_ptr<grpc::ServerBuilder> builder_;
    std::unique_ptr<grpc::Server> server_;
};

class GrpcClient {
public:
    explicit GrpcClient(std::shared_ptr<Runtime> runtime,
                        std::string target);
    ~GrpcClient();

    std::shared_ptr<grpc::Channel> channel() const noexcept {
        return channel_;
    }

    std::string target() const noexcept { return target_; }

private:
    std::shared_ptr<Runtime> runtime_;
    std::string target_;
    std::shared_ptr<grpc::Channel> channel_;
};

} // namespace dynamo::transport
