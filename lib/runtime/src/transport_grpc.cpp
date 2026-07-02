#include <dynamo/transport/grpc.h>

#include <spdlog/spdlog.h>

#include <grpcpp/grpcpp.h>

namespace dynamo::transport {

GrpcServer::GrpcServer(std::shared_ptr<Runtime> runtime, std::string addr)
    : runtime_(std::move(runtime)), addr_(std::move(addr))
    , builder_(std::make_unique<grpc::ServerBuilder>()) {
    builder_->AddListeningPort(addr_, grpc::InsecureServerCredentials(), &port_);
}

GrpcServer::~GrpcServer() {
    shutdown();
}

void GrpcServer::start() {
    server_ = builder_->BuildAndStart();
    spdlog::info("gRPC server listening on {} (port {})", addr_, port_);
}

void GrpcServer::shutdown() {
    if (server_) {
        auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(5);
        server_->Shutdown(deadline);
        spdlog::debug("gRPC server shut down");
    }
}

int GrpcServer::port() const noexcept { return port_; }
std::string GrpcServer::addr() const noexcept { return addr_; }

GrpcClient::GrpcClient(std::shared_ptr<Runtime> runtime, std::string target)
    : runtime_(std::move(runtime)), target_(std::move(target)) {
    channel_ = grpc::CreateChannel(target_, grpc::InsecureChannelCredentials());
    spdlog::debug("gRPC client created for target '{}'", target_);
}

GrpcClient::~GrpcClient() = default;

} // namespace dynamo::transport
