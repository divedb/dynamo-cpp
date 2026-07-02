#pragma once

#include <dynamo/runtime.h>
#include <dynamo/engine.h>

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <folly/futures/Future.h>

namespace dynamo {

class Component;
class Endpoint;

// ---------------------------------------------------------------------------
// Namespace — logical grouping of components
// ---------------------------------------------------------------------------

class Namespace {
public:
    Namespace(std::shared_ptr<Runtime> runtime, std::string name);

    const std::string& name() const noexcept { return name_; }

    Component component(std::string component_name);

    std::string etcd_prefix() const;

private:
    std::shared_ptr<Runtime> runtime_;
    std::string name_;
};

// ---------------------------------------------------------------------------
// Component — a deployable unit hosting endpoints
// ---------------------------------------------------------------------------

class Component {
public:
    Component(std::shared_ptr<Runtime> runtime,
              Namespace* ns,
              std::string name);

    const std::string& name() const noexcept { return name_; }
    Namespace& ns() noexcept { return *ns_; }
    const Namespace& ns() const noexcept { return *ns_; }
    std::shared_ptr<Runtime> runtime() const noexcept { return runtime_; }

    Endpoint endpoint(std::string endpoint_name);

    std::string etcd_path() const;
    std::string service_name() const;

private:
    std::shared_ptr<Runtime> runtime_;
    Namespace* ns_;
    std::string name_;
};

// ---------------------------------------------------------------------------
// Endpoint — a network-accessible service on a Component
// ---------------------------------------------------------------------------

class Endpoint {
public:
    Endpoint(std::shared_ptr<Runtime> runtime,
             Component* component,
             std::string name);

    const std::string& name() const noexcept { return name_; }
    Component& component() noexcept { return *component_; }
    const Component& component() const noexcept { return *component_; }

    std::string subject() const;
    std::string etcd_path() const;

    // Register this endpoint with etcd service discovery
    folly::Future<folly::Unit> register_service(int port);

    // Create a client that can call this endpoint remotely
    template <typename Req, typename Resp>
    class Client {
    public:
        Client(std::shared_ptr<Endpoint> endpoint,
               std::string remote_addr,
               int remote_port)
            : endpoint_(std::move(endpoint))
            , addr_(std::move(remote_addr))
            , port_(remote_port) {}

        folly::coro::Task<void> generate(
            Req request,
            std::shared_ptr<ResponseStream<Resp>> stream);

        const std::string& addr() const noexcept { return addr_; }
        int port() const noexcept { return port_; }

    private:
        std::shared_ptr<Endpoint> endpoint_;
        std::string addr_;
        int port_ = 0;
    };

private:
    std::shared_ptr<Runtime> runtime_;
    Component* component_;
    std::string name_;
};

} // namespace dynamo
