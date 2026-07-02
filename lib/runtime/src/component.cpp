#include <dynamo/component.h>

#include <algorithm>
#include <cctype>
#include <spdlog/spdlog.h>

namespace dynamo {

// ---------------------------------------------------------------------------
// Namespace
// ---------------------------------------------------------------------------

Namespace::Namespace(std::shared_ptr<Runtime> runtime, std::string name)
    : runtime_(std::move(runtime)), name_(std::move(name)) {}

Component Namespace::component(std::string component_name) {
    return Component(runtime_, this, std::move(component_name));
}

std::string Namespace::etcd_prefix() const {
    return "/dynamo/namespaces/" + name_;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

Component::Component(std::shared_ptr<Runtime> runtime,
                     Namespace* ns,
                     std::string name)
    : runtime_(std::move(runtime)), ns_(ns), name_(std::move(name)) {}

Endpoint Component::endpoint(std::string endpoint_name) {
    return Endpoint(runtime_, this, std::move(endpoint_name));
}

std::string Component::etcd_path() const {
    return ns_->etcd_prefix() + "/components/" + name_;
}

std::string Component::service_name() const {
    auto name = ns_->name() + "|" + name_;
    // Slugify: replace non-alphanumeric with underscores
    std::transform(name.begin(), name.end(), name.begin(),
                   [](char c) { return std::isalnum(c) ? c : '_'; });
    return name;
}

// ---------------------------------------------------------------------------
// Endpoint
// ---------------------------------------------------------------------------

Endpoint::Endpoint(std::shared_ptr<Runtime> runtime,
                   Component* component,
                   std::string name)
    : runtime_(std::move(runtime)), component_(component), name_(std::move(name)) {}

std::string Endpoint::subject() const {
    return component_->service_name() + "." + name_;
}

std::string Endpoint::etcd_path() const {
    return component_->etcd_path() + "/endpoints/" + name_;
}

folly::Future<folly::Unit> Endpoint::register_service(int port) {
    spdlog::info("Registering endpoint {} on port {}",
                 subject(), port);
    // etcd registration happens via DiscoveryClient
    return folly::makeFuture(folly::Unit{});
}

} // namespace dynamo
