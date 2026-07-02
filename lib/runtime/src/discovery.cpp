#include <dynamo/discovery.h>

#include <map>
#include <mutex>
#include <vector>
#include <atomic>
#include <thread>
#include <spdlog/spdlog.h>

namespace dynamo {

struct DiscoveryClient::Impl {
    std::shared_ptr<Runtime> runtime;
    EtcdConfig config;

    // In-memory store for local development/testing
    std::mutex mutex;
    std::map<std::string, EndpointInfo> endpoints;
    std::atomic<int64_t> next_lease_id_{1};
    std::atomic<int64_t> next_watch_id_{1};

    int64_t next_lease_id() { return next_lease_id_++; }
    int64_t next_watch_id() { return next_watch_id_++; }
};

DiscoveryClient::DiscoveryClient(
    std::shared_ptr<Runtime> runtime, const EtcdConfig& cfg)
    : impl_(std::make_unique<Impl>()) {
    impl_->runtime = std::move(runtime);
    impl_->config = cfg;
    spdlog::info("DiscoveryClient created (in-memory mode)");
}

DiscoveryClient::~DiscoveryClient() {
    shutdown();
}

folly::Future<int64_t> DiscoveryClient::create_lease(int ttl_seconds) {
    auto lease_id = impl_->next_lease_id();
    spdlog::debug("DiscoveryClient: created lease {} (TTL={}s)",
                  lease_id, ttl_seconds);
    return folly::makeFuture(lease_id);
}

folly::Future<folly::Unit> DiscoveryClient::register_endpoint(
    const EndpointInfo& info, int64_t lease_id) {
    auto key = info.namespace_name + "/" +
               info.component_name + "/" +
               info.endpoint_name + "/" +
               info.addr + ":" + std::to_string(info.port);
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        auto ep = info;
        ep.lease_id = lease_id;
        impl_->endpoints[key] = std::move(ep);
    }
    spdlog::info("DiscoveryClient: registered endpoint {}:{} ({})",
                 info.addr, info.port, key);
    return folly::makeFuture(folly::Unit{});
}

folly::Future<folly::Unit> DiscoveryClient::unregister_endpoint(
    const EndpointInfo& info) {
    auto key = info.namespace_name + "/" +
               info.component_name + "/" +
               info.endpoint_name + "/" +
               info.addr + ":" + std::to_string(info.port);
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->endpoints.erase(key);
    }
    return folly::makeFuture(folly::Unit{});
}

folly::Future<std::vector<EndpointInfo>> DiscoveryClient::discover_endpoints(
    const std::string& ns,
    const std::string& component,
    const std::string& endpoint) {
    auto prefix = ns + "/" + component + "/" + endpoint;
    std::vector<EndpointInfo> result;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        for (const auto& [key, ep] : impl_->endpoints) {
            if (key.find(prefix) == 0) {
                result.push_back(ep);
            }
        }
    }
    return folly::makeFuture(std::move(result));
}

folly::Future<folly::Unit> DiscoveryClient::watch_endpoints(
    const std::string& ns,
    const std::string& component,
    const std::string& endpoint,
    WatchCallback callback) {
    auto watch_id = impl_->next_watch_id();
    spdlog::debug("DiscoveryClient: watch #{} on {}/{}/{}",
                  watch_id, ns, component, endpoint);
    // In-memory mode: call callback once with current snapshot
    discover_endpoints(ns, component, endpoint)
        .thenValue([cb = std::move(callback)](
                       std::vector<EndpointInfo> eps) {
            cb(std::move(eps));
        });
    return folly::makeFuture(folly::Unit{});
}

void DiscoveryClient::shutdown() {
    spdlog::debug("DiscoveryClient shut down");
}

} // namespace dynamo
