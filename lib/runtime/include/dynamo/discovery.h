#pragma once

#include <dynamo/runtime.h>
#include <dynamo/config.h>

#include <string>
#include <vector>
#include <functional>
#include <folly/futures/Future.h>
#include <folly/coro/Task.h>

namespace dynamo {

struct EndpointInfo {
    std::string namespace_name;
    std::string component_name;
    std::string endpoint_name;
    std::string addr;
    int port = 0;
    int64_t lease_id = 0;
    std::string metadata_json;
};

class DiscoveryClient {
public:
    explicit DiscoveryClient(std::shared_ptr<Runtime> runtime,
                             const EtcdConfig& cfg = EtcdConfig::load());

    ~DiscoveryClient();

    folly::Future<int64_t> create_lease(int ttl_seconds);

    folly::Future<folly::Unit> register_endpoint(
        const EndpointInfo& info, int64_t lease_id);

    folly::Future<folly::Unit> unregister_endpoint(
        const EndpointInfo& info);

    folly::Future<std::vector<EndpointInfo>> discover_endpoints(
        const std::string& ns,
        const std::string& component,
        const std::string& endpoint);

    using WatchCallback = std::function<void(std::vector<EndpointInfo>)>;
    folly::Future<folly::Unit> watch_endpoints(
        const std::string& ns,
        const std::string& component,
        const std::string& endpoint,
        WatchCallback callback);

    void shutdown();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace dynamo
