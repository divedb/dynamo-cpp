#pragma once

#include <dynamo/config.h>

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <folly/futures/Future.h>

namespace dynamo::transport {

class EtcdClient {
public:
    explicit EtcdClient(const EtcdConfig& cfg = EtcdConfig::load());
    ~EtcdClient();

    folly::Future<bool> put(const std::string& key,
                            const std::string& value);

    folly::Future<std::string> get(const std::string& key);

    folly::Future<bool> del(const std::string& key);

    folly::Future<std::vector<std::pair<std::string, std::string>>>
    get_prefix(const std::string& prefix);

    // Lease management
    folly::Future<int64_t> grant_lease(int ttl_seconds);
    folly::Future<bool> keep_alive_lease(int64_t lease_id);
    folly::Future<bool> revoke_lease(int64_t lease_id);
    folly::Future<bool> put_with_lease(const std::string& key,
                                        const std::string& value,
                                        int64_t lease_id);

    using WatchCallback = std::function<void(
        std::string key, std::string value, bool deleted)>;
    folly::Future<int64_t> watch_prefix(
        const std::string& prefix, WatchCallback callback);
    folly::Future<bool> cancel_watch(int64_t watch_id);

    void shutdown();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace dynamo::transport
