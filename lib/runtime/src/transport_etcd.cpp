#include <dynamo/transport/etcd.h>

#include <map>
#include <mutex>
#include <atomic>
#include <spdlog/spdlog.h>

namespace dynamo::transport {

struct EtcdClient::Impl {
    EtcdConfig config;
    std::mutex mutex;
    std::map<std::string, std::string> store;
    std::atomic<int64_t> next_lease_id_{1};
    std::atomic<int64_t> next_watch_id_{1};

    int64_t next_lease_id() { return next_lease_id_++; }
    int64_t next_watch_id() { return next_watch_id_++; }
};

EtcdClient::EtcdClient(const EtcdConfig& cfg)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = cfg;
    spdlog::info("EtcdClient created (in-memory mode, endpoints: {})",
                 cfg.endpoints);
}

EtcdClient::~EtcdClient() {
    shutdown();
}

folly::Future<bool> EtcdClient::put(
    const std::string& key, const std::string& value) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->store[key] = value;
    return folly::makeFuture(true);
}

folly::Future<std::string> EtcdClient::get(const std::string& key) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto it = impl_->store.find(key);
    if (it == impl_->store.end()) {
        return folly::makeFuture<std::string>("");
    }
    return folly::makeFuture(it->second);
}

folly::Future<bool> EtcdClient::del(const std::string& key) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return folly::makeFuture(impl_->store.erase(key) > 0);
}

folly::Future<std::vector<std::pair<std::string, std::string>>>
EtcdClient::get_prefix(const std::string& prefix) {
    std::vector<std::pair<std::string, std::string>> result;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    for (const auto& [key, value] : impl_->store) {
        if (key.find(prefix) == 0) {
            result.emplace_back(key, value);
        }
    }
    return folly::makeFuture(std::move(result));
}

folly::Future<int64_t> EtcdClient::grant_lease(int ttl_seconds) {
    auto id = impl_->next_lease_id();
    spdlog::debug("EtcdClient: granted lease {} (TTL={}s)", id, ttl_seconds);
    return folly::makeFuture(id);
}

folly::Future<bool> EtcdClient::keep_alive_lease(int64_t lease_id) {
    spdlog::debug("EtcdClient: keep-alive for lease {}", lease_id);
    return folly::makeFuture(true);
}

folly::Future<bool> EtcdClient::revoke_lease(int64_t lease_id) {
    spdlog::debug("EtcdClient: revoked lease {}", lease_id);
    return folly::makeFuture(true);
}

folly::Future<bool> EtcdClient::put_with_lease(
    const std::string& key, const std::string& value, int64_t lease_id) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->store[key] = value;
    spdlog::debug("EtcdClient: put '{}' with lease {}", key, lease_id);
    return folly::makeFuture(true);
}

folly::Future<int64_t> EtcdClient::watch_prefix(
    const std::string& prefix, WatchCallback callback) {
    auto id = impl_->next_watch_id();
    spdlog::debug("EtcdClient: watch #{} on prefix '{}'", id, prefix);
    // Immediately call callback with current snapshot
    auto results = get_prefix(prefix).get();
    for (const auto& [k, v] : results) {
        callback(k, v, false);
    }
    return folly::makeFuture(id);
}

folly::Future<bool> EtcdClient::cancel_watch(int64_t watch_id) {
    spdlog::debug("EtcdClient: cancelled watch #{}", watch_id);
    return folly::makeFuture(true);
}

void EtcdClient::shutdown() {
    spdlog::debug("EtcdClient shut down");
}

} // namespace dynamo::transport
