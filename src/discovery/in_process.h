// SPDX-License-Identifier: Apache-2.0
//
// In-memory discovery backend. Multiple DistributedRuntimes in one process
// share a Store, so registration/watch semantics work across "instances"
// without a server. Lease TTLs are not enforced by time here: a lease dies
// when revoked or when its owning runtime shuts down (stated assumption in
// docs/architecture.md).

#pragma once

#include "discovery/discovery.h"
#include "runtime/runtime.h"

namespace dynamo::discovery {

class InProcessDiscovery final : public Discovery {
 public:
  struct Store;

  /// Process-wide default store.
  static std::shared_ptr<Store> shared_store();

  /// Fresh isolated store (tests).
  static std::shared_ptr<Store> new_store();

  InProcessDiscovery(Runtime runtime, std::shared_ptr<Store> store = shared_store());
  ~InProcessDiscovery() override;

  coro::Task<Lease> primary_lease() override;
  coro::Task<Lease> create_lease(std::chrono::seconds ttl) override;
  coro::Task<void> kv_create(std::string key, std::string value,
                             std::optional<int64_t> lease_id) override;
  coro::Task<void> kv_put(std::string key, std::string value,
                          std::optional<int64_t> lease_id) override;
  coro::Task<void> kv_create_or_validate(std::string key, std::string value,
                                         std::optional<int64_t> lease_id) override;
  coro::Task<std::vector<KeyValue>> kv_get_prefix(std::string prefix) override;
  coro::Task<WatchStream> kv_get_and_watch_prefix(std::string prefix) override;
  coro::Task<void> publish(std::string subject, std::string payload) override;
  coro::Task<EventStream> subscribe(std::string subject) override;
  void shutdown() override;

 private:
  coro::Task<Lease> make_lease(CancellationToken token);

  Runtime runtime_;
  std::shared_ptr<Store> store_;
  std::mutex mutex_;
  std::optional<Lease> primary_;
};

}  // namespace dynamo::discovery
