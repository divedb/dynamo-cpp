// SPDX-License-Identifier: Apache-2.0
//
// TcpDiscovery: client for discoveryd. Preserves Dynamo's etcd liveness
// coupling: the primary lease shares the runtime's root token (lease loss ⇒
// runtime shutdown, shutdown ⇒ revoke); keep-alive loops run on the
// secondary executor at TTL/2 cadence with retry-until-TTL-deadline.
//
// The connection is managed by a background thread that reconnects after a
// drop (backoff up to 2s) and re-establishes watches and subscriptions.
// Watches resync via the server's snapshot + sync marker: puts are replayed
// (idempotent downstream) and deletes missed during the outage are
// synthesized by diffing the folded key set against the snapshot.

#pragma once

#include "discovery/discovery.h"
#include "runtime/runtime.h"

namespace dynamo::discovery {

class TcpDiscovery final : public Discovery,
                           public std::enable_shared_from_this<TcpDiscovery> {
 public:
  /// Connects to a discoveryd at `address` ("host:port"). Throws on failure.
  static std::shared_ptr<TcpDiscovery> connect(Runtime runtime, const std::string& address);
  ~TcpDiscovery() override;

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
  coro::Task<void> queue_dispatch(std::string subject, std::string payload) override;
  void shutdown() override;

  /// Test hook: drops the current connection without shutting down, forcing
  /// the manager through its reconnect + resync path.
  void debug_drop_connection();

  struct State;  // opaque; public so implementation helpers can name it

 private:
  TcpDiscovery() = default;
  coro::Task<Lease> make_lease(std::chrono::seconds ttl, CancellationToken token);

  std::shared_ptr<State> state_;
};

}  // namespace dynamo::discovery
