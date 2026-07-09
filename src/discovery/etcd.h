// SPDX-License-Identifier: Apache-2.0
//
// EtcdDiscovery: real etcd v3 backend via the system etcd-cpp-apiv3 client
// (compiled only when CMake finds it; see DYNAMO_WITH_ETCD). Preserves the
// same contracts as the other backends:
//   - leases: etcd leasegrant + a KeepAlive per lease; keep-alive failure
//     cancels the token, token cancellation revokes the lease. The primary
//     lease shares the runtime's root token (Dynamo's liveness coupling).
//   - kv: kv_create = etcd's create-if-absent; kv_create_or_validate runs as
//     a single txn (create_revision==0 ? put : range) — atomic, no race.
//   - watch: snapshot at revision R (as Puts), then a native etcd watch from
//     R+1 — no gaps, no dupes. Dropping the receiver ends the watch.
//
// Deliberately NOT supported: publish/subscribe/queue_dispatch. etcd has no
// transient pub/sub; Dynamo pairs etcd with NATS for events. These throw —
// use discoveryd or in-process discovery when events/queue groups are needed.
//
// Blocking etcd calls run on the calling pool thread (SyncClient), matching
// the TcpDiscovery pattern; watch/keep-alive callbacks run on the client
// library's own threads and never resume coroutines inline (channel sends
// resume consumers on the runtime's secondary executor).

#pragma once

#include "discovery/discovery.h"
#include "runtime/runtime.h"

namespace dynamo::discovery {

class EtcdDiscovery final : public Discovery,
                            public std::enable_shared_from_this<EtcdDiscovery> {
 public:
  /// Connects to etcd at `address` ("host:port" or a full "http://..." URL).
  /// Throws std::runtime_error if the endpoint is unreachable.
  static std::shared_ptr<EtcdDiscovery> connect(Runtime runtime, const std::string& address);
  ~EtcdDiscovery() override;

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

  struct State;  // opaque; keeps the etcd headers out of this header

 private:
  EtcdDiscovery() = default;
  coro::Task<Lease> make_lease(std::chrono::seconds ttl, CancellationToken token);

  std::shared_ptr<State> state_;
};

}  // namespace dynamo::discovery
