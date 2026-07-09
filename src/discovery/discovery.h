// SPDX-License-Identifier: Apache-2.0
//
// Discovery abstraction: lease-based liveness plus get-and-watch registration
// (the behavioral contract of Dynamo's etcd usage). Two backends:
//   - InProcessDiscovery: shared in-memory store (single process, tests)
//   - TcpDiscovery: client for the dynamo discoveryd server (multi-process)

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "runtime/cancellation.h"
#include "runtime/coro/channel.h"
#include "runtime/coro/task.h"

namespace dynamo::discovery {

struct KeyValue {
  std::string key;
  std::string value;
  int64_t lease_id = 0;
  /// Store revision at which this key was last modified (etcd mod_revision).
  int64_t mod_revision = 0;
};

struct WatchEvent {
  enum class Kind { Put, Delete };
  Kind kind = Kind::Put;
  KeyValue kv;
};

/// A liveness lease. Keys registered under it disappear when it dies.
/// Cancelling the token revokes the lease; if the backend expires the lease
/// (missed keep-alives), the token is cancelled.
struct Lease {
  int64_t id = 0;
  CancellationToken token;
  void revoke() const { token.cancel(); }
  bool is_live() const { return !token.is_cancelled(); }
};

/// Live watch: current keys arrive first as Put events (no gaps, no dupes),
/// followed by updates. The channel closes when the watch or backend dies.
/// Dropping the receiver ends the watch.
struct WatchStream {
  coro::Receiver<WatchEvent> events;
};

/// Transient pub/sub event (Dynamo's EventPublisher/EventSubscriber role,
/// carried by NATS there). Not persisted: only current subscribers see it.
struct Event {
  std::string subject;
  std::string payload;
};

/// Live event subscription; the channel closes when the subscription or
/// backend dies. Dropping the receiver unsubscribes.
struct EventStream {
  coro::Receiver<Event> events;
};

class Discovery {
 public:
  virtual ~Discovery() = default;

  /// The lease tied to this process's runtime lifetime (created lazily).
  virtual coro::Task<Lease> primary_lease() = 0;

  virtual coro::Task<Lease> create_lease(std::chrono::seconds ttl) = 0;

  /// Atomic create-if-absent. Throws std::runtime_error if the key exists.
  virtual coro::Task<void> kv_create(std::string key, std::string value,
                                     std::optional<int64_t> lease_id) = 0;

  /// Upsert; rebinds the key's lease when `lease_id` is given.
  virtual coro::Task<void> kv_put(std::string key, std::string value,
                                  std::optional<int64_t> lease_id) = 0;

  /// Creates the key, or succeeds iff the existing value is identical
  /// (Dynamo's kv_create_or_validate transaction).
  virtual coro::Task<void> kv_create_or_validate(std::string key, std::string value,
                                                 std::optional<int64_t> lease_id) = 0;

  virtual coro::Task<std::vector<KeyValue>> kv_get_prefix(std::string prefix) = 0;

  virtual coro::Task<WatchStream> kv_get_and_watch_prefix(std::string prefix) = 0;

  /// Publishes a transient event to all current subscribers of `subject`.
  virtual coro::Task<void> publish(std::string subject, std::string payload) = 0;

  /// Subscribes to events on `subject` (exact match).
  virtual coro::Task<EventStream> subscribe(std::string subject) = 0;

  virtual void shutdown() = 0;
};

using DiscoveryPtr = std::shared_ptr<Discovery>;

}  // namespace dynamo::discovery
