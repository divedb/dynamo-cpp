// SPDX-License-Identifier: Apache-2.0

#include "discovery/etcd.h"

#include <unordered_map>

#include <etcd/KeepAlive.hpp>
#include <etcd/SyncClient.hpp>
#include <etcd/Watcher.hpp>
#include <etcd/v3/Transaction.hpp>
#include <spdlog/spdlog.h>

namespace dynamo::discovery {

namespace {

std::string with_scheme(const std::string& address) {
  if (address.rfind("http://", 0) == 0 || address.rfind("https://", 0) == 0) return address;
  return "http://" + address;
}

[[noreturn]] void throw_etcd(const char* op, const etcd::Response& resp) {
  throw std::runtime_error(std::string(op) + " failed: " + resp.error_message() +
                           " (etcd error " + std::to_string(resp.error_code()) + ")");
}

}  // namespace

// Lifecycle rule (this file's one hard constraint): etcd::KeepAlive and
// etcd::Watcher own threads that their own callbacks run on, so they must
// never be destroyed — directly or transitively — from those callbacks, and
// nothing a callback captures may ever hold the last reference to State
// (State → Runtime → ~ThreadPool joins, the recorded teardown hazard).
// Callbacks therefore capture only what they use: the lease token, a channel
// sender, or the SyncClient (shared_ptr, no threads of its own). Cancelled
// KeepAlives and abandoned Watchers stay parked in State — a few dead
// objects per process — until shutdown()/~State destroys them on the
// caller's thread.
struct EtcdDiscovery::State {
  Runtime runtime;
  std::string endpoints;
  // SyncClient calls are thread-safe (each op builds a fresh gRPC call on
  // shared stubs). shared_ptr: revoke callbacks capture it without State.
  std::shared_ptr<etcd::SyncClient> client;

  std::mutex mutex;
  bool shutdown_requested = false;
  std::optional<Lease> primary;

  struct LeaseRecord {
    std::unique_ptr<etcd::KeepAlive> keepalive;
    CancellationToken token;
    // Keeps the revoke-callback registration alive for the lease's lifetime.
    std::shared_ptr<CancellationRegistration> on_revoke;
  };
  std::unordered_map<int64_t, LeaseRecord> leases;

  std::vector<std::unique_ptr<etcd::Watcher>> watches;

  explicit State(Runtime rt) : runtime(std::move(rt)) {}
};

std::shared_ptr<EtcdDiscovery> EtcdDiscovery::connect(Runtime runtime,
                                                      const std::string& address) {
  auto client = std::shared_ptr<EtcdDiscovery>(new EtcdDiscovery());
  client->state_ = std::make_shared<State>(std::move(runtime));
  auto s = client->state_;
  s->endpoints = with_scheme(address);
  s->client = std::make_shared<etcd::SyncClient>(s->endpoints);
  s->client->set_grpc_timeout(std::chrono::seconds(5));

  // gRPC connects lazily; probe now so a bad address fails loudly here
  // (matching TcpDiscovery::connect) instead of on the first operation.
  auto head = s->client->head();
  if (!head.is_ok()) {
    throw std::runtime_error("failed to connect to etcd at " + s->endpoints + ": " +
                             head.error_message());
  }
  spdlog::debug("etcd discovery connected to {}", s->endpoints);
  return client;
}

EtcdDiscovery::~EtcdDiscovery() { shutdown(); }

coro::Task<Lease> EtcdDiscovery::make_lease(std::chrono::seconds ttl, CancellationToken token) {
  auto s = state_;
  auto granted = s->client->leasegrant(static_cast<int>(ttl.count()));
  if (!granted.is_ok()) throw_etcd("leasegrant", granted);
  int64_t id = granted.value().lease();

  // Keep-alive failure (etcd down past the TTL, lease expired server-side)
  // cancels the token. Runs on the KeepAlive's own thread: capture no State.
  auto keepalive = std::make_unique<etcd::KeepAlive>(
      *s->client,
      [id, token](std::exception_ptr eptr) mutable {
        try {
          if (eptr) std::rethrow_exception(eptr);
        } catch (const std::exception& e) {
          spdlog::warn("etcd lease {:x} keep-alive lost: {}", id, e.what());
        }
        token.cancel();
      },
      static_cast<int>(ttl.count()), id);

  // Revocation is a plain client call — no State capture, no KeepAlive
  // teardown (see the lifecycle rule above): the cancelled KeepAlive's next
  // refresh fails against the revoked lease and its loop exits on its own.
  auto registration = token.register_callback([client = s->client, id] {
    try {
      client->leaserevoke(id);
    } catch (const std::exception& e) {
      // Unreachable server: the server-side TTL reaps the lease.
      spdlog::debug("etcd leaserevoke({:x}) failed: {}", id, e.what());
    }
  });

  std::lock_guard lock(s->mutex);
  if (s->shutdown_requested) throw std::runtime_error("etcd discovery shut down");
  s->leases[id] = State::LeaseRecord{
      std::move(keepalive), token,
      std::make_shared<CancellationRegistration>(std::move(registration))};
  co_return Lease{id, token};
}

coro::Task<Lease> EtcdDiscovery::primary_lease() {
  {
    std::lock_guard lock(state_->mutex);
    if (state_->primary) co_return *state_->primary;
  }
  // Primary lease shares the runtime root token (Dynamo's liveness coupling).
  auto lease = co_await make_lease(std::chrono::seconds(10), state_->runtime.token());
  std::lock_guard lock(state_->mutex);
  if (!state_->primary) state_->primary = lease;
  co_return *state_->primary;
}

coro::Task<Lease> EtcdDiscovery::create_lease(std::chrono::seconds ttl) {
  co_return co_await make_lease(ttl, state_->runtime.child_token());
}

coro::Task<void> EtcdDiscovery::kv_create(std::string key, std::string value,
                                          std::optional<int64_t> lease_id) {
  // etcd's add() is native create-if-absent.
  auto resp = state_->client->add(key, value, lease_id.value_or(0));
  if (!resp.is_ok()) {
    if (resp.error_code() == etcd::ERROR_KEY_ALREADY_EXISTS) {
      throw std::runtime_error("kv_create: key already exists: " + key);
    }
    throw_etcd("kv_create", resp);
  }
  co_return;
}

coro::Task<void> EtcdDiscovery::kv_put(std::string key, std::string value,
                                       std::optional<int64_t> lease_id) {
  // put with a lease binds the key to it; lease 0 unbinds (upsert semantics
  // identical to the other backends).
  auto resp = state_->client->put(key, value, lease_id.value_or(0));
  if (!resp.is_ok()) throw_etcd("kv_put", resp);
  co_return;
}

coro::Task<void> EtcdDiscovery::kv_create_or_validate(std::string key, std::string value,
                                                      std::optional<int64_t> lease_id) {
  // One atomic txn: absent (create_revision == 0) ? put : read the existing
  // value for comparison. The other backends do this under their store lock.
  etcdv3::Transaction txn;
  txn.add_compare_create(key, 0);
  txn.add_success_put(key, value, lease_id.value_or(0));
  txn.add_failure_range(key);
  auto resp = state_->client->txn(txn);
  if (resp.is_ok()) co_return;  // absent: created
  // The library surfaces a failed txn compare as ERROR_COMPARE_FAILED with
  // the failure branch's results attached — here, the existing kv to compare.
  if (resp.error_code() != etcd::ERROR_COMPARE_FAILED) {
    throw_etcd("kv_create_or_validate", resp);
  }
  if (resp.values().empty() || resp.values().front().as_string() != value) {
    throw std::runtime_error("kv_create_or_validate: existing value differs for " + key);
  }
  co_return;  // validated identical
}

coro::Task<std::vector<KeyValue>> EtcdDiscovery::kv_get_prefix(std::string prefix) {
  auto resp = state_->client->ls(prefix);
  if (!resp.is_ok()) throw_etcd("kv_get_prefix", resp);
  std::vector<KeyValue> out;
  out.reserve(resp.keys().size());
  for (size_t i = 0; i < resp.keys().size(); ++i) {
    const auto& value = resp.values()[i];
    out.push_back({resp.keys()[i], value.as_string(), value.lease(), value.modified_index()});
  }
  co_return out;
}

coro::Task<WatchStream> EtcdDiscovery::kv_get_and_watch_prefix(std::string prefix) {
  auto s = state_;
  // Snapshot at revision R, then watch from R+1: no gaps, no dupes.
  auto snapshot = s->client->ls(prefix);
  if (!snapshot.is_ok()) throw_etcd("kv_get_and_watch_prefix", snapshot);
  int64_t revision = snapshot.index();

  auto [tx, rx] = coro::make_channel<WatchEvent>(snapshot.keys().size() + 1024,
                                                 resume_on(s->runtime.secondary()));
  for (size_t i = 0; i < snapshot.keys().size(); ++i) {
    const auto& value = snapshot.values()[i];
    tx.send({WatchEvent::Kind::Put,
             KeyValue{snapshot.keys()[i], value.as_string(), value.lease(),
                      value.modified_index()}});
  }

  // The callback runs on the watcher's own thread: capture no State, never
  // touch the watcher (the lifecycle rule above). Releasing the sender closes
  // the consumer's stream; the watcher object itself stays parked until
  // shutdown — its later events are discarded against the closed channel.
  auto callback = [tx = std::move(tx)](etcd::Response resp) mutable {
    if (!resp.is_ok()) {
      // Watch broken (server gone, compaction, cancel): end the stream so
      // consumers observe closure, like the discoveryd client on shutdown.
      spdlog::debug("etcd watch ended: {}", resp.error_message());
      tx = coro::Sender<WatchEvent>{};
      return;
    }
    for (const auto& event : resp.events()) {
      WatchEvent out;
      out.kind = event.event_type() == etcd::Event::EventType::PUT ? WatchEvent::Kind::Put
                                                                   : WatchEvent::Kind::Delete;
      out.kv = KeyValue{event.kv().key(), event.kv().as_string(), event.kv().lease(),
                        event.kv().modified_index()};
      if (!tx.send(std::move(out))) {  // receiver dropped: stop feeding
        tx = coro::Sender<WatchEvent>{};
        return;
      }
    }
  };

  auto watcher =
      std::make_unique<etcd::Watcher>(*s->client, prefix, revision + 1, std::move(callback),
                                      /*recursive=*/true);
  std::lock_guard lock(s->mutex);
  if (s->shutdown_requested) throw std::runtime_error("etcd discovery shut down");
  s->watches.push_back(std::move(watcher));
  co_return WatchStream{std::move(rx)};
}

coro::Task<void> EtcdDiscovery::publish(std::string subject, std::string) {
  throw std::runtime_error(
      "the etcd backend does not support transient events (publish '" + subject +
      "'): Dynamo pairs etcd with NATS for eventing — use discoveryd or in-process discovery");
  co_return;  // unreachable
}

coro::Task<EventStream> EtcdDiscovery::subscribe(std::string subject) {
  throw std::runtime_error(
      "the etcd backend does not support transient events (subscribe '" + subject +
      "'): Dynamo pairs etcd with NATS for eventing — use discoveryd or in-process discovery");
  co_return EventStream{};  // unreachable
}

coro::Task<void> EtcdDiscovery::queue_dispatch(std::string subject, std::string) {
  throw std::runtime_error(
      "the etcd backend does not support queue groups (queue_dispatch '" + subject +
      "'): use discoveryd or in-process discovery");
  co_return;  // unreachable
}

void EtcdDiscovery::shutdown() {
  if (!state_) return;
  auto s = state_;
  std::unordered_map<int64_t, State::LeaseRecord> leases;
  std::vector<std::unique_ptr<etcd::Watcher>> watches;
  {
    std::lock_guard lock(s->mutex);
    if (s->shutdown_requested) return;
    s->shutdown_requested = true;
    leases.swap(s->leases);
    watches.swap(s->watches);
  }
  // Cancel + destroy on this thread — the only place these objects die (see
  // the lifecycle rule above). Leases are NOT revoked here: shutdown is not
  // revocation — unexpired leases lapse via their server-side TTL, mirroring
  // the other backends.
  for (auto& watcher : watches) {
    if (watcher) watcher->Cancel();
  }
  for (auto& [id, record] : leases) {
    if (record.keepalive) record.keepalive->Cancel();
  }
}

}  // namespace dynamo::discovery
