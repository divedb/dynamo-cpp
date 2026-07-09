// SPDX-License-Identifier: Apache-2.0

#include "discovery/in_process.h"

#include <algorithm>
#include <atomic>
#include <list>
#include <map>
#include <unordered_map>

#include <spdlog/spdlog.h>

namespace dynamo::discovery {

struct InProcessDiscovery::Store {
  std::mutex mutex;
  std::map<std::string, KeyValue> kvs;  // ordered: prefix scans
  int64_t revision = 0;                 // bumps on every mutation
  std::atomic<int64_t> next_lease_id{1};

  // Must hold mutex. Detaches `key` from whichever lease owns it.
  void unbind_lease_locked(const std::string& key, int64_t lease_id) {
    if (lease_id == 0) return;
    auto it = leases.find(lease_id);
    if (it == leases.end()) return;
    auto& keys = it->second.keys;
    keys.erase(std::remove(keys.begin(), keys.end(), key), keys.end());
  }

  struct LeaseRecord {
    CancellationToken token;
    std::vector<std::string> keys;
    // Keeps the revoke-callback registration alive for the lease's lifetime.
    std::shared_ptr<CancellationRegistration> on_revoke;
  };
  std::unordered_map<int64_t, LeaseRecord> leases;

  struct Watcher {
    std::string prefix;
    coro::Sender<WatchEvent> sink;
  };
  std::list<Watcher> watchers;

  struct Subscriber {
    std::string subject;
    coro::Sender<Event> sink;
  };
  std::list<Subscriber> subscribers;

  /// Per-subject round-robin cursor for queue_dispatch.
  std::unordered_map<std::string, size_t> queue_cursor;

  void notify_locked(const WatchEvent& event) {
    spdlog::debug("store notify: {} {} (watchers={})",
                  event.kind == WatchEvent::Kind::Put ? "put" : "delete", event.kv.key,
                  watchers.size());
    for (auto it = watchers.begin(); it != watchers.end();) {
      if (event.kv.key.rfind(it->prefix, 0) == 0) {
        // send() may block if a consumer is slow; watch channels are sized
        // generously so this stays rare and bounded.
        if (!it->sink.send(event)) {
          it = watchers.erase(it);
          continue;
        }
      }
      ++it;
    }
  }

  void revoke_lease(int64_t lease_id) {
    std::vector<WatchEvent> events;
    {
      std::lock_guard lock(mutex);
      auto it = leases.find(lease_id);
      if (it == leases.end()) return;
      for (auto& key : it->second.keys) {
        auto kv_it = kvs.find(key);
        if (kv_it != kvs.end()) {
          events.push_back({WatchEvent::Kind::Delete, kv_it->second});
          kvs.erase(kv_it);
        }
      }
      leases.erase(it);
      for (auto& e : events) notify_locked(e);
    }
  }
};

std::shared_ptr<InProcessDiscovery::Store> InProcessDiscovery::shared_store() {
  static auto store = std::make_shared<Store>();
  return store;
}

std::shared_ptr<InProcessDiscovery::Store> InProcessDiscovery::new_store() {
  return std::make_shared<Store>();
}

InProcessDiscovery::InProcessDiscovery(Runtime runtime, std::shared_ptr<Store> store)
    : runtime_(std::move(runtime)), store_(std::move(store)) {}

InProcessDiscovery::~InProcessDiscovery() = default;

coro::Task<Lease> InProcessDiscovery::primary_lease() {
  {
    std::lock_guard lock(mutex_);
    if (primary_) co_return *primary_;
  }
  // The primary lease shares the runtime's root token: revoking it shuts the
  // runtime down, and runtime shutdown revokes it (Dynamo's coupling).
  auto lease = co_await make_lease(runtime_.token());
  std::lock_guard lock(mutex_);
  if (!primary_) primary_ = lease;
  co_return *primary_;
}

coro::Task<Lease> InProcessDiscovery::create_lease(std::chrono::seconds /*ttl*/) {
  co_return co_await make_lease(runtime_.child_token());
}

coro::Task<Lease> InProcessDiscovery::make_lease(CancellationToken token) {
  int64_t id = store_->next_lease_id.fetch_add(1);
  Lease lease{id, token};

  auto store = store_;
  auto registration = token.register_callback([store, id] { store->revoke_lease(id); });

  std::lock_guard lock(store_->mutex);
  store_->leases[id] = Store::LeaseRecord{
      token, {}, std::make_shared<CancellationRegistration>(std::move(registration))};
  co_return lease;
}

coro::Task<void> InProcessDiscovery::kv_create(std::string key, std::string value,
                                               std::optional<int64_t> lease_id) {
  std::lock_guard lock(store_->mutex);
  if (store_->kvs.count(key)) {
    throw std::runtime_error("kv_create: key already exists: " + key);
  }
  KeyValue kv{key, std::move(value), lease_id.value_or(0), ++store_->revision};
  if (lease_id && *lease_id != 0) {
    auto it = store_->leases.find(*lease_id);
    if (it == store_->leases.end()) {
      throw std::runtime_error("kv_create: unknown or expired lease");
    }
    it->second.keys.push_back(key);
  }
  store_->kvs[key] = kv;
  store_->notify_locked({WatchEvent::Kind::Put, std::move(kv)});
  co_return;
}

coro::Task<void> InProcessDiscovery::kv_put(std::string key, std::string value,
                                            std::optional<int64_t> lease_id) {
  std::lock_guard lock(store_->mutex);
  int64_t new_lease = lease_id.value_or(0);
  if (new_lease != 0 && !store_->leases.count(new_lease)) {
    throw std::runtime_error("kv_put: unknown or expired lease");
  }
  auto existing = store_->kvs.find(key);
  if (existing != store_->kvs.end()) {
    if (existing->second.lease_id != new_lease) {
      store_->unbind_lease_locked(key, existing->second.lease_id);
      if (new_lease != 0) store_->leases[new_lease].keys.push_back(key);
    }
  } else if (new_lease != 0) {
    store_->leases[new_lease].keys.push_back(key);
  }
  KeyValue kv{key, std::move(value), new_lease, ++store_->revision};
  store_->kvs[key] = kv;
  store_->notify_locked({WatchEvent::Kind::Put, std::move(kv)});
  co_return;
}

coro::Task<void> InProcessDiscovery::kv_create_or_validate(std::string key, std::string value,
                                                           std::optional<int64_t> lease_id) {
  {
    std::lock_guard lock(store_->mutex);
    auto existing = store_->kvs.find(key);
    if (existing != store_->kvs.end()) {
      if (existing->second.value == value) co_return;  // matches: validated
      throw std::runtime_error("kv_create_or_validate: existing value differs for " + key);
    }
  }
  co_await kv_create(std::move(key), std::move(value), lease_id);
}

coro::Task<std::vector<KeyValue>> InProcessDiscovery::kv_get_prefix(std::string prefix) {
  std::vector<KeyValue> out;
  std::lock_guard lock(store_->mutex);
  for (auto it = store_->kvs.lower_bound(prefix);
       it != store_->kvs.end() && it->first.rfind(prefix, 0) == 0; ++it) {
    out.push_back(it->second);
  }
  co_return out;
}

coro::Task<WatchStream> InProcessDiscovery::kv_get_and_watch_prefix(std::string prefix) {
  std::lock_guard lock(store_->mutex);

  std::vector<KeyValue> snapshot;
  for (auto it = store_->kvs.lower_bound(prefix);
       it != store_->kvs.end() && it->first.rfind(prefix, 0) == 0; ++it) {
    snapshot.push_back(it->second);
  }

  // Snapshot + registration happen under one lock: no gaps, no duplicates.
  // Consumers resume on the background pool, never on the mutating thread
  // (which holds the store mutex while notifying).
  auto [tx, rx] =
      coro::make_channel<WatchEvent>(snapshot.size() + 256, resume_on(runtime_.secondary()));
  for (auto& kv : snapshot) {
    tx.send({WatchEvent::Kind::Put, std::move(kv)});
  }
  store_->watchers.push_back({std::move(prefix), std::move(tx)});
  co_return WatchStream{std::move(rx)};
}

coro::Task<void> InProcessDiscovery::publish(std::string subject, std::string payload) {
  std::lock_guard lock(store_->mutex);
  for (auto it = store_->subscribers.begin(); it != store_->subscribers.end();) {
    if (it->subject == subject) {
      if (!it->sink.send(Event{subject, payload})) {
        it = store_->subscribers.erase(it);
        continue;
      }
    }
    ++it;
  }
  co_return;
}

coro::Task<EventStream> InProcessDiscovery::subscribe(std::string subject) {
  auto [tx, rx] = coro::make_channel<Event>(256, resume_on(runtime_.secondary()));
  std::lock_guard lock(store_->mutex);
  store_->subscribers.push_back({std::move(subject), std::move(tx)});
  co_return EventStream{std::move(rx)};
}

coro::Task<void> InProcessDiscovery::queue_dispatch(std::string subject, std::string payload) {
  std::lock_guard lock(store_->mutex);
  std::vector<decltype(store_->subscribers.begin())> matching;
  for (auto it = store_->subscribers.begin(); it != store_->subscribers.end(); ++it) {
    if (it->subject == subject) matching.push_back(it);
  }
  size_t& cursor = store_->queue_cursor[subject];
  // Start at the cursor; a dead sink is dropped and the next member tried.
  for (size_t attempt = 0; attempt < matching.size(); ++attempt) {
    auto it = matching[cursor++ % matching.size()];
    if (it->sink.send(Event{subject, payload})) co_return;
    store_->subscribers.erase(it);
  }
  throw std::runtime_error("no responders for queue subject " + subject);
}

void InProcessDiscovery::shutdown() {}

}  // namespace dynamo::discovery
