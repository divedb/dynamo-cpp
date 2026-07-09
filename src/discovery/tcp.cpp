// SPDX-License-Identifier: Apache-2.0

#include "discovery/tcp.h"

#include <set>
#include <thread>
#include <unordered_map>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "transports/socket.h"

namespace dynamo::discovery {

using nlohmann::json;
using transports::Socket;
using transports::TwoPartMessage;

namespace {

struct Reply {
  json header;
  std::string data;
};

}  // namespace

struct TcpDiscovery::State {
  Runtime runtime;
  std::string address;

  // Current connection; writes serialize on wmutex; the manager thread
  // replaces the socket on reconnect.
  std::shared_ptr<Socket> sock;
  std::mutex wmutex;

  std::mutex mutex;
  bool shutdown_requested = false;
  bool connected = false;
  uint64_t next_req_id = 1;
  uint64_t next_watch_id = 1;
  uint64_t next_sub_id = 1;
  std::unordered_map<uint64_t, coro::Sender<Reply>> pending;

  struct WatchRecord {
    std::string prefix;
    coro::Sender<WatchEvent> sink;
    std::set<std::string> live_keys;  // keys currently believed alive
    std::set<std::string> sync_keys;  // snapshot keys gathered during a resync
    bool syncing = false;
    int64_t last_rev = 0;  // highest revision observed; resync starts here
  };
  std::unordered_map<uint64_t, WatchRecord> watches;

  struct SubRecord {
    std::string subject;
    coro::Sender<Event> sink;
  };
  std::unordered_map<uint64_t, SubRecord> subs;

  // Lease tokens created through this client (their keep-alive loops enforce
  // the TTL deadline; connection loss does not cancel them directly).
  std::unordered_map<int64_t, CancellationToken> lease_tokens;

  std::optional<Lease> primary;
  std::thread manager;

  explicit State(Runtime rt) : runtime(std::move(rt)) {}

  bool write(const json& header, std::string data = {}) {
    std::lock_guard lock(wmutex);
    if (!sock) return false;
    return sock->write_frame(TwoPartMessage::from_parts(header.dump(), std::move(data)));
  }

  /// Sends a request and awaits its reply. Throws if disconnected.
  coro::Task<Reply> request(json header, std::string data = {}) {
    // Replies resume off the manager thread so continuations can issue
    // further requests without deadlocking against it.
    auto [tx, rx] = coro::make_channel<Reply>(1, resume_on(runtime.secondary()));
    uint64_t id;
    {
      std::lock_guard lock(mutex);
      if (shutdown_requested || !connected) {
        throw std::runtime_error("discovery connection lost");
      }
      id = next_req_id++;
      pending.emplace(id, std::move(tx));
    }
    header["req"] = id;
    if (!write(header, std::move(data))) {
      std::lock_guard lock(mutex);
      pending.erase(id);
      throw std::runtime_error("discovery connection lost");
    }
    auto reply = co_await rx.recv();
    if (!reply) throw std::runtime_error("discovery connection lost");
    co_return std::move(*reply);
  }

  // --- manager thread -------------------------------------------------------

  void manager_loop() {
    int backoff_ms = 200;
    bool first = true;
    for (;;) {
      {
        std::lock_guard lock(mutex);
        if (shutdown_requested) break;
      }
      if (!first) {
        auto [host, port] = transports::parse_address(address);
        auto fresh = Socket::connect(host, port);
        if (!fresh) {
          std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
          backoff_ms = std::min(backoff_ms * 2, 2000);
          continue;
        }
        {
          std::lock_guard lock(wmutex);
          sock = std::make_shared<Socket>(std::move(*fresh));
        }
        spdlog::info("discovery: reconnected to {}", address);
      }
      first = false;
      backoff_ms = 200;

      resubscribe_all();
      {
        std::lock_guard lock(mutex);
        connected = true;
      }

      read_until_disconnect();

      // Disconnected: fail in-flight requests; watches/subs survive for resync.
      std::unordered_map<uint64_t, coro::Sender<Reply>> dropped;
      {
        std::lock_guard lock(mutex);
        connected = false;
        dropped.swap(pending);
        if (shutdown_requested) break;
      }
      spdlog::warn("discovery connection to {} lost; reconnecting", address);
    }

    // Permanent shutdown: close watch/sub channels (leases lapse via their
    // keep-alive deadlines; an explicit runtime shutdown revokes them).
    std::unordered_map<uint64_t, WatchRecord> dropped_watches;
    std::unordered_map<uint64_t, SubRecord> dropped_subs;
    std::unordered_map<uint64_t, coro::Sender<Reply>> dropped_pending;
    {
      std::lock_guard lock(mutex);
      connected = false;
      dropped_watches.swap(watches);
      dropped_subs.swap(subs);
      dropped_pending.swap(pending);
    }
  }

  /// Re-registers all watches (entering resync mode) and subscriptions on a
  /// fresh connection. Watches pass their last seen revision so the server
  /// can replay incrementally instead of resending a full snapshot. Uses
  /// req id 0: replies are not awaited.
  void resubscribe_all() {
    struct WatchResub {
      uint64_t id;
      std::string prefix;
      int64_t from_rev;
    };
    std::vector<WatchResub> watch_list;
    std::vector<std::pair<uint64_t, std::string>> sub_list;
    {
      std::lock_guard lock(mutex);
      for (auto& [id, record] : watches) {
        record.syncing = true;
        record.sync_keys.clear();
        watch_list.push_back({id, record.prefix, record.last_rev});
      }
      for (auto& [id, record] : subs) sub_list.emplace_back(id, record.subject);
    }
    for (auto& resub : watch_list) {
      json header{{"op", "watch"}, {"watch_id", resub.id}, {"prefix", resub.prefix}, {"req", 0}};
      if (resub.from_rev > 0) header["from_rev"] = resub.from_rev;
      write(header);
    }
    for (auto& [id, subject] : sub_list) {
      write({{"op", "subscribe"}, {"sub_id", id}, {"subject", subject}, {"req", 0}});
    }
  }

  void read_until_disconnect() {
    std::shared_ptr<Socket> reader_sock;
    {
      std::lock_guard lock(wmutex);
      reader_sock = sock;
    }
    try {
      for (;;) {
        auto frame = reader_sock->read_frame();
        if (!frame || !frame->has_header()) break;
        json header = json::parse(frame->header);
        if (header.contains("reply")) {
          route_reply(header, std::move(frame->data));
        } else if (header.contains("watch")) {
          route_watch_frame(header, std::move(frame->data));
        } else if (header.contains("event")) {
          route_event(header, std::move(frame->data));
        }
      }
    } catch (const std::exception& e) {
      spdlog::debug("discovery reader: {}", e.what());
    }
  }

  void route_reply(json& header, std::string data) {
    uint64_t id = header.at("reply").get<uint64_t>();
    coro::Sender<Reply> tx;
    {
      std::lock_guard lock(mutex);
      auto it = pending.find(id);
      if (it == pending.end()) return;
      tx = std::move(it->second);
      pending.erase(it);
    }
    tx.send(Reply{std::move(header), std::move(data)});
  }

  void route_watch_frame(json& header, std::string data) {
    uint64_t watch_id = header.at("watch").get<uint64_t>();
    std::string kind = header.at("kind").get<std::string>();

    coro::Sender<WatchEvent> sink;
    std::vector<WatchEvent> to_send;
    {
      std::lock_guard lock(mutex);
      auto it = watches.find(watch_id);
      if (it == watches.end()) return;
      auto& record = it->second;
      sink = record.sink;

      if (kind == "sync") {
        // A marker outside a sync window (duplicate registration during a
        // reconnect race) must be ignored, or it would wipe the live set.
        if (!record.syncing) return;
        std::string mode = header.value("mode", "snapshot");
        if (mode == "snapshot") {
          // End of a snapshot: emit deletes for keys that vanished while we
          // were away, then adopt the snapshot as the live set.
          for (auto& key : record.live_keys) {
            if (!record.sync_keys.count(key)) {
              to_send.push_back({WatchEvent::Kind::Delete, KeyValue{key, "", 0, 0}});
            }
          }
          record.live_keys = std::move(record.sync_keys);
        }
        // Replay mode: the replayed events already updated live_keys —
        // including explicit deletes — so no diff is needed.
        record.sync_keys.clear();
        record.syncing = false;
        record.last_rev = std::max(record.last_rev,
                                   header.value("rev", static_cast<int64_t>(0)));
      } else if (kind == "put") {
        WatchEvent event;
        event.kind = WatchEvent::Kind::Put;
        event.kv.key = header.at("key").get<std::string>();
        event.kv.lease_id = header.value("lease_id", static_cast<int64_t>(0));
        event.kv.mod_revision = header.value("rev", static_cast<int64_t>(0));
        event.kv.value = std::move(data);
        record.live_keys.insert(event.kv.key);
        if (record.syncing) record.sync_keys.insert(event.kv.key);
        record.last_rev = std::max(record.last_rev, event.kv.mod_revision);
        to_send.push_back(std::move(event));
      } else {  // delete
        WatchEvent event;
        event.kind = WatchEvent::Kind::Delete;
        event.kv.key = header.at("key").get<std::string>();
        event.kv.lease_id = header.value("lease_id", static_cast<int64_t>(0));
        event.kv.mod_revision = header.value("rev", static_cast<int64_t>(0));
        event.kv.value = std::move(data);
        record.live_keys.erase(event.kv.key);
        record.last_rev = std::max(record.last_rev, event.kv.mod_revision);
        to_send.push_back(std::move(event));
      }
    }

    for (auto& event : to_send) {
      if (!sink.send(std::move(event))) {
        // Consumer gone: unregister on both sides (best effort).
        {
          std::lock_guard lock(mutex);
          watches.erase(watch_id);
        }
        write({{"op", "unwatch"}, {"watch_id", watch_id}, {"req", 0}});
        return;
      }
    }
  }

  void route_event(json& header, std::string data) {
    uint64_t sub_id = header.at("event").get<uint64_t>();
    coro::Sender<Event> sink;
    std::string subject;
    {
      std::lock_guard lock(mutex);
      auto it = subs.find(sub_id);
      if (it == subs.end()) return;
      sink = it->second.sink;
      subject = it->second.subject;
    }
    if (!sink.send(Event{std::move(subject), std::move(data)})) {
      {
        std::lock_guard lock(mutex);
        subs.erase(sub_id);
      }
      write({{"op", "unsubscribe"}, {"sub_id", sub_id}, {"req", 0}});
    }
  }
};

namespace {

/// Keep-alive loop for one lease: heartbeat at TTL/2; revoke on cancellation.
/// Transient failures are retried until the TTL deadline (etcd semantics);
/// the token is cancelled only when the deadline passes or the backend
/// authoritatively reports the lease dead.
coro::Task<void> keep_alive_loop(std::shared_ptr<TcpDiscovery::State> s, int64_t lease_id,
                                 std::chrono::seconds ttl, CancellationToken token) {
  using Clock = std::chrono::steady_clock;
  const auto ttl_ms = std::chrono::duration_cast<std::chrono::milliseconds>(ttl);
  const auto healthy_interval = ttl_ms / 2;
  const auto retry_interval = std::min(ttl_ms / 4, std::chrono::milliseconds(500));

  auto deadline = Clock::now() + ttl_ms;
  auto interval = healthy_interval;

  for (;;) {
    bool cancelled = co_await token.wait_for(s->runtime.secondary(), interval);
    if (cancelled) {
      try {
        co_await s->request({{"op", "revoke"}, {"lease_id", lease_id}});
      } catch (...) {
        // Connection gone; the server-side TTL will reap the lease.
      }
      break;
    }
    try {
      auto reply = co_await s->request({{"op", "keepalive"}, {"lease_id", lease_id}});
      if (!reply.header.value("ok", false)) {
        spdlog::warn("lease {:x} expired or revoked by server", lease_id);
        token.cancel();
        break;
      }
      deadline = Clock::now() + ttl_ms;  // heartbeat landed: reset the window
      interval = healthy_interval;
    } catch (const std::exception& e) {
      if (Clock::now() >= deadline) {
        spdlog::warn("lease {:x} lost: no heartbeat within ttl ({}s): {}", lease_id,
                     ttl.count(), e.what());
        token.cancel();
        break;
      }
      spdlog::debug("lease {:x} keep-alive failed ({}); retrying until deadline", lease_id,
                    e.what());
      interval = retry_interval;
    }
  }
  std::lock_guard lock(s->mutex);
  s->lease_tokens.erase(lease_id);
}

}  // namespace

std::shared_ptr<TcpDiscovery> TcpDiscovery::connect(Runtime runtime, const std::string& address) {
  auto [host, port] = transports::parse_address(address);
  auto sock = Socket::connect(host, port);
  if (!sock) throw std::runtime_error("failed to connect to discoveryd at " + address);

  auto client = std::shared_ptr<TcpDiscovery>(new TcpDiscovery());
  client->state_ = std::make_shared<State>(std::move(runtime));
  auto s = client->state_;
  s->address = address;
  s->sock = std::make_shared<Socket>(std::move(*sock));
  s->connected = true;
  s->manager = std::thread([s] { s->manager_loop(); });
  return client;
}

TcpDiscovery::~TcpDiscovery() {
  shutdown();
  if (state_ && state_->manager.joinable()) state_->manager.join();
}

coro::Task<Lease> TcpDiscovery::make_lease(std::chrono::seconds ttl, CancellationToken token) {
  auto s = state_;
  auto reply =
      co_await s->request({{"op", "create_lease"}, {"ttl", static_cast<int64_t>(ttl.count())}});
  if (!reply.header.value("ok", false)) {
    throw std::runtime_error("create_lease failed: " + reply.header.value("error", ""));
  }
  int64_t id = reply.header.at("lease_id").get<int64_t>();
  {
    std::lock_guard lock(s->mutex);
    s->lease_tokens.emplace(id, token);
  }
  s->runtime.spawn_background(keep_alive_loop(s, id, ttl, token));
  co_return Lease{id, token};
}

coro::Task<Lease> TcpDiscovery::primary_lease() {
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

coro::Task<Lease> TcpDiscovery::create_lease(std::chrono::seconds ttl) {
  co_return co_await make_lease(ttl, state_->runtime.child_token());
}

coro::Task<void> TcpDiscovery::kv_create(std::string key, std::string value,
                                         std::optional<int64_t> lease_id) {
  auto reply = co_await state_->request(
      {{"op", "kv_create"}, {"key", key}, {"lease_id", lease_id.value_or(0)}}, std::move(value));
  if (!reply.header.value("ok", false)) {
    throw std::runtime_error("kv_create failed: " + reply.header.value("error", ""));
  }
}

coro::Task<void> TcpDiscovery::kv_put(std::string key, std::string value,
                                      std::optional<int64_t> lease_id) {
  auto reply = co_await state_->request(
      {{"op", "kv_put"}, {"key", key}, {"lease_id", lease_id.value_or(0)}}, std::move(value));
  if (!reply.header.value("ok", false)) {
    throw std::runtime_error("kv_put failed: " + reply.header.value("error", ""));
  }
}

coro::Task<void> TcpDiscovery::kv_create_or_validate(std::string key, std::string value,
                                                     std::optional<int64_t> lease_id) {
  auto reply = co_await state_->request(
      {{"op", "kv_create_or_validate"}, {"key", key}, {"lease_id", lease_id.value_or(0)}},
      std::move(value));
  if (!reply.header.value("ok", false)) {
    throw std::runtime_error("kv_create_or_validate failed: " +
                             reply.header.value("error", ""));
  }
}

coro::Task<std::vector<KeyValue>> TcpDiscovery::kv_get_prefix(std::string prefix) {
  auto reply = co_await state_->request({{"op", "get_prefix"}, {"prefix", prefix}});
  std::vector<KeyValue> out;
  for (auto& item : reply.header.at("kvs")) {
    out.push_back({item.at("key").get<std::string>(), item.at("value").get<std::string>(),
                   item.value("lease_id", static_cast<int64_t>(0)),
                   item.value("rev", static_cast<int64_t>(0))});
  }
  co_return out;
}

coro::Task<WatchStream> TcpDiscovery::kv_get_and_watch_prefix(std::string prefix) {
  auto s = state_;
  auto [tx, rx] = coro::make_channel<WatchEvent>(1024, resume_on(s->runtime.secondary()));
  uint64_t watch_id;
  {
    std::lock_guard lock(s->mutex);
    if (s->shutdown_requested) throw std::runtime_error("discovery connection lost");
    watch_id = s->next_watch_id++;
    State::WatchRecord record;
    record.prefix = prefix;
    record.sink = std::move(tx);
    record.syncing = true;  // the initial snapshot ends with a sync marker too
    s->watches.emplace(watch_id, std::move(record));
  }
  // The server streams the snapshot (as watch events) and the sync marker
  // before the reply; the manager routes them into the channel above.
  auto reply =
      co_await s->request({{"op", "watch"}, {"prefix", prefix}, {"watch_id", watch_id}});
  if (!reply.header.value("ok", false)) {
    std::lock_guard lock(s->mutex);
    s->watches.erase(watch_id);
    throw std::runtime_error("watch failed: " + reply.header.value("error", ""));
  }
  co_return WatchStream{std::move(rx)};
}

coro::Task<void> TcpDiscovery::publish(std::string subject, std::string payload) {
  auto reply =
      co_await state_->request({{"op", "publish"}, {"subject", subject}}, std::move(payload));
  if (!reply.header.value("ok", false)) {
    throw std::runtime_error("publish failed: " + reply.header.value("error", ""));
  }
}

coro::Task<EventStream> TcpDiscovery::subscribe(std::string subject) {
  auto s = state_;
  auto [tx, rx] = coro::make_channel<Event>(256, resume_on(s->runtime.secondary()));
  uint64_t sub_id;
  {
    std::lock_guard lock(s->mutex);
    if (s->shutdown_requested) throw std::runtime_error("discovery connection lost");
    sub_id = s->next_sub_id++;
    s->subs.emplace(sub_id, State::SubRecord{subject, std::move(tx)});
  }
  auto reply =
      co_await s->request({{"op", "subscribe"}, {"sub_id", sub_id}, {"subject", subject}});
  if (!reply.header.value("ok", false)) {
    std::lock_guard lock(s->mutex);
    s->subs.erase(sub_id);
    throw std::runtime_error("subscribe failed: " + reply.header.value("error", ""));
  }
  co_return EventStream{std::move(rx)};
}

void TcpDiscovery::shutdown() {
  if (!state_) return;
  {
    std::lock_guard lock(state_->mutex);
    state_->shutdown_requested = true;
  }
  std::lock_guard lock(state_->wmutex);
  if (state_->sock) state_->sock->shutdown();  // unblocks the manager
}

void TcpDiscovery::debug_drop_connection() {
  std::lock_guard lock(state_->wmutex);
  if (state_->sock) state_->sock->shutdown();
}

}  // namespace dynamo::discovery
