// SPDX-License-Identifier: Apache-2.0
//
// Wire protocol (two-part frames):
//   request  header: {"op": ..., "req": N, ...}   data: value bytes (kv_create)
//   reply    header: {"reply": N, "ok": bool, "error"?, "lease_id"?, "kvs"?}
//   watch    header: {"watch": W, "kind": "put"|"delete", "key", "lease_id"}
//            data: value bytes
// Watch snapshot events are sent before the watch request's reply, so a
// client sees current state first, then live updates — no gaps, no dupes.

#include "discovery/server.h"

#include <algorithm>
#include <chrono>
#include <deque>
#include <condition_variable>
#include <list>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "transports/auth.h"
#include "transports/socket.h"
#include "transports/tls.h"

namespace dynamo::discovery {

using nlohmann::json;
using transports::Listener;
using transports::Socket;
using transports::TwoPartMessage;
using Clock = std::chrono::steady_clock;

namespace {

struct Conn {
  std::shared_ptr<Socket> sock;
  std::mutex wmutex;

  bool write(const json& header, std::string data = {}) {
    std::lock_guard lock(wmutex);
    return sock->write_frame(TwoPartMessage::from_parts(header.dump(), std::move(data)));
  }
};

}  // namespace

struct DiscoveryServer::State {
  std::optional<Listener> listener;

  std::mutex mutex;
  std::condition_variable cv;
  bool stopping = false;

  struct Entry {
    std::string value;
    int64_t lease_id = 0;
    int64_t mod_revision = 0;
  };
  std::map<std::string, Entry> kvs;

  /// Monotonic store revision (etcd-style); bumps on every put/delete.
  int64_t revision = 0;

  /// Bounded event log enabling incremental watch resync (from_rev replay).
  struct LoggedEvent {
    int64_t rev;
    bool is_put;
    std::string key;
    int64_t lease_id;
    std::string value;
  };
  std::deque<LoggedEvent> event_log;
  static constexpr size_t kEventLogCapacity = 4096;

  // Must hold mutex. Records a mutation and returns its revision.
  int64_t log_mutation_locked(bool is_put, const std::string& key, int64_t lease_id,
                              const std::string& value) {
    int64_t rev = ++revision;
    event_log.push_back({rev, is_put, key, lease_id, value});
    if (event_log.size() > kEventLogCapacity) event_log.pop_front();
    return rev;
  }

  /// True when every event after `from_rev` is still in the log.
  bool can_replay_from(int64_t from_rev) const {
    if (from_rev >= revision) return true;  // nothing missed
    if (event_log.empty()) return false;
    return event_log.front().rev <= from_rev + 1;
  }

  struct LeaseRecord {
    Clock::time_point deadline;
    std::chrono::seconds ttl{10};
    std::vector<std::string> keys;
  };
  std::unordered_map<int64_t, LeaseRecord> leases;
  int64_t next_lease_id = 1;

  struct Watcher {
    uint64_t id;
    std::string prefix;
    std::shared_ptr<Conn> conn;
  };
  std::list<Watcher> watchers;

  struct Subscriber {
    uint64_t id;  // client-chosen, unique per connection
    std::string subject;
    std::shared_ptr<Conn> conn;
  };
  std::list<Subscriber> subscribers;

  /// Per-subject round-robin cursor for queue_dispatch.
  std::unordered_map<std::string, uint64_t> queue_cursor;

  size_t active_conns = 0;
  std::unordered_map<uint64_t, std::shared_ptr<Conn>> conns;
  uint64_t next_conn_id = 0;

  std::thread accept_thread;
  std::thread sweeper_thread;

  // Must hold mutex. Logs the mutation and fans it out to matching watchers.
  void mutate_and_notify_locked(bool is_put, const std::string& key, int64_t lease_id,
                                const std::string& value) {
    int64_t rev = log_mutation_locked(is_put, key, lease_id, value);
    const char* kind = is_put ? "put" : "delete";
    for (auto it = watchers.begin(); it != watchers.end();) {
      if (key.rfind(it->prefix, 0) == 0) {
        json header{{"watch", it->id},
                    {"kind", kind},
                    {"key", key},
                    {"lease_id", lease_id},
                    {"rev", rev}};
        if (!it->conn->write(header, value)) {
          it = watchers.erase(it);
          continue;
        }
      }
      ++it;
    }
  }

  // Must hold mutex. Detaches `key` from whichever lease owns it.
  void unbind_lease_locked(const std::string& key, int64_t lease_id) {
    if (lease_id == 0) return;
    auto it = leases.find(lease_id);
    if (it == leases.end()) return;
    auto& keys = it->second.keys;
    keys.erase(std::remove(keys.begin(), keys.end(), key), keys.end());
  }

  // Must hold mutex.
  void expire_lease_locked(int64_t lease_id) {
    auto it = leases.find(lease_id);
    if (it == leases.end()) return;
    for (auto& key : it->second.keys) {
      auto kv_it = kvs.find(key);
      if (kv_it != kvs.end()) {
        mutate_and_notify_locked(false, key, lease_id, kv_it->second.value);
        kvs.erase(kv_it);
      }
    }
    leases.erase(it);
  }
};

namespace {

void handle_request(DiscoveryServer::State& s, Conn& conn, uint64_t conn_id, const json& req,
                    std::string data) {
  uint64_t req_id = req.at("req").get<uint64_t>();
  std::string op = req.at("op").get<std::string>();
  json reply{{"reply", req_id}, {"ok", true}};
  std::string reply_data;

  std::lock_guard lock(s.mutex);

  if (op == "create_lease") {
    auto ttl = std::chrono::seconds(req.value("ttl", 10));
    int64_t id = s.next_lease_id++;
    s.leases[id] = {Clock::now() + ttl, ttl, {}};
    reply["lease_id"] = id;
  } else if (op == "keepalive") {
    int64_t id = req.at("lease_id").get<int64_t>();
    auto it = s.leases.find(id);
    if (it == s.leases.end()) {
      reply["ok"] = false;
      reply["error"] = "lease expired or unknown";
    } else {
      it->second.deadline = Clock::now() + it->second.ttl;
      reply["ttl"] = static_cast<int64_t>(it->second.ttl.count());
    }
  } else if (op == "revoke") {
    s.expire_lease_locked(req.at("lease_id").get<int64_t>());
  } else if (op == "kv_create") {
    std::string key = req.at("key").get<std::string>();
    int64_t lease_id = req.value("lease_id", static_cast<int64_t>(0));
    if (s.kvs.count(key)) {
      reply["ok"] = false;
      reply["error"] = "key already exists";
    } else if (lease_id != 0 && !s.leases.count(lease_id)) {
      reply["ok"] = false;
      reply["error"] = "lease expired or unknown";
    } else {
      if (lease_id != 0) s.leases[lease_id].keys.push_back(key);
      s.mutate_and_notify_locked(true, key, lease_id, data);
      s.kvs[key] = {data, lease_id, s.revision};
    }
  } else if (op == "kv_put") {
    std::string key = req.at("key").get<std::string>();
    int64_t lease_id = req.value("lease_id", static_cast<int64_t>(0));
    if (lease_id != 0 && !s.leases.count(lease_id)) {
      reply["ok"] = false;
      reply["error"] = "lease expired or unknown";
    } else {
      auto existing = s.kvs.find(key);
      if (existing != s.kvs.end()) {
        if (existing->second.lease_id != lease_id) {
          s.unbind_lease_locked(key, existing->second.lease_id);
          if (lease_id != 0) s.leases[lease_id].keys.push_back(key);
        }
      } else if (lease_id != 0) {
        s.leases[lease_id].keys.push_back(key);
      }
      s.mutate_and_notify_locked(true, key, lease_id, data);
      s.kvs[key] = {data, lease_id, s.revision};
    }
  } else if (op == "kv_create_or_validate") {
    std::string key = req.at("key").get<std::string>();
    int64_t lease_id = req.value("lease_id", static_cast<int64_t>(0));
    auto existing = s.kvs.find(key);
    if (existing != s.kvs.end()) {
      if (existing->second.value != data) {
        reply["ok"] = false;
        reply["error"] = "existing value differs";
      }
      // else: validated, no mutation
    } else if (lease_id != 0 && !s.leases.count(lease_id)) {
      reply["ok"] = false;
      reply["error"] = "lease expired or unknown";
    } else {
      if (lease_id != 0) s.leases[lease_id].keys.push_back(key);
      s.mutate_and_notify_locked(true, key, lease_id, data);
      s.kvs[key] = {data, lease_id, s.revision};
    }
  } else if (op == "get_prefix") {
    std::string prefix = req.at("prefix").get<std::string>();
    json kvs = json::array();
    for (auto it = s.kvs.lower_bound(prefix);
         it != s.kvs.end() && it->first.rfind(prefix, 0) == 0; ++it) {
      kvs.push_back({{"key", it->first}, {"value", it->second.value},
                     {"lease_id", it->second.lease_id}, {"rev", it->second.mod_revision}});
    }
    reply["kvs"] = std::move(kvs);
    reply["revision"] = s.revision;
  } else if (op == "watch") {
    std::string prefix = req.at("prefix").get<std::string>();
    uint64_t watch_id = req.at("watch_id").get<uint64_t>();
    int64_t from_rev = req.value("from_rev", static_cast<int64_t>(-1));

    if (from_rev >= 0 && s.can_replay_from(from_rev)) {
      // Incremental resync: replay logged events after from_rev, then a
      // replay-mode sync marker (no snapshot diff needed client-side).
      for (auto& logged : s.event_log) {
        if (logged.rev <= from_rev || logged.key.rfind(prefix, 0) != 0) continue;
        json header{{"watch", watch_id},
                    {"kind", logged.is_put ? "put" : "delete"},
                    {"key", logged.key},
                    {"lease_id", logged.lease_id},
                    {"rev", logged.rev}};
        conn.write(header, logged.value);
      }
      conn.write(json{{"watch", watch_id}, {"kind", "sync"}, {"mode", "replay"},
                      {"rev", s.revision}});
    } else {
      // Full snapshot (as watch events), then a snapshot-mode sync marker so
      // a resubscribing client can diff away deletes missed in an outage.
      for (auto it = s.kvs.lower_bound(prefix);
           it != s.kvs.end() && it->first.rfind(prefix, 0) == 0; ++it) {
        json header{{"watch", watch_id}, {"kind", "put"}, {"key", it->first},
                    {"lease_id", it->second.lease_id}, {"rev", it->second.mod_revision}};
        conn.write(header, it->second.value);
      }
      conn.write(json{{"watch", watch_id}, {"kind", "sync"}, {"mode", "snapshot"},
                      {"rev", s.revision}});
    }
    // Re-registration with the same id replaces the previous entry.
    s.watchers.remove_if(
        [&](auto& w) { return w.id == watch_id && w.conn == s.conns.at(conn_id); });
    s.watchers.push_back({watch_id, prefix, s.conns.at(conn_id)});
  } else if (op == "unwatch") {
    uint64_t watch_id = req.at("watch_id").get<uint64_t>();
    s.watchers.remove_if([&](auto& w) { return w.id == watch_id; });
  } else if (op == "subscribe") {
    uint64_t sub_id = req.at("sub_id").get<uint64_t>();
    std::string subject = req.at("subject").get<std::string>();
    s.subscribers.remove_if(
        [&](auto& e) { return e.id == sub_id && e.conn == s.conns.at(conn_id); });
    s.subscribers.push_back({sub_id, std::move(subject), s.conns.at(conn_id)});
  } else if (op == "unsubscribe") {
    uint64_t sub_id = req.at("sub_id").get<uint64_t>();
    s.subscribers.remove_if(
        [&](auto& e) { return e.id == sub_id && e.conn == s.conns.at(conn_id); });
  } else if (op == "publish") {
    std::string subject = req.at("subject").get<std::string>();
    for (auto it = s.subscribers.begin(); it != s.subscribers.end();) {
      if (it->subject == subject) {
        json header{{"event", it->id}, {"subject", subject}};
        if (!it->conn->write(header, data)) {
          it = s.subscribers.erase(it);
          continue;
        }
      }
      ++it;
    }
  } else if (op == "queue_dispatch") {
    // Queue-group delivery: exactly one subscriber of `subject` gets the
    // frame (round-robin); a dead connection is dropped and the next member
    // tried. No members ⇒ NATS-style no-responders error, nothing buffered.
    std::string subject = req.at("subject").get<std::string>();
    std::vector<std::list<DiscoveryServer::State::Subscriber>::iterator> matching;
    for (auto it = s.subscribers.begin(); it != s.subscribers.end(); ++it) {
      if (it->subject == subject) matching.push_back(it);
    }
    uint64_t& cursor = s.queue_cursor[subject];
    bool delivered = false;
    for (size_t attempt = 0; attempt < matching.size(); ++attempt) {
      auto it = matching[cursor++ % matching.size()];
      json header{{"event", it->id}, {"subject", subject}};
      if (it->conn->write(header, data)) {
        delivered = true;
        break;
      }
      s.subscribers.erase(it);
    }
    if (!delivered) {
      reply["ok"] = false;
      reply["error"] = "no responders for queue subject " + subject;
    }
  } else {
    reply["ok"] = false;
    reply["error"] = "unknown op: " + op;
  }

  conn.write(reply, std::move(reply_data));
}

void serve_discovery_connection(std::shared_ptr<DiscoveryServer::State> s, uint64_t conn_id,
                                std::shared_ptr<Conn> conn) {
  try {
    if (!transports::auth::expect(*conn->sock)) {
      spdlog::warn("discoveryd: rejecting unauthenticated connection");
      conn->sock->shutdown();
    } else {
      for (;;) {
        auto frame = conn->sock->read_frame();
        if (!frame || !frame->has_header()) break;
        json req = json::parse(frame->header);
        handle_request(*s, *conn, conn_id, req, std::move(frame->data));
      }
    }
  } catch (const std::exception& e) {
    spdlog::debug("discoveryd connection ended: {}", e.what());
  }
  // Connection gone: drop its watchers/subscribers. Leases are NOT dropped
  // here — they expire via TTL, preserving semantics across reconnects.
  std::lock_guard lock(s->mutex);
  s->watchers.remove_if([&](auto& w) { return w.conn == s->conns.at(conn_id); });
  s->subscribers.remove_if([&](auto& e) { return e.conn == s->conns.at(conn_id); });
  s->conns.erase(conn_id);
  --s->active_conns;
  s->cv.notify_all();
}

}  // namespace

std::shared_ptr<DiscoveryServer> DiscoveryServer::start(const std::string& host, uint16_t port) {
  auto server = std::shared_ptr<DiscoveryServer>(new DiscoveryServer());
  server->state_ = std::make_shared<State>();
  auto s = server->state_;

  s->listener = Listener::bind(host, port, transports::tls::enabled());
  if (!s->listener) throw std::runtime_error("discoveryd: failed to bind " + host);
  spdlog::info("discoveryd listening on {}", s->listener->address());

  s->accept_thread = std::thread([s] {
    for (;;) {
      auto sock = s->listener->accept();
      if (!sock) break;
      auto conn = std::make_shared<Conn>();
      conn->sock = std::make_shared<Socket>(std::move(*sock));
      uint64_t id;
      {
        std::lock_guard lock(s->mutex);
        if (s->stopping) break;
        id = s->next_conn_id++;
        s->conns[id] = conn;
        ++s->active_conns;
      }
      std::thread(serve_discovery_connection, s, id, conn).detach();
    }
  });

  s->sweeper_thread = std::thread([s] {
    std::unique_lock lock(s->mutex);
    while (!s->stopping) {
      s->cv.wait_for(lock, std::chrono::milliseconds(200));
      if (s->stopping) break;
      auto now = Clock::now();
      std::vector<int64_t> expired;
      for (auto& [id, lease] : s->leases) {
        if (lease.deadline <= now) expired.push_back(id);
      }
      for (int64_t id : expired) {
        spdlog::debug("discoveryd: lease {} expired", id);
        s->expire_lease_locked(id);
      }
    }
  });

  return server;
}

DiscoveryServer::~DiscoveryServer() { stop(); }

std::string DiscoveryServer::address() const { return state_->listener->address(); }
uint16_t DiscoveryServer::port() const { return state_->listener->port(); }

void DiscoveryServer::stop() {
  if (!state_) return;
  {
    std::lock_guard lock(state_->mutex);
    if (state_->stopping) return;
    state_->stopping = true;
    for (auto& [id, conn] : state_->conns) conn->sock->shutdown();
  }
  state_->cv.notify_all();
  state_->listener->shutdown();
  if (state_->accept_thread.joinable()) state_->accept_thread.join();
  if (state_->sweeper_thread.joinable()) state_->sweeper_thread.join();
  std::unique_lock lock(state_->mutex);
  state_->cv.wait(lock, [&] { return state_->active_conns == 0; });
}

}  // namespace dynamo::discovery
