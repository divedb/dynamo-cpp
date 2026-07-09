// SPDX-License-Identifier: Apache-2.0

#include "transports/control_plane.h"

#include <condition_variable>
#include <thread>
#include <unordered_map>

#include <spdlog/spdlog.h>

#include "transports/socket.h"

namespace dynamo::transports {

using nlohmann::json;

struct ControlPlaneServer::State {
  Runtime runtime;
  std::optional<Listener> listener;

  std::mutex mutex;
  std::condition_variable cv;
  bool stopping = false;
  std::unordered_map<std::string, std::shared_ptr<PushHandler>> handlers;
  std::unordered_map<std::string, QueryHandler> queries;

  uint64_t next_conn_id = 0;
  size_t active_conns = 0;
  std::unordered_map<uint64_t, std::shared_ptr<Socket>> conn_sockets;

  std::thread accept_thread;

  explicit State(Runtime rt) : runtime(std::move(rt)) {}
};

namespace {

void send_ack(Socket& sock, bool ok, const std::string& error, std::string payload = {}) {
  DispatchAck ack{ok, error};
  sock.write_frame(TwoPartMessage::from_parts(json(ack).dump(), std::move(payload)));
}

/// Owns the handler for the duration of the (member-coroutine) handle()
/// frame: without this, unregistering the subject mid-request frees the
/// handler under a running coroutine's feet.
coro::Task<void> run_handler(std::shared_ptr<PushHandler> handler,
                             RequestControlMessage control, std::string payload) {
  co_await handler->handle(std::move(control), std::move(payload));
}

void serve_control_connection(std::shared_ptr<ControlPlaneServer::State> s, uint64_t conn_id,
                              std::shared_ptr<Socket> sock) {
  struct Cleanup {
    std::shared_ptr<ControlPlaneServer::State>& s;
    uint64_t id;
    ~Cleanup() {
      std::lock_guard lock(s->mutex);
      s->conn_sockets.erase(id);
      --s->active_conns;
      s->cv.notify_all();
    }
  } cleanup{s, conn_id};

  try {
    // Persistent connection: serve requests until the client closes or errs.
    for (;;) {
      auto frame = sock->read_frame();
      if (!frame || !frame->has_header()) break;
      DispatchHeader header = json::parse(frame->header);

      std::shared_ptr<PushHandler> handler;
      QueryHandler query;
      {
        std::lock_guard lock(s->mutex);
        if (auto it = s->handlers.find(header.subject); it != s->handlers.end()) {
          handler = it->second;
        } else if (auto qit = s->queries.find(header.subject); qit != s->queries.end()) {
          query = qit->second;
        }
      }

      if (handler) {
        // Ack first (fire-and-forget dispatch, like Dynamo's PushEndpoint);
        // subsequent errors travel via the data-plane prologue.
        send_ack(*sock, true, "");
        s->runtime.spawn(
            run_handler(std::move(handler), std::move(header.control), std::move(frame->data)));
      } else if (query) {
        send_ack(*sock, true, "", query(frame->data));
      } else {
        send_ack(*sock, false, "unknown subject: " + header.subject);
      }
    }
  } catch (const std::exception& e) {
    spdlog::warn("control plane connection error: {}", e.what());
  }
  sock->shutdown();  // not close(): see data_plane.cpp — stop() may race a shutdown()
}

}  // namespace

std::shared_ptr<ControlPlaneServer> ControlPlaneServer::start(Runtime runtime,
                                                              const std::string& host,
                                                              uint16_t port) {
  auto server = std::shared_ptr<ControlPlaneServer>(new ControlPlaneServer());
  server->state_ = std::make_shared<State>(std::move(runtime));
  auto s = server->state_;

  s->listener = Listener::bind(host, port);
  if (!s->listener) throw std::runtime_error("control plane: failed to bind " + host);
  spdlog::debug("control plane listening on {}", s->listener->address());

  s->accept_thread = std::thread([s] {
    for (;;) {
      auto sock = s->listener->accept();
      if (!sock) break;
      auto shared_sock = std::make_shared<Socket>(std::move(*sock));
      uint64_t id;
      {
        std::lock_guard lock(s->mutex);
        if (s->stopping) break;
        id = s->next_conn_id++;
        s->conn_sockets[id] = shared_sock;
        ++s->active_conns;
      }
      std::thread(serve_control_connection, s, id, shared_sock).detach();
    }
  });

  return server;
}

ControlPlaneServer::~ControlPlaneServer() { stop(); }

std::string ControlPlaneServer::address() const { return state_->listener->address(); }

void ControlPlaneServer::register_handler(const std::string& subject,
                                          std::shared_ptr<PushHandler> handler) {
  std::lock_guard lock(state_->mutex);
  state_->handlers[subject] = std::move(handler);
}

void ControlPlaneServer::unregister_handler(const std::string& subject) {
  std::lock_guard lock(state_->mutex);
  state_->handlers.erase(subject);
}

void ControlPlaneServer::register_query(const std::string& subject, QueryHandler handler) {
  std::lock_guard lock(state_->mutex);
  state_->queries[subject] = std::move(handler);
}

void ControlPlaneServer::unregister_query(const std::string& subject) {
  std::lock_guard lock(state_->mutex);
  state_->queries.erase(subject);
}

void ControlPlaneServer::stop() {
  if (!state_) return;
  {
    std::lock_guard lock(state_->mutex);
    if (state_->stopping) return;
    state_->stopping = true;
    for (auto& [id, sock] : state_->conn_sockets) sock->shutdown();
  }
  state_->listener->shutdown();
  if (state_->accept_thread.joinable()) state_->accept_thread.join();
  std::unique_lock lock(state_->mutex);
  state_->cv.wait(lock, [&] { return state_->active_conns == 0; });
}

namespace {

/// Per-target pool of idle control-plane connections. Sockets are used by
/// one request at a time; stale entries (server restarted) are detected by
/// the roundtrip's retry-once-with-fresh-connection logic.
class ConnectionPool {
 public:
  std::optional<Socket> acquire(const std::string& address) {
    std::lock_guard lock(mutex_);
    auto it = idle_.find(address);
    if (it == idle_.end() || it->second.empty()) return std::nullopt;
    Socket sock = std::move(it->second.back());
    it->second.pop_back();
    return sock;
  }

  void release(const std::string& address, Socket sock) {
    sock.set_recv_timeout(std::chrono::milliseconds::zero());
    std::lock_guard lock(mutex_);
    auto& idle = idle_[address];
    if (idle.size() < kMaxIdlePerTarget) idle.push_back(std::move(sock));
    // else: dropped, closing the connection
  }

 private:
  static constexpr size_t kMaxIdlePerTarget = 8;
  std::mutex mutex_;
  std::unordered_map<std::string, std::vector<Socket>> idle_;
};

ConnectionPool& pool() {
  static ConnectionPool instance;
  return instance;
}

std::pair<Socket, TwoPartMessage> roundtrip(const std::string& address,
                                            const DispatchHeader& header, std::string payload,
                                            std::chrono::milliseconds recv_timeout) {
  TwoPartMessage msg = TwoPartMessage::from_parts(json(header).dump(), std::move(payload));

  // Attempt 0 reuses a pooled connection (may be stale); attempt 1 is fresh.
  for (int attempt = 0; attempt < 2; ++attempt) {
    bool pooled = false;
    std::optional<Socket> sock;
    if (attempt == 0) {
      sock = pool().acquire(address);
      pooled = sock.has_value();
    }
    if (!sock) {
      auto [host, port] = parse_address(address);
      sock = Socket::connect(host, port);
      if (!sock) throw std::runtime_error("control plane: cannot reach instance at " + address);
    }
    if (recv_timeout.count() > 0) sock->set_recv_timeout(recv_timeout);

    if (!sock->write_frame(msg)) {
      if (pooled) continue;  // stale pooled connection
      throw std::runtime_error("control plane: failed to send request to " + address);
    }
    std::optional<TwoPartMessage> ack_frame;
    try {
      ack_frame = sock->read_frame();
    } catch (const std::exception&) {
      if (pooled) continue;
      throw;
    }
    if (!ack_frame || !ack_frame->has_header()) {
      if (pooled) continue;
      throw std::runtime_error("control plane: no ack from " + address +
                               " (peer closed or timed out)");
    }
    return {std::move(*sock), std::move(*ack_frame)};
  }
  throw std::runtime_error("control plane: request to " + address + " failed");
}

}  // namespace

void dispatch_request(const std::string& address, const DispatchHeader& header,
                      std::string payload, std::chrono::milliseconds ack_timeout) {
  auto [sock, ack_frame] = roundtrip(address, header, std::move(payload), ack_timeout);
  DispatchAck ack = json::parse(ack_frame.header);
  if (!ack.ok) throw std::runtime_error("control plane: dispatch rejected: " + ack.error);
  pool().release(address, std::move(sock));
}

std::string dispatch_query(const std::string& address, const std::string& subject,
                           std::string payload, std::chrono::milliseconds timeout) {
  DispatchHeader header;
  header.subject = subject;
  auto [sock, ack_frame] = roundtrip(address, header, std::move(payload), timeout);
  DispatchAck ack = json::parse(ack_frame.header);
  if (!ack.ok) throw std::runtime_error("control plane: query rejected: " + ack.error);
  pool().release(address, std::move(sock));
  return std::move(ack_frame.data);
}

}  // namespace dynamo::transports
