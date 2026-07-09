// SPDX-License-Identifier: Apache-2.0

#include "transports/data_plane.h"

#include <atomic>
#include <condition_variable>
#include <thread>
#include <unordered_map>

#include <spdlog/spdlog.h>

namespace dynamo::transports {

using nlohmann::json;

namespace {

bool write_control(Socket& sock, std::mutex& wmutex, ControlKind kind) {
  ControlFrame frame{kind};
  std::lock_guard lock(wmutex);
  return sock.write_frame(TwoPartMessage::from_header(json(frame).dump()));
}

}  // namespace

// ---------------------------------------------------------------------------
// DataPlaneServer
// ---------------------------------------------------------------------------

struct DataPlaneServer::State {
  std::optional<Runtime> runtime;
  std::optional<Listener> listener;

  std::mutex mutex;
  std::condition_variable cv;
  bool stopping = false;

  struct Pending {
    pipeline::ContextPtr ctx;
    coro::Sender<StreamArrival> fulfill;
    size_t recv_buffer_count = 64;
  };
  std::unordered_map<std::string, Pending> pending;

  uint64_t next_conn_id = 0;
  size_t active_conns = 0;
  std::unordered_map<uint64_t, std::shared_ptr<Socket>> conn_sockets;

  std::thread accept_thread;
};

namespace {

void serve_data_connection(std::shared_ptr<DataPlaneServer::State> s, uint64_t conn_id,
                           std::shared_ptr<Socket> sock) {
  struct Cleanup {
    std::shared_ptr<DataPlaneServer::State>& s;
    uint64_t id;
    ~Cleanup() {
      std::lock_guard lock(s->mutex);
      s->conn_sockets.erase(id);
      --s->active_conns;
      s->cv.notify_all();
    }
  } cleanup{s, conn_id};

  try {
    auto hs_frame = sock->read_frame();
    if (!hs_frame || !hs_frame->has_header()) return;
    CallHomeHandshake handshake = json::parse(hs_frame->header);

    if (handshake.stream_type != StreamKind::Response) {
      // Request streams (ManyIn) are not implemented — reject explicitly
      // (Dynamo's process_request_stream is likewise a stub).
      spdlog::warn("data plane: rejecting unsupported stream_type on subject {}",
                   handshake.subject);
      return;
    }

    DataPlaneServer::State::Pending entry;
    {
      std::lock_guard lock(s->mutex);
      auto it = s->pending.find(handshake.subject);
      if (it == s->pending.end()) {
        spdlog::warn("data plane: call-home for unknown subject {}", handshake.subject);
        return;
      }
      entry = std::move(it->second);
      s->pending.erase(it);
    }

    auto prologue_frame = sock->read_frame();
    if (!prologue_frame || !prologue_frame->has_header()) return;  // fulfill dropped → error upstream
    Prologue prologue = json::parse(prologue_frame->header);
    if (prologue.error) {
      entry.fulfill.send(StreamArrival{prologue.error, std::nullopt});
      return;
    }

    // Consumers resume on the primary pool: this conn thread must keep
    // servicing the socket regardless of what consumers do next.
    auto [tx, rx] = coro::make_channel<std::string>(entry.recv_buffer_count,
                                                    resume_on(s->runtime->primary()));

    // Relay requester-side stop/kill to the worker as control frames.
    auto wmutex = std::make_shared<std::mutex>();
    std::weak_ptr<Socket> weak_sock = sock;
    if (auto controller = std::dynamic_pointer_cast<pipeline::Controller>(entry.ctx)) {
      controller->on_stop([weak_sock, wmutex] {
        if (auto ss = weak_sock.lock()) write_control(*ss, *wmutex, ControlKind::Stop);
      });
      controller->on_kill([weak_sock, wmutex] {
        if (auto ss = weak_sock.lock()) write_control(*ss, *wmutex, ControlKind::Kill);
      });
    }

    entry.fulfill.send(StreamArrival{std::nullopt, IncomingStream{std::move(rx)}});

    for (;;) {
      auto frame = sock->read_frame();
      if (!frame) {
        spdlog::debug("data plane: stream {} closed by worker without sentinel", handshake.subject);
        break;
      }
      if (frame->has_header()) {
        ControlFrame control = json::parse(frame->header);
        if (control.kind == ControlKind::Sentinel) break;
      }
      if (frame->has_data()) {
        if (!tx.send(std::move(frame->data))) {
          // Consumer dropped the stream: tell the worker to kill generation.
          write_control(*sock, *wmutex, ControlKind::Kill);
          break;
        }
      }
    }
  } catch (const std::exception& e) {
    spdlog::warn("data plane connection error: {}", e.what());
  }
  // FIN via shutdown, NOT close: stop() may concurrently call shutdown() on
  // this socket through the conn map, and close() would free the fd under it
  // (a racing shutdown could then hit a recycled fd). The fd is closed by the
  // Socket destructor once the map entry is gone (cleanup guard, same mutex).
  sock->shutdown();
}

}  // namespace

std::shared_ptr<DataPlaneServer> DataPlaneServer::start(Runtime runtime, const std::string& host,
                                                        uint16_t port) {
  auto server = std::shared_ptr<DataPlaneServer>(new DataPlaneServer());
  server->state_ = std::make_shared<State>();
  auto s = server->state_;
  s->runtime = std::move(runtime);

  s->listener = Listener::bind(host, port);
  if (!s->listener) throw std::runtime_error("data plane: failed to bind " + host);
  spdlog::debug("data plane listening on {}", s->listener->address());

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
      std::thread(serve_data_connection, s, id, shared_sock).detach();
    }
  });

  return server;
}

DataPlaneServer::~DataPlaneServer() { stop(); }

std::string DataPlaneServer::address() const { return state_->listener->address(); }

RegisteredStream DataPlaneServer::register_response_stream(pipeline::ContextPtr ctx,
                                                           size_t recv_buffer_count) {
  // Capacity 2: the worker's arrival and a synthetic timeout marker can both
  // land without blocking their (timer/conn) threads.
  auto [tx, rx] = coro::make_channel<StreamArrival>(2, resume_on(state_->runtime->primary()));
  std::string subject = pipeline::Controller::generate_id();

  ConnectionInfo info;
  info.address = address();
  info.subject = subject;
  info.context_id = ctx->id();
  info.stream_type = StreamKind::Response;

  {
    std::lock_guard lock(state_->mutex);
    if (!state_->stopping) {
      state_->pending.emplace(subject,
                              State::Pending{std::move(ctx), tx, recv_buffer_count});
    }
    // If stopping, tx is dropped here and the arrival channel reports closure.
  }
  return RegisteredStream{std::move(info), std::move(rx), std::move(tx)};
}

void DataPlaneServer::deregister(const std::string& subject) {
  std::lock_guard lock(state_->mutex);
  state_->pending.erase(subject);
}

void DataPlaneServer::stop() {
  if (!state_) return;
  {
    std::lock_guard lock(state_->mutex);
    if (state_->stopping) return;
    state_->stopping = true;
    // Fail pending registrations explicitly (their arrival channels stay
    // open via the requester-held spare sender, so closure alone is not a
    // signal). Capacity 2 makes these sends non-blocking.
    for (auto& [subject, entry] : state_->pending) {
      entry.fulfill.send(StreamArrival{std::string("data plane shutting down"), std::nullopt});
    }
    state_->pending.clear();
    for (auto& [id, sock] : state_->conn_sockets) sock->shutdown();
  }
  state_->listener->shutdown();
  if (state_->accept_thread.joinable()) state_->accept_thread.join();
  std::unique_lock lock(state_->mutex);
  state_->cv.wait(lock, [&] { return state_->active_conns == 0; });
}

// ---------------------------------------------------------------------------
// StreamSender (worker side)
// ---------------------------------------------------------------------------

struct StreamSender::State {
  std::shared_ptr<Socket> sock;
  std::mutex wmutex;
  std::thread reader;
  std::mutex reader_mutex;
  std::condition_variable reader_cv;
  bool reader_done = false;
  std::atomic<bool> finished{false};
  bool prologue_sent = false;
};

std::optional<StreamSender> StreamSender::connect(pipeline::ContextPtr ctx,
                                                  const ConnectionInfo& info,
                                                  Executor* signal_pool) {
  if (info.transport != "tcp" || info.stream_type != StreamKind::Response) {
    spdlog::error("data plane: unsupported connection info (transport={})", info.transport);
    return std::nullopt;
  }
  if (info.context_id != ctx->id()) {
    spdlog::error("data plane: context mismatch ({} vs {})", info.context_id, ctx->id());
    return std::nullopt;
  }

  auto [host, port] = parse_address(info.address);
  std::optional<Socket> sock;
  for (int attempt = 0; attempt < 3 && !sock; ++attempt) {
    if (attempt > 0) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    sock = Socket::connect(host, port);
  }
  if (!sock) {
    spdlog::warn("data plane: failed to call home to {}", info.address);
    return std::nullopt;
  }

  StreamSender sender;
  sender.state_ = std::make_shared<State>();
  auto s = sender.state_;
  s->sock = std::make_shared<Socket>(std::move(*sock));

  CallHomeHandshake handshake{info.subject, StreamKind::Response};
  if (!s->sock->write_frame(TwoPartMessage::from_header(json(handshake).dump()))) {
    return std::nullopt;
  }

  // Reader: relays requester stop/kill onto the worker-side context; exits on FIN.
  s->reader = std::thread([s, ctx, signal_pool] {
    auto signal = [&](bool kill) {
      auto fire = [ctx, kill] { kill ? ctx->kill() : ctx->stop_generating(); };
      // Continuations woken by stop/kill must not run on this thread: they
      // may call finish(), which joins it.
      if (signal_pool) {
        signal_pool->post(fire);
      } else {
        fire();
      }
    };
    try {
      for (;;) {
        auto frame = s->sock->read_frame();
        if (!frame) break;
        if (!frame->has_header()) continue;
        ControlFrame control = json::parse(frame->header);
        if (control.kind == ControlKind::Stop) {
          signal(false);
        } else if (control.kind == ControlKind::Kill) {
          signal(true);
        }
      }
    } catch (const std::exception& e) {
      spdlog::debug("data plane reader: {}", e.what());
    }
    {
      std::lock_guard lock(s->reader_mutex);
      s->reader_done = true;
    }
    s->reader_cv.notify_all();
  });

  return sender;
}

bool StreamSender::send_prologue(std::optional<std::string> error) {
  Prologue prologue{std::move(error)};
  std::lock_guard lock(state_->wmutex);
  state_->prologue_sent = true;
  return state_->sock->write_frame(TwoPartMessage::from_header(json(prologue).dump()));
}

bool StreamSender::send(std::string bytes) {
  std::lock_guard lock(state_->wmutex);
  return state_->sock->write_frame(TwoPartMessage::from_data(std::move(bytes)));
}

void StreamSender::finish() {
  auto s = state_;
  if (!s || s->finished.exchange(true)) return;
  {
    std::lock_guard lock(s->wmutex);
    ControlFrame sentinel{ControlKind::Sentinel};
    s->sock->write_frame(TwoPartMessage::from_header(json(sentinel).dump()));
  }
  // Wait for the requester to close the socket; force-unblock the reader if
  // it takes too long.
  {
    std::unique_lock lock(s->reader_mutex);
    if (!s->reader_cv.wait_for(lock, std::chrono::seconds(10), [&] { return s->reader_done; })) {
      spdlog::debug("data plane: requester did not close stream in time");
      s->sock->shutdown();
    }
  }
  if (s->reader.joinable()) s->reader.join();
  s->sock->close();
}

StreamSender::~StreamSender() {
  if (state_) finish();
}

}  // namespace dynamo::transports
