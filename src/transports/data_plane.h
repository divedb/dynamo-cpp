// SPDX-License-Identifier: Apache-2.0
//
// Data plane: the streaming return path. The requester runs a DataPlaneServer
// and registers a pending response stream per request; the worker "calls
// home" with a StreamSender, sends the prologue, then data frames, and
// finally a sentinel. Stop/kill control frames flow requester→worker on the
// same socket.

#pragma once

#include <memory>
#include <optional>
#include <string>

#include "pipeline/engine.h"
#include "runtime/coro/channel.h"
#include "runtime/runtime.h"
#include "transports/socket.h"
#include "transports/wire.h"

namespace dynamo::transports {

/// Requester-side view of an established response stream.
struct IncomingStream {
  coro::Receiver<std::string> data;
};

/// Outcome of the call-home handshake for a registered stream.
struct StreamArrival {
  std::optional<std::string> error;      // worker-side generate() failure
  std::optional<IncomingStream> stream;  // present iff error is empty
};

/// Handle returned by register_response_stream(): the ConnectionInfo to ship
/// to the worker, and a one-shot receiver resolving when the worker connects
/// (closed without a value if the server shuts down first). `arrival_tx` is
/// a spare sender for injecting a synthetic arrival (e.g. a timeout marker).
struct RegisteredStream {
  ConnectionInfo info;
  coro::Receiver<StreamArrival> arrival;
  coro::Sender<StreamArrival> arrival_tx;
};

class DataPlaneServer {
 public:
  /// Binds and starts the accept loop. Throws on bind failure.
  static std::shared_ptr<DataPlaneServer> start(Runtime runtime,
                                                const std::string& host = "127.0.0.1",
                                                uint16_t port = 0);
  ~DataPlaneServer();

  std::string address() const;

  /// Registers a pending response stream tied to `ctx`. The context's
  /// stop/kill signals are forwarded to the worker as control frames once
  /// the stream is connected. `recv_buffer_count` bounds the number of
  /// response frames buffered before backpressuring the worker.
  RegisteredStream register_response_stream(pipeline::ContextPtr ctx,
                                            size_t recv_buffer_count = 64);

  /// Withdraws a pending registration (e.g. after an arrival timeout); a
  /// later call-home for the subject is then rejected.
  void deregister(const std::string& subject);

  void stop();

  struct State;  // opaque; public so implementation helpers can name it

 private:
  DataPlaneServer() = default;
  std::shared_ptr<State> state_;
};

/// Worker-side sender for one response stream (movable handle).
class StreamSender {
 public:
  /// Connects back to the requester and performs the handshake. Returns
  /// nullopt if the requester is unreachable. `ctx` receives stop/kill
  /// signals relayed from the requester; if `signal_pool` is given they are
  /// posted there instead of running on the internal reader thread (required
  /// when stop/kill continuations may call back into this sender).
  static std::optional<StreamSender> connect(pipeline::ContextPtr ctx, const ConnectionInfo& info,
                                             Executor* signal_pool = nullptr);

  /// Reports the generate() outcome. Must be called exactly once, first.
  bool send_prologue(std::optional<std::string> error);

  /// Sends one data frame. Returns false if the stream is gone.
  bool send(std::string bytes);

  /// Sends the sentinel and waits briefly for the server's FIN. Idempotent;
  /// also invoked by the destructor.
  void finish();

  ~StreamSender();
  StreamSender(StreamSender&&) noexcept = default;
  StreamSender& operator=(StreamSender&&) noexcept = default;

 private:
  StreamSender() = default;
  struct State;
  std::shared_ptr<State> state_;
};

}  // namespace dynamo::transports
