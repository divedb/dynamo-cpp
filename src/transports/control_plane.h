// SPDX-License-Identifier: Apache-2.0
//
// Control plane: request dispatch. Every worker hosts a ControlPlaneServer;
// its address is advertised through discovery as part of each endpoint
// instance's info. Clients dispatch a request straight to the chosen
// instance's server and receive an ack; the response never flows here (it
// streams back over the data plane).
//
// This replaces Dynamo's NATS subject-addressed push (see docs/architecture.md).

#pragma once

#include <functional>
#include <memory>
#include <string>

#include "runtime/coro/task.h"
#include "runtime/runtime.h"
#include "transports/wire.h"

namespace dynamo::transports {

/// Handles one pushed request. Runs detached on the worker's primary pool.
class PushHandler {
 public:
  virtual ~PushHandler() = default;
  virtual coro::Task<void> handle(RequestControlMessage control, std::string payload) = 0;
};

/// Handles a synchronous query (e.g. stats scrape); returns the reply payload.
using QueryHandler = std::function<std::string(const std::string& payload)>;

class ControlPlaneServer {
 public:
  static std::shared_ptr<ControlPlaneServer> start(Runtime runtime,
                                                   const std::string& host = "127.0.0.1",
                                                   uint16_t port = 0);
  ~ControlPlaneServer();

  std::string address() const;

  void register_handler(const std::string& subject, std::shared_ptr<PushHandler> handler);
  void unregister_handler(const std::string& subject);

  void register_query(const std::string& subject, QueryHandler handler);
  void unregister_query(const std::string& subject);

  void stop();

  struct State;  // opaque; public so implementation helpers can name it

 private:
  ControlPlaneServer() = default;
  std::shared_ptr<State> state_;
};

/// Sends one request frame to `address` and awaits the ack (blocking I/O on
/// the calling thread — call from a pool-thread coroutine). Connections are
/// pooled per target and reused across calls.
/// Throws std::runtime_error if the instance is unreachable, rejects the
/// request, or does not ack within `ack_timeout`.
void dispatch_request(const std::string& address, const DispatchHeader& header,
                      std::string payload,
                      std::chrono::milliseconds ack_timeout = std::chrono::seconds(5));

/// Sends a query and returns the reply payload. Throws on failure or if the
/// reply does not arrive within `timeout`.
std::string dispatch_query(const std::string& address, const std::string& subject,
                           std::string payload,
                           std::chrono::milliseconds timeout = std::chrono::seconds(5));

}  // namespace dynamo::transports
