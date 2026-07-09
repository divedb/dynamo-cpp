// SPDX-License-Identifier: Apache-2.0
//
// Minimal HTTP/1.1 server on the transports Socket/Listener stack — just
// enough for the OpenAI frontend (Rust uses axum): exact-path routing,
// keep-alive, Content-Length request bodies, fixed or chunked (SSE)
// streaming responses. Thread-per-connection, matching the blocking-I/O
// model of the rest of the transports layer.

#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>

namespace dynamo::llm::http {

struct HttpRequest {
  std::string method;
  std::string target;  // as received (path + query)
  std::string path;    // target without the query string
  std::map<std::string, std::string> headers;  // names lowercased
  std::string body;
};

/// Writes one chunk of a streaming response body; returns false once the
/// client has disconnected (the handler should stop producing).
using StreamSink = std::function<bool(std::string_view)>;

struct HttpResponse {
  int status = 200;
  std::string content_type = "application/json";
  std::map<std::string, std::string> headers;  // extra headers
  std::string body;

  /// When set, the response streams with Transfer-Encoding: chunked and
  /// `body` is ignored. The callback runs after headers are sent.
  std::function<void(const StreamSink&)> stream;
};

using HttpHandler = std::function<HttpResponse(const HttpRequest&)>;

/// Exact-path HTTP server. Register handlers, then start().
class HttpServer {
 public:
  HttpServer();
  ~HttpServer();

  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;

  /// Registers a handler for an exact method+path. Not thread-safe against
  /// a running server; register everything before start().
  void handle(std::string method, std::string path, HttpHandler handler);

  /// Binds and starts accepting (port 0 = ephemeral). Throws on failure.
  void start(const std::string& host, uint16_t port);

  uint16_t port() const;
  std::string address() const;

  /// Stops accepting, disconnects clients, joins all threads. Idempotent.
  void stop();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace dynamo::llm::http
