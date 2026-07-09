// SPDX-License-Identifier: Apache-2.0

#include "llm/http/http_server.h"

#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include "transports/socket.h"

namespace dynamo::llm::http {

namespace {

constexpr size_t kMaxHeaderBytes = 64 * 1024;
constexpr size_t kMaxBodyBytes = 32 * 1024 * 1024;

std::string lowercase(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

std::string_view trim_sv(std::string_view s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) {
    s.remove_suffix(1);
  }
  return s;
}

const char* status_text(int status) {
  switch (status) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 500: return "Internal Server Error";
    case 503: return "Service Unavailable";
    default: return "Unknown";
  }
}

/// Buffered reader over a blocking socket fd.
class LineReader {
 public:
  explicit LineReader(int fd) : fd_(fd) {}

  /// Reads a CRLF/LF-terminated line (terminator stripped); nullopt on EOF.
  std::optional<std::string> read_line(size_t max_bytes) {
    std::string line;
    while (line.size() < max_bytes) {
      if (pos_ == len_ && !fill()) return std::nullopt;
      char c = buf_[pos_++];
      if (c == '\n') {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        return line;
      }
      line.push_back(c);
    }
    return std::nullopt;  // header line too long
  }

  bool read_exact(std::string& out, size_t n) {
    out.clear();
    out.reserve(n);
    while (out.size() < n) {
      if (pos_ == len_ && !fill()) return false;
      size_t take = std::min(n - out.size(), len_ - pos_);
      out.append(buf_ + pos_, take);
      pos_ += take;
    }
    return true;
  }

 private:
  bool fill() {
    ssize_t r;
    do {
      r = ::recv(fd_, buf_, sizeof(buf_), 0);
    } while (r < 0 && errno == EINTR);
    if (r <= 0) return false;
    pos_ = 0;
    len_ = static_cast<size_t>(r);
    return true;
  }

  int fd_;
  char buf_[16 * 1024];
  size_t pos_ = 0;
  size_t len_ = 0;
};

bool write_all_fd(int fd, std::string_view data) {
  size_t sent = 0;
  while (sent < data.size()) {
    ssize_t r = ::send(fd, data.data() + sent, data.size() - sent,
#ifdef MSG_NOSIGNAL
                       MSG_NOSIGNAL
#else
                       0
#endif
    );
    if (r < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    sent += static_cast<size_t>(r);
  }
  return true;
}

}  // namespace

struct HttpServer::Impl {
  std::unordered_map<std::string, HttpHandler> routes;  // "METHOD path"
  std::optional<transports::Listener> listener;
  std::thread accept_thread;
  std::mutex mutex;
  std::vector<std::thread> connection_threads;
  std::unordered_map<uint64_t, int> live_fds;
  uint64_t next_conn_id = 0;
  std::atomic<bool> stopping{false};
  bool started = false;

  void serve_connection(uint64_t conn_id, transports::Socket socket);
  bool handle_one(LineReader& reader, int fd);
  static void write_response(int fd, const HttpResponse& response, bool keep_alive);
};

HttpServer::HttpServer() : impl_(std::make_unique<Impl>()) {}

HttpServer::~HttpServer() { stop(); }

void HttpServer::handle(std::string method, std::string path, HttpHandler handler) {
  impl_->routes[method + " " + path] = std::move(handler);
}

void HttpServer::start(const std::string& host, uint16_t port) {
  auto listener = transports::Listener::bind(host, port);
  if (!listener) {
    throw std::runtime_error(fmt::format("http server failed to bind {}:{}", host, port));
  }
  impl_->listener = std::move(*listener);
  impl_->started = true;
  Impl* impl = impl_.get();
  impl_->accept_thread = std::thread([impl] {
    while (auto socket = impl->listener->accept()) {
      if (impl->stopping.load()) break;
      std::lock_guard lock(impl->mutex);
      uint64_t id = impl->next_conn_id++;
      impl->live_fds[id] = socket->fd();
      impl->connection_threads.emplace_back(
          [impl, id, s = std::move(*socket)]() mutable { impl->serve_connection(id, std::move(s)); });
    }
  });
}

uint16_t HttpServer::port() const { return impl_->listener ? impl_->listener->port() : 0; }

std::string HttpServer::address() const {
  return impl_->listener ? impl_->listener->address() : std::string{};
}

void HttpServer::stop() {
  if (!impl_ || !impl_->started) return;
  bool expected = false;
  if (!impl_->stopping.compare_exchange_strong(expected, true)) {
    // Another stop() already ran; just make sure threads are joined.
  }
  impl_->listener->shutdown();
  if (impl_->accept_thread.joinable()) impl_->accept_thread.join();
  std::vector<std::thread> threads;
  {
    std::lock_guard lock(impl_->mutex);
    for (auto& [id, fd] : impl_->live_fds) ::shutdown(fd, SHUT_RDWR);
    threads.swap(impl_->connection_threads);
  }
  for (auto& t : threads) {
    if (t.joinable()) t.join();
  }
}

void HttpServer::Impl::serve_connection(uint64_t conn_id, transports::Socket socket) {
  LineReader reader(socket.fd());
  while (!stopping.load()) {
    if (!handle_one(reader, socket.fd())) break;
  }
  std::lock_guard lock(mutex);
  live_fds.erase(conn_id);
}

/// Serves one request; returns false when the connection should close.
bool HttpServer::Impl::handle_one(LineReader& reader, int fd) {
  auto request_line = reader.read_line(kMaxHeaderBytes);
  if (!request_line || request_line->empty()) return false;

  HttpRequest request;
  {
    size_t sp1 = request_line->find(' ');
    size_t sp2 = request_line->rfind(' ');
    if (sp1 == std::string::npos || sp2 == sp1) {
      write_response(fd, HttpResponse{400, "application/json", {}, R"({"error":"bad request"})",
                                      nullptr},
                     false);
      return false;
    }
    request.method = request_line->substr(0, sp1);
    request.target = request_line->substr(sp1 + 1, sp2 - sp1 - 1);
    request.path = request.target.substr(0, request.target.find('?'));
  }

  size_t header_bytes = 0;
  while (true) {
    auto line = reader.read_line(kMaxHeaderBytes);
    if (!line) return false;
    if (line->empty()) break;
    header_bytes += line->size();
    if (header_bytes > kMaxHeaderBytes) return false;
    size_t colon = line->find(':');
    if (colon == std::string::npos) continue;
    std::string name = lowercase(line->substr(0, colon));
    request.headers[name] = std::string(trim_sv(std::string_view(*line).substr(colon + 1)));
  }

  bool keep_alive = true;
  if (auto it = request.headers.find("connection"); it != request.headers.end()) {
    keep_alive = lowercase(it->second) != "close";
  }

  if (auto it = request.headers.find("content-length"); it != request.headers.end()) {
    size_t length = 0;
    try {
      length = static_cast<size_t>(std::stoull(it->second));
    } catch (const std::exception&) {
      return false;
    }
    if (length > kMaxBodyBytes) {
      write_response(fd, HttpResponse{400, "application/json", {},
                                      R"({"error":"body too large"})", nullptr},
                     false);
      return false;
    }
    if (!reader.read_exact(request.body, length)) return false;
  }

  auto route = routes.find(request.method + " " + request.path);
  if (route == routes.end()) {
    bool path_exists = std::any_of(routes.begin(), routes.end(), [&](const auto& r) {
      return r.first.substr(r.first.find(' ') + 1) == request.path;
    });
    int status = path_exists ? 405 : 404;
    write_response(fd, HttpResponse{status, "application/json", {},
                                    fmt::format(R"({{"error":"{}"}})", status_text(status)),
                                    nullptr},
                   keep_alive);
    return keep_alive;
  }

  HttpResponse response;
  try {
    response = route->second(request);
  } catch (const std::exception& e) {
    spdlog::error("http handler for {} {} threw: {}", request.method, request.path, e.what());
    response = HttpResponse{500, "application/json", {},
                            R"({"error":"internal server error"})", nullptr};
  }

  write_response(fd, response, keep_alive);
  return keep_alive;
}

void HttpServer::Impl::write_response(int fd, const HttpResponse& response, bool keep_alive) {
  std::string head = fmt::format("HTTP/1.1 {} {}\r\n", response.status,
                                 status_text(response.status));
  head += fmt::format("Content-Type: {}\r\n", response.content_type);
  for (const auto& [name, value] : response.headers) {
    head += fmt::format("{}: {}\r\n", name, value);
  }
  head += keep_alive ? "Connection: keep-alive\r\n" : "Connection: close\r\n";

  if (response.stream) {
    head += "Transfer-Encoding: chunked\r\n\r\n";
    if (!write_all_fd(fd, head)) return;
    StreamSink sink = [fd](std::string_view chunk) {
      if (chunk.empty()) return true;
      std::string framed = fmt::format("{:x}\r\n", chunk.size());
      framed.append(chunk);
      framed += "\r\n";
      return write_all_fd(fd, framed);
    };
    response.stream(sink);
    write_all_fd(fd, "0\r\n\r\n");
    return;
  }

  head += fmt::format("Content-Length: {}\r\n\r\n", response.body.size());
  std::string full = head + response.body;
  write_all_fd(fd, full);
}

}  // namespace dynamo::llm::http
