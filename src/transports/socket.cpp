// SPDX-License-Identifier: Apache-2.0

#include "transports/socket.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>

#include <spdlog/spdlog.h>

namespace dynamo::transports {

Socket& Socket::operator=(Socket&& other) noexcept {
  if (this != &other) {
    close();
    fd_ = std::exchange(other.fd_, -1);
  }
  return *this;
}

void Socket::shutdown() {
  if (fd_ >= 0) ::shutdown(fd_, SHUT_RDWR);
}

void Socket::close() {
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

bool Socket::write_all(std::string_view buf) {
  size_t sent = 0;
  while (sent < buf.size()) {
    ssize_t n = ::send(fd_, buf.data() + sent, buf.size() - sent,
#ifdef MSG_NOSIGNAL
                       MSG_NOSIGNAL
#else
                       0
#endif
    );
    if (n <= 0) {
      if (n < 0 && errno == EINTR) continue;
      return false;
    }
    sent += static_cast<size_t>(n);
  }
  return true;
}

bool Socket::read_exact(char* out, size_t n) {
  size_t got = 0;
  while (got < n) {
    ssize_t r = ::recv(fd_, out + got, n - got, 0);
    if (r == 0) return false;  // EOF
    if (r < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    got += static_cast<size_t>(r);
  }
  return true;
}

void Socket::set_recv_timeout(std::chrono::milliseconds timeout) {
  struct timeval tv;
  tv.tv_sec = static_cast<time_t>(timeout.count() / 1000);
  tv.tv_usec = static_cast<suseconds_t>((timeout.count() % 1000) * 1000);
  ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
}

void Socket::drain(std::chrono::milliseconds timeout) {
  set_recv_timeout(timeout);
  char buf[1024];
  for (;;) {
    ssize_t r = ::recv(fd_, buf, sizeof(buf), 0);
    if (r <= 0) return;  // EOF, error, or timeout
  }
}

namespace {

std::atomic<size_t> g_max_frame_bytes = [] {
  size_t bytes = 256ull * 1024 * 1024;
  if (const char* mb = std::getenv("DYN_MAX_FRAME_MB")) {
    if (auto parsed = std::strtoull(mb, nullptr, 10); parsed > 0) {
      bytes = static_cast<size_t>(parsed) * 1024 * 1024;
    }
  }
  return bytes;
}();

}  // namespace

size_t Socket::max_frame_bytes() { return g_max_frame_bytes.load(std::memory_order_relaxed); }
void Socket::set_max_frame_bytes(size_t bytes) {
  g_max_frame_bytes.store(bytes, std::memory_order_relaxed);
}

std::optional<TwoPartMessage> Socket::read_frame() {
  char prelude[TwoPartCodec::kPreludeSize];
  if (!read_exact(prelude, sizeof(prelude))) return std::nullopt;

  auto get_u64 = [](const char* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v = (v << 8) | static_cast<unsigned char>(p[i]);
    return v;
  };
  uint64_t header_len = get_u64(prelude);
  uint64_t body_len = get_u64(prelude + 8);
  if (header_len + body_len > max_frame_bytes()) throw std::runtime_error("frame too large");

  std::string rest;
  rest.resize(header_len + body_len);
  if (!rest.empty() && !read_exact(rest.data(), rest.size())) {
    throw std::runtime_error("connection closed mid-frame");
  }

  std::string full;
  full.reserve(TwoPartCodec::kPreludeSize + rest.size());
  full.append(prelude, sizeof(prelude));
  full.append(rest);
  return TwoPartCodec().decode(full);
}

bool Socket::write_frame(const TwoPartMessage& msg) {
  return write_all(TwoPartCodec().encode(msg));
}

std::optional<Socket> Socket::connect(const std::string& host, uint16_t port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return std::nullopt;

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
    ::close(fd);
    return std::nullopt;
  }
  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    ::close(fd);
    return std::nullopt;
  }
  int one = 1;
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
#ifdef SO_NOSIGPIPE
  ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
  return Socket(fd);
}

Listener::Fd::~Fd() {
  if (fd >= 0) ::close(fd);
  if (wake_rd >= 0) ::close(wake_rd);
  if (wake_wr >= 0) ::close(wake_wr);
}

std::optional<Listener> Listener::bind(const std::string& host, uint16_t port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return std::nullopt;
  int one = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
    ::close(fd);
    return std::nullopt;
  }
  if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 || ::listen(fd, 64) != 0) {
    ::close(fd);
    return std::nullopt;
  }

  int wake[2];
  if (::pipe(wake) != 0) {
    ::close(fd);
    return std::nullopt;
  }

  sockaddr_in bound{};
  socklen_t len = sizeof(bound);
  ::getsockname(fd, reinterpret_cast<sockaddr*>(&bound), &len);
  return Listener(fd, wake[0], wake[1], host, ntohs(bound.sin_port));
}

std::optional<Socket> Listener::accept() {
  for (;;) {
    struct pollfd fds[2] = {{fd_->fd, POLLIN, 0}, {fd_->wake_rd, POLLIN, 0}};
    int r = ::poll(fds, 2, -1);
    if (r < 0) {
      if (errno == EINTR) continue;
      return std::nullopt;
    }
    if (fds[1].revents & (POLLIN | POLLHUP)) return std::nullopt;  // shutdown requested
    if (!(fds[0].revents & POLLIN)) continue;

    int conn = ::accept(fd_->fd, nullptr, nullptr);
    if (conn >= 0) {
      int one = 1;
      ::setsockopt(conn, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
#ifdef SO_NOSIGPIPE
      ::setsockopt(conn, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
      return Socket(conn);
    }
    if (errno == EINTR || errno == ECONNABORTED) continue;
    return std::nullopt;
  }
}

void Listener::shutdown() {
  char byte = 1;
  [[maybe_unused]] ssize_t n = ::write(fd_->wake_wr, &byte, 1);
}

std::pair<std::string, uint16_t> parse_address(const std::string& address) {
  auto pos = address.rfind(':');
  if (pos == std::string::npos || pos + 1 >= address.size()) {
    throw std::invalid_argument("malformed address: " + address);
  }
  return {address.substr(0, pos), static_cast<uint16_t>(std::stoul(address.substr(pos + 1)))};
}

}  // namespace dynamo::transports
