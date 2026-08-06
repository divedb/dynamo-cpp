// SPDX-License-Identifier: Apache-2.0
//
// Minimal RAII TCP primitives with two-part-message framing. I/O is blocking;
// higher layers run it on dedicated threads and bridge to coroutines with
// channels. shutdown() may be called from any thread to unblock reads.
//
// Optional mTLS (see tls.h): pass with_tls to connect()/bind() — the internal
// planes pass tls::enabled(). TLS sockets keep the same blocking read/write
// semantics, including full-duplex use from two threads.

#pragma once

#include <chrono>
#include <memory>
#include <optional>
#include <string>

#include "transports/codec.h"

namespace dynamo::transports {

class TlsSession;

class Socket {
 public:
  Socket();
  explicit Socket(int fd);
  Socket(Socket&& other) noexcept;
  Socket& operator=(Socket&& other) noexcept;
  Socket(const Socket&) = delete;
  Socket& operator=(const Socket&) = delete;
  ~Socket();

  bool valid() const { return fd_ >= 0; }
  int fd() const { return fd_; }

  /// Half-closes both directions, unblocking any reader; keeps the fd.
  void shutdown();
  void close();

  /// Blocking write of the whole buffer. Returns false on error/peer close.
  bool write_all(std::string_view buf);

  /// Blocking read of exactly n bytes. Returns false on EOF/error.
  bool read_exact(char* out, size_t n);

  /// Bounds subsequent blocking reads (SO_RCVTIMEO); reads then fail instead
  /// of blocking forever. Zero restores unbounded reads.
  void set_recv_timeout(std::chrono::milliseconds timeout);

  /// Reads and discards inbound bytes until EOF, error, or per-read timeout.
  void drain(std::chrono::milliseconds timeout);

  /// Framed I/O. read_frame returns nullopt on clean EOF and throws on
  /// malformed frames (checksum mismatch, oversize, mid-frame close);
  /// write_frame returns false on transport errors.
  std::optional<TwoPartMessage> read_frame();
  bool write_frame(const TwoPartMessage& msg);

  /// with_tls wraps the connection in a client-side TLS session (lazy
  /// handshake). Throws on TLS misconfiguration.
  static std::optional<Socket> connect(const std::string& host, uint16_t port,
                                       bool with_tls = false);
  bool is_tls() const { return tls_ != nullptr; }

  /// Process-wide inbound frame-size cap (default 256 MiB, or
  /// DYN_MAX_FRAME_MB at startup). Oversized frames fail the connection.
  static size_t max_frame_bytes();
  static void set_max_frame_bytes(size_t bytes);

 private:
  friend class Listener;
  int fd_ = -1;
  std::unique_ptr<TlsSession> tls_;
};

/// Listening socket bound to `host:port` (port 0 = ephemeral).
class Listener {
 public:
  /// with_tls wraps every accepted connection in a server-side TLS session
  /// (lazy handshake, so a stalling peer never blocks the accept loop).
  static std::optional<Listener> bind(const std::string& host, uint16_t port,
                                      bool with_tls = false);

  uint16_t port() const { return port_; }
  const std::string& host() const { return host_; }
  std::string address() const { return host_ + ":" + std::to_string(port_); }

  /// Blocking accept. Returns nullopt once shutdown() has been called.
  std::optional<Socket> accept();

  /// Unblocks accept() from another thread (self-pipe wakeup: shutdown() on
  /// a listening socket does not wake accept() on macOS).
  void shutdown();

 private:
  Listener(int fd, int wake_rd, int wake_wr, std::string host, uint16_t port, bool with_tls)
      : fd_(std::make_shared<Fd>(fd, wake_rd, wake_wr)),
        host_(std::move(host)),
        port_(port),
        with_tls_(with_tls) {}

  struct Fd {
    Fd(int fd, int wake_rd, int wake_wr) : fd(fd), wake_rd(wake_rd), wake_wr(wake_wr) {}
    ~Fd();
    int fd;
    int wake_rd;
    int wake_wr;
  };
  std::shared_ptr<Fd> fd_;
  std::string host_;
  uint16_t port_ = 0;
  bool with_tls_ = false;
};

/// Splits "host:port". Throws std::invalid_argument on malformed input.
std::pair<std::string, uint16_t> parse_address(const std::string& address);

}  // namespace dynamo::transports
