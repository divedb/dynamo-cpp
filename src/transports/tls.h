// SPDX-License-Identifier: Apache-2.0
//
// Optional mutual-TLS layer for the internal planes (discoveryd, control
// plane, data plane). Configured process-wide from DYN_TLS_CERT / DYN_TLS_KEY
// / DYN_TLS_CA (all three required together) or programmatically via
// tls::configure(). Every node presents the same kind of identity and
// verifies the peer against the pinned CA in both directions (mTLS); hostname
// verification is deliberately skipped — trust is anchored in the private CA,
// not in names. TLS 1.3 only.
//
// The HTTP frontend is NOT covered (it does raw-fd I/O and speaks to external
// OpenAI clients); terminate TLS for it in a proxy.
//
// Built only when OpenSSL is available (DYNAMO_HAVE_TLS); otherwise a stub is
// linked that fails loudly if TLS is requested.

#pragma once

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace dynamo::transports {

namespace tls {

struct Options {
  std::string cert_path;  // PEM certificate (chain) presented to peers
  std::string key_path;   // PEM private key for cert_path
  std::string ca_path;    // PEM trust anchor peers must chain to
};

/// Overrides the environment configuration (nullopt disables TLS). Invalidates
/// cached SSL contexts; sockets already wrapped keep their session.
void configure(std::optional<Options> options);

/// True when TLS is configured. Throws when DYN_TLS_* is set but the build has
/// no TLS support, or when only some of the three variables are set.
bool enabled();

}  // namespace tls

/// One TLS session over an already-connected fd (not owned). The fd is
/// switched to non-blocking; the handshake runs lazily, driven by the first
/// read/write. Safe for one concurrent reader plus one concurrent writer
/// (each SSL call runs under an internal mutex held only for the
/// non-blocking call itself; waiting happens outside the lock).
class TlsSession {
 public:
  static std::unique_ptr<TlsSession> make_client(int fd);
  static std::unique_ptr<TlsSession> make_server(int fd);
  ~TlsSession();

  TlsSession(const TlsSession&) = delete;
  TlsSession& operator=(const TlsSession&) = delete;

  /// Reads up to n bytes; returns bytes read, 0 on EOF/timeout/error.
  size_t read_some(char* out, size_t n);
  bool read_exact(char* out, size_t n);
  bool write_all(std::string_view buf);

  /// Bounds each blocking wait in subsequent reads (zero = unbounded),
  /// mirroring SO_RCVTIMEO on plaintext sockets.
  void set_read_timeout(std::chrono::milliseconds timeout);

 private:
  TlsSession();
  static std::unique_ptr<TlsSession> make(int fd, bool server);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace dynamo::transports
