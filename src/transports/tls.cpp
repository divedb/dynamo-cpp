// SPDX-License-Identifier: Apache-2.0

#include "transports/tls.h"

#include <fcntl.h>
#include <poll.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <stdexcept>

#include <openssl/err.h>
#include <openssl/ssl.h>

namespace dynamo::transports {

namespace {

std::string last_openssl_error(const std::string& what) {
  char buf[256] = "unknown error";
  if (unsigned long err = ERR_peek_last_error()) ERR_error_string_n(err, buf, sizeof(buf));
  ERR_clear_error();
  return what + ": " + buf;
}

struct TlsGlobals {
  std::mutex mutex;
  std::optional<tls::Options> options;
  bool env_loaded = false;
  SSL_CTX* client_ctx = nullptr;
  SSL_CTX* server_ctx = nullptr;

  void load_env_locked() {
    if (env_loaded) return;
    env_loaded = true;
    const char* cert = std::getenv("DYN_TLS_CERT");
    const char* key = std::getenv("DYN_TLS_KEY");
    const char* ca = std::getenv("DYN_TLS_CA");
    if (!cert && !key && !ca) return;
    if (!cert || !key || !ca) {
      throw std::runtime_error(
          "TLS misconfigured: DYN_TLS_CERT, DYN_TLS_KEY, and DYN_TLS_CA must all be set");
    }
    options = tls::Options{cert, key, ca};
  }

  void drop_ctxs_locked() {
    if (client_ctx) SSL_CTX_free(client_ctx);
    if (server_ctx) SSL_CTX_free(server_ctx);
    client_ctx = server_ctx = nullptr;
  }

  SSL_CTX* make_ctx_locked(bool server) {
    SSL_CTX* ctx = SSL_CTX_new(server ? TLS_server_method() : TLS_client_method());
    if (!ctx) throw std::runtime_error(last_openssl_error("TLS: SSL_CTX_new failed"));
    try {
      const auto& o = *options;
      if (SSL_CTX_set_min_proto_version(ctx, TLS1_3_VERSION) != 1) {
        throw std::runtime_error(last_openssl_error("TLS: cannot require TLS 1.3"));
      }
      if (SSL_CTX_use_certificate_chain_file(ctx, o.cert_path.c_str()) != 1) {
        throw std::runtime_error(last_openssl_error("TLS: cannot load cert " + o.cert_path));
      }
      if (SSL_CTX_use_PrivateKey_file(ctx, o.key_path.c_str(), SSL_FILETYPE_PEM) != 1) {
        throw std::runtime_error(last_openssl_error("TLS: cannot load key " + o.key_path));
      }
      if (SSL_CTX_check_private_key(ctx) != 1) {
        throw std::runtime_error(last_openssl_error("TLS: key does not match cert"));
      }
      if (SSL_CTX_load_verify_locations(ctx, o.ca_path.c_str(), nullptr) != 1) {
        throw std::runtime_error(last_openssl_error("TLS: cannot load CA " + o.ca_path));
      }
      // mTLS both ways; trust is pinned to the private CA, no hostname check.
      SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT, nullptr);
      SSL_CTX_set_mode(ctx, SSL_MODE_AUTO_RETRY);
    } catch (...) {
      SSL_CTX_free(ctx);
      throw;
    }
    return ctx;
  }

  SSL_CTX* ctx(bool server) {
    std::lock_guard lock(mutex);
    load_env_locked();
    if (!options) throw std::logic_error("TLS session requested but TLS is not configured");
    SSL_CTX*& slot = server ? server_ctx : client_ctx;
    if (!slot) slot = make_ctx_locked(server);
    return slot;
  }
};

TlsGlobals& globals() {
  static TlsGlobals instance;
  return instance;
}

}  // namespace

namespace tls {

void configure(std::optional<Options> options) {
  auto& g = globals();
  std::lock_guard lock(g.mutex);
  g.env_loaded = true;  // explicit config wins over env
  g.options = std::move(options);
  g.drop_ctxs_locked();
}

bool enabled() {
  auto& g = globals();
  std::lock_guard lock(g.mutex);
  g.load_env_locked();
  return g.options.has_value();
}

}  // namespace tls

struct TlsSession::Impl {
  SSL* ssl = nullptr;
  int fd = -1;
  std::mutex mutex;
  std::atomic<int64_t> read_timeout_ms{0};

  ~Impl() {
    if (ssl) {
      SSL_shutdown(ssl);  // best-effort close_notify; non-blocking, one shot
      SSL_free(ssl);
    }
  }

  /// Waits for fd readiness; also returns true on HUP/ERR so the next SSL
  /// call observes and reports the condition. False = timeout/poll failure.
  bool wait(short events, bool timed) {
    int64_t ms = timed ? read_timeout_ms.load(std::memory_order_relaxed) : 0;
    struct pollfd p{fd, events, 0};
    int r = ::poll(&p, 1, ms > 0 ? static_cast<int>(ms) : -1);
    return r > 0;
  }

  size_t read_some(char* out, size_t n) {
    for (;;) {
      int r;
      int err = SSL_ERROR_NONE;
      {
        std::lock_guard lock(mutex);
        ERR_clear_error();
        r = SSL_read(ssl, out, static_cast<int>(std::min(n, size_t{1} << 30)));
        if (r <= 0) err = SSL_get_error(ssl, r);
      }
      if (r > 0) return static_cast<size_t>(r);
      if (err == SSL_ERROR_WANT_READ) {
        if (!wait(POLLIN, /*timed=*/true)) return 0;
      } else if (err == SSL_ERROR_WANT_WRITE) {
        if (!wait(POLLOUT, /*timed=*/true)) return 0;
      } else {
        return 0;  // clean EOF, abrupt close, or protocol error — caller treats alike
      }
    }
  }

  bool write_all(std::string_view buf) {
    size_t sent = 0;
    while (sent < buf.size()) {
      int r;
      int err = SSL_ERROR_NONE;
      {
        std::lock_guard lock(mutex);
        ERR_clear_error();
        r = SSL_write(ssl, buf.data() + sent,
                      static_cast<int>(std::min(buf.size() - sent, size_t{1} << 30)));
        if (r <= 0) err = SSL_get_error(ssl, r);
      }
      if (r > 0) {
        sent += static_cast<size_t>(r);
      } else if (err == SSL_ERROR_WANT_WRITE) {
        if (!wait(POLLOUT, /*timed=*/false)) return false;
      } else if (err == SSL_ERROR_WANT_READ) {
        if (!wait(POLLIN, /*timed=*/false)) return false;
      } else {
        return false;
      }
    }
    return true;
  }
};

TlsSession::TlsSession() : impl_(std::make_unique<Impl>()) {}
TlsSession::~TlsSession() = default;

std::unique_ptr<TlsSession> TlsSession::make(int fd, bool server) {
  SSL_CTX* ctx = globals().ctx(server);
  SSL* ssl = SSL_new(ctx);
  if (!ssl) throw std::runtime_error(last_openssl_error("TLS: SSL_new failed"));
  if (SSL_set_fd(ssl, fd) != 1) {
    SSL_free(ssl);
    throw std::runtime_error(last_openssl_error("TLS: SSL_set_fd failed"));
  }
  if (server) {
    SSL_set_accept_state(ssl);
  } else {
    SSL_set_connect_state(ssl);
  }
  ::fcntl(fd, F_SETFL, ::fcntl(fd, F_GETFL, 0) | O_NONBLOCK);
  auto session = std::unique_ptr<TlsSession>(new TlsSession());
  session->impl_->ssl = ssl;
  session->impl_->fd = fd;
  return session;
}

std::unique_ptr<TlsSession> TlsSession::make_client(int fd) { return make(fd, /*server=*/false); }

std::unique_ptr<TlsSession> TlsSession::make_server(int fd) { return make(fd, /*server=*/true); }

size_t TlsSession::read_some(char* out, size_t n) { return impl_->read_some(out, n); }

bool TlsSession::read_exact(char* out, size_t n) {
  size_t got = 0;
  while (got < n) {
    size_t r = impl_->read_some(out + got, n - got);
    if (r == 0) return false;
    got += r;
  }
  return true;
}

bool TlsSession::write_all(std::string_view buf) { return impl_->write_all(buf); }

void TlsSession::set_read_timeout(std::chrono::milliseconds timeout) {
  impl_->read_timeout_ms.store(timeout.count(), std::memory_order_relaxed);
}

}  // namespace dynamo::transports
