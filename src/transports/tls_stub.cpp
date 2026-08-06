// SPDX-License-Identifier: Apache-2.0
//
// Linked instead of tls.cpp when OpenSSL is not available. TLS stays off, and
// any attempt to enable it fails loudly rather than silently running plaintext.

#include "transports/tls.h"

#include <cstdlib>
#include <stdexcept>

namespace dynamo::transports {

namespace tls {

void configure(std::optional<Options> options) {
  if (options) {
    throw std::runtime_error("dynamo was built without TLS support (OpenSSL not found)");
  }
}

bool enabled() {
  if (std::getenv("DYN_TLS_CERT") || std::getenv("DYN_TLS_KEY") || std::getenv("DYN_TLS_CA")) {
    throw std::runtime_error(
        "DYN_TLS_* is set but dynamo was built without TLS support (OpenSSL not found)");
  }
  return false;
}

}  // namespace tls

struct TlsSession::Impl {};

TlsSession::TlsSession() = default;
TlsSession::~TlsSession() = default;

std::unique_ptr<TlsSession> TlsSession::make(int, bool) {
  throw std::logic_error("TLS session requested in a build without TLS support");
}
std::unique_ptr<TlsSession> TlsSession::make_client(int fd) { return make(fd, false); }
std::unique_ptr<TlsSession> TlsSession::make_server(int fd) { return make(fd, true); }

size_t TlsSession::read_some(char*, size_t) { return 0; }
bool TlsSession::read_exact(char*, size_t) { return false; }
bool TlsSession::write_all(std::string_view) { return false; }
void TlsSession::set_read_timeout(std::chrono::milliseconds) {}

}  // namespace dynamo::transports
