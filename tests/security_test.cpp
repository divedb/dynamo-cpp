// SPDX-License-Identifier: Apache-2.0
//
// Transport security: shared-token connection auth (always built) and the
// optional mTLS layer (DYNAMO_HAVE_TLS). TLS tests generate throwaway
// self-signed certificates at runtime — no fixtures, no OpenSSL CLI.

#include <catch2/catch_test_macros.hpp>

#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

#include "component/component.h"
#include "discovery/server.h"
#include "discovery/tcp.h"
#include "runtime/coro/sync_wait.h"
#include "transports/auth.h"
#include "transports/control_plane.h"
#include "transports/socket.h"
#include "transports/tls.h"

#ifdef DYNAMO_HAVE_TLS
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/x509.h>
#endif

using namespace dynamo;
using namespace std::chrono_literals;
using nlohmann::json;
using transports::Listener;
using transports::Socket;
using transports::TwoPartMessage;

namespace {

RuntimeConfig test_config() {
  RuntimeConfig config;
  config.num_worker_threads = 4;
  config.num_background_threads = 2;
  return config;
}

/// Scoped token config; resets to "no token" so later tests are unaffected.
struct AuthGuard {
  explicit AuthGuard(std::string token) { transports::auth::set_token(std::move(token)); }
  ~AuthGuard() { transports::auth::set_token(std::nullopt); }
};

/// Reads one frame off a raw socket, tolerating both clean close (nullopt)
/// and mid-frame/garbage errors (throw) — either means "rejected".
bool peer_rejected(Socket& sock) {
  try {
    return !sock.read_frame().has_value();
  } catch (const std::exception&) {
    return true;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Token auth (no TLS required)
// ---------------------------------------------------------------------------

TEST_CASE("control plane requires the shared token", "[transports][auth]") {
  AuthGuard guard("sekrit-token");
  auto rt = Runtime::create(test_config());
  {
    auto server = transports::ControlPlaneServer::start(rt);
    server->register_query("q", [](const std::string& payload) { return payload + "!"; });
    auto [host, port] = transports::parse_address(server->address());

    SECTION("the authenticated client path works") {
      REQUIRE(transports::dispatch_query(server->address(), "q", "ping") == "ping!");
    }

    SECTION("a wrong token is rejected") {
      auto sock = Socket::connect(host, port);
      REQUIRE(sock);
      REQUIRE(sock->write_frame(TwoPartMessage::from_header(json{{"auth_token", "nope"}}.dump())));
      REQUIRE(peer_rejected(*sock));
    }

    SECTION("skipping the auth frame is rejected") {
      auto sock = Socket::connect(host, port);
      REQUIRE(sock);
      // First frame is treated as the auth frame no matter what it claims to be.
      REQUIRE(sock->write_frame(TwoPartMessage::from_header(json{{"subject", "q"}}.dump())));
      REQUIRE(peer_rejected(*sock));
    }

    server->stop();
  }
  rt.shutdown();
  REQUIRE(rt.join_tasks(3000ms));
}

TEST_CASE("discoveryd and tcp client authenticate end to end", "[transports][auth]") {
  AuthGuard guard("cluster-token");
  auto server = discovery::DiscoveryServer::start();
  auto rt = Runtime::create(test_config());
  {
    auto client = discovery::TcpDiscovery::connect(rt, server->address());
    coro::sync_wait([&]() -> coro::Task<void> {
      auto lease = co_await client->create_lease(10s);
      co_await client->kv_create("auth/x", "payload", lease.id);
      auto seen = co_await client->kv_get_prefix("auth/");
      REQUIRE(seen.size() == 1);
      REQUIRE(seen[0].value == "payload");
    }());
    rt.shutdown();
    REQUIRE(rt.join_tasks(3000ms));
  }

  // An unauthenticated raw connection gets nothing.
  auto [host, port] = transports::parse_address(server->address());
  auto sock = Socket::connect(host, port);
  REQUIRE(sock);
  REQUIRE(sock->write_frame(
      TwoPartMessage::from_header(json{{"op", "kv_get_prefix"}, {"req", 1}}.dump())));
  REQUIRE(peer_rejected(*sock));

  server->stop();
}

// ---------------------------------------------------------------------------
// mTLS
// ---------------------------------------------------------------------------

#ifdef DYNAMO_HAVE_TLS

namespace {

std::string unique_ns() {
  static std::atomic<int> counter{0};
  return "securityns" + std::to_string(counter.fetch_add(1));
}

coro::AsyncGenerator<std::string> echo_chars(std::string data, pipeline::ContextPtr ctx) {
  for (char c : data) {
    if (ctx->is_stopped()) break;
    std::string item(1, c);
    co_yield item;
  }
}

struct EchoEngine final : pipeline::AsyncEngine<std::string, std::string> {
  coro::Task<pipeline::ManyOut<std::string>> generate(
      pipeline::SingleIn<std::string> in) override {
    auto [data, controller] = std::move(in).into_parts();
    pipeline::ContextPtr ctx = controller;
    co_return pipeline::ManyOut<std::string>(echo_chars(std::move(data), ctx), ctx);
  }
};

struct CertFiles {
  std::string cert_path;
  std::string key_path;
};

/// Self-signed cert + key (P-256, one year), written as PEM under a fresh
/// temp dir. The cert doubles as its own trust anchor.
CertFiles write_self_signed(const std::string& cn) {
  namespace fs = std::filesystem;
  static std::atomic<long> serial{static_cast<long>(::time(nullptr))};

  std::string dir = (fs::temp_directory_path() / "dynamo-tls-XXXXXX").string();
  REQUIRE(::mkdtemp(dir.data()) != nullptr);

  EVP_PKEY* pkey = EVP_EC_gen("P-256");
  REQUIRE(pkey != nullptr);

  X509* x = X509_new();
  REQUIRE(x != nullptr);
  ASN1_INTEGER_set(X509_get_serialNumber(x), serial.fetch_add(1));
  X509_gmtime_adj(X509_getm_notBefore(x), -3600);
  X509_gmtime_adj(X509_getm_notAfter(x), 365L * 24 * 3600);
  X509_set_pubkey(x, pkey);
  X509_NAME* name = X509_get_subject_name(x);
  X509_NAME_add_entry_by_txt(name, "CN", MBSTRING_ASC,
                             reinterpret_cast<const unsigned char*>(cn.c_str()), -1, -1, 0);
  X509_set_issuer_name(x, name);
  REQUIRE(X509_sign(x, pkey, EVP_sha256()) > 0);

  CertFiles files{dir + "/cert.pem", dir + "/key.pem"};
  FILE* cf = std::fopen(files.cert_path.c_str(), "w");
  REQUIRE(cf != nullptr);
  REQUIRE(PEM_write_X509(cf, x) == 1);
  std::fclose(cf);
  FILE* kf = std::fopen(files.key_path.c_str(), "w");
  REQUIRE(kf != nullptr);
  REQUIRE(PEM_write_PrivateKey(kf, pkey, nullptr, nullptr, 0, nullptr, nullptr) == 1);
  std::fclose(kf);

  X509_free(x);
  EVP_PKEY_free(pkey);
  return files;
}

/// Scoped TLS config; resets to "TLS off" so later tests are unaffected.
struct TlsGuard {
  explicit TlsGuard(const CertFiles& identity, const std::string& ca_path) {
    transports::tls::configure(
        transports::tls::Options{identity.cert_path, identity.key_path, ca_path});
  }
  ~TlsGuard() { transports::tls::configure(std::nullopt); }
};

}  // namespace

TEST_CASE("frames roundtrip full-duplex over TLS", "[transports][tls]") {
  auto certs = write_self_signed("dynamo-test");
  TlsGuard guard(certs, certs.cert_path);

  auto listener = Listener::bind("127.0.0.1", 0, /*with_tls=*/true);
  REQUIRE(listener);
  std::thread echo_server([&] {
    auto sock = listener->accept();
    if (!sock) return;
    try {
      while (auto frame = sock->read_frame()) {
        if (!sock->write_frame(*frame)) break;
      }
    } catch (const std::exception&) {
    }
  });

  auto client = Socket::connect("127.0.0.1", listener->port(), /*with_tls=*/true);
  REQUIRE(client);
  REQUIRE(client->is_tls());

  constexpr int kFrames = 64;
  auto payload_for = [](int i) {
    // Frame 0 is 1 MiB to force many TLS records and partial reads.
    return std::string(i == 0 ? (1u << 20) : 128u, static_cast<char>('a' + i % 26));
  };

  // Writer and reader run concurrently on the same socket (the data-plane
  // usage pattern the TLS session must support).
  std::thread writer([&] {
    for (int i = 0; i < kFrames; ++i) {
      if (!client->write_frame(
              TwoPartMessage::from_parts(json{{"i", i}}.dump(), payload_for(i)))) {
        return;
      }
    }
  });

  for (int i = 0; i < kFrames; ++i) {
    auto frame = client->read_frame();
    REQUIRE(frame.has_value());
    REQUIRE(json::parse(frame->header).at("i").get<int>() == i);
    REQUIRE(frame->data == payload_for(i));
  }

  writer.join();
  client->shutdown();
  client->close();
  echo_server.join();
  listener->shutdown();
}

TEST_CASE("a peer outside the trust store is rejected", "[transports][tls]") {
  auto identity = write_self_signed("dynamo-node");
  auto other_ca = write_self_signed("unrelated-ca");
  // Everyone presents `identity` but trusts only `other_ca`: verification
  // must fail in both directions. This proves peer verification is on — a
  // context that skipped verification would happily roundtrip here.
  TlsGuard guard(identity, other_ca.cert_path);

  auto listener = Listener::bind("127.0.0.1", 0, /*with_tls=*/true);
  REQUIRE(listener);
  std::thread server([&] {
    auto sock = listener->accept();
    if (!sock) return;
    (void)peer_rejected(*sock);
  });

  auto client = Socket::connect("127.0.0.1", listener->port(), /*with_tls=*/true);
  REQUIRE(client);
  // The first I/O drives the handshake, which must fail.
  bool wrote = client->write_frame(TwoPartMessage::from_header("{}"));
  bool rejected = !wrote || peer_rejected(*client);
  REQUIRE(rejected);

  server.join();
  listener->shutdown();
}

TEST_CASE("plaintext and TLS peers do not interoperate", "[transports][tls]") {
  auto certs = write_self_signed("dynamo-test");
  TlsGuard guard(certs, certs.cert_path);

  auto listener = Listener::bind("127.0.0.1", 0, /*with_tls=*/true);
  REQUIRE(listener);
  std::thread server([&] {
    // Connection 1: a plaintext client — its bytes are not a ClientHello, so
    // the read must fail without wedging the listener.
    if (auto sock = listener->accept()) {
      (void)peer_rejected(*sock);
    }
    // Connection 2: a real TLS client — echo one frame.
    if (auto sock = listener->accept()) {
      try {
        if (auto frame = sock->read_frame()) sock->write_frame(*frame);
      } catch (const std::exception&) {
      }
    }
  });

  auto plain = Socket::connect("127.0.0.1", listener->port());
  REQUIRE(plain);
  (void)plain->write_frame(TwoPartMessage::from_header(json{{"hello", 1}}.dump()));
  REQUIRE(peer_rejected(*plain));
  plain->close();

  auto secure = Socket::connect("127.0.0.1", listener->port(), /*with_tls=*/true);
  REQUIRE(secure);
  REQUIRE(secure->write_frame(TwoPartMessage::from_header(json{{"hello", 2}}.dump())));
  auto frame = secure->read_frame();
  REQUIRE(frame.has_value());
  REQUIRE(json::parse(frame->header).at("hello").get<int>() == 2);
  secure->close();

  server.join();
  listener->shutdown();
}

TEST_CASE("endpoint round trip works over TLS with token auth", "[transports][tls][auth]") {
  auto certs = write_self_signed("dynamo-cluster");
  TlsGuard tls_guard(certs, certs.cert_path);
  AuthGuard auth_guard("cluster-token");

  auto rt = Runtime::create(test_config());
  {
    auto drt = component::DistributedRuntime::create(rt, {});
    auto endpoint = drt.ns(unique_ns()).component("secure").endpoint("generate");

    coro::sync_wait([&]() -> coro::Task<void> {
      co_await rt.primary().schedule();
      rt.spawn(endpoint.serve<std::string, std::string>(std::make_shared<EchoEngine>()));

      auto client = co_await endpoint.client<std::string, std::string>();
      co_await client.wait_for_instances();

      auto stream = co_await client.generate(pipeline::SingleIn<std::string>("secure hi"));
      std::string reassembled;
      while (auto item = co_await stream.next()) reassembled += *item;
      REQUIRE(reassembled == "secure hi");
    }());

    rt.shutdown();
    REQUIRE(rt.join_tasks(5000ms));
  }
}

#endif  // DYNAMO_HAVE_TLS
