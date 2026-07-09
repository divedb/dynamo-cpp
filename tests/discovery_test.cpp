// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>

#include <thread>

#include "discovery/in_process.h"
#include "discovery/server.h"
#include "discovery/tcp.h"
#include "runtime/coro/sync_wait.h"

using namespace dynamo;
using namespace std::chrono_literals;

namespace {

RuntimeConfig test_config() {
  RuntimeConfig config;
  config.num_worker_threads = 4;
  config.num_background_threads = 2;
  return config;
}

/// Collects watch events until `n` arrived or the stream closed.
coro::Task<std::vector<discovery::WatchEvent>> take_events(discovery::WatchStream& stream,
                                                           size_t n) {
  std::vector<discovery::WatchEvent> out;
  while (out.size() < n) {
    auto event = co_await stream.events.recv();
    if (!event) break;
    out.push_back(std::move(*event));
  }
  co_return out;
}

}  // namespace

TEST_CASE("in-process discovery: create, duplicate, prefix get", "[discovery]") {
  auto rt = Runtime::create(test_config());
  auto store = discovery::InProcessDiscovery::new_store();
  auto disco = std::make_shared<discovery::InProcessDiscovery>(rt, store);

  coro::sync_wait([&]() -> coro::Task<void> {
    co_await disco->kv_create("a/b/1", "one", std::nullopt);
    co_await disco->kv_create("a/b/2", "two", std::nullopt);
    co_await disco->kv_create("a/c/1", "other", std::nullopt);

    REQUIRE_THROWS(co_await disco->kv_create("a/b/1", "dup", std::nullopt));

    auto kvs = co_await disco->kv_get_prefix("a/b/");
    REQUIRE(kvs.size() == 2);
  }());

  rt.shutdown();
  REQUIRE(rt.join_tasks(2000ms));
}

TEST_CASE("in-process discovery: watch sees snapshot then updates then lease death",
          "[discovery]") {
  auto rt = Runtime::create(test_config());
  auto store = discovery::InProcessDiscovery::new_store();
  auto disco = std::make_shared<discovery::InProcessDiscovery>(rt, store);

  coro::sync_wait([&]() -> coro::Task<void> {
    co_await disco->kv_create("w/existing", "0", std::nullopt);

    auto watch = co_await disco->kv_get_and_watch_prefix("w/");

    // Snapshot arrives first.
    auto snapshot = co_await take_events(watch, 1);
    REQUIRE(snapshot.size() == 1);
    REQUIRE(snapshot[0].kind == discovery::WatchEvent::Kind::Put);
    REQUIRE(snapshot[0].kv.key == "w/existing");

    // Lease-bound key: put visible, delete arrives when the lease dies.
    auto lease = co_await disco->create_lease(10s);
    co_await disco->kv_create("w/leased", "1", lease.id);

    auto put = co_await take_events(watch, 1);
    REQUIRE(put[0].kind == discovery::WatchEvent::Kind::Put);
    REQUIRE(put[0].kv.key == "w/leased");

    lease.revoke();
    auto del = co_await take_events(watch, 1);
    REQUIRE(del[0].kind == discovery::WatchEvent::Kind::Delete);
    REQUIRE(del[0].kv.key == "w/leased");

    auto remaining = co_await disco->kv_get_prefix("w/leased");
    REQUIRE(remaining.empty());
  }());

  rt.shutdown();
  REQUIRE(rt.join_tasks(2000ms));
}

TEST_CASE("tcp discovery: registration, watch across clients, revoke", "[discovery][tcp]") {
  auto server = discovery::DiscoveryServer::start();
  auto rt_a = Runtime::create(test_config());
  auto rt_b = Runtime::create(test_config());
  {
    auto client_a = discovery::TcpDiscovery::connect(rt_a, server->address());
    auto client_b = discovery::TcpDiscovery::connect(rt_b, server->address());

    coro::sync_wait([&]() -> coro::Task<void> {
      auto watch = co_await client_b->kv_get_and_watch_prefix("svc/");

      auto lease = co_await client_a->create_lease(10s);
      co_await client_a->kv_create("svc/x", "payload", lease.id);
      REQUIRE_THROWS(co_await client_a->kv_create("svc/x", "dup", lease.id));

      auto put = co_await take_events(watch, 1);
      REQUIRE(put[0].kind == discovery::WatchEvent::Kind::Put);
      REQUIRE(put[0].kv.key == "svc/x");
      REQUIRE(put[0].kv.value == "payload");
      REQUIRE(put[0].kv.lease_id == lease.id);

      auto seen = co_await client_b->kv_get_prefix("svc/");
      REQUIRE(seen.size() == 1);

      lease.revoke();
      auto del = co_await take_events(watch, 1);
      REQUIRE(del[0].kind == discovery::WatchEvent::Kind::Delete);
      REQUIRE(del[0].kv.key == "svc/x");
    }());

    rt_a.shutdown();
    rt_b.shutdown();
    REQUIRE(rt_a.join_tasks(3000ms));
    REQUIRE(rt_b.join_tasks(3000ms));
  }
  server->stop();
}

TEST_CASE("kv_put upserts and rebinds leases; create_or_validate validates",
          "[discovery][kv]") {
  auto rt = Runtime::create(test_config());
  auto store = discovery::InProcessDiscovery::new_store();
  auto disco = std::make_shared<discovery::InProcessDiscovery>(rt, store);

  coro::sync_wait([&]() -> coro::Task<void> {
    // kv_put: create then overwrite.
    co_await disco->kv_put("p/x", "v1", std::nullopt);
    co_await disco->kv_put("p/x", "v2", std::nullopt);
    auto kvs = co_await disco->kv_get_prefix("p/x");
    REQUIRE(kvs.size() == 1);
    REQUIRE(kvs[0].value == "v2");
    REQUIRE(kvs[0].mod_revision == 2);  // revisions are monotonic per mutation

    // Rebinding to a lease: revoking it must now delete the key.
    auto lease = co_await disco->create_lease(10s);
    co_await disco->kv_put("p/x", "v3", lease.id);
    lease.revoke();
    REQUIRE((co_await disco->kv_get_prefix("p/x")).empty());

    // create_or_validate: create, revalidate same value, reject different.
    co_await disco->kv_create_or_validate("p/y", "cfg", std::nullopt);
    co_await disco->kv_create_or_validate("p/y", "cfg", std::nullopt);
    REQUIRE_THROWS(co_await disco->kv_create_or_validate("p/y", "other", std::nullopt));
  }());

  rt.shutdown();
  REQUIRE(rt.join_tasks(2000ms));
}

TEST_CASE("kv_put and create_or_validate over discoveryd", "[discovery][kv][tcp]") {
  auto server = discovery::DiscoveryServer::start();
  auto rt = Runtime::create(test_config());
  {
    auto client = discovery::TcpDiscovery::connect(rt, server->address());

    coro::sync_wait([&]() -> coro::Task<void> {
      co_await client->kv_put("t/x", "v1", std::nullopt);
      co_await client->kv_put("t/x", "v2", std::nullopt);
      auto kvs = co_await client->kv_get_prefix("t/");
      REQUIRE(kvs.size() == 1);
      REQUIRE(kvs[0].value == "v2");
      REQUIRE(kvs[0].mod_revision == 2);

      co_await client->kv_create_or_validate("t/y", "cfg", std::nullopt);
      co_await client->kv_create_or_validate("t/y", "cfg", std::nullopt);
      REQUIRE_THROWS(co_await client->kv_create_or_validate("t/y", "other", std::nullopt));

      // Watch events carry revisions.
      auto watch = co_await client->kv_get_and_watch_prefix("t/");
      auto snapshot = co_await take_events(watch, 2);
      REQUIRE(snapshot.size() == 2);
      for (auto& event : snapshot) REQUIRE(event.kv.mod_revision > 0);
    }());

    rt.shutdown();
    REQUIRE(rt.join_tasks(3000ms));
  }
  server->stop();
}

TEST_CASE("events: publish reaches current subscribers (in-process)", "[discovery][events]") {
  auto rt = Runtime::create(test_config());
  auto store = discovery::InProcessDiscovery::new_store();
  auto disco = std::make_shared<discovery::InProcessDiscovery>(rt, store);

  coro::sync_wait([&]() -> coro::Task<void> {
    auto stream = co_await disco->subscribe("model.ready");
    co_await disco->publish("model.ready", "payload-1");
    co_await disco->publish("other.subject", "ignored");
    co_await disco->publish("model.ready", "payload-2");

    auto e1 = co_await stream.events.recv();
    REQUIRE(e1);
    REQUIRE(e1->subject == "model.ready");
    REQUIRE(e1->payload == "payload-1");
    auto e2 = co_await stream.events.recv();
    REQUIRE(e2);
    REQUIRE(e2->payload == "payload-2");
  }());

  rt.shutdown();
  REQUIRE(rt.join_tasks(2000ms));
}

TEST_CASE("events: publish crosses processes via discoveryd", "[discovery][events][tcp]") {
  auto server = discovery::DiscoveryServer::start();
  auto rt_a = Runtime::create(test_config());
  auto rt_b = Runtime::create(test_config());
  {
    auto publisher = discovery::TcpDiscovery::connect(rt_a, server->address());
    auto subscriber = discovery::TcpDiscovery::connect(rt_b, server->address());

    coro::sync_wait([&]() -> coro::Task<void> {
      auto stream = co_await subscriber->subscribe("kv.events");
      co_await publisher->publish("kv.events", "hello");
      auto event = co_await stream.events.recv();
      REQUIRE(event);
      REQUIRE(event->subject == "kv.events");
      REQUIRE(event->payload == "hello");
    }());

    rt_a.shutdown();
    rt_b.shutdown();
    REQUIRE(rt_a.join_tasks(3000ms));
    REQUIRE(rt_b.join_tasks(3000ms));
  }
  server->stop();
}

TEST_CASE("tcp discovery: reconnect resyncs watches and revives requests",
          "[discovery][tcp][reconnect]") {
  auto server = discovery::DiscoveryServer::start();
  auto rt_watcher = Runtime::create(test_config());
  auto rt_writer = Runtime::create(test_config());
  {
    auto watcher = discovery::TcpDiscovery::connect(rt_watcher, server->address());
    auto writer = discovery::TcpDiscovery::connect(rt_writer, server->address());

    coro::sync_wait([&]() -> coro::Task<void> {
      auto watch = co_await watcher->kv_get_and_watch_prefix("rc/");

      auto lease1 = co_await writer->create_lease(10s);
      co_await writer->kv_create("rc/k1", "v1", lease1.id);
      auto put1 = co_await take_events(watch, 1);
      REQUIRE(put1[0].kv.key == "rc/k1");

      // Network blip on the watcher: while it is away, k1 disappears and k2
      // appears. After resync it must observe both changes.
      watcher->debug_drop_connection();

      auto lease2 = co_await writer->create_lease(10s);
      co_await writer->kv_create("rc/k2", "v2", lease2.id);
      lease1.revoke();

      bool saw_k2_put = false;
      bool saw_k1_delete = false;
      for (int i = 0; i < 20 && !(saw_k2_put && saw_k1_delete); ++i) {
        auto events = co_await take_events(watch, 1);
        REQUIRE_FALSE(events.empty());  // stream must survive the blip
        for (auto& event : events) {
          if (event.kind == discovery::WatchEvent::Kind::Put && event.kv.key == "rc/k2") {
            saw_k2_put = true;
          }
          if (event.kind == discovery::WatchEvent::Kind::Delete && event.kv.key == "rc/k1") {
            saw_k1_delete = true;
          }
        }
      }
      REQUIRE(saw_k2_put);
      REQUIRE(saw_k1_delete);

      // Requests work again on the revived connection.
      auto kvs = co_await watcher->kv_get_prefix("rc/");
      REQUIRE(kvs.size() == 1);
      REQUIRE(kvs[0].key == "rc/k2");
    }());

    rt_watcher.shutdown();
    rt_writer.shutdown();
    REQUIRE(rt_watcher.join_tasks(3000ms));
    REQUIRE(rt_writer.join_tasks(3000ms));
  }
  server->stop();
}

TEST_CASE("tcp discovery: lease survives a connection blip shorter than its TTL",
          "[discovery][tcp][reconnect]") {
  auto server = discovery::DiscoveryServer::start();
  auto rt = Runtime::create(test_config());
  {
    auto client = discovery::TcpDiscovery::connect(rt, server->address());

    coro::sync_wait([&]() -> coro::Task<void> {
      auto lease = co_await client->create_lease(5s);
      co_await client->kv_create("blip/key", "v", lease.id);

      client->debug_drop_connection();
      std::this_thread::sleep_for(500ms);  // blip well under the 5s TTL

      // The lease's keep-alive retried through the blip: still live.
      REQUIRE(lease.is_live());
      auto kvs = co_await client->kv_get_prefix("blip/");
      REQUIRE(kvs.size() == 1);
    }());

    rt.shutdown();
    REQUIRE(rt.join_tasks(3000ms));
  }
  server->stop();
}

TEST_CASE("tcp discovery: lease TTL expires when owner vanishes", "[discovery][tcp]") {
  auto server = discovery::DiscoveryServer::start();
  auto rt_owner = Runtime::create(test_config());
  auto rt_watcher = Runtime::create(test_config());
  {
    auto owner = discovery::TcpDiscovery::connect(rt_owner, server->address());
    auto watcher = discovery::TcpDiscovery::connect(rt_watcher, server->address());

    coro::sync_wait([&]() -> coro::Task<void> {
      auto watch = co_await watcher->kv_get_and_watch_prefix("ttl/");

      auto lease = co_await owner->create_lease(1s);
      co_await owner->kv_create("ttl/instance", "info", lease.id);

      auto put = co_await take_events(watch, 1);
      REQUIRE(put[0].kind == discovery::WatchEvent::Kind::Put);

      // Simulate a crash: drop the connection without revoking. The server
      // must reap the lease after its TTL and watchers see the delete.
      owner->shutdown();

      auto del = co_await take_events(watch, 1);
      REQUIRE(del.size() == 1);
      REQUIRE(del[0].kind == discovery::WatchEvent::Kind::Delete);
      REQUIRE(del[0].kv.key == "ttl/instance");
    }());

    rt_owner.shutdown();
    rt_watcher.shutdown();
    REQUIRE(rt_owner.join_tasks(3000ms));
    REQUIRE(rt_watcher.join_tasks(3000ms));
  }
  server->stop();
}
