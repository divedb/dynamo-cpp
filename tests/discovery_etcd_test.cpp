// SPDX-License-Identifier: Apache-2.0
//
// EtcdDiscovery tests. They need a live etcd server (default
// 127.0.0.1:2379, override with DYN_TEST_ETCD=host:port) and SKIP
// themselves when none is reachable, so plain `ctest` runs stay green
// without etcd. All test keys are lease-bound: revocation cleans the store.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <set>
#include <thread>

#include "component/component.h"
#include "discovery/etcd.h"
#include "runtime/coro/sync_wait.h"
#include "transports/socket.h"

using namespace dynamo;
using namespace std::chrono_literals;

namespace {

RuntimeConfig test_config() {
  RuntimeConfig config;
  config.num_worker_threads = 4;
  config.num_background_threads = 2;
  return config;
}

std::string etcd_address() {
  const char* env = std::getenv("DYN_TEST_ETCD");
  return env ? env : "127.0.0.1:2379";
}

/// Fast TCP probe: gRPC would block for its full deadline on a dead address.
bool etcd_reachable() {
  auto [host, port] = transports::parse_address(etcd_address());
  return transports::Socket::connect(host, port).has_value();
}

#define REQUIRE_ETCD() \
  if (!etcd_reachable()) SKIP("no etcd server at " << etcd_address())

/// Unique key prefix per test run: the etcd store persists across runs.
std::string unique_prefix() {
  static std::atomic<int> counter{0};
  return "dynamo-cpp-test/" + std::to_string(::getpid()) + "-" +
         std::to_string(counter.fetch_add(1)) + "/";
}

coro::Task<discovery::WatchEvent> next_event(discovery::WatchStream& stream) {
  auto event = co_await stream.events.recv();
  REQUIRE(event);
  co_return std::move(*event);
}

}  // namespace

TEST_CASE("etcd: kv create/put/validate/get_prefix", "[discovery][etcd]") {
  REQUIRE_ETCD();
  auto rt = Runtime::create(test_config());
  {
    auto disco = discovery::EtcdDiscovery::connect(rt, etcd_address());
    auto prefix = unique_prefix();

    coro::sync_wait([&]() -> coro::Task<void> {
      auto lease = co_await disco->create_lease(10s);

      co_await disco->kv_create(prefix + "a", "one", lease.id);
      REQUIRE_THROWS_WITH(co_await disco->kv_create(prefix + "a", "dup", lease.id),
                          Catch::Matchers::ContainsSubstring("already exists"));

      // Upsert + same-value validation + differing-value rejection.
      co_await disco->kv_put(prefix + "a", "two", lease.id);
      co_await disco->kv_create_or_validate(prefix + "a", "two", lease.id);
      REQUIRE_THROWS_WITH(co_await disco->kv_create_or_validate(prefix + "a", "three", lease.id),
                          Catch::Matchers::ContainsSubstring("differs"));
      co_await disco->kv_create_or_validate(prefix + "b", "fresh", lease.id);

      auto kvs = co_await disco->kv_get_prefix(prefix);
      REQUIRE(kvs.size() == 2);
      REQUIRE(kvs[0].key == prefix + "a");
      REQUIRE(kvs[0].value == "two");
      REQUIRE(kvs[0].lease_id == lease.id);
      REQUIRE(kvs[0].mod_revision > 0);
      REQUIRE(kvs[1].value == "fresh");

      // Revocation deletes every bound key.
      lease.revoke();
      for (int i = 0; i < 100; ++i) {
        if ((co_await disco->kv_get_prefix(prefix)).empty()) break;
        std::this_thread::sleep_for(20ms);
      }
      REQUIRE((co_await disco->kv_get_prefix(prefix)).empty());
    }());

    disco->shutdown();
  }
  rt.shutdown();
  REQUIRE(rt.join_tasks(5000ms));
}

TEST_CASE("etcd: watch sees snapshot, live puts, lease-death deletes", "[discovery][etcd]") {
  REQUIRE_ETCD();
  auto rt = Runtime::create(test_config());
  {
    auto disco = discovery::EtcdDiscovery::connect(rt, etcd_address());
    auto prefix = unique_prefix();

    coro::sync_wait([&]() -> coro::Task<void> {
      auto lease = co_await disco->create_lease(10s);
      co_await disco->kv_create(prefix + "pre", "existing", lease.id);

      auto watch = co_await disco->kv_get_and_watch_prefix(prefix);

      // Snapshot first.
      auto snapshot = co_await next_event(watch);
      REQUIRE(snapshot.kind == discovery::WatchEvent::Kind::Put);
      REQUIRE(snapshot.kv.key == prefix + "pre");
      REQUIRE(snapshot.kv.value == "existing");

      // Then live updates.
      co_await disco->kv_create(prefix + "live", "update", lease.id);
      auto put = co_await next_event(watch);
      REQUIRE(put.kind == discovery::WatchEvent::Kind::Put);
      REQUIRE(put.kv.key == prefix + "live");
      REQUIRE(put.kv.mod_revision > snapshot.kv.mod_revision);

      // Lease death deletes both keys; the watch reports both.
      lease.revoke();
      std::set<std::string> deleted;
      for (int i = 0; i < 2; ++i) {
        auto del = co_await next_event(watch);
        REQUIRE(del.kind == discovery::WatchEvent::Kind::Delete);
        deleted.insert(del.kv.key);
      }
      REQUIRE(deleted == std::set<std::string>{prefix + "pre", prefix + "live"});
    }());

    disco->shutdown();
  }
  rt.shutdown();
  REQUIRE(rt.join_tasks(5000ms));
}

TEST_CASE("etcd: keep-alive outlives the ttl; events are unsupported", "[discovery][etcd]") {
  REQUIRE_ETCD();
  auto rt = Runtime::create(test_config());
  {
    auto disco = discovery::EtcdDiscovery::connect(rt, etcd_address());
    auto prefix = unique_prefix();

    coro::sync_wait([&]() -> coro::Task<void> {
      auto lease = co_await disco->create_lease(2s);
      co_await disco->kv_create(prefix + "k", "v", lease.id);

      // Past the raw TTL: the KeepAlive must have refreshed the lease.
      std::this_thread::sleep_for(3s);
      REQUIRE(lease.is_live());
      REQUIRE((co_await disco->kv_get_prefix(prefix)).size() == 1);
      lease.revoke();

      // etcd carries no transient pub/sub (Dynamo pairs it with NATS).
      REQUIRE_THROWS_WITH(co_await disco->publish("s", "p"),
                          Catch::Matchers::ContainsSubstring("does not support"));
      REQUIRE_THROWS_WITH(co_await disco->subscribe("s"),
                          Catch::Matchers::ContainsSubstring("does not support"));
      REQUIRE_THROWS_WITH(co_await disco->queue_dispatch("s", "p"),
                          Catch::Matchers::ContainsSubstring("does not support"));
    }());

    disco->shutdown();
  }
  rt.shutdown();
  REQUIRE(rt.join_tasks(5000ms));
}

namespace {

coro::AsyncGenerator<std::string> etcd_echo_chars(std::string data, pipeline::ContextPtr ctx) {
  for (char c : data) {
    if (ctx->is_stopped()) break;
    std::string item(1, c);
    co_yield item;
  }
}

struct EtcdEchoEngine final : pipeline::AsyncEngine<std::string, std::string> {
  coro::Task<pipeline::ManyOut<std::string>> generate(
      pipeline::SingleIn<std::string> in) override {
    auto [data, controller] = std::move(in).into_parts();
    pipeline::ContextPtr ctx = controller;
    co_return pipeline::ManyOut<std::string>(etcd_echo_chars(std::move(data), ctx), ctx);
  }
};

}  // namespace

TEST_CASE("etcd: component layer round-trip over etcd discovery", "[discovery][etcd][endpoint]") {
  REQUIRE_ETCD();
  auto rt = Runtime::create(test_config());
  {
    auto drt = component::DistributedRuntime::create(
        rt, {.discovery_address = "etcd://" + etcd_address()});
    // Unique namespace: instance keys persist only as long as their lease.
    auto endpoint = drt.ns("etcdtest" + std::to_string(::getpid()))
                        .component("backend")
                        .endpoint("generate");

    coro::sync_wait([&]() -> coro::Task<void> {
      co_await rt.primary().schedule();
      rt.spawn(endpoint.serve<std::string, std::string>(std::make_shared<EtcdEchoEngine>()));

      auto client = co_await endpoint.client<std::string, std::string>();
      co_await client.wait_for_instances();

      auto stream = co_await client.generate(pipeline::SingleIn<std::string>("etcd!"));
      std::string reassembled;
      while (auto item = co_await stream.next()) reassembled += *item;
      REQUIRE(reassembled == "etcd!");
    }());

    rt.shutdown();
    REQUIRE(rt.join_tasks(5000ms));
  }
}
