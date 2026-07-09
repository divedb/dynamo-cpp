// SPDX-License-Identifier: Apache-2.0
//
// End-to-end component-layer tests: serving, watching, routing, streaming,
// cancellation. In-process discovery, real TCP control/data planes.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <atomic>
#include <set>
#include <thread>

#include "component/component.h"
#include "pipeline/annotated.h"
#include "pipeline/operators.h"
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

/// Unique namespace per test: the in-process store is process-wide.
std::string unique_ns() {
  static std::atomic<int> counter{0};
  return "testns" + std::to_string(counter.fetch_add(1));
}

// --- engines ---------------------------------------------------------------

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

coro::AsyncGenerator<int64_t> label_stream(int64_t label) {
  co_yield label;
}

/// Replies with a fixed label; used to observe routing decisions.
struct LabelEngine final : pipeline::AsyncEngine<std::string, int64_t> {
  explicit LabelEngine(int64_t label) : label(label) {}
  coro::Task<pipeline::ManyOut<int64_t>> generate(pipeline::SingleIn<std::string> in) override {
    auto [data, controller] = std::move(in).into_parts();
    pipeline::ContextPtr ctx = controller;
    co_return pipeline::ManyOut<int64_t>(label_stream(label), ctx);
  }
  int64_t label;
};

coro::AsyncGenerator<int> endless_ints(pipeline::ContextPtr ctx, std::atomic<bool>& finished) {
  // Generation "ends" either by observing stop/kill (loop exit) or by the
  // pump tearing the stream down after a dead socket — both count.
  struct Ended {
    std::atomic<bool>& flag;
    ~Ended() { flag = true; }
  } ended{finished};

  int i = 0;
  while (!ctx->is_stopped() && !ctx->is_killed()) {
    co_yield i;
    ++i;
  }
}

/// Streams forever until stop/kill; records that the producer loop exited.
struct EndlessEngine final : pipeline::AsyncEngine<std::string, int> {
  coro::Task<pipeline::ManyOut<int>> generate(pipeline::SingleIn<std::string> in) override {
    auto [data, controller] = std::move(in).into_parts();
    pipeline::ContextPtr ctx = controller;
    co_return pipeline::ManyOut<int>(endless_ints(ctx, finished), ctx);
  }
  std::atomic<bool> finished{false};
};

struct FailingEngine final : pipeline::AsyncEngine<std::string, std::string> {
  coro::Task<pipeline::ManyOut<std::string>> generate(pipeline::SingleIn<std::string>) override {
    throw std::runtime_error("engine exploded");
    co_return pipeline::ManyOut<std::string>();  // unreachable
  }
};

}  // namespace

TEST_CASE("unary request streams a response and completes", "[endpoint]") {
  auto rt = Runtime::create(test_config());
  {
    auto drt = component::DistributedRuntime::create(rt, {});
    auto endpoint = drt.ns(unique_ns()).component("backend").endpoint("generate");

    coro::sync_wait([&]() -> coro::Task<void> {
      co_await rt.primary().schedule();
      rt.spawn(endpoint.serve<std::string, std::string>(std::make_shared<EchoEngine>()));

      auto client = co_await endpoint.client<std::string, std::string>();
      co_await client.wait_for_instances();

      auto stream = co_await client.generate(pipeline::SingleIn<std::string>("hi!"));
      std::string reassembled;
      while (auto item = co_await stream.next()) reassembled += *item;
      REQUIRE(reassembled == "hi!");
    }());

    rt.shutdown();
    REQUIRE(rt.join_tasks(5000ms));
  }
}

TEST_CASE("clients observe registration and disappearance", "[endpoint][watch]") {
  auto rt = Runtime::create(test_config());
  {
    auto drt = component::DistributedRuntime::create(rt, {});
    auto endpoint = drt.ns(unique_ns()).component("backend").endpoint("generate");

    coro::sync_wait([&]() -> coro::Task<void> {
      co_await rt.primary().schedule();
      auto client = co_await endpoint.client<std::string, std::string>();
      REQUIRE(client.instance_ids().empty());

      // Serve under an explicit lease so we can end just this instance.
      auto lease = co_await drt.discovery()->create_lease(10s);
      rt.spawn(endpoint.serve<std::string, std::string>(std::make_shared<EchoEngine>(), {.lease = lease}));

      co_await client.wait_for_instances();
      auto ids = client.instance_ids();
      REQUIRE(ids.size() == 1);
      REQUIRE(ids[0] == lease.id);

      lease.revoke();
      for (int i = 0; i < 100 && !client.instance_ids().empty(); ++i) {
        std::this_thread::sleep_for(10ms);
      }
      REQUIRE(client.instance_ids().empty());
    }());

    rt.shutdown();
    REQUIRE(rt.join_tasks(5000ms));
  }
}

TEST_CASE("round-robin and direct routing over two instances", "[endpoint][routing]") {
  auto rt = Runtime::create(test_config());
  {
    auto drt = component::DistributedRuntime::create(rt, {});
    auto endpoint = drt.ns(unique_ns()).component("backend").endpoint("generate");

    coro::sync_wait([&]() -> coro::Task<void> {
      co_await rt.primary().schedule();

      auto lease_a = co_await drt.discovery()->create_lease(10s);
      auto lease_b = co_await drt.discovery()->create_lease(10s);
      rt.spawn(endpoint.serve<std::string, int64_t>(
          std::make_shared<LabelEngine>(lease_a.id), {.lease = lease_a}));
      rt.spawn(endpoint.serve<std::string, int64_t>(
          std::make_shared<LabelEngine>(lease_b.id), {.lease = lease_b}));

      auto client = co_await endpoint.client<std::string, int64_t>();
      co_await client.wait_for_instances();
      for (int i = 0; i < 100 && client.instance_ids().size() < 2; ++i) {
        std::this_thread::sleep_for(10ms);
      }
      REQUIRE(client.instance_ids().size() == 2);

      auto one_reply = [&](coro::Task<pipeline::ManyOut<int64_t>> call) -> coro::Task<int64_t> {
        auto stream = co_await std::move(call);
        auto item = co_await stream.next();
        REQUIRE(item.has_value());
        while (co_await stream.next()) {
        }
        co_return *item;
      };

      // Round-robin alternates strictly between the two instances.
      std::vector<int64_t> order;
      for (int i = 0; i < 6; ++i) {
        order.push_back(
            co_await one_reply(client.round_robin(pipeline::SingleIn<std::string>("r"))));
      }
      REQUIRE(order[0] != order[1]);
      for (int i = 2; i < 6; ++i) REQUIRE(order[i] == order[i - 2]);

      // Random hits only live instances (and answers correctly).
      std::set<int64_t> seen;
      for (int i = 0; i < 20; ++i) {
        seen.insert(co_await one_reply(client.random(pipeline::SingleIn<std::string>("r"))));
      }
      std::set<int64_t> live{lease_a.id, lease_b.id};
      for (auto id : seen) REQUIRE(live.count(id) == 1);

      // Direct reaches the chosen instance; unknown ids fail.
      REQUIRE(co_await one_reply(client.direct(pipeline::SingleIn<std::string>("r"),
                                               lease_b.id)) == lease_b.id);
      REQUIRE_THROWS(co_await client.direct(pipeline::SingleIn<std::string>("r"), 424242));
    }());

    rt.shutdown();
    REQUIRE(rt.join_tasks(5000ms));
  }
}

TEST_CASE("client stop propagates to the worker engine", "[endpoint][cancellation]") {
  auto rt = Runtime::create(test_config());
  {
    auto drt = component::DistributedRuntime::create(rt, {});
    auto endpoint = drt.ns(unique_ns()).component("backend").endpoint("generate");
    auto engine = std::make_shared<EndlessEngine>();

    coro::sync_wait([&]() -> coro::Task<void> {
      co_await rt.primary().schedule();
      rt.spawn(endpoint.serve<std::string, int>(engine));

      auto client = co_await endpoint.client<std::string, int>();
      co_await client.wait_for_instances();

      auto stream = co_await client.generate(pipeline::SingleIn<std::string>("go"));
      int received = 0;
      while (auto item = co_await stream.next()) {
        if (++received == 5) stream.context()->stop_generating();
      }
      // The stream terminated even though the engine was endless.
      REQUIRE(received >= 5);
    }());

    for (int i = 0; i < 200 && !engine->finished; ++i) std::this_thread::sleep_for(10ms);
    REQUIRE(engine->finished);

    rt.shutdown();
    REQUIRE(rt.join_tasks(5000ms));
  }
}

TEST_CASE("abandoning the stream kills worker-side generation", "[endpoint][cancellation]") {
  auto rt = Runtime::create(test_config());
  {
    auto drt = component::DistributedRuntime::create(rt, {});
    auto endpoint = drt.ns(unique_ns()).component("backend").endpoint("generate");
    auto engine = std::make_shared<EndlessEngine>();

    coro::sync_wait([&]() -> coro::Task<void> {
      co_await rt.primary().schedule();
      rt.spawn(endpoint.serve<std::string, int>(engine));

      auto client = co_await endpoint.client<std::string, int>();
      co_await client.wait_for_instances();

      {
        auto stream = co_await client.generate(pipeline::SingleIn<std::string>("go"));
        auto first = co_await stream.next();
        REQUIRE(first.has_value());
        // Dropped here without draining → kill must reach the worker.
      }
      co_return;
    }());

    for (int i = 0; i < 200 && !engine->finished; ++i) std::this_thread::sleep_for(10ms);
    REQUIRE(engine->finished);

    rt.shutdown();
    REQUIRE(rt.join_tasks(5000ms));
  }
}

TEST_CASE("worker-side generate failure fails the call via the prologue", "[endpoint]") {
  auto rt = Runtime::create(test_config());
  {
    auto drt = component::DistributedRuntime::create(rt, {});
    auto endpoint = drt.ns(unique_ns()).component("backend").endpoint("generate");

    coro::sync_wait([&]() -> coro::Task<void> {
      co_await rt.primary().schedule();
      rt.spawn(endpoint.serve<std::string, std::string>(std::make_shared<FailingEngine>()));

      auto client = co_await endpoint.client<std::string, std::string>();
      co_await client.wait_for_instances();

      REQUIRE_THROWS_WITH(co_await client.generate(pipeline::SingleIn<std::string>("x")),
                          Catch::Matchers::ContainsSubstring("engine exploded"));
    }());

    rt.shutdown();
    REQUIRE(rt.join_tasks(5000ms));
  }
}

TEST_CASE("unary call returns a single value from a streaming endpoint", "[endpoint][unary]") {
  auto rt = Runtime::create(test_config());
  {
    auto drt = component::DistributedRuntime::create(rt, {});
    auto endpoint = drt.ns(unique_ns()).component("backend").endpoint("generate");

    coro::sync_wait([&]() -> coro::Task<void> {
      co_await rt.primary().schedule();

      // A unary-shaped engine built with the adapter.
      auto reverse = pipeline::make_unary_engine<std::string, std::string>(
          [](pipeline::SingleIn<std::string> in) -> coro::Task<std::string> {
            auto [data, controller] = std::move(in).into_parts();
            std::reverse(data.begin(), data.end());
            co_return data;
          });
      rt.spawn(endpoint.serve<std::string, std::string>(reverse));

      auto client = co_await endpoint.client<std::string, std::string>();
      co_await client.wait_for_instances();

      auto value = co_await client.unary(pipeline::SingleIn<std::string>("abc"));
      REQUIRE(value == "cba");
    }());

    rt.shutdown();
    REQUIRE(rt.join_tasks(5000ms));
  }
}

TEST_CASE("unary call against a streaming engine takes the first item", "[endpoint][unary]") {
  auto rt = Runtime::create(test_config());
  {
    auto drt = component::DistributedRuntime::create(rt, {});
    auto endpoint = drt.ns(unique_ns()).component("backend").endpoint("generate");

    coro::sync_wait([&]() -> coro::Task<void> {
      co_await rt.primary().schedule();
      // EchoEngine streams one item per character; single_out must cut the
      // stream after the first.
      rt.spawn(endpoint.serve<std::string, std::string>(std::make_shared<EchoEngine>()));

      auto client = co_await endpoint.client<std::string, std::string>();
      co_await client.wait_for_instances();

      auto value = co_await client.unary(pipeline::SingleIn<std::string>("hi!"));
      REQUIRE(value == "h");
    }());

    rt.shutdown();
    REQUIRE(rt.join_tasks(5000ms));
  }
}

TEST_CASE("component-wide stats scrape aggregates all instances", "[endpoint][stats]") {
  auto rt = Runtime::create(test_config());
  {
    auto drt = component::DistributedRuntime::create(rt, {});
    auto component = drt.ns(unique_ns()).component("backend");
    auto endpoint = component.endpoint("generate");

    coro::sync_wait([&]() -> coro::Task<void> {
      co_await rt.primary().schedule();

      auto lease_a = co_await drt.discovery()->create_lease(10s);
      auto lease_b = co_await drt.discovery()->create_lease(10s);
      rt.spawn(endpoint.serve<std::string, int64_t>(
          std::make_shared<LabelEngine>(lease_a.id), {.lease = lease_a}));
      rt.spawn(endpoint.serve<std::string, int64_t>(
          std::make_shared<LabelEngine>(lease_b.id), {.lease = lease_b}));

      auto client = co_await endpoint.client<std::string, int64_t>();
      co_await client.wait_for_instances();
      for (int i = 0; i < 100 && client.instance_ids().size() < 2; ++i) {
        std::this_thread::sleep_for(10ms);
      }
      REQUIRE(client.instance_ids().size() == 2);

      // Issue three requests round-robin, then scrape the whole component.
      for (int i = 0; i < 3; ++i) {
        auto stream = co_await client.round_robin(pipeline::SingleIn<std::string>("r"));
        while (co_await stream.next()) {
        }
      }

      auto set = co_await component.scrape_stats(2000ms);
      REQUIRE(set.endpoints.size() == 2);
      int total_requests = 0;
      for (auto& entry : set.endpoints) {
        REQUIRE(entry.ok);
        REQUIRE(entry.info.component == "backend");
        total_requests += entry.stats.at("requests").get<int>();
      }
      REQUIRE(total_requests == 3);
    }());

    rt.shutdown();
    REQUIRE(rt.join_tasks(5000ms));
  }
}

TEST_CASE("instance watch closes when the last client drops", "[endpoint][watch]") {
  auto rt = Runtime::create(test_config());
  {
    auto drt = component::DistributedRuntime::create(rt, {});
    auto endpoint = drt.ns(unique_ns()).component("backend").endpoint("generate");

    coro::sync_wait([&]() -> coro::Task<void> {
      co_await rt.primary().schedule();
      std::weak_ptr<component::InstanceSource> weak_source;
      {
        auto client = co_await endpoint.client<std::string, std::string>();
        auto source = co_await drt.instance_source(endpoint);
        weak_source = source;
        REQUIRE(weak_source.lock() != nullptr);
      }
      // Both holders (client + local ref) gone: source destroyed, watch ends.
      for (int i = 0; i < 100 && weak_source.lock(); ++i) {
        std::this_thread::sleep_for(10ms);
      }
      REQUIRE(weak_source.lock() == nullptr);

      // A fresh client re-creates the source and works normally.
      rt.spawn(endpoint.serve<std::string, std::string>(std::make_shared<EchoEngine>()));
      auto client = co_await endpoint.client<std::string, std::string>();
      co_await client.wait_for_instances();
      REQUIRE(client.instance_ids().size() == 1);
    }());

    rt.shutdown();
    REQUIRE(rt.join_tasks(5000ms));
  }
}

TEST_CASE("serve failures are observable through the task handle", "[endpoint][handles]") {
  auto rt = Runtime::create(test_config());
  {
    auto drt = component::DistributedRuntime::create(rt, {});
    auto endpoint = drt.ns(unique_ns()).component("backend").endpoint("generate");

    coro::sync_wait([&]() -> coro::Task<void> {
      co_await rt.primary().schedule();
      auto lease = co_await drt.discovery()->create_lease(10s);

      auto first = rt.spawn(
          endpoint.serve<std::string, std::string>(std::make_shared<EchoEngine>(),
                                                   {.lease = lease}));
      auto client = co_await endpoint.client<std::string, std::string>();
      co_await client.wait_for_instances();

      // Same endpoint + same lease ⇒ same instance key: registration fails,
      // and the failure surfaces on the handle instead of being lost.
      auto second = rt.spawn(
          endpoint.serve<std::string, std::string>(std::make_shared<EchoEngine>(),
                                                   {.lease = lease}));
      REQUIRE_THROWS(co_await second.join());
      REQUIRE_FALSE(first.finished());

      lease.revoke();
      co_await first.join();  // clean exit after revocation
    }());

    rt.shutdown();
    REQUIRE(rt.join_tasks(5000ms));
  }
}

TEST_CASE("operators compose with a remote endpoint (distributed segment)",
          "[endpoint][operators]") {
  auto rt = Runtime::create(test_config());
  {
    auto drt = component::DistributedRuntime::create(rt, {});
    auto endpoint = drt.ns(unique_ns()).component("backend").endpoint("generate");

    coro::sync_wait([&]() -> coro::Task<void> {
      co_await rt.primary().schedule();
      rt.spawn(endpoint.serve<std::string, std::string>(std::make_shared<EchoEngine>()));

      auto client = co_await endpoint.client<std::string, std::string>();
      co_await client.wait_for_instances();

      // Caller-side operator in front of the network segment: uppercase the
      // request downstream, bracket each streamed item upstream.
      auto op =
          pipeline::make_map_operator<std::string, std::string, std::string, std::string>(
              "front",
              [](std::string req) {
                std::transform(req.begin(), req.end(), req.begin(),
                               [](unsigned char c) { return std::toupper(c); });
                return req;
              },
              [](std::string item) { return "[" + item + "]"; });
      auto engine = pipeline::link(op, client.as_engine());

      auto out = co_await engine->generate(pipeline::SingleIn<std::string>("ab"));
      std::string collected;
      while (auto item = co_await out.next()) collected += *item;
      REQUIRE(collected == "[A][B]");
    }());

    rt.shutdown();
    REQUIRE(rt.join_tasks(5000ms));
  }
}

TEST_CASE("component events reach subscribers", "[endpoint][events]") {
  auto rt = Runtime::create(test_config());
  {
    auto drt = component::DistributedRuntime::create(rt, {});
    auto ns = drt.ns(unique_ns());
    auto component = ns.component("backend");

    coro::sync_wait([&]() -> coro::Task<void> {
      co_await rt.primary().schedule();

      auto component_stream = co_await component.subscribe("kv-cache");
      auto ns_stream = co_await ns.subscribe("announce");

      co_await component.publish("kv-cache", {{"blocks", 3}});
      co_await ns.publish("announce", {{"model", "m1"}});

      auto ce = co_await component_stream.events.recv();
      REQUIRE(ce);
      REQUIRE(nlohmann::json::parse(ce->payload).at("blocks") == 3);

      auto ne = co_await ns_stream.events.recv();
      REQUIRE(ne);
      REQUIRE(nlohmann::json::parse(ne->payload).at("model") == "m1");
    }());

    rt.shutdown();
    REQUIRE(rt.join_tasks(5000ms));
  }
}

TEST_CASE("registered description and version are advertised", "[endpoint][metadata]") {
  auto rt = Runtime::create(test_config());
  {
    auto drt = component::DistributedRuntime::create(rt, {});
    auto endpoint = drt.ns(unique_ns()).component("backend").endpoint("generate");

    coro::sync_wait([&]() -> coro::Task<void> {
      co_await rt.primary().schedule();
      rt.spawn(endpoint.serve<std::string, std::string>(
          std::make_shared<EchoEngine>(),
          {.description = "test echo service", .version = "9.9.9"}));

      auto client = co_await endpoint.client<std::string, std::string>();
      co_await client.wait_for_instances();

      auto instances = client.instances();
      REQUIRE(instances.size() == 1);
      REQUIRE(instances[0].description == "test echo service");
      REQUIRE(instances[0].version == "9.9.9");
    }());

    rt.shutdown();
    REQUIRE(rt.join_tasks(5000ms));
  }
}

TEST_CASE("arrival timeout fails the call when a worker acks but never calls home",
          "[endpoint][timeouts]") {
  auto rt = Runtime::create(test_config());
  {
    auto drt = component::DistributedRuntime::create(rt, {});
    auto endpoint = drt.ns(unique_ns()).component("backend").endpoint("generate");

    coro::sync_wait([&]() -> coro::Task<void> {
      co_await rt.primary().schedule();

      // A rogue "instance": acks control-plane dispatches but never opens the
      // data-plane return stream.
      struct BlackHole final : transports::PushHandler {
        coro::Task<void> handle(transports::RequestControlMessage, std::string) override {
          co_return;
        }
      };
      auto lease = co_await drt.discovery()->create_lease(10s);
      auto control_plane = drt.control_plane();
      std::string subject = endpoint.subject_for(lease.id);
      control_plane->register_handler(subject, std::make_shared<BlackHole>());

      component::EndpointInfo info;
      info.namespace_name = endpoint.component().ns().name();
      info.component = endpoint.component().name();
      info.endpoint = endpoint.name();
      info.instance_id = lease.id;
      info.address = control_plane->address();
      info.subject = subject;
      co_await drt.discovery()->kv_create(endpoint.instance_key(lease.id),
                                          nlohmann::json(info).dump(), lease.id);

      component::CallOptions options;
      options.arrival_timeout = std::chrono::milliseconds(300);
      auto client = co_await endpoint.client<std::string, std::string>(options);
      co_await client.wait_for_instances();

      auto start = std::chrono::steady_clock::now();
      REQUIRE_THROWS_WITH(co_await client.generate(pipeline::SingleIn<std::string>("x")),
                          Catch::Matchers::ContainsSubstring("call-home"));
      auto elapsed = std::chrono::steady_clock::now() - start;
      REQUIRE(elapsed < 5s);  // bounded by the option, not hanging

      control_plane->unregister_handler(subject);
    }());

    rt.shutdown();
    REQUIRE(rt.join_tasks(5000ms));
  }
}

TEST_CASE("many_in requests are rejected via the prologue", "[endpoint][manyin]") {
  auto rt = Runtime::create(test_config());
  {
    auto drt = component::DistributedRuntime::create(rt, {});
    auto endpoint = drt.ns(unique_ns()).component("backend").endpoint("generate");

    coro::sync_wait([&]() -> coro::Task<void> {
      co_await rt.primary().schedule();
      rt.spawn(endpoint.serve<std::string, std::string>(std::make_shared<EchoEngine>()));

      auto client = co_await endpoint.client<std::string, std::string>();
      co_await client.wait_for_instances();
      auto target = client.instances().at(0);

      // Hand-craft a dispatch with the unsupported request_type.
      auto controller = std::make_shared<pipeline::Controller>();
      pipeline::ContextPtr ctx = controller;
      auto registered = drt.data_plane()->register_response_stream(ctx);

      transports::DispatchHeader header;
      header.subject = target.subject;
      header.control.id = ctx->id();
      header.control.request_type = "many_in";
      header.control.connection_info = std::move(registered.info);
      transports::dispatch_request(target.address, header, "\"x\"");

      auto arrival = co_await registered.arrival.recv();
      REQUIRE(arrival);
      REQUIRE(arrival->error);
      REQUIRE(arrival->error->find("not supported") != std::string::npos);
    }());

    rt.shutdown();
    REQUIRE(rt.join_tasks(5000ms));
  }
}

TEST_CASE("component list_instances enumerates every endpoint", "[endpoint][list]") {
  auto rt = Runtime::create(test_config());
  {
    auto drt = component::DistributedRuntime::create(rt, {});
    auto component = drt.ns(unique_ns()).component("backend");

    coro::sync_wait([&]() -> coro::Task<void> {
      co_await rt.primary().schedule();
      auto lease_a = co_await drt.discovery()->create_lease(10s);
      auto lease_b = co_await drt.discovery()->create_lease(10s);
      rt.spawn(component.endpoint("generate").serve<std::string, std::string>(
          std::make_shared<EchoEngine>(), {.lease = lease_a}));
      rt.spawn(component.endpoint("tokenize").serve<std::string, std::string>(
          std::make_shared<EchoEngine>(), {.lease = lease_b}));

      // Wait for both registrations.
      for (int i = 0; i < 100; ++i) {
        if ((co_await component.list_instances()).size() == 2) break;
        std::this_thread::sleep_for(10ms);
      }
      auto instances = co_await component.list_instances();
      REQUIRE(instances.size() == 2);
      std::set<std::string> endpoints;
      for (auto& info : instances) endpoints.insert(info.endpoint);
      REQUIRE(endpoints == std::set<std::string>{"generate", "tokenize"});
    }());

    rt.shutdown();
    REQUIRE(rt.join_tasks(5000ms));
  }
}

TEST_CASE("malformed control-plane frames do not take the server down",
          "[endpoint][hardening]") {
  auto rt = Runtime::create(test_config());
  {
    auto drt = component::DistributedRuntime::create(rt, {});
    auto endpoint = drt.ns(unique_ns()).component("backend").endpoint("generate");

    coro::sync_wait([&]() -> coro::Task<void> {
      co_await rt.primary().schedule();
      rt.spawn(endpoint.serve<std::string, std::string>(std::make_shared<EchoEngine>()));

      auto client = co_await endpoint.client<std::string, std::string>();
      co_await client.wait_for_instances();
      auto target = client.instances().at(0);

      // Raw garbage at the control plane: the connection dies, the server
      // survives.
      auto [host, port] = transports::parse_address(target.address);
      {
        auto sock = transports::Socket::connect(host, port);
        REQUIRE(sock);
        REQUIRE(sock->write_all("this is not a two-part frame at all........."));
      }

      // Valid requests still work afterwards.
      auto stream = co_await client.generate(pipeline::SingleIn<std::string>("ok"));
      std::string collected;
      while (auto item = co_await stream.next()) collected += *item;
      REQUIRE(collected == "ok");
    }());

    rt.shutdown();
    REQUIRE(rt.join_tasks(5000ms));
  }
}

TEST_CASE("stats query reports request counts", "[endpoint][stats]") {
  auto rt = Runtime::create(test_config());
  {
    auto drt = component::DistributedRuntime::create(rt, {});
    auto endpoint = drt.ns(unique_ns()).component("backend").endpoint("generate");

    coro::sync_wait([&]() -> coro::Task<void> {
      co_await rt.primary().schedule();
      auto lease = co_await drt.discovery()->create_lease(10s);
      rt.spawn(endpoint.serve<std::string, std::string>(std::make_shared<EchoEngine>(), {.lease = lease}));

      auto client = co_await endpoint.client<std::string, std::string>();
      co_await client.wait_for_instances();

      auto stream = co_await client.generate(pipeline::SingleIn<std::string>("abc"));
      while (co_await stream.next()) {
      }

      auto stats = co_await client.scrape_stats(lease.id);
      REQUIRE(stats.at("requests").get<int>() == 1);
      REQUIRE(stats.at("errors").get<int>() == 0);
    }());

    rt.shutdown();
    REQUIRE(rt.join_tasks(5000ms));
  }
}
