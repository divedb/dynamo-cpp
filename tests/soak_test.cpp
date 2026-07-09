// SPDX-License-Identifier: Apache-2.0
//
// Short soak: concurrent streaming clients hammering two instances while a
// third instance churns (serve → revoke → serve). Success = no crashes, no
// hangs, and every request either completes or fails with a routing error.
// (Dynamo's tests/soak.rs role, sized to stay CI-friendly.)

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <string>

#include "component/component.h"
#include "runtime/coro/sync_wait.h"

using namespace dynamo;
using namespace std::chrono_literals;

namespace {

coro::AsyncGenerator<std::string> echo_chars_soak(std::string data) {
  for (char c : data) {
    std::string item(1, c);
    co_yield item;
  }
}

struct SoakEcho final : pipeline::AsyncEngine<std::string, std::string> {
  coro::Task<pipeline::ManyOut<std::string>> generate(
      pipeline::SingleIn<std::string> in) override {
    auto [data, controller] = std::move(in).into_parts();
    pipeline::ContextPtr ctx = controller;
    co_return pipeline::ManyOut<std::string>(echo_chars_soak(std::move(data)), ctx);
  }
};

}  // namespace

TEST_CASE("soak: concurrent streams under instance churn", "[soak]") {
  RuntimeConfig config;
  config.num_worker_threads = 8;
  config.num_background_threads = 2;
  auto rt = Runtime::create(config);
  {
    auto drt = component::DistributedRuntime::create(rt, {});
    auto endpoint = drt.ns("soakns").component("backend").endpoint("generate");

    constexpr int kClients = 8;
    constexpr int kRequestsPerClient = 25;

    std::atomic<int> completed{0};
    std::atomic<int> routing_errors{0};
    std::atomic<int> payload_mismatches{0};

    coro::sync_wait([&]() -> coro::Task<void> {
      co_await rt.primary().schedule();

      // Two stable instances.
      auto lease_a = co_await drt.discovery()->create_lease(30s);
      auto lease_b = co_await drt.discovery()->create_lease(30s);
      rt.spawn(endpoint.serve<std::string, std::string>(std::make_shared<SoakEcho>(),
                                                        {.lease = lease_a}));
      rt.spawn(endpoint.serve<std::string, std::string>(std::make_shared<SoakEcho>(),
                                                        {.lease = lease_b}));

      auto client = co_await endpoint.client<std::string, std::string>();
      co_await client.wait_for_instances();

      // Churner: a third instance repeatedly joins and leaves.
      auto churn = rt.spawn([](component::DistributedRuntime drt, component::Endpoint ep,
                               Runtime rt) -> coro::Task<void> {
        for (int i = 0; i < 6; ++i) {
          auto lease = co_await drt.discovery()->create_lease(30s);
          rt.spawn(ep.serve<std::string, std::string>(std::make_shared<SoakEcho>(),
                                                      {.lease = lease}));
          co_await rt.token().wait_for(rt.primary(), std::chrono::milliseconds(60));
          lease.revoke();
          co_await rt.token().wait_for(rt.primary(), std::chrono::milliseconds(40));
        }
      }(drt, endpoint, rt));

      // Client workers.
      std::vector<TaskHandle> workers;
      for (int w = 0; w < kClients; ++w) {
        workers.push_back(rt.spawn(
            [](component::Client<std::string, std::string> client, std::atomic<int>& done,
               std::atomic<int>& errors, std::atomic<int>& mismatches,
               int requests) -> coro::Task<void> {
              for (int i = 0; i < requests; ++i) {
                std::string payload = "soak-" + std::to_string(i);
                try {
                  auto stream = co_await client.round_robin(
                      pipeline::SingleIn<std::string>(payload));
                  std::string collected;
                  while (auto item = co_await stream.next()) collected += *item;
                  if (collected != payload) mismatches.fetch_add(1);
                  done.fetch_add(1);
                } catch (const std::exception&) {
                  // Racing a churned-away instance is legal; losing data is not.
                  errors.fetch_add(1);
                }
              }
            }(client, completed, routing_errors, payload_mismatches, kRequestsPerClient)));
      }

      for (auto& worker : workers) co_await worker.join();
      co_await churn.join();
    }());

    REQUIRE(payload_mismatches == 0);
    REQUIRE(completed + routing_errors == kClients * kRequestsPerClient);
    REQUIRE(completed > 0);

    rt.shutdown();
    REQUIRE(rt.join_tasks(10000ms));
  }
}
