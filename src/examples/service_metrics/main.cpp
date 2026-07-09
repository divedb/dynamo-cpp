// SPDX-License-Identifier: Apache-2.0
//
// service_metrics (port of Dynamo's example): serves an endpoint with a
// custom stats handler, issues a batch of requests, then scrapes the whole
// component and prints the aggregated per-instance stats.

#include <cstdio>
#include <memory>
#include <string>

#include "component/component.h"
#include "runtime/logging.h"
#include "runtime/worker.h"

using namespace dynamo;

namespace {

coro::AsyncGenerator<std::string> echo_stream(std::string data) {
  co_yield data;
}

struct EchoOnce final : pipeline::AsyncEngine<std::string, std::string> {
  coro::Task<pipeline::ManyOut<std::string>> generate(
      pipeline::SingleIn<std::string> in) override {
    auto [data, controller] = std::move(in).into_parts();
    pipeline::ContextPtr ctx = controller;
    co_return pipeline::ManyOut<std::string>(echo_stream(std::move(data)), ctx);
  }
};

}  // namespace

int main() {
  logging::init();
  auto worker = Worker::from_env();
  return worker.execute([](Runtime rt) -> coro::Task<void> {
    auto drt = component::DistributedRuntime::create(rt);
    auto component = drt.ns("dynamo").component("metrics-demo");
    auto endpoint = component.endpoint("echo");

    component::ServeOptions options;
    options.description = "service_metrics demo endpoint";
    options.version = "1.2.3";
    options.stats_handler = [](const std::string&) {
      return nlohmann::json{{"queue_depth", 0}, {"gpu_util", 0.42}}.dump();
    };
    auto lease = co_await drt.discovery()->create_lease(std::chrono::seconds(10));
    options.lease = lease;
    rt.spawn(endpoint.serve<std::string, std::string>(std::make_shared<EchoOnce>(), options));

    auto client = co_await endpoint.client<std::string, std::string>();
    co_await client.wait_for_instances();

    for (int i = 0; i < 5; ++i) {
      auto stream =
          co_await client.generate(pipeline::SingleIn<std::string>("req " + std::to_string(i)));
      while (co_await stream.next()) {
      }
    }

    auto set = co_await component.scrape_stats(std::chrono::milliseconds(2000));
    for (auto& entry : set.endpoints) {
      printf("instance %s (%s, v%s): ok=%d stats=%s\n",
             component::hex_id(entry.info.instance_id).c_str(),
             entry.info.description.c_str(), entry.info.version.c_str(), entry.ok,
             entry.stats.dump().c_str());
    }
  });
}
