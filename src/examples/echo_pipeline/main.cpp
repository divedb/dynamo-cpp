// SPDX-License-Identifier: Apache-2.0
//
// echo_pipeline: composable operators, locally and across the network.
// An uppercase operator is linked in front of a char-streaming echo backend;
// the same operator chain is then linked in front of a remote endpoint
// (Client::as_engine — the caller side of a distributed segment).

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>

#include "component/component.h"
#include "pipeline/operators.h"
#include "runtime/logging.h"
#include "runtime/worker.h"

using namespace dynamo;

namespace {

coro::AsyncGenerator<std::string> char_stream(std::string data) {
  for (char c : data) {
    std::string item(1, c);
    co_yield item;
  }
}

struct EchoChars final : pipeline::AsyncEngine<std::string, std::string> {
  coro::Task<pipeline::ManyOut<std::string>> generate(
      pipeline::SingleIn<std::string> in) override {
    auto [data, controller] = std::move(in).into_parts();
    pipeline::ContextPtr ctx = controller;
    co_return pipeline::ManyOut<std::string>(char_stream(std::move(data)), ctx);
  }
};

coro::Task<void> run_pipeline(const char* label,
                              pipeline::EnginePtr<std::string, std::string> engine) {
  auto out = co_await engine->generate(pipeline::SingleIn<std::string>("hello pipeline"));
  std::string collected;
  while (auto item = co_await out.next()) collected += *item;
  printf("%s: %s\n", label, collected.c_str());
}

}  // namespace

int main() {
  logging::init();
  auto worker = Worker::from_env();
  return worker.execute([](Runtime rt) -> coro::Task<void> {
    auto uppercase =
        pipeline::make_map_operator<std::string, std::string, std::string, std::string>(
            "uppercase",
            [](std::string req) {
              std::transform(req.begin(), req.end(), req.begin(),
                             [](unsigned char c) { return std::toupper(c); });
              return req;
            },
            [](std::string item) { return item; });

    // Local pipeline: operator → in-process engine.
    auto backend = pipeline::EnginePtr<std::string, std::string>(std::make_shared<EchoChars>());
    co_await run_pipeline("local ", pipeline::link(uppercase, backend));

    // Distributed pipeline: operator → network segment → served engine.
    auto drt = component::DistributedRuntime::create(rt);
    auto endpoint = drt.ns("dynamo").component("pipeline-demo").endpoint("echo");
    auto lease = co_await drt.discovery()->create_lease(std::chrono::seconds(10));
    component::ServeOptions options;
    options.lease = lease;
    rt.spawn(endpoint.serve<std::string, std::string>(std::make_shared<EchoChars>(), options));

    auto client = co_await endpoint.client<std::string, std::string>();
    co_await client.wait_for_instances();
    co_await run_pipeline("remote", pipeline::link(uppercase, client.as_engine()));
  });
}
