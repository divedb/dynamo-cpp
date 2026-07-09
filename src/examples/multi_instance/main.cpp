// SPDX-License-Identifier: Apache-2.0
//
// Multi-instance routing demo (single process, in-process discovery, real
// TCP planes): two instances of dynamo/backend/generate serve under separate
// leases; a client round-robins requests across them and prints which
// instance answered.

#include <cstdio>
#include <memory>
#include <string>

#include "component/component.h"
#include "pipeline/annotated.h"
#include "runtime/logging.h"
#include "runtime/worker.h"

using namespace dynamo;
using Response = pipeline::Annotated<std::string>;

namespace {

coro::AsyncGenerator<Response> reply_stream(std::string data, std::string label) {
  auto item = Response::from_data("[" + label + "] " + data);
  co_yield item;
}

struct LabelledHandler final : pipeline::AsyncEngine<std::string, Response> {
  explicit LabelledHandler(std::string label) : label(std::move(label)) {}

  coro::Task<pipeline::ManyOut<Response>> generate(pipeline::SingleIn<std::string> in) override {
    auto [data, controller] = std::move(in).into_parts();
    pipeline::ContextPtr ctx = controller;
    co_return pipeline::ManyOut<Response>(reply_stream(std::move(data), label), ctx);
  }

  std::string label;
};

}  // namespace

int main() {
  logging::init();
  auto worker = Worker::from_env();
  return worker.execute([](Runtime rt) -> coro::Task<void> {
    auto drt = component::DistributedRuntime::create(rt);
    auto endpoint = drt.ns("dynamo").component("backend").endpoint("generate");

    // Two instances, each with its own lease (= its own instance id).
    for (auto* label : {"instance-a", "instance-b"}) {
      auto lease = co_await drt.discovery()->create_lease(std::chrono::seconds(10));
      rt.spawn(endpoint.serve<std::string, Response>(
          std::make_shared<LabelledHandler>(label), {.lease = lease}));
    }

    auto client = co_await endpoint.client<std::string, Response>();
    co_await client.wait_for_instances();

    for (int i = 0; i < 6; ++i) {
      auto stream =
          co_await client.round_robin(pipeline::SingleIn<std::string>("request " + std::to_string(i)));
      while (auto item = co_await stream.next()) {
        if (item->data) printf("%s\n", item->data->c_str());
      }
    }
  });
}
