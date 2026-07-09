// SPDX-License-Identifier: Apache-2.0
//
// hello_world client: waits for a live instance of dynamo/backend/generate,
// sends one request, prints the streamed response.

#include <cstdio>
#include <string>

#include "component/component.h"
#include "pipeline/annotated.h"
#include "runtime/logging.h"
#include "runtime/worker.h"

using namespace dynamo;
using Response = pipeline::Annotated<std::string>;

int main() {
  logging::init();
  auto worker = Worker::from_env();
  return worker.execute([](Runtime rt) -> coro::Task<void> {
    auto drt = component::DistributedRuntime::create(rt);
    auto endpoint = drt.ns("dynamo").component("backend").endpoint("generate");

    auto client = co_await endpoint.client<std::string, Response>();
    co_await client.wait_for_instances();

    auto stream = co_await client.random(pipeline::SingleIn<std::string>("hello world"));
    while (auto item = co_await stream.next()) {
      if (item->data) printf("%s", item->data->c_str());
      if (item->is_error()) printf("[error: %s]", item->comment->front().c_str());
    }
    printf("\n");
  });
}
