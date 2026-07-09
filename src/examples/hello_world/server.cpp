// SPDX-License-Identifier: Apache-2.0
//
// hello_world server: serves dynamo/backend/generate, streaming each
// character of the request back as an Annotated<std::string>.
//
// Single process: run with the in-process discovery (also run the client in
// the same process — see multi_instance). Multi process: start `discoveryd`,
// then run server and client with DYN_DISCOVERY=127.0.0.1:7787.

#include <memory>
#include <string>

#include "component/component.h"
#include "pipeline/annotated.h"
#include "runtime/logging.h"
#include "runtime/worker.h"

using namespace dynamo;
using Response = pipeline::Annotated<std::string>;

namespace {

coro::AsyncGenerator<Response> char_stream(std::string data, pipeline::ContextPtr ctx) {
  for (char c : data) {
    if (ctx->is_stopped()) break;
    auto item = Response::from_data(std::string(1, c));
    co_yield item;
  }
}

struct RequestHandler final : pipeline::AsyncEngine<std::string, Response> {
  coro::Task<pipeline::ManyOut<Response>> generate(pipeline::SingleIn<std::string> in) override {
    auto [data, controller] = std::move(in).into_parts();
    pipeline::ContextPtr ctx = controller;
    co_return pipeline::ManyOut<Response>(char_stream(std::move(data), ctx), ctx);
  }
};

}  // namespace

int main() {
  logging::init();
  auto worker = Worker::from_env();
  return worker.execute([](Runtime rt) -> coro::Task<void> {
    auto drt = component::DistributedRuntime::create(rt);
    auto endpoint = drt.ns("dynamo").component("backend").endpoint("generate");
    co_await endpoint.serve<std::string, Response>(std::make_shared<RequestHandler>());
  });
}
