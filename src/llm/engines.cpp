// SPDX-License-Identifier: Apache-2.0

#include "llm/engines.h"

#include <thread>

namespace dynamo::llm {

namespace {

class EchoEngineCore final : public pipeline::AsyncEngine<BackendInput, ExecutionOutputStream> {
 public:
  explicit EchoEngineCore(EchoEngineOptions options) : options_(options) {}

  coro::Task<pipeline::ManyOut<ExecutionOutputStream>> generate(
      pipeline::SingleIn<BackendInput> request) override {
    auto [input, controller] = std::move(request).into_parts();
    pipeline::ContextPtr context = controller;
    co_return pipeline::ManyOut<ExecutionOutputStream>(
        echo(std::move(input), context, options_), std::move(context));
  }

 private:
  static coro::AsyncGenerator<ExecutionOutputStream> echo(BackendInput input,
                                                          pipeline::ContextPtr context,
                                                          EchoEngineOptions options) {
    uint32_t emitted = 0;
    uint32_t max_tokens =
        input.stop_conditions.max_tokens.value_or(static_cast<uint32_t>(~0u));
    for (TokenIdType token_id : input.token_ids) {
      if (context->is_stopped()) break;
      if (emitted >= max_tokens) {
        co_yield pipeline::Annotated<LLMEngineOutput>::from_data(LLMEngineOutput::length());
        co_return;
      }
      if (options.token_delay.count() > 0) std::this_thread::sleep_for(options.token_delay);
      LLMEngineOutput delta;
      delta.token_ids = {token_id};
      ++emitted;
      co_yield pipeline::Annotated<LLMEngineOutput>::from_data(std::move(delta));
    }
    co_yield pipeline::Annotated<LLMEngineOutput>::from_data(LLMEngineOutput::stop());
  }

  EchoEngineOptions options_;
};

}  // namespace

ExecutionContext make_echo_engine_core(EchoEngineOptions options) {
  return std::make_shared<EchoEngineCore>(options);
}

}  // namespace dynamo::llm
