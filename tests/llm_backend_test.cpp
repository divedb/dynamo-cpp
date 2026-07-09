// SPDX-License-Identifier: Apache-2.0
//
// M3 tests: the Decoder (stop tokens/sequences, min_tokens, jail windows),
// the Backend operator over an inline engine, the echo engine, and the full
// local pipeline: preprocessor -> backend -> echo engine.

#include <catch2/catch_test_macros.hpp>

#include <memory>

#include "llm/backend.h"
#include "llm/engines.h"
#include "llm/preprocessor.h"
#include "runtime/coro/sync_wait.h"

using namespace dynamo::llm;
using dynamo::pipeline::Annotated;
namespace pipeline = dynamo::pipeline;
namespace coro = dynamo::coro;

namespace {

TokenizerPtr byte_tokenizer() { return std::make_shared<ByteLevelTokenizer>(); }

std::vector<TokenIdType> ids_of(const std::string& text) {
  return ByteLevelTokenizer().encode(text).token_ids;
}

}  // namespace

// ---------------------------------------------------------------------------
// Decoder

TEST_CASE("decoder: hidden stop token cuts with reason stop", "[llm][backend]") {
  StopConditions sc;
  sc.stop_token_ids_hidden = std::vector<TokenIdType>{ByteLevelTokenizer::kEosId};
  Decoder decoder(byte_tokenizer(), sc);

  auto ids = ids_of("hi");
  ids.push_back(ByteLevelTokenizer::kEosId);
  auto result = decoder.process_token_ids(ids);

  CHECK(result.text == "hi");
  REQUIRE(result.stop_trigger.has_value());
  CHECK(result.stop_trigger->kind == StopTrigger::Kind::hidden_stop_token);
  CHECK(result.stop_trigger->finish_reason() == FinishReason::stop());
  CHECK(result.tokens.size() == 3);
}

TEST_CASE("decoder: hidden stop sequence hides matched text", "[llm][backend]") {
  StopConditions sc;
  sc.stop = std::vector<std::string>{"ox"};
  Decoder decoder(byte_tokenizer(), sc);

  // Stop "ox" arrives inside "boxes": only "b" survives from that step.
  auto result = decoder.process_token_ids(ids_of("boxes"));
  REQUIRE(result.stop_trigger.has_value());
  CHECK(result.stop_trigger->kind == StopTrigger::Kind::hidden_stop_sequence);
  CHECK(result.stop_trigger->sequence == "ox");
  CHECK(result.text == "b");
}

TEST_CASE("decoder: stop sequence spanning steps", "[llm][backend]") {
  StopConditions sc;
  sc.stop = std::vector<std::string>{"END"};
  Decoder decoder(byte_tokenizer(), sc);

  auto first = decoder.process_token_ids(ids_of("value: E"));
  CHECK_FALSE(first.stop_trigger.has_value());
  CHECK(first.text == "value: ");  // the "E" is jailed as a possible stop prefix

  auto second = decoder.process_token_ids(ids_of("ND"));
  REQUIRE(second.stop_trigger.has_value());
  CHECK(second.stop_trigger->sequence == "END");
  CHECK_FALSE(second.text.has_value());  // the whole "END" stays hidden
}

TEST_CASE("decoder: jailed text is released when it stops matching", "[llm][backend]") {
  StopConditions sc;
  sc.stop = std::vector<std::string>{"END"};
  Decoder decoder(byte_tokenizer(), sc);

  auto result = decoder.process_token_ids(ids_of("EN"));
  CHECK_FALSE(result.text.has_value());  // jailed
  result = decoder.process_token_ids(ids_of("!"));
  CHECK(result.text == "EN!");  // released: "EN!" cannot start "END"

  // On flush, held text that never resolved is returned.
  result = decoder.process_token_ids(ids_of("E"));
  CHECK_FALSE(result.text.has_value());
  CHECK(decoder.flush() == "E");
}

TEST_CASE("decoder: min_tokens suppresses stops", "[llm][backend]") {
  StopConditions sc;
  sc.stop_token_ids_hidden = std::vector<TokenIdType>{'x'};
  sc.min_tokens = 3;
  Decoder decoder(byte_tokenizer(), sc);

  auto result = decoder.process_token_ids(ids_of("xx"));
  CHECK_FALSE(result.stop_trigger.has_value());
  CHECK(result.text == "xx");

  result = decoder.process_token_ids(ids_of("x"));
  REQUIRE(result.stop_trigger.has_value());  // third token: stops apply now
}

// ---------------------------------------------------------------------------
// Backend operator + echo engine

TEST_CASE("backend over echo engine detokenizes and stops", "[llm][backend][operator]") {
  auto backend = std::make_shared<Backend>(byte_tokenizer(), "mdc-sum");
  auto engine = pipeline::link<BackendInput, Annotated<BackendOutput>, BackendInput,
                               Annotated<LLMEngineOutput>>(backend, make_echo_engine_core());

  BackendInput input;
  input.token_ids = ids_of("hello ⚡ world");
  input.stop_conditions.stop_token_ids_hidden =
      std::vector<TokenIdType>{ByteLevelTokenizer::kEosId};
  input.mdc_sum = "mdc-sum";

  pipeline::SingleIn<BackendInput> request(input);
  auto controller = request.controller();

  auto outputs = coro::sync_wait([&]() -> coro::Task<std::vector<Annotated<BackendOutput>>> {
    std::vector<Annotated<BackendOutput>> collected;
    auto out = co_await engine->generate(std::move(request));
    while (auto item = co_await out.next()) collected.push_back(std::move(*item));
    co_return collected;
  }());

  // Echo emits one delta per token + a stop delta; multi-byte UTF-8 tokens
  // decode to empty and text arrives on the completing byte.
  std::string text;
  std::optional<FinishReason> final_reason;
  for (const auto& item : outputs) {
    REQUIRE(item.data.has_value());
    if (item.data->text.has_value()) text += *item.data->text;
    if (item.data->finish_reason.has_value()) final_reason = item.data->finish_reason;
    CHECK(item.data->mdcsum == "mdc-sum");
  }
  CHECK(text == "hello ⚡ world");
  REQUIRE(final_reason.has_value());
  CHECK(*final_reason == FinishReason::stop());
}

TEST_CASE("backend cuts the stream on a stop sequence", "[llm][backend][operator]") {
  auto backend = std::make_shared<Backend>(byte_tokenizer(), "sum");
  auto engine = pipeline::link<BackendInput, Annotated<BackendOutput>, BackendInput,
                               Annotated<LLMEngineOutput>>(backend, make_echo_engine_core());

  BackendInput input;
  input.token_ids = ids_of("say STOP then keep going");
  input.stop_conditions.stop = std::vector<std::string>{"STOP"};

  pipeline::SingleIn<BackendInput> request(input);
  auto controller = request.controller();

  auto outputs = coro::sync_wait([&]() -> coro::Task<std::vector<Annotated<BackendOutput>>> {
    std::vector<Annotated<BackendOutput>> collected;
    auto out = co_await engine->generate(std::move(request));
    while (auto item = co_await out.next()) collected.push_back(std::move(*item));
    co_return collected;
  }());

  std::string text;
  std::optional<FinishReason> final_reason;
  for (const auto& item : outputs) {
    if (item.data->text.has_value()) text += *item.data->text;
    if (item.data->finish_reason.has_value()) final_reason = item.data->finish_reason;
  }
  CHECK(text == "say ");  // nothing at or after the stop sequence is emitted
  REQUIRE(final_reason.has_value());
  CHECK(*final_reason == FinishReason::stop());
  // The backend told the engine to stop generating.
  CHECK(controller->is_stopped());
}

TEST_CASE("echo engine honors max_tokens with a length finish", "[llm][backend][operator]") {
  auto backend = std::make_shared<Backend>(byte_tokenizer(), "sum");
  auto engine = pipeline::link<BackendInput, Annotated<BackendOutput>, BackendInput,
                               Annotated<LLMEngineOutput>>(backend, make_echo_engine_core());

  BackendInput input;
  input.token_ids = ids_of("abcdef");
  input.stop_conditions.max_tokens = 3;

  auto outputs = coro::sync_wait([&]() -> coro::Task<std::vector<Annotated<BackendOutput>>> {
    std::vector<Annotated<BackendOutput>> collected;
    auto out = co_await engine->generate(pipeline::SingleIn<BackendInput>(input));
    while (auto item = co_await out.next()) collected.push_back(std::move(*item));
    co_return collected;
  }());

  std::string text;
  std::optional<FinishReason> final_reason;
  for (const auto& item : outputs) {
    if (item.data->text.has_value()) text += *item.data->text;
    if (item.data->finish_reason.has_value()) final_reason = item.data->finish_reason;
  }
  CHECK(text == "abc");
  REQUIRE(final_reason.has_value());
  // The engine-issued Length survives the backend (deviation from Rust,
  // which drops it for token-only engines).
  CHECK(*final_reason == FinishReason::length());
}

// ---------------------------------------------------------------------------
// Full local pipeline: preprocessor -> backend -> echo

TEST_CASE("openai chat request through the full local pipeline", "[llm][backend][pipeline]") {
  const char* tmpl =
      "{% for message in messages %}{{ message.content }}{% endfor %}"
      "{% if add_generation_prompt %}!{% endif %}";
  auto formatter = std::make_shared<HfChatTemplateFormatter>(tmpl, "", "");
  auto tokenizer = byte_tokenizer();

  auto preprocessor = std::make_shared<OpenAIPreprocessor>(
      formatter, tokenizer, std::vector<TokenIdType>{ByteLevelTokenizer::kEosId}, "sum");
  auto backend = std::make_shared<Backend>(tokenizer, "sum");

  auto engine_core = make_echo_engine_core();
  auto backend_engine =
      pipeline::link<BackendInput, Annotated<BackendOutput>, BackendInput,
                     Annotated<LLMEngineOutput>>(backend, engine_core);
  auto chat_engine =
      pipeline::link<openai::NvCreateChatCompletionRequest,
                     Annotated<openai::NvCreateChatCompletionStreamResponse>, BackendInput,
                     Annotated<BackendOutput>>(preprocessor, backend_engine);

  openai::NvCreateChatCompletionRequest request;
  request.model = "echo-model";
  request.messages.push_back({.role = "user", .content = "hola"});

  auto chunks = coro::sync_wait(
      [&]() -> coro::Task<
                std::vector<Annotated<openai::NvCreateChatCompletionStreamResponse>>> {
        std::vector<Annotated<openai::NvCreateChatCompletionStreamResponse>> collected;
        auto out = co_await chat_engine->generate(
            pipeline::SingleIn<openai::NvCreateChatCompletionRequest>(request));
        while (auto item = co_await out.next()) collected.push_back(std::move(*item));
        co_return collected;
      }());

  // Aggregate the stream like a unary client would.
  openai::ChatDeltaAggregator aggregator;
  for (const auto& chunk : chunks) aggregator.push(chunk);
  auto response = std::move(aggregator).finalize();

  REQUIRE(response.choices.size() == 1);
  CHECK(response.choices[0].message.content == "hola!");  // template applied
  CHECK(response.choices[0].finish_reason == "stop");
  CHECK(response.model == "echo-model");
  REQUIRE(response.usage.has_value());
  CHECK(response.usage->prompt_tokens == 5);      // "hola!" as bytes
  CHECK(response.usage->completion_tokens == 5);  // echoed back
}
