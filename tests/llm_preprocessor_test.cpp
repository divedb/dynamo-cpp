// SPDX-License-Identifier: Apache-2.0
//
// M2 tests: ByteLevelTokenizer, incremental Sequence decoding (UTF-8
// partials), StopSequenceDecoder jail semantics, HF chat-template rendering
// via minja, and the OpenAIPreprocessor operator end-to-end against an
// inline backend engine.

#include <catch2/catch_test_macros.hpp>

#include <memory>

#include "llm/preprocessor.h"
#include "llm/preprocessor/prompt.h"
#include "llm/tokenizers.h"
#include "runtime/coro/sync_wait.h"

using namespace dynamo::llm;
using dynamo::pipeline::Annotated;
namespace pipeline = dynamo::pipeline;
namespace coro = dynamo::coro;

namespace {

const char* kChatMlTemplate =
    "{% for message in messages %}"
    "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

std::shared_ptr<const PromptFormatter> chatml_formatter() {
  return std::make_shared<HfChatTemplateFormatter>(kChatMlTemplate, "<s>", "</s>");
}

}  // namespace

// ---------------------------------------------------------------------------
// tokenizer

TEST_CASE("byte tokenizer round-trips utf-8 text", "[llm][tokenizer]") {
  ByteLevelTokenizer tok;
  std::string text = "héllo ⚡";
  auto encoding = tok.encode(text);
  CHECK(encoding.token_ids.size() == text.size());  // bytes, not chars
  CHECK(tok.decode(encoding.token_ids, false) == text);
}

TEST_CASE("byte tokenizer specials and lossy decode", "[llm][tokenizer]") {
  ByteLevelTokenizer tok;
  std::vector<TokenIdType> ids{ByteLevelTokenizer::kBosId, 'h', 'i', ByteLevelTokenizer::kEosId};
  CHECK(tok.decode(ids, false) == "<s>hi</s>");
  CHECK(tok.decode(ids, true) == "hi");

  // Incomplete UTF-8 tail decodes to U+FFFD.
  std::vector<TokenIdType> partial{0xE2, 0x9A};  // first two bytes of ⚡
  CHECK(tok.decode(partial, false) == "\xEF\xBF\xBD");
}

TEST_CASE("sequence holds partial characters", "[llm][tokenizer]") {
  auto tok = std::make_shared<ByteLevelTokenizer>();
  Sequence seq(tok);

  // "⚡" = E2 9A A1: nothing emitted until the final byte arrives.
  CHECK(seq.append_token_id(0xE2).empty());
  CHECK(seq.append_token_id(0x9A).empty());
  CHECK(seq.append_token_id(0xA1) == "⚡");

  // Plain ASCII streams out immediately.
  CHECK(seq.append_token_id('!') == "!");
  CHECK(seq.text() == "⚡!");
}

TEST_CASE("stop sequence decoder: hidden and visible stop tokens", "[llm][tokenizer]") {
  auto tok = std::make_shared<ByteLevelTokenizer>();

  StopSequenceConfig config;
  config.stop_token_ids_hidden = {ByteLevelTokenizer::kEosId};
  StopSequenceDecoder hidden(tok, config);
  auto out = hidden.append_token_id('a');
  CHECK(out.kind == SequenceDecoderOutput::Kind::text);
  CHECK(out.text == "a");
  out = hidden.append_token_id(ByteLevelTokenizer::kEosId);
  CHECK(out.kind == SequenceDecoderOutput::Kind::stopped);
  CHECK(hidden.is_complete());
  CHECK_THROWS_AS(hidden.append_token_id('b'), std::runtime_error);

  StopSequenceConfig visible_config;
  visible_config.stop_token_ids_visible = {'!'};
  StopSequenceDecoder visible(tok, visible_config);
  out = visible.append_token_id('!');
  CHECK(out.kind == SequenceDecoderOutput::Kind::stopped_with_text);
  CHECK(out.text == "!");
}

TEST_CASE("stop sequence decoder: hidden stop sequence jails text", "[llm][tokenizer]") {
  auto tok = std::make_shared<ByteLevelTokenizer>();
  StopSequenceConfig config;
  config.stop_sequences_hidden = {"STOP"};
  StopSequenceDecoder decoder(tok, config);

  auto out = decoder.append_token_id('S');
  CHECK(out.kind == SequenceDecoderOutput::Kind::held);
  out = decoder.append_token_id('T');
  CHECK(out.kind == SequenceDecoderOutput::Kind::held);
  out = decoder.append_token_id('O');
  CHECK(out.kind == SequenceDecoderOutput::Kind::held);
  out = decoder.append_token_id('P');
  CHECK(out.kind == SequenceDecoderOutput::Kind::stopped);
  CHECK(decoder.is_complete());
}

TEST_CASE("stop sequence decoder: jailed text released on mismatch", "[llm][tokenizer]") {
  auto tok = std::make_shared<ByteLevelTokenizer>();
  StopSequenceConfig config;
  config.stop_sequences_hidden = {"STOP"};
  StopSequenceDecoder decoder(tok, config);

  CHECK(decoder.append_token_id('S').kind == SequenceDecoderOutput::Kind::held);
  auto out = decoder.append_token_id('o');  // "So" is not a prefix of "STOP"
  CHECK(out.kind == SequenceDecoderOutput::Kind::text);
  CHECK(out.text == "So");
}

// ---------------------------------------------------------------------------
// prompt formatting

TEST_CASE("chatml template renders messages and generation prompt", "[llm][prompt]") {
  auto formatter = chatml_formatter();
  CHECK(formatter->supports_add_generation_prompt());

  openai::NvCreateChatCompletionRequest request;
  request.model = "m";
  request.messages.push_back({.role = "system", .content = "be brief"});
  request.messages.push_back({.role = "user", .content = "hi"});

  auto input = chat_template_input(request);
  CHECK(input.add_generation_prompt);  // last message is a user turn
  CHECK(formatter->render(input) ==
        "<|im_start|>system\nbe brief<|im_end|>\n"
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\n");

  // Assistant-last turns do not add the generation prompt.
  request.messages.push_back({.role = "assistant", .content = "hello"});
  auto continued = chat_template_input(request);
  CHECK_FALSE(continued.add_generation_prompt);
  auto rendered = formatter->render(continued);
  CHECK(rendered.find("hello<|im_end|>\n") != std::string::npos);
  CHECK(rendered.rfind("<|im_start|>assistant\n") != rendered.size() - 22);
}

TEST_CASE("completion requests render as a single user turn", "[llm][prompt]") {
  openai::NvCreateCompletionRequest request;
  request.model = "m";
  request.prompt = "once upon a time";
  auto input = chat_template_input(request);
  REQUIRE(input.messages.size() == 1);
  CHECK(input.messages[0]["role"] == "user");
  CHECK(input.messages[0]["content"] == "once upon a time");
  CHECK(input.add_generation_prompt);
}

TEST_CASE("tools flow into the template input and context", "[llm][prompt]") {
  openai::NvCreateChatCompletionRequest request;
  request.model = "m";
  request.messages.push_back({.role = "user", .content = "weather?"});
  request.tools = nlohmann::json::parse(R"([{"type": "function",
    "function": {"name": "get_weather", "description": "d",
                 "parameters": {"type": "object", "properties": {}}}}])");

  auto input = chat_template_input(request);
  REQUIRE(input.tools.is_array());

  // A tools-aware template (prints tool names, so minja detects native
  // support and applies no polyfill) sees the definitions.
  HfChatTemplateFormatter formatter(
      "{% for t in tools %}<tool:{{ t.function.name }}>{% endfor %}"
      "{% for m in messages %}{{ m.content }}{% endfor %}",
      "<s>", "</s>");
  CHECK(formatter.render(input) == "<tool:get_weather>weather?");

  // Tool-call turns survive into the rendered message list.
  request.messages.push_back(
      {.role = "assistant",
       .tool_calls = nlohmann::json::parse(
           R"([{"id": "call_1", "type": "function",
                "function": {"name": "get_weather", "arguments": "{}"}}])")});
  request.messages.push_back({.role = "tool", .content = "sunny", .tool_call_id = "call_1"});
  auto full = chat_template_input(request);
  CHECK(full.messages[1]["tool_calls"][0]["id"] == "call_1");
  CHECK(full.messages[2]["tool_call_id"] == "call_1");
}

TEST_CASE("tool_use template selected when tools are present", "[llm][prompt]") {
  HfChatTemplateFormatter formatter(
      "D:{% for m in messages %}{{ m.content }}{% endfor %}", "<s>", "</s>",
      "T:{% for t in tools %}{{ t.function.name }}{% endfor %}");

  ChatTemplateInput input;
  input.messages.push_back({{"role", "user"}, {"content", "hi"}});
  CHECK(formatter.render(input) == "D:hi");

  input.tools = nlohmann::json::parse(
      R"([{"type": "function", "function": {"name": "get_weather"}}])");
  CHECK(formatter.render(input) == "T:get_weather");
}

TEST_CASE("llama3_datetime mixin injects datetime at render time", "[llm][prompt]") {
  const std::string tmpl = "{% if datetime %}<{{ datetime }}>{% endif %}chat";
  ChatTemplateInput input;

  // Without the mixin the variable is undefined.
  HfChatTemplateFormatter plain(tmpl, "<s>", "</s>");
  CHECK(plain.render(input) == "chat");

  HfChatTemplateFormatter dated(tmpl, "<s>", "</s>", std::nullopt, {"llama3_datetime"});
  auto rendered = dated.render(input);
  // "<DD, Month, YYYY>chat" — structural check (exact-date equality would
  // race the UTC midnight rollover).
  REQUIRE(rendered.size() > std::string("<01, May, 2026>chat").size() - 3);
  REQUIRE(rendered.front() == '<');
  auto close = rendered.find('>');
  REQUIRE(close != std::string::npos);
  auto date = rendered.substr(1, close - 1);
  REQUIRE(date.size() >= 14);  // "01, May, 2026"
  CHECK(std::isdigit(static_cast<unsigned char>(date[0])));
  CHECK(std::isdigit(static_cast<unsigned char>(date[1])));
  CHECK(date.substr(2, 2) == ", ");
  CHECK(date.substr(date.size() - 6, 2) == ", ");
  CHECK(rendered.substr(close + 1) == "chat");

  // oai_chat is a recognized no-op (as in Rust); unknown mixins are rejected
  // (Rust fails MDC deserialization on unknown enum variants).
  CHECK(HfChatTemplateFormatter(tmpl, "<s>", "</s>", std::nullopt, {"oai_chat"}).render(input) ==
        "chat");
  CHECK_THROWS_AS(HfChatTemplateFormatter(tmpl, "<s>", "</s>", std::nullopt, {"bogus"}),
                  std::runtime_error);
}

TEST_CASE("invalid template throws", "[llm][prompt]") {
  CHECK_THROWS_AS(HfChatTemplateFormatter("{% for x in %}", "<s>", "</s>"),
                  std::runtime_error);
  CHECK_THROWS_AS(HfChatTemplateFormatter("", "<s>", "</s>"), std::runtime_error);
}

// ---------------------------------------------------------------------------
// preprocessor

namespace {

std::shared_ptr<OpenAIPreprocessor> make_preprocessor() {
  return std::make_shared<OpenAIPreprocessor>(
      chatml_formatter(), std::make_shared<ByteLevelTokenizer>(),
      std::vector<TokenIdType>{ByteLevelTokenizer::kEosId}, "mdc-test-sum");
}

openai::NvCreateChatCompletionRequest simple_chat_request() {
  openai::NvCreateChatCompletionRequest request;
  request.model = "test-model";
  request.messages.push_back({.role = "user", .content = "hi"});
  return request;
}

/// Inline backend: replays the request token ids as one delta (detokenized
/// via ByteLevelTokenizer), then a final eos delta.
class ReplayBackend final
    : public pipeline::AsyncEngine<BackendInput, Annotated<BackendOutput>> {
 public:
  coro::Task<pipeline::ManyOut<Annotated<BackendOutput>>> generate(
      pipeline::SingleIn<BackendInput> request) override {
    auto [input, controller] = std::move(request).into_parts();
    co_return pipeline::ManyOut<Annotated<BackendOutput>>(replay(std::move(input)), controller);
  }

  bool fail = false;

 private:
  coro::AsyncGenerator<Annotated<BackendOutput>> replay(BackendInput input) {
    ByteLevelTokenizer tok;
    BackendOutput delta;
    delta.token_ids = input.token_ids;
    delta.text = tok.decode(input.token_ids, true);
    delta.mdcsum = input.mdc_sum.value_or("");
    co_yield Annotated<BackendOutput>::from_data(std::move(delta));

    BackendOutput final_delta;
    final_delta.finish_reason = fail ? FinishReason::error("backend exploded")
                                     : FinishReason::eos();
    final_delta.mdcsum = input.mdc_sum.value_or("");
    co_yield Annotated<BackendOutput>::from_data(std::move(final_delta));
  }
};

}  // namespace

TEST_CASE("preprocess: renders, tokenizes, merges eos stops", "[llm][preprocessor]") {
  auto pre = make_preprocessor();
  auto request = simple_chat_request();
  request.max_completion_tokens = 32;
  request.nvext = openai::NvExt{};
  request.nvext->annotations = std::vector<std::string>{kAnnotationFormattedPrompt,
                                                        kAnnotationTokenIds};

  auto [input, annotations] = pre->preprocess(request);

  std::string expected_prompt = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n";
  CHECK(annotations.at(kAnnotationFormattedPrompt) == expected_prompt);
  CHECK(annotations.count(kAnnotationTokenIds) == 1);

  ByteLevelTokenizer tok;
  CHECK(input.token_ids == tok.encode(expected_prompt).token_ids);
  CHECK(input.stop_conditions.max_tokens == 32u);
  REQUIRE(input.stop_conditions.stop_token_ids_hidden.has_value());
  CHECK(*input.stop_conditions.stop_token_ids_hidden ==
        std::vector<TokenIdType>{ByteLevelTokenizer::kEosId});
  CHECK(input.eos_token_ids == std::vector<TokenIdType>{ByteLevelTokenizer::kEosId});
  CHECK(input.mdc_sum == "mdc-test-sum");
  CHECK(input.annotations.size() == 2);
}

TEST_CASE("preprocess: ignore_eos clears eos enforcement", "[llm][preprocessor]") {
  auto pre = make_preprocessor();
  auto request = simple_chat_request();
  request.max_completion_tokens = 8;
  request.nvext = openai::NvExt{};
  request.nvext->ignore_eos = true;

  auto [input, annotations] = pre->preprocess(request);
  CHECK(annotations.empty());
  CHECK(input.eos_token_ids.empty());
  CHECK_FALSE(input.stop_conditions.stop_token_ids_hidden.has_value());
  CHECK(input.stop_conditions.min_tokens == 8u);
}

TEST_CASE("preprocessor operator maps backend deltas to chat chunks",
          "[llm][preprocessor][operator]") {
  auto pre = make_preprocessor();
  auto backend = std::make_shared<ReplayBackend>();
  auto engine = pipeline::link<openai::NvCreateChatCompletionRequest,
                               Annotated<openai::NvCreateChatCompletionStreamResponse>,
                               BackendInput, Annotated<BackendOutput>>(pre, backend);

  auto request = simple_chat_request();
  request.nvext = openai::NvExt{};
  request.nvext->annotations = std::vector<std::string>{kAnnotationFormattedPrompt};

  auto chunks = coro::sync_wait([&]() -> coro::Task<
                                          std::vector<Annotated<
                                              openai::NvCreateChatCompletionStreamResponse>>> {
    std::vector<Annotated<openai::NvCreateChatCompletionStreamResponse>> collected;
    auto out = co_await engine->generate(
        pipeline::SingleIn<openai::NvCreateChatCompletionRequest>(request));
    while (auto item = co_await out.next()) collected.push_back(std::move(*item));
    co_return collected;
  }());

  REQUIRE(chunks.size() == 3);

  // 1) the formatted_prompt annotation rides ahead of the data
  CHECK(chunks[0].event == kAnnotationFormattedPrompt);
  CHECK_FALSE(chunks[0].data.has_value());

  // 2) the replayed prompt text as a chat chunk with usage
  REQUIRE(chunks[1].data.has_value());
  const auto& chunk = *chunks[1].data;
  CHECK(chunk.model == "test-model");
  REQUIRE(chunk.choices.size() == 1);
  CHECK(chunk.choices[0].delta.content ==
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
  REQUIRE(chunk.usage.has_value());
  CHECK(chunk.usage->prompt_tokens == static_cast<int32_t>(chunk.usage->completion_tokens));

  // 3) the eos delta becomes finish_reason "stop"
  REQUIRE(chunks[2].data.has_value());
  CHECK(chunks[2].data->choices[0].finish_reason == "stop");
}

TEST_CASE("preprocessor operator surfaces backend errors and stops the context",
          "[llm][preprocessor][operator]") {
  auto pre = make_preprocessor();
  auto backend = std::make_shared<ReplayBackend>();
  backend->fail = true;
  auto engine = pipeline::link<openai::NvCreateChatCompletionRequest,
                               Annotated<openai::NvCreateChatCompletionStreamResponse>,
                               BackendInput, Annotated<BackendOutput>>(pre, backend);

  auto request = simple_chat_request();
  pipeline::SingleIn<openai::NvCreateChatCompletionRequest> in(request);
  auto controller = in.controller();

  auto chunks = coro::sync_wait([&]() -> coro::Task<
                                          std::vector<Annotated<
                                              openai::NvCreateChatCompletionStreamResponse>>> {
    std::vector<Annotated<openai::NvCreateChatCompletionStreamResponse>> collected;
    auto out = co_await engine->generate(std::move(in));
    while (auto item = co_await out.next()) collected.push_back(std::move(*item));
    co_return collected;
  }());

  REQUIRE(chunks.size() == 2);  // data chunk, then the error annotation
  CHECK(chunks[1].is_error());
  REQUIRE(chunks[1].comment.has_value());
  CHECK(chunks[1].comment->at(0) == "backend exploded");
  CHECK(controller->is_stopped());
}
