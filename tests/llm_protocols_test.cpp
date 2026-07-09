// SPDX-License-Identifier: Apache-2.0
//
// M1 LLM protocol tests: common types, OpenAI requests/responses, delta
// generation/aggregation, SSE codec, and token block hashing. Golden hash
// values and several fixtures are ported from the Rust test suite so the two
// implementations stay interoperable.

#include <catch2/catch_test_macros.hpp>

#include <nlohmann/json.hpp>

#include "llm/protocols/common.h"
#include "llm/protocols/llm_backend.h"
#include "llm/protocols/openai.h"
#include "llm/protocols/sse.h"
#include "llm/tokens.h"

using namespace dynamo::llm;
using dynamo::pipeline::Annotated;
using nlohmann::json;

// ---------------------------------------------------------------------------
// common

TEST_CASE("finish reason string and json round-trips", "[llm][common]") {
  CHECK(FinishReason::eos().to_string() == "eos");
  CHECK(FinishReason::from_string("length") == FinishReason::length());
  CHECK(FinishReason::from_string("error: boom") == FinishReason::error("boom"));
  CHECK(FinishReason::error("boom").to_string() == "error: boom");
  CHECK_THROWS_AS(FinishReason::from_string("nope"), std::invalid_argument);

  // Rust serde encoding: unit variants as strings, error as {"error": msg}.
  CHECK(json(FinishReason::stop()) == json("stop"));
  CHECK(json(FinishReason::error("x")) == json({{"error", "x"}}));
  CHECK(json("cancelled").get<FinishReason>() == FinishReason::cancelled());
  CHECK(json({{"error", "x"}}).get<FinishReason>() == FinishReason::error("x"));
}

TEST_CASE("stop conditions apply_ignore_eos", "[llm][common]") {
  StopConditions sc;
  sc.max_tokens = 128;
  sc.stop = std::vector<std::string>{"a"};
  sc.stop_token_ids_hidden = std::vector<TokenIdType>{7};
  sc.ignore_eos = true;
  sc.apply_ignore_eos();
  CHECK(sc.min_tokens == 128u);
  CHECK_FALSE(sc.stop.has_value());
  CHECK_FALSE(sc.stop_token_ids_hidden.has_value());
}

TEST_CASE("backend output json round-trip with null tokens", "[llm][common]") {
  BackendOutput out;
  out.token_ids = {1, 2, 3};
  out.tokens = {std::string("a"), std::nullopt, std::string("c")};
  out.text = "ac";
  out.finish_reason = FinishReason::eos();
  out.mdcsum = "sum";

  json j = out;
  auto back = j.get<BackendOutput>();
  CHECK(back.token_ids == out.token_ids);
  REQUIRE(back.tokens.size() == 3);
  CHECK(back.tokens[0] == std::string("a"));
  CHECK_FALSE(back.tokens[1].has_value());
  CHECK(back.text == "ac");
  CHECK(back.finish_reason == FinishReason::eos());
  CHECK(back.mdcsum == "sum");
}

TEST_CASE("preprocessed request round-trip and annotations", "[llm][common]") {
  PreprocessedRequest req;
  req.token_ids = {10, 20};
  req.stop_conditions.max_tokens = 16;
  req.sampling_options.temperature = 0.5f;
  req.eos_token_ids = {2};
  req.annotations = {"formatted_prompt"};

  json j = req;
  auto back = j.get<PreprocessedRequest>();
  CHECK(back.token_ids == req.token_ids);
  CHECK(back.stop_conditions.max_tokens == 16u);
  CHECK(back.sampling_options.temperature == 0.5f);
  CHECK(back.eos_token_ids == std::vector<TokenIdType>{2});
  CHECK(back.has_annotation("formatted_prompt"));
  CHECK_FALSE(back.has_annotation("other"));
}

// ---------------------------------------------------------------------------
// openai requests

TEST_CASE("chat request parses openai json with nvext", "[llm][openai]") {
  auto j = json::parse(R"({
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [
      {"role": "system", "content": "be brief"},
      {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    ],
    "temperature": 0.7,
    "max_completion_tokens": 64,
    "stop": "END",
    "stream": true,
    "unknown_field": {"ignored": true},
    "nvext": {"ignore_eos": true, "annotations": ["formatted_prompt"]}
  })");

  auto req = j.get<openai::NvCreateChatCompletionRequest>();
  CHECK(req.model == "meta/llama-3.1-8b-instruct");
  REQUIRE(req.messages.size() == 2);
  CHECK(req.messages[0].content == "be brief");
  CHECK(req.messages[1].content == "hi");  // flattened from parts
  CHECK(req.stream == true);
  REQUIRE(req.stop.has_value());
  CHECK(*req.stop == std::vector<std::string>{"END"});
  CHECK(req.has_annotation("formatted_prompt"));

  auto sampling = openai::extract_sampling_options(req);
  CHECK(sampling.temperature == 0.7f);
  auto stops = openai::extract_stop_conditions(req);
  CHECK(stops.max_tokens == 64u);
  CHECK(stops.ignore_eos == true);
  REQUIRE(stops.stop.has_value());
  CHECK(stops.stop->size() == 1);
}

TEST_CASE("chat request tools and tool messages round-trip", "[llm][openai]") {
  auto j = json::parse(R"({
    "model": "m",
    "messages": [
      {"role": "user", "content": "weather?"},
      {"role": "assistant", "content": null,
       "tool_calls": [{"id": "call_1", "type": "function",
                       "function": {"name": "get_weather", "arguments": "{}"}}]},
      {"role": "tool", "content": "sunny", "tool_call_id": "call_1"}
    ],
    "tools": [{"type": "function",
               "function": {"name": "get_weather", "description": "d",
                            "parameters": {"type": "object"}}}],
    "tool_choice": "auto"
  })");

  auto req = j.get<openai::NvCreateChatCompletionRequest>();
  REQUIRE(req.tools.is_array());
  CHECK(req.tools[0]["function"]["name"] == "get_weather");
  CHECK(req.tool_choice == "auto");
  REQUIRE(req.messages.size() == 3);
  CHECK(req.messages[1].tool_calls[0]["id"] == "call_1");
  CHECK(req.messages[2].tool_call_id == "call_1");

  json out = req;
  CHECK(out["tools"] == j["tools"]);
  CHECK(out["tool_choice"] == "auto");
  CHECK(out["messages"][1]["tool_calls"] == j["messages"][1]["tool_calls"]);
  CHECK(out["messages"][2]["tool_call_id"] == "call_1");

  // Absent tool fields stay absent on re-serialization.
  openai::NvCreateChatCompletionRequest bare;
  json bare_out = bare;
  CHECK_FALSE(bare_out.contains("tools"));
  CHECK_FALSE(bare_out.contains("tool_choice"));

  // Present-but-mistyped tool fields are rejected, not fed to the template.
  CHECK_THROWS_AS(json::parse(R"({"model": "m", "messages": [], "tools": {}})")
                      .get<openai::NvCreateChatCompletionRequest>(),
                  std::invalid_argument);
  CHECK_THROWS_AS(json::parse(R"({"model": "m", "messages": [], "tool_choice": 3})")
                      .get<openai::NvCreateChatCompletionRequest>(),
                  std::invalid_argument);
}

TEST_CASE("sampling extraction validates ranges and applies greedy", "[llm][openai]") {
  openai::NvCreateChatCompletionRequest req;
  req.model = "m";
  req.temperature = 2.5f;
  CHECK_THROWS_AS(openai::extract_sampling_options(req), std::invalid_argument);

  req.temperature = 0.7f;
  req.top_p = 0.9f;
  req.nvext = openai::NvExt{};
  req.nvext->greed_sampling = true;
  auto sampling = openai::extract_sampling_options(req);
  CHECK_FALSE(sampling.temperature.has_value());
  CHECK_FALSE(sampling.top_p.has_value());

  req.nvext.reset();
  req.stop = std::vector<std::string>{"a", "b", "c", "d", "e"};
  CHECK_THROWS_AS(openai::extract_stop_conditions(req), std::invalid_argument);
}

TEST_CASE("nvext validation", "[llm][openai]") {
  openai::NvExt e;
  e.top_k = -1;
  e.repetition_penalty = 1.5;
  CHECK_NOTHROW(e.validate());
  e.top_k = 0;
  CHECK_THROWS_AS(e.validate(), std::invalid_argument);
  e.top_k = 10;
  e.repetition_penalty = -0.5;
  CHECK_THROWS_AS(e.validate(), std::invalid_argument);
}

TEST_CASE("completion request prompt flattening", "[llm][openai]") {
  openai::NvCreateCompletionRequest req;
  req.prompt = "hello";
  CHECK(req.prompt_to_string() == "hello");
  req.prompt = json::array({"a", "b"});
  CHECK(req.prompt_to_string() == "a b");
  req.prompt = json::array({1, 2, 3});
  CHECK(req.prompt_to_string() == "1 2 3");
  req.prompt = json::array({json::array({1, 2}), json::array({3, 4})});
  CHECK(req.prompt_to_string() == "1 2 | 3 4");

  CHECK_FALSE(req.raw_prompt().has_value());
  req.nvext = openai::NvExt{};
  req.nvext->use_raw_prompt = true;
  CHECK(req.raw_prompt() == "1 2 | 3 4");
}

// ---------------------------------------------------------------------------
// delta generators

TEST_CASE("chat delta generator maps backend output", "[llm][openai][delta]") {
  openai::NvCreateChatCompletionRequest req;
  req.model = "test-model";
  auto gen = openai::response_generator(req);
  gen.update_isl(5);

  BackendOutput delta;
  delta.token_ids = {1, 2};
  delta.text = "hi";
  auto chunk = gen.choice_from_postprocessor(delta);

  CHECK(chunk.model == "test-model");
  CHECK(chunk.object == "chat.completion.chunk");
  CHECK(chunk.id.rfind("chatcmpl-", 0) == 0);
  REQUIRE(chunk.choices.size() == 1);
  CHECK(chunk.choices[0].delta.role == "assistant");
  CHECK(chunk.choices[0].delta.content == "hi");
  CHECK_FALSE(chunk.choices[0].finish_reason.has_value());
  REQUIRE(chunk.usage.has_value());
  CHECK(chunk.usage->prompt_tokens == 5);
  CHECK(chunk.usage->completion_tokens == 2);
  CHECK(chunk.usage->total_tokens == 7);

  BackendOutput final_delta;
  final_delta.finish_reason = FinishReason::eos();
  auto final_chunk = gen.choice_from_postprocessor(final_delta);
  CHECK(final_chunk.choices[0].finish_reason == "stop");

  BackendOutput err_delta;
  err_delta.finish_reason = FinishReason::error("engine exploded");
  CHECK_THROWS_AS(gen.choice_from_postprocessor(err_delta), std::runtime_error);
}

TEST_CASE("completion delta generator maps cancelled distinctly", "[llm][openai][delta]") {
  openai::NvCreateCompletionRequest req;
  req.model = "test-model";
  auto gen = openai::response_generator(req);

  BackendOutput delta;
  delta.token_ids = {1};
  delta.text = "x";
  delta.finish_reason = FinishReason::cancelled();
  auto chunk = gen.choice_from_postprocessor(delta);
  CHECK(chunk.object == "text_completion");
  CHECK(chunk.id.rfind("cmpl-", 0) == 0);
  REQUIRE(chunk.choices.size() == 1);
  CHECK(chunk.choices[0].text == "x");
  CHECK(chunk.choices[0].finish_reason == "cancelled");
}

// ---------------------------------------------------------------------------
// aggregators (ported from the Rust aggregator tests)

namespace {

Annotated<openai::NvCreateChatCompletionStreamResponse> chat_test_delta(
    uint32_t index, const std::string& text, std::optional<std::string> role,
    std::optional<std::string> finish_reason) {
  openai::ChatChoiceStream choice;
  choice.index = index;
  choice.delta.role = std::move(role);
  choice.delta.content = text;
  choice.finish_reason = std::move(finish_reason);

  openai::NvCreateChatCompletionStreamResponse chunk;
  chunk.id = "test_id";
  chunk.model = "meta/llama-3.1-8b-instruct";
  chunk.created = 1234567890;
  chunk.choices.push_back(std::move(choice));

  auto annotated = Annotated<openai::NvCreateChatCompletionStreamResponse>::from_data(chunk);
  annotated.id = "test_id";
  return annotated;
}

}  // namespace

TEST_CASE("chat aggregator: empty stream yields defaults", "[llm][openai][aggregate]") {
  openai::ChatDeltaAggregator agg;
  auto response = std::move(agg).finalize();
  CHECK(response.id.empty());
  CHECK(response.model.empty());
  CHECK(response.created == 0);
  CHECK(response.choices.empty());
  CHECK_FALSE(response.usage.has_value());
}

TEST_CASE("chat aggregator: single and multiple deltas", "[llm][openai][aggregate]") {
  openai::ChatDeltaAggregator agg;
  agg.push(chat_test_delta(0, "Hello,", "user", std::nullopt));
  agg.push(chat_test_delta(0, " world!", std::nullopt, "stop"));
  auto response = std::move(agg).finalize();

  CHECK(response.id == "test_id");
  CHECK(response.model == "meta/llama-3.1-8b-instruct");
  CHECK(response.created == 1234567890);
  REQUIRE(response.choices.size() == 1);
  CHECK(response.choices[0].index == 0);
  CHECK(response.choices[0].message.role == "user");
  CHECK(response.choices[0].message.content == "Hello, world!");
  CHECK(response.choices[0].finish_reason == "stop");
}

TEST_CASE("chat aggregator: multiple choices sorted by index", "[llm][openai][aggregate]") {
  openai::ChatDeltaAggregator agg;
  agg.push(chat_test_delta(1, "Choice 1", "assistant", "stop"));
  agg.push(chat_test_delta(0, "Choice 0", "assistant", "stop"));
  auto response = std::move(agg).finalize();

  REQUIRE(response.choices.size() == 2);
  CHECK(response.choices[0].index == 0);
  CHECK(response.choices[0].message.content == "Choice 0");
  CHECK(response.choices[1].index == 1);
  CHECK(response.choices[1].message.content == "Choice 1");
}

TEST_CASE("chat aggregator: error annotation surfaces", "[llm][openai][aggregate]") {
  openai::ChatDeltaAggregator agg;
  agg.push(chat_test_delta(0, "partial", "assistant", std::nullopt));
  agg.push(Annotated<openai::NvCreateChatCompletionStreamResponse>::from_error("boom"));
  CHECK_THROWS_AS(std::move(agg).finalize(), std::runtime_error);
}

TEST_CASE("completion aggregator folds text and finish reason", "[llm][openai][aggregate]") {
  openai::CompletionDeltaAggregator agg;

  openai::NvCreateCompletionResponse chunk;
  chunk.id = "cmpl-1";
  chunk.model = "m";
  chunk.created = 42;
  openai::CompletionChoice choice;
  choice.text = "Hello,";
  chunk.choices.push_back(choice);
  agg.push(Annotated<openai::NvCreateCompletionResponse>::from_data(chunk));

  chunk.choices[0].text = " world!";
  chunk.choices[0].finish_reason = "stop";
  agg.push(Annotated<openai::NvCreateCompletionResponse>::from_data(chunk));

  auto response = std::move(agg).finalize();
  CHECK(response.id == "cmpl-1");
  CHECK(response.object == "text_completion");
  REQUIRE(response.choices.size() == 1);
  CHECK(response.choices[0].text == "Hello, world!");
  CHECK(response.choices[0].finish_reason == "stop");
}

// ---------------------------------------------------------------------------
// SSE codec (ported from the Rust codec tests)

TEST_CASE("sse: message with all fields", "[llm][sse]") {
  SseDecoder dec;
  auto messages = dec.feed(
      "id: 123\n"
      "event: test\n"
      "data: {\"message\": \"Hello World\"}\n"
      ": This is a comment\n"
      "\n");
  REQUIRE(messages.size() == 1);
  CHECK(messages[0].id == "123");
  CHECK(messages[0].event == "test");
  REQUIRE(messages[0].comments.has_value());
  CHECK(messages[0].comments->at(0) == "This is a comment");
  auto data = messages[0].decode_data<json>();
  CHECK(data.at("message") == "Hello World");
}

TEST_CASE("sse: chunked delivery across feeds", "[llm][sse]") {
  SseDecoder dec;
  auto first = dec.feed("data: {\"messa");
  CHECK(first.empty());
  auto second = dec.feed("ge\": \"split\"}\n\n");
  REQUIRE(second.size() == 1);
  CHECK(second[0].decode_data<json>().at("message") == "split");
}

TEST_CASE("sse: multiple comments, missing data, unknown fields", "[llm][sse]") {
  SseDecoder dec;
  auto messages = dec.feed(": First comment\n: Second comment\n\n");
  REQUIRE(messages.size() == 1);
  CHECK_FALSE(messages[0].data.has_value());
  CHECK(messages[0].comments->size() == 2);

  messages = dec.feed("id: 789\nevent: test_event\n\n");
  REQUIRE(messages.size() == 1);
  CHECK(messages[0].id == "789");
  CHECK(messages[0].event == "test_event");
  CHECK_FALSE(messages[0].data.has_value());

  messages = dec.feed("unknown: value\ndata: {\"message\": \"Hello\"}\n\n");
  REQUIRE(messages.size() == 1);
  CHECK_FALSE(messages[0].id.has_value());
  CHECK(messages[0].decode_data<json>().at("message") == "Hello");
}

TEST_CASE("sse: empty data line emits nothing", "[llm][sse]") {
  SseDecoder dec;
  auto messages = dec.feed("data:\n\n");
  CHECK(messages.empty());
  CHECK_FALSE(dec.finish().has_value());
}

TEST_CASE("sse: multi-line data joined with newline", "[llm][sse]") {
  SseDecoder dec;
  auto messages = dec.feed("data: line1\ndata: line2\n\n");
  REQUIRE(messages.size() == 1);
  CHECK(messages[0].data == "line1\nline2");
}

TEST_CASE("sse: [DONE] sentinel is dropped", "[llm][sse]") {
  SseDecoder dec;
  auto messages = dec.feed("data: {\"x\": 1}\n\ndata: [DONE]\n\n");
  REQUIRE(messages.size() == 1);  // only the payload message
  CHECK(messages[0].data == "{\"x\": 1}");
}

TEST_CASE("sse: crlf line endings", "[llm][sse]") {
  SseDecoder dec;
  auto messages = dec.feed("data: {\"x\": 1}\r\n\r\n");
  REQUIRE(messages.size() == 1);
  CHECK(messages[0].data == "{\"x\": 1}");
}

TEST_CASE("sse: finish flushes a partial trailing event", "[llm][sse]") {
  SseDecoder dec;
  auto messages = dec.feed("data: {\"x\": 1}\ndata: tail");
  CHECK(messages.empty());
  auto tail = dec.finish();
  REQUIRE(tail.has_value());
  CHECK(tail->data == "{\"x\": 1}\ntail");
}

TEST_CASE("sse: openai chat stream sample decodes", "[llm][sse]") {
  // Abbreviated version of the Rust fixture recorded from an OpenAI-style
  // endpoint: role chunk, content chunks, finish chunk, [DONE].
  const char* sample =
      "\n"
      "data: {\"id\":\"chat-1\",\"object\":\"chat.completion.chunk\",\"created\":1727750141,"
      "\"model\":\"mixtral\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\","
      "\"content\":null},\"logprobs\":null,\"finish_reason\":null}]}\n"
      "\n"
      "data: {\"id\":\"chat-1\",\"object\":\"chat.completion.chunk\",\"created\":1727750141,"
      "\"model\":\"mixtral\",\"choices\":[{\"index\":0,\"delta\":{\"role\":null,"
      "\"content\":\"A\"},\"logprobs\":null,\"finish_reason\":null}]}\n"
      "\n"
      "data: {\"id\":\"chat-1\",\"object\":\"chat.completion.chunk\",\"created\":1727750141,"
      "\"model\":\"mixtral\",\"choices\":[{\"index\":0,\"delta\":{\"role\":null,"
      "\"content\":\" GPU\"},\"logprobs\":null,\"finish_reason\":null}]}\n"
      "\n"
      "data: {\"id\":\"chat-1\",\"object\":\"chat.completion.chunk\",\"created\":1727750141,"
      "\"model\":\"mixtral\",\"choices\":[{\"index\":0,\"delta\":{\"role\":null,"
      "\"content\":\"\"},\"logprobs\":null,\"finish_reason\":\"stop\",\"stop_reason\":null}]}\n"
      "\n"
      "data: [DONE]\n"
      "\n";

  SseDecoder dec;
  auto messages = dec.feed(sample);
  REQUIRE(messages.size() == 4);

  openai::ChatDeltaAggregator agg;
  for (const auto& m : messages) {
    agg.push(annotated_from_sse<openai::NvCreateChatCompletionStreamResponse>(m));
  }
  auto response = std::move(agg).finalize();
  CHECK(response.id == "chat-1");
  REQUIRE(response.choices.size() == 1);
  CHECK(response.choices[0].message.role == "assistant");
  CHECK(response.choices[0].message.content == "A GPU");
  CHECK(response.choices[0].finish_reason == "stop");
}

TEST_CASE("sse: error event becomes error annotation", "[llm][sse]") {
  SseMessage message;
  message.event = "error";
  message.comments = std::vector<std::string>{"An error occurred"};
  auto annotated = annotated_from_sse<json>(message);
  CHECK(annotated.is_error());
  REQUIRE(annotated.comment.has_value());
  CHECK(annotated.comment->at(0) == "An error occurred");

  SseMessage bad;
  bad.data = "Invalid JSON";
  auto bad_annotated = annotated_from_sse<json>(bad);
  CHECK(bad_annotated.is_error());
}

TEST_CASE("sse: encode round-trips through the decoder", "[llm][sse]") {
  SseMessage message;
  message.id = "42";
  message.event = "update";
  message.data = "{\"a\": 1}\n{\"b\": 2}";
  message.comments = std::vector<std::string>{"note"};

  SseDecoder dec;
  auto messages = dec.feed(encode_sse(message) + encode_sse_done());
  REQUIRE(messages.size() == 1);
  CHECK(messages[0].id == "42");
  CHECK(messages[0].event == "update");
  CHECK(messages[0].data == message.data);
  CHECK(messages[0].comments->at(0) == "note");
}

// ---------------------------------------------------------------------------
// tokens (golden values from the Rust tokens.rs test)

TEST_CASE("token blocks match rust golden hashes", "[llm][tokens]") {
  std::vector<Token> tokens{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  TokenSequence sequence(std::move(tokens), 4);

  REQUIRE(sequence.blocks().size() == 2);
  CHECK(sequence.current_block().tokens() == std::vector<Token>{9, 10});

  CHECK(sequence.blocks()[0].tokens == std::vector<Token>{1, 2, 3, 4});
  CHECK(sequence.blocks()[0].block_hash == 14643705804678351452ull);
  CHECK(sequence.blocks()[0].sequence_hash == 14643705804678351452ull);

  CHECK(sequence.blocks()[1].tokens == std::vector<Token>{5, 6, 7, 8});
  CHECK(sequence.blocks()[1].block_hash == 16777012769546811212ull);
  CHECK(sequence.blocks()[1].sequence_hash == 4945711292740353085ull);
  CHECK(sequence.blocks()[1].parent_sequence_hash == 14643705804678351452ull);
}

TEST_CASE("token sequence grows by pushing", "[llm][tokens]") {
  TokenSequence sequence(std::vector<Token>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 4);

  CHECK(sequence.push_token(11) == nullptr);
  REQUIRE(sequence.blocks().size() == 2);

  const TokenBlock* block = sequence.push_token(12);
  REQUIRE(block != nullptr);
  REQUIRE(sequence.blocks().size() == 3);
  CHECK(sequence.current_block().tokens().empty());
  CHECK(block->tokens == std::vector<Token>{9, 10, 11, 12});
  CHECK(block->parent_sequence_hash == sequence.blocks()[1].sequence_hash);

  // The pushed block chains like a split block would.
  uint64_t pair[2] = {sequence.blocks()[1].sequence_hash, block->block_hash};
  CHECK(block->sequence_hash == compute_hash(pair, sizeof(pair)));

  // And the partial tail advances its parent to the new block.
  auto [blocks, current] = std::move(sequence).into_parts();
  CHECK(blocks.size() == 3);
  CHECK(current.parent_sequence_hash() == blocks[2].sequence_hash);

  auto next = current.push_token(13);
  CHECK_FALSE(next.has_value());
  CHECK(current.tokens().size() == 1);
}

TEST_CASE("empty token sequence is valid", "[llm][tokens]") {
  // Deviation from Rust, which panics on empty input to split_tokens.
  TokenSequence sequence({}, 4);
  CHECK(sequence.blocks().empty());
  CHECK(sequence.current_block().tokens().empty());

  TokenSequence seq2({1, 2, 3}, 4);
  CHECK(seq2.blocks().empty());
  CHECK(seq2.current_block().tokens() == std::vector<Token>{1, 2, 3});
}
