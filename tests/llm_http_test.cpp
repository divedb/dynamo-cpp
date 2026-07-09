// SPDX-License-Identifier: Apache-2.0
//
// M5 tests: the OpenAI HTTP frontend — model listing, unary and SSE
// streaming chat completions against a local pipeline, 404s, metrics
// exposition, and the model watcher wiring a remote worker through
// discovery + the component layer.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sys/socket.h>

#include <memory>
#include <thread>

#include "component/component.h"
#include "llm/backend.h"
#include "llm/engines.h"
#include "llm/http/model_watcher.h"
#include "llm/http/service.h"
#include "llm/preprocessor.h"
#include "llm/protocols/sse.h"
#include "runtime/coro/sync_wait.h"
#include "transports/socket.h"

using namespace dynamo::llm;
using namespace std::chrono_literals;
using dynamo::pipeline::Annotated;
namespace pipeline = dynamo::pipeline;
namespace coro = dynamo::coro;
namespace http = dynamo::llm::http;
using nlohmann::json;

namespace {

dynamo::RuntimeConfig test_config() {
  dynamo::RuntimeConfig config;
  config.num_worker_threads = 4;
  config.num_background_threads = 2;
  return config;
}

std::string unique_ns() {
  static std::atomic<int> counter{100};
  return "httpns" + std::to_string(counter.fetch_add(1));
}

/// Minimal HTTP/1.1 test client: one request per connection
/// (Connection: close), handles Content-Length and chunked bodies.
struct HttpResult {
  int status = 0;
  std::string body;
};

HttpResult http_request(uint16_t port, const std::string& method, const std::string& path,
                        const std::string& body = {}) {
  auto socket = dynamo::transports::Socket::connect("127.0.0.1", port);
  REQUIRE(socket.has_value());

  std::string request = method + " " + path + " HTTP/1.1\r\n";
  request += "Host: 127.0.0.1\r\nConnection: close\r\n";
  if (!body.empty()) {
    request += "Content-Type: application/json\r\n";
    request += "Content-Length: " + std::to_string(body.size()) + "\r\n";
  }
  request += "\r\n" + body;
  REQUIRE(socket->write_all(request));

  std::string raw;
  char buf[4096];
  while (true) {
    ssize_t r = ::recv(socket->fd(), buf, sizeof(buf), 0);
    if (r <= 0) break;
    raw.append(buf, static_cast<size_t>(r));
  }

  HttpResult result;
  size_t line_end = raw.find("\r\n");
  REQUIRE(line_end != std::string::npos);
  result.status = std::stoi(raw.substr(9, 3));

  size_t header_end = raw.find("\r\n\r\n");
  REQUIRE(header_end != std::string::npos);
  std::string headers = raw.substr(0, header_end);
  std::string payload = raw.substr(header_end + 4);

  if (headers.find("Transfer-Encoding: chunked") != std::string::npos) {
    // De-chunk.
    size_t pos = 0;
    while (pos < payload.size()) {
      size_t nl = payload.find("\r\n", pos);
      if (nl == std::string::npos) break;
      size_t len = std::stoul(payload.substr(pos, nl - pos), nullptr, 16);
      if (len == 0) break;
      result.body += payload.substr(nl + 2, len);
      pos = nl + 2 + len + 2;
    }
  } else {
    result.body = payload;
  }
  return result;
}

/// A local chat engine: preprocessor -> backend -> echo over the byte
/// tokenizer with a trivial template.
http::ChatEngine make_local_chat_engine() {
  const char* tmpl = "{% for message in messages %}{{ message.content }}{% endfor %}";
  auto formatter = std::make_shared<HfChatTemplateFormatter>(tmpl, "", "");
  auto tokenizer = std::make_shared<ByteLevelTokenizer>();
  auto preprocessor = std::make_shared<OpenAIPreprocessor>(
      formatter, tokenizer, std::vector<TokenIdType>{ByteLevelTokenizer::kEosId}, "sum");
  auto backend = std::make_shared<Backend>(tokenizer, "sum");
  auto backend_engine = pipeline::link<BackendInput, Annotated<BackendOutput>, BackendInput,
                                       Annotated<LLMEngineOutput>>(backend,
                                                                   make_echo_engine_core());
  return pipeline::link<openai::NvCreateChatCompletionRequest,
                        Annotated<openai::NvCreateChatCompletionStreamResponse>, BackendInput,
                        Annotated<BackendOutput>>(preprocessor, backend_engine);
}

}  // namespace

TEST_CASE("http: health, model list, and 404s", "[llm][http]") {
  http::HttpService service({.host = "127.0.0.1", .port = 0});
  service.start();

  auto health = http_request(service.port(), "GET", "/health");
  CHECK(health.status == 200);
  CHECK(json::parse(health.body).at("status") == "ok");

  auto models = http_request(service.port(), "GET", "/v1/models");
  CHECK(models.status == 200);
  CHECK(json::parse(models.body).at("data").empty());

  service.model_manager()->add_chat_model("echo-model", make_local_chat_engine());
  models = http_request(service.port(), "GET", "/v1/models");
  auto listing = json::parse(models.body);
  REQUIRE(listing.at("data").size() == 1);
  CHECK(listing["data"][0]["id"] == "echo-model");
  CHECK(listing["data"][0]["object"] == "model");

  auto missing = http_request(service.port(), "GET", "/nope");
  CHECK(missing.status == 404);

  auto chat_404 = http_request(service.port(), "POST", "/v1/chat/completions",
                               R"({"model": "unknown", "messages": []})");
  CHECK(chat_404.status == 404);
  CHECK(json::parse(chat_404.body).at("error") == "Model not found");

  auto bad_json = http_request(service.port(), "POST", "/v1/chat/completions", "{nope");
  CHECK(bad_json.status == 400);

  service.stop();
}

TEST_CASE("http: unary chat completion folds the stream", "[llm][http]") {
  http::HttpService service({.host = "127.0.0.1", .port = 0});
  service.model_manager()->add_chat_model("echo-model", make_local_chat_engine());
  service.start();

  auto result = http_request(service.port(), "POST", "/v1/chat/completions",
                             R"({"model": "echo-model",
                                 "messages": [{"role": "user", "content": "hello"}]})");
  REQUIRE(result.status == 200);
  auto response = json::parse(result.body);
  CHECK(response.at("object") == "chat.completion");
  REQUIRE(response.at("choices").size() == 1);
  CHECK(response["choices"][0]["message"]["content"] == "hello");
  CHECK(response["choices"][0]["finish_reason"] == "stop");
  CHECK(response["usage"]["prompt_tokens"] == 5);

  // Metrics recorded the unary success.
  CHECK(service.metrics().request_count("echo-model", "chat_completions", "unary", "success") ==
        1);
  auto metrics = http_request(service.port(), "GET", "/metrics");
  CHECK_THAT(metrics.body,
             Catch::Matchers::ContainsSubstring(
                 "nv_llm_http_service_requests_total{model=\"echo-model\","
                 "endpoint=\"chat_completions\",request_type=\"unary\",status=\"success\"} 1"));
  CHECK_THAT(metrics.body, Catch::Matchers::ContainsSubstring(
                               "nv_llm_http_service_inflight_requests{model=\"echo-model\"} 0"));

  service.stop();
}

TEST_CASE("http: streaming chat completion emits SSE and [DONE]", "[llm][http]") {
  http::HttpService service({.host = "127.0.0.1", .port = 0});
  service.model_manager()->add_chat_model("echo-model", make_local_chat_engine());
  service.start();

  auto result = http_request(service.port(), "POST", "/v1/chat/completions",
                             R"({"model": "echo-model", "stream": true,
                                 "messages": [{"role": "user", "content": "hi"}],
                                 "nvext": {"annotations": ["formatted_prompt"]}})");
  REQUIRE(result.status == 200);

  // Parse the SSE body with our own decoder.
  SseDecoder decoder;
  auto messages = decoder.feed(result.body);

  REQUIRE(!messages.empty());
  // First event: the formatted_prompt annotation (kept because we preserve
  // nvext through the frontend, unlike Rust v0.1.0).
  CHECK(messages[0].event == "formatted_prompt");

  std::string content;
  std::optional<std::string> finish;
  for (const auto& message : messages) {
    if (!message.data.has_value()) continue;
    auto chunk = json::parse(*message.data)
                     .get<openai::NvCreateChatCompletionStreamResponse>();
    for (const auto& choice : chunk.choices) {
      if (choice.delta.content.has_value()) content += *choice.delta.content;
      if (choice.finish_reason.has_value()) finish = choice.finish_reason;
    }
  }
  CHECK(content == "hi");
  CHECK(finish == "stop");
  // The final "data: [DONE]" is consumed by the decoder as the sentinel.
  CHECK_THAT(result.body, Catch::Matchers::ContainsSubstring("data: [DONE]"));

  CHECK(service.metrics().request_count("echo-model", "chat_completions", "stream", "success") ==
        1);
  service.stop();
}

TEST_CASE("http: model watcher wires remote workers through discovery",
          "[llm][http][watcher]") {
  auto rt = dynamo::Runtime::create(test_config());
  {
    auto drt = dynamo::component::DistributedRuntime::create(rt, {});
    auto ns = unique_ns();

    // Frontend: service + watcher.
    http::HttpService service({.host = "127.0.0.1", .port = 0});
    service.start();
    std::string prefix = ns + "/components/http/models/";

    // Worker: serve the chat engine on an endpoint, then register the model
    // entry (as llmctl would), bound to the worker lease.
    auto endpoint = drt.ns(ns).component("backend").endpoint("generate");
    coro::sync_wait([&]() -> coro::Task<void> {
      co_await rt.primary().schedule();
      rt.spawn(endpoint.serve<openai::NvCreateChatCompletionRequest,
                              Annotated<openai::NvCreateChatCompletionStreamResponse>>(
          make_local_chat_engine()));
      rt.spawn(run_model_watcher(drt, service.model_manager(), prefix));

      // Wait until the instance is visible before publishing the entry.
      auto probe =
          co_await endpoint.client<openai::NvCreateChatCompletionRequest,
                                   Annotated<openai::NvCreateChatCompletionStreamResponse>>();
      co_await probe.wait_for_instances();

      auto lease = co_await drt.primary_lease();
      http::ModelEntry entry;
      entry.name = "remote-echo";
      entry.namespace_name = ns;
      entry.component = "backend";
      entry.endpoint = "generate";
      entry.model_type = "chat";
      co_await register_model_entry(*drt.discovery(), entry, prefix, lease.id);
    }());

    // The watcher picks the model up asynchronously.
    bool appeared = false;
    for (int i = 0; i < 100 && !appeared; ++i) {
      appeared = service.model_manager()->get_chat_engine("remote-echo") != nullptr;
      if (!appeared) std::this_thread::sleep_for(20ms);
    }
    REQUIRE(appeared);

    // And a real HTTP request round-trips through discovery + TCP planes.
    auto result = http_request(service.port(), "POST", "/v1/chat/completions",
                               R"({"model": "remote-echo",
                                   "messages": [{"role": "user", "content": "remote!"}]})");
    REQUIRE(result.status == 200);
    auto response = json::parse(result.body);
    CHECK(response["choices"][0]["message"]["content"] == "remote!");

    service.stop();
    // Closing discovery ends the watch stream (and with it the watcher task).
    drt.discovery()->shutdown();
    rt.shutdown();
    REQUIRE(rt.join_tasks(5000ms));
  }
}
