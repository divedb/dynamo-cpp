// SPDX-License-Identifier: Apache-2.0
//
// Context metadata (registry/stages/map) and composable pipeline operators.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <string>

#include "pipeline/engine.h"
#include "pipeline/operators.h"
#include "runtime/coro/sync_wait.h"

using namespace dynamo;

namespace {

coro::AsyncGenerator<std::string> echo_chars(std::string data) {
  for (char c : data) {
    std::string item(1, c);
    co_yield item;
  }
}

struct EchoEngine final : pipeline::AsyncEngine<std::string, std::string> {
  coro::Task<pipeline::ManyOut<std::string>> generate(
      pipeline::SingleIn<std::string> in) override {
    auto [data, controller] = std::move(in).into_parts();
    pipeline::ContextPtr ctx = controller;
    co_return pipeline::ManyOut<std::string>(echo_chars(std::move(data)), ctx);
  }
};

}  // namespace

TEST_CASE("context registry stores shared and unique values", "[pipeline][context]") {
  pipeline::Context<int> ctx(7);

  ctx.registry().insert_shared("model", std::string("llama"));
  ctx.registry().insert_unique("budget", 42);

  auto model = ctx.registry().get<std::string>("model");
  REQUIRE(model);
  REQUIRE(*model == "llama");
  REQUIRE(ctx.registry().get<int>("model") == nullptr);  // type mismatch

  auto budget = ctx.registry().take_unique<int>("budget");
  REQUIRE(budget == 42);
  REQUIRE_FALSE(ctx.registry().take_unique<int>("budget").has_value());  // taken

  // Registry and stages survive transfer and map.
  ctx.add_stage("ingest");
  auto [old_payload, next] = std::move(ctx).transfer(std::string("payload"));
  REQUIRE(old_payload == 7);
  REQUIRE(next.registry().get<std::string>("model") != nullptr);
  REQUIRE(next.stages() == std::vector<std::string>{"ingest"});

  auto mapped = std::move(next).map([](std::string s) { return s.size(); });
  REQUIRE(mapped.payload() == 7u);
  REQUIRE(mapped.registry().get<std::string>("model") != nullptr);
}

TEST_CASE("map operator transforms request and response stream", "[pipeline][operators]") {
  // Uppercase the request on the way down, bracket each item on the way up.
  auto op = pipeline::make_map_operator<std::string, std::string, std::string, std::string>(
      "upper",
      [](std::string req) {
        std::transform(req.begin(), req.end(), req.begin(),
                       [](unsigned char c) { return std::toupper(c); });
        return req;
      },
      [](std::string item) { return "[" + item + "]"; });

  auto engine = pipeline::link(op, pipeline::EnginePtr<std::string, std::string>(
                                       std::make_shared<EchoEngine>()));

  auto result = coro::sync_wait([&]() -> coro::Task<std::string> {
    auto out = co_await engine->generate(pipeline::SingleIn<std::string>("abc"));
    std::string collected;
    while (auto item = co_await out.next()) collected += *item;
    co_return collected;
  }());
  REQUIRE(result == "[A][B][C]");
}

TEST_CASE("operators compose into multi-stage pipelines", "[pipeline][operators]") {
  auto shout = pipeline::make_map_operator<std::string, std::string, std::string, std::string>(
      "shout", [](std::string req) { return req + "!"; },
      [](std::string item) { return item; });
  auto twice = pipeline::make_map_operator<std::string, std::string, std::string, std::string>(
      "twice", [](std::string req) { return req + req; },
      [](std::string item) { return item + item; });

  auto engine = pipeline::link(
      shout, pipeline::link(twice, pipeline::EnginePtr<std::string, std::string>(
                                       std::make_shared<EchoEngine>())));

  auto result = coro::sync_wait([&]() -> coro::Task<std::string> {
    // shout: "a" -> "a!" ; twice: -> "a!a!" ; echo streams 4 chars, each
    // doubled by `twice`'s backward map on the way up.
    auto out = co_await engine->generate(pipeline::SingleIn<std::string>("a"));
    std::string collected;
    while (auto item = co_await out.next()) collected += *item;
    co_return collected;
  }());
  REQUIRE(result == "aa!!aa!!");
}
