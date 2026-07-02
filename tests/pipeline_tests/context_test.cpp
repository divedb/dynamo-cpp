#include <catch2/catch_test_macros.hpp>
#include <dynamo/pipeline.h>

using namespace dynamo;

TEST_CASE("Context holds value and metadata", "[pipeline][context]") {
    auto ctx = Context<int>(42);
    CHECK(ctx.value() == 42);

    ctx.registry().insert("key", std::string("value"));
    auto* val = ctx.registry().get<std::string>("key");
    REQUIRE(val != nullptr);
    CHECK(*val == "value");
}

TEST_CASE("Context can be mapped to new type", "[pipeline][context]") {
    auto ctx = Context<int>(42);
    auto mapped = ctx.map<std::string>([](const int& v) {
        return "number: " + std::to_string(v);
    });
    CHECK(mapped.value() == "number: 42");
}

TEST_CASE("Context transfers value preserving metadata", "[pipeline][context]") {
    auto ctx = Context<int>(42);
    ctx.registry().insert("tag", std::string("test"));
    auto transferred = ctx.transfer(std::string("hello"));
    CHECK(transferred.value() == "hello");
    auto* tag = transferred.registry().get<std::string>("tag");
    REQUIRE(tag != nullptr);
    CHECK(*tag == "test");
}

TEST_CASE("Context detects cancellation", "[pipeline][context]") {
    auto cancel_tok = CancellationToken::create_source();
    auto ctx = Context<int>(
        42, std::make_shared<AsyncEngineContext>(cancel_tok));
    CHECK_FALSE(ctx.is_cancelled());
    cancel_tok.cancel();
    CHECK(ctx.is_cancelled());
}
