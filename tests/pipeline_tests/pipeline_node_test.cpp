#include <catch2/catch_test_macros.hpp>
#include <dynamo/runtime.h>
#include <dynamo/engine.h>
#include <dynamo/pipeline.h>
#include <folly/coro/BlockingWait.h>

using namespace dynamo;

TEST_CASE("PipelineNode transforms input to output", "[pipeline][node]") {
    auto runtime = Runtime::create(RuntimeConfig{});

    std::string received;
    auto ctx = std::make_shared<AsyncEngineContext>();
    auto stream = std::make_shared<ResponseStream<std::string>>(
        ctx,
        [&](std::string v) -> folly::coro::Task<void> {
            received = std::move(v);
            co_return;
        });

    PipelineNode<int, std::string> node(
        [](int input) -> folly::coro::Task<std::string> {
            co_return "value: " + std::to_string(input);
        });

    auto task = node.generate(Context<int>(42), stream);
    folly::coro::blockingWait(std::move(task));

    CHECK(received == "value: 42");

    runtime->shutdown();
}

TEST_CASE("PipelineNode handles errors", "[pipeline][node]") {
    auto runtime = Runtime::create(RuntimeConfig{});

    bool errored = false;
    auto ctx = std::make_shared<AsyncEngineContext>();
    auto stream = std::make_shared<ResponseStream<int>>(
        ctx, nullptr, nullptr,
        [&](std::exception_ptr) { errored = true; });

    PipelineNode<int, int> node(
        [](int) -> folly::coro::Task<int> {
            throw std::runtime_error("test error");
            co_return 0;
        });

    auto task = node.generate(Context<int>(1), stream);
    folly::coro::blockingWait(std::move(task));

    CHECK(errored);

    runtime->shutdown();
}
