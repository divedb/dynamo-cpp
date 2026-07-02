#include <catch2/catch_test_macros.hpp>
#include <dynamo/runtime.h>
#include <dynamo/engine.h>
#include <folly/coro/BlockingWait.h>

using namespace dynamo;

TEST_CASE("ResponseStream delivers values", "[pipeline][engine]") {
    auto runtime = Runtime::create(RuntimeConfig{});

    std::vector<std::string> received;
    auto ctx = std::make_shared<AsyncEngineContext>();
    auto stream = std::make_shared<ResponseStream<std::string>>(
        ctx,
        [&](std::string v) -> folly::coro::Task<void> {
            received.push_back(std::move(v));
            co_return;
        });

    REQUIRE(stream->is_active());

    auto task = stream->push("hello");
    folly::coro::blockingWait(std::move(task));

    CHECK(received.size() == 1);
    CHECK(received[0] == "hello");

    runtime->shutdown();
}

TEST_CASE("ResponseStream completes", "[pipeline][engine]") {
    auto runtime = Runtime::create(RuntimeConfig{});

    bool completed = false;
    auto ctx = std::make_shared<AsyncEngineContext>();
    auto stream = std::make_shared<ResponseStream<int>>(
        ctx, nullptr, [&]() { completed = true; });

    stream->complete();
    CHECK(completed);

    runtime->shutdown();
}

TEST_CASE("ResponseStream reports error", "[pipeline][engine]") {
    auto runtime = Runtime::create(RuntimeConfig{});

    bool errored = false;
    auto ctx = std::make_shared<AsyncEngineContext>();
    auto stream = std::make_shared<ResponseStream<int>>(
        ctx, nullptr, nullptr,
        [&](std::exception_ptr) { errored = true; });

    stream->error(std::make_exception_ptr(std::runtime_error("test")));
    CHECK(errored);

    runtime->shutdown();
}

TEST_CASE("AsyncEngineContext lifecycle", "[pipeline][engine]") {
    auto ctx = AsyncEngineContext();
    CHECK_FALSE(ctx.is_stopped());
    CHECK_FALSE(ctx.is_killed());

    ctx.stop_generating();
    CHECK(ctx.is_stopped());
    CHECK_FALSE(ctx.is_killed());

    ctx.kill();
    CHECK(ctx.is_killed());
}
