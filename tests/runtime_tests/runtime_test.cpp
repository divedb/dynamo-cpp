#include <catch2/catch_test_macros.hpp>
#include <dynamo/runtime.h>

using namespace dynamo;

TEST_CASE("Runtime creates and shuts down", "[runtime]") {
    RuntimeConfig cfg;
    cfg.num_worker_threads = 2;

    auto rt = Runtime::create(cfg);
    CHECK_FALSE(rt->id().empty());
    CHECK_FALSE(rt->cancellation_token().is_cancelled());

    rt->shutdown();
    CHECK(rt->cancellation_token().is_cancelled());
}

TEST_CASE("Runtime provides executors", "[runtime]") {
    RuntimeConfig cfg;
    cfg.num_worker_threads = 4;

    auto rt = Runtime::create(cfg);
    CHECK(rt->primary().getPoolId() != nullptr);
    CHECK(rt->io().getPoolId() != nullptr);
    CHECK(rt->secondary().getPoolId() != nullptr);

    rt->shutdown();
}

TEST_CASE("Multiple runtimes can coexist", "[runtime]") {
    auto rt1 = Runtime::create(RuntimeConfig{});
    auto rt2 = Runtime::create(RuntimeConfig{});

    CHECK(rt1->id() != rt2->id());

    rt1->shutdown();
    rt2->shutdown();
}
