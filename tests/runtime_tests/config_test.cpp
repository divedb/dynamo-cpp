#include <catch2/catch_test_macros.hpp>
#include <dynamo/config.h>

using namespace dynamo;

TEST_CASE("RuntimeConfig loads with defaults", "[config]") {
    auto cfg = RuntimeConfig::load_from_env();
    CHECK_FALSE(cfg.runtime_name.empty());
    CHECK(cfg.num_worker_threads > 0);
    CHECK(cfg.max_blocking_threads > 0);
}

TEST_CASE("RuntimeConfig loads from environment", "[config]") {
    setenv("DYN_RUNTIME_NAME", "test-runtime", 1);
    setenv("DYN_RUNTIME_NUM_WORKER_THREADS", "8", 1);
    auto cfg = RuntimeConfig::load_from_env();
    CHECK(cfg.runtime_name == "test-runtime");
    CHECK(cfg.num_worker_threads == 8);
    unsetenv("DYN_RUNTIME_NAME");
    unsetenv("DYN_RUNTIME_NUM_WORKER_THREADS");
}

TEST_CASE("WorkerConfig loads defaults", "[config]") {
    auto cfg = WorkerConfig::load();
    CHECK(cfg.graceful_shutdown_timeout.count() > 0);
}

TEST_CASE("EtcdConfig loads defaults", "[config]") {
    auto cfg = EtcdConfig::load();
    CHECK_FALSE(cfg.endpoints.empty());
}

TEST_CASE("NatsConfig loads defaults", "[config]") {
    auto cfg = NatsConfig::load();
    CHECK_FALSE(cfg.server.empty());
}
