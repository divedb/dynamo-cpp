// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include <atomic>

#include "runtime/cancellation.h"
#include "runtime/coro/sync_wait.h"
#include "runtime/runtime.h"
#include "runtime/worker.h"

using namespace dynamo;
using namespace std::chrono_literals;

namespace {

RuntimeConfig test_config() {
  RuntimeConfig config;
  config.num_worker_threads = 4;
  config.num_background_threads = 2;
  config.graceful_shutdown_timeout = 5s;
  return config;
}

}  // namespace

TEST_CASE("cancellation cancels children but not parents", "[cancellation]") {
  CancellationToken root;
  auto child = root.child_token();
  auto grandchild = child.child_token();

  child.cancel();
  REQUIRE(child.is_cancelled());
  REQUIRE(grandchild.is_cancelled());
  REQUIRE_FALSE(root.is_cancelled());

  root.cancel();
  REQUIRE(root.is_cancelled());
}

TEST_CASE("cancellation callbacks fire once, immediately if already cancelled", "[cancellation]") {
  CancellationToken token;
  std::atomic<int> calls{0};

  auto registration = token.register_callback([&] { calls++; });
  token.cancel();
  token.cancel();
  REQUIRE(calls == 1);

  auto late = token.register_callback([&] { calls++; });
  REQUIRE(calls == 2);
}

TEST_CASE("dropped registration does not fire", "[cancellation]") {
  CancellationToken token;
  std::atomic<int> calls{0};
  {
    auto registration = token.register_callback([&] { calls++; });
  }
  token.cancel();
  REQUIRE(calls == 0);
}

TEST_CASE("wait_for resolves on cancel and on timeout", "[cancellation]") {
  auto rt = Runtime::create(test_config());

  SECTION("timeout") {
    bool cancelled = coro::sync_wait(rt.token().wait_for(rt.primary(), 50ms));
    REQUIRE_FALSE(cancelled);
  }

  SECTION("cancel") {
    std::thread canceller([&] {
      std::this_thread::sleep_for(30ms);
      rt.shutdown();
    });
    bool cancelled = coro::sync_wait(rt.token().wait_for(rt.primary(), 10s));
    REQUIRE(cancelled);
    canceller.join();
  }

  rt.shutdown();
  REQUIRE(rt.join_tasks(1000ms));
}

TEST_CASE("runtime startup, spawn tracking, shutdown", "[runtime]") {
  auto rt = Runtime::create(test_config());
  REQUIRE_FALSE(rt.id().empty());

  std::atomic<int> completed{0};
  for (int i = 0; i < 16; ++i) {
    rt.spawn([](std::atomic<int>& done) -> coro::Task<void> {
      done++;
      co_return;
    }(completed));
  }
  REQUIRE(rt.join_tasks(2000ms));
  REQUIRE(completed == 16);

  REQUIRE_FALSE(rt.token().is_cancelled());
  rt.shutdown();
  REQUIRE(rt.token().is_cancelled());
}

TEST_CASE("background tasks observe shutdown", "[runtime]") {
  auto rt = Runtime::create(test_config());
  std::atomic<bool> saw_cancel{false};

  rt.spawn_background([](Runtime rt, std::atomic<bool>& flag) -> coro::Task<void> {
    co_await rt.token().cancelled();
    flag = true;
  }(rt, saw_cancel));

  rt.shutdown();
  REQUIRE(rt.join_tasks(2000ms));
  REQUIRE(saw_cancel);
}

TEST_CASE("task handles observe completion and failures", "[runtime][handles]") {
  auto rt = Runtime::create(test_config());

  SECTION("success") {
    auto handle = rt.spawn([]() -> coro::Task<void> { co_return; }());
    REQUIRE(handle.sync_join(2000ms));
    REQUIRE(handle.finished());
  }

  SECTION("failure is rethrown on join") {
    auto handle = rt.spawn([]() -> coro::Task<void> {
      throw std::runtime_error("task boom");
      co_return;  // makes this a coroutine: the throw happens on resume
    }());
    REQUIRE_THROWS_WITH(handle.sync_join(2000ms), "task boom");
  }

  SECTION("coroutine join") {
    auto handle = rt.spawn([]() -> coro::Task<void> {
      throw std::runtime_error("async boom");
      co_return;
    }());
    REQUIRE_THROWS_WITH(coro::sync_wait(handle.join()), "async boom");
  }

  rt.shutdown();
  REQUIRE(rt.join_tasks(2000ms));
}

TEST_CASE("runtime over external executors runs tasks on the provided scheduler",
          "[runtime][executors]") {
  // The application owns this pool; the runtime only borrows it.
  ThreadPool app_pool(2, "app-owned");
  std::atomic<int> posted{0};

  auto rt = Runtime::from_executors([&](std::function<void()> fn) {
    posted++;
    app_pool.post(std::move(fn));
  });

  std::atomic<bool> ran{false};
  auto handle = rt.spawn([](std::atomic<bool>& flag) -> coro::Task<void> {
    flag = true;
    co_return;
  }(ran));
  REQUIRE(handle.sync_join(2000ms));
  REQUIRE(ran);
  REQUIRE(posted > 0);  // work actually flowed through the app's scheduler

  rt.shutdown();
  REQUIRE(rt.join_tasks(2000ms));
}

TEST_CASE("config layers TOML under environment overrides", "[runtime][config]") {
  std::string path = "/tmp/dynamo_cpp_test_config.toml";
  {
    FILE* f = fopen(path.c_str(), "w");
    REQUIRE(f);
    fputs("[runtime]\nworker_threads = 3\ngraceful_shutdown_timeout_s = 17\n", f);
    fclose(f);
  }
  setenv("DYN_CONFIG", path.c_str(), 1);
  unsetenv("DYN_RUNTIME_NUM_WORKER_THREADS");

  auto config = RuntimeConfig::from_env();
  REQUIRE(config.num_worker_threads == 3);
  REQUIRE(config.graceful_shutdown_timeout == std::chrono::seconds(17));

  // Environment wins over the file.
  setenv("DYN_RUNTIME_NUM_WORKER_THREADS", "5", 1);
  auto overridden = RuntimeConfig::from_env();
  REQUIRE(overridden.num_worker_threads == 5);

  unsetenv("DYN_RUNTIME_NUM_WORKER_THREADS");
  unsetenv("DYN_CONFIG");
  remove(path.c_str());
}

TEST_CASE("only one worker may be alive; execute is one-shot", "[worker]") {
  SECTION("concurrent second worker is rejected") {
    Worker first(test_config());
    REQUIRE_THROWS(Worker(test_config()));
  }

  SECTION("sequential workers are allowed") {
    { Worker first(test_config()); }
    Worker second(test_config());
    REQUIRE(second.execute([](Runtime) -> coro::Task<void> { co_return; }) == 0);
  }

  SECTION("second execute on the same worker throws") {
    Worker worker(test_config());
    REQUIRE(worker.execute([](Runtime) -> coro::Task<void> { co_return; }) == 0);
    REQUIRE_THROWS_AS(worker.execute([](Runtime) -> coro::Task<void> { co_return; }),
                      std::logic_error);
  }
}

TEST_CASE("worker executes app and returns its status", "[worker]") {
  SECTION("success") {
    Worker worker(test_config());
    int code = worker.execute([](Runtime) -> coro::Task<void> { co_return; });
    REQUIRE(code == 0);
  }

  SECTION("app error maps to exit code 1") {
    Worker worker(test_config());
    int code = worker.execute(
        [](Runtime) -> coro::Task<void> { throw std::runtime_error("app failed"); });
    REQUIRE(code == 1);
  }

  SECTION("app that waits for shutdown finishes when cancelled") {
    Worker worker(test_config());
    int code = worker.execute([](Runtime rt) -> coro::Task<void> {
      rt.spawn_background([](Runtime rt) -> coro::Task<void> {
        rt.shutdown();  // simulate an external shutdown signal
        co_return;
      }(rt));
      co_await rt.token().cancelled();
    });
    REQUIRE(code == 0);
  }
}
