// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include "runtime/coro/async_generator.h"
#include "runtime/coro/channel.h"
#include "runtime/coro/event.h"
#include "runtime/coro/sync_wait.h"
#include "runtime/coro/task.h"
#include "runtime/coro/utils.h"

using namespace dynamo;

namespace {

coro::Task<int> forty_two() { co_return 42; }

coro::Task<int> add(int a, int b) {
  int x = co_await forty_two();
  co_return a + b + x - 42;
}

coro::Task<int> throws() {
  co_await forty_two();
  throw std::runtime_error("boom");
}

coro::AsyncGenerator<int> count_to(int n) {
  for (int i = 0; i < n; ++i) co_yield i;
}

}  // namespace

TEST_CASE("task returns value and chains", "[coro]") {
  REQUIRE(coro::sync_wait(forty_two()) == 42);
  REQUIRE(coro::sync_wait(add(1, 2)) == 3);
}

TEST_CASE("task propagates exceptions", "[coro]") {
  REQUIRE_THROWS_WITH(coro::sync_wait(throws()), "boom");
}

TEST_CASE("async generator yields all items then ends", "[coro]") {
  auto result = coro::sync_wait([]() -> coro::Task<std::vector<int>> {
    std::vector<int> out;
    auto gen = count_to(5);
    while (auto v = co_await gen.next()) out.push_back(*v);
    co_return out;
  }());
  REQUIRE(result == std::vector<int>{0, 1, 2, 3, 4});
}

TEST_CASE("channel bridges threads and coroutines", "[coro]") {
  auto [tx, rx] = coro::make_channel<int>(4);

  std::thread producer([tx = std::move(tx)]() mutable {
    for (int i = 0; i < 100; ++i) REQUIRE(tx.send(i));
    // Sender dropped at scope end → channel closes after drain.
  });

  auto sum = coro::sync_wait([](coro::Receiver<int> rx) -> coro::Task<int> {
    int total = 0;
    while (auto v = co_await rx.recv()) total += *v;
    co_return total;
  }(std::move(rx)));

  producer.join();
  REQUIRE(sum == 4950);
}

TEST_CASE("channel send fails after receiver closes", "[coro]") {
  auto [tx, rx] = coro::make_channel<int>(2);
  rx.close();
  REQUIRE_FALSE(tx.send(1));
}

TEST_CASE("until_deadline cuts a stream off at the deadline", "[coro][utils]") {
  auto slow_counter = []() -> coro::AsyncGenerator<int> {
    for (int i = 0;; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      co_yield i;
    }
  };

  auto result = coro::sync_wait([&]() -> coro::Task<int> {
    auto bounded =
        coro::until_timeout(slow_counter(), std::chrono::milliseconds(100));
    int count = 0;
    while (auto item = co_await bounded.next()) ++count;
    co_return count;
  }());
  REQUIRE(result >= 1);
  REQUIRE(result < 20);  // an unbounded stream would keep going
}

TEST_CASE("pool recycles items through RAII handles", "[coro][utils]") {
  int resets = 0;
  auto pool = coro::Pool<std::string>::create(
      {"a", "b"}, [&resets](std::string& s) {
        s.clear();
        ++resets;
      });

  {
    auto first = pool->try_acquire();
    auto second = pool->try_acquire();
    REQUIRE(first);
    REQUIRE(second);
    REQUIRE_FALSE(pool->try_acquire().has_value());  // exhausted
    **first += "-used";
  }  // both returned (and reset)

  REQUIRE(pool->available() == 2);
  REQUIRE(resets == 2);
  auto again = pool->try_acquire();
  REQUIRE(again);
  REQUIRE((*again)->empty());  // on_return cleared it

  // take() detaches from the pool permanently.
  { auto taken = std::move(*pool->try_acquire()).take(); }
  REQUIRE(pool->available() == 0);
}

TEST_CASE("async event resume hook routes waiters", "[coro]") {
  coro::AsyncEvent event;
  std::atomic<int> hook_calls{0};
  event.set_resume_hook([&hook_calls](std::coroutine_handle<> h) {
    hook_calls++;
    h.resume();
  });

  std::thread setter([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    event.set();
  });
  coro::sync_wait([&]() -> coro::Task<void> { co_await event.wait(); }());
  setter.join();
  REQUIRE(hook_calls == 1);
}

TEST_CASE("async event wakes waiters", "[coro]") {
  coro::AsyncEvent event;
  std::thread setter([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    event.set();
  });
  coro::sync_wait([&]() -> coro::Task<void> { co_await event.wait(); }());
  REQUIRE(event.is_set());
  setter.join();
}
