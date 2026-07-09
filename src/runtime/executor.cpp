// SPDX-License-Identifier: Apache-2.0

#include "runtime/executor.h"

#include <spdlog/spdlog.h>

namespace dynamo {

ThreadPool::ThreadPool(size_t num_threads, std::string name) : name_(std::move(name)) {
  if (num_threads == 0) num_threads = 1;
  threads_.reserve(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
    threads_.emplace_back([this] { run(); });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard lock(mutex_);
    stopping_ = true;
  }
  cv_.notify_all();
  for (auto& t : threads_) t.join();
}

void ThreadPool::post(std::function<void()> fn) {
  {
    std::lock_guard lock(mutex_);
    if (stopping_) {
      spdlog::warn("ThreadPool[{}]: post() after shutdown; dropping work", name_);
      return;
    }
    queue_.push_back(std::move(fn));
  }
  cv_.notify_one();
}

void ThreadPool::run() {
  for (;;) {
    std::function<void()> fn;
    {
      std::unique_lock lock(mutex_);
      cv_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
      if (queue_.empty()) return;  // stopping and drained
      fn = std::move(queue_.front());
      queue_.pop_front();
    }
    try {
      fn();
    } catch (const std::exception& e) {
      spdlog::error("ThreadPool[{}]: task threw: {}", name_, e.what());
    } catch (...) {
      spdlog::error("ThreadPool[{}]: task threw unknown exception", name_);
    }
  }
}

namespace {

struct DetachedTask {
  struct promise_type {
    DetachedTask get_return_object() { return {}; }
    std::suspend_never initial_suspend() noexcept { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    void return_void() noexcept {}
    void unhandled_exception() noexcept {
      spdlog::error("detached task: unhandled exception escaped");
    }
  };
};

DetachedTask run_detached(Executor& executor, coro::Task<void> task, TaskTracker* tracker,
                          std::shared_ptr<TaskHandle::State> state) {
  struct Guard {
    TaskTracker* tracker;
    ~Guard() {
      if (tracker) tracker->remove();
    }
  } guard{tracker};

  co_await executor.schedule();
  std::exception_ptr error;
  try {
    // Owned in a nested scope so the inner frame (and anything it captured,
    // e.g. the last Runtime handle) is destroyed BEFORE the tracker guard
    // fires. Frame locals destroy in reverse construction order: without
    // this, tracker.remove() would run first, letting the owner tear down
    // thread pools while this pool thread still holds runtime references —
    // a self-join in ~ThreadPool.
    coro::Task<void> owned = std::move(task);
    co_await std::move(owned);
  } catch (const std::exception& e) {
    error = std::current_exception();
    spdlog::error("detached task failed: {}", e.what());
  } catch (...) {
    error = std::current_exception();
    spdlog::error("detached task failed with unknown exception");
  }
  state->complete(error);
}

}  // namespace

TaskHandle spawn_detached(Executor& executor, coro::Task<void> task, TaskTracker* tracker) {
  if (tracker) tracker->add();
  auto state = std::make_shared<TaskHandle::State>();
  run_detached(executor, std::move(task), tracker, state);
  return TaskHandle(std::move(state));
}

}  // namespace dynamo
