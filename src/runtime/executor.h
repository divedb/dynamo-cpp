// SPDX-License-Identifier: Apache-2.0
//
// Thread-pool executor with a coroutine `schedule()` awaitable, plus
// detached-task spawning with a tracker for deterministic teardown.

#pragma once

#include <chrono>
#include <condition_variable>
#include <coroutine>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "runtime/coro/event.h"
#include "runtime/coro/task.h"

namespace dynamo {

/// Where work runs. ThreadPool is the owned implementation; ExternalExecutor
/// adapts a caller-provided scheduler (Dynamo's Runtime::from_handle role).
class Executor {
 public:
  virtual ~Executor() = default;
  virtual void post(std::function<void()> fn) = 0;
  virtual const std::string& name() const = 0;

  /// co_await executor.schedule() resumes the coroutine on the executor.
  auto schedule() noexcept {
    struct Awaiter {
      Executor& executor;
      bool await_ready() const noexcept { return false; }
      void await_suspend(std::coroutine_handle<> h) {
        executor.post([h] { h.resume(); });
      }
      void await_resume() const noexcept {}
    };
    return Awaiter{*this};
  }
};

class ThreadPool final : public Executor {
 public:
  explicit ThreadPool(size_t num_threads, std::string name = "pool");
  ~ThreadPool() override;

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  void post(std::function<void()> fn) override;
  size_t size() const { return threads_.size(); }
  const std::string& name() const override { return name_; }

 private:
  void run();

  std::string name_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> queue_;
  bool stopping_ = false;
  std::vector<std::thread> threads_;
};

/// Adapts an application-owned scheduler. The post function must run the
/// callable exactly once, on some thread, and must outlive the runtime.
class ExternalExecutor final : public Executor {
 public:
  using PostFn = std::function<void(std::function<void()>)>;

  explicit ExternalExecutor(PostFn post_fn, std::string name = "external")
      : post_fn_(std::move(post_fn)), name_(std::move(name)) {}

  void post(std::function<void()> fn) override { post_fn_(std::move(fn)); }
  const std::string& name() const override { return name_; }

 private:
  PostFn post_fn_;
  std::string name_;
};

/// Counts in-flight detached tasks so owners can drain them before teardown.
class TaskTracker {
 public:
  void add() {
    std::lock_guard lock(mutex_);
    ++count_;
  }
  void remove() {
    std::lock_guard lock(mutex_);
    if (--count_ == 0) cv_.notify_all();
  }
  /// Blocks until all tracked tasks finished. Returns false on timeout.
  bool join(std::chrono::milliseconds timeout = std::chrono::milliseconds::max()) {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, timeout, [&] { return count_ == 0; });
  }
  size_t count() const {
    std::lock_guard lock(mutex_);
    return count_;
  }

 private:
  mutable std::mutex mutex_;
  mutable std::condition_variable cv_;
  size_t count_ = 0;
};

/// Observes a spawned task: completion, and the exception if it failed
/// (Dynamo's ExecutionHandle role). Cheap to copy; safe to discard.
class TaskHandle {
 public:
  TaskHandle() = default;

  bool valid() const { return state_ != nullptr; }
  bool finished() const { return state_ && state_->finished.load(std::memory_order_acquire); }

  /// Awaits completion; rethrows the task's exception if it failed.
  coro::Task<void> join() {
    auto s = state_;
    if (!s) co_return;
    co_await s->done.wait();
    if (s->error) std::rethrow_exception(s->error);
  }

  /// Blocking join for non-coroutine contexts; rethrows on failure.
  /// Returns false on timeout.
  bool sync_join(std::chrono::milliseconds timeout = std::chrono::milliseconds::max()) {
    auto s = state_;
    if (!s) return true;
    {
      std::unique_lock lock(s->mutex);
      if (!s->cv.wait_for(lock, timeout, [&] { return s->finished.load(); })) return false;
    }
    if (s->error) std::rethrow_exception(s->error);
    return true;
  }

  struct State {
    coro::AsyncEvent done;
    std::exception_ptr error;
    std::atomic<bool> finished{false};
    std::mutex mutex;
    std::condition_variable cv;

    void complete(std::exception_ptr e) {
      error = e;
      {
        std::lock_guard lock(mutex);
        finished.store(true, std::memory_order_release);
      }
      cv.notify_all();
      done.set();
    }
  };

  explicit TaskHandle(std::shared_ptr<State> s) : state_(std::move(s)) {}

 private:
  std::shared_ptr<State> state_;
};

/// Runs `task` detached on `executor`. If `tracker` is non-null the task is
/// tracked until completion. Exceptions escaping the task are logged and
/// stored in the returned handle.
TaskHandle spawn_detached(Executor& executor, coro::Task<void> task,
                          TaskTracker* tracker = nullptr);

/// Channel resume hook posting consumer continuations onto `executor` instead
/// of running them on the producer's thread. The executor must outlive the
/// channel's activity (true for runtime executors + tracked tasks).
inline std::function<void(std::coroutine_handle<>)> resume_on(Executor& executor) {
  return [&executor](std::coroutine_handle<> h) { executor.post([h] { h.resume(); }); };
}

}  // namespace dynamo
