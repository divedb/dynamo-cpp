// SPDX-License-Identifier: Apache-2.0
//
// Lazy coroutine task with symmetric-transfer continuations (cppcoro-style).

#pragma once

#include <coroutine>
#include <exception>
#include <stdexcept>
#include <utility>
#include <variant>

namespace dynamo::coro {

template <typename T>
class Task;

namespace detail {

class TaskPromiseBase {
 public:
  std::suspend_always initial_suspend() noexcept { return {}; }

  struct FinalAwaiter {
    bool await_ready() noexcept { return false; }
    template <typename P>
    std::coroutine_handle<> await_suspend(std::coroutine_handle<P> h) noexcept {
      auto cont = h.promise().continuation_;
      return cont ? cont : std::noop_coroutine();
    }
    void await_resume() noexcept {}
  };

  FinalAwaiter final_suspend() noexcept { return {}; }

  void set_continuation(std::coroutine_handle<> c) noexcept { continuation_ = c; }

 protected:
  std::coroutine_handle<> continuation_ = nullptr;
};

template <typename T>
class TaskPromise final : public TaskPromiseBase {
 public:
  Task<T> get_return_object() noexcept;

  template <typename U>
    requires std::convertible_to<U&&, T>
  void return_value(U&& value) {
    result_.template emplace<1>(std::forward<U>(value));
  }

  void unhandled_exception() noexcept { result_.template emplace<2>(std::current_exception()); }

  T take_result() {
    if (result_.index() == 2) std::rethrow_exception(std::get<2>(result_));
    if (result_.index() != 1) throw std::logic_error("task completed without a value");
    return std::move(std::get<1>(result_));
  }

 private:
  std::variant<std::monostate, T, std::exception_ptr> result_;
};

template <>
class TaskPromise<void> final : public TaskPromiseBase {
 public:
  Task<void> get_return_object() noexcept;
  void return_void() noexcept {}
  void unhandled_exception() noexcept { exception_ = std::current_exception(); }
  void take_result() {
    if (exception_) std::rethrow_exception(exception_);
  }

 private:
  std::exception_ptr exception_ = nullptr;
};

}  // namespace detail

/// A lazily-started coroutine returning `T`. Move-only; must be awaited (or
/// passed to sync_wait / spawn) exactly once.
template <typename T = void>
class [[nodiscard]] Task {
 public:
  using promise_type = detail::TaskPromise<T>;

  Task() noexcept = default;
  explicit Task(std::coroutine_handle<promise_type> h) noexcept : handle_(h) {}
  Task(Task&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
  Task& operator=(Task&& other) noexcept {
    if (this != &other) {
      if (handle_) handle_.destroy();
      handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
  }
  Task(const Task&) = delete;
  Task& operator=(const Task&) = delete;
  ~Task() {
    if (handle_) handle_.destroy();
  }

  bool valid() const noexcept { return handle_ != nullptr; }

  auto operator co_await() && noexcept {
    struct Awaiter {
      std::coroutine_handle<promise_type> handle;
      bool await_ready() const noexcept { return !handle || handle.done(); }
      std::coroutine_handle<> await_suspend(std::coroutine_handle<> awaiting) noexcept {
        handle.promise().set_continuation(awaiting);
        return handle;
      }
      T await_resume() { return handle.promise().take_result(); }
    };
    return Awaiter{handle_};
  }

 private:
  std::coroutine_handle<promise_type> handle_ = nullptr;
};

namespace detail {

template <typename T>
Task<T> TaskPromise<T>::get_return_object() noexcept {
  return Task<T>(std::coroutine_handle<TaskPromise<T>>::from_promise(*this));
}

inline Task<void> TaskPromise<void>::get_return_object() noexcept {
  return Task<void>(std::coroutine_handle<TaskPromise<void>>::from_promise(*this));
}

}  // namespace detail

}  // namespace dynamo::coro
