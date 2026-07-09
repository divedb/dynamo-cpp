// SPDX-License-Identifier: Apache-2.0
//
// Asynchronous pull-based generator. Consumed with:
//   while (auto item = co_await gen.next()) { use(*item); }
// Values are yielded by value or movable lvalue (do not co_yield a const local).

#pragma once

#include <coroutine>
#include <exception>
#include <optional>
#include <utility>

namespace dynamo::coro {

template <typename T>
class AsyncGenerator {
 public:
  struct promise_type {
    AsyncGenerator get_return_object() noexcept {
      return AsyncGenerator(std::coroutine_handle<promise_type>::from_promise(*this));
    }
    std::suspend_always initial_suspend() noexcept { return {}; }

    struct YieldAwaiter {
      bool await_ready() noexcept { return false; }
      std::coroutine_handle<> await_suspend(std::coroutine_handle<promise_type> h) noexcept {
        auto cont = h.promise().consumer_;
        return cont ? cont : std::noop_coroutine();
      }
      void await_resume() noexcept {}
    };

    YieldAwaiter final_suspend() noexcept { return {}; }

    YieldAwaiter yield_value(T&& value) noexcept {
      value_ = std::addressof(value);
      return {};
    }
    YieldAwaiter yield_value(T& value) noexcept {
      value_ = std::addressof(value);
      return {};
    }

    void return_void() noexcept {}
    void unhandled_exception() noexcept { exception_ = std::current_exception(); }

    T* value_ = nullptr;
    std::exception_ptr exception_ = nullptr;
    std::coroutine_handle<> consumer_ = nullptr;
  };

  AsyncGenerator() noexcept = default;
  explicit AsyncGenerator(std::coroutine_handle<promise_type> h) noexcept : handle_(h) {}
  AsyncGenerator(AsyncGenerator&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
  AsyncGenerator& operator=(AsyncGenerator&& other) noexcept {
    if (this != &other) {
      if (handle_) handle_.destroy();
      handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
  }
  AsyncGenerator(const AsyncGenerator&) = delete;
  AsyncGenerator& operator=(const AsyncGenerator&) = delete;
  ~AsyncGenerator() {
    if (handle_) handle_.destroy();
  }

  bool valid() const noexcept { return handle_ != nullptr; }

  /// Awaitable advancing the generator; resolves to the next item or nullopt.
  auto next() noexcept {
    struct NextAwaiter {
      std::coroutine_handle<promise_type> handle;
      bool await_ready() const noexcept { return !handle || handle.done(); }
      std::coroutine_handle<> await_suspend(std::coroutine_handle<> consumer) noexcept {
        handle.promise().consumer_ = consumer;
        handle.promise().value_ = nullptr;
        return handle;
      }
      std::optional<T> await_resume() {
        if (!handle) return std::nullopt;
        auto& p = handle.promise();
        if (p.exception_) std::rethrow_exception(p.exception_);
        if (handle.done() || p.value_ == nullptr) return std::nullopt;
        return std::optional<T>(std::move(*p.value_));
      }
    };
    return NextAwaiter{handle_};
  }

 private:
  std::coroutine_handle<promise_type> handle_ = nullptr;
};

}  // namespace dynamo::coro
