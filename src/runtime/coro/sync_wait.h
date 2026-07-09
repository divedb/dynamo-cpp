// SPDX-License-Identifier: Apache-2.0
//
// Blocks the calling thread until a Task completes. Bridge from synchronous
// code (main, tests) into the coroutine world.

#pragma once

#include <coroutine>
#include <exception>
#include <optional>
#include <semaphore>

#include "runtime/coro/task.h"

namespace dynamo::coro {

namespace detail {

struct SyncWaitTask {
  struct promise_type {
    SyncWaitTask get_return_object() { return {}; }
    std::suspend_never initial_suspend() noexcept { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    void return_void() noexcept {}
    void unhandled_exception() noexcept { std::terminate(); }
  };
};

template <typename T>
SyncWaitTask sync_wait_impl(Task<T> task, std::binary_semaphore& done,
                            std::optional<T>& result, std::exception_ptr& error) {
  try {
    result.emplace(co_await std::move(task));
  } catch (...) {
    error = std::current_exception();
  }
  done.release();
}

inline SyncWaitTask sync_wait_impl(Task<void> task, std::binary_semaphore& done,
                                   std::exception_ptr& error) {
  try {
    co_await std::move(task);
  } catch (...) {
    error = std::current_exception();
  }
  done.release();
}

}  // namespace detail

template <typename T>
T sync_wait(Task<T> task) {
  std::binary_semaphore done(0);
  std::optional<T> result;
  std::exception_ptr error;
  detail::sync_wait_impl(std::move(task), done, result, error);
  done.acquire();
  if (error) std::rethrow_exception(error);
  return std::move(*result);
}

inline void sync_wait(Task<void> task) {
  std::binary_semaphore done(0);
  std::exception_ptr error;
  detail::sync_wait_impl(std::move(task), done, error);
  done.acquire();
  if (error) std::rethrow_exception(error);
}

}  // namespace dynamo::coro
