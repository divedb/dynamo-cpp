// SPDX-License-Identifier: Apache-2.0
//
// Hierarchical cancellation. Cancelling a token cancels all of its children;
// cancelling a child never affects the parent. Mirrors the semantics of
// tokio_util's CancellationToken used throughout Dynamo.

#pragma once

#include <chrono>
#include <functional>
#include <list>
#include <memory>
#include <mutex>

#include "runtime/coro/event.h"
#include "runtime/coro/task.h"

namespace dynamo {

class Executor;
class CancellationToken;

/// Free coroutine so the token is copied into the frame (safe on temporaries).
coro::Task<bool> wait_for_cancellation(CancellationToken token, Executor& executor,
                                       std::chrono::milliseconds timeout);

namespace detail {

struct CancellationState : std::enable_shared_from_this<CancellationState> {
  std::mutex mutex;
  bool cancelled = false;
  std::list<std::function<void()>> callbacks;
  coro::AsyncEvent event;

  // Keeps the registration on the parent alive for this state's lifetime.
  std::shared_ptr<void> parent_link;

  void cancel();
};

}  // namespace detail

/// RAII handle for a registered cancellation callback; unregisters on destruction.
class CancellationRegistration {
 public:
  CancellationRegistration() = default;
  CancellationRegistration(std::shared_ptr<detail::CancellationState> state,
                           std::list<std::function<void()>>::iterator it)
      : state_(std::move(state)), it_(it), armed_(true) {}
  CancellationRegistration(CancellationRegistration&& other) noexcept { *this = std::move(other); }
  CancellationRegistration& operator=(CancellationRegistration&& other) noexcept {
    reset();
    state_ = std::move(other.state_);
    it_ = other.it_;
    armed_ = std::exchange(other.armed_, false);
    return *this;
  }
  CancellationRegistration(const CancellationRegistration&) = delete;
  CancellationRegistration& operator=(const CancellationRegistration&) = delete;
  ~CancellationRegistration() { reset(); }

  void reset() {
    if (!armed_) return;
    armed_ = false;
    std::lock_guard lock(state_->mutex);
    if (!state_->cancelled) state_->callbacks.erase(it_);
    state_.reset();
  }

 private:
  std::shared_ptr<detail::CancellationState> state_;
  std::list<std::function<void()>>::iterator it_;
  bool armed_ = false;
};

class CancellationToken {
 public:
  CancellationToken() : state_(std::make_shared<detail::CancellationState>()) {}

  bool is_cancelled() const {
    std::lock_guard lock(state_->mutex);
    return state_->cancelled;
  }

  void cancel() const { state_->cancel(); }

  /// New token cancelled when this one is; cancelling the child is independent.
  CancellationToken child_token() const;

  /// Registers `fn` to run on cancellation (immediately if already cancelled,
  /// on the cancelling thread otherwise). Hold the registration to keep it.
  CancellationRegistration register_callback(std::function<void()> fn) const;

  /// Awaitable that resolves when the token is cancelled. Owns the token
  /// state, so it is safe to await on a temporary token.
  auto cancelled() const noexcept {
    struct Awaiter {
      std::shared_ptr<detail::CancellationState> s;
      bool await_ready() const { return s->event.is_set(); }
      bool await_suspend(std::coroutine_handle<> h) { return s->event.add_waiter(h); }
      void await_resume() const noexcept {}
    };
    return Awaiter{state_};
  }

  /// Waits for cancellation with a timeout; resumes on `executor`.
  /// Resolves true if cancelled, false on timeout.
  coro::Task<bool> wait_for(Executor& executor, std::chrono::milliseconds timeout) const {
    return wait_for_cancellation(*this, executor, timeout);
  }

  bool operator==(const CancellationToken& other) const { return state_ == other.state_; }

 private:
  explicit CancellationToken(std::shared_ptr<detail::CancellationState> s) : state_(std::move(s)) {}
  std::shared_ptr<detail::CancellationState> state_;
};

}  // namespace dynamo
