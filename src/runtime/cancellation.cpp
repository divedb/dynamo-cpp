// SPDX-License-Identifier: Apache-2.0

#include "runtime/cancellation.h"

#include <coroutine>
#include <vector>

#include "runtime/executor.h"
#include "runtime/timer.h"

namespace dynamo {

namespace detail {

void CancellationState::cancel() {
  std::list<std::function<void()>> to_run;
  {
    std::lock_guard lock(mutex);
    if (cancelled) return;
    cancelled = true;
    to_run.swap(callbacks);
  }
  for (auto& fn : to_run) fn();
  event.set();
}

}  // namespace detail

CancellationToken CancellationToken::child_token() const {
  auto child_state = std::make_shared<detail::CancellationState>();
  auto weak_child = std::weak_ptr<detail::CancellationState>(child_state);
  auto registration = register_callback([weak_child] {
    if (auto child = weak_child.lock()) child->cancel();
  });
  child_state->parent_link =
      std::make_shared<CancellationRegistration>(std::move(registration));
  return CancellationToken(std::move(child_state));
}

CancellationRegistration CancellationToken::register_callback(std::function<void()> fn) const {
  {
    std::unique_lock lock(state_->mutex);
    if (!state_->cancelled) {
      state_->callbacks.push_back(std::move(fn));
      auto it = std::prev(state_->callbacks.end());
      return CancellationRegistration(state_, it);
    }
  }
  fn();  // already cancelled: fire inline
  return CancellationRegistration();
}

namespace {

struct TimedWaitState {
  std::mutex mutex;
  bool done = false;
  bool was_cancelled = false;
  std::coroutine_handle<> handle;
  Executor* executor = nullptr;
  uint64_t timer_id = 0;
  CancellationRegistration registration;

  void complete(bool cancelled_flag) {
    std::coroutine_handle<> h;
    {
      std::lock_guard lock(mutex);
      if (done) return;
      done = true;
      was_cancelled = cancelled_flag;
      h = handle;
    }
    if (cancelled_flag) TimerQueue::instance().cancel(timer_id);
    if (h) executor->post([h] { h.resume(); });
  }
};

struct TimedWaitAwaiter {
  CancellationToken token;
  Executor& executor;
  std::chrono::milliseconds timeout;
  std::shared_ptr<TimedWaitState> state;

  bool await_ready() const noexcept { return false; }
  void await_suspend(std::coroutine_handle<> h) {
    state = std::make_shared<TimedWaitState>();
    state->handle = h;
    state->executor = &executor;
    auto s = state;
    // Arm the timer first so a cancel-during-registration still completes once.
    state->timer_id = TimerQueue::instance().schedule_at(
        TimerQueue::Clock::now() + timeout, [s] { s->complete(false); });
    state->registration = token.register_callback([s] { s->complete(true); });
  }
  bool await_resume() const noexcept { return state->was_cancelled; }
};

}  // namespace

coro::Task<bool> wait_for_cancellation(CancellationToken token, Executor& executor,
                                       std::chrono::milliseconds timeout) {
  if (token.is_cancelled()) co_return true;
  co_return co_await TimedWaitAwaiter{token, executor, timeout, nullptr};
}

}  // namespace dynamo
