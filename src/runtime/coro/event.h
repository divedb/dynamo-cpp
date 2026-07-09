// SPDX-License-Identifier: Apache-2.0
//
// Thread-safe async manual-reset event. Multiple coroutines may await it;
// set() resumes all current and future waiters — on the setter's thread by
// default, or via a resume hook (see set_resume_hook) when the setter must
// not run arbitrary continuations (locks held, I/O threads).

#pragma once

#include <coroutine>
#include <functional>
#include <mutex>
#include <vector>

namespace dynamo::coro {

class AsyncEvent {
 public:
  using ResumeHook = std::function<void(std::coroutine_handle<>)>;

  bool is_set() const {
    std::lock_guard lock(mutex_);
    return set_;
  }

  /// Routes waiter resumption through `hook` (e.g. resume_on(pool)). Set
  /// before waiters can park; not synchronized against a concurrent set().
  void set_resume_hook(ResumeHook hook) {
    std::lock_guard lock(mutex_);
    resume_hook_ = std::move(hook);
  }

  void set() {
    std::vector<std::coroutine_handle<>> waiters;
    ResumeHook hook;
    {
      std::lock_guard lock(mutex_);
      if (set_) return;
      set_ = true;
      waiters.swap(waiters_);
      hook = resume_hook_;
    }
    for (auto h : waiters) {
      if (hook) {
        hook(h);
      } else {
        h.resume();
      }
    }
  }

  /// Parks `h` unless the event is already set. Returns false if `h` should
  /// resume immediately. Building block for external awaiters.
  bool add_waiter(std::coroutine_handle<> h) {
    std::lock_guard lock(mutex_);
    if (set_) return false;
    waiters_.push_back(h);
    return true;
  }

  auto wait() noexcept {
    struct Awaiter {
      AsyncEvent& event;
      bool await_ready() const { return event.is_set(); }
      bool await_suspend(std::coroutine_handle<> h) { return event.add_waiter(h); }
      void await_resume() const noexcept {}
    };
    return Awaiter{*this};
  }

 private:
  mutable std::mutex mutex_;
  bool set_ = false;
  ResumeHook resume_hook_;
  std::vector<std::coroutine_handle<>> waiters_;
};

}  // namespace dynamo::coro
