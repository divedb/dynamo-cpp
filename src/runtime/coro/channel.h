// SPDX-License-Identifier: Apache-2.0
//
// Bounded MPMC channel bridging threads and coroutines: producers may block
// (sync_send) or fail fast (try_send); consumers may block (sync_recv) or
// await (recv). The channel closes when all Senders or all Receivers are
// gone, or explicitly via close().

#pragma once

#include <condition_variable>
#include <coroutine>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

namespace dynamo::coro {

/// How a channel resumes parked consumers. Defaults to inline (on the
/// producer's thread); pass an executor-posting hook to decouple producer
/// threads from consumer continuations (avoids reentrancy deadlocks when
/// producers hold locks or must keep servicing I/O).
using ResumeHook = std::function<void(std::coroutine_handle<>)>;

namespace detail {

template <typename T>
struct ChannelState {
  explicit ChannelState(size_t capacity, ResumeHook resume_hook)
      : capacity(capacity), resume_hook(std::move(resume_hook)) {}

  void resume(std::coroutine_handle<> h) {
    if (resume_hook) {
      resume_hook(h);
    } else {
      h.resume();
    }
  }

  std::mutex mutex;
  std::condition_variable cv_send;  // senders waiting for space
  std::condition_variable cv_recv;  // sync receivers waiting for items
  std::deque<T> queue;
  const size_t capacity;
  ResumeHook resume_hook;
  bool closed = false;
  size_t senders = 0;
  size_t receivers = 0;

  struct AsyncWaiter {
    std::coroutine_handle<> handle;
    std::optional<T>* slot;
  };
  std::deque<AsyncWaiter> async_receivers;

  void close_locked(std::deque<AsyncWaiter>& out_waiters) {
    if (closed) return;
    closed = true;
    out_waiters.swap(async_receivers);
    cv_send.notify_all();
    cv_recv.notify_all();
  }

  void close() {
    std::deque<AsyncWaiter> waiters;
    {
      std::lock_guard lock(mutex);
      close_locked(waiters);
    }
    for (auto& w : waiters) {
      w.slot->reset();
      resume(w.handle);
    }
  }

  // Returns a waiter to resume outside the lock, if a parked receiver took the value.
  bool push(T value, std::optional<AsyncWaiter>& woken) {
    std::unique_lock lock(mutex);
    cv_send.wait(lock, [&] { return closed || queue.size() < capacity || !async_receivers.empty(); });
    if (closed) return false;
    if (!async_receivers.empty()) {
      woken = std::move(async_receivers.front());
      async_receivers.pop_front();
      woken->slot->emplace(std::move(value));
      return true;
    }
    queue.push_back(std::move(value));
    cv_recv.notify_one();
    return true;
  }
};

}  // namespace detail

template <typename T>
class Sender {
 public:
  Sender() = default;
  explicit Sender(std::shared_ptr<detail::ChannelState<T>> s) : state_(std::move(s)) {
    if (state_) {
      std::lock_guard lock(state_->mutex);
      ++state_->senders;
    }
  }
  Sender(const Sender& other) : Sender(other.state_) {}
  Sender(Sender&& other) noexcept : state_(std::move(other.state_)) {}
  Sender& operator=(Sender other) noexcept {
    std::swap(state_, other.state_);
    return *this;
  }
  ~Sender() { release(); }

  /// Blocks while the channel is full. Returns false if the channel is closed.
  bool send(T value) {
    if (!state_) return false;
    std::optional<typename detail::ChannelState<T>::AsyncWaiter> woken;
    bool ok = state_->push(std::move(value), woken);
    if (woken) state_->resume(woken->handle);
    return ok;
  }

  void close() {
    if (state_) state_->close();
  }

  explicit operator bool() const { return state_ != nullptr; }

 private:
  void release() {
    if (!state_) return;
    bool last = false;
    {
      std::lock_guard lock(state_->mutex);
      last = (--state_->senders == 0);
    }
    if (last) state_->close();
    state_.reset();
  }

  std::shared_ptr<detail::ChannelState<T>> state_;
};

template <typename T>
class Receiver {
 public:
  Receiver() = default;
  explicit Receiver(std::shared_ptr<detail::ChannelState<T>> s) : state_(std::move(s)) {
    if (state_) {
      std::lock_guard lock(state_->mutex);
      ++state_->receivers;
    }
  }
  Receiver(const Receiver& other) : Receiver(other.state_) {}
  Receiver(Receiver&& other) noexcept : state_(std::move(other.state_)) {}
  Receiver& operator=(Receiver other) noexcept {
    std::swap(state_, other.state_);
    return *this;
  }
  ~Receiver() { release(); }

  /// Awaitable receive; resolves to nullopt once the channel is closed and drained.
  auto recv() noexcept {
    struct Awaiter {
      std::shared_ptr<detail::ChannelState<T>> state;
      std::optional<T> slot;

      bool await_ready() const noexcept { return state == nullptr; }
      bool await_suspend(std::coroutine_handle<> h) {
        std::lock_guard lock(state->mutex);
        if (!state->queue.empty()) {
          slot.emplace(std::move(state->queue.front()));
          state->queue.pop_front();
          state->cv_send.notify_one();
          return false;
        }
        if (state->closed) return false;  // resolves to nullopt
        state->async_receivers.push_back({h, &slot});
        return true;
      }
      std::optional<T> await_resume() noexcept { return std::move(slot); }
    };
    return Awaiter{state_, std::nullopt};
  }

  /// Blocking receive for plain threads.
  std::optional<T> sync_recv() {
    if (!state_) return std::nullopt;
    std::unique_lock lock(state_->mutex);
    state_->cv_recv.wait(lock, [&] { return state_->closed || !state_->queue.empty(); });
    if (state_->queue.empty()) return std::nullopt;
    T value = std::move(state_->queue.front());
    state_->queue.pop_front();
    state_->cv_send.notify_one();
    return value;
  }

  void close() {
    if (state_) state_->close();
  }

  explicit operator bool() const { return state_ != nullptr; }

 private:
  void release() {
    if (!state_) return;
    bool last = false;
    {
      std::lock_guard lock(state_->mutex);
      last = (--state_->receivers == 0);
    }
    if (last) state_->close();
    state_.reset();
  }

  std::shared_ptr<detail::ChannelState<T>> state_;
};

template <typename T>
std::pair<Sender<T>, Receiver<T>> make_channel(size_t capacity, ResumeHook resume_hook = nullptr) {
  auto state = std::make_shared<detail::ChannelState<T>>(capacity, std::move(resume_hook));
  return {Sender<T>(state), Receiver<T>(state)};
}

}  // namespace dynamo::coro
