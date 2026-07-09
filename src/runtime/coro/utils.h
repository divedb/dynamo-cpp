// SPDX-License-Identifier: Apache-2.0
//
// Stream and pooling utilities (Dynamo's utils/stream.rs and utils/pool.rs).

#pragma once

#include <chrono>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "runtime/coro/async_generator.h"

namespace dynamo::coro {

/// Bounds a generator by a deadline: items yielded after `deadline` are cut
/// off and the stream ends early (Dynamo's DeadlineStream/until_deadline).
/// The check applies between items — an in-flight await on the source is not
/// interrupted, matching the Rust adaptor's poll-based semantics.
template <typename T>
AsyncGenerator<T> until_deadline(AsyncGenerator<T> source,
                                 std::chrono::steady_clock::time_point deadline) {
  while (std::chrono::steady_clock::now() < deadline) {
    auto item = co_await source.next();
    if (!item) co_return;
    if (std::chrono::steady_clock::now() >= deadline) co_return;
    co_yield *item;
  }
}

template <typename T>
AsyncGenerator<T> until_timeout(AsyncGenerator<T> source, std::chrono::milliseconds timeout) {
  return until_deadline(std::move(source), std::chrono::steady_clock::now() + timeout);
}

/// A returnable object pool (Dynamo's utils/pool.rs). Acquired items return
/// to the pool automatically when their handle is destroyed; `on_return` is
/// invoked to reset them. Exhausted pools report nullopt (no blocking).
template <typename T>
class Pool : public std::enable_shared_from_this<Pool<T>> {
 public:
  /// RAII handle; releases the item back to its pool on destruction.
  class Item {
   public:
    Item() = default;
    Item(Item&& other) noexcept
        : pool_(std::move(other.pool_)), value_(std::move(other.value_)) {
      other.value_.reset();
    }
    Item& operator=(Item&& other) noexcept {
      release();
      pool_ = std::move(other.pool_);
      value_ = std::move(other.value_);
      other.value_.reset();
      return *this;
    }
    Item(const Item&) = delete;
    Item& operator=(const Item&) = delete;
    ~Item() { release(); }

    bool has_value() const { return value_.has_value(); }
    T& operator*() { return *value_; }
    T* operator->() { return &*value_; }

    /// Detaches the value from the pool (it will not be returned).
    T take() && {
      T out = std::move(*value_);
      value_.reset();
      return out;
    }

   private:
    friend class Pool;
    Item(std::shared_ptr<Pool> pool, T value) : pool_(std::move(pool)), value_(std::move(value)) {}

    void release() {
      if (!value_) return;
      if (auto pool = pool_.lock()) pool->put_back(std::move(*value_));
      value_.reset();
    }

    std::weak_ptr<Pool> pool_;
    std::optional<T> value_;
  };

  static std::shared_ptr<Pool> create(std::vector<T> initial,
                                      std::function<void(T&)> on_return = nullptr) {
    auto pool = std::shared_ptr<Pool>(new Pool(std::move(on_return)));
    for (auto& value : initial) pool->idle_.push_back(std::move(value));
    return pool;
  }

  /// Takes an item if one is available.
  std::optional<Item> try_acquire() {
    std::lock_guard lock(mutex_);
    if (idle_.empty()) return std::nullopt;
    T value = std::move(idle_.front());
    idle_.pop_front();
    return Item(this->shared_from_this(), std::move(value));
  }

  size_t available() const {
    std::lock_guard lock(mutex_);
    return idle_.size();
  }

 private:
  explicit Pool(std::function<void(T&)> on_return) : on_return_(std::move(on_return)) {}

  void put_back(T value) {
    if (on_return_) on_return_(value);
    std::lock_guard lock(mutex_);
    idle_.push_back(std::move(value));
  }

  mutable std::mutex mutex_;
  std::deque<T> idle_;
  std::function<void(T&)> on_return_;
};

}  // namespace dynamo::coro
