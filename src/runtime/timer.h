// SPDX-License-Identifier: Apache-2.0
//
// Process-wide timer queue. Entries are cancellable so completed waits do not
// pin resources until their deadline.

#pragma once

#include <chrono>
#include <cstdint>
#include <functional>

namespace dynamo {

class TimerQueue {
 public:
  using Clock = std::chrono::steady_clock;

  static TimerQueue& instance();

  /// Schedules `fn` to run on the timer thread at `deadline`. Callbacks must
  /// be brief and non-blocking (post to an executor for real work).
  uint64_t schedule_at(Clock::time_point deadline, std::function<void()> fn);

  /// Cancels a pending entry. Returns false if it already fired or was cancelled.
  bool cancel(uint64_t id);

 private:
  TimerQueue();
  ~TimerQueue();
  struct Impl;
  Impl* impl_;
};

}  // namespace dynamo
