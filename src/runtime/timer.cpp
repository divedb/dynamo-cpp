// SPDX-License-Identifier: Apache-2.0

#include "runtime/timer.h"

#include <condition_variable>
#include <map>
#include <mutex>
#include <thread>

#include <spdlog/spdlog.h>

namespace dynamo {

struct TimerQueue::Impl {
  std::mutex mutex;
  std::condition_variable cv;
  std::multimap<Clock::time_point, std::pair<uint64_t, std::function<void()>>> entries;
  uint64_t next_id = 1;
  bool stopping = false;
  std::thread thread;

  void run() {
    std::unique_lock lock(mutex);
    while (!stopping) {
      if (entries.empty()) {
        cv.wait(lock);
        continue;
      }
      auto next = entries.begin()->first;
      if (cv.wait_until(lock, next) == std::cv_status::no_timeout && entries.empty()) continue;

      auto now = Clock::now();
      while (!entries.empty() && entries.begin()->first <= now) {
        auto fn = std::move(entries.begin()->second.second);
        entries.erase(entries.begin());
        lock.unlock();
        try {
          fn();
        } catch (const std::exception& e) {
          spdlog::error("timer callback threw: {}", e.what());
        }
        lock.lock();
      }
    }
  }
};

TimerQueue::TimerQueue() : impl_(new Impl) {
  impl_->thread = std::thread([this] { impl_->run(); });
}

TimerQueue::~TimerQueue() {
  {
    std::lock_guard lock(impl_->mutex);
    impl_->stopping = true;
  }
  impl_->cv.notify_all();
  impl_->thread.join();
  delete impl_;
}

TimerQueue& TimerQueue::instance() {
  static TimerQueue queue;
  return queue;
}

uint64_t TimerQueue::schedule_at(Clock::time_point deadline, std::function<void()> fn) {
  uint64_t id;
  {
    std::lock_guard lock(impl_->mutex);
    id = impl_->next_id++;
    impl_->entries.emplace(deadline, std::make_pair(id, std::move(fn)));
  }
  impl_->cv.notify_all();
  return id;
}

bool TimerQueue::cancel(uint64_t id) {
  std::lock_guard lock(impl_->mutex);
  for (auto it = impl_->entries.begin(); it != impl_->entries.end(); ++it) {
    if (it->second.first == id) {
      impl_->entries.erase(it);
      return true;
    }
  }
  return false;
}

}  // namespace dynamo
