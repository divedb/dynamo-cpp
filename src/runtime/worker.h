// SPDX-License-Identifier: Apache-2.0
//
// Bootstrap wrapper: builds the Runtime, installs SIGINT/SIGTERM handling,
// runs the user's async app to completion, and enforces a graceful-shutdown
// window after cancellation (hard exit 911 on overrun) — mirroring Dynamo's
// Worker::execute contract.

#pragma once

#include <atomic>
#include <functional>

#include "runtime/coro/task.h"
#include "runtime/runtime.h"

namespace dynamo {

/// At most one Worker may be alive per process (it owns process-wide signal
/// handling), and execute() may run at most once per Worker. Unlike Dynamo's
/// Rust Worker (whose tokio runtime is a process-global, so a Worker may only
/// ever be created once), a new Worker may be constructed after the previous
/// one is destroyed — our runtimes are self-contained.
class Worker {
 public:
  static Worker from_env() { return Worker(RuntimeConfig::from_env()); }

  /// Throws std::runtime_error if another Worker is currently alive.
  explicit Worker(const RuntimeConfig& config) : slot_(), runtime_(Runtime::create(config)) {}

  Worker(const Worker&) = delete;
  Worker& operator=(const Worker&) = delete;
  Worker(Worker&&) = delete;
  Worker& operator=(Worker&&) = delete;

  Runtime& runtime() { return runtime_; }

  /// Blocks until `app` completes (or is cancelled and finishes shutting
  /// down). Returns 0 on success, 1 if the app threw. If the app overruns the
  /// graceful-shutdown window after cancellation, the process exits with 911.
  /// Throws std::logic_error if called more than once.
  int execute(std::function<coro::Task<void>(Runtime)> app);

  static constexpr int kGracefulShutdownExitCode = 911;

 private:
  // Claims the process-wide worker slot; declared before runtime_ so the
  // check runs before any pools are created.
  struct Slot {
    Slot();
    ~Slot();
  };

  Slot slot_;
  Runtime runtime_;
  std::atomic<bool> executed_{false};
};

}  // namespace dynamo
