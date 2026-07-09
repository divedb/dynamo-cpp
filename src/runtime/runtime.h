// SPDX-License-Identifier: Apache-2.0
//
// Process-local runtime: primary + background thread pools, a process-unique
// worker id, and the root CancellationToken. Cheap to copy (shared handle).

#pragma once

#include <memory>
#include <string>

#include "runtime/cancellation.h"
#include "runtime/config.h"
#include "runtime/coro/task.h"
#include "runtime/executor.h"

namespace dynamo {

class Runtime {
 public:
  /// Creates a runtime with owned thread pools.
  static Runtime create(const RuntimeConfig& config = RuntimeConfig::from_env());
  static Runtime single_threaded() { return create(RuntimeConfig::single_threaded()); }

  /// Creates a runtime over application-owned schedulers (Dynamo's
  /// Runtime::from_handle role). `post_primary` runs application work;
  /// `post_secondary` runs background chores (defaults to the primary).
  /// The schedulers must outlive the runtime and all its tasks.
  static Runtime from_executors(ExternalExecutor::PostFn post_primary,
                                ExternalExecutor::PostFn post_secondary = nullptr);

  /// Unique identifier for this runtime/worker process.
  const std::string& id() const;

  /// Primary (application) executor.
  Executor& primary() const;

  /// Secondary executor for background chores (keep-alives, watchers).
  Executor& secondary() const;

  /// Root cancellation token; cancelling it shuts the whole runtime down.
  CancellationToken token() const;
  CancellationToken child_token() const;
  void shutdown() const { token().cancel(); }

  /// Runs a detached task on the primary pool, tracked for teardown.
  /// The returned handle observes completion/failure; safe to discard.
  TaskHandle spawn(coro::Task<void> task) const;

  /// Runs a detached task on the secondary pool, tracked for teardown.
  TaskHandle spawn_background(coro::Task<void> task) const;

  /// Blocks until all tracked tasks finished. Returns false on timeout.
  bool join_tasks(std::chrono::milliseconds timeout = std::chrono::milliseconds::max()) const;

  const RuntimeConfig& config() const;

  bool operator==(const Runtime& other) const { return state_ == other.state_; }

 private:
  struct State;
  explicit Runtime(std::shared_ptr<State> s) : state_(std::move(s)) {}
  std::shared_ptr<State> state_;
};

}  // namespace dynamo
