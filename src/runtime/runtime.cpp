// SPDX-License-Identifier: Apache-2.0

#include "runtime/runtime.h"

#include <atomic>
#include <random>

#include <spdlog/spdlog.h>

namespace dynamo {

namespace {

std::string generate_worker_id() {
  // 128-bit random hex id (UUID-shaped without the dashes).
  std::random_device rd;
  std::mt19937_64 gen((static_cast<uint64_t>(rd()) << 32) ^ rd());
  char buf[33];
  snprintf(buf, sizeof(buf), "%016llx%016llx",
           static_cast<unsigned long long>(gen()), static_cast<unsigned long long>(gen()));
  return std::string(buf);
}

}  // namespace

struct Runtime::State {
  State(const RuntimeConfig& cfg, std::shared_ptr<Executor> primary_executor,
        std::shared_ptr<Executor> secondary_executor)
      : config(cfg),
        id(generate_worker_id()),
        primary(std::move(primary_executor)),
        secondary(std::move(secondary_executor)) {}

  RuntimeConfig config;
  std::string id;
  std::shared_ptr<Executor> primary;
  std::shared_ptr<Executor> secondary;
  CancellationToken root;
  TaskTracker tracker;
};

Runtime Runtime::create(const RuntimeConfig& config) {
  auto state = std::make_shared<State>(
      config, std::make_shared<ThreadPool>(config.num_worker_threads, "primary"),
      std::make_shared<ThreadPool>(config.num_background_threads, "secondary"));
  spdlog::debug("runtime {} created (primary={} secondary={})", state->id,
                config.num_worker_threads, config.num_background_threads);
  return Runtime(std::move(state));
}

Runtime Runtime::from_executors(ExternalExecutor::PostFn post_primary,
                                ExternalExecutor::PostFn post_secondary) {
  auto primary = std::make_shared<ExternalExecutor>(std::move(post_primary), "external-primary");
  auto secondary =
      post_secondary
          ? std::make_shared<ExternalExecutor>(std::move(post_secondary), "external-secondary")
          : primary;
  auto state = std::make_shared<State>(RuntimeConfig{}, std::move(primary), std::move(secondary));
  spdlog::debug("runtime {} created over external executors", state->id);
  return Runtime(std::move(state));
}

const std::string& Runtime::id() const { return state_->id; }
Executor& Runtime::primary() const { return *state_->primary; }
Executor& Runtime::secondary() const { return *state_->secondary; }
CancellationToken Runtime::token() const { return state_->root; }
CancellationToken Runtime::child_token() const { return state_->root.child_token(); }
const RuntimeConfig& Runtime::config() const { return state_->config; }

TaskHandle Runtime::spawn(coro::Task<void> task) const {
  return spawn_detached(*state_->primary, std::move(task), &state_->tracker);
}

TaskHandle Runtime::spawn_background(coro::Task<void> task) const {
  return spawn_detached(*state_->secondary, std::move(task), &state_->tracker);
}

bool Runtime::join_tasks(std::chrono::milliseconds timeout) const {
  return state_->tracker.join(timeout);
}

}  // namespace dynamo
