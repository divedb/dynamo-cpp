// SPDX-License-Identifier: Apache-2.0
//
// KvRouter — Dynamo's kv_router.rs + metrics_aggregator.rs: KV-aware worker
// selection as a routing policy in front of `Client::direct`. Owns
//   - a KvIndexer fed by the component's kv_events subscription
//   - a KvScheduler fed by periodic ForwardPassMetrics scrapes of the
//     component's load_metrics endpoint
// schedule(token_ids) hashes the request into blocks, scores the overlap,
// and picks the cheapest worker; hit-rate events go out on kv-hit-rate.
//
// Background tasks are explicit coroutines to spawn on the runtime; they end
// when the runtime shuts down.

#pragma once

#include <memory>

#include "component/component.h"
#include "llm/kv_router/indexer.h"
#include "llm/kv_router/scheduler.h"

namespace dynamo::llm::kv {

class KvRouter : public std::enable_shared_from_this<KvRouter> {
 public:
  KvRouter(component::Component component, size_t kv_block_size)
      : component_(std::move(component)),
        indexer_(std::make_shared<KvIndexer>(kv_block_size)),
        scheduler_(std::make_shared<KvScheduler>(kv_block_size)) {}

  /// Consumes the component's kv_events subscription into the indexer.
  /// Spawn on the runtime; ends when the event stream closes or the runtime
  /// shuts down.
  coro::Task<void> run_event_consumer();

  /// Scrapes load_metrics stats into the scheduler every `interval`.
  /// Spawn on the runtime; ends on runtime shutdown.
  coro::Task<void> run_metrics_aggregator(
      std::chrono::milliseconds interval = std::chrono::milliseconds(100),
      std::chrono::milliseconds scrape_timeout = std::chrono::milliseconds(300));

  /// Waits until at least one worker's metrics have been scraped.
  coro::Task<void> wait_for_endpoints();

  /// Picks the best worker for a tokenized request; publishes a hit-rate
  /// event. Throws SchedulerError subtypes when nothing can be scheduled.
  coro::Task<WorkerId> schedule(const std::vector<Token>& token_ids);

  KvIndexer& indexer() { return *indexer_; }
  KvScheduler& scheduler() { return *scheduler_; }
  const component::Component& component() const { return component_; }

 private:
  component::Component component_;
  std::shared_ptr<KvIndexer> indexer_;
  std::shared_ptr<KvScheduler> scheduler_;
};

}  // namespace dynamo::llm::kv
