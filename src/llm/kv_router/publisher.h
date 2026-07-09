// SPDX-License-Identifier: Apache-2.0
//
// Worker-side KV publishing — Dynamo's kv_router/publisher.rs:
//   KvEventPublisher: emits RouterEvents on the component's kv_events subject
//   KvMetricsPublisher: holds the latest ForwardPassMetrics and serves the
//     load_metrics endpoint whose stats handler exposes them to scrapes

#pragma once

#include <memory>
#include <mutex>

#include "component/component.h"
#include "llm/kv_router/protocols.h"
#include "pipeline/annotated.h"

namespace dynamo::llm::kv {

/// Publishes this worker's KV cache events for router indexers.
class KvEventPublisher {
 public:
  KvEventPublisher(component::Component component, WorkerId worker_id, size_t kv_block_size)
      : component_(std::move(component)), worker_id_(worker_id), kv_block_size_(kv_block_size) {}

  size_t kv_block_size() const { return kv_block_size_; }
  WorkerId worker_id() const { return worker_id_; }

  coro::Task<void> publish(KvCacheEvent event) const {
    RouterEvent router_event{worker_id_, std::move(event)};
    co_await component_.publish(kKvEventSubject, nlohmann::json(router_event));
  }

 private:
  component::Component component_;
  WorkerId worker_id_;
  size_t kv_block_size_;
};

/// Holds the worker's latest ForwardPassMetrics and serves them: the
/// load_metrics endpoint's stats handler returns them to scrape_stats, and
/// the endpoint itself answers unary queries with the current snapshot.
class KvMetricsPublisher : public std::enable_shared_from_this<KvMetricsPublisher> {
 public:
  void publish(ForwardPassMetrics metrics) {
    std::lock_guard lock(mutex_);
    metrics_ = std::move(metrics);
  }

  ForwardPassMetrics current() const {
    std::lock_guard lock(mutex_);
    return metrics_;
  }

  /// Serves the component's load_metrics endpoint (spawn on the runtime;
  /// runs for the endpoint's lifetime, like Endpoint::serve). Pass a lease
  /// to identify this worker instance (defaults to the primary lease).
  coro::Task<void> create_endpoint(component::Component component,
                                   std::optional<discovery::Lease> lease = std::nullopt) const;

 private:
  mutable std::mutex mutex_;
  ForwardPassMetrics metrics_;
};

}  // namespace dynamo::llm::kv
