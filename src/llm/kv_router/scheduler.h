// SPDX-License-Identifier: Apache-2.0
//
// KV scheduler — Dynamo's kv_router/{scheduler,scoring}.rs: combines the
// indexer's overlap scores with scraped ForwardPassMetrics into a worker
// choice. cost = alpha*load_deviation + (1-alpha)*normalized_new_tokens
// + gamma*request_load_ratio, with alpha raised when load spread is high
// (balance mode).
//
// Deviations from Rust v0.1.0 (its own FIXMEs): worker_ids[i] is aligned
// with endpoints[i] (Rust collects them through a HashSet, misaligning the
// two arrays), and load statistics are computed over the kv load *ratio*
// (Rust averages absolute block counts but subtracts that average from a
// ratio). The scheduler is a snapshot + pure selection here rather than a
// channel-fed task; the aggregator calls update_endpoints().

#pragma once

#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "llm/kv_router/indexer.h"
#include "llm/kv_router/protocols.h"

namespace dynamo::llm::kv {

/// A worker endpoint with its latest forward-pass metrics.
struct Endpoint {
  std::string name;
  std::string subject;
  WorkerId worker_id = 0;
  ForwardPassMetrics data;
};

/// Endpoint set with precomputed load statistics (over kv load ratios).
struct ProcessedEndpoints {
  std::vector<Endpoint> endpoints;
  double load_avg = 0;
  double load_std = 0;

  ProcessedEndpoints() = default;
  explicit ProcessedEndpoints(std::vector<Endpoint> eps);
};

struct SchedulerError : std::runtime_error {
  using std::runtime_error::runtime_error;
};
struct NoEndpointsError : SchedulerError {
  NoEndpointsError() : SchedulerError("no endpoints available to route work") {}
};
struct AllWorkersBusyError : SchedulerError {
  AllWorkersBusyError() : SchedulerError("all workers busy") {}
};

/// Pure selection over a mutable endpoint snapshot (the chosen worker's
/// predicted load is bumped in place). Throws NoEndpointsError /
/// AllWorkersBusyError.
WorkerId select_worker(ProcessedEndpoints& workers, const OverlapScores& overlap,
                       size_t isl_tokens, size_t kv_block_size, KVHitRateEvent* hit_rate_event);

class KvScheduler {
 public:
  explicit KvScheduler(size_t kv_block_size) : kv_block_size_(kv_block_size) {}

  /// Replaces the endpoint snapshot (called by the metrics aggregator).
  void update_endpoints(ProcessedEndpoints endpoints);

  bool has_endpoints() const;

  /// Optional sink for KVHitRateEvents emitted on each decision.
  void set_hit_rate_sink(std::function<void(const KVHitRateEvent&)> sink);

  /// Picks a worker for a request; fills `out_event` (also delivered to the
  /// sink) when given. Throws SchedulerError subtypes.
  WorkerId schedule(const OverlapScores& overlap, size_t isl_tokens,
                    KVHitRateEvent* out_event = nullptr);

 private:
  mutable std::mutex mutex_;
  ProcessedEndpoints endpoints_;
  std::function<void(const KVHitRateEvent&)> hit_rate_sink_;
  size_t kv_block_size_;
};

}  // namespace dynamo::llm::kv
