// SPDX-License-Identifier: Apache-2.0

#include "llm/kv_router/scheduler.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <spdlog/spdlog.h>

namespace dynamo::llm::kv {

ProcessedEndpoints::ProcessedEndpoints(std::vector<Endpoint> eps) : endpoints(std::move(eps)) {
  if (endpoints.empty()) return;
  double sum = 0;
  std::vector<double> ratios;
  ratios.reserve(endpoints.size());
  for (const auto& e : endpoints) {
    double ratio = e.data.kv_total_blocks == 0
                       ? 0.0
                       : static_cast<double>(e.data.kv_active_blocks) /
                             static_cast<double>(e.data.kv_total_blocks);
    ratios.push_back(ratio);
    sum += ratio;
  }
  load_avg = sum / static_cast<double>(ratios.size());
  double variance = 0;
  for (double r : ratios) variance += (r - load_avg) * (r - load_avg);
  variance /= static_cast<double>(ratios.size());
  load_std = std::sqrt(variance);
}

WorkerId select_worker(ProcessedEndpoints& workers, const OverlapScores& overlap,
                       size_t isl_tokens, size_t kv_block_size,
                       KVHitRateEvent* hit_rate_event) {
  if (workers.endpoints.empty()) throw NoEndpointsError();

  // Balance mode prioritizes evening out load across workers.
  constexpr double kBalanceThreshold = 0.1;
  bool balance_mode = workers.load_std > kBalanceThreshold * workers.load_avg;
  double alpha = balance_mode ? 0.7 : 0.3;
  constexpr double kGamma = 0.1;

  std::optional<size_t> best_index;
  double best_cost = std::numeric_limits<double>::infinity();

  for (size_t i = 0; i < workers.endpoints.size(); ++i) {
    const Endpoint& w = workers.endpoints[i];
    // Exclude workers at capacity.
    if (w.data.request_active_slots >= w.data.request_total_slots ||
        w.data.kv_active_blocks >= w.data.kv_total_blocks) {
      continue;
    }

    double kv_load_ratio = static_cast<double>(w.data.kv_active_blocks) /
                           static_cast<double>(w.data.kv_total_blocks);
    double load_deviation = kv_load_ratio - workers.load_avg;

    size_t overlap_tokens = static_cast<size_t>(overlap.score(w.worker_id)) * kv_block_size;
    size_t new_tokens = isl_tokens > overlap_tokens ? isl_tokens - overlap_tokens : 0;
    double normalized_new_tokens =
        isl_tokens == 0 ? 0.0
                        : static_cast<double>(new_tokens) / static_cast<double>(isl_tokens);

    double request_load_ratio = static_cast<double>(w.data.request_active_slots) /
                                static_cast<double>(w.data.request_total_slots);

    double cost = alpha * load_deviation + (1.0 - alpha) * normalized_new_tokens +
                  kGamma * request_load_ratio;

    spdlog::debug(
        "kv scheduler: worker {} load_deviation {:.4f} new_tokens {:.4f} request_load {:.4f} "
        "cost {:.4f}",
        w.worker_id, load_deviation, normalized_new_tokens, request_load_ratio, cost);

    if (cost < best_cost) {
      best_cost = cost;
      best_index = i;
    }
  }

  if (!best_index.has_value()) throw AllWorkersBusyError();

  Endpoint& chosen = workers.endpoints[*best_index];
  // Bump the snapshot's predicted load until the next scrape lands.
  // (Rust adds min(isl_blocks, 1) here — at most one block — which reads as
  // an inverted min/max; kept as-is for parity of tuning behavior.)
  size_t isl_blocks = kv_block_size == 0 ? 0 : isl_tokens / kv_block_size;
  chosen.data.request_active_slots += 1;
  chosen.data.kv_active_blocks += std::min<size_t>(isl_blocks, 1);

  if (hit_rate_event != nullptr) {
    hit_rate_event->worker_id = chosen.worker_id;
    hit_rate_event->isl_blocks = isl_blocks;
    hit_rate_event->overlap_blocks = overlap.score(chosen.worker_id);
  }

  spdlog::debug("kv scheduler: selected worker {} (cost {:.4f})", chosen.worker_id, best_cost);
  return chosen.worker_id;
}

void KvScheduler::update_endpoints(ProcessedEndpoints endpoints) {
  std::lock_guard lock(mutex_);
  endpoints_ = std::move(endpoints);
}

bool KvScheduler::has_endpoints() const {
  std::lock_guard lock(mutex_);
  return !endpoints_.endpoints.empty();
}

void KvScheduler::set_hit_rate_sink(std::function<void(const KVHitRateEvent&)> sink) {
  std::lock_guard lock(mutex_);
  hit_rate_sink_ = std::move(sink);
}

WorkerId KvScheduler::schedule(const OverlapScores& overlap, size_t isl_tokens,
                               KVHitRateEvent* out_event) {
  KVHitRateEvent event;
  std::function<void(const KVHitRateEvent&)> sink;
  WorkerId worker;
  {
    std::lock_guard lock(mutex_);
    worker = select_worker(endpoints_, overlap, isl_tokens, kv_block_size_, &event);
    sink = hit_rate_sink_;
  }
  if (out_event != nullptr) *out_event = event;
  if (sink) sink(event);
  return worker;
}

}  // namespace dynamo::llm::kv
