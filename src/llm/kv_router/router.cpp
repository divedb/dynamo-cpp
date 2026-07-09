// SPDX-License-Identifier: Apache-2.0

#include "llm/kv_router/router.h"

#include <spdlog/spdlog.h>

namespace dynamo::llm::kv {

namespace {

// Free coroutines own a shared_ptr to the router so spawned frames never
// dangle (see docs/architecture.md on member coroutines).

coro::Task<void> event_consumer(std::shared_ptr<KvRouter> self) {
  auto component = self->component();
  auto stream = co_await component.subscribe(kKvEventSubject);

  // Unblock recv on runtime shutdown.
  auto token = component.drt().runtime().child_token();
  auto registration =
      token.register_callback([events = stream.events]() mutable { events.close(); });

  spdlog::debug("kv router: event consumer started on {}", component.path());
  while (auto event = co_await stream.events.recv()) {
    try {
      auto router_event = nlohmann::json::parse(event->payload).get<RouterEvent>();
      self->indexer().apply_event(router_event);
    } catch (const std::exception& e) {
      spdlog::warn("kv router: dropping malformed kv event: {}", e.what());
    }
  }
  spdlog::debug("kv router: event consumer ended");
}

coro::Task<void> metrics_aggregator(std::shared_ptr<KvRouter> self,
                                    std::chrono::milliseconds interval,
                                    std::chrono::milliseconds scrape_timeout) {
  auto component = self->component();
  auto runtime = component.drt().runtime();
  auto token = runtime.child_token();

  while (true) {
    bool cancelled = co_await wait_for_cancellation(token, runtime.primary(), interval);
    if (cancelled) break;

    component::ServiceSet stats;
    try {
      stats = co_await component.scrape_stats(scrape_timeout);
    } catch (const std::exception& e) {
      spdlog::warn("kv router: failed to scrape metrics: {}", e.what());
      continue;
    }

    std::vector<Endpoint> endpoints;
    for (const auto& entry : stats.endpoints) {
      if (!entry.ok || entry.info.endpoint != kKvMetricsEndpoint) continue;
      try {
        Endpoint endpoint;
        endpoint.name = entry.info.endpoint;
        endpoint.subject = entry.info.subject;
        // The worker id is the instance (lease) id — no subject parsing
        // needed, unlike Rust's hex-suffix split.
        endpoint.worker_id = entry.info.instance_id;
        // Custom stats-handler output rides under "custom" in the scrape.
        const auto& stats_json =
            entry.stats.contains("custom") ? entry.stats.at("custom") : entry.stats;
        endpoint.data = stats_json.get<ForwardPassMetrics>();
        endpoints.push_back(std::move(endpoint));
      } catch (const std::exception& e) {
        spdlog::debug("kv router: skipping stats not parseable as ForwardPassMetrics: {}",
                      e.what());
      }
    }
    self->scheduler().update_endpoints(ProcessedEndpoints(std::move(endpoints)));
  }
  spdlog::debug("kv router: metrics aggregator ended");
}

coro::Task<void> wait_endpoints(std::shared_ptr<KvRouter> self) {
  auto runtime = self->component().drt().runtime();
  auto token = runtime.child_token();
  while (!self->scheduler().has_endpoints()) {
    bool cancelled =
        co_await wait_for_cancellation(token, runtime.primary(), std::chrono::milliseconds(20));
    if (cancelled) co_return;
  }
}

coro::Task<WorkerId> schedule_impl(std::shared_ptr<KvRouter> self,
                                   std::vector<Token> token_ids) {
  OverlapScores overlap = self->indexer().find_matches_for_request(token_ids);

  KVHitRateEvent event;
  WorkerId worker = self->scheduler().schedule(overlap, token_ids.size(), &event);

  auto component = self->component();
  try {
    co_await component.publish(kKvHitRateSubject, nlohmann::json(event));
  } catch (const std::exception& e) {
    spdlog::warn("kv router: failed to publish hit-rate event: {}", e.what());
  }
  co_return worker;
}

}  // namespace

coro::Task<void> KvRouter::run_event_consumer() { return event_consumer(shared_from_this()); }

coro::Task<void> KvRouter::run_metrics_aggregator(std::chrono::milliseconds interval,
                                                  std::chrono::milliseconds scrape_timeout) {
  return metrics_aggregator(shared_from_this(), interval, scrape_timeout);
}

coro::Task<void> KvRouter::wait_for_endpoints() { return wait_endpoints(shared_from_this()); }

coro::Task<WorkerId> KvRouter::schedule(const std::vector<Token>& token_ids) {
  return schedule_impl(shared_from_this(), token_ids);
}

}  // namespace dynamo::llm::kv
