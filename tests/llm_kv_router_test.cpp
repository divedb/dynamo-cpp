// SPDX-License-Identifier: Apache-2.0
//
// M6 tests: KV event protocol round-trips, radix indexer semantics
// (store/remove/eviction, prefix scoring), scheduler selection (overlap vs
// load), and the end-to-end KvRouter: two workers publishing KV events +
// metrics; prefix-affinity routing beats blind routing and follows load.

#include <catch2/catch_test_macros.hpp>

#include <thread>

#include "component/component.h"
#include "llm/kv_router/publisher.h"
#include "llm/kv_router/router.h"
#include "runtime/coro/sync_wait.h"

using namespace dynamo::llm;
using namespace dynamo::llm::kv;
using namespace std::chrono_literals;
namespace coro = dynamo::coro;
namespace component = dynamo::component;

namespace {

dynamo::RuntimeConfig test_config() {
  dynamo::RuntimeConfig config;
  config.num_worker_threads = 4;
  config.num_background_threads = 2;
  return config;
}

std::string unique_ns() {
  static std::atomic<int> counter{500};
  return "kvns" + std::to_string(counter.fetch_add(1));
}

/// A stored event chaining `n` blocks starting from `first_hash`, parented
/// on `parent` (0 = root). Engine block hashes are offset by 1000 to keep
/// them distinct from token hashes.
RouterEvent stored_event(WorkerId worker, uint64_t event_id,
                         std::optional<ExternalSequenceBlockHash> parent,
                         std::vector<LocalBlockHash> tokens_hashes) {
  KvCacheStoreData data;
  data.parent_hash = parent;
  for (LocalBlockHash h : tokens_hashes) {
    data.blocks.push_back({h + 1000, h});
  }
  RouterEvent event;
  event.worker_id = worker;
  event.event.event_id = event_id;
  event.event.data = KvCacheEventData::make_stored(std::move(data));
  return event;
}

RouterEvent removed_event(WorkerId worker, uint64_t event_id,
                          std::vector<ExternalSequenceBlockHash> hashes) {
  RouterEvent event;
  event.worker_id = worker;
  event.event.event_id = event_id;
  event.event.data = KvCacheEventData::make_removed({std::move(hashes)});
  return event;
}

ForwardPassMetrics idle_metrics() {
  ForwardPassMetrics m;
  m.request_active_slots = 0;
  m.request_total_slots = 8;
  m.kv_active_blocks = 1;
  m.kv_total_blocks = 100;
  return m;
}

}  // namespace

TEST_CASE("kv protocols: serde-compatible json", "[llm][kv]") {
  // Fixture mirrors the Rust protocols test.
  auto event = stored_event(7, 1, ExternalSequenceBlockHash{1}, {3});
  event.event.data.stored.blocks[0].block_hash = 2;

  nlohmann::json j = event;
  CHECK(j["worker_id"] == 7);
  CHECK(j["event"]["event_id"] == 1);
  CHECK(j["event"]["data"]["stored"]["parent_hash"] == 1);
  CHECK(j["event"]["data"]["stored"]["blocks"][0]["block_hash"] == 2);
  CHECK(j["event"]["data"]["stored"]["blocks"][0]["tokens_hash"] == 3);

  auto back = j.get<RouterEvent>();
  CHECK(back.event.data.kind == KvCacheEventData::Kind::stored);
  CHECK(back.event.data.stored.parent_hash == 1u);

  nlohmann::json removed = removed_event(7, 2, {4, 5});
  auto removed_back = removed.get<RouterEvent>();
  CHECK(removed_back.event.data.kind == KvCacheEventData::Kind::removed);
  CHECK(removed_back.event.data.removed.block_hashes ==
        std::vector<ExternalSequenceBlockHash>{4, 5});
}

TEST_CASE("kv block hashes chunk exactly", "[llm][kv]") {
  std::vector<Token> tokens{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto hashes = compute_block_hashes_for_seq(tokens, 4);
  REQUIRE(hashes.size() == 2);  // trailing partial chunk dropped
  // Same primitive as tokens.h block hashing (golden value from Rust).
  CHECK(hashes[0] == 14643705804678351452ull);
  CHECK(hashes[1] == 16777012769546811212ull);
}

TEST_CASE("radix indexer: store, match, remove, evict", "[llm][kv][indexer]") {
  KvIndexer indexer(4);

  // Worker 1 has blocks [A, B, C]; worker 2 has [A].
  indexer.apply_event(stored_event(1, 1, std::nullopt, {10, 11, 12}));
  indexer.apply_event(stored_event(2, 1, std::nullopt, {10}));

  auto scores = indexer.find_matches({10, 11, 12});
  CHECK(scores.score(1) == 3);
  CHECK(scores.score(2) == 1);

  // Prefix mismatch scores nothing.
  CHECK(indexer.find_matches({99}).scores.empty());

  // Worker 2 extends under A with D (parented on A's engine hash 1010).
  indexer.apply_event(stored_event(2, 2, ExternalSequenceBlockHash{1010}, {13}));
  scores = indexer.find_matches({10, 13});
  CHECK(scores.score(2) == 2);
  CHECK(scores.score(1) == 1);  // shares only A

  // Unknown parent is skipped (warn, no crash).
  indexer.apply_event(stored_event(3, 1, ExternalSequenceBlockHash{4242}, {20}));
  CHECK(indexer.find_matches({20}).scores.empty());

  // Removal: worker 1 drops C.
  indexer.apply_event(removed_event(1, 2, {1012}));
  scores = indexer.find_matches({10, 11, 12});
  CHECK(scores.score(1) == 2);

  // Worker eviction clears all attribution.
  indexer.remove_worker(1);
  scores = indexer.find_matches({10, 11, 12});
  CHECK(scores.score(1) == 0);
  CHECK(scores.score(2) == 1);  // worker 2 still owns A
}

TEST_CASE("kv scheduler: overlap wins, load balances, busy throws",
          "[llm][kv][scheduler]") {
  auto make_endpoints = [] {
    Endpoint a{"load_metrics", "s-a", 1, idle_metrics()};
    Endpoint b{"load_metrics", "s-b", 2, idle_metrics()};
    return std::vector<Endpoint>{a, b};
  };

  // Overlap dominates when load is even.
  {
    KvScheduler scheduler(4);
    scheduler.update_endpoints(ProcessedEndpoints(make_endpoints()));
    OverlapScores overlap;
    overlap.scores[2] = 5;  // worker 2 has 5 blocks cached
    KVHitRateEvent event;
    CHECK(scheduler.schedule(overlap, 40, &event) == 2);
    CHECK(event.worker_id == 2);
    CHECK(event.isl_blocks == 10);
    CHECK(event.overlap_blocks == 5);
  }

  // With no overlap anywhere, the less loaded worker wins.
  {
    KvScheduler scheduler(4);
    auto endpoints = make_endpoints();
    endpoints[0].data.kv_active_blocks = 90;  // worker 1 heavily loaded
    scheduler.update_endpoints(ProcessedEndpoints(std::move(endpoints)));
    CHECK(scheduler.schedule({}, 40) == 2);
  }

  // All at capacity -> AllWorkersBusy; empty -> NoEndpoints.
  {
    KvScheduler scheduler(4);
    auto endpoints = make_endpoints();
    for (auto& e : endpoints) e.data.request_active_slots = e.data.request_total_slots;
    scheduler.update_endpoints(ProcessedEndpoints(std::move(endpoints)));
    CHECK_THROWS_AS(scheduler.schedule({}, 40), AllWorkersBusyError);

    scheduler.update_endpoints(ProcessedEndpoints{});
    CHECK_THROWS_AS(scheduler.schedule({}, 40), NoEndpointsError);
  }
}

TEST_CASE("kv router end to end: prefix affinity through events and scrapes",
          "[llm][kv][router]") {
  auto rt = dynamo::Runtime::create(test_config());
  {
    auto drt = component::DistributedRuntime::create(rt, {});
    auto comp = drt.ns(unique_ns()).component("backend");
    constexpr size_t kBlockSize = 4;

    auto router = std::make_shared<KvRouter>(comp, kBlockSize);

    coro::sync_wait([&]() -> coro::Task<void> {
      co_await rt.primary().schedule();

      // Router background tasks.
      rt.spawn(router->run_event_consumer());
      rt.spawn(router->run_metrics_aggregator(20ms, 300ms));

      // Two workers: metrics endpoints with distinct instance ids (leases).
      auto metrics_a = std::make_shared<KvMetricsPublisher>();
      auto metrics_b = std::make_shared<KvMetricsPublisher>();
      metrics_a->publish(idle_metrics());
      metrics_b->publish(idle_metrics());
      auto lease_a = co_await drt.discovery()->create_lease(30s);
      auto lease_b = co_await drt.discovery()->create_lease(30s);
      rt.spawn(metrics_a->create_endpoint(comp, lease_a));
      rt.spawn(metrics_b->create_endpoint(comp, lease_b));

      // Wait until both workers' metrics have been scraped.
      co_await router->wait_for_endpoints();
      for (int i = 0; i < 200; ++i) {
        auto stats = co_await comp.scrape_stats(300ms);
        size_t metrics_endpoints = 0;
        for (auto& e : stats.endpoints) {
          if (e.ok && e.info.endpoint == kKvMetricsEndpoint) ++metrics_endpoints;
        }
        if (metrics_endpoints >= 2) break;
        co_await wait_for_cancellation(rt.child_token(), rt.primary(), 20ms);
      }

      // Discover the two worker ids from the scrape.
      auto stats = co_await comp.scrape_stats(300ms);
      std::vector<WorkerId> workers;
      for (auto& e : stats.endpoints) {
        if (e.ok && e.info.endpoint == kKvMetricsEndpoint) {
          workers.push_back(e.info.instance_id);
        }
      }
      REQUIRE(workers.size() == 2);
      WorkerId worker_a = workers[0];
      WorkerId worker_b = workers[1];

      // Worker B announces it has the first two blocks of `prompt` cached.
      std::vector<Token> prompt{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
      auto hashes = compute_block_hashes_for_seq(prompt, kBlockSize);
      KvEventPublisher events_b(comp, worker_b, kBlockSize);
      co_await events_b.publish([&] {
        KvCacheStoreData data;
        for (auto h : hashes) data.blocks.push_back({h + 1, h});
        KvCacheEvent event;
        event.event_id = 1;
        event.data = KvCacheEventData::make_stored(std::move(data));
        return event;
      }());

      // Wait for the event to land in the indexer.
      for (int i = 0; i < 200; ++i) {
        if (router->indexer().find_matches(hashes).score(worker_b) == 2) break;
        co_await wait_for_cancellation(rt.child_token(), rt.primary(), 10ms);
      }
      REQUIRE(router->indexer().find_matches(hashes).score(worker_b) == 2);

      // The router now routes the matching prompt to worker B every time.
      for (int i = 0; i < 5; ++i) {
        WorkerId chosen = co_await router->schedule(prompt);
        CHECK(chosen == worker_b);
      }

      // A prompt with no cached prefix still schedules (load-based pick).
      std::vector<Token> cold{100, 101, 102, 103, 104, 105, 106, 107};
      WorkerId cold_choice = co_await router->schedule(cold);
      CHECK((cold_choice == worker_a || cold_choice == worker_b));

      // Worker eviction removes B's affinity.
      router->indexer().remove_worker(worker_b);
      CHECK(router->indexer().find_matches(hashes).score(worker_b) == 0);
    }());

    drt.discovery()->shutdown();
    rt.shutdown();
    REQUIRE(rt.join_tasks(5000ms));
  }
}
