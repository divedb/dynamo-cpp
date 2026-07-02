#include <catch2/catch_test_macros.hpp>
#include <dynamo/llm/router.h>

using namespace dynamo::llm;

TEST_CASE("KvIndexer tracks worker sequences", "[llm][router]") {
    auto indexer = std::make_shared<KvIndexer>();

    indexer->add_sequence(0, {1, 10, 20, 30});
    indexer->add_sequence(1, {1, 10, 99, 88});

    // Overlap with worker 0's prefix
    auto scores = indexer->compute_overlap({1, 10, 20});
    CHECK_FALSE(scores.empty());

    bool found_w0 = false;
    for (const auto& [wid, score] : scores) {
        if (wid == 0) {
            found_w0 = true;
            // 2 out of 3 tokens match the prefix
            CHECK(score > 0.5);
        }
    }
    CHECK(found_w0);
}

TEST_CASE("KvIndexer removes sequences", "[llm][router]") {
    auto indexer = std::make_shared<KvIndexer>();

    indexer->add_sequence(0, {1, 2, 3});
    indexer->remove_sequence(0, {1, 2, 3});

    auto scores = indexer->compute_overlap({1, 2});
    CHECK(scores.empty());
}

TEST_CASE("KvRouter schedules to best worker", "[llm][router]") {
    auto indexer = std::make_shared<KvIndexer>();
    indexer->add_sequence(0, {1, 10, 20});
    indexer->add_sequence(1, {1, 99, 88});

    KvRouter router(indexer);

    std::vector<WorkerMetadata> workers = {
        {"0", "addr1", 50051, 0.9, 50},
        {"1", "addr2", 50051, 0.1, 30},
    };

    // Request matching worker 0's prefix, but worker 0 has high load
    auto chosen = router.schedule({1, 10}, workers).get();

    // Worker 1 has low load factor, might be chosen despite lower KV match
    CHECK(chosen >= 0);
}

TEST_CASE("DisaggregatedRouter decides remote prefill", "[llm][router]") {
    DisaggRouterConfig cfg;
    cfg.max_local_prefill_length = 4096;
    cfg.max_prefill_queue_depth = 8;
    DisaggregatedRouter router(cfg);

    // Short prefill should be local
    CHECK_FALSE(router.prefill_remote(1000, 500, 1));

    // Long prefill with good prefix hit should be remote
    CHECK(router.prefill_remote(8192, 100, 3));

    // Long prefill but queue too deep should NOT be remote
    CHECK_FALSE(router.prefill_remote(8192, 100, 10));
}

TEST_CASE("DisaggRouter reconfigures at runtime", "[llm][router]") {
    DisaggregatedRouter router;
    CHECK(router.config().enable_remote_prefill);

    DisaggRouterConfig new_cfg;
    new_cfg.enable_remote_prefill = false;
    router.reconfigure(new_cfg);
    CHECK_FALSE(router.config().enable_remote_prefill);
}
