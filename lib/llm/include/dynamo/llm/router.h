#pragma once

#include <dynamo/llm/protocols.h>

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <folly/futures/Future.h>

namespace dynamo::llm {

// ---------------------------------------------------------------------------
// KvRouter — KV-cache-aware request routing
// ---------------------------------------------------------------------------

struct WorkerMetadata {
    std::string worker_id;
    std::string addr;
    int port = 0;
    double load_factor = 0.0;
    int kv_cache_usage_pct = 0;
};

class KvIndexer {
public:
    void add_sequence(int32_t worker_id, const std::vector<int32_t>& tokens);
    void remove_sequence(int32_t worker_id, const std::vector<int32_t>& tokens);

    // Returns overlap scores for each worker
    std::vector<std::pair<int32_t, double>> compute_overlap(
        const std::vector<int32_t>& prefix) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class KvRouter {
public:
    explicit KvRouter(std::shared_ptr<KvIndexer> indexer);

    // Schedule a request to the best worker
    // Returns worker_id
    folly::Future<int32_t> schedule(
        const std::vector<int32_t>& token_ids,
        const std::vector<WorkerMetadata>& workers);

private:
    std::shared_ptr<KvIndexer> indexer_;
};

// ---------------------------------------------------------------------------
// DisaggregatedRouter — decide prefill vs decode placement
// ---------------------------------------------------------------------------

struct DisaggRouterConfig {
    int max_local_prefill_length = 4096;
    int max_prefill_queue_depth = 8;
    bool enable_remote_prefill = true;
};

class DisaggregatedRouter {
public:
    explicit DisaggregatedRouter(DisaggRouterConfig config = {});

    // Returns true if prefill should be offloaded to a remote worker
    bool prefill_remote(int prefill_length,
                        int prefix_hit_length,
                        int prefill_queue_depth) const;

    const DisaggRouterConfig& config() const noexcept { return config_; }
    void reconfigure(DisaggRouterConfig cfg) { config_ = cfg; }

private:
    DisaggRouterConfig config_;
};

} // namespace dynamo::llm
