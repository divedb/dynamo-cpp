#include <dynamo/llm/router.h>

#include <map>
#include <set>
#include <algorithm>
#include <spdlog/spdlog.h>

namespace dynamo::llm {

// ---------------------------------------------------------------------------
// KvIndexer
// ---------------------------------------------------------------------------

struct KvIndexer::Impl {
    // Radix tree node storing worker IDs
    struct Node {
        std::map<int32_t, std::unique_ptr<Node>> children;
        std::set<int32_t> workers;  // workers that have this prefix
    };

    std::unique_ptr<Node> root = std::make_unique<Node>();

    void insert(const std::vector<int32_t>& tokens, int32_t worker_id) {
        Node* node = root.get();
        for (auto t : tokens) {
            if (!node->children[t]) {
                node->children[t] = std::make_unique<Node>();
            }
            node = node->children[t].get();
            node->workers.insert(worker_id);
        }
    }

    void remove(const std::vector<int32_t>& tokens, int32_t worker_id) {
        Node* node = root.get();
        for (auto t : tokens) {
            auto it = node->children.find(t);
            if (it == node->children.end()) return;
            node = it->second.get();
            node->workers.erase(worker_id);
        }
    }

    std::vector<std::pair<int32_t, double>> compute_overlap(
        const std::vector<int32_t>& prefix) const {
        // Walk the tree following the prefix, collect workers and depths
        std::map<int32_t, int> worker_depth;
        Node* node = root.get();
        int depth = 0;

        for (auto t : prefix) {
            auto it = node->children.find(t);
            if (it == node->children.end()) break;
            node = it->second.get();
            depth++;

            for (auto w : node->workers) {
                worker_depth[w] = std::max(worker_depth[w], depth);
            }
        }

        // Convert to scores (0.0 - 1.0)
        std::vector<std::pair<int32_t, double>> scores;
        int max_depth = std::max(1, static_cast<int>(prefix.size()));
        for (const auto& [w, d] : worker_depth) {
            scores.emplace_back(w, static_cast<double>(d) / max_depth);
        }
        return scores;
    }
};

KvIndexer::KvIndexer() : impl_(std::make_unique<Impl>()) {}
KvIndexer::~KvIndexer() = default;

void KvIndexer::add_sequence(
    int32_t worker_id, const std::vector<int32_t>& tokens) {
    impl_->insert(tokens, worker_id);
}

void KvIndexer::remove_sequence(
    int32_t worker_id, const std::vector<int32_t>& tokens) {
    impl_->remove(tokens, worker_id);
}

std::vector<std::pair<int32_t, double>> KvIndexer::compute_overlap(
    const std::vector<int32_t>& prefix) const {
    return impl_->compute_overlap(prefix);
}

// ---------------------------------------------------------------------------
// KvRouter
// ---------------------------------------------------------------------------

KvRouter::KvRouter(std::shared_ptr<KvIndexer> indexer)
    : indexer_(std::move(indexer)) {}

folly::Future<int32_t> KvRouter::schedule(
    const std::vector<int32_t>& token_ids,
    const std::vector<WorkerMetadata>& workers) {
    auto scores = indexer_->compute_overlap(token_ids);

    // Combine overlap scores with load factors
    int32_t best_worker = -1;
    double best_score = -1e9;

    for (const auto& wm : workers) {
        int32_t wid = std::stoi(wm.worker_id);
        double kv_score = 0.0;
        for (const auto& [sid, score] : scores) {
            if (sid == wid) {
                kv_score = score;
                break;
            }
        }
        // Cost function: KV match ratio - load factor
        double total_score = kv_score - wm.load_factor;
        if (total_score > best_score) {
            best_score = total_score;
            best_worker = wid;
        }
    }

    return folly::makeFuture(best_worker >= 0 ? best_worker : 0);
}

// ---------------------------------------------------------------------------
// DisaggregatedRouter
// ---------------------------------------------------------------------------

DisaggregatedRouter::DisaggregatedRouter(DisaggRouterConfig config)
    : config_(std::move(config)) {}

bool DisaggregatedRouter::prefill_remote(
    int prefill_length,
    int prefix_hit_length,
    int prefill_queue_depth) const {
    if (!config_.enable_remote_prefill) return false;

    int unique_prefill = prefill_length - prefix_hit_length;
    bool too_long = unique_prefill > config_.max_local_prefill_length;
    bool queue_full = prefill_queue_depth >= config_.max_prefill_queue_depth;

    return too_long && !queue_full;
}

} // namespace dynamo::llm
