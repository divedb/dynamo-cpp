// SPDX-License-Identifier: Apache-2.0
//
// KV radix indexer — Dynamo's kv_router/indexer.rs: a prefix tree over
// block hashes with per-worker attribution, fed by the worker KV event
// stream. find_matches() scores how many leading blocks of a request each
// worker already has cached.
//
// Deviation from Rust: the Rust KvIndexer owns the tree on a dedicated
// thread and talks over channels (the tree is !Send there); here the tree
// is guarded by a mutex behind the same interface.

#pragma once

#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "llm/kv_router/protocols.h"

namespace dynamo::llm::kv {

/// Per-worker match scores: worker -> number of leading blocks it has.
struct OverlapScores {
  std::unordered_map<WorkerId, uint32_t> scores;
  /// Non-zero recent-use counts of traversed blocks (frequency tracking on).
  std::vector<size_t> frequencies;

  uint32_t score(WorkerId worker) const {
    auto it = scores.find(worker);
    return it == scores.end() ? 0 : it->second;
  }
};

class RadixTree {
 public:
  RadixTree() = default;
  explicit RadixTree(std::optional<std::chrono::steady_clock::duration> expiration_duration)
      : expiration_duration_(expiration_duration) {}

  /// Walks the tree along `sequence`, scoring every worker present on each
  /// matched block; `early_exit` stops once a block is unique to one worker.
  OverlapScores find_matches(const std::vector<LocalBlockHash>& sequence, bool early_exit);

  /// Applies a stored/removed event from a worker.
  void apply_event(const RouterEvent& event);

  /// Drops all blocks attributed to a worker (lease gone).
  void remove_worker(WorkerId worker);

 private:
  struct Block {
    std::unordered_map<LocalBlockHash, std::shared_ptr<Block>> children;
    std::unordered_set<WorkerId> workers;
    std::deque<std::chrono::steady_clock::time_point> recent_uses;
  };
  using BlockPtr = std::shared_ptr<Block>;

  BlockPtr root_ = std::make_shared<Block>();
  /// Per-worker jump table from engine block hash into the tree.
  std::unordered_map<WorkerId, std::unordered_map<ExternalSequenceBlockHash, BlockPtr>> lookup_;
  std::optional<std::chrono::steady_clock::duration> expiration_duration_;
};

/// Thread-safe indexer over a RadixTree, plus the request-side hash helper.
class KvIndexer {
 public:
  explicit KvIndexer(size_t kv_block_size) : kv_block_size_(kv_block_size) {}

  size_t block_size() const { return kv_block_size_; }

  void apply_event(const RouterEvent& event) {
    std::lock_guard lock(mutex_);
    tree_.apply_event(event);
  }

  void remove_worker(WorkerId worker) {
    std::lock_guard lock(mutex_);
    tree_.remove_worker(worker);
  }

  OverlapScores find_matches(const std::vector<LocalBlockHash>& sequence) {
    std::lock_guard lock(mutex_);
    return tree_.find_matches(sequence, /*early_exit=*/false);
  }

  OverlapScores find_matches_for_request(const std::vector<Token>& tokens) {
    return find_matches(compute_block_hashes_for_seq(tokens, kv_block_size_));
  }

 private:
  std::mutex mutex_;
  RadixTree tree_;
  size_t kv_block_size_;
};

}  // namespace dynamo::llm::kv
