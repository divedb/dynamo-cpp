// SPDX-License-Identifier: Apache-2.0

#include "llm/kv_router/indexer.h"

#include <spdlog/spdlog.h>

namespace dynamo::llm::kv {

OverlapScores RadixTree::find_matches(const std::vector<LocalBlockHash>& sequence,
                                      bool early_exit) {
  OverlapScores scores;
  BlockPtr current = root_;
  auto now = std::chrono::steady_clock::now();

  for (LocalBlockHash block_hash : sequence) {
    auto it = current->children.find(block_hash);
    if (it == current->children.end()) break;
    BlockPtr block = it->second;

    for (WorkerId worker : block->workers) {
      ++scores.scores[worker];
    }

    if (expiration_duration_.has_value()) {
      while (!block->recent_uses.empty() &&
             now - block->recent_uses.front() > *expiration_duration_) {
        block->recent_uses.pop_front();
      }
      if (!block->recent_uses.empty()) {
        scores.frequencies.push_back(block->recent_uses.size());
      }
      block->recent_uses.push_back(now);
    }

    if (early_exit && block->workers.size() == 1) break;
    current = std::move(block);
  }
  return scores;
}

void RadixTree::apply_event(const RouterEvent& event) {
  WorkerId worker_id = event.worker_id;
  auto& worker_lookup = lookup_[worker_id];

  if (event.event.data.kind == KvCacheEventData::Kind::stored) {
    const KvCacheStoreData& op = event.event.data.stored;

    BlockPtr current;
    if (op.parent_hash.has_value()) {
      auto it = worker_lookup.find(*op.parent_hash);
      if (it == worker_lookup.end()) {
        spdlog::warn("kv indexer: worker {} event {}: parent block {:x} not found; skipping",
                     worker_id, event.event.event_id, *op.parent_hash);
        return;
      }
      current = it->second;
    } else {
      current = root_;
    }

    for (const KvCacheStoredBlockData& block_data : op.blocks) {
      BlockPtr block;
      if (auto it = current->children.find(block_data.tokens_hash);
          it != current->children.end()) {
        block = it->second;
      } else {
        // New edge; reuse the worker's existing node for this engine hash
        // if there is one (re-parent case), else create a fresh block.
        if (auto lookup_it = worker_lookup.find(block_data.block_hash);
            lookup_it != worker_lookup.end()) {
          block = lookup_it->second;
        } else {
          block = std::make_shared<Block>();
        }
        current->children.emplace(block_data.tokens_hash, block);
      }
      block->workers.insert(worker_id);
      worker_lookup[block_data.block_hash] = block;
      current = std::move(block);
    }
    return;
  }

  // removed
  for (ExternalSequenceBlockHash hash : event.event.data.removed.block_hashes) {
    auto it = worker_lookup.find(hash);
    if (it == worker_lookup.end()) {
      spdlog::warn("kv indexer: worker {} event {}: block {:x} not found for removal; skipping",
                   worker_id, event.event.event_id, hash);
      continue;
    }
    BlockPtr block = it->second;
    block->workers.erase(worker_id);
    if (block->workers.empty()) {
      // No worker holds this block, so none holds any of its children.
      block->children.clear();
    }
    worker_lookup.erase(it);
  }
}

void RadixTree::remove_worker(WorkerId worker) {
  auto it = lookup_.find(worker);
  if (it == lookup_.end()) return;
  for (auto& [hash, block] : it->second) {
    block->workers.erase(worker);
  }
  lookup_.erase(it);
}

}  // namespace dynamo::llm::kv
