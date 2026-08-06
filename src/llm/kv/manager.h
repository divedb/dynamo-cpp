// SPDX-License-Identifier: Apache-2.0
//
// KV storage manager — Dynamo's lib/llm kv/manager.rs.
//
// Manages reservation and priority reuse of KV blocks for a single storage
// tier (a GPU, host memory). Prefill preparation is two-phase so a scheduler
// can look at the match result (how many net-new blocks a request needs)
// before committing pool capacity:
//   1. prepare_prefill_sequence: split tokens into blocks, match the prefix
//      against inflight blocks, then against parked reusable blocks (which
//      get promoted to inflight).
//   2. prepare_prefill_offload: take fresh blocks from the pool for the
//      unmatched remainder plus one for the partial tail.

#pragma once

#include <cstddef>
#include <vector>

#include "llm/kv/reserved.h"
#include "llm/kv/reuse.h"

namespace dynamo::llm::kv {

/// A partial (tail) token block paired with the pool block that will hold it.
struct PartialKvBlock {
  PartialTokenBlock token_block;
  UniqueBlock kv_block;
};

/// Result of the match phase: blocks already resident (inflight), completed
/// token blocks still needing prefill, and the partial tail.
struct PrefillMatched {
  std::vector<ReservedBlock> inflight_blocks;
  std::vector<TokenBlock> remaining_blocks;
  PartialTokenBlock tail_block;
};

/// Result of the allocation phase: pool blocks assigned to every remaining
/// token block plus the tail.
struct PrefillOffload {
  std::vector<ReservedBlock> inflight_blocks;
  std::vector<UniqueBlock> complete_prefill_blocks;
  PartialKvBlock tail_prefill_block;
};

class KvStorageManager {
 public:
  explicit KvStorageManager(size_t block_size);

  PrefillMatched prepare_prefill_sequence(std::vector<Token> tokens);

  /// Throws if the pool cannot supply remaining_blocks.size() + 1 blocks.
  PrefillOffload prepare_prefill_offload(PrefillMatched matched);

  /// The reuse pool (seed capacity with insert(), inspect counters).
  AvailableBlocks& available_blocks() { return available_blocks_; }
  size_t block_size() const { return block_size_; }

 private:
  AvailableBlocks available_blocks_;
  ReservedBlocks inflight_blocks_;
  size_t block_size_;
};

}  // namespace dynamo::llm::kv
