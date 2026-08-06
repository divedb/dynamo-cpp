// SPDX-License-Identifier: Apache-2.0
//
// KV block available pool — Dynamo's lib/llm kv/reuse.rs.
//
// Blocks that are not actively in use park here WITH their previous state
// (token block + hashes), so a later request whose prefix matches can reuse
// them without recomputation. Eviction for fresh capacity is priority-based
// FIFO: lowest priority first, oldest return first within a priority.
//
// Deviation from Rust v0.1.0 (same call made for the kv_router radix tree):
// Rust runs the pool as a dedicated tokio task fed by mpsc channels; we guard
// one state struct with a mutex and make the API synchronous. Returns happen
// inline in the UniqueBlock destructor, so there is no fence() — there is
// nothing asynchronous to wait for.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "llm/tokens.h"

namespace dynamo::llm::kv {

/// A KV cache block: the token state it holds plus reuse metadata. At Rust
/// v0.1.0 the block carries no storage reference (that field is commented
/// out upstream); pairing pool blocks with tensor storage indices is the
/// engine integration's job.
struct KvBlock {
  TokenBlock token_block;
  uint32_t priority = 0;
  uint64_t return_tick = 0;

  KvBlock() = default;
  explicit KvBlock(TokenBlock block) : token_block(std::move(block)) {}

  void reset() {
    token_block = TokenBlock{};
    priority = 0;
    return_tick = 0;
  }
};

class AvailableBlocks;
struct AvailableBlocksState;  // defined in reuse.cpp

/// Exclusive handle to a block taken from the pool; the destructor returns
/// the block (with its state) to the pool it came from.
class UniqueBlock {
 public:
  UniqueBlock() = default;
  UniqueBlock(UniqueBlock&&) noexcept = default;
  UniqueBlock& operator=(UniqueBlock&& other) noexcept;
  UniqueBlock(const UniqueBlock&) = delete;
  UniqueBlock& operator=(const UniqueBlock&) = delete;
  ~UniqueBlock() { release(); }

  bool valid() const { return block_.has_value(); }
  KvBlock& operator*() { return *block_; }
  const KvBlock& operator*() const { return *block_; }
  KvBlock* operator->() { return &*block_; }
  const KvBlock* operator->() const { return &*block_; }

  void update_token_block(TokenBlock token_block) { block_->token_block = std::move(token_block); }

  /// Detaches the block from the pool (it will not be returned).
  KvBlock take() &&;

 private:
  friend class AvailableBlocks;
  UniqueBlock(KvBlock block, std::weak_ptr<AvailableBlocksState> pool);
  void release();

  std::optional<KvBlock> block_;
  std::weak_ptr<AvailableBlocksState> pool_;
};

/// Priority-reuse pool of idle KV blocks. Copyable handle over shared state;
/// blocks returned by dropped UniqueBlock handles come back even if the
/// original AvailableBlocks handle is gone.
class AvailableBlocks {
 public:
  AvailableBlocks();

  uint64_t total_blocks() const;
  uint64_t available_blocks() const;

  /// Matches hashes in order against parked blocks; stops at the first miss
  /// and returns the matched prefix (possibly empty).
  std::vector<UniqueBlock> match_blocks(const std::vector<SequenceHash>& hashes);
  std::vector<UniqueBlock> match_token_blocks(const std::vector<TokenBlock>& token_blocks);

  /// Takes up to `count` blocks for reuse: uninitialized blocks first, then
  /// lowest-priority / longest-parked blocks (their previous state is lost
  /// to the taker, which overwrites it).
  std::vector<UniqueBlock> take_blocks(uint32_t count);

  /// Adds a brand-new block to the pool (grows capacity).
  void insert(KvBlock block);

  /// Re-prioritizes parked blocks by sequence hash (missing hashes ignored).
  void update_priority(SequenceHash hash, uint32_t priority);
  void update_priorities(const std::vector<std::pair<SequenceHash, uint32_t>>& updates);

  /// Wipes the state of parked blocks by hash / of every parked block.
  void reset(const std::vector<SequenceHash>& hashes);
  void reset_all();

 private:
  std::shared_ptr<AvailableBlocksState> state_;
};

}  // namespace dynamo::llm::kv
