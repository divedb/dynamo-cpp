// SPDX-License-Identifier: Apache-2.0
//
// Inflight KV block registry — Dynamo's lib/llm kv/reserved.rs.
//
// Blocks actively used by running requests are shared by sequence hash: two
// requests over the same prefix hold the same reserved block. The registry
// keeps weak references; when the last holder drops, the block leaves the
// registry and its UniqueBlock returns to the available pool.

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "llm/kv/reuse.h"

namespace dynamo::llm::kv {

class ReservedBlocks;
struct ReservedRegistry;    // defined in reserved.cpp
struct ReservedBlockInner;  // defined in reserved.cpp

/// Shared handle to an inflight block.
class ReservedBlock {
 public:
  ReservedBlock() = default;

  bool valid() const { return inner_ != nullptr; }
  const KvBlock& block() const;

  /// How many requests currently hold this block.
  size_t inflight_count() const { return static_cast<size_t>(inner_.use_count()); }

 private:
  friend class ReservedBlocks;
  explicit ReservedBlock(std::shared_ptr<ReservedBlockInner> inner) : inner_(std::move(inner)) {}
  std::shared_ptr<ReservedBlockInner> inner_;
};

class ReservedBlocks {
 public:
  explicit ReservedBlocks(size_t block_size);

  /// Matches hashes / token blocks in order against inflight blocks; stops at
  /// the first miss and returns the matched prefix.
  std::vector<ReservedBlock> match_sequence_hashes(const std::vector<SequenceHash>& hashes);
  std::vector<ReservedBlock> match_token_blocks(const std::vector<TokenBlock>& token_blocks);

  /// Registers a full block as inflight. If a block with the same sequence
  /// hash is already registered, the existing one is returned and the passed
  /// block goes back to its pool (two requests raced to build the same
  /// block). Throws if the block is not exactly block_size tokens.
  ReservedBlock register_block(UniqueBlock block);

 private:
  size_t block_size_;
  std::shared_ptr<ReservedRegistry> registry_;
};

}  // namespace dynamo::llm::kv
