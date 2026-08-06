// SPDX-License-Identifier: Apache-2.0

#include "llm/kv/reserved.h"

#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace dynamo::llm::kv {

struct ReservedRegistry {
  std::mutex mutex;
  std::unordered_map<SequenceHash, std::weak_ptr<ReservedBlockInner>> map;
};

struct ReservedBlockInner {
  UniqueBlock block;  // returns to the available pool when Inner dies
  SequenceHash sequence_hash = 0;
  std::shared_ptr<ReservedRegistry> registry;

  ~ReservedBlockInner() {
    // Deviation from Rust (cleaner, same effect): Rust removes the entry and
    // re-inserts it if it was a different, still-live registration; we only
    // erase when the entry is the expired one.
    std::lock_guard lock(registry->mutex);
    auto it = registry->map.find(sequence_hash);
    if (it != registry->map.end() && it->second.expired()) registry->map.erase(it);
  }
};

const KvBlock& ReservedBlock::block() const { return *inner_->block; }

ReservedBlocks::ReservedBlocks(size_t block_size)
    : block_size_(block_size), registry_(std::make_shared<ReservedRegistry>()) {}

std::vector<ReservedBlock> ReservedBlocks::match_sequence_hashes(
    const std::vector<SequenceHash>& hashes) {
  std::vector<ReservedBlock> matched;
  // Reserve before locking: emplace_back must not throw under the registry
  // mutex (unwinding would destroy Inners, whose destructors take it again).
  matched.reserve(hashes.size());
  std::lock_guard lock(registry_->mutex);
  for (SequenceHash hash : hashes) {
    auto it = registry_->map.find(hash);
    if (it == registry_->map.end()) break;
    auto inner = it->second.lock();
    if (!inner) break;
    matched.emplace_back(ReservedBlock(std::move(inner)));
  }
  return matched;
}

std::vector<ReservedBlock> ReservedBlocks::match_token_blocks(
    const std::vector<TokenBlock>& token_blocks) {
  std::vector<SequenceHash> hashes;
  hashes.reserve(token_blocks.size());
  for (const auto& block : token_blocks) hashes.push_back(block.sequence_hash);
  return match_sequence_hashes(hashes);
}

ReservedBlock ReservedBlocks::register_block(UniqueBlock block) {
  if (!block.valid()) throw std::invalid_argument("cannot register an empty block");
  if (block->token_block.tokens.size() != block_size_) {
    throw std::runtime_error("block size mismatch");
  }
  SequenceHash hash = block->token_block.sequence_hash;

  std::lock_guard lock(registry_->mutex);
  if (auto it = registry_->map.find(hash); it != registry_->map.end()) {
    if (auto existing = it->second.lock()) {
      // Another request finished building the same block first; ours returns
      // to the pool as `block` goes out of scope.
      return ReservedBlock(std::move(existing));
    }
  }

  auto inner = std::make_shared<ReservedBlockInner>();
  inner->block = std::move(block);
  inner->sequence_hash = hash;
  inner->registry = registry_;
  registry_->map[hash] = inner;
  return ReservedBlock(std::move(inner));
}

}  // namespace dynamo::llm::kv
