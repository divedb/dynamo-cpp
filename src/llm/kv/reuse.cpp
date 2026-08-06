// SPDX-License-Identifier: Apache-2.0

#include "llm/kv/reuse.h"

#include <atomic>
#include <deque>
#include <map>
#include <mutex>
#include <unordered_map>

#include <spdlog/spdlog.h>

namespace dynamo::llm::kv {

namespace {

/// Eviction order: lowest priority first, then oldest return first. The
/// sequence hash is a tiebreaker only (in Rust it sits in the key but is
/// excluded from Ord; return ticks are unique so it never decides).
struct PriorityKey {
  uint32_t priority = 0;
  uint64_t return_tick = 0;
  SequenceHash sequence_hash = 0;

  bool operator<(const PriorityKey& other) const {
    if (priority != other.priority) return priority < other.priority;
    if (return_tick != other.return_tick) return return_tick < other.return_tick;
    return sequence_hash < other.sequence_hash;
  }
};

PriorityKey key_of(const KvBlock& block) {
  return {block.priority, block.return_tick, block.token_block.sequence_hash};
}

}  // namespace

struct AvailableBlocksState {
  std::mutex mutex;

  // One parked block per sequence hash; duplicates and hash-0 blocks go to
  // the uninitialized set (state not worth matching).
  std::unordered_map<SequenceHash, KvBlock> lookup_map;
  std::map<PriorityKey, SequenceHash> priority_set;
  std::deque<KvBlock> uninitialized_set;

  uint64_t return_tick = 0;
  std::atomic<uint64_t> total_blocks{0};
  std::atomic<uint64_t> available_blocks{0};

  void insert_locked(KvBlock block) {
    SequenceHash hash = block.token_block.sequence_hash;
    if (hash == 0 || lookup_map.count(hash) > 0) {
      uninitialized_set.push_back(std::move(block));
      return;
    }
    priority_set.emplace(key_of(block), hash);
    lookup_map.emplace(hash, std::move(block));
  }

  std::optional<KvBlock> take_with_hash_locked(SequenceHash hash) {
    auto it = lookup_map.find(hash);
    if (it == lookup_map.end()) return std::nullopt;
    KvBlock block = std::move(it->second);
    lookup_map.erase(it);
    priority_set.erase(key_of(block));
    return block;
  }

  std::optional<KvBlock> take_any_locked() {
    // Uninitialized first: they carry no reusable state, so spending them
    // preserves matchable blocks.
    if (!uninitialized_set.empty()) {
      KvBlock block = std::move(uninitialized_set.front());
      uninitialized_set.pop_front();
      return block;
    }
    if (!priority_set.empty()) {
      auto first = priority_set.begin();
      SequenceHash hash = first->second;
      priority_set.erase(first);
      auto it = lookup_map.find(hash);
      if (it == lookup_map.end()) {
        throw std::logic_error("kv reuse pool: priority entry missing from lookup map");
      }
      KvBlock block = std::move(it->second);
      lookup_map.erase(it);
      return block;
    }
    return std::nullopt;
  }

  void return_block(KvBlock block) {
    std::lock_guard lock(mutex);
    available_blocks.fetch_add(1, std::memory_order_seq_cst);
    block.return_tick = ++return_tick;
    insert_locked(std::move(block));
  }
};

UniqueBlock::UniqueBlock(KvBlock block, std::weak_ptr<AvailableBlocksState> pool)
    : block_(std::move(block)), pool_(std::move(pool)) {}

UniqueBlock& UniqueBlock::operator=(UniqueBlock&& other) noexcept {
  if (this != &other) {
    release();
    block_ = std::move(other.block_);
    pool_ = std::move(other.pool_);
    other.block_.reset();
  }
  return *this;
}

void UniqueBlock::release() {
  if (!block_) return;
  if (auto pool = pool_.lock()) pool->return_block(std::move(*block_));
  block_.reset();
}

KvBlock UniqueBlock::take() && {
  KvBlock block = std::move(*block_);
  block_.reset();
  return block;
}

AvailableBlocks::AvailableBlocks() : state_(std::make_shared<AvailableBlocksState>()) {}

uint64_t AvailableBlocks::total_blocks() const {
  return state_->total_blocks.load(std::memory_order_seq_cst);
}

uint64_t AvailableBlocks::available_blocks() const {
  return state_->available_blocks.load(std::memory_order_seq_cst);
}

std::vector<UniqueBlock> AvailableBlocks::match_blocks(const std::vector<SequenceHash>& hashes) {
  std::vector<UniqueBlock> matched;
  std::lock_guard lock(state_->mutex);
  for (SequenceHash hash : hashes) {
    auto block = state_->take_with_hash_locked(hash);
    if (!block) break;
    matched.emplace_back(UniqueBlock(std::move(*block), state_));
  }
  state_->available_blocks.fetch_sub(matched.size(), std::memory_order_seq_cst);
  return matched;
}

std::vector<UniqueBlock> AvailableBlocks::match_token_blocks(
    const std::vector<TokenBlock>& token_blocks) {
  std::vector<SequenceHash> hashes;
  hashes.reserve(token_blocks.size());
  for (const auto& block : token_blocks) hashes.push_back(block.sequence_hash);
  return match_blocks(hashes);
}

std::vector<UniqueBlock> AvailableBlocks::take_blocks(uint32_t count) {
  std::vector<UniqueBlock> taken;
  std::lock_guard lock(state_->mutex);
  for (uint32_t i = 0; i < count; ++i) {
    auto block = state_->take_any_locked();
    if (!block) break;
    taken.emplace_back(UniqueBlock(std::move(*block), state_));
  }
  state_->available_blocks.fetch_sub(taken.size(), std::memory_order_seq_cst);
  return taken;
}

void AvailableBlocks::insert(KvBlock block) {
  std::lock_guard lock(state_->mutex);
  state_->available_blocks.fetch_add(1, std::memory_order_seq_cst);
  state_->total_blocks.fetch_add(1, std::memory_order_seq_cst);
  block.return_tick = ++state_->return_tick;
  state_->insert_locked(std::move(block));
}

void AvailableBlocks::update_priority(SequenceHash hash, uint32_t priority) {
  update_priorities({{hash, priority}});
}

void AvailableBlocks::update_priorities(
    const std::vector<std::pair<SequenceHash, uint32_t>>& updates) {
  std::lock_guard lock(state_->mutex);
  for (const auto& [hash, priority] : updates) {
    if (auto block = state_->take_with_hash_locked(hash)) {
      block->priority = priority;
      state_->insert_locked(std::move(*block));
    }
  }
}

void AvailableBlocks::reset(const std::vector<SequenceHash>& hashes) {
  std::lock_guard lock(state_->mutex);
  for (SequenceHash hash : hashes) {
    if (auto block = state_->take_with_hash_locked(hash)) {
      block->reset();
      state_->insert_locked(std::move(*block));
    }
  }
}

void AvailableBlocks::reset_all() {
  std::lock_guard lock(state_->mutex);
  while (!state_->priority_set.empty()) {
    auto first = state_->priority_set.begin();
    SequenceHash hash = first->second;
    state_->priority_set.erase(first);
    auto it = state_->lookup_map.find(hash);
    if (it == state_->lookup_map.end()) {
      throw std::logic_error("kv reuse pool: priority entry missing from lookup map");
    }
    KvBlock block = std::move(it->second);
    state_->lookup_map.erase(it);
    block.reset();
    state_->insert_locked(std::move(block));
  }
}

}  // namespace dynamo::llm::kv
