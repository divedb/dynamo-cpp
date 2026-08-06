// SPDX-License-Identifier: Apache-2.0

#include "llm/kv/manager.h"

#include <stdexcept>

#include <spdlog/spdlog.h>

namespace dynamo::llm::kv {

KvStorageManager::KvStorageManager(size_t block_size)
    : inflight_blocks_(block_size), block_size_(block_size) {}

PrefillMatched KvStorageManager::prepare_prefill_sequence(std::vector<Token> tokens) {
  spdlog::debug("kv manager: adding request with {} tokens", tokens.size());
  auto [blocks, tail_block] = TokenSequence::split_tokens(std::move(tokens), block_size_);
  spdlog::debug("kv manager: request translates to {} blocks; remaining tokens: {}", blocks.size(),
                tail_block.tokens().size());

  // First match against inflight blocks (shared with running requests)...
  auto inflight = inflight_blocks_.match_token_blocks(blocks);
  spdlog::debug("kv manager: matched {} inflight blocks", inflight.size());

  // ...then the unmatched suffix against parked reusable blocks, which are
  // promoted to inflight registrations.
  std::vector<TokenBlock> unmatched(blocks.begin() + static_cast<ptrdiff_t>(inflight.size()),
                                    blocks.end());
  auto reusable = available_blocks_.match_token_blocks(unmatched);
  spdlog::debug("kv manager: matched {} freed blocks", reusable.size());
  for (auto& block : reusable) {
    inflight.push_back(inflight_blocks_.register_block(std::move(block)));
  }

  std::vector<TokenBlock> remaining(
      std::make_move_iterator(blocks.begin() + static_cast<ptrdiff_t>(inflight.size())),
      std::make_move_iterator(blocks.end()));
  return PrefillMatched{std::move(inflight), std::move(remaining), std::move(tail_block)};
}

PrefillOffload KvStorageManager::prepare_prefill_offload(PrefillMatched matched) {
  auto blocks_to_reuse =
      available_blocks_.take_blocks(static_cast<uint32_t>(matched.remaining_blocks.size()) + 1);
  if (blocks_to_reuse.size() != matched.remaining_blocks.size() + 1) {
    throw std::runtime_error("kv manager: expected " +
                             std::to_string(matched.remaining_blocks.size() + 1) + " blocks, got " +
                             std::to_string(blocks_to_reuse.size()));
  }

  // Assign each remaining token block to a fresh pool block (matching Rust,
  // blocks pair back-to-front; pool blocks are interchangeable here).
  std::vector<UniqueBlock> complete_prefill_blocks;
  complete_prefill_blocks.reserve(matched.remaining_blocks.size());
  for (auto& token_block : matched.remaining_blocks) {
    UniqueBlock block = std::move(blocks_to_reuse.back());
    blocks_to_reuse.pop_back();
    block.update_token_block(std::move(token_block));
    complete_prefill_blocks.push_back(std::move(block));
  }

  PrefillOffload offload{std::move(matched.inflight_blocks), std::move(complete_prefill_blocks),
                         PartialKvBlock{std::move(matched.tail_block),
                                        std::move(blocks_to_reuse.back())}};
  blocks_to_reuse.pop_back();
  return offload;
}

}  // namespace dynamo::llm::kv
