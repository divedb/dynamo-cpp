// SPDX-License-Identifier: Apache-2.0

#include "llm/tokens.h"

#include <bit>

#include <xxhash.h>

static_assert(std::endian::native == std::endian::little,
              "block hashing assumes little-endian token/hash byte layout");

namespace dynamo::llm {

uint64_t compute_hash(const void* data, size_t len) {
  return XXH3_64bits_withSeed(data, len, kXxh3Seed);
}

namespace {

SequenceHash chain_hash(SequenceHash parent, BlockHash block_hash) {
  uint64_t pair[2] = {parent, block_hash};
  return compute_hash(pair, sizeof(pair));
}

}  // namespace

std::optional<TokenBlock> PartialTokenBlock::push_token(Token token) {
  tokens_.push_back(token);
  if (tokens_.size() != block_size_) return std::nullopt;

  TokenBlock block;
  block.tokens = std::move(tokens_);
  tokens_ = {};
  block.block_hash = compute_block_hash(block.tokens);
  block.sequence_hash = chain_hash(parent_sequence_hash_.value_or(0), block.block_hash);
  block.parent_sequence_hash = parent_sequence_hash_;
  parent_sequence_hash_ = block.sequence_hash;
  return block;
}

TokenSequence::TokenSequence(std::vector<Token> tokens, size_t block_size)
    : current_block_({}, block_size, std::nullopt) {
  auto [blocks, current] = split_tokens(std::move(tokens), block_size);
  blocks_ = std::move(blocks);
  current_block_ = std::move(current);
}

const TokenBlock* TokenSequence::push_token(Token token) {
  if (auto block = current_block_.push_token(token)) {
    blocks_.push_back(std::move(*block));
    return &blocks_.back();
  }
  return nullptr;
}

std::pair<std::vector<TokenBlock>, PartialTokenBlock> TokenSequence::split_tokens(
    std::vector<Token> tokens, size_t block_size) {
  std::vector<TokenBlock> blocks;
  size_t full = block_size == 0 ? 0 : tokens.size() / block_size;
  blocks.reserve(full);

  for (size_t i = 0; i < full; ++i) {
    TokenBlock block;
    block.tokens.assign(tokens.begin() + static_cast<ptrdiff_t>(i * block_size),
                        tokens.begin() + static_cast<ptrdiff_t>((i + 1) * block_size));
    block.block_hash = compute_block_hash(block.tokens);
    if (i == 0) {
      // Rust quirk: the first split block's sequence hash is its block hash.
      block.sequence_hash = block.block_hash;
    } else {
      block.parent_sequence_hash = blocks.back().sequence_hash;
      block.sequence_hash = chain_hash(*block.parent_sequence_hash, block.block_hash);
    }
    blocks.push_back(std::move(block));
  }

  std::vector<Token> remainder(tokens.begin() + static_cast<ptrdiff_t>(full * block_size),
                               tokens.end());
  std::optional<SequenceHash> parent;
  if (!blocks.empty()) parent = blocks.back().sequence_hash;

  return {std::move(blocks), PartialTokenBlock(std::move(remainder), block_size, parent)};
}

}  // namespace dynamo::llm
