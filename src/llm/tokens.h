// SPDX-License-Identifier: Apache-2.0
//
// Token sequences and fixed-size block hashing — Dynamo's lib/llm tokens.rs.
// Block hashes feed the KV-router radix indexer; the hash function and its
// seed are ported exactly (XXH3-64, seed 1337, over the little-endian bytes
// of the u32 token array) so hashes interoperate with recorded Rust fixtures.
//
// Sequence hashes chain blocks: seq[i] = hash(bytes of u64[2]{seq[i-1], bh[i]}).
// Quirk preserved from Rust: a first block created by split_tokens() gets
// seq = block_hash, while one grown via push_token() from an empty sequence
// gets seq = hash({0, block_hash}).
// Deviation from Rust v0.1.0: their partial block never advances its parent
// hash after completing a block (every pushed block chains to the same stale
// parent); here the parent advances so pushed blocks chain like split blocks.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace dynamo::llm {

using Token = uint32_t;
using BlockHash = uint64_t;
using SequenceHash = uint64_t;

inline constexpr uint64_t kXxh3Seed = 1337;

/// XXH3-64 with the Dynamo seed; the KV-event/block hash primitive.
uint64_t compute_hash(const void* data, size_t len);

inline BlockHash compute_block_hash(const std::vector<Token>& tokens) {
  return compute_hash(tokens.data(), tokens.size() * sizeof(Token));
}

/// A completed fixed-size block of tokens with its hashes.
struct TokenBlock {
  std::vector<Token> tokens;
  BlockHash block_hash = 0;
  SequenceHash sequence_hash = 0;
  std::optional<SequenceHash> parent_sequence_hash;
};

/// The trailing (not yet full) block of a sequence.
class PartialTokenBlock {
 public:
  PartialTokenBlock(std::vector<Token> tokens, size_t block_size,
                    std::optional<SequenceHash> parent_sequence_hash)
      : tokens_(std::move(tokens)),
        block_size_(block_size),
        parent_sequence_hash_(parent_sequence_hash) {}

  /// Appends a token; when the block fills up, returns the completed
  /// TokenBlock and resets to empty (parented on the new block).
  std::optional<TokenBlock> push_token(Token token);

  const std::vector<Token>& tokens() const { return tokens_; }
  std::optional<SequenceHash> parent_sequence_hash() const { return parent_sequence_hash_; }

 private:
  std::vector<Token> tokens_;
  size_t block_size_;
  std::optional<SequenceHash> parent_sequence_hash_;
};

/// A token sequence maintained as completed blocks plus a partial tail.
class TokenSequence {
 public:
  TokenSequence(std::vector<Token> tokens, size_t block_size);

  /// Appends a token; returns the newly completed block, if any.
  const TokenBlock* push_token(Token token);

  const std::vector<TokenBlock>& blocks() const { return blocks_; }
  const PartialTokenBlock& current_block() const { return current_block_; }

  std::pair<std::vector<TokenBlock>, PartialTokenBlock> into_parts() && {
    return {std::move(blocks_), std::move(current_block_)};
  }

  /// Splits tokens into completed blocks + partial tail, computing hashes.
  static std::pair<std::vector<TokenBlock>, PartialTokenBlock> split_tokens(
      std::vector<Token> tokens, size_t block_size);

 private:
  std::vector<TokenBlock> blocks_;
  PartialTokenBlock current_block_;
};

}  // namespace dynamo::llm
