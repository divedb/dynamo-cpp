// SPDX-License-Identifier: Apache-2.0
//
// Tokenizer interface and incremental detokenization — Dynamo's lib/llm
// tokenizers.rs: Encoding, the Encoder/Decoder/Tokenizer traits, Sequence
// (incremental decode with UTF-8 partials held back), and StopSequenceDecoder
// (stop token/sequence enforcement with jailed text).
//
// Backends: ByteLevelTokenizer is a small self-contained reference backend
// (token id = UTF-8 byte, lossy decode) used by tests and the echo engines.
// A HuggingFace tokenizer.json backend slots in behind the same interface
// when a real engine integration lands (see TODO.md M2).

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "llm/protocols/common.h"

namespace dynamo::llm {

/// The result of encoding a string.
struct Encoding {
  std::vector<TokenIdType> token_ids;
  std::vector<std::string> tokens;
  std::vector<std::pair<size_t, size_t>> spans;
};

/// Encode/decode interface implemented by tokenizer backends. Implementations
/// must be thread-safe; methods throw std::runtime_error on failure.
class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  virtual Encoding encode(const std::string& input) const = 0;

  virtual std::string decode(const std::vector<TokenIdType>& token_ids,
                             bool skip_special_tokens) const = 0;
};

using TokenizerPtr = std::shared_ptr<const Tokenizer>;

/// Reference backend: token id = UTF-8 byte value; ids 256/257 are the
/// <s>/</s> special tokens. Decode is lossy (invalid or incomplete UTF-8
/// becomes U+FFFD), which gives real partial-character semantics to the
/// incremental decoders below.
class ByteLevelTokenizer final : public Tokenizer {
 public:
  static constexpr TokenIdType kBosId = 256;
  static constexpr TokenIdType kEosId = 257;

  Encoding encode(const std::string& input) const override;
  std::string decode(const std::vector<TokenIdType>& token_ids,
                     bool skip_special_tokens) const override;
};

/// Incrementally decodes an append-only token sequence, releasing only text
/// that ends on a complete character (vLLM-style prefix/read offsets).
class Sequence {
 public:
  explicit Sequence(TokenizerPtr tokenizer) : tokenizer_(std::move(tokenizer)) {}

  bool empty() const { return token_ids_.empty(); }
  size_t size() const { return token_ids_.size(); }
  const std::vector<TokenIdType>& token_ids() const { return token_ids_; }

  void clear() {
    token_ids_.clear();
    prefix_offset_ = 0;
    read_offset_ = 0;
  }

  /// Tokenizes and appends text (no decode side effects).
  void append_text(const std::string& input);

  /// Appends one token and returns newly-decodable text; returns "" while the
  /// tail decodes to an incomplete character (U+FFFD).
  std::string append_token_id(TokenIdType token_id);

  /// Full decoded text of the sequence.
  std::string text() const { return tokenizer_->decode(token_ids_, false); }

  const TokenizerPtr& tokenizer() const { return tokenizer_; }

 private:
  TokenizerPtr tokenizer_;
  std::vector<TokenIdType> token_ids_;
  size_t prefix_offset_ = 0;
  size_t read_offset_ = 0;
};

/// One step of StopSequenceDecoder output.
struct SequenceDecoderOutput {
  enum class Kind {
    text,               ///< emit `text`
    held,               ///< text jailed pending a possible hidden stop match
    stopped,            ///< stop hit; nothing to emit
    stopped_with_text,  ///< stop hit; emit `text` first
  };
  Kind kind = Kind::text;
  std::string text;
};

struct StopSequenceConfig {
  std::vector<TokenIdType> stop_token_ids_visible;
  std::vector<TokenIdType> stop_token_ids_hidden;
  std::vector<std::string> stop_sequences_visible;  // parity: unused in Rust v0.1.0 too
  std::vector<std::string> stop_sequences_hidden;
};

/// Incremental detokenizer that enforces stop tokens/sequences; text that
/// prefixes a hidden stop sequence is held (jailed) until disambiguated.
class StopSequenceDecoder {
 public:
  StopSequenceDecoder(TokenizerPtr tokenizer, StopSequenceConfig config)
      : sequence_(std::move(tokenizer)), config_(std::move(config)) {}

  /// Throws std::runtime_error if called after a stop.
  SequenceDecoderOutput append_token_id(TokenIdType token_id);

  bool empty() const { return sequence_.empty(); }
  size_t size() const { return sequence_.size(); }
  bool is_complete() const { return stopped_; }
  void close() { stopped_ = true; }

 private:
  Sequence sequence_;
  StopSequenceConfig config_;
  bool stopped_ = false;
  std::string state_;  // jailed text
};

}  // namespace dynamo::llm
