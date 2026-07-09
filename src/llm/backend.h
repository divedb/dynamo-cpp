// SPDX-License-Identifier: Apache-2.0
//
// Backend operator — Dynamo's lib/llm backend.rs: the stage between the
// preprocessor and the LLM engine (ExecutionContext). On the way up it
// detokenizes engine deltas (incremental decode, UTF-8 partials held) and
// enforces stop conditions: hidden stop tokens cut the stream, hidden stop
// sequences cut it retroactively (matched text is never emitted), and a stop
// detected here (when the engine did not finish on its own) issues
// stop_generating() to free engine resources.

#pragma once

#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "llm/protocols/llm_backend.h"
#include "llm/tokenizers.h"
#include "pipeline/annotated.h"
#include "pipeline/operators.h"

namespace dynamo::llm {

/// The output stream shape produced by LLM engines.
using ExecutionOutputStream = pipeline::Annotated<LLMEngineOutput>;

/// The engine seam (Rust's ExecutionContext): consumes tokenized requests,
/// produces raw engine deltas.
using ExecutionContext = pipeline::EnginePtr<BackendInput, ExecutionOutputStream>;

/// Why the decoder stopped a stream.
struct StopTrigger {
  enum class Kind { max_tokens_limit, hidden_stop_token, hidden_stop_sequence };
  Kind kind = Kind::max_tokens_limit;
  TokenIdType token_id = 0;  // for hidden_stop_token
  std::string sequence;      // for hidden_stop_sequence

  FinishReason finish_reason() const {
    return kind == Kind::max_tokens_limit ? FinishReason::length() : FinishReason::stop();
  }
};

struct StepResult {
  /// Text released by this step, already stop-filtered (jailed stop-sequence
  /// text is withheld and never appears here).
  std::optional<std::string> token;
  std::optional<StopTrigger> stop_trigger;
};

struct SeqResult {
  std::vector<std::optional<std::string>> tokens;
  std::optional<std::string> text;
  std::optional<StopTrigger> stop_trigger;
};

/// Incremental detokenizer + stop-condition enforcement for one request
/// (Rust's backend::Decoder).
///
/// Deviation from Rust v0.1.0: text that could be the start of a hidden stop
/// sequence is jailed (withheld) until disambiguated, so stop-sequence text
/// never leaks to the client. Rust's backend decoder emits it and only cuts
/// once the full sequence has been streamed (its own TODO notes the missing
/// jailing; tokenizers.rs' StopSequenceDecoder already behaves this way).
class Decoder {
 public:
  Decoder(TokenizerPtr tokenizer, const StopConditions& stop_conditions);

  /// Decodes one token and evaluates stop conditions (not applied until
  /// min_tokens have been generated).
  StepResult step(TokenIdType token_id);

  /// Runs step() over a delta's token ids, accumulating text; stops early on
  /// a trigger.
  SeqResult process_token_ids(const std::vector<TokenIdType>& token_ids);

  /// Releases any jailed text (call when generation finishes without a stop
  /// trigger — the held text turned out not to be a stop sequence).
  std::string flush();

 private:
  Sequence sequence_;
  uint32_t min_tokens_ = 0;
  std::unordered_set<TokenIdType> hidden_stop_ids_;
  std::vector<std::string> hidden_stop_sequences_;
  uint32_t generated_tokens_ = 0;
  std::string pending_;  // jailed text: a prefix of some stop sequence
};

using BackendOperator =
    pipeline::Operator<BackendInput, pipeline::Annotated<BackendOutput>, BackendInput,
                       pipeline::Annotated<LLMEngineOutput>>;

/// The Backend pipeline operator. Engine deltas that already carry decoded
/// text pass through untouched (unless validate_engine_decode is set); token-
/// only deltas are detokenized and stop-checked here.
struct ModelDeploymentCard;

class Backend final : public BackendOperator {
 public:
  Backend(TokenizerPtr tokenizer, std::string mdcsum, bool validate_engine_decode = false)
      : tokenizer_(std::move(tokenizer)),
        mdcsum_(std::move(mdcsum)),
        validate_engine_decode_(validate_engine_decode) {}

  /// Rust's Backend::from_mdc: tokenizer + checksum from the card.
  static std::shared_ptr<Backend> from_mdc(const ModelDeploymentCard& mdc);

  coro::Task<pipeline::ManyOut<pipeline::Annotated<BackendOutput>>> generate(
      pipeline::SingleIn<BackendInput> request, ExecutionContext next) override;

 private:
  TokenizerPtr tokenizer_;
  std::string mdcsum_;
  bool validate_engine_decode_ = false;
};

}  // namespace dynamo::llm
