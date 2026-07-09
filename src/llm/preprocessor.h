// SPDX-License-Identifier: Apache-2.0
//
// OpenAIPreprocessor — Dynamo's lib/llm preprocessor.rs: a pipeline Operator
// that turns OpenAI chat/completions requests into tokenized BackendInput on
// the way down (template render + tokenize + option extraction) and maps
// BackendOutput deltas back into OpenAI stream responses on the way up,
// prepending any requested annotations (formatted_prompt, token_ids).

#pragma once

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "llm/preprocessor/prompt.h"
#include "llm/protocols/llm_backend.h"
#include "llm/protocols/openai.h"
#include "llm/tokenizers.h"
#include "pipeline/annotated.h"
#include "pipeline/operators.h"

namespace dynamo::llm {

inline constexpr const char* kAnnotationFormattedPrompt = "formatted_prompt";
inline constexpr const char* kAnnotationTokenIds = "token_ids";

using ChatPreprocessOperator =
    pipeline::Operator<openai::NvCreateChatCompletionRequest,
                       pipeline::Annotated<openai::NvCreateChatCompletionStreamResponse>,
                       BackendInput, pipeline::Annotated<BackendOutput>>;
using CompletionPreprocessOperator =
    pipeline::Operator<openai::NvCreateCompletionRequest,
                       pipeline::Annotated<openai::NvCreateCompletionResponse>, BackendInput,
                       pipeline::Annotated<BackendOutput>>;

struct ModelDeploymentCard;

/// Builds the tokenizer named by an MDC (byte_level supported in-tree;
/// hf_tokenizer_json requires a backend that is not vendored yet — throws).
TokenizerPtr tokenizer_from_mdc(const ModelDeploymentCard& mdc);

/// Builds the prompt formatter from an MDC's tokenizer_config.json artifact
/// (chat_template required, as in Rust). Throws std::runtime_error.
std::shared_ptr<const PromptFormatter> formatter_from_mdc(const ModelDeploymentCard& mdc);

class OpenAIPreprocessor final : public ChatPreprocessOperator,
                                 public CompletionPreprocessOperator {
 public:
  /// `eos_token_ids` comes from the model info (the MDC wires it);
  /// `mdcsum` is the Model Deployment Card checksum stamped on requests.
  OpenAIPreprocessor(std::shared_ptr<const PromptFormatter> formatter, TokenizerPtr tokenizer,
                     std::vector<TokenIdType> eos_token_ids, std::string mdcsum);

  /// Rust's OpenAIPreprocessor::new(mdc): formatter + tokenizer + model info
  /// all resolved from the card's artifacts.
  static std::shared_ptr<OpenAIPreprocessor> from_mdc(const ModelDeploymentCard& mdc);

  /// Renders + tokenizes a request into BackendInput; also returns the
  /// annotation map the caller asked for via nvext.annotations.
  std::pair<BackendInput, std::map<std::string, std::string>> preprocess(
      const openai::NvCreateChatCompletionRequest& request) const;
  std::pair<BackendInput, std::map<std::string, std::string>> preprocess(
      const openai::NvCreateCompletionRequest& request) const;

  coro::Task<pipeline::ManyOut<pipeline::Annotated<openai::NvCreateChatCompletionStreamResponse>>>
  generate(pipeline::SingleIn<openai::NvCreateChatCompletionRequest> request,
           pipeline::EnginePtr<BackendInput, pipeline::Annotated<BackendOutput>> next) override;

  coro::Task<pipeline::ManyOut<pipeline::Annotated<openai::NvCreateCompletionResponse>>> generate(
      pipeline::SingleIn<openai::NvCreateCompletionRequest> request,
      pipeline::EnginePtr<BackendInput, pipeline::Annotated<BackendOutput>> next) override;

  const std::string& mdcsum() const { return mdcsum_; }

 private:
  template <typename Request>
  std::pair<BackendInput, std::map<std::string, std::string>> preprocess_impl(
      const Request& request) const;

  std::shared_ptr<const PromptFormatter> formatter_;
  TokenizerPtr tokenizer_;
  std::vector<TokenIdType> eos_token_ids_;
  std::string mdcsum_;
};

/// Maps a BackendOutput stream into a response stream via a delta generator
/// (ChatDeltaGenerator or CompletionDeltaGenerator), yielding `prologue`
/// first. A generator error becomes an error annotation, stops the request
/// context, and closes the stream (Rust transform_postprocessor_stream).
template <typename Resp, typename Generator>
coro::AsyncGenerator<pipeline::Annotated<Resp>> transform_postprocessor_stream(
    pipeline::ManyOut<pipeline::Annotated<BackendOutput>> in, Generator generator,
    std::vector<pipeline::Annotated<Resp>> prologue, pipeline::ContextPtr context) {
  for (auto& annotation : prologue) {
    co_yield annotation;
  }
  while (auto item = co_await in.next()) {
    pipeline::Annotated<Resp> out;
    out.id = item->id;
    out.event = item->event;
    out.comment = item->comment;
    std::optional<std::string> error;
    if (item->data.has_value()) {
      try {
        out.data = generator.choice_from_postprocessor(*item->data);
      } catch (const std::exception& e) {
        error = e.what();
      }
    }
    if (error.has_value()) {
      context->stop_generating();
      auto error_item = pipeline::Annotated<Resp>::from_error(std::move(*error));
      co_yield error_item;
      break;
    }
    co_yield out;
  }
}

}  // namespace dynamo::llm
