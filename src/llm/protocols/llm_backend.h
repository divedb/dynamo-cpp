// SPDX-License-Identifier: Apache-2.0
//
// Internal LLM request/response representation — Dynamo's
// protocols::common::{preprocessor, llm_backend}:
//   PreprocessedRequest (= BackendInput): tokenized request handed to engines
//   LLMEngineOutput: minimal raw engine output (pre-detokenization)
//   BackendOutput: post-processed output emitted by the Backend operator

#pragma once

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "llm/protocols/common.h"

namespace dynamo::llm {

/// A detokenized token; engines may not produce text for every id.
using TokenType = std::optional<std::string>;
using LogProbs = std::vector<double>;

/// The internal representation of an LLM request: the preprocessor converts
/// public-API requests into this before they reach an engine.
struct PreprocessedRequest {
  std::vector<TokenIdType> token_ids;
  StopConditions stop_conditions;
  SamplingOptions sampling_options;
  /// EOS token id(s) for the model, for backends that need them.
  std::vector<TokenIdType> eos_token_ids;
  /// Checksum of the Model Deployment Card this request was prepared against.
  std::optional<std::string> mdc_sum;
  /// User-requested annotations (out-of-band events in the response stream).
  std::vector<std::string> annotations;

  bool has_annotation(const std::string& annotation) const {
    for (const auto& a : annotations) {
      if (a == annotation) return true;
    }
    return false;
  }
};

using BackendInput = PreprocessedRequest;

inline void to_json(nlohmann::json& j, const PreprocessedRequest& r) {
  j = nlohmann::json{{"token_ids", r.token_ids},
                     {"stop_conditions", r.stop_conditions},
                     {"sampling_options", r.sampling_options},
                     {"eos_token_ids", r.eos_token_ids},
                     {"annotations", r.annotations}};
  set_opt(j, "mdc_sum", r.mdc_sum);
}

inline void from_json(const nlohmann::json& j, PreprocessedRequest& r) {
  get_or(j, "token_ids", r.token_ids, {});
  get_or(j, "stop_conditions", r.stop_conditions, {});
  get_or(j, "sampling_options", r.sampling_options, {});
  get_or(j, "eos_token_ids", r.eos_token_ids, {});
  get_or(j, "annotations", r.annotations, {});
  get_opt(j, "mdc_sum", r.mdc_sum);
}

namespace detail {

inline nlohmann::json tokens_to_json(const std::vector<TokenType>& tokens) {
  nlohmann::json arr = nlohmann::json::array();
  for (const auto& t : tokens) {
    if (t.has_value()) {
      arr.push_back(*t);
    } else {
      arr.push_back(nullptr);
    }
  }
  return arr;
}

inline std::vector<TokenType> tokens_from_json(const nlohmann::json& arr) {
  std::vector<TokenType> tokens;
  tokens.reserve(arr.size());
  for (const auto& t : arr) {
    if (t.is_null()) {
      tokens.emplace_back(std::nullopt);
    } else {
      tokens.emplace_back(t.get<std::string>());
    }
  }
  return tokens;
}

}  // namespace detail

/// Minimal raw output of an LLM engine. If `tokens` is absent the Backend
/// operator is responsible for detokenization.
struct LLMEngineOutput {
  std::vector<TokenIdType> token_ids;
  std::optional<std::vector<TokenType>> tokens;
  std::optional<std::string> text;
  std::optional<double> cum_log_probs;
  std::optional<LogProbs> log_probs;
  std::optional<FinishReason> finish_reason;

  static LLMEngineOutput cancelled() { return with_reason(FinishReason::cancelled()); }
  static LLMEngineOutput stop() { return with_reason(FinishReason::stop()); }
  static LLMEngineOutput length() { return with_reason(FinishReason::length()); }
  static LLMEngineOutput error(std::string message) {
    return with_reason(FinishReason::error(std::move(message)));
  }

 private:
  static LLMEngineOutput with_reason(FinishReason reason) {
    LLMEngineOutput out;
    out.finish_reason = std::move(reason);
    return out;
  }
};

inline void to_json(nlohmann::json& j, const LLMEngineOutput& o) {
  j = nlohmann::json{{"token_ids", o.token_ids}};
  if (o.tokens) j["tokens"] = detail::tokens_to_json(*o.tokens);
  set_opt(j, "text", o.text);
  set_opt(j, "cum_log_probs", o.cum_log_probs);
  set_opt(j, "log_probs", o.log_probs);
  set_opt(j, "finish_reason", o.finish_reason);
}

inline void from_json(const nlohmann::json& j, LLMEngineOutput& o) {
  get_or(j, "token_ids", o.token_ids, {});
  o.tokens.reset();
  if (auto it = j.find("tokens"); it != j.end() && !it->is_null()) {
    o.tokens = detail::tokens_from_json(*it);
  }
  get_opt(j, "text", o.text);
  get_opt(j, "cum_log_probs", o.cum_log_probs);
  get_opt(j, "log_probs", o.log_probs);
  get_opt(j, "finish_reason", o.finish_reason);
}

/// Post-processed engine output emitted by the Backend operator; `tokens`
/// always has the same length as `token_ids`.
struct BackendOutput {
  std::vector<TokenIdType> token_ids;
  std::vector<TokenType> tokens;
  std::optional<std::string> text;
  std::optional<double> cum_log_probs;
  std::optional<LogProbs> log_probs;
  std::optional<FinishReason> finish_reason;
  /// Model Deployment Card checksum.
  std::string mdcsum;
};

inline void to_json(nlohmann::json& j, const BackendOutput& o) {
  j = nlohmann::json{{"token_ids", o.token_ids},
                     {"tokens", detail::tokens_to_json(o.tokens)},
                     {"mdcsum", o.mdcsum}};
  set_opt(j, "text", o.text);
  set_opt(j, "cum_log_probs", o.cum_log_probs);
  set_opt(j, "log_probs", o.log_probs);
  set_opt(j, "finish_reason", o.finish_reason);
}

inline void from_json(const nlohmann::json& j, BackendOutput& o) {
  get_or(j, "token_ids", o.token_ids, {});
  o.tokens.clear();
  if (auto it = j.find("tokens"); it != j.end() && !it->is_null()) {
    o.tokens = detail::tokens_from_json(*it);
  }
  get_or(j, "mdcsum", o.mdcsum, {});
  get_opt(j, "text", o.text);
  get_opt(j, "cum_log_probs", o.cum_log_probs);
  get_opt(j, "log_probs", o.log_probs);
  get_opt(j, "finish_reason", o.finish_reason);
}

}  // namespace dynamo::llm
