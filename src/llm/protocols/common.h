// SPDX-License-Identifier: Apache-2.0
//
// Common LLM engine protocol types — Dynamo's lib/llm protocols::common:
// FinishReason, StopConditions, SamplingOptions, OutputOptions.
//
// Wire format matches the Rust serde encoding: FinishReason unit variants
// serialize as plain strings ("eos", "length", "stop", "cancelled") and the
// error variant as {"error": "<message>"}. Optional fields are omitted when
// absent and tolerated as missing or null when parsing.

#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "llm/protocols/json.h"

namespace dynamo::llm {

using TokenIdType = uint32_t;

// ---------------------------------------------------------------------------
// FinishReason

struct FinishReason {
  enum class Kind { eos, length, stop, error, cancelled };

  Kind kind = Kind::stop;
  std::string error_message;  // meaningful iff kind == error

  static FinishReason eos() { return {Kind::eos, {}}; }
  static FinishReason length() { return {Kind::length, {}}; }
  static FinishReason stop() { return {Kind::stop, {}}; }
  static FinishReason cancelled() { return {Kind::cancelled, {}}; }
  static FinishReason error(std::string message) { return {Kind::error, std::move(message)}; }

  bool is_error() const { return kind == Kind::error; }

  /// Display form; the error variant renders as "error: <message>".
  std::string to_string() const {
    switch (kind) {
      case Kind::eos: return "eos";
      case Kind::length: return "length";
      case Kind::stop: return "stop";
      case Kind::cancelled: return "cancelled";
      case Kind::error: return "error: " + error_message;
    }
    return "stop";
  }

  /// Inverse of to_string(); throws std::invalid_argument on unknown input.
  static FinishReason from_string(const std::string& s) {
    if (s == "eos") return eos();
    if (s == "length") return length();
    if (s == "stop") return stop();
    if (s == "cancelled") return cancelled();
    if (s.rfind("error: ", 0) == 0) return error(s.substr(7));
    throw std::invalid_argument("Invalid FinishReason variant: '" + s + "'");
  }

  friend bool operator==(const FinishReason& a, const FinishReason& b) {
    return a.kind == b.kind && (a.kind != Kind::error || a.error_message == b.error_message);
  }
};

inline void to_json(nlohmann::json& j, const FinishReason& r) {
  if (r.kind == FinishReason::Kind::error) {
    j = nlohmann::json{{"error", r.error_message}};
  } else {
    j = r.to_string();
  }
}

inline void from_json(const nlohmann::json& j, FinishReason& r) {
  if (j.is_string()) {
    r = FinishReason::from_string(j.get<std::string>());
    return;
  }
  if (j.is_object() && j.contains("error")) {
    r = FinishReason::error(j.at("error").get<std::string>());
    return;
  }
  throw std::invalid_argument("Invalid FinishReason encoding: " + j.dump());
}

// ---------------------------------------------------------------------------
// StopConditions

/// Server-side stop conditions evaluated against the generated sequence.
struct StopConditions {
  std::optional<uint32_t> max_tokens;
  /// Strings that stop generation; the output will not contain them.
  std::optional<std::vector<std::string>> stop;
  /// Token ids that stop generation; the output will not contain them.
  std::optional<std::vector<TokenIdType>> stop_token_ids_hidden;
  /// Minimum number of tokens to generate (set to max_tokens to ignore EOS).
  std::optional<uint32_t> min_tokens;
  std::optional<bool> ignore_eos;

  void apply_ignore_eos() {
    if (ignore_eos.value_or(false)) {
      min_tokens = max_tokens;
      stop.reset();
      stop_token_ids_hidden.reset();
    }
  }
};

inline void to_json(nlohmann::json& j, const StopConditions& s) {
  j = nlohmann::json::object();
  set_opt(j, "max_tokens", s.max_tokens);
  set_opt(j, "stop", s.stop);
  set_opt(j, "stop_token_ids_hidden", s.stop_token_ids_hidden);
  set_opt(j, "min_tokens", s.min_tokens);
  set_opt(j, "ignore_eos", s.ignore_eos);
}

inline void from_json(const nlohmann::json& j, StopConditions& s) {
  get_opt(j, "max_tokens", s.max_tokens);
  get_opt(j, "stop", s.stop);
  get_opt(j, "stop_token_ids_hidden", s.stop_token_ids_hidden);
  get_opt(j, "min_tokens", s.min_tokens);
  get_opt(j, "ignore_eos", s.ignore_eos);
}

// ---------------------------------------------------------------------------
// SamplingOptions

struct SamplingOptions {
  std::optional<int32_t> n;
  std::optional<int32_t> best_of;
  std::optional<float> presence_penalty;
  std::optional<float> frequency_penalty;
  std::optional<float> repetition_penalty;
  std::optional<float> temperature;
  std::optional<float> top_p;
  std::optional<int32_t> top_k;
  std::optional<float> min_p;
  std::optional<bool> use_beam_search;
  std::optional<float> length_penalty;
  std::optional<int64_t> seed;

  void force_greedy() {
    presence_penalty.reset();
    frequency_penalty.reset();
    repetition_penalty.reset();
    temperature.reset();
    top_p.reset();
    top_k.reset();
    min_p.reset();
  }
};

inline void to_json(nlohmann::json& j, const SamplingOptions& s) {
  j = nlohmann::json::object();
  set_opt(j, "n", s.n);
  set_opt(j, "best_of", s.best_of);
  set_opt(j, "presence_penalty", s.presence_penalty);
  set_opt(j, "frequency_penalty", s.frequency_penalty);
  set_opt(j, "repetition_penalty", s.repetition_penalty);
  set_opt(j, "temperature", s.temperature);
  set_opt(j, "top_p", s.top_p);
  set_opt(j, "top_k", s.top_k);
  set_opt(j, "min_p", s.min_p);
  set_opt(j, "use_beam_search", s.use_beam_search);
  set_opt(j, "length_penalty", s.length_penalty);
  set_opt(j, "seed", s.seed);
}

inline void from_json(const nlohmann::json& j, SamplingOptions& s) {
  get_opt(j, "n", s.n);
  get_opt(j, "best_of", s.best_of);
  get_opt(j, "presence_penalty", s.presence_penalty);
  get_opt(j, "frequency_penalty", s.frequency_penalty);
  get_opt(j, "repetition_penalty", s.repetition_penalty);
  get_opt(j, "temperature", s.temperature);
  get_opt(j, "top_p", s.top_p);
  get_opt(j, "top_k", s.top_k);
  get_opt(j, "min_p", s.min_p);
  get_opt(j, "use_beam_search", s.use_beam_search);
  get_opt(j, "length_penalty", s.length_penalty);
  get_opt(j, "seed", s.seed);
}

// ---------------------------------------------------------------------------
// OutputOptions

/// Controls what auxiliary information the engine returns.
struct OutputOptions {
  std::optional<uint32_t> logprobs;
  std::optional<uint32_t> prompt_logprobs;
  std::optional<bool> skip_special_tokens;
  std::optional<bool> formatted_prompt;
};

inline void to_json(nlohmann::json& j, const OutputOptions& o) {
  j = nlohmann::json::object();
  set_opt(j, "logprobs", o.logprobs);
  set_opt(j, "prompt_logprobs", o.prompt_logprobs);
  set_opt(j, "skip_special_tokens", o.skip_special_tokens);
  set_opt(j, "formatted_prompt", o.formatted_prompt);
}

inline void from_json(const nlohmann::json& j, OutputOptions& o) {
  get_opt(j, "logprobs", o.logprobs);
  get_opt(j, "prompt_logprobs", o.prompt_logprobs);
  get_opt(j, "skip_special_tokens", o.skip_special_tokens);
  get_opt(j, "formatted_prompt", o.formatted_prompt);
}

}  // namespace dynamo::llm
