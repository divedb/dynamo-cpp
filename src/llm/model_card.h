// SPDX-License-Identifier: Apache-2.0
//
// Model Deployment Card — Dynamo's lib/llm model_card: the model
// configuration record shared by every component that touches a model.
// Built from a local HuggingFace-format checkout (config.json,
// tokenizer.json, optional tokenizer_config.json); published under the
// discovery kv prefix "mdc/" with the worker's lease (hub downloading is
// out of scope — local paths only).
//
// JSON layout matches the Rust serde encoding (externally-tagged artifact
// enums like {"hf_config_json": path}); mdcsum/slug use XXH3 where Rust
// uses blake3 — cross-implementation checksums were never comparable anyway.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "discovery/discovery.h"
#include "llm/protocols/common.h"

namespace dynamo::llm {

/// Discovery kv prefix for published cards.
inline constexpr const char* kMdcBucketName = "mdc";

/// Cards not re-published within this window are considered expired
/// (their worker is likely gone).
inline constexpr std::chrono::seconds kMdcMaxAge{5 * 60};

/// An externally-tagged file artifact ({"<kind>": "<path>"} in JSON).
struct FileRef {
  std::string kind;
  std::string path;
};

// Artifact kinds (Rust enum variants, snake_case tags).
inline constexpr const char* kModelInfoHfConfigJson = "hf_config_json";
inline constexpr const char* kTokenizerHfJson = "hf_tokenizer_json";
inline constexpr const char* kPromptFormatterHfTokenizerConfigJson = "hf_tokenizer_config_json";
/// C++ extension: the in-tree byte-level reference tokenizer (no file).
inline constexpr const char* kTokenizerByteLevel = "byte_level";

void to_json(nlohmann::json& j, const FileRef& f);
void from_json(const nlohmann::json& j, FileRef& f);

/// Parsed HuggingFace config.json (Rust's HFConfigJsonFile / ModelInfo).
struct ModelInfo {
  std::string model_type;
  TokenIdType bos_token_id = 0;
  std::vector<TokenIdType> eos_token_ids;  // config's int-or-array flattened
  size_t max_position_embeddings = 0;
  size_t vocab_size = 0;

  /// Throws std::runtime_error on missing file or malformed config.
  static ModelInfo from_hf_config_json(const std::string& path);
};

/// Parsed subset of tokenizer_config.json needed for prompt formatting
/// (Rust's tokcfg::HfTokenizerConfig).
struct HfTokenizerConfig {
  std::optional<std::string> chat_template;
  /// Distinct "tool_use" entry of a named-template list, when present.
  std::optional<std::string> tool_use_chat_template;
  std::string bos_token;  // string or {"content": ...} in the file
  std::string eos_token;

  static HfTokenizerConfig from_file(const std::string& path);
};

struct ModelDeploymentCard {
  /// Human readable name, e.g. "Meta Llama 3.1 8B Instruct".
  std::string display_name;
  /// Identifier expected in OpenAI-compatible requests.
  std::string service_name;
  FileRef model_info;
  FileRef tokenizer;
  std::optional<FileRef> prompt_formatter;
  std::optional<std::vector<std::string>> prompt_context;
  /// Unix seconds of the last worker publish; unset if never published.
  std::optional<int64_t> last_published;
  /// Publish count; not serialized (parity with Rust's skip_serializing).
  uint64_t revision = 0;
  /// True when the worker consumes BackendInput (pre-tokenized) rather than
  /// OpenAI requests.
  bool requires_preprocessing = false;

  /// Builds a card from a local HF-format model directory: requires
  /// config.json and tokenizer.json; tokenizer_config.json is picked up as
  /// the prompt formatter when present. Throws std::runtime_error.
  static ModelDeploymentCard from_local_path(const std::string& local_root_dir,
                                             std::optional<std::string> model_name = {});

  static ModelDeploymentCard load_from_json_str(const std::string& json);
  static ModelDeploymentCard load_from_json_file(const std::string& path);
  void save_to_json_file(const std::string& path) const;
  std::string to_json_string() const;

  /// URL/subject-safe unique id derived from service_name
  /// (lowercased, invalid chars -> '_', 8-hex content-hash suffix).
  std::string slug() const;

  /// Discovery kv key: "mdc/{slug}".
  std::string kv_key() const;

  /// Content checksum stamped on requests (XXH3-128 hex here; blake3 in Rust).
  std::string mdcsum() const;

  bool is_expired() const;
  static std::chrono::seconds expiry_check_period() { return kMdcMaxAge / 3; }

  /// Loads and parses the model info artifact. Throws std::runtime_error.
  ModelInfo load_model_info() const;
};

void to_json(nlohmann::json& j, const ModelDeploymentCard& c);
void from_json(const nlohmann::json& j, ModelDeploymentCard& c);

/// Slugify helper (Rust's Slug::from_string).
std::string slugify(const std::string& input);

// ---------------------------------------------------------------------------
// Publish/discover over the discovery kv store

/// Publishes (upserts) the card under mdc/{slug}, bound to `lease_id` so it
/// disappears with the worker; stamps last_published and bumps revision.
coro::Task<void> publish_model_card(discovery::Discovery& discovery, ModelDeploymentCard& card,
                                    std::optional<int64_t> lease_id);

/// All currently published cards (malformed entries are skipped with a
/// warning).
coro::Task<std::vector<ModelDeploymentCard>> list_model_cards(discovery::Discovery& discovery);

}  // namespace dynamo::llm
