// SPDX-License-Identifier: Apache-2.0

#include "llm/model_card.h"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <xxhash.h>

#include "llm/protocols/json.h"

namespace dynamo::llm {

namespace fs = std::filesystem;

namespace {

std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("failed to read file: " + path);
  std::ostringstream out;
  out << in.rdbuf();
  return out.str();
}

int64_t now_unix_seconds() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

/// RFC3339 UTC (seconds precision), the readable form of last_published.
std::string to_rfc3339(int64_t unix_seconds) {
  std::time_t t = static_cast<std::time_t>(unix_seconds);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buf;
}

int64_t from_rfc3339(const std::string& text) {
  std::tm tm{};
  if (strptime(text.c_str(), "%Y-%m-%dT%H:%M:%S", &tm) == nullptr) {
    throw std::runtime_error("invalid RFC3339 timestamp: " + text);
  }
  return static_cast<int64_t>(timegm(&tm));
}

std::string find_file(const std::string& dir, const char* name, bool required) {
  fs::path candidate = fs::path(dir) / name;
  if (fs::exists(candidate)) return candidate.string();
  if (required) {
    throw std::runtime_error(fmt::format("file {} not found in {}", name, dir));
  }
  return {};
}

/// tokenizer_config.json token fields are either a plain string or an
/// AddedToken object {"content": "...", ...}.
std::string token_string(const nlohmann::json& j, const char* key) {
  auto it = j.find(key);
  if (it == j.end() || it->is_null()) return {};
  if (it->is_string()) return it->get<std::string>();
  if (it->is_object() && it->contains("content")) return (*it)["content"].get<std::string>();
  return {};
}

}  // namespace

// ---------------------------------------------------------------------------
// FileRef

void to_json(nlohmann::json& j, const FileRef& f) {
  j = nlohmann::json{{f.kind, f.path}};
}

void from_json(const nlohmann::json& j, FileRef& f) {
  if (!j.is_object() || j.size() != 1) {
    throw std::runtime_error("expected externally-tagged artifact object, got: " + j.dump());
  }
  auto it = j.begin();
  f.kind = it.key();
  f.path = it.value().is_null() ? std::string{} : it.value().get<std::string>();
}

// ---------------------------------------------------------------------------
// ModelInfo

ModelInfo ModelInfo::from_hf_config_json(const std::string& path) {
  nlohmann::json config = nlohmann::json::parse(read_file(path));
  ModelInfo info;
  info.model_type = config.value("model_type", "");
  info.bos_token_id = config.value("bos_token_id", TokenIdType{0});
  if (auto it = config.find("eos_token_id"); it != config.end() && !it->is_null()) {
    if (it->is_array()) {
      info.eos_token_ids = it->get<std::vector<TokenIdType>>();
    } else {
      info.eos_token_ids = {it->get<TokenIdType>()};
    }
  }
  info.max_position_embeddings = config.value("max_position_embeddings", size_t{0});
  info.vocab_size = config.value("vocab_size", size_t{0});
  return info;
}

// ---------------------------------------------------------------------------
// HfTokenizerConfig

HfTokenizerConfig HfTokenizerConfig::from_file(const std::string& path) {
  nlohmann::json config = nlohmann::json::parse(read_file(path));
  HfTokenizerConfig out;
  if (auto it = config.find("chat_template"); it != config.end() && !it->is_null()) {
    if (it->is_string()) {
      out.chat_template = it->get<std::string>();
    } else if (it->is_array()) {
      // Named-template list [{"name": ..., "template": ...}]; keep "default"
      // and "tool_use". (Rust intends the same but registers each map's raw
      // k/v pairs as templates — i.e. templates named "name"/"template" — so
      // its named-list path never resolves; we follow the HF convention it
      // aimed for.)
      for (const auto& entry : *it) {
        const std::string name = entry.value("name", "");
        if (name == "default") {
          out.chat_template = entry.value("template", "");
        } else if (name == "tool_use") {
          out.tool_use_chat_template = entry.value("template", "");
        }
      }
    }
  }
  out.bos_token = token_string(config, "bos_token");
  out.eos_token = token_string(config, "eos_token");
  return out;
}

// ---------------------------------------------------------------------------
// slug / checksum

std::string slugify(const std::string& input) {
  std::string out;
  out.reserve(input.size() + 9);
  for (char c : input) {
    char lower = (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
    bool valid = (lower >= 'a' && lower <= 'z') || (lower >= '0' && lower <= '9') || lower == '_';
    out.push_back(valid ? lower : '_');
  }
  // Rust appends the last 8 hex chars of a content hash for uniqueness
  // (blake3 there, XXH3 here) and trims leading replacement chars.
  uint64_t hash = XXH3_64bits(input.data(), input.size());
  out += fmt::format("_{:08x}", static_cast<uint32_t>(hash));
  size_t start = out.find_first_not_of('_');
  return start == std::string::npos ? out : out.substr(start);
}

// ---------------------------------------------------------------------------
// ModelDeploymentCard

ModelDeploymentCard ModelDeploymentCard::from_local_path(
    const std::string& local_root_dir, std::optional<std::string> model_name) {
  fs::path dir(local_root_dir);
  if (!fs::exists(dir)) {
    throw std::runtime_error("Model path does not exist: " + local_root_dir);
  }
  if (!fs::is_directory(dir)) {
    throw std::runtime_error("Model path is not a directory: " + local_root_dir);
  }

  std::string name = model_name.value_or(fs::canonical(dir).filename().string());

  ModelDeploymentCard card;
  card.display_name = name;
  card.service_name = name;
  card.model_info = {kModelInfoHfConfigJson, find_file(local_root_dir, "config.json", true)};
  card.tokenizer = {kTokenizerHfJson, find_file(local_root_dir, "tokenizer.json", true)};
  if (auto path = find_file(local_root_dir, "tokenizer_config.json", false); !path.empty()) {
    card.prompt_formatter = FileRef{kPromptFormatterHfTokenizerConfigJson, path};
  }
  card.requires_preprocessing = true;
  return card;
}

ModelDeploymentCard ModelDeploymentCard::load_from_json_str(const std::string& json) {
  return nlohmann::json::parse(json).get<ModelDeploymentCard>();
}

ModelDeploymentCard ModelDeploymentCard::load_from_json_file(const std::string& path) {
  auto card = load_from_json_str(read_file(path));
  card.requires_preprocessing = false;  // parity with Rust load_from_json_file
  return card;
}

void ModelDeploymentCard::save_to_json_file(const std::string& path) const {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("failed to write file: " + path);
  out << to_json_string();
}

std::string ModelDeploymentCard::to_json_string() const {
  return nlohmann::json(*this).dump();
}

std::string ModelDeploymentCard::slug() const { return slugify(service_name); }

std::string ModelDeploymentCard::kv_key() const {
  return std::string(kMdcBucketName) + "/" + slug();
}

std::string ModelDeploymentCard::mdcsum() const {
  std::string json = to_json_string();
  XXH128_hash_t hash = XXH3_128bits(json.data(), json.size());
  return fmt::format("{:016x}{:016x}", hash.high64, hash.low64);
}

bool ModelDeploymentCard::is_expired() const {
  if (!last_published.has_value()) return false;
  return now_unix_seconds() - *last_published > kMdcMaxAge.count();
}

ModelInfo ModelDeploymentCard::load_model_info() const {
  if (model_info.kind != kModelInfoHfConfigJson) {
    throw std::runtime_error("unsupported model_info kind: " + model_info.kind);
  }
  return ModelInfo::from_hf_config_json(model_info.path);
}

void to_json(nlohmann::json& j, const ModelDeploymentCard& c) {
  j = nlohmann::json{{"display_name", c.display_name},
                     {"service_name", c.service_name},
                     {"model_info", c.model_info},
                     {"tokenizer", c.tokenizer},
                     {"requires_preprocessing", c.requires_preprocessing}};
  set_opt(j, "prompt_formatter", c.prompt_formatter);
  set_opt(j, "prompt_context", c.prompt_context);
  // revision is intentionally not serialized (parity with Rust).
  j["last_published"] = c.last_published.has_value()
                            ? nlohmann::json(to_rfc3339(*c.last_published))
                            : nlohmann::json(nullptr);
}

void from_json(const nlohmann::json& j, ModelDeploymentCard& c) {
  get_or(j, "display_name", c.display_name, {});
  get_or(j, "service_name", c.service_name, {});
  get_or(j, "model_info", c.model_info, {});
  get_or(j, "tokenizer", c.tokenizer, {});
  get_opt(j, "prompt_formatter", c.prompt_formatter);
  get_opt(j, "prompt_context", c.prompt_context);
  get_or(j, "requires_preprocessing", c.requires_preprocessing, false);
  get_or(j, "revision", c.revision, uint64_t{0});
  c.last_published.reset();
  if (auto it = j.find("last_published"); it != j.end() && !it->is_null()) {
    c.last_published = from_rfc3339(it->get<std::string>());
  }
}

// ---------------------------------------------------------------------------
// Publish/discover

coro::Task<void> publish_model_card(discovery::Discovery& discovery, ModelDeploymentCard& card,
                                    std::optional<int64_t> lease_id) {
  card.last_published = now_unix_seconds();
  ++card.revision;
  co_await discovery.kv_put(card.kv_key(), card.to_json_string(), lease_id);
}

coro::Task<std::vector<ModelDeploymentCard>> list_model_cards(discovery::Discovery& discovery) {
  auto kvs = co_await discovery.kv_get_prefix(std::string(kMdcBucketName) + "/");
  std::vector<ModelDeploymentCard> cards;
  cards.reserve(kvs.size());
  for (const auto& kv : kvs) {
    try {
      cards.push_back(ModelDeploymentCard::load_from_json_str(kv.value));
    } catch (const std::exception& e) {
      spdlog::warn("skipping malformed model card at {}: {}", kv.key, e.what());
    }
  }
  co_return cards;
}

}  // namespace dynamo::llm
