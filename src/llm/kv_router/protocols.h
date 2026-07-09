// SPDX-License-Identifier: Apache-2.0
//
// KV-router wire types — Dynamo's kv_router/protocols.rs: worker KV-cache
// events (stored/removed block hashes) and per-worker forward-pass metrics.
// JSON matches the Rust serde encoding (hashes as bare u64, event data
// externally tagged {"stored": ...} / {"removed": ...}).

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "llm/protocols/json.h"
#include "llm/tokens.h"

namespace dynamo::llm::kv {

/// Identifier of an LLM worker emitting events (= instance/lease id).
using WorkerId = int64_t;

// Subjects/endpoints (Rust kv_router.rs constants).
inline constexpr const char* kKvEventSubject = "kv_events";
inline constexpr const char* kKvHitRateSubject = "kv-hit-rate";
inline constexpr const char* kKvMetricsEndpoint = "load_metrics";

/// Hash of the tokens within one block (compute_block_hash).
using LocalBlockHash = uint64_t;
/// Engine-provided sequence-aware block hash (opaque to the router).
using ExternalSequenceBlockHash = uint64_t;

/// Block hashes for a token sequence, one per full kv_block_size chunk
/// (Rust's compute_block_hash_for_seq; same XXH3-1337 over LE bytes).
inline std::vector<LocalBlockHash> compute_block_hashes_for_seq(const std::vector<Token>& tokens,
                                                                size_t kv_block_size) {
  std::vector<LocalBlockHash> hashes;
  if (kv_block_size == 0) return hashes;
  size_t full = tokens.size() / kv_block_size;
  hashes.reserve(full);
  for (size_t i = 0; i < full; ++i) {
    hashes.push_back(compute_hash(tokens.data() + i * kv_block_size,
                                  kv_block_size * sizeof(Token)));
  }
  return hashes;
}

struct ForwardPassMetrics {
  uint64_t request_active_slots = 0;
  uint64_t request_total_slots = 0;
  uint64_t kv_active_blocks = 0;
  uint64_t kv_total_blocks = 0;
  uint64_t num_requests_waiting = 0;
  float gpu_cache_usage_perc = 0;       // 0..1
  float gpu_prefix_cache_hit_rate = 0;  // 0..1
};

inline void to_json(nlohmann::json& j, const ForwardPassMetrics& m) {
  j = nlohmann::json{{"request_active_slots", m.request_active_slots},
                     {"request_total_slots", m.request_total_slots},
                     {"kv_active_blocks", m.kv_active_blocks},
                     {"kv_total_blocks", m.kv_total_blocks},
                     {"num_requests_waiting", m.num_requests_waiting},
                     {"gpu_cache_usage_perc", m.gpu_cache_usage_perc},
                     {"gpu_prefix_cache_hit_rate", m.gpu_prefix_cache_hit_rate}};
}

inline void from_json(const nlohmann::json& j, ForwardPassMetrics& m) {
  get_or(j, "request_active_slots", m.request_active_slots, {});
  get_or(j, "request_total_slots", m.request_total_slots, {});
  get_or(j, "kv_active_blocks", m.kv_active_blocks, {});
  get_or(j, "kv_total_blocks", m.kv_total_blocks, {});
  get_or(j, "num_requests_waiting", m.num_requests_waiting, {});
  get_or(j, "gpu_cache_usage_perc", m.gpu_cache_usage_perc, {});
  get_or(j, "gpu_prefix_cache_hit_rate", m.gpu_prefix_cache_hit_rate, {});
}

struct KvCacheStoredBlockData {
  ExternalSequenceBlockHash block_hash = 0;
  LocalBlockHash tokens_hash = 0;
};

inline void to_json(nlohmann::json& j, const KvCacheStoredBlockData& d) {
  j = nlohmann::json{{"block_hash", d.block_hash}, {"tokens_hash", d.tokens_hash}};
}
inline void from_json(const nlohmann::json& j, KvCacheStoredBlockData& d) {
  get_or(j, "block_hash", d.block_hash, {});
  get_or(j, "tokens_hash", d.tokens_hash, {});
}

struct KvCacheStoreData {
  std::optional<ExternalSequenceBlockHash> parent_hash;
  std::vector<KvCacheStoredBlockData> blocks;
};

inline void to_json(nlohmann::json& j, const KvCacheStoreData& d) {
  j = nlohmann::json{{"blocks", d.blocks}};
  j["parent_hash"] = d.parent_hash.has_value() ? nlohmann::json(*d.parent_hash)
                                               : nlohmann::json(nullptr);
}
inline void from_json(const nlohmann::json& j, KvCacheStoreData& d) {
  get_or(j, "blocks", d.blocks, {});
  get_opt(j, "parent_hash", d.parent_hash);
}

struct KvCacheRemoveData {
  std::vector<ExternalSequenceBlockHash> block_hashes;
};

inline void to_json(nlohmann::json& j, const KvCacheRemoveData& d) {
  j = nlohmann::json{{"block_hashes", d.block_hashes}};
}
inline void from_json(const nlohmann::json& j, KvCacheRemoveData& d) {
  get_or(j, "block_hashes", d.block_hashes, {});
}

/// Stored-or-removed event payload (externally tagged, as in serde).
struct KvCacheEventData {
  enum class Kind { stored, removed };
  Kind kind = Kind::stored;
  KvCacheStoreData stored;
  KvCacheRemoveData removed;

  static KvCacheEventData make_stored(KvCacheStoreData data) {
    return {Kind::stored, std::move(data), {}};
  }
  static KvCacheEventData make_removed(KvCacheRemoveData data) {
    return {Kind::removed, {}, std::move(data)};
  }
};

inline void to_json(nlohmann::json& j, const KvCacheEventData& d) {
  if (d.kind == KvCacheEventData::Kind::stored) {
    j = nlohmann::json{{"stored", d.stored}};
  } else {
    j = nlohmann::json{{"removed", d.removed}};
  }
}
inline void from_json(const nlohmann::json& j, KvCacheEventData& d) {
  if (j.contains("stored")) {
    d.kind = KvCacheEventData::Kind::stored;
    d.stored = j.at("stored").get<KvCacheStoreData>();
  } else if (j.contains("removed")) {
    d.kind = KvCacheEventData::Kind::removed;
    d.removed = j.at("removed").get<KvCacheRemoveData>();
  } else {
    throw std::invalid_argument("invalid KvCacheEventData: " + j.dump());
  }
}

struct KvCacheEvent {
  uint64_t event_id = 0;
  KvCacheEventData data;
};

inline void to_json(nlohmann::json& j, const KvCacheEvent& e) {
  j = nlohmann::json{{"event_id", e.event_id}, {"data", e.data}};
}
inline void from_json(const nlohmann::json& j, KvCacheEvent& e) {
  get_or(j, "event_id", e.event_id, {});
  get_or(j, "data", e.data, {});
}

/// A KvCacheEvent attributed to a worker (what rides the event plane).
struct RouterEvent {
  WorkerId worker_id = 0;
  KvCacheEvent event;
};

inline void to_json(nlohmann::json& j, const RouterEvent& e) {
  j = nlohmann::json{{"worker_id", e.worker_id}, {"event", e.event}};
}
inline void from_json(const nlohmann::json& j, RouterEvent& e) {
  get_or(j, "worker_id", e.worker_id, {});
  get_or(j, "event", e.event, {});
}

/// Published on kv-hit-rate after each scheduling decision.
struct KVHitRateEvent {
  WorkerId worker_id = 0;
  size_t isl_blocks = 0;
  size_t overlap_blocks = 0;
};

inline void to_json(nlohmann::json& j, const KVHitRateEvent& e) {
  j = nlohmann::json{{"worker_id", e.worker_id},
                     {"isl_blocks", e.isl_blocks},
                     {"overlap_blocks", e.overlap_blocks}};
}
inline void from_json(const nlohmann::json& j, KVHitRateEvent& e) {
  get_or(j, "worker_id", e.worker_id, {});
  get_or(j, "isl_blocks", e.isl_blocks, {});
  get_or(j, "overlap_blocks", e.overlap_blocks, {});
}

}  // namespace dynamo::llm::kv
