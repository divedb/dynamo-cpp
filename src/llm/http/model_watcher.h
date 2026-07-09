// SPDX-License-Identifier: Apache-2.0
//
// Model discovery for the HTTP frontend — Dynamo's http/service/discovery.rs
// + llmctl's registration: workers (or an operator CLI) publish ModelEntry
// records ({name, endpoint, model_type}) under "{prefix}{type}/{name}" in
// the discovery kv store; the frontend watches the prefix and adds/removes
// engines (component Clients) as entries come and go.

#pragma once

#include <memory>
#include <string>

#include "component/component.h"
#include "llm/http/service.h"

namespace dynamo::llm::http {

/// Default kv prefix for model entries served by an HTTP frontend component.
inline constexpr const char* kModelsPrefix = "models/";

struct ModelEntry {
  /// Model name as exposed on /v1/* (OpenAI `model` field).
  std::string name;
  /// The component endpoint serving this model.
  std::string namespace_name;
  std::string component;
  std::string endpoint;
  /// "chat" or "completion".
  std::string model_type;

  std::string kv_key(const std::string& prefix) const {
    return prefix + model_type + "/" + name;
  }
};

void to_json(nlohmann::json& j, const ModelEntry& e);
void from_json(const nlohmann::json& j, ModelEntry& e);

/// Publishes a model entry (llmctl `add model`); bound to `lease_id` when
/// given so it disappears with its worker.
coro::Task<void> register_model_entry(discovery::Discovery& discovery, ModelEntry entry,
                                      std::string prefix = kModelsPrefix,
                                      std::optional<int64_t> lease_id = std::nullopt);

/// Removes a model entry. (Deletion happens via lease revocation today:
/// entries published with a lease vanish with it; discoveryd has no
/// unconditional delete op — parity gap tracked in TODO.md.)

/// Watches `prefix` and keeps `manager` in sync: Put -> build a component
/// Client for the entry's endpoint and register it under the entry's name;
/// Delete -> remove. Runs until the watch stream closes (discovery
/// shutdown). Spawn on the runtime:
///   runtime.spawn(run_model_watcher(drt, service.model_manager()));
coro::Task<void> run_model_watcher(component::DistributedRuntime drt,
                                   std::shared_ptr<ModelManager> manager,
                                   std::string prefix = kModelsPrefix);

}  // namespace dynamo::llm::http
