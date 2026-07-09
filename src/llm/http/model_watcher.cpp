// SPDX-License-Identifier: Apache-2.0

#include "llm/http/model_watcher.h"

#include <spdlog/spdlog.h>

#include "llm/protocols/json.h"

namespace dynamo::llm::http {

void to_json(nlohmann::json& j, const ModelEntry& e) {
  j = nlohmann::json{{"name", e.name},
                     {"endpoint",
                      {{"namespace", e.namespace_name},
                       {"component", e.component},
                       {"name", e.endpoint}}},
                     {"model_type", e.model_type}};
}

void from_json(const nlohmann::json& j, ModelEntry& e) {
  get_or(j, "name", e.name, {});
  get_or(j, "model_type", e.model_type, {});
  if (auto it = j.find("endpoint"); it != j.end() && it->is_object()) {
    e.namespace_name = it->value("namespace", "");
    e.component = it->value("component", "");
    e.endpoint = it->value("name", "");
  }
}

coro::Task<void> register_model_entry(discovery::Discovery& discovery, ModelEntry entry,
                                      std::string prefix, std::optional<int64_t> lease_id) {
  std::string key = entry.kv_key(prefix);
  co_await discovery.kv_put(key, nlohmann::json(entry).dump(), lease_id);
}

namespace {

coro::Task<void> add_model(component::DistributedRuntime drt,
                           std::shared_ptr<ModelManager> manager, ModelEntry entry) {
  auto endpoint =
      drt.ns(entry.namespace_name).component(entry.component).endpoint(entry.endpoint);
  if (entry.model_type == "chat") {
    auto client =
        co_await endpoint
            .client<openai::NvCreateChatCompletionRequest,
                    pipeline::Annotated<openai::NvCreateChatCompletionStreamResponse>>();
    manager->add_chat_model(entry.name, client.as_engine());
  } else if (entry.model_type == "completion") {
    auto client = co_await endpoint.client<openai::NvCreateCompletionRequest,
                                           pipeline::Annotated<openai::NvCreateCompletionResponse>>();
    manager->add_completion_model(entry.name, client.as_engine());
  } else {
    throw std::runtime_error("unknown model_type: " + entry.model_type);
  }
}

/// "{prefix}{type}/{name}" -> (type, name); empty type on mismatch.
std::pair<std::string, std::string> split_key(const std::string& key,
                                              const std::string& prefix) {
  if (key.rfind(prefix, 0) != 0) return {};
  std::string rest = key.substr(prefix.size());
  size_t slash = rest.find('/');
  if (slash == std::string::npos) return {};
  return {rest.substr(0, slash), rest.substr(slash + 1)};
}

}  // namespace

coro::Task<void> run_model_watcher(component::DistributedRuntime drt,
                                   std::shared_ptr<ModelManager> manager, std::string prefix) {
  auto watch = co_await drt.discovery()->kv_get_and_watch_prefix(prefix);
  // End the watch (unblocking recv) when the runtime shuts down; discovery
  // backends may keep watch channels open past shutdown (in-process store).
  auto shutdown_token = drt.runtime().child_token();
  auto registration =
      shutdown_token.register_callback([events = watch.events]() mutable { events.close(); });
  spdlog::debug("model watcher started on prefix {}", prefix);

  while (auto event = co_await watch.events.recv()) {
    if (event->kind == discovery::WatchEvent::Kind::Put) {
      try {
        auto entry = nlohmann::json::parse(event->kv.value).get<ModelEntry>();
        co_await add_model(drt, manager, entry);
        spdlog::info("added {} model: {}", entry.model_type, entry.name);
      } catch (const std::exception& e) {
        spdlog::error("error adding model at {}: {}", event->kv.key, e.what());
      }
    } else {
      auto [type, name] = split_key(event->kv.key, prefix);
      if (type == "chat") {
        manager->remove_chat_model(name);
        spdlog::info("removed chat model: {}", name);
      } else if (type == "completion") {
        manager->remove_completion_model(name);
        spdlog::info("removed completion model: {}", name);
      }
    }
  }
  spdlog::debug("model watcher on prefix {} ended", prefix);
}

}  // namespace dynamo::llm::http
