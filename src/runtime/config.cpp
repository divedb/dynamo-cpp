// SPDX-License-Identifier: Apache-2.0

#include "runtime/config.h"

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <thread>

#include <spdlog/spdlog.h>
#include <toml.hpp>

namespace dynamo {

std::string env_or(const char* name, const std::string& fallback) {
  const char* v = std::getenv(name);
  return v ? std::string(v) : fallback;
}

bool env_is_truthy(const char* name) {
  std::string v = env_or(name, "");
  std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return std::tolower(c); });
  return v == "1" || v == "true" || v == "yes" || v == "on";
}

namespace {

/// Parses the DYN_CONFIG file once; nullptr when unset or unparsable.
const toml::value* config_file() {
  static std::optional<toml::value> parsed = []() -> std::optional<toml::value> {
    auto path = env_or("DYN_CONFIG", "");
    if (path.empty()) return std::nullopt;
    try {
      return toml::parse(path);
    } catch (const std::exception& e) {
      spdlog::error("failed to parse DYN_CONFIG file {}: {}", path, e.what());
      return std::nullopt;
    }
  }();
  return parsed ? &*parsed : nullptr;
}

}  // namespace

std::string config_or(const char* table, const char* key, const std::string& fallback) {
  const auto* file = config_file();
  if (!file || !file->contains(table)) return fallback;
  return toml::find_or(toml::find(*file, table), key, fallback);
}

int64_t config_or(const char* table, const char* key, int64_t fallback) {
  const auto* file = config_file();
  if (!file || !file->contains(table)) return fallback;
  return toml::find_or(toml::find(*file, table), key, fallback);
}

RuntimeConfig RuntimeConfig::from_env() {
  RuntimeConfig config;

  // Layer 1: TOML config file ([runtime] table).
  config.num_worker_threads =
      static_cast<size_t>(config_or("runtime", "worker_threads", int64_t{0}));
  config.num_background_threads = static_cast<size_t>(config_or(
      "runtime", "background_threads", static_cast<int64_t>(config.num_background_threads)));
  config.graceful_shutdown_timeout = std::chrono::seconds(
      config_or("runtime", "graceful_shutdown_timeout_s",
                static_cast<int64_t>(config.graceful_shutdown_timeout.count())));

  // Layer 2: environment overrides.
  if (auto v = env_or("DYN_RUNTIME_NUM_WORKER_THREADS", ""); !v.empty()) {
    config.num_worker_threads = static_cast<size_t>(std::stoul(v));
  }
  if (auto v = env_or("DYN_RUNTIME_NUM_BACKGROUND_THREADS", ""); !v.empty()) {
    config.num_background_threads = static_cast<size_t>(std::stoul(v));
  }
  if (auto v = env_or("DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT", ""); !v.empty()) {
    config.graceful_shutdown_timeout = std::chrono::seconds(std::stoul(v));
  }

  if (config.num_worker_threads == 0) {
    config.num_worker_threads = std::max(2u, std::thread::hardware_concurrency());
  }
  if (config.num_background_threads == 0) config.num_background_threads = 1;
  return config;
}

RuntimeConfig RuntimeConfig::single_threaded() {
  RuntimeConfig config;
  config.num_worker_threads = 1;
  config.num_background_threads = 1;
  return config;
}

}  // namespace dynamo
