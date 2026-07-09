// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstddef>
#include <string>

namespace dynamo {

/// Runtime settings, layered: built-in defaults ← TOML config file (path in
/// DYN_CONFIG, `[runtime]` table) ← environment variables (highest priority).
struct RuntimeConfig {
  /// Primary (application) pool size; 0 = hardware concurrency.
  size_t num_worker_threads = 0;  // toml: worker_threads / env: DYN_RUNTIME_NUM_WORKER_THREADS

  /// Secondary (background) pool size.
  size_t num_background_threads = 1;  // toml: background_threads

  /// Graceful shutdown window granted after cancellation before hard exit.
  std::chrono::seconds graceful_shutdown_timeout{5};  // DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT

  static RuntimeConfig from_env();
  static RuntimeConfig single_threaded();
};

/// Reads an environment variable, or `fallback` when unset.
std::string env_or(const char* name, const std::string& fallback);

/// Reads `[table] key` from the DYN_CONFIG TOML file, or `fallback` when the
/// file/table/key is absent. The file is parsed once per process.
std::string config_or(const char* table, const char* key, const std::string& fallback);
int64_t config_or(const char* table, const char* key, int64_t fallback);

/// "1", "true", "yes", "on" (case-insensitive) — Dynamo's is_truthy.
bool env_is_truthy(const char* name);

}  // namespace dynamo
