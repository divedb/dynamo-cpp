// SPDX-License-Identifier: Apache-2.0

#include "runtime/logging.h"

#include <cstdlib>
#include <mutex>
#include <string>

#include <spdlog/spdlog.h>

#include "runtime/config.h"

namespace dynamo::logging {

void init() {
  static std::once_flag once;
  std::call_once(once, [] {
    if (env_is_truthy("DYN_LOGGING_JSONL")) {
      // JSONL output (Dynamo's DYN_LOGGING_JSONL). Note: the message body is
      // not JSON-escaped by spdlog; messages containing quotes/backslashes
      // can produce invalid lines. Acceptable for machine-scraping typical
      // log content; a custom escaping formatter is future work.
      spdlog::set_pattern(
          R"({"ts":"%Y-%m-%dT%H:%M:%S.%e%z","level":"%l","thread":%t,"msg":"%v"})");
    } else if (env_is_truthy("DYN_LOGGING_DISABLE_ANSI")) {
      spdlog::set_pattern("[%Y-%m-%dT%H:%M:%S.%e] [%l] [t:%t] %v");
    } else {
      spdlog::set_pattern("[%Y-%m-%dT%H:%M:%S.%e] [%^%l%$] [t:%t] %v");
    }

    std::string level = env_or("DYN_LOG", "info");
    if (level == "trace")
      spdlog::set_level(spdlog::level::trace);
    else if (level == "debug")
      spdlog::set_level(spdlog::level::debug);
    else if (level == "warn")
      spdlog::set_level(spdlog::level::warn);
    else if (level == "error")
      spdlog::set_level(spdlog::level::err);
    else
      spdlog::set_level(spdlog::level::info);
  });
}

}  // namespace dynamo::logging
