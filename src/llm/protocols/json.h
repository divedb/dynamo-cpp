// SPDX-License-Identifier: Apache-2.0
//
// Small helpers for tolerant JSON (de)serialization of optional fields:
// absent and null are both "no value"; present fields must convert.

#pragma once

#include <optional>

#include <nlohmann/json.hpp>

namespace dynamo::llm {

template <typename T>
void set_opt(nlohmann::json& j, const char* key, const std::optional<T>& v) {
  if (v.has_value()) j[key] = *v;
}

template <typename T>
void get_opt(const nlohmann::json& j, const char* key, std::optional<T>& v) {
  v.reset();
  if (auto it = j.find(key); it != j.end() && !it->is_null()) v = it->get<T>();
}

template <typename T>
void get_or(const nlohmann::json& j, const char* key, T& v, T fallback) {
  if (auto it = j.find(key); it != j.end() && !it->is_null()) {
    v = it->get<T>();
  } else {
    v = std::move(fallback);
  }
}

}  // namespace dynamo::llm
