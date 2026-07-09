// SPDX-License-Identifier: Apache-2.0
//
// SSE-like response envelope carrying data, events (including in-band
// errors), ids and comments — Dynamo's protocols::annotated::Annotated.

#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace dynamo::pipeline {

template <typename R>
struct Annotated {
  std::optional<R> data;
  std::optional<std::string> id;
  std::optional<std::string> event;
  std::optional<std::vector<std::string>> comment;

  static Annotated from_data(R value) {
    Annotated a;
    a.data = std::move(value);
    return a;
  }

  static Annotated from_error(std::string message) {
    Annotated a;
    a.event = "error";
    a.comment = std::vector<std::string>{std::move(message)};
    return a;
  }

  static Annotated from_annotation(std::string name, const nlohmann::json& value) {
    Annotated a;
    a.event = std::move(name);
    a.comment = std::vector<std::string>{value.dump()};
    return a;
  }

  bool is_error() const { return event.has_value() && *event == "error"; }
  bool is_event() const { return event.has_value(); }

  /// Data if present; throws if this is an error annotation.
  std::optional<R> into_result() && {
    if (is_error()) {
      std::string msg = comment && !comment->empty() ? comment->front() : "unknown error";
      throw std::runtime_error(msg);
    }
    return std::move(data);
  }
};

template <typename R>
void to_json(nlohmann::json& j, const Annotated<R>& a) {
  j = nlohmann::json::object();
  if (a.data) j["data"] = *a.data;
  if (a.id) j["id"] = *a.id;
  if (a.event) j["event"] = *a.event;
  if (a.comment) j["comment"] = *a.comment;
}

template <typename R>
void from_json(const nlohmann::json& j, Annotated<R>& a) {
  if (j.contains("data")) a.data = j.at("data").get<R>();
  if (j.contains("id")) a.id = j.at("id").get<std::string>();
  if (j.contains("event")) a.event = j.at("event").get<std::string>();
  if (j.contains("comment")) a.comment = j.at("comment").get<std::vector<std::string>>();
}

}  // namespace dynamo::pipeline
