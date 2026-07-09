// SPDX-License-Identifier: Apache-2.0
//
// JSON wire structures shared by the control plane and data plane.

#pragma once

#include <optional>
#include <string>

#include <nlohmann/json.hpp>

namespace dynamo::transports {

enum class StreamKind { Request, Response };

NLOHMANN_JSON_SERIALIZE_ENUM(StreamKind, {{StreamKind::Request, "request"},
                                          {StreamKind::Response, "response"}})

/// Where and how a pending stream can be reached (advertised by the
/// requester, used by the worker to call home).
struct ConnectionInfo {
  std::string transport = "tcp";
  std::string address;     // host:port of the requester's data-plane server
  std::string subject;     // per-stream routing key
  std::string context_id;  // request context id (validated on connect)
  StreamKind stream_type = StreamKind::Response;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ConnectionInfo, transport, address, subject, context_id,
                                   stream_type)

/// First frame a worker sends when calling home.
struct CallHomeHandshake {
  std::string subject;
  StreamKind stream_type = StreamKind::Response;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(CallHomeHandshake, subject, stream_type)

/// Second frame: outcome of the worker-side generate() call. The requester's
/// dispatch resolves only once this arrives; an error fails the call.
struct Prologue {
  std::optional<std::string> error;
};

inline void to_json(nlohmann::json& j, const Prologue& p) {
  j = nlohmann::json::object();
  if (p.error) j["error"] = *p.error;
}
inline void from_json(const nlohmann::json& j, Prologue& p) {
  if (j.contains("error") && !j.at("error").is_null()) p.error = j.at("error").get<std::string>();
}

/// In-band stream control. stop/kill flow requester→worker; sentinel flows
/// worker→requester to terminate the stream cleanly.
enum class ControlKind { Stop, Kill, Sentinel };

NLOHMANN_JSON_SERIALIZE_ENUM(ControlKind, {{ControlKind::Stop, "stop"},
                                           {ControlKind::Kill, "kill"},
                                           {ControlKind::Sentinel, "sentinel"}})

struct ControlFrame {
  ControlKind kind = ControlKind::Stop;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ControlFrame, kind)

/// Control-plane request header accompanying the serialized request payload.
struct RequestControlMessage {
  std::string id;  // request/context id
  std::string request_type = "single_in";
  std::string response_type = "many_out";
  ConnectionInfo connection_info;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(RequestControlMessage, id, request_type, response_type,
                                   connection_info)

/// Envelope routing a control-plane frame to a specific endpoint instance.
struct DispatchHeader {
  std::string subject;
  RequestControlMessage control;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DispatchHeader, subject, control)

struct DispatchAck {
  bool ok = false;
  std::string error;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DispatchAck, ok, error)

}  // namespace dynamo::transports
