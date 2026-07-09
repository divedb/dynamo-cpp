// SPDX-License-Identifier: Apache-2.0
//
// Identity scheme (Dynamo's path model):
//   component key prefix : {ns}/components/{component}
//   endpoint key prefix  : {ns}/components/{component}/{endpoint}
//   instance key         : {endpoint prefix}:{instance_id:x}
//   instance subject     : {ns}/{component}/{endpoint}:{instance_id:x}
// A running endpoint instance is identified by its lease id (= instance id).

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace dynamo::component {

/// Registered per instance in discovery (Dynamo's ComponentEndpointInfo).
struct EndpointInfo {
  std::string namespace_name;
  std::string component;
  std::string endpoint;
  int64_t instance_id = 0;  // lease id
  std::string transport = "dyn-tcp";
  std::string address;      // control-plane address of the serving worker
  std::string subject;      // instance-addressed routing subject
  std::string description;  // human-readable service description
  std::string version;      // service version advertised by the instance
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(EndpointInfo, namespace_name, component,
                                                endpoint, instance_id, transport, address,
                                                subject, description, version)

/// One instance's reply to a cluster-wide stats scrape.
struct EndpointStats {
  EndpointInfo info;
  bool ok = false;
  std::string error;
  nlohmann::json stats;
};

/// Aggregate of a scrape across all live instances of a component
/// (Dynamo's ServiceSet, collected via NATS service broadcast there).
struct ServiceSet {
  std::vector<EndpointStats> endpoints;
};

inline std::string hex_id(int64_t id) {
  char buf[24];
  snprintf(buf, sizeof(buf), "%llx", static_cast<unsigned long long>(id));
  return buf;
}

inline void validate_name(const std::string& name, const char* what) {
  if (name.empty()) throw std::invalid_argument(std::string(what) + " must not be empty");
  for (char c : name) {
    bool ok = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '_';
    if (!ok) {
      throw std::invalid_argument(std::string(what) + " has invalid character: " + name);
    }
  }
}

}  // namespace dynamo::component
