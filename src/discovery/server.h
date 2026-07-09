// SPDX-License-Identifier: Apache-2.0
//
// discoveryd: a small TCP discovery server reproducing the etcd behaviors
// Dynamo relies on — leases with TTL + keep-alive, atomic create-if-absent
// keys bound to leases, and get-and-watch prefix streams. Hostable in-process
// (tests) or as the standalone `discoveryd` example binary.

#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace dynamo::discovery {

class DiscoveryServer {
 public:
  static std::shared_ptr<DiscoveryServer> start(const std::string& host = "127.0.0.1",
                                                uint16_t port = 0);
  ~DiscoveryServer();

  std::string address() const;
  uint16_t port() const;
  void stop();

  struct State;  // opaque; public so implementation helpers can name it

 private:
  DiscoveryServer() = default;
  std::shared_ptr<State> state_;
};

}  // namespace dynamo::discovery
