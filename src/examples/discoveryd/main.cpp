// SPDX-License-Identifier: Apache-2.0
//
// Standalone discovery server. Usage: discoveryd [port]  (default 7787)

#include <cstdlib>

#include "discovery/server.h"
#include "runtime/logging.h"
#include "runtime/worker.h"

using namespace dynamo;

int main(int argc, char** argv) {
  logging::init();
  uint16_t port = argc > 1 ? static_cast<uint16_t>(std::atoi(argv[1])) : 7787;

  auto worker = Worker::from_env();
  return worker.execute([port](Runtime rt) -> coro::Task<void> {
    auto server = discovery::DiscoveryServer::start("127.0.0.1", port);
    co_await rt.token().cancelled();  // run until SIGINT/SIGTERM
    server->stop();
  });
}
