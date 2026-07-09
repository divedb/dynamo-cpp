// SPDX-License-Identifier: Apache-2.0
//
// Lists live instances of an endpoint. Usage:
//   instance_ls <ns> <component> <endpoint>     (DYN_DISCOVERY must be set)
// Prints one "instance_id subject address" line per instance; the count is
// the exit-visible signal for scripts (also printed as the last line).

#include <cstdio>

#include "component/component.h"
#include "runtime/coro/sync_wait.h"
#include "runtime/logging.h"

using namespace dynamo;

int main(int argc, char** argv) {
  logging::init();
  if (argc != 4) {
    fprintf(stderr, "usage: instance_ls <namespace> <component> <endpoint>\n");
    return 2;
  }

  auto rt = Runtime::create(RuntimeConfig::single_threaded());
  int count = 0;
  {
    auto drt = component::DistributedRuntime::create(rt);
    auto endpoint = drt.ns(argv[1]).component(argv[2]).endpoint(argv[3]);

    coro::sync_wait([&]() -> coro::Task<void> {
      auto kvs = co_await drt.discovery()->kv_get_prefix(endpoint.key_prefix() + ":");
      for (auto& kv : kvs) {
        auto info = nlohmann::json::parse(kv.value).get<component::EndpointInfo>();
        printf("%s %s %s\n", component::hex_id(info.instance_id).c_str(),
               info.subject.c_str(), info.address.c_str());
        ++count;
      }
    }());

    rt.shutdown();
    rt.join_tasks(std::chrono::milliseconds(3000));
  }
  printf("count=%d\n", count);
  return 0;
}
