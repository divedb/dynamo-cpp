// SPDX-License-Identifier: Apache-2.0
//
// Test helper: an app that ignores cancellation, to exercise the Worker's
// graceful-shutdown hard-exit (911) path. Run with a short
// DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT and send SIGTERM.

#include <cstdio>
#include <thread>

#include "runtime/worker.h"

using namespace dynamo;

int main() {
  Worker worker(RuntimeConfig::from_env());
  return worker.execute([](Runtime) -> coro::Task<void> {
    printf("hang_worker: running\n");
    fflush(stdout);
    for (;;) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    co_return;  // unreachable; makes this a coroutine
  });
}
