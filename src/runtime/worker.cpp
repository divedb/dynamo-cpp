// SPDX-License-Identifier: Apache-2.0

#include "runtime/worker.h"

#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <unistd.h>

#include <poll.h>
#include <spdlog/spdlog.h>

#include "runtime/coro/sync_wait.h"

namespace dynamo {

namespace {

// Self-pipe written from the (async-signal-safe) handler; drained by the
// signal thread, which performs the actual cancellation. Shared process
// state — the reason only one Worker may be alive at a time.
int g_signal_pipe[2] = {-1, -1};

std::atomic<bool> g_worker_alive{false};

void signal_handler(int) {
  char byte = 1;
  [[maybe_unused]] ssize_t n = ::write(g_signal_pipe[1], &byte, 1);
}

}  // namespace

Worker::Slot::Slot() {
  bool expected = false;
  if (!g_worker_alive.compare_exchange_strong(expected, true)) {
    throw std::runtime_error(
        "a Worker already exists in this process; only one may be alive at a time");
  }
}

Worker::Slot::~Slot() { g_worker_alive.store(false); }

int Worker::execute(std::function<coro::Task<void>(Runtime)> app) {
  if (executed_.exchange(true)) {
    throw std::logic_error("Worker::execute() may only be called once");
  }

  Runtime runtime = runtime_;
  auto token = runtime.token();

  // Self-pipe pair: [0]=signal events, plus a shutdown pipe to stop the thread.
  int shutdown_pipe[2];
  if (::pipe(g_signal_pipe) != 0 || ::pipe(shutdown_pipe) != 0) {
    spdlog::error("worker: failed to create signal pipes");
    return 1;
  }

  struct sigaction sa {};
  sa.sa_handler = signal_handler;
  sigemptyset(&sa.sa_mask);
  ::sigaction(SIGINT, &sa, nullptr);
  ::sigaction(SIGTERM, &sa, nullptr);

  std::thread signal_thread([&token, shutdown_fd = shutdown_pipe[0]] {
    struct pollfd fds[2] = {{g_signal_pipe[0], POLLIN, 0}, {shutdown_fd, POLLIN, 0}};
    if (::poll(fds, 2, -1) > 0 && (fds[0].revents & POLLIN)) {
      spdlog::info("shutdown signal received; starting graceful shutdown");
      token.cancel();
    }
  });

  // Graceful-shutdown enforcement: once cancelled, the app gets a bounded
  // window to finish before the process is terminated.
  std::mutex done_mutex;
  std::condition_variable done_cv;
  bool app_done = false;
  auto timeout = runtime.config().graceful_shutdown_timeout;

  std::thread grace_thread([&] {
    std::unique_lock lock(done_mutex);
    while (!app_done && !token.is_cancelled()) {
      done_cv.wait_for(lock, std::chrono::milliseconds(50));
    }
    if (app_done) return;
    spdlog::debug("graceful shutdown window: {}s", timeout.count());
    if (!done_cv.wait_for(lock, timeout, [&] { return app_done; })) {
      spdlog::error("application did not shut down within {}s; terminating", timeout.count());
      std::_Exit(kGracefulShutdownExitCode);
    }
  });

  int exit_code = 0;
  try {
    coro::sync_wait([](Runtime rt, std::function<coro::Task<void>(Runtime)> f) -> coro::Task<void> {
      co_await rt.primary().schedule();
      co_await f(rt);
    }(runtime, std::move(app)));
  } catch (const std::exception& e) {
    spdlog::error("application exited with error: {}", e.what());
    exit_code = 1;
  }

  // App finished: shut everything down and drain background tasks.
  runtime.shutdown();
  {
    std::lock_guard lock(done_mutex);
    app_done = true;
  }
  done_cv.notify_all();
  grace_thread.join();

  if (!runtime.join_tasks(std::chrono::duration_cast<std::chrono::milliseconds>(timeout))) {
    spdlog::warn("background tasks did not drain within the graceful window");
  }

  // Stop the signal thread and restore default handlers.
  {
    char byte = 1;
    [[maybe_unused]] ssize_t n = ::write(shutdown_pipe[1], &byte, 1);
  }
  signal_thread.join();
  ::signal(SIGINT, SIG_DFL);
  ::signal(SIGTERM, SIG_DFL);
  for (int fd : {g_signal_pipe[0], g_signal_pipe[1], shutdown_pipe[0], shutdown_pipe[1]}) {
    ::close(fd);
  }
  g_signal_pipe[0] = g_signal_pipe[1] = -1;

  return exit_code;
}

}  // namespace dynamo
