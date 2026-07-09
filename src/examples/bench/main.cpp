// SPDX-License-Identifier: Apache-2.0
//
// Microbenchmarks: codec throughput, control-plane dispatch (unary) rate,
// and data-plane streaming throughput over loopback. Baseline numbers gate
// optimization decisions (e.g. whether per-stream data-plane connections are
// worth multiplexing away).

#include <chrono>
#include <cstdio>
#include <memory>
#include <numeric>
#include <string>

#include "component/component.h"
#include "runtime/logging.h"
#include "runtime/worker.h"
#include "transports/codec.h"

using namespace dynamo;
using Clock = std::chrono::steady_clock;

namespace {

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

void bench_codec() {
  transports::TwoPartCodec codec;
  std::string header(64, 'h');
  std::string body(1024, 'b');
  constexpr int kIters = 200000;

  auto start = Clock::now();
  size_t bytes = 0;
  for (int i = 0; i < kIters; ++i) {
    auto encoded = codec.encode(transports::TwoPartMessage::from_parts(header, body));
    auto decoded = codec.decode(encoded);
    bytes += encoded.size();
  }
  double secs = seconds_since(start);
  printf("codec      : %.0f msg/s, %.1f MB/s (1KB messages, encode+decode)\n",
         kIters / secs, bytes / secs / 1e6);
}

coro::AsyncGenerator<std::string> stream_n(int n) {
  std::string item(256, 'x');
  for (int i = 0; i < n; ++i) co_yield item;
}

struct BenchEngine final : pipeline::AsyncEngine<int, std::string> {
  coro::Task<pipeline::ManyOut<std::string>> generate(pipeline::SingleIn<int> in) override {
    auto [n, controller] = std::move(in).into_parts();
    pipeline::ContextPtr ctx = controller;
    co_return pipeline::ManyOut<std::string>(stream_n(n), ctx);
  }
};

coro::Task<void> bench_endpoint(Runtime rt) {
  auto drt = component::DistributedRuntime::create(rt);
  auto endpoint = drt.ns("bench").component("backend").endpoint("generate");
  auto lease = co_await drt.discovery()->create_lease(std::chrono::seconds(30));
  component::ServeOptions options;
  options.lease = lease;
  rt.spawn(endpoint.serve<int, std::string>(std::make_shared<BenchEngine>(), options));

  auto client = co_await endpoint.client<int, std::string>();
  co_await client.wait_for_instances();

  // Unary round trips: dispatch + call-home connect + 1 item + teardown.
  // This is the worst case for per-stream connection setup cost.
  {
    constexpr int kCalls = 2000;
    std::vector<double> latencies_us;
    latencies_us.reserve(kCalls);
    auto start = Clock::now();
    for (int i = 0; i < kCalls; ++i) {
      auto call_start = Clock::now();
      auto value = co_await client.unary(pipeline::SingleIn<int>(1));
      latencies_us.push_back(seconds_since(call_start) * 1e6);
    }
    double secs = seconds_since(start);
    std::sort(latencies_us.begin(), latencies_us.end());
    printf("unary      : %.0f req/s, p50 %.0fus, p99 %.0fus (full stream setup per call)\n",
           kCalls / secs, latencies_us[kCalls / 2], latencies_us[kCalls * 99 / 100]);
  }

  // Streaming throughput: one stream, many items.
  {
    constexpr int kItems = 100000;
    auto start = Clock::now();
    auto stream = co_await client.generate(pipeline::SingleIn<int>(kItems));
    size_t received = 0;
    while (auto item = co_await stream.next()) ++received;
    double secs = seconds_since(start);
    printf("streaming  : %.0f items/s, %.1f MB/s (256B items, single stream)\n",
           received / secs, received * 256 / secs / 1e6);
  }

  lease.revoke();
}

}  // namespace

int main() {
  logging::init();
  bench_codec();
  Worker worker(RuntimeConfig::from_env());
  return worker.execute([](Runtime rt) -> coro::Task<void> {
    co_await bench_endpoint(std::move(rt));
  });
}
