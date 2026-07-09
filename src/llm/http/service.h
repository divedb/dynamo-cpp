// SPDX-License-Identifier: Apache-2.0
//
// OpenAI-compatible HTTP frontend — Dynamo's lib/llm http/service (axum
// there, our minimal HTTP server here):
//   POST /v1/chat/completions   (streaming SSE or folded unary)
//   POST /v1/completions        (legacy)
//   GET  /v1/models             (OpenAI model listing)
//   GET  /metrics               (Prometheus text exposition)
//   GET  /health
// plus the ModelManager (model name -> engine) and per-request metrics
// (requests_total / inflight / duration, matching the Rust label scheme).

#pragma once

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "llm/http/http_server.h"
#include "llm/protocols/openai.h"
#include "pipeline/annotated.h"
#include "pipeline/engine.h"

namespace dynamo::llm::http {

using ChatEngine =
    pipeline::EnginePtr<openai::NvCreateChatCompletionRequest,
                        pipeline::Annotated<openai::NvCreateChatCompletionStreamResponse>>;
using CompletionEngine =
    pipeline::EnginePtr<openai::NvCreateCompletionRequest,
                        pipeline::Annotated<openai::NvCreateCompletionResponse>>;

/// Thread-safe model name -> engine registry (Rust's ModelManager).
class ModelManager {
 public:
  void add_chat_model(const std::string& name, ChatEngine engine);
  void add_completion_model(const std::string& name, CompletionEngine engine);
  void remove_chat_model(const std::string& name);
  void remove_completion_model(const std::string& name);

  ChatEngine get_chat_engine(const std::string& name) const;            // nullptr if absent
  CompletionEngine get_completion_engine(const std::string& name) const;

  std::vector<std::string> chat_models() const;
  std::vector<std::string> completion_models() const;

 private:
  mutable std::mutex mutex_;
  std::map<std::string, ChatEngine> chat_engines_;
  std::map<std::string, CompletionEngine> completion_engines_;
};

/// HTTP service metrics with Prometheus text exposition (Rust's Metrics:
/// {prefix}_http_service_requests_total{model,endpoint,request_type,status},
/// {prefix}_http_service_inflight_requests{model},
/// {prefix}_http_service_request_duration_seconds{model}).
class Metrics {
 public:
  explicit Metrics(std::string prefix = "nv_llm") : prefix_(std::move(prefix)) {}

  /// RAII per-request guard: inflight while alive; on destruction records
  /// the counter (status label from mark_ok) and the duration histogram.
  class InflightGuard {
   public:
    InflightGuard(Metrics& metrics, std::string model, std::string endpoint, bool streaming);
    ~InflightGuard();
    InflightGuard(InflightGuard&&) = delete;
    void mark_ok() { ok_ = true; }

   private:
    Metrics& metrics_;
    std::string model_;
    std::string endpoint_;
    bool streaming_;
    bool ok_ = false;
    std::chrono::steady_clock::time_point start_;
  };

  uint64_t request_count(const std::string& model, const std::string& endpoint,
                         const std::string& request_type, const std::string& status) const;
  int64_t inflight(const std::string& model) const;

  /// Prometheus text format.
  std::string render() const;

 private:
  friend class InflightGuard;
  struct Histogram {
    std::vector<uint64_t> bucket_counts;  // cumulative at render time; raw here
    double sum = 0;
    uint64_t count = 0;
  };

  static const std::vector<double>& buckets();

  using CounterKey = std::tuple<std::string, std::string, std::string, std::string>;

  mutable std::mutex mutex_;
  std::string prefix_;
  std::map<CounterKey, uint64_t> request_counter_;  // (model, endpoint, type, status)
  std::map<std::string, int64_t> inflight_gauge_;   // model
  std::map<std::string, Histogram> duration_;       // model
};

struct HttpServiceOptions {
  std::string host = "127.0.0.1";
  uint16_t port = 8080;  // 0 = ephemeral
  bool enable_chat_endpoints = true;
  bool enable_completion_endpoints = true;
};

/// The frontend: routes + state. start() binds and serves; engines come and
/// go through model_manager() (directly or via the model watcher).
class HttpService {
 public:
  explicit HttpService(HttpServiceOptions options = {});
  ~HttpService();

  void start();
  void stop();
  uint16_t port() const { return server_.port(); }

  const std::shared_ptr<ModelManager>& model_manager() { return manager_; }
  Metrics& metrics() { return *metrics_; }

 private:
  HttpResponse handle_chat(const HttpRequest& request);
  HttpResponse handle_completions(const HttpRequest& request);
  HttpResponse handle_list_models(const HttpRequest& request);

  HttpServiceOptions options_;
  std::shared_ptr<ModelManager> manager_ = std::make_shared<ModelManager>();
  std::shared_ptr<Metrics> metrics_ = std::make_shared<Metrics>();
  HttpServer server_;
};

}  // namespace dynamo::llm::http
