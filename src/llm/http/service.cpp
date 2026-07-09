// SPDX-License-Identifier: Apache-2.0

#include "llm/http/service.h"

#include <chrono>
#include <random>
#include <set>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include "llm/protocols/sse.h"
#include "runtime/coro/sync_wait.h"

namespace dynamo::llm::http {

using nlohmann::json;

// ---------------------------------------------------------------------------
// ModelManager

void ModelManager::add_chat_model(const std::string& name, ChatEngine engine) {
  std::lock_guard lock(mutex_);
  chat_engines_[name] = std::move(engine);
}

void ModelManager::add_completion_model(const std::string& name, CompletionEngine engine) {
  std::lock_guard lock(mutex_);
  completion_engines_[name] = std::move(engine);
}

void ModelManager::remove_chat_model(const std::string& name) {
  std::lock_guard lock(mutex_);
  chat_engines_.erase(name);
}

void ModelManager::remove_completion_model(const std::string& name) {
  std::lock_guard lock(mutex_);
  completion_engines_.erase(name);
}

ChatEngine ModelManager::get_chat_engine(const std::string& name) const {
  std::lock_guard lock(mutex_);
  auto it = chat_engines_.find(name);
  return it == chat_engines_.end() ? nullptr : it->second;
}

CompletionEngine ModelManager::get_completion_engine(const std::string& name) const {
  std::lock_guard lock(mutex_);
  auto it = completion_engines_.find(name);
  return it == completion_engines_.end() ? nullptr : it->second;
}

std::vector<std::string> ModelManager::chat_models() const {
  std::lock_guard lock(mutex_);
  std::vector<std::string> names;
  names.reserve(chat_engines_.size());
  for (const auto& [name, engine] : chat_engines_) names.push_back(name);
  return names;
}

std::vector<std::string> ModelManager::completion_models() const {
  std::lock_guard lock(mutex_);
  std::vector<std::string> names;
  names.reserve(completion_engines_.size());
  for (const auto& [name, engine] : completion_engines_) names.push_back(name);
  return names;
}

// ---------------------------------------------------------------------------
// Metrics

const std::vector<double>& Metrics::buckets() {
  static const std::vector<double> kBuckets = {0.0, 1.0,  2.0,  4.0,   8.0,
                                               16.0, 32.0, 64.0, 128.0, 256.0};
  return kBuckets;
}

Metrics::InflightGuard::InflightGuard(Metrics& metrics, std::string model, std::string endpoint,
                                      bool streaming)
    : metrics_(metrics),
      model_(std::move(model)),
      endpoint_(std::move(endpoint)),
      streaming_(streaming),
      start_(std::chrono::steady_clock::now()) {
  std::lock_guard lock(metrics_.mutex_);
  ++metrics_.inflight_gauge_[model_];
}

Metrics::InflightGuard::~InflightGuard() {
  double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
  std::lock_guard lock(metrics_.mutex_);
  --metrics_.inflight_gauge_[model_];
  const char* type = streaming_ ? "stream" : "unary";
  const char* status = ok_ ? "success" : "error";
  ++metrics_.request_counter_[{model_, endpoint_, type, status}];

  auto& histogram = metrics_.duration_[model_];
  if (histogram.bucket_counts.size() != buckets().size()) {
    histogram.bucket_counts.assign(buckets().size(), 0);
  }
  for (size_t i = 0; i < buckets().size(); ++i) {
    if (seconds <= buckets()[i]) {
      ++histogram.bucket_counts[i];
      break;
    }
  }
  histogram.sum += seconds;
  ++histogram.count;
}

uint64_t Metrics::request_count(const std::string& model, const std::string& endpoint,
                                const std::string& request_type,
                                const std::string& status) const {
  std::lock_guard lock(mutex_);
  auto it = request_counter_.find({model, endpoint, request_type, status});
  return it == request_counter_.end() ? 0 : it->second;
}

int64_t Metrics::inflight(const std::string& model) const {
  std::lock_guard lock(mutex_);
  auto it = inflight_gauge_.find(model);
  return it == inflight_gauge_.end() ? 0 : it->second;
}

std::string Metrics::render() const {
  std::lock_guard lock(mutex_);
  std::string out;

  std::string counter_name = prefix_ + "_http_service_requests_total";
  out += fmt::format("# HELP {} Total number of LLM requests processed\n", counter_name);
  out += fmt::format("# TYPE {} counter\n", counter_name);
  for (const auto& [key, value] : request_counter_) {
    const auto& [model, endpoint, type, status] = key;
    out += fmt::format(
        "{}{{model=\"{}\",endpoint=\"{}\",request_type=\"{}\",status=\"{}\"}} {}\n",
        counter_name, model, endpoint, type, status, value);
  }

  std::string gauge_name = prefix_ + "_http_service_inflight_requests";
  out += fmt::format("# HELP {} Number of inflight requests\n", gauge_name);
  out += fmt::format("# TYPE {} gauge\n", gauge_name);
  for (const auto& [model, value] : inflight_gauge_) {
    out += fmt::format("{}{{model=\"{}\"}} {}\n", gauge_name, model, value);
  }

  std::string histogram_name = prefix_ + "_http_service_request_duration_seconds";
  out += fmt::format("# HELP {} Duration of LLM requests\n", histogram_name);
  out += fmt::format("# TYPE {} histogram\n", histogram_name);
  for (const auto& [model, histogram] : duration_) {
    uint64_t cumulative = 0;
    for (size_t i = 0; i < buckets().size() && i < histogram.bucket_counts.size(); ++i) {
      cumulative += histogram.bucket_counts[i];
      out += fmt::format("{}_bucket{{model=\"{}\",le=\"{}\"}} {}\n", histogram_name, model,
                         buckets()[i], cumulative);
    }
    out += fmt::format("{}_bucket{{model=\"{}\",le=\"+Inf\"}} {}\n", histogram_name, model,
                       histogram.count);
    out += fmt::format("{}_sum{{model=\"{}\"}} {}\n", histogram_name, model, histogram.sum);
    out += fmt::format("{}_count{{model=\"{}\"}} {}\n", histogram_name, model, histogram.count);
  }

  return out;
}

// ---------------------------------------------------------------------------
// HttpService

namespace {

HttpResponse json_error(int status, const std::string& message) {
  return HttpResponse{status, "application/json", {},
                      json{{"error", message}}.dump(), nullptr};
}

std::string request_uuid() {
  thread_local std::mt19937_64 rng{std::random_device{}()};
  return fmt::format("{:016x}{:016x}", rng(), rng());
}

/// Streams annotated responses as SSE (Rust's EventConverter + [DONE]);
/// stops the request context if the client disconnects. Marks the guard ok
/// when the stream drains fully.
template <typename Resp>
void stream_sse(pipeline::ManyOut<pipeline::Annotated<Resp>>& out,
                std::shared_ptr<Metrics::InflightGuard> guard, const StreamSink& sink) {
  bool completed = coro::sync_wait([&]() -> coro::Task<bool> {
    while (auto item = co_await out.next()) {
      SseMessage message;
      if (item->data.has_value()) message.data = json(*item->data).dump();
      message.event = item->event;
      message.id = item->id;
      message.comments = item->comment;
      if (!sink(encode_sse(message))) {
        spdlog::debug("SSE client disconnected; stopping generation");
        out.context()->stop_generating();
        co_return false;
      }
    }
    co_return true;
  }());
  if (completed) {
    sink(encode_sse_done());
    guard->mark_ok();
  }
}

/// Shared route body for chat + legacy completions: parse, force internal
/// streaming, dispatch by model, then SSE-stream or fold to a unary reply.
template <typename Request, typename StreamResp, typename Aggregator>
HttpResponse run_openai_route(const HttpRequest& http_request, const char* endpoint_label,
                              Metrics& metrics,
                              pipeline::EnginePtr<Request, pipeline::Annotated<StreamResp>> engine,
                              Request request) {
  (void)http_request;
  bool streaming = request.stream.value_or(false);
  // Engines always stream internally; unary folds below. Unlike Rust v0.1.0
  // we keep nvext (it carries the annotation triggers).
  request.stream = true;

  std::string model = request.model;
  auto guard =
      std::make_shared<Metrics::InflightGuard>(metrics, model, endpoint_label, streaming);

  pipeline::SingleIn<Request> context_request(std::move(request), request_uuid());

  pipeline::ManyOut<pipeline::Annotated<StreamResp>> out;
  try {
    out = coro::sync_wait(engine->generate(std::move(context_request)));
  } catch (const std::exception& e) {
    spdlog::error("generate failed for model {}: {}", model, e.what());
    return json_error(500, "Failed to generate completions");
  }

  if (streaming) {
    HttpResponse response;
    response.content_type = "text/event-stream";
    response.headers["Cache-Control"] = "no-cache";
    response.stream = [out = std::make_shared<pipeline::ManyOut<pipeline::Annotated<StreamResp>>>(
                           std::move(out)),
                       guard](const StreamSink& sink) mutable {
      stream_sse(*out, std::move(guard), sink);
    };
    return response;
  }

  try {
    Aggregator aggregator;
    coro::sync_wait([&]() -> coro::Task<void> {
      while (auto item = co_await out.next()) aggregator.push(*item);
    }());
    auto response_body = std::move(aggregator).finalize();
    guard->mark_ok();
    return HttpResponse{200, "application/json", {}, json(response_body).dump(), nullptr};
  } catch (const std::exception& e) {
    spdlog::error("failed to fold response stream for model {}: {}", model, e.what());
    return json_error(500, std::string("Failed to fold response stream: ") + e.what());
  }
}

template <typename Request>
std::optional<Request> parse_request(const HttpRequest& http_request, HttpResponse& error) {
  try {
    return json::parse(http_request.body).get<Request>();
  } catch (const std::exception& e) {
    error = json_error(400, std::string("invalid request: ") + e.what());
    return std::nullopt;
  }
}

}  // namespace

HttpService::HttpService(HttpServiceOptions options) : options_(std::move(options)) {
  if (options_.enable_chat_endpoints) {
    server_.handle("POST", "/v1/chat/completions",
                   [this](const HttpRequest& request) { return handle_chat(request); });
  }
  if (options_.enable_completion_endpoints) {
    server_.handle("POST", "/v1/completions",
                   [this](const HttpRequest& request) { return handle_completions(request); });
  }
  server_.handle("GET", "/v1/models",
                 [this](const HttpRequest& request) { return handle_list_models(request); });
  server_.handle("GET", "/metrics", [this](const HttpRequest&) {
    return HttpResponse{200, "text/plain; version=0.0.4", {}, metrics_->render(), nullptr};
  });
  server_.handle("GET", "/health", [](const HttpRequest&) {
    return HttpResponse{200, "application/json", {}, R"({"status":"ok"})", nullptr};
  });
}

HttpService::~HttpService() { stop(); }

void HttpService::start() { server_.start(options_.host, options_.port); }

void HttpService::stop() { server_.stop(); }

HttpResponse HttpService::handle_chat(const HttpRequest& http_request) {
  HttpResponse error;
  auto request = parse_request<openai::NvCreateChatCompletionRequest>(http_request, error);
  if (!request) return error;

  auto engine = manager_->get_chat_engine(request->model);
  if (!engine) return json_error(404, "Model not found");

  return run_openai_route<openai::NvCreateChatCompletionRequest,
                          openai::NvCreateChatCompletionStreamResponse,
                          openai::ChatDeltaAggregator>(http_request, "chat_completions",
                                                       *metrics_, std::move(engine),
                                                       std::move(*request));
}

HttpResponse HttpService::handle_completions(const HttpRequest& http_request) {
  HttpResponse error;
  auto request = parse_request<openai::NvCreateCompletionRequest>(http_request, error);
  if (!request) return error;

  auto engine = manager_->get_completion_engine(request->model);
  if (!engine) return json_error(404, "Model not found");

  return run_openai_route<openai::NvCreateCompletionRequest, openai::NvCreateCompletionResponse,
                          openai::CompletionDeltaAggregator>(http_request, "completions",
                                                             *metrics_, std::move(engine),
                                                             std::move(*request));
}

HttpResponse HttpService::handle_list_models(const HttpRequest&) {
  uint64_t created = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
                                               std::chrono::system_clock::now().time_since_epoch())
                                               .count());
  std::set<std::string> names;
  for (auto& name : manager_->chat_models()) names.insert(name);
  for (auto& name : manager_->completion_models()) names.insert(name);

  json data = json::array();
  for (const auto& name : names) {
    // OpenAI model listing; object is "model" per the API (Rust emits the
    // literal "object" here, which looks unintended).
    data.push_back({{"id", name},
                    {"object", "model"},
                    {"created", created},
                    {"owned_by", "nvidia"}});
  }
  return HttpResponse{200, "application/json", {},
                      json{{"object", "list"}, {"data", data}}.dump(), nullptr};
}

}  // namespace dynamo::llm::http
