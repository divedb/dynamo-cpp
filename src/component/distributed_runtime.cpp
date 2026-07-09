// SPDX-License-Identifier: Apache-2.0

#include "component/distributed_runtime.h"

#include <mutex>

#include <spdlog/spdlog.h>

#include "discovery/in_process.h"
#include "discovery/tcp.h"
#include "runtime/config.h"

namespace dynamo::component {

using nlohmann::json;

// ---------------------------------------------------------------------------
// InstanceSource
// ---------------------------------------------------------------------------

struct InstanceSource::State {
  std::optional<Runtime> runtime;
  mutable std::mutex mutex;
  std::map<std::string, EndpointInfo> instances;  // key → info
  bool closed = false;
  // Replaced on every change; waiters grab the current event and await it.
  std::shared_ptr<coro::AsyncEvent> changed = std::make_shared<coro::AsyncEvent>();

  // Swaps in a fresh event under the lock; the caller must fire the returned
  // one AFTER releasing the lock (waiters resume inline and may re-lock).
  [[nodiscard]] std::shared_ptr<coro::AsyncEvent> bump_locked() {
    return std::exchange(changed, std::make_shared<coro::AsyncEvent>());
  }
};

InstanceSource::~InstanceSource() {
  // Last holder gone: end the underlying discovery watch (the registry only
  // holds this object weakly, mirroring Dynamo's watcher-exits-when-all-
  // receivers-drop behavior).
  if (close_watch_) close_watch_();
}

std::vector<EndpointInfo> InstanceSource::snapshot() const {
  std::vector<EndpointInfo> out;
  std::lock_guard lock(state_->mutex);
  out.reserve(state_->instances.size());
  for (auto& [key, info] : state_->instances) out.push_back(info);
  return out;
}

bool InstanceSource::closed() const {
  std::lock_guard lock(state_->mutex);
  return state_->closed;
}

coro::Task<void> InstanceSource::wait_changed() {
  std::shared_ptr<coro::AsyncEvent> event;
  {
    std::lock_guard lock(state_->mutex);
    if (state_->closed) co_return;
    event = state_->changed;
  }
  co_await event->wait();
  // Hop off the notifying thread (the watch loop) so it can keep folding
  // events while our continuation runs.
  co_await state_->runtime->primary().schedule();
}

coro::Task<void> InstanceSource::wait_nonempty() {
  for (;;) {
    std::shared_ptr<coro::AsyncEvent> event;
    {
      std::lock_guard lock(state_->mutex);
      if (!state_->instances.empty()) co_return;
      if (state_->closed) {
        throw std::runtime_error("instance watch closed (discovery lost or shutting down)");
      }
      event = state_->changed;
    }
    co_await event->wait();
    co_await state_->runtime->primary().schedule();  // see wait_changed
  }
}

namespace {

coro::Task<void> instance_watch_loop(std::shared_ptr<InstanceSource::State> state,
                                     coro::Receiver<discovery::WatchEvent> events,
                                     std::string prefix) {
  while (auto event = co_await events.recv()) {
    spdlog::debug("instance watch {}: {} {}", prefix,
                  event->kind == discovery::WatchEvent::Kind::Put ? "put" : "delete",
                  event->kv.key);
    std::shared_ptr<coro::AsyncEvent> fire;
    {
      std::lock_guard lock(state->mutex);
      if (event->kind == discovery::WatchEvent::Kind::Put) {
        try {
          state->instances[event->kv.key] = json::parse(event->kv.value).get<EndpointInfo>();
        } catch (const std::exception& e) {
          spdlog::warn("instance watch {}: bad endpoint info for {}: {}", prefix, event->kv.key,
                       e.what());
          continue;
        }
      } else {
        state->instances.erase(event->kv.key);
      }
      fire = state->bump_locked();
    }
    fire->set();
  }
  spdlog::debug("instance watch for {} closed", prefix);
  std::shared_ptr<coro::AsyncEvent> fire;
  {
    std::lock_guard lock(state->mutex);
    state->closed = true;
    state->instances.clear();
    fire = state->bump_locked();
  }
  fire->set();
}

}  // namespace

// ---------------------------------------------------------------------------
// DistributedRuntime
// ---------------------------------------------------------------------------

struct DistributedRuntime::State {
  Runtime runtime;
  Options options;
  discovery::DiscoveryPtr discovery;

  std::mutex mutex;
  std::shared_ptr<transports::ControlPlaneServer> control_plane;
  std::shared_ptr<transports::DataPlaneServer> data_plane;
  // Weak: sources die (and their watches close) when the last client drops.
  std::map<std::string, std::weak_ptr<InstanceSource>> sources;

  State(Runtime rt, Options opts) : runtime(std::move(rt)), options(std::move(opts)) {}

  ~State() {
    if (control_plane) control_plane->stop();
    if (data_plane) data_plane->stop();
    if (discovery) discovery->shutdown();
  }
};

DistributedRuntime::Options DistributedRuntime::Options::from_env() {
  Options options;
  options.discovery_address = env_or("DYN_DISCOVERY", "");
  options.host = env_or("DYN_HOST", "127.0.0.1");
  return options;
}

DistributedRuntime DistributedRuntime::create(Runtime runtime, Options options) {
  auto state = std::make_shared<State>(runtime, options);
  if (options.discovery_address.empty()) {
    state->discovery = std::make_shared<discovery::InProcessDiscovery>(runtime);
    spdlog::debug("distributed runtime {} using in-process discovery", runtime.id());
  } else {
    state->discovery = discovery::TcpDiscovery::connect(runtime, options.discovery_address);
    spdlog::debug("distributed runtime {} using discoveryd at {}", runtime.id(),
                  options.discovery_address);
  }
  return DistributedRuntime(std::move(state));
}

Runtime& DistributedRuntime::runtime() const { return state_->runtime; }
discovery::DiscoveryPtr DistributedRuntime::discovery() const { return state_->discovery; }

coro::Task<discovery::Lease> DistributedRuntime::primary_lease() const {
  // Eager forward (not a coroutine): safe on temporaries.
  return state_->discovery->primary_lease();
}

std::shared_ptr<transports::ControlPlaneServer> DistributedRuntime::control_plane() const {
  std::lock_guard lock(state_->mutex);
  if (!state_->control_plane) {
    state_->control_plane =
        transports::ControlPlaneServer::start(state_->runtime, state_->options.host);
  }
  return state_->control_plane;
}

std::shared_ptr<transports::DataPlaneServer> DistributedRuntime::data_plane() const {
  std::lock_guard lock(state_->mutex);
  if (!state_->data_plane) {
    state_->data_plane = transports::DataPlaneServer::start(state_->runtime, state_->options.host);
  }
  return state_->data_plane;
}

Namespace DistributedRuntime::ns(std::string name) const {
  return Namespace(*this, std::move(name));
}

coro::Task<std::shared_ptr<InstanceSource>> DistributedRuntime::instance_source(
    Endpoint endpoint) const {
  // Eager copy into the impl frame: safe on temporaries (see serve()).
  return instance_source_impl(*this, std::move(endpoint));
}

coro::Task<std::shared_ptr<InstanceSource>> DistributedRuntime::instance_source_impl(
    DistributedRuntime self, Endpoint endpoint) {
  auto& state_ = self.state_;
  std::string prefix = endpoint.key_prefix();
  {
    std::lock_guard lock(state_->mutex);
    if (auto it = state_->sources.find(prefix); it != state_->sources.end()) {
      if (auto existing = it->second.lock()) co_return existing;
      state_->sources.erase(it);  // expired: last client dropped it
    }
  }

  auto watch = co_await state_->discovery->kv_get_and_watch_prefix(prefix);

  auto source = std::shared_ptr<InstanceSource>(new InstanceSource());
  source->state_ = std::make_shared<InstanceSource::State>();
  source->state_->runtime = state_->runtime;

  {
    std::lock_guard lock(state_->mutex);
    // Lost the race? Keep the existing one; our watch closes with `receiver`.
    if (auto it = state_->sources.find(prefix); it != state_->sources.end()) {
      if (auto existing = it->second.lock()) co_return existing;
      state_->sources.erase(it);
    }
    state_->sources.emplace(prefix, source);
  }

  // The watch ends when the last holder drops the source, or on runtime
  // cancellation — whichever comes first.
  auto receiver = std::move(watch.events);
  source->close_watch_ = [rx = receiver]() mutable { rx.close(); };
  auto registration = state_->runtime.token().register_callback(
      [rx = receiver]() mutable { rx.close(); });

  auto keep_registration =
      std::make_shared<CancellationRegistration>(std::move(registration));
  state_->runtime.spawn_background(
      [](std::shared_ptr<InstanceSource::State> st, coro::Receiver<discovery::WatchEvent> rx,
         std::string pfx, std::shared_ptr<CancellationRegistration> reg) -> coro::Task<void> {
        co_await instance_watch_loop(std::move(st), std::move(rx), std::move(pfx));
        reg.reset();
      }(source->state_, std::move(receiver), prefix, std::move(keep_registration)));

  co_return source;
}

// ---------------------------------------------------------------------------
// Namespace / Component / Endpoint
// ---------------------------------------------------------------------------

Namespace::Namespace(DistributedRuntime drt, std::string name)
    : drt_(std::move(drt)), name_(std::move(name)) {
  validate_name(name_, "namespace");
}

Component Namespace::component(std::string name) const { return Component(*this, std::move(name)); }

std::string Namespace::event_subject(const std::string& event_name) const {
  return name_ + ".events." + event_name;
}

namespace {

// Free coroutines with by-value params: the fluent API (drt.ns("a").publish(...))
// runs these on temporaries, so nothing may capture `this` lazily.
coro::Task<void> publish_impl(discovery::DiscoveryPtr disco, std::string subject,
                              std::string payload) {
  co_await disco->publish(std::move(subject), std::move(payload));
}

coro::Task<discovery::EventStream> subscribe_impl(discovery::DiscoveryPtr disco,
                                                  std::string subject) {
  co_return co_await disco->subscribe(std::move(subject));
}

}  // namespace

coro::Task<void> Namespace::publish(const std::string& event_name,
                                    const nlohmann::json& value) const {
  return publish_impl(drt_.discovery(), event_subject(event_name), value.dump());
}

coro::Task<discovery::EventStream> Namespace::subscribe(const std::string& event_name) const {
  return subscribe_impl(drt_.discovery(), event_subject(event_name));
}

Component::Component(Namespace ns, std::string name)
    : namespace_(std::move(ns)), name_(std::move(name)) {
  validate_name(name_, "component");
}

std::string Component::key_prefix() const {
  return namespace_.name() + "/components/" + name_;
}

std::string Component::path() const { return namespace_.name() + "." + name_; }

Endpoint Component::endpoint(std::string name) const { return Endpoint(*this, std::move(name)); }

std::string Component::event_subject(const std::string& event_name) const {
  return namespace_.name() + "." + name_ + ".events." + event_name;
}

coro::Task<void> Component::publish(const std::string& event_name,
                                    const nlohmann::json& value) const {
  return publish_impl(drt().discovery(), event_subject(event_name), value.dump());
}

coro::Task<discovery::EventStream> Component::subscribe(const std::string& event_name) const {
  return subscribe_impl(drt().discovery(), event_subject(event_name));
}

namespace {

coro::Task<void> scrape_one(EndpointInfo info, std::chrono::milliseconds timeout,
                            coro::Sender<EndpointStats> out) {
  EndpointStats result;
  result.info = info;
  try {
    result.stats = json::parse(
        transports::dispatch_query(info.address, info.subject + "/stats", "{}", timeout));
    result.ok = true;
  } catch (const std::exception& e) {
    result.error = e.what();
  }
  out.send(std::move(result));
  co_return;
}

}  // namespace

namespace {

coro::Task<std::vector<EndpointInfo>> list_instances_impl(Component self) {
  auto kvs = co_await self.drt().discovery()->kv_get_prefix(self.key_prefix() + "/");
  std::vector<EndpointInfo> instances;
  for (auto& kv : kvs) {
    try {
      instances.push_back(json::parse(kv.value).get<EndpointInfo>());
    } catch (const std::exception& e) {
      spdlog::warn("{}: skipping malformed instance info at {}: {}", self.path(), kv.key,
                   e.what());
    }
  }
  co_return instances;
}

}  // namespace

coro::Task<std::vector<EndpointInfo>> Component::list_instances() const {
  return list_instances_impl(*this);
}

namespace {

coro::Task<ServiceSet> scrape_stats_impl(Component self, std::chrono::milliseconds timeout) {
  auto instances = co_await self.list_instances();

  // Fan the queries out concurrently; each is bounded by `timeout`, so the
  // whole scrape is too. The channel closes once every query task finished.
  auto [tx, rx] = coro::make_channel<EndpointStats>(
      instances.size() + 1, resume_on(self.drt().runtime().primary()));
  for (auto& info : instances) {
    self.drt().runtime().spawn(scrape_one(info, timeout, tx));
  }
  tx = coro::Sender<EndpointStats>();  // drop our sender: tasks hold the rest

  ServiceSet set;
  while (auto result = co_await rx.recv()) {
    set.endpoints.push_back(std::move(*result));
  }
  co_return set;
}

}  // namespace

coro::Task<ServiceSet> Component::scrape_stats(std::chrono::milliseconds timeout) const {
  return scrape_stats_impl(*this, timeout);
}

Endpoint::Endpoint(Component component, std::string name)
    : component_(std::move(component)), name_(std::move(name)) {
  validate_name(name_, "endpoint");
}

std::string Endpoint::key_prefix() const { return component_.key_prefix() + "/" + name_; }

std::string Endpoint::instance_key(int64_t instance_id) const {
  return key_prefix() + ":" + hex_id(instance_id);
}

std::string Endpoint::subject_for(int64_t instance_id) const {
  return component_.ns().name() + "/" + component_.name() + "/" + name_ + ":" +
         hex_id(instance_id);
}

}  // namespace dynamo::component
