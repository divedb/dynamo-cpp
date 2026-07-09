// SPDX-License-Identifier: Apache-2.0
//
// DistributedRuntime: local Runtime + discovery + control/data plane servers
// (lazily started) + a registry sharing one instance watcher per endpoint
// across all clients in the process. Namespace/Component/Endpoint are cheap
// value handles over it.
//
// Include component/component.h (umbrella) to get the templated serve/client
// APIs; this header keeps the non-template core.

#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "discovery/discovery.h"
#include "pipeline/engine.h"
#include "runtime/runtime.h"
#include "transports/control_plane.h"
#include "transports/data_plane.h"

#include "component/info.h"

namespace dynamo::component {

class Namespace;
class Component;
class Endpoint;
class InstanceSource;

template <typename Req, typename Resp>
class Client;

/// Options for serving an endpoint instance.
struct ServeOptions {
  /// Liveness lease; defaults to the runtime's primary lease.
  std::optional<discovery::Lease> lease;
  /// Custom stats handler merged into the built-in counters (as "custom").
  transports::QueryHandler stats_handler;
  /// Advertised in the instance's discovery record.
  std::string description;
  std::string version = "0.1.0";
  /// Also joins the endpoint's shared work queue: requests sent with
  /// Client::queue() are broker-balanced across all instances serving with
  /// this flag (NATS queue-group semantics). Instance-addressed dispatch
  /// keeps working alongside.
  bool queue_group = false;
};

/// Per-client knobs for the request path (Dynamo's StreamOptions role).
struct CallOptions {
  /// Response frames buffered client-side before backpressuring the worker.
  size_t recv_buffer_count = 64;
  /// Deadline for the control-plane ack.
  std::chrono::milliseconds dispatch_timeout{5000};
  /// Deadline for the worker's call-home after a successful dispatch.
  std::chrono::milliseconds arrival_timeout{30000};
};

class DistributedRuntime {
 public:
  struct Options {
    /// discoveryd address ("host:port"); empty selects the in-process store.
    std::string discovery_address;
    /// Bind host for this process's control/data plane listeners.
    std::string host = "127.0.0.1";

    /// Reads DYN_DISCOVERY / DYN_HOST.
    static Options from_env();
  };

  static DistributedRuntime create(Runtime runtime, Options options = Options::from_env());

  Runtime& runtime() const;
  discovery::DiscoveryPtr discovery() const;
  coro::Task<discovery::Lease> primary_lease() const;

  std::shared_ptr<transports::ControlPlaneServer> control_plane() const;  // lazy
  std::shared_ptr<transports::DataPlaneServer> data_plane() const;        // lazy

  Namespace ns(std::string name) const;

  void shutdown() const { runtime().shutdown(); }

  /// One shared watcher per endpoint path (Dynamo's component registry role).
  coro::Task<std::shared_ptr<InstanceSource>> instance_source(Endpoint endpoint) const;

  bool operator==(const DistributedRuntime& other) const { return state_ == other.state_; }

  struct State;  // opaque; public so implementation helpers can name it

 private:
  explicit DistributedRuntime(std::shared_ptr<State> s) : state_(std::move(s)) {}
  static coro::Task<std::shared_ptr<InstanceSource>> instance_source_impl(DistributedRuntime self,
                                                                          Endpoint endpoint);
  std::shared_ptr<State> state_;
};

class Namespace {
 public:
  Namespace(DistributedRuntime drt, std::string name);

  const std::string& name() const { return name_; }
  Component component(std::string name) const;
  const DistributedRuntime& drt() const { return drt_; }

  /// Namespace-scoped transient events (Dynamo's EventPublisher/Subscriber).
  /// Subject: "{ns}.events.{event_name}".
  std::string event_subject(const std::string& event_name) const;
  coro::Task<void> publish(const std::string& event_name, const nlohmann::json& value) const;
  coro::Task<discovery::EventStream> subscribe(const std::string& event_name) const;

 private:
  DistributedRuntime drt_;
  std::string name_;
};

class Component {
 public:
  Component(Namespace ns, std::string name);

  const std::string& name() const { return name_; }
  const Namespace& ns() const { return namespace_; }
  const DistributedRuntime& drt() const { return namespace_.drt(); }

  /// Discovery key prefix: {ns}/components/{name}
  std::string key_prefix() const;
  /// Display path: {ns}.{name}
  std::string path() const;

  Endpoint endpoint(std::string name) const;

  /// Lists all live instances of this component, across every endpoint
  /// (Dynamo's list_endpoints intent, unimplemented there).
  coro::Task<std::vector<EndpointInfo>> list_instances() const;

  /// Scrapes the stats endpoint of every live instance of this component
  /// (all endpoints), concurrently, each bounded by `timeout`. Unreachable
  /// instances are reported per-entry (`ok == false`), never thrown.
  coro::Task<ServiceSet> scrape_stats(std::chrono::milliseconds timeout) const;

  /// Component-scoped transient events. Subject: "{ns}.{comp}.events.{name}".
  std::string event_subject(const std::string& event_name) const;
  coro::Task<void> publish(const std::string& event_name, const nlohmann::json& value) const;
  coro::Task<discovery::EventStream> subscribe(const std::string& event_name) const;

 private:
  Namespace namespace_;
  std::string name_;
};

class Endpoint {
 public:
  Endpoint(Component component, std::string name);

  const std::string& name() const { return name_; }
  const Component& component() const { return component_; }
  const DistributedRuntime& drt() const { return component_.drt(); }

  /// Discovery key prefix for all instances of this endpoint.
  std::string key_prefix() const;
  /// Discovery key for one instance.
  std::string instance_key(int64_t instance_id) const;
  /// Instance-addressed subject.
  std::string subject_for(int64_t instance_id) const;
  /// Instance-agnostic shared work-queue subject (queue groups).
  std::string queue_subject() const;

  /// Serves `engine` on this endpoint until the lease (default: primary) or
  /// runtime is cancelled. Registers the instance in discovery; on
  /// registration failure the endpoint is torn down and the error rethrown
  /// (observable through the TaskHandle when spawned).
  /// Defined in component/serve.h (via the umbrella header).
  template <typename Req, typename Resp>
  coro::Task<void> serve(pipeline::EnginePtr<Req, Resp> engine, ServeOptions options = {}) const;

  /// Creates a typed client for this endpoint.
  /// Defined in component/client.h (via the umbrella header).
  template <typename Req, typename Resp>
  coro::Task<Client<Req, Resp>> client(CallOptions options = {}) const;

 private:
  Component component_;
  std::string name_;
};

/// Live set of endpoint instances, folded from a discovery watch stream.
/// The underlying watch is closed when the last holder drops this object
/// (the DistributedRuntime registry only holds it weakly).
class InstanceSource {
 public:
  ~InstanceSource();

  std::vector<EndpointInfo> snapshot() const;
  bool closed() const;

  /// Resolves once at least one instance is live. Throws std::runtime_error
  /// if the watch closed (discovery lost / runtime shut down).
  coro::Task<void> wait_nonempty();

  /// Resolves on the next membership change (or closure).
  coro::Task<void> wait_changed();

  struct State;  // opaque; public so implementation helpers can name it

 private:
  friend class DistributedRuntime;
  InstanceSource() = default;
  std::function<void()> close_watch_;
  std::shared_ptr<State> state_;
};

}  // namespace dynamo::component
