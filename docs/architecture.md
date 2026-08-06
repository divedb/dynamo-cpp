# dynamo-cpp: Architecture

A C++20 reimplementation of the runtime layer of [NVIDIA Dynamo]
(`third_party/dynamo/lib/runtime`), preserving its distributed-runtime
behaviors while redesigning the APIs idiomatically for C++.

---

## 1. Dynamo Architecture Summary

Dynamo's runtime crate is the foundation of a distributed inference framework.
It is organized around a small number of cooperating layers:

### Local runtime (`runtime.rs`, `worker.rs`, `config.rs`)

- `Runtime` is a process-local handle bundling **two tokio thread pools** — a
  *primary* pool for application work and a *secondary* (usually
  single-threaded) pool for background chores (etcd keep-alives, watchers) —
  plus a process-unique worker id (UUID) and the **root `CancellationToken`**.
  `Runtime::shutdown()` simply cancels the root token; everything else in the
  process is expected to hang off child tokens.
- `Worker` is the bootstrap wrapper: it builds the runtime from environment
  settings, installs a SIGINT/SIGTERM handler that cancels the root token, and
  runs the user's async `app(Runtime)` to completion. After cancellation it
  grants a **graceful-shutdown window** (`DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT`,
  default 5 s debug / 30 s release) and hard-exits with code 911 if the app
  doesn't finish in time.

### Distributed runtime (`distributed.rs`)

`DistributedRuntime` = `Runtime` + cluster connectivity:

- an **etcd client** — discovery, leases, liveness (control-plane metadata);
- a **NATS client** — the request/control plane transport (addressed message
  dispatch to service subjects);
- a lazily created **TCP stream server** — the response/data plane (streams
  are "called home" directly over TCP, bypassing the broker);
- a **component registry** — process-local sharing so that, e.g., two clients
  of the same endpoint share one etcd watcher.

### Component / namespace / endpoint model (`component.rs`)

Hierarchy: `Namespace` → `Component` → `Endpoint`.

- Identity is path-like: component etcd path `{ns}/components/{comp}`,
  endpoint path `{...}/{endpoint}`, and a *per-instance* key
  `{endpoint_path}:{lease_id:x}`. The NATS subject for an instance is
  `{ns}|{comp}.{endpoint}-{lease_id:x}` (slugged).
- **A running endpoint instance is identified by its etcd lease id** — the
  lease id doubles as the instance id used for routing.
- Serving is a two-step builder flow: `component.service_builder().create()`
  registers a NATS service (with a stats handler registry); then
  `endpoint.endpoint_builder().handler(h).start()`:
  1. picks a lease (defaults to the client's *primary lease*),
  2. subscribes a `PushEndpoint` work loop on the instance subject,
  3. writes `ComponentEndpointInfo` JSON (`{namespace, component, endpoint,
     lease_id, transport}`) to etcd **bound to the lease** — atomically via a
     create-if-absent transaction,
  4. blocks until the lease/runtime is cancelled (that call *is* "serving").
  If etcd registration fails the endpoint is torn down.

### Discovery and lease-based liveness (`transports/etcd.rs`, `discovery.rs`)

- Connecting creates a **primary lease (TTL 10 s)** whose keep-alive loop runs
  on the secondary pool at TTL/2 cadence with a hard deadline: if heartbeats
  can't be issued within TTL, the lease is considered lost and the **runtime's
  root token is cancelled** (lease death ⇒ process shutdown). Conversely,
  cancelling the token revokes the lease (shutdown ⇒ prompt deregistration).
- Registration keys are written under a lease, so **an instance's keys vanish
  automatically when its process dies** — that's the liveness model.
- `kv_get_and_watch_prefix` returns the current keys as synthetic `Put` events
  followed by a live watch stream (`Put`/`Delete`) from the next revision —
  clients never miss or double-see an instance.

### Client, endpoint watching, and routing (`component/client.rs`)

`Client<Req, Resp>`:

- spawns (on the secondary pool) a **prefix watcher** for the endpoint's etcd
  path, folds Put/Delete events into a `key → lease_id` map, and publishes the
  current instance-id vector through a `watch` channel;
- `wait_for_endpoints()` awaits a non-empty instance vector;
- routing policies pick an instance id then dispatch to its subject:
  - `round_robin` — atomic counter modulo instance count,
  - `random` — random index,
  - `direct(lease_id)` — validated against the live set;
  - the generic `generate()` defaults to `random`.

### Control plane vs data plane (`pipeline/network/*`)

A request is **pushed over NATS; the response streams back over a direct TCP
connection** established *from the worker to the requester* ("call home"):

1. Requester registers a pending **response stream** with its local
   `TcpStreamServer` under a fresh subject (UUID) and gets `ConnectionInfo`
   (address, subject, context id).
2. Requester encodes a **two-part message** — header =
   `RequestControlMessage{id, request_type, response_type, connection_info}`,
   body = serialized request — and publishes it to the instance's NATS
   subject. (Codec framing: `u64 header_len | u64 body_len | u64 xxh3
   checksum | header | body`.)
3. Worker's `PushEndpoint` loop acks the NATS message and spawns
   `handle_payload`: decode both parts, rebuild `Context<T>` **with the
   propagated request id**, TCP-connect back to the requester
   (`CallHomeHandshake{subject, stream_type}`), then invoke the local
   engine's `generate()`.
4. Worker sends a **prologue** frame first — `{error: None}` on success or the
   generate error — the requester's `generate()` call doesn't return until the
   prologue arrives (errors fail the call, not the stream).
5. Response items flow as data frames (JSON), ending with a **Sentinel**
   control frame; the server then closes the socket (client awaits FIN).
6. **Cancellation flows the other way on the same socket**: if the requester
   drops the stream or calls stop/kill, the TCP server sends `Stop`/`Kill`
   control frames, which the worker maps onto the request context
   (`stop_generating()` / `kill()`), which the user handler observes.

### Execution model (`engine.rs`, `pipeline/context.rs`)

- `AsyncEngine<Req, Resp, E>` — the single generate-style interface:
  `async generate(Req) -> Result<Resp, E>`. The canonical shape is
  `SingleIn<T> = Context<T>` → `ManyOut<U> = ResponseStream<U>`.
- `Context<T>` wraps the request payload with an id, a **controller**
  (stop/kill state + wakers) and a typed key/value registry; `transfer()`
  moves the context onto a new payload type as it flows through pipeline
  stages.
- `AsyncEngineContext` is the control surface carried by every response
  stream: `id()`, `is_stopped()/is_killed()`, awaitable `stopped()/killed()`,
  `stop_generating()`, `kill()`.
- `Annotated<R>` (`protocols/annotated.rs`) is the SSE-like response envelope
  (`data`, `id`, `event`, `comment`) used to carry either data or in-band
  errors/events.
- Observability: NATS services expose per-endpoint **stats handlers**
  (`scrape_stats`); metrics/instrumentation elsewhere are TODO-grade.

---

## 2. Behavioral Requirements to Preserve

1. **Process-local runtime**: primary + background executors, process id,
   root cancellation token; `shutdown()` = cancel root.
2. **Worker bootstrap**: run an async app to completion; SIGINT/SIGTERM →
   graceful cancel; bounded graceful-shutdown window, then hard exit (911).
3. **Distributed runtime**: local runtime + discovery client + control-plane
   transport + lazily-started data-plane stream server + shared registry.
4. **Namespace/Component/Endpoint** naming with path-like identities and
   per-instance keys derived from a lease/instance id.
5. **Lease-based liveness**: registrations bound to leases with TTL +
   keep-alive; lease loss cancels the owner's runtime; runtime shutdown
   revokes leases; instance keys disappear with their lease.
6. **Registration/deregistration** is atomic create-if-absent; serving blocks
   until cancelled; failure to register tears the endpoint down.
7. **Client-side watching**: get-then-watch prefix semantics (no gaps, no
   dupes), folded into a live instance set observable through a watch channel;
   `wait_for_endpoints()`.
8. **Routing**: random (default), round-robin (atomic counter), direct
   (validated instance id) over the live instance set.
9. **Unary request → streaming response** with explicit **separation of
   control plane (request dispatch) and data plane (call-home TCP stream)**.
10. **Prologue contract**: the client's `generate()` resolves only after the
    worker reports success/failure; failures surface as call errors.
11. **Request-context propagation**: request id travels to the worker;
    the worker-side context is remotely controllable.
12. **Cancellation propagation**: client stop/kill (or stream abandonment)
    reaches the worker's handler context via control frames; graceful stream
    termination via sentinel.
13. **Streaming envelope** (`Annotated`) supporting data, events, and in-band
    errors.
14. **Observability hooks**: per-endpoint stats handler registration.

---

## 3. Proposed C++20 Architecture

### Dependency and platform decisions

| Dynamo (Rust) | dynamo-cpp | Rationale |
|---|---|---|
| tokio | in-tree C++20 coroutine library (`Task`, `AsyncGenerator`, channels, events) + thread-pool executors | folly/asio integration cost is high; the surface we need is small and self-contained (decision carried over from the earlier M1 milestone). |
| etcd | `Discovery` interface with two backends: **in-process** (single process, tests) and **dynamo-discoveryd** (a small TCP discovery server implementing leases + prefix watch) | An etcd C++ client would drag in grpc/protobuf; the *behavioral contract* (lease TTL, keep-alive, get+watch) is what matters and is reproduced faithfully. |
| NATS request plane | **direct TCP dispatch**: every worker hosts a control-plane listener; its address is part of the registered instance info; clients dispatch requests straight to the chosen instance | Without a broker dependency, instance-addressed push (`subject_to(lease_id)`) maps naturally onto "connect to the instance's advertised address, name the subject in the frame". Assumption stated in §Assumptions. |
| TCP call-home data plane | same design: data-plane server on the client side, worker connects back | Preserved as-is — this is the load-bearing streaming behavior. |
| serde_json | nlohmann/json | wire payloads and instance info stay JSON. |
| xxh3-checksummed two-part codec | same framing (`u64 header_len, u64 body_len, u64 checksum` + parts), FNV-1a 64 checksum | wire compatibility with Rust is a non-goal; framing semantics are kept. |

Socket I/O uses **blocking POSIX sockets serviced by dedicated threads**,
bridged into the coroutine world with channels. This trades maximum
scalability for simplicity and robustness; the transport interfaces hide it,
so an asio/io_uring backend can be swapped in later.

### Modules

```
src/runtime/    coro/ (Task, AsyncGenerator, sync_wait, Event, Channel, oneshot)
                cancellation, executor (ThreadPool), Runtime, Worker, logging, config
src/pipeline/   engine (AsyncEngineContext, Context<T>, ResponseStream), annotated
src/discovery/  Discovery interface, Lease, WatchEvent/WatchStream,
                InProcessDiscovery, TcpDiscovery client + discoveryd server
src/transports/ two-part codec, socket/framing primitives,
                control-plane server/client, data-plane server/client
src/component/  DistributedRuntime, Namespace/Component/Endpoint,
                ServiceBuilder/EndpointBuilder, Client (watch + routing)
src/examples/   hello_world server/client, multi-instance routing demo, discoveryd
tests/          Catch2 suites per module
```

Ownership: `Runtime` and `DistributedRuntime` are handle types over
shared-ptr state (cheap to copy, like the Rust `Clone` types). Transports and
discovery are interfaces (`std::shared_ptr<Discovery>` etc.) so tests can
substitute in-process fakes. Endpoint serving, watchers, and keep-alive loops
are coroutines running on the runtime's executors, all rooted in the
hierarchical `CancellationToken` tree.

### Assumptions (where Dynamo is ambiguous or broker-specific)

1. **NATS replacement.** Dynamo relies on NATS for instance-addressed request
   push and acks. We preserve *observable* behavior — a request lands on
   exactly the chosen live instance's endpoint queue, the dispatch call fails
   if the instance is gone — by dispatching directly to the instance's
   advertised control-plane address. Queue-group load balancing (unused by the
   client routing paths) is not reproduced.
2. **NATS service stats.** `scrape_stats` broadcast collection is reduced to
   local, per-endpoint stats handlers (request counts, errors) queryable via a
   control-plane `stats` message to a specific instance.
3. **Request streams** (`enable_request_stream`, ManyIn) are declared but
   unimplemented in Dynamo (`process_request_stream` returns Ok(())); we keep
   the option in the data-plane registration API but implement SingleIn only.
4. **Lease TTL semantics** for the in-process backend: expiry only occurs via
   revocation or missed keep-alives, which is simulated deterministically in
   tests (the TCP backend enforces real TTL deadlines).
5. `PushEndpoint` acks before processing (fire-and-forget dispatch); we keep
   the early-ack: control-plane dispatch succeeds once the worker accepts the
   frame, and all subsequent errors travel via the data-plane prologue.

---

## 4. Deliberate limitations

### discoveryd: single node, in-memory

`dynamo-discoveryd` is a **single-node, in-memory** discovery server by
design. There is no snapshot/restore, no write-ahead log, and no replication
(raft). If the process dies, all leases, keys, watches, and subscriptions are
gone; clients reconnect with backoff and re-register what they own (workers
re-serve, watchers resync via the snapshot + sync-marker protocol), so a
*restarted* discoveryd converges back to a correct view, but anything not
re-asserted by a live client is lost.

This mirrors the deployment posture of the Rust reference, which delegates
durability/HA to etcd. Deployments that need a highly available control plane
should use the etcd backend (`DYN_DISCOVERY=etcd://host:port`, built when
etcd-cpp-apiv3 is available) and run a real etcd cluster; discoveryd is meant
for development, tests, and single-node deployments where "restart it and let
clients re-register" is acceptable.

### Transport security (mTLS + shared-token auth)

The internal planes — discoveryd, the control plane, and the data plane — can
be secured with two independently togglable, cluster-wide settings
(`src/transports/tls.h`, `src/transports/auth.h`):

- **Mutual TLS** (`DYN_TLS_CERT` / `DYN_TLS_KEY` / `DYN_TLS_CA`, all three
  required together; needs an OpenSSL 3 build, `DYNAMO_WITH_TLS`). Every node
  presents the cert and verifies the peer against the pinned CA in *both*
  directions (every node is both client and server — the data plane calls
  home). TLS 1.3 only; hostname verification is deliberately skipped — trust
  is anchored in the private CA, not in names. Implementation note: sockets
  are used full-duplex from two threads (data-plane reader + control-frame
  writers), which OpenSSL's `SSL` objects don't support natively; the session
  therefore runs the fd non-blocking and serializes each `SSL_read`/`SSL_write`
  under a mutex held only for the non-blocking call, waiting in `poll()`
  outside the lock. Handshakes are lazy (driven by the first read/write), so
  a stalling peer never blocks an accept loop.
- **Shared-token auth** (`DYN_AUTH_TOKEN`). When set, every new connection on
  the internal planes starts with one auth frame; servers verify the token
  (constant-time) before serving anything. The token authenticates cluster
  membership only — pair it with TLS on untrusted networks or it travels in
  cleartext.

Deliberately **not** covered: the **HTTP frontend** (standard HTTP to external
OpenAI clients — terminate TLS and authenticate in a proxy, as the Rust
reference does) and the **etcd backend's own connection** (etcd deployments
bring their own TLS/auth story). Without OpenSSL, the build links a stub that
fails loudly at startup if `DYN_TLS_*` is set — TLS never silently degrades
to plaintext.
