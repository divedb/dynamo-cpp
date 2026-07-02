# Dynamo C++20 — Implementation TODO

This document enumerates every gap between the NVIDIA Dynamo Rust implementation and the C++20 reimplementation, organized by module. Each item includes rationale, affected components, dependencies, estimated complexity (XS–XL), and recommended implementation order.

---

## Phase 0: Make Existing Code Compile (Order: 0)

These items block all further work — without functioning headers and a working build, nothing else can be validated.

### 0.1 Create all header files
| Rust File | C++ Status | Gap |
|---|---|---|
| `lib/runtime/src/lib.rs` (86L) | ❌ No `lib.rs` equivalent | No module declaration file, no re-exports, no `DistributedRuntime` type. C++ namespace `dynamo` is declared piecemeal across headers. |
| `runtime.rs` (137L) | ⚠️ `runtime.cpp` exists | **Missing header**: `<dynamo/runtime.h>` is included but not created. Missing `RuntimeType` enum, `child_token()`, `id()` accessor. |
| `config.rs` (236L) | ⚠️ `config.cpp` exists | **Missing header**: `<dynamo/config.h>` is included but not created. Missing `RuntimeConfigBuilder`, `single_threaded()`, `create_runtime()`, `env_is_truthy()`, `is_truthy()`, `jsonl_logging_enabled()`, `disable_ansi_logging()`. |
| `engine.rs` (168L) | ⚠️ `engine.cpp` exists | **Missing header**: `<dynamo/engine.h>` is included but not created. Missing `Data` trait concept, `DataUnary`/`DataStream` aliases, `AsyncEngineContext` trait, `AsyncEngineContextProvider`, `AsyncEngineUnary`, `AsyncEngineStream`, `ResponseStream`. |
| `component.rs` (436L) | ⚠️ `component.cpp` exists | **Missing header**: `<dynamo/component.h>` is included but not created. Missing `TransportType` enum, `ComponentEndpointInfo`, `Registry`, `ComponentBuilder`, `EndpointConfig`, all `list_endpoints()/scrape_stats()/stats_stream()` methods, `validate_allowed_chars()`. |
| `discovery.rs` (86L) | ⚠️ `discovery.cpp` exists | **Missing header**: `<dynamo/discovery.h>`. Missing `primary_lease_id()`, real etcd integration. |
| `pipeline.rs` (121L) | ⚠️ Stub .cpp | **Missing header**: `<dynamo/pipeline.h>`. Missing all type aliases (`SingleIn`, `ManyIn`, `SingleOut`, `ManyOut`, `ServiceEngine`, `UnaryEngine`, `ServerStreamingEngine`, `ClientStreamingEngine`, `BidirectionalStreamingEngine`), `PipelineIO` trait, `Event` struct. |
| `pipeline/context.rs` (467L) | ⚠️ Stub .cpp | **Missing header**: `<dynamo/pipeline/context.h>`. Missing `Context<T>`, `Controller` (state machine), `StreamContext`, `Registry` (type-erased), `State` enum, `IntoContext` trait, `map()/try_map()/transfer()/into_parts()/insert()/get()/clone_unique()/take_unique()`. |
| `pipeline/nodes.rs` (351L) | ⚠️ Stub .cpp | **Missing header**: `<dynamo/pipeline/nodes.h>`. Missing `Source<T>`, `Sink<T>`, `Edge<T>`, `Operator<UI,UO,DI,DO>`, `PipelineOperator<...>`, `PipelineNode<In,Out>`, `NodeFn`, `ServiceFrontend`, `ServiceBackend`, `SegmentSource`, `SegmentSink`. |
| `protocols/annotated.rs` (168L) | ❌ Missing | **Missing header**: `<dynamo/protocols/annotated.h>`. Missing `Annotated<T>` with `from_error()`, `from_data()`, `from_annotation()`, `ok()`, `is_ok()`, `is_err()`, `is_event()`, `transfer()`, `map_data()`, `into_result()`. |
| `worker.rs` (211L) | ❌ Missing | **Missing header**: `<dynamo/worker.h>`. Missing `Worker` struct with `from_settings()`, `execute()`, `signal_handler()`, shutdown timeout constants. |
| `slug.rs` (163L) | ❌ Missing | **Missing header**: `<dynamo/slug.h>`. Missing `Slug` class with BLAKE3-based unique slugification, `InvalidSlugError`. |
| `traits.rs` (33L) | ❌ Missing | **Missing header**: `<dynamo/traits.h>`. Missing `RuntimeProvider`, `DistributedRuntimeProvider`, `EventPublisher`, `EventSubscriber` traits. |

**Rationale**: Without headers, nothing compiles. The existing `.cpp` files `#include` files that don't exist.
**Dependencies**: None (prerequisite for everything).
**Complexity**: M — 12 header files, mostly porting Rust declarations to C++.
**Implementation order**: **0 — MUST GO FIRST**.

### 0.2 Fix proto build integration
**Gap**: The root `CMakeLists.txt` references `dynamo_protos` but the protobuf generation logic uses `protobuf_generate()` which requires `find_package(Protobuf)` and `find_program(grpc_cpp_plugin)`. The `find_package(Protobuf)` call occurs before submodules are configured, which will fail until protobuf is built.

**Rationale**: The gRPC service stubs (EndpointService, LLMService) are needed by transport and llm modules.
**Dependencies**: 0.1 (headers exist), protobuf + grpc submodule availability.
**Complexity**: M — restructure CMake to build protobuf first or use an ExternalProject/bundle approach.
**Implementation order**: **0** (parallel with 0.1).

### 0.3 Restructure CMake to handle Folly + gRPC complexity
**Gap**: Folly has complex build requirements (double-conversion, glog, gflags, libevent, fmt, etc.). gRPC requires protobuf, cares about ABI, and has its own submodules. The current CMakeLists.txt naively does `add_subdirectory` on each — this will fail in practice.

**Rationale**: Neither Folly nor gRPC are simple "drop-in subdirectory" dependencies. They need proper handling.
**Dependencies**: 0.1, 0.2.
**Complexity**: XL — Folly and gRPC both have deep dependency chains and platform-specific build issues.
**Implementation order**: **0** (parallel with 0.1, 0.2).

---

## Phase 1: Core Runtime Completeness (Order: 1)

### 1.1 Create `DistributedRuntime` class
**Rust**: `lib.rs:71-86` — `DistributedRuntime { runtime, etcd_client, nats_client, tcp_server, component_registry }`
**C++**: ❌ Missing.

**What to add**: A new class `dynamo::DistributedRuntime` that holds:
- `std::shared_ptr<Runtime> runtime`
- `std::shared_ptr<transport::EtcdClient> etcd_client`
- `std::shared_ptr<transport::GrpcServer> grpc_server`
- `std::shared_ptr<ComponentRegistry> component_registry`

Implements `RuntimeProvider` (delegates to `runtime`).

**Rationale**: This is the central coordination point in Dynamo. Every component, endpoint, and discovery operation flows through it. Without it, nothing in the distributed model works.
**Dependencies**: 0.1, 0.3.
**Complexity**: S.
**Implementation order**: **1**.

### 1.2 Add `Runtime::id()` and `Runtime::child_token()`
**Rust**: `runtime.rs:57`, `runtime.rs:93`, `runtime.rs:106`
**C++**: ❌ Missing `id()` accessor; `child_token()` not implemented.

**What to add**:
- `const std::string& Runtime::id() const noexcept`
- `CancellationToken Runtime::child_token()` — creates a linked token that cancels when the parent cancels.

**Rationale**: `id()` is used throughout Dynamo for logging, lease naming, and endpoint registration. `child_token()` is essential for hierarchical cancellation (stream-level cancellation inheriting from runtime-level shutdown).
**Dependencies**: 1.1.
**Complexity**: XS.
**Implementation order**: **1** (parallel with 1.1).

### 1.3 Add `RuntimeType` enum and `from_current()`/`single_threaded()` factories
**Rust**: `runtime.rs:23-29` — `RuntimeType { Shared(Arc<Runtime>), External(Handle) }`, `Runtime::from_current()`, `Runtime::from_handle()`, `Runtime::single_threaded()`
**C++**: ❌ Missing.

**What to add**:
- `enum class RuntimeType { Internal, External }` — or use folly executor types
- `static std::shared_ptr<Runtime> Runtime::single_threaded()` — creates a runtime with one worker thread
- `static std::shared_ptr<Runtime> Runtime::from_existing(folly::Executor&)` — wraps an existing executor

**Rationale**: The Rust code supports multiple construction modes for embedding scenarios. Without these, Dynamo can't be embedded in other applications.
**Dependencies**: 1.1.
**Complexity**: S.
**Implementation order**: **1**.

### 1.4 Implement `Worker` class with signal handling
**Rust**: `worker.rs:60-211`
**C++**: ❌ Missing.

**What to add**: A `Worker` class that:
- Takes a `RuntimeConfig` or `Runtime`
- Provides `execute(Fn)` that blocks and handles `SIGINT`/`SIGTERM`
- Implements graceful shutdown with configurable timeout (default: 30s release, 5s debug)
- Returns exit code 911 on signal (matching Rust behavior)

**Rationale**: Every Dynamo binary (dynamo-run, http component, llmctl) uses `Worker::execute()` as its entry point. Without it, no application can use the runtime properly.
**Dependencies**: 1.1, 1.2.
**Complexity**: M — need signal handling (sigaction + signalfd or folly::AsyncSignalHandler).
**Implementation order**: **2**.

### 1.5 Add config helpers: `env_is_truthy()`, `is_truthy()`, `RuntimeConfigBuilder`
**Rust**: `config.rs:146-171`
**C++**: ❌ Missing.

**What to add**:
- `bool env_is_truthy(const char* env)` / `bool is_truthy(std::string_view)` — "true", "1", "yes" → true
- `RuntimeConfigBuilder` class with fluent API: `.num_worker_threads(N)`, `.max_blocking_threads(N)`, `.build()`
- `RuntimeConfig::single_threaded()` convenience factory

**Rationale**: These helpers are used across the Rust codebase and make config construction ergonomic. Without the builder, C++ code uses aggregate initialization which is fragile.
**Dependencies**: 0.1.
**Complexity**: XS.
**Implementation order**: **2** (parallel with 1.4).

---

## Phase 2: Component Model (Order: 2)

### 2.1 Create `Endpoint::Client<T,U>` with routing strategies
**Rust**: `component/client.rs:52-256`
**C++**: ❌ Missing.

**What to add**: A `Client<Req, Resp>` template class on `Endpoint` with:
- `round_robin()` — round-robin across discovered endpoint instances
- `random()` — random selection
- `direct(endpoint_id)` — direct to specific instance
- `wait_for_endpoints()` — block until endpoints appear
- `endpoint_ids()` — current live endpoint list
- Implements `AsyncEngine<Req, Resp>` delegating to gRPC stub

Uses `DiscoveryClient` to watch for endpoint changes (etcd prefix watch → update local routing table).

**Rationale**: Without `Client`, endpoints are one-way. The whole point of the component model is RPC — `Client` is the caller side.
**Dependencies**: 1.1 (DistributedRuntime), 1.5 (DiscoveryClient), gRPC service stubs.
**Complexity**: L — requires etcd watcher integration, routing state machine, gRPC stub lifetime management.
**Implementation order**: **3**.

### 2.2 Implement `ServiceConfig` / `EndpointConfig` with gRPC-based endpoint lifecycle
**Rust**: `component/service.rs:110L`, `component/endpoint.rs:143L`
**C++**: ❌ Missing.

**What to add**:
- `EndpointConfig` class with `from_endpoint(endpoint)`, `stats_handler(callback)`, `start()`
- `ServiceConfig` class with `from_component(component)`, `description()`, `create()`
- `start()` registers the gRPC service, creates an etcd lease, writes `ComponentRegistration` to etcd

**Rationale**: The Rust code uses `ServiceConfig::create()` to bring up a component with NATS service + etcd registration. Without this, endpoints can't be discovered.
**Dependencies**: 2.1, real etcd client (3.2), gRPC service implementation (4.1).
**Complexity**: L — ties together gRPC service registration, etcd lease management, and component lifecycle.
**Implementation order**: **4**.

### 2.3 Add `Registry` for shared endpoint watchers
**Rust**: `component/registry.rs:99L`
**C++**: ❌ Missing.

**What to add**: A thread-safe singleton-per-`DistributedRuntime` that tracks active etcd watchers so multiple `Client` instances targeting the same remote endpoint share a single watcher task.

**Rationale**: Without this, creating 100 `Client` objects for the same remote endpoint creates 100 redundant etcd watchers.
**Dependencies**: 2.1.
**Complexity**: S — mutex-guarded map of watcher IDs.
**Implementation order**: **4** (parallel with 2.2).

### 2.4 Add `Slug` class with BLAKE3-based unique slugification
**Rust**: `slug.rs:163L`
**C++**: ❌ Missing.

**What to add**: `Slug(std::string)` that:
- Lowercases, keeps only `[a-z0-9_]`
- Appends a 4-byte BLAKE3 hash suffix for uniqueness
- Implements `Display`, comparison, `TryFrom<std::string>`, `AsRef<std::string>`

**Rationale**: Slugs are used for NATS subject generation and service naming. With gRPC they're less critical, but still needed for etcd paths, component names, and human-readable identifiers.
**Dependencies**: None (can use `<openssl/sha.h>` or `<blake3.h>`).
**Complexity**: S — handful of free functions, could use abseil strings + a hash library.
**Implementation order**: **2** (parallel with 1.4, 1.5).

---

## Phase 3: Transport & Service Discovery (Order: 3)

### 3.1 Implement real etcd client via gRPC
**Rust**: `transports/etcd/` (implied via `etcd-client` crate)
**C++**: In-memory mock only.

**What to add**: Replace `transport_etcd.cpp`'s in-memory `std::map` with real etcd gRPC calls:
- Connect via gRPC to `EtcdConfig::endpoints`
- `put(key, value)` → `etcdserverpb::PutRequest`
- `get(key)` → `etcdserverpb::RangeRequest`
- `del(key)` → `etcdserverpb::DeleteRangeRequest`
- `get_prefix(prefix)` → `etcdserverpb::RangeRequest` with range_end
- `grant_lease(ttl)` → `etcdserverpb::LeaseGrantRequest`
- `keep_alive_lease(id)` → `etcdserverpb::LeaseKeepAliveRequest` (bidirectional streaming)
- `watch_prefix(prefix, cb)` → `etcdserverpb::WatchRequest` (bidirectional streaming)
- `revoke_lease(id)` → `etcdserverpb::LeaseRevokeRequest`

**Rationale**: In-memory etcd is useless for distributed operation. Without real etcd, service discovery doesn't work across nodes.
**Dependencies**: 0.2 (protobuf build), gRPC availability.
**Complexity**: XL — etcd gRPC API is extensive, lease keep-alive is a streaming RPC with complex lifecycle management, and the watcher protocol requires maintaining watch IDs and reconnections.
**Implementation order**: **5**.

### 3.2 Implement etcd lease keep-alive loop
**Rust**: `transports/etcd/lease.rs:117L`
**C++**: ❌ Missing.

**What to add**: A background coroutine that:
- Takes a lease ID, TTL, and cancellation token
- Sends periodic `LeaseKeepAliveRequest` via gRPC bidirectional stream
- Retries with exponential backoff on failure
- Revokes lease on cancellation
- Logs warnings if heartbeat fails repeatedly

**Rationale**: Without keep-alive, etcd leases expire in seconds and component registrations disappear. Every production deployment needs this.
**Dependencies**: 3.1 (real etcd client).
**Complexity**: L — background task lifecycle, streaming RPC management, retry logic.
**Implementation order**: **5** (parallel with 3.1).

### 3.3 Implement etcd prefix watcher with reconnection
**Rust**: `transports/etcd/` (watcher via `etcd-client` crate)
**C++**: ❌ Missing.

**What to add**: A watcher that:
- Opens a `WatchRequest` bidirectional gRPC stream for a prefix
- Receives `WatchResponse` events (Put, Delete)
- Translates to `WatchCallback(key, value, deleted)` invocations
- Reconnects on stream failure with backoff
- Maintains a map of watch IDs for cancellation

**Rationale**: Service discovery relies on prefix watching. Without this, `Client` can't react to endpoint changes.
**Dependencies**: 3.1.
**Complexity**: L — streaming gRPC management, reconnection logic, event deduplication.
**Implementation order**: **5** (parallel with 3.1, 3.2).

### 3.4 Implement TCP call-handshake protocol
**Rust**: `pipeline/network/tcp/server.rs:614L`
**C++**: ⚠️ `transport_tcp.cpp` — basic `TcpServer`/`TcpClient` only.

**What to add**:
- `CallHomeHandshake` message — first TCP bytes identifying stream subject and type (request/response)
- `ControlMessage` — STOP, KILL, SENTINEL sentinel values for stream lifecycle
- `TwoPartCodec` — length-delimited framing: header (control message) + body (payload), xxhash3 checksums
- `PendingConnections` / `RegisteredStream` — registration pattern for call-home response streaming
- `ResponseService` trait — `register(options) -> PendingConnections`
- `StreamOptions` — configurable buffer sizes, timeouts

**Rationale**: The TCP call-home pattern is fundamental to Dynamo's data plane. Without the full handshake and codec, TCP transport is not interoperable with the Rust implementation.
**Dependencies**: 0.1, asio.
**Complexity**: L — wire protocol implementation, buffer management, framing codec.
**Implementation order**: **6**.

---

## Phase 4: Pipeline Execution Model (Order: 4)

### 4.1 Implement full `Context<T>` with `Controller` state machine
**Rust**: `pipeline/context.rs:467L`
**C++**: ❌ Stub file only.

**What to add**: Full `Context<T>` class in `<dynamo/pipeline/context.h>`:
- `Controller` with watch-channel-based state machine (`State::Live | Stopped | Killed`)
- `Context<T>` with `id()`, `controller()`, `insert()`, `get()`, `clone_unique()`, `take_unique()`, `transfer()`, `into_parts()`, `map()`, `try_map()`
- `StreamContext` for detached context sharing (no value, only metadata)
- `Registry` with type-erased `std::any`-like storage supporting shared and unique ownership
- `IntoContext` concept

**Rationale**: `Context<T>` is the heart of the pipeline — it carries the request value, metadata, and lifecycle state through every pipeline stage. Without it, there is no pipeline.
**Dependencies**: 0.1.
**Complexity**: M — template metaprogramming for type-erased registry, async channel for Controller.
**Implementation order**: **7**.

### 4.2 Implement `Source`, `Sink`, `Edge`, `Operator` pipeline nodes
**Rust**: `pipeline/nodes.rs:351L`
**C++**: ❌ Stub file only.

**What to add**: Full pipeline node types in `<dynamo/pipeline/nodes.h>`:
- `Source<T>` — emits data downstream via `on_next()`, `set_edge()`, `link(Sink)`
- `Sink<T>` — receives data via `on_data()`
- `Edge<T>` — typed connection between `Source` and `Sink`
- `Operator<UI,UO,DI,DO>` — bidirectional transform: `forward(UpIn) -> DownIn`, `backward(DownOut) -> UpOut`
- `PipelineOperator<...>` — wraps an `Operator`, exposes forward/backward edges
- `PipelineNode<In,Out>` — simple unidirectional transform
- `ServiceFrontend<In,Out>` / `ServiceBackend<In,Out>` — pipeline entry/exit points

**Rationale**: Without these, the pipeline is just a concept document. These types make the pipeline executable.
**Dependencies**: 4.1 (Context).
**Complexity**: M — template-heavy, but straightforward port from Rust.
**Implementation order**: **7** (parallel with 4.1).

### 4.3 Implement `AsyncEngine` hierarchy
**Rust**: `engine.rs:168L`
**C++**: ⚠️ `engine.h` attempts this but header doesn't exist.

**What to add**: Create `<dynamo/engine.h>` with:
- `Data` concept (C++20 concept: `std::movable && std::destructible`)
- `AsyncEngineContext` interface (not just a class — Rust has `dyn AsyncEngineContext`)
- `AsyncEngineContextProvider` trait
- `AsyncEngine<Req, Resp, E>` abstract class with `virtual Task<void> generate(Req, ResponseStream<Resp>) = 0`
- `AsyncEngineUnary<Resp>` — a `SemiFuture<Resp>` + context
- `AsyncEngineStream<Resp>` — an `AsyncGenerator<Resp>` + context
- `ResponseStream<R>` — wraps data stream + context, Stream impl

**Rationale**: `AsyncEngine` is the fundamental execution trait in Dynamo. Every component, pipeline node, client, and backend implements it. Without it, nothing can be composed.
**Dependencies**: 4.1 (Context).
**Complexity**: M — template concepts, Folly coro integration.
**Implementation order**: **7** (parallel with 4.1, 4.2).

---

## Phase 5: LLM Layer (Order: 5)

### 5.1 Implement `Annotated<T>` envelope
**Rust**: `protocols/annotated.rs:168L`
**C++**: ❌ Missing.

**What to add**: `<dynamo/protocols/annotated.h>` with:
- `Annotated<T>` struct: `data: optional<T>`, `id: string`, `event: optional<string>`, `comment: optional<string>`
- `static from_data(T)`, `from_error(error)`, `from_event(string)`, `from_annotation(name, value)`
- `is_ok()`, `is_err()`, `is_event()`, `is_error()`
- `transfer(U) -> Annotated<U>`, `map_data(F) -> Annotated<U>`
- `into_result() -> Result<optional<T>, Error>`

**Rationale**: `Annotated<T>` is the universal response envelope used by SSE streaming, error propagation, and pipeline outputs. Without it, streaming responses lack metadata.
**Dependencies**: None.
**Complexity**: S — header-only template class.
**Implementation order**: **8**.

### 5.2 Add `Usage` struct and `ContentProvider` trait
**Rust**: `llm/protocols.rs:76L`
**C++**: ❌ Missing.

**What to add**:
- `struct Usage { int prompt_tokens, completion_tokens, total_tokens; }` with `to_json`/`from_json`
- `struct ContentProvider` concept — `fn content(&self) -> std::string`

**Rationale**: The OpenAI API returns `Usage` in every response. `ContentProvider` is used for extracting text from response choices.
**Dependencies**: 5.1.
**Complexity**: XS.
**Implementation order**: **8** (parallel with 5.1).

### 5.3 Add SSE codec: `convert_sse_stream()`
**Rust**: `llm/protocols.rs:76L` — `convert_sse_stream(stream) -> DataStream<Annotated<R>>`
**C++**: ❌ Missing — HTTP server does manual SSE formatting.

**What to add**: A reusable SSE codec:
- `SSEEncoder` class: `encode(Annotated<T>) -> string` (formats `data: ...\n\n`)
- `SSEDecoder` class: `decode(string_view) -> optional<Annotated<json>>` (parses SSE frames)
- `convert_to_sse_stream(AsyncGenerator<Annotated<T>>) -> AsyncGenerator<string>`

**Rationale**: SSE is how OpenAI-compatible streaming works. The HTTP server currently hardcodes SSE formatting. A reusable codec enables testing and reuse.
**Dependencies**: 5.1.
**Complexity**: S — parsing/formatting logic.
**Implementation order**: **8**.

### 5.4 Integrate HuggingFace Tokenizer or equivalent
**Rust**: `tokenizers.rs:570L` — HuggingFace + SentencePiece wrappers
**C++**: ⚠️ `SimpleTokenizer` — character-level demo only.

**What to add**:
- Option A: Use `sentencepiece` C++ library (if available) or `huggingface/tokenizers` Rust FFI (complex)
- Option B: Implement a minimal BPE tokenizer that can load a merged BPE vocab + merges file (simpler but limited)
- `Tokenizer` abstract class with `encode()`, `decode()`, `encode_batch()`, `decode_batch()`
- `HuggingFaceTokenizer` wrapper loading `tokenizer.json` files
- `DecodeStream` — stateful incremental decoder (`add_token(id) -> optional<string>`)
- `StopSequenceDecoder` with visible/hidden stop tokens and sequences, builder pattern

**Rationale**: Tokenization is essential for any LLM serving. `SimpleTokenizer` is a placeholder that can't serve real models. Without proper tokenization, the LLM examples are non-functional.
**Dependencies**: 5.1.
**Complexity**: XL — HuggingFace tokenizer is complex. SentencePiece is easier but still significant. Option B (minimal BPE) is M.
**Implementation order**: **9**.

### 5.5 Wire `Backend` as a pipeline `Operator`
**Rust**: `backend.rs:508L` — Backend implements `Operator<BackendInput, BackendOutput, BackendInput, LLMEngineOutput>`
**C++**: `Backend` is a standalone abstract class, not a pipeline operator.

**What to add**: Make `Backend` implement `Operator`:
- `Backend::forward(BackendInput) -> BackendInput` (pass-through)
- `Backend::backward(LLMEngineOutput) -> BackendOutput` (decode tokens, check stop conditions)
- Integrate `Decoder` as the stop-condition-checking step in the backward path
- Add `StepResult` / `SeqResult` / `StopTrigger` types for more diagnostic information

**Rationale**: The Rust `Backend` is a pipeline `Operator`. Making it an operator allows it to be composed in pipelines naturally.
**Dependencies**: 4.2 (Operator), 5.4 (tokenizer).
**Complexity**: M — port from Rust's backend.rs.
**Implementation order**: **10**.

### 5.6 Add KV router metrics aggregation + scoring
**Rust**: `kv_router/indexer.rs`, `kv_router/metrics_aggregator.rs`, `kv_router/scheduler.rs`, `kv_router/publisher.rs`, `kv_router/scoring.rs`
**C++**: ⚠️ `KvIndexer` exists, `KvRouter` exists, but no metrics/scheduler/publisher/scoring.

**What to add**:
- `KvMetricsAggregator` — collects `load_factor`, `kv_cache_usage_pct` from worker heartbeats
- `KvScheduler` — takes indexer scores + metrics → final decision
- `KvScorer` — cost function: `(KV match ratio) - (load_factor * weight) + (cache_usage * weight)`
- `KvPublisher` — publishes KV events to the event bus (gRPC broadcast)
- Subscribe to KV events from workers and update the indexer

**Rationale**: Without metrics aggregation, the router schedules based on prefix match alone — ignoring load, which is the whole point of KV-aware routing.
**Dependencies**: 3.1 (real etcd for event bus).
**Complexity**: L — event subscription, metrics collection, configurable scoring weights.
**Implementation order**: **11**.

### 5.7 Disaggregated router with etcd config watcher
**Rust**: `disagg_router.rs:259L` — `new_with_etcd_and_default()`, `start_config_watcher()`, `check_for_updates()`
**C++**: ⚠️ Simple boolean decision, no dynamic config.

**What to add**:
- `DisaggregatedRouter::new_with_etcd(drt, model_name, default)` — reads config from `etcd:/dynamo/disagg_router/models/{model_name}`
- `start_config_watcher()` — etcd prefix watch → `reconfigure()`
- `get_model_name()` accessor

**Rationale**: The whole point of disaggregated routing is runtime configurability. Without etcd watching, the config is static.
**Dependencies**: 3.2 (etcd watcher).
**Complexity**: S — mostly wrapping etcd watcher + calling `reconfigure()`.
**Implementation order**: **11** (parallel with 5.6).

---

## Phase 6: Network Transport (Order: 6)

### 6.1 Implement gRPC `EndpointService` handler
**Rust**: NATS-based `PushEndpoint` + two-part message dispatch
**C++**: ❌ Missing.

**What to add**: Implement the `EndpointService::Call` bidirectional streaming RPC from `runtime.proto`:
- Receives `EndpointMessage` from clients
- Demultiplexes by `endpoint_name` to the correct pipeline
- Returns `EndpointMessage` responses with stream control
- Handles `ControlMessage::STOP`, `KILL`, `SENTINEL`

This replaces the NATS `PushEndpoint` + `AddressedPushRouter` pattern from Rust.

**Rationale**: Without this, gRPC transport is a stub — no actual RPC happens.
**Dependencies**: 0.2 (protobuf), 2.2 (EndpointConfig).
**Complexity**: L — gRPC bidirectional streaming, endpoint routing, control message handling.
**Implementation order**: **12**.

### 6.2 Implement gRPC `LLMService` handler
**Rust**: Various engine backends (vLLM, TRT-LLM, etc.)
**C++**: ❌ Missing.

**What to add**: Implement the `LLMService::Generate` server-streaming RPC from `llm.proto`:
- Takes `GenerateRequest` (model, input_ids, params)
- Returns stream of `GenerateResponse` (token_id, finished, finish_reason)
- Delegates to `Backend::generate()`

**Rationale**: This is the primary serving API. Without it, clients can't send generation requests.
**Dependencies**: 0.2 (protobuf), 5.5 (Backend as Operator).
**Complexity**: M — streaming RPC handler, token streaming.
**Implementation order**: **12** (parallel with 6.1).

### 6.3 Implement gRPC client-side load balancing for `EndpointService`
**Rust**: `component/client.rs` — round_robin/random/direct
**C++**: ❌ Missing.

**What to add**: `GrpcEndpointClient<Req, Resp>` class that:
- Maintains a list of discovered gRPC endpoints
- Creates/reuses gRPC stubs per endpoint
- Implements round-robin/random/direct routing strategies
- Watches for endpoint changes (etcd → update routing table)
- Thread-safe, lock-free or fine-grained locking

**Rationale**: Without client-side LB, the gRPC client hits a single endpoint. This is the C++ equivalent of the Rust `Client` struct but using gRPC channels instead of NATS subjects.
**Dependencies**: 3.1 (real etcd), 3.3 (etcd watcher), 6.1 (EndpointService).
**Complexity**: L — stub pool management, health checking, reconnection.
**Implementation order**: **13**.

---

## Phase 7: HTTP & API Completeness (Order: 7)

### 7.1 Full OpenAI-compatible HTTP API
**Rust**: `components/http/` — axum-based with `/v1/chat/completions`, `/v1/completions`, `/v1/models`
**C++**: ⚠️ `http/server.cpp` — basic asio HTTP/1.1, only `/v1/completions`.

**What to add**:
- `/v1/chat/completions` — takes `ChatCompletionRequest`, returns SSE stream of `ChatCompletionResponse`
- `/v1/completions` — takes `CompletionRequest`, returns SSE stream of `CompletionResponse`
- `/v1/models` — returns model list queried from etcd registrations
- Proper HTTP routing (not raw asio; use a library or add a simple router)
- Content-Type negotiation, error responses with OpenAI-compatible error format
- CORS headers for web clients

**Rationale**: The HTTP component is the primary user-facing API for Dynamo. Without full OpenAI compatibility, users can't use standard client libraries.
**Dependencies**: 6.2 (LLMService), 5.3 (SSE codec).
**Complexity**: L — HTTP routing, SSE streaming, model listing, error formatting.
**Implementation order**: **14**.

### 7.2 OpenAI-compatible error format
**Rust**: Uses `http-api-problem` crate for RFC 7807 problem details
**C++**: ❌ Returns raw exception strings.

**What to add**: Standard error response format:
- `{"error": {"message": "...", "type": "...", "param": null, "code": null}}`
- Error types: `invalid_request_error`, `server_error`, `rate_limit_error`, etc.

**Rationale**: OpenAI API clients parse this specific error format.
**Dependencies**: 7.1.
**Complexity**: XS.
**Implementation order**: **14** (parallel with 7.1).

---

## Phase 8: Testing & Validation (Order: 8)

### 8.1 Integration tests with gRPC end-to-end
**Rust**: Integration tests in `dynamo/tests/` (workflow tests)
**C++**: ❌ No integration tests.

**What to add**: Integration tests that:
- Start `DistributedRuntime` with in-memory etcd
- Create two components (server + client) on different "nodes"
- Register endpoints, discover each other
- Send a request through the pipeline and verify the response

**Rationale**: Without integration tests, there's no assurance the components work together.
**Dependencies**: Everything in phases 1–6.
**Complexity**: M — test fixtures, process/thread management.
**Implementation order**: **15**.

### 8.2 Performance benchmarks
**Rust**: Uses `criterion` for benchmarks
**C++**: ❌ Missing.

**What to add**: Google Benchmark (or folly Benchmark) tests for:
- Pipeline throughput (Context creation, transfer, map)
- gRPC round-trip latency
- TCP call-home throughput
- Tokenizer encode/decode throughput
- Router scheduling latency

**Rationale**: Dynamo is a performance-critical system. Without benchmarks, regressions can't be detected.
**Dependencies**: All modules must be functional.
**Complexity**: M — benchmark setup, statistical analysis.
**Implementation order**: **16**.

### 8.3 Testing with real etcd (docker-compose)
**Rust**: `deploy/docker-compose.yml` — etcd + NATS + Prometheus + Grafana
**C++**: ❌ Missing.

**What to add**: `docker-compose.yml` for test infrastructure:
- etcd (for real integration tests)
- gRPC reflection (for debugging)
- Optionally: Grafana + Prometheus for metrics

**Rationale**: In-memory etcd is fine for unit tests but distributed features need real etcd.
**Dependencies**: 3.1 (real etcd client).
**Complexity**: S — docker-compose file + CI integration.
**Implementation order**: **15** (parallel with 8.1).

---

## Phase 9: Production Polish (Order: 9)

### 9.1 Metrics and observability
**Rust**: `components/metrics/` — Prometheus metrics with OpenTelemetry
**C++**: ❌ Missing.

**What to add**: Prometheus metrics via `prometheus-cpp`:
- `dynamo_requests_total` counter
- `dynamo_request_duration_seconds` histogram
- `dynamo_tokens_generated_total` counter
- `dynamo_active_requests` gauge
- Metrics HTTP endpoint for Prometheus scraping

**Rationale**: Without metrics, operators can't observe system behavior.
**Dependencies**: 6.1, 6.2 (gRPC services).
**Complexity**: M — metrics collection, HTTP exposition endpoint.
**Implementation order**: **17**.

### 9.2 Structured logging / tracing
**Rust**: `tracing` crate with structured fields and spans
**C++**: ⚠️ `spdlog` basic logging.

**What to add**:
- Structured log fields (request_id, component, endpoint, duration)
- Trace context propagation through gRPC metadata
- Configurable log level per-module
- JSON log output (for log aggregation systems)

**Rationale**: Structured logging is essential for debugging distributed systems.
**Dependencies**: None.
**Complexity**: S — spdlog already handles most of this; just need convention and propagation.
**Implementation order**: **17** (parallel with 9.1).

### 9.3 Graceful shutdown with drain
**Rust**: `worker.rs` — signal → start drain → stop accepting → wait for inflight → shutdown
**C++**: ❌ Missing.

**What to add**: Shutdown sequence:
1. Signal received (or `Runtime::shutdown()` called)
2. Stop accepting new gRPC connections
3. Send `ControlMessage::STOP` to inflight streams
4. Wait for inflight streams to complete (with timeout)
5. `CancellationToken::cancel()` → cascade to all children
6. Join all thread pools

**Rationale**: Without graceful draining, in-flight requests are aborted and clients see errors during rolling updates.
**Dependencies**: 1.4 (Worker), 6.1 (gRPC services).
**Complexity**: M — drain orchestration, stream tracking.
**Implementation order**: **18**.

### 9.4 `llmctl` equivalent — CLI for deployment management
**Rust**: `launch/llmctl/` — CLI tool using `clap` + `tabled`
**C++**: ❌ Missing.

**What to add**: A CLI tool that:
- Lists registered components/endpoints via etcd
- Shows model deployment cards
- Controls per-endpoint configuration
- Tabular output via `tabulate` or similar

**Rationale**: Operators need a way to inspect and control deployments.
**Dependencies**: 3.1 (real etcd).
**Complexity**: M — CLI framework, etcd queries, table formatting.
**Implementation order**: **19**.

---

## Summary Table

| Phase | Items | Complexity | Order |
|---|---|---|---|
| **0: Make it compile** | 0.1–0.3 | XL | **0 — MUST GO FIRST** |
| **1: Core Runtime** | 1.1–1.5 | M | **1** |
| **2: Component Model** | 2.1–2.4 | L | **2** |
| **3: Transport & Discovery** | 3.1–3.4 | XL | **3** |
| **4: Pipeline Execution** | 4.1–4.3 | L | **4** |
| **5: LLM Layer** | 5.1–5.7 | XL | **5** |
| **6: Network Transport** | 6.1–6.3 | L | **6** |
| **7: HTTP & API** | 7.1–7.2 | L | **7** |
| **8: Testing** | 8.1–8.3 | M | **8** |
| **9: Production Polish** | 9.1–9.4 | L | **9** |

Total estimated complexity to reach parity with the Rust codebase: ~25–30 new files, ~15,000–20,000 lines of C++ code beyond the current ~2,000 lines. This reflects the Rust codebase's ~30,000 lines of implementation code (excluding vendored/third-party code).

**Key differentiator vs Rust**: The gRPC-for-NATS replacement (items 6.1, 6.3, 3.1–3.3) is the single largest deviation. In Rust, NATS handles ~80% of the networking complexity. In C++, gRPC fills this role but requires explicit implementation of endpoint discovery, load balancing, and streaming control that NATS provides for free.
