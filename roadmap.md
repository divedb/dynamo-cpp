# Dynamo C++20 Reimplementation — Roadmap

## 1. What NVIDIA Dynamo Does

NVIDIA Dynamo is a **distributed inference serving runtime** designed for large language models (LLMs) and generative AI workloads. It provides:

- **Disaggregated serving**: Separate prefill (compute-heavy) and decode (latency-sensitive) onto different workers.
- **KV-cache-aware routing**: Route requests to workers with the highest cache hit rate.
- **Dynamic GPU scheduling**: Allocate GPUs based on real-time demand.
- **Pipeline-based execution model**: Composable operators connected in a directed graph.
- **Distributed service discovery**: Components register with etcd, discover each other via NATS.
- **Streaming responses**: Request via NATS control plane, response streaming over direct TCP (call-home pattern).
- **Multi-engine support**: vLLM, TensorRT-LLM, SGLang, llama.cpp, Mistral.rs.

## 2. Key Modules in the Original Implementation

| Module | Crate | Responsibility |
|---|---|---|
| **Runtime** | `dynamo-runtime` | Dual-threaded async runtime (primary + secondary), cancellation, worker entry point |
| **Config** | `dynamo-runtime::config` | Layered config (env → TOML → defaults) via `figment` |
| **Component Model** | `dynamo-runtime::component` | `Namespace` → `Component` → `Endpoint` hierarchy, etcd-based registration |
| **Pipeline** | `dynamo-runtime::pipeline` | `Context<T>`, `AsyncEngine`, `Source`/`Sink`/`Operator` node graph |
| **Network** | `dynamo-runtime::pipeline::network` | NATS-based request dispatch, TCP call-home streaming, two-part message codec |
| **Transports** | `dynamo-runtime::transports` | NATS client, etcd client/leases, ZMQ, TCP server |
| **Service Discovery** | `dynamo-runtime::discovery` | etcd-based lease management and prefix watching |
| **LLM Backend** | `dynamo-llm` | Backend, Decoder, stop conditions, tokenization, engine wrappers |
| **KV Router** | `dynamo-llm::kv_router` | Radix-tree KV index, metrics aggregation, scheduler |
| **Disagg Router** | `dynamo-llm::disagg_router` | Prefill/decode split decision |
| **Protocols** | `dynamo-llm::protocols` | OpenAI-compatible request/response types, `Annotated<R>` envelope |
| **HTTP Frontend** | `components/http` | OpenAI-compatible HTTP server (axum) |

## 3. Scope of C++20 Reimplementation

We reimplement the **core distributed runtime** — the skeleton upon which inference engines attach. The following are in scope:

| Component | Priority | Rationale |
|---|---|---|
| **Runtime** (dual thread pool, cancellation) | P0 | Foundation of everything |
| **Configuration system** | P0 | Required before any component can run |
| **Component/Endpoint model** | P0 | Core distributed abstraction |
| **Pipeline execution model** | P0 | The data-flow abstraction |
| **Service discovery (etcd)** | P0 | How components find each other |
| **Network transport (gRPC + TCP streaming)** | P0 | Inter-component communication |
| **Pipeline operators and context** | P1 | Composable execution |
| **LLM primitives** (Backend skeleton, Decoder) | P1 | High-level LLM abstraction |
| **KV Cache Router skeleton** | P2 | Routing abstraction |
| **Disaggregated Router skeleton** | P2 | Prefill/decode split |
| **HTTP frontend** | P2 | Entry point for clients |

The following are **out of scope** or deferred:
- Actual integration with inference engines (vLLM, TRT-LLM, etc.)
- CUDA/CUDAGraph internals
- Python bindings
- Production-grade metrics (Prometheus)
- Kubernetes operator / Helm charts

## 4. Proposed C++20 Architecture

```
dynamo/
├── CMakeLists.txt                  # Root CMake
├── third_party/                    # Git submodules
│   ├── folly/                      # Async, futures, IO
│   ├── fmt/                        # Formatting
│   ├── spdlog/                     # Logging
│   ├── abseil-cpp/                 # Flat maps, strings, etc.
│   ├── grpc/                       # RPC framework
│   ├── protobuf/                   # Serialization
│   ├── asio/                       # Networking (TCP streaming)
│   ├── nlohmann_json/              # JSON
│   ├── toml11/                     # TOML config parsing
│   └── catch2/                     # Unit testing
├── lib/
│   ├── runtime/                    → dynamo-runtime equivalent
│   │   ├── include/dynamo/
│   │   │   ├── runtime.h           # Dual-threaded runtime
│   │   │   ├── config.h            # Layered config (env → toml → defaults)
│   │   │   ├── component.h         # Component, Namespace, Endpoint
│   │   │   ├── pipeline.h          # Context<T>, Engine traits
│   │   │   ├── engine.h            # AsyncEngine, ResponseStream
│   │   │   ├── transport/          # etcd, gRPC, TCP
│   │   │   └── discovery.h         # etcd-based service discovery
│   │   └── src/
│   ├── llm/                        → dynamo-llm equivalent
│   │   ├── include/dynamo/llm/
│   │   │   ├── backend.h           # Backend interface, Decoder
│   │   │   ├── tokenizer.h         # Tokenizer abstraction
│   │   │   ├── protocols.h         # Request/Response types
│   │   │   └── router.h            # KV + Disagg router skeletons
│   │   └── src/
│   └── http/                       → components/http equivalent
│       ├── include/dynamo/http/
│       │   └── server.h            # HTTP frontend
│       └── src/
├── examples/
│   ├── hello_runtime               # Minimal runtime startup
│   ├── echo_pipeline               # Pipeline with simple operator
│   ├── distributed_echo            # Two-component distributed pipeline
│   └── llm_minimal                 # Minimal LLM serving demo
└── tests/
    ├── runtime_tests
    ├── pipeline_tests
    ├── component_tests
    └── network_tests
```

### Component Responsibilities

| C++ Component | Original Equivalent | Description |
|---|---|---|
| `runtime::Runtime` | `Runtime` | Primary + secondary thread pools, cancellation token tree, graceful shutdown |
| `runtime::Config` | `RuntimeConfig` + `WorkerConfig` | Layered config: env vars → TOML files → built-in defaults |
| `runtime::Component` | `Component` + `Namespace` + `Endpoint` | Hierarchical naming, etcd registration, gRPC service exposure |
| `runtime::Pipeline` | `pipeline` module | `Context<T>`, `Engine<Req,Resp>`, `Source<T>`, `Sink<T>`, `Operator` |
| `runtime::Discovery` | `DiscoveryClient` | etcd-based service registration with leases + prefix watching |
| `runtime::transport::grpc` | NATS transport | gRPC bidirectional streaming for request dispatch |
| `runtime::transport::tcp` | TCP call-home | Direct TCP streaming for response data plane |
| `runtime::transport::etcd` | etcd transport | etcd KV client, lease management, watchers |
| `llm::Backend` | `Backend` | Engine interface: `generate()`, token decode, stop conditions |
| `llm::Tokenizer` | `Tokenizer` | HF-compatible tokenizer interface |
| `llm::Protocols` | `Protocols` | Request/response types, `Annotated<T>` envelope |
| `llm::Router` | `KvRouter` + `DisaggregatedRouter` | Routing interfaces (concrete routing logic is future work) |
| `http::Server` | HTTP component | Simple HTTP server using Boost.Beast or similar |

## 5. Dependency Choices

| Dependency | Version | Purpose | Why Not Alternatives |
|---|---|---|---|
| **Folly** | Latest (submodule) | Async runtime, futures, `SemiFuture`, `coro::Task`, `ThreadPoolExecutor` | More mature C++ async than raw asio; provides `co_await` integration; used widely in production C++ systems |
| **Abseil** | LTS (submodule) | `flat_hash_map`, `flat_hash_set`, `string_view`, `StatusOr`, `Span` | Industry standard; replaces handwritten containers |
| **fmt** | Stable (submodule) | Type-safe formatting | C++20 `std::format` still not universally available post-CMake; fmt is the de facto standard |
| **spdlog** | Stable (submodule) | Logging | Header-only, fast, widely used; simple replacement for `tracing` |
| **gRPC** | LTS (submodule) | RPC framework | Replaces NATS for most RPC needs; provides bidirectional streaming; protobuf-based IDL |
| **protobuf** | LTS (submodule) | Message serialization | De facto standard for gRPC; type-safe serialization (replaces serde) |
| **asio** | Standalone (submodule) | Low-level TCP streaming | For call-home TCP data plane; complements gRPC for high-throughput streaming |
| **nlohmann/json** | Stable (submodule) | JSON parsing | Replaces serde_json; widely used, header-only |
| **toml11** | Stable (submodule) | TOML config parsing | Replaces figment's TOML layer; modern C++ |
| **Catch2** | v3 (submodule) | Unit testing | Modern C++ test framework with BDD and sections |

**Key design deviations from NVIDIA Dynamo:**
- gRPC replaces NATS as the primary RPC mechanism (NATS is great but gRPC is more standard in C++ ecosystems and provides native streaming, flow control, and load balancing).
- Folly replaces tokio (obvious choice in C++ for async with coroutine support).
- No Python bindings (would require Pybind11 and significantly more scope).
- Config uses toml11 + env vars directly instead of figment.

## 6. Milestones and Implementation Order

### Milestone 1: Core Runtime Foundation (Week 1)
- [ ] Project scaffolding: CMake, third-party submodules, CI config
- [ ] `runtime::Config`: Layered config reader (env → TOML → defaults)
- [ ] `runtime::Runtime`: Dual thread pool (primary + secondary), `CancellationToken`, graceful shutdown
- [ ] **Test**: Config parsing, Runtime startup/shutdown

### Milestone 2: Distributed Abstractions (Week 2)
- [ ] `runtime::transport::etcd`: Client wrapper, leases, prefix watch
- [ ] `runtime::component`: Namespace, Component, Endpoint hierarchy + etcd registration
- [ ] `runtime::discovery`: Watch for endpoint changes, callbacks
- [ ] **Test**: Component lifecycle, etcd operations (with embedded etcd or mock)

### Milestone 3: Pipeline Execution Model (Week 2-3)
- [ ] `runtime::pipeline::Context`: Metadata registry + lifecycle controller
- [ ] `runtime::engine`: `AsyncEngine` trait, `ResponseStream`
- [ ] `runtime::pipeline::nodes`: `Source`, `Sink`, `Operator`, `PipelineNode`
- [ ] **Test**: Context propagation, engine composition, stream lifecycle

### Milestone 4: Network Transport (Week 3)
- [ ] gRPC service definitions (protobuf) for component communication
- [ ] `runtime::transport::grpc`: gRPC server/client for request dispatch
- [ ] `runtime::transport::tcp`: Direct TCP streaming (call-home pattern)
- [ ] Integration: Pipeline → Network → Remote Pipeline
- [ ] **Test**: gRPC roundtrip, TCP streaming, end-to-end pipeline

### Milestone 5: LLM Primitives (Week 4)
- [ ] `llm::protocols`: Request/Response types, `Annotated<T>`
- [ ] `llm::tokenizer`: Tokenizer interface (BPE/Unigram)
- [ ] `llm::backend`: `Backend` interface, `Decoder` skeleton (stop conditions, token processing)
- [ ] `llm::router`: Router interface
- [ ] **Test**: Tokenizer roundtrip, Decoder stop conditions

### Milestone 6: Examples and HTTP Frontend (Week 4-5)
- [ ] `examples/hello_runtime`: Minimal runtime
- [ ] `examples/echo_pipeline`: Pipeline with a single operator
- [ ] `examples/distributed_echo`: Two-component pipeline over gRPC
- [ ] `examples/llm_minimal`: Tokenize → LLM backend → Decode
- [ ] `http::Server`: Optional HTTP frontend
- [ ] **Test**: Integration tests for each example

### Milestone 7: Polish (Week 5)
- [ ] Documentation (README, API docs)
- [ ] CMake install targets
- [ ] CI workflow
- [ ] Address tech debt from earlier milestones

## 7. Testing Strategy

| Level | Tool | Scope |
|---|---|---|
| **Unit tests** | Catch2 | Each module: Config, Runtime, Component, Pipeline (Context, Engine, nodes), Transport (etcd, gRPC) |
| **Integration tests** | Catch2 + test fixtures | End-to-end pipeline over gRPC, etcd registration → discovery → connect |
| **Examples** | Build-and-run | Each example is a runnable demo that exercises real functionality |
| **Mocking** | Manual mock classes or GMock | For network/etcd tests where an external process is undesirable; use in-memory etcd or gRPC in-process tests |

Key testing principles:
- **Deterministic**: Use mock clocks and single-threaded executors where possible.
- **Fast**: Unit tests must not depend on network or external processes.
- **Coverage**: Every public API has at least one positive and one negative test.
- **Leak-free**: Tests verify that all threads are joined and resources are released.

## 8. Risks, Missing Information, and Open Questions

### Risks

| Risk | Impact | Mitigation |
|---|---|---|
| **Folly complexity** | Steep learning curve, build complexity | Use Folly sparingly: `ThreadPoolExecutor`, `SemiFuture`, `coro::Task` only; avoid `Proxy` |
| **gRPC vs NATS semantics** | NATS is pub/sub + request/reply; gRPC is primarily RPC | Model component interaction as bidirectional gRPC streams (request → response streaming) |
| **etcd availability** | Tests need etcd; CI without etcd is harder | Provide in-process etcd mock; integration tests with docker-compose |
| **C++20 coroutine interoperability** | Folly coro vs C++20 std::coroutine compatibility | Follow Folly's coro patterns (`Task`, `co_await`, `co_viaIfAsync`) |
| **Scope creep** | Reimplementing too much | Strictly scope to runtime; LLM backends are skeletal interfaces only |

### Missing Information

- **NATS-specific features**: JetStream exactly-once delivery, KV store, object store. We replace these with gRPC + etcd.
- **NIXL**: NVIDIA's high-speed data transfer library is proprietary. We use vanilla TCP/gRPC.
- **Production metrics**: Detailed Prometheus metric semantics. We provide basic instrumentation callbacks.

### Open Questions

1. **Should we embed etcd for tests?** Using `etcd-cpp-apiv3` with a real embedded etcd process or a mock? Decision: provide a mock etcd that implements the subset of the API we need.
2. **Coroutine style**: `Folly::coro::Task` or raw `asio::awaitable`? Decision: use Folly coro for consistency with the executor model, asio for low-level TCP.
3. **gRPC server-per-endpoint vs single server**: In Dynamo, each component registers endpoints. Do we run one gRPC server per component or one per endpoint? Decision: one server per component with endpoint demultiplexing (gRPC routing via service name).
4. **Serialization format for pipeline context**: Protobuf for well-defined messages; JSON for debugging/testing.
5. **Support level for non-Linux platforms**: macOS for development (Folly, gRPC buildable via Homebrew).
