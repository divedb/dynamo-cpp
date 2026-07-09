# dynamo-cpp

A C++20 reimplementation of [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo)
(the Rust crates under `third_party/dynamo/lib`): a reusable distributed
runtime framework — local runtime + worker lifecycle, lease-based discovery
with live watches, a component/namespace/endpoint model, dynamic client
routing, and unary-request → streaming-response over an explicit
control-plane / data-plane split — plus the LLM serving layer on top:
OpenAI-compatible protocols and HTTP frontend, prompt templating +
tokenization preprocessor, detokenizing/stop-condition backend, model
deployment cards, and KV-aware routing.

**Start with [docs/architecture.md](docs/architecture.md)** — it contains the
Dynamo architecture summary, the behavioral requirements this port preserves,
the C++ design, and the stated assumptions (notably: NATS/etcd are replaced by
direct TCP dispatch + a small `discoveryd` server that reproduces the lease/
watch contract).

## Layout

```
CMakeLists.txt, CMakePresets.json
docs/            architecture notes
third_party/     git submodules / vendored deps (fmt, spdlog, nlohmann_json,
                 Catch2, toml11, xxHash, minja, dynamo)
src/runtime/     coroutine library (Task, AsyncGenerator, Channel, Event),
                 cancellation, thread-pool executors, Runtime, Worker, config, logging
src/pipeline/    AsyncEngine model: Context/Controller, ResponseStream, Annotated, Operators
src/discovery/   Discovery interface; in-process backend; discoveryd server + TCP client
src/transports/  two-part codec, sockets, control plane (dispatch), data plane (call-home streams)
src/component/   DistributedRuntime, Namespace/Component/Endpoint, serve(), Client (routing)
src/llm/         LLM layer (Dynamo's lib/llm): protocols/ (common types, OpenAI
                 chat+completions with delta generators/aggregators, SSE codec),
                 tokens (KV block hashing), tokenizers (interface, byte-level
                 reference backend, stop-sequence decoding), preprocessor
                 (HF chat templates via minja -> BackendInput), backend
                 (detokenize + stop conditions), engines (echo), model_card
                 (MDC + publish/discover), http/ (OpenAI frontend: SSE
                 streaming, model watcher, Prometheus metrics), kv_router/
                 (radix indexer, scheduler, publishers, KvRouter)
src/examples/    hello_world server+client, multi_instance routing demo, discoveryd
tests/           Catch2 suites (coro, runtime/cancellation, discovery, endpoint
                 e2e, and the llm_* suites for protocols/preprocessor/backend/
                 model cards/HTTP frontend/KV router)
```

## Build

Requires CMake ≥ 3.20, Ninja, and a C++20 compiler (tested: Apple clang 17).

```sh
git submodule update --init --depth 1 third_party/fmt third_party/spdlog
# toml11/Catch2/nlohmann_json/xxHash/minja are plain clones if not registered:
#   git clone --depth 1 https://github.com/catchorg/Catch2 third_party/Catch2
#   git clone --depth 1 https://github.com/nlohmann/json third_party/nlohmann_json
#   git clone --depth 1 https://github.com/ToruNiina/toml11 third_party/toml11
#   git clone --depth 1 https://github.com/Cyan4973/xxHash third_party/xxHash
#   git clone --depth 1 https://github.com/google/minja third_party/minja

cmake --preset default && cmake --build build          # Debug
cmake --preset asan    && cmake --build build-asan     # + ASan/UBSan
cmake --preset release && cmake --build build-release  # Release, -Werror
```

## Run the examples

Single process (in-process discovery, real TCP planes over loopback):

```sh
./build/src/examples/multi_instance   # two instances + round-robin client
./build/src/examples/echo_pipeline    # composable operators, local + remote
./build/src/examples/service_metrics  # custom stats + component-wide scrape
./build/src/examples/dynamo_bench     # codec/unary/streaming microbenchmarks
```

Multi process (via discoveryd):

```sh
./build/src/examples/discoveryd 7787 &
DYN_DISCOVERY=127.0.0.1:7787 ./build/src/examples/hello_world_server &
DYN_DISCOVERY=127.0.0.1:7787 ./build/src/examples/hello_world_client
# prints "hello world", streamed back one character at a time
```

Environment: `DYN_DISCOVERY` (discoveryd address; empty = in-process),
`DYN_HOST` (bind host for the control/data planes), `DYN_LOG`
(trace|debug|info|warn|error), `DYN_LOGGING_JSONL` / `DYN_LOGGING_DISABLE_ANSI`,
`DYN_RUNTIME_NUM_WORKER_THREADS`, `DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT`
(seconds; exceed it after a shutdown signal and the worker hard-exits with
code 911, as in Dynamo), `DYN_CONFIG` (TOML file layered under the env
vars — see `[runtime]` keys in `src/runtime/config.h`), and
`DYN_MAX_FRAME_MB` (inbound frame-size cap, default 256).

## Tests

```sh
ctest --test-dir build --output-on-failure
```

122 cases covering: coroutine primitives and utilities (deadline streams,
object pool, event resume hooks); cancellation hierarchy/callbacks/timed
waits; runtime startup/shutdown, task handles, config layering, and
external-executor runtimes; worker execution, single-worker enforcement, and
the 911 hard-exit path (subprocess); pipeline context metadata and operator
composition; codec hardening (malformed/corrupt/oversize frames); discovery
(create-if-absent, kv_put/create_or_validate, revisions, snapshot-then-watch
ordering, lease revocation, TTL expiry, reconnect + revision-aware watch
resync, pub/sub events); end-to-end endpoint behavior (streaming and unary
calls, many_in rejection, membership updates, watcher teardown, instance
enumeration, routing, distributed operator pipelines, stop propagation,
kill-on-abandoned-stream, error prologues and observable serve failures,
arrival timeouts, stats, component events, malformed-frame resilience); a
short soak under instance churn; and a multi-process integration test
(discoveryd + server + client, SIGKILL + TTL disappearance).

The LLM layer (65 of those cases, `[llm]`) covers: protocol round-trips with
Rust-serde-compatible JSON (FinishReason, BackendInput/Output, OpenAI chat +
completions, delta generation/aggregation from ported Rust fixtures); the SSE
codec (chunked feeds, [DONE], recorded OpenAI stream); token block hashing
against Rust golden values; the byte-level tokenizer, incremental Sequence
decoding of UTF-8 partials, and stop-sequence jailing; HF chat-template
rendering (minja); the preprocessor and backend operators end-to-end over the
echo engine; model deployment cards (local HF dirs, publish/list via
discovery kv, lease binding, from_mdc factories); the HTTP frontend (unary +
SSE chat completions, model listing, 404s, Prometheus metrics, and the model
watcher wiring a remote worker through discovery); and the KV router (radix
indexer store/remove/evict, scheduler cost selection, and an end-to-end
prefix-affinity test over live pub/sub + metrics scrapes).

The suite runs in Debug, ASan+UBSan, and Release (`-Werror`) locally and in
CI (`.github/workflows/ci.yml`, macOS + Linux).
