# TODO — dynamo-cpp vs. Dynamo gap analysis

Comparison of this implementation against `third_party/dynamo` (Rust, v0.1.0,
commit `3983830e80` — the same commit as the `~/Documents/git/dynamo`
checkout). Sections 1–7 cover `lib/runtime` parity (complete except P3s);
section 8 plans the `lib/llm` layer as milestones; section 9 lists what stays
out of scope; section 10 records what the fresh audit found (open gaps +
verified dead code in the Rust reference).
Each task names the Rust reference and the C++ module it lands in.
Priorities: **P0** = core behavioral parity, **P1** = functional parity, **P2** = robustness/scale, **P3** = optional/out-of-scope candidates.

**Last full re-audit: 2026-07-09.** Every `.rs` file under `lib/runtime` and
`lib/llm` was mapped to a TODO entry, a C++ module, a §9 out-of-scope item, or
the §10.2 dead-code list; completed claims were spot-checked against both
source trees; test suite green at 122/122 (Debug).

---

## 1. Execution model & pipeline (`src/pipeline/`)

- [x] **P0 — Unary responses (`SingleIn → SingleOut`).** *(done)*
  `pipeline::make_unary_engine<Req,Resp>(fn)` adapts a value-returning handler
  into a one-item-stream engine; `Client::unary()` dispatches with
  `response_type: single_out` and resolves to the single value;
  `EngineIngress` cuts the stream after the first item for unary requests
  (works against streaming engines too). Note: a distinct `SingleOut` engine
  *type* was not added — Dynamo's own network layer only ships
  `SingleIn/ManyOut` handlers, so unary is a wire/consumption shape here, not
  a separate engine hierarchy. Tests: `[endpoint][unary]` ×2.

- [x] **P1 — Context registry and stage tracking.** *(done)*
  `pipeline::ContextRegistry` (typed shared + take-once unique entries),
  `Context::registry()`, `stages()/add_stage()`, and `map()` — all carried
  across `transfer()`/`map()`. Tests: `[pipeline][context]`.

- [x] **P1 — Composable pipeline node graph.** *(done)*
  `pipeline/operators.h`: `Operator` (bidirectional transform with the
  downstream engine as its forward edge), `link()` composing operators into
  plain engines, `make_map_operator` (PipelineNode role, records its stage on
  the context trail). Segment boundaries are `Client::as_engine()` (caller
  side) + `Endpoint::serve()` (worker side) — validated locally, multi-stage,
  and across the network. Example: `echo_pipeline`; tests:
  `[pipeline][operators]`, `[endpoint][operators]`. Note: kept engine-shaped
  rather than porting Rust's Source/Sink/Edge trait machinery — same
  composition power, idiomatic to our AsyncEngine model.

- [x] **P2 — `ManyIn` request streams.** *(done: explicit rejection)*
  Decision: not implemented (matching Dynamo, whose handler is a stub), but
  now rejected loudly at both layers — the ingress fails the call fast via a
  prologue error ("request_type 'many_in' is not supported"), and the data
  plane rejects Request-type call-home handshakes. Test: `[endpoint][manyin]`.

- [x] **P2 — Stream utilities.** *(done)*
  `coro/utils.h`: `until_deadline`/`until_timeout` generator adaptors
  (checked between items, matching the Rust adaptor's poll semantics) and
  `Pool<T>` with RAII `Item` handles, `on_return` reset, and `take()`
  detachment. Tests: `[coro][utils]`.

## 2. Component & service layer (`src/component/`)

- [x] **P0 — Cluster-wide stats collection (`scrape_stats(duration)`).** *(done)*
  `Component::scrape_stats(timeout)` enumerates all live instances of the
  component from discovery, fans out concurrent control-plane stats queries
  (each bounded by `timeout` via SO_RCVTIMEO on the ack read), and returns a
  `ServiceSet` of per-instance `EndpointStats` (unreachable instances are
  reported per-entry, not thrown). Test:
  "component-wide stats scrape aggregates all instances".
  Follow-up completed with the P1 batch: `examples/service_metrics` ported
  (custom stats handler + component-wide scrape, printed per instance).

- [x] **P1 — Stop instance watchers when the last client is gone.** *(done)*
  The registry now holds `weak_ptr<InstanceSource>`; the last client dropping
  the source closes the watch receiver (ending the fold loop and the
  server-side watch), and a later client recreates it. Test:
  "instance watch closes when the last client drops".

- [x] **P1 — Service/endpoint description metadata.** *(done)*
  `serve()` now takes `ServeOptions{lease, stats_handler, description,
  version}`; description/version are registered in `EndpointInfo` (parsing is
  default-tolerant) and visible via `Client::instances()` / scrapes. Test:
  `[endpoint][metadata]`.

- [x] **P2 — `Component::list_endpoints` / instance enumeration API.** *(done)*
  `Component::list_instances()` returns every live `EndpointInfo` across the
  component's endpoints; `scrape_stats` now builds on it. Its test tripped a
  real bug: fluent-chain member coroutines dangled `this` when called on
  temporaries (`component.endpoint("x").serve(...)`) — the whole fluent
  surface (serve/client/publish/subscribe/scrape/list/instance_source/
  primary_lease) now copies handles into free-coroutine frames eagerly.

- [x] **P3 — Shared work-queue dispatch (queue groups).** *(done)*
  NATS-faithful queue-group semantics as a broker-side op (note: this
  exceeds Rust parity — `egress/queue.rs` is a 14-line placeholder there):
  `Discovery::queue_dispatch(subject, payload)` delivers to exactly ONE
  current subscriber of the subject, round-robin, dropping dead members and
  trying the next; an empty group fails fast with a NATS-style
  "no responders" error (nothing is buffered — at-most-once, like NATS).
  Both backends implement it (in-process store + discoveryd
  `queue_dispatch` op); workers reuse the existing event-subscription
  machinery, so reconnect re-registration comes for free. Component layer:
  `ServeOptions::queue_group` subscribes the instance to
  `Endpoint::queue_subject()` ("{ns}/{comp}/{endpoint}:queue") and pumps
  broker-balanced envelopes into the same `EngineIngress` as
  instance-addressed dispatches; `Client::queue()` ships the
  {control, payload} envelope through the broker and then follows the
  normal call-home/arrival path (a worker that dies after pickup surfaces
  as the arrival timeout). Tests: `[discovery][queue]` ×2 (both backends:
  round-robin, dead-member skip, no-responders) and `[endpoint][queue]`
  e2e (even 4/4 split across two instances, survivor takes over after a
  lease revoke, direct dispatch still works alongside). 131/131 in
  Debug/ASan/Release.

## 3. Local runtime & worker (`src/runtime/`)

- [x] **P0 — Enforce single Worker per process.** *(done)*
  Constructing a second Worker while one is alive throws (process-wide atomic
  slot claimed before any pools are created); `execute()` is one-shot
  (`std::logic_error` on reuse); Worker is non-copyable/non-movable.
  Deliberate deviation from Rust: a new Worker may be created after the
  previous one is destroyed (our runtimes are self-contained; Rust's
  process-global tokio runtime forces ever-once semantics). Tests:
  "only one worker may be alive; execute is one-shot".

- [x] **P1 — Config layering (TOML + env).** *(done)*
  `DYN_CONFIG` names a TOML file; `config_or(table, key, fallback)` layers
  `[runtime]` (worker_threads, background_threads, graceful_shutdown_timeout_s)
  under env overrides; `env_is_truthy` ported. Test: `[runtime][config]`.
  `[distributed]` table keys can be added the same way as needed.

- [x] **P1 — Task handles (`runnable.rs::ExecutionHandle`).** *(done)*
  `Runtime::spawn`/`spawn_background` return a `TaskHandle` (finished(),
  awaitable `join()` rethrowing failures, blocking `sync_join(timeout)`).
  serve() registration failures are now observable ("serve failures are
  observable through the task handle" test). Cancellation stays cooperative
  via tokens (no abort-by-handle — matches our cancellation model). Fixing
  this also surfaced and fixed a real teardown race: the spawned frame's
  tracker guard fired before the inner task frame was destroyed, allowing a
  pool thread to hold the last Runtime reference and self-join in
  ~ThreadPool.

- [x] **P1 — Structured logging parity (partial).** *(done, descoped)*
  `DYN_LOGGING_JSONL` (JSONL pattern; note: message bodies are not
  JSON-escaped — custom formatter is future work) and
  `DYN_LOGGING_DISABLE_ANSI`; `DYN_LOG` unchanged. Per-module level filters
  deliberately descoped: all call sites use spdlog's default logger, so
  module-scoped filtering needs a logger-per-module refactor first (folded
  into P2 if ever needed).

- [x] **P2 — External-executor runtimes.** *(done)*
  `Executor` interface (post + schedule awaitable); `ThreadPool` implements
  it; `ExternalExecutor` adapts an application post-fn;
  `Runtime::from_executors(primary, secondary = primary)`. All internal APIs
  (resume_on, wait_for, spawn_detached, signal pools) now take `Executor&`.
  Test: `[runtime][executors]`.

- [x] **P2 — Inline-resume policy for events/hooks.** *(done)*
  `AsyncEvent::set_resume_hook` routes waiter resumption;
  `Controller::set_signal_executor(post)` routes stop/kill event waiters AND
  registered hooks; `EngineIngress` sets it to the primary pool, so engines
  awaiting `stopped()/killed()` never run on transport threads.
  `CancellationToken` callbacks remain inline by design (they are used for
  lightweight teardown like channel closes; documented). Test:
  "async event resume hook routes waiters".

## 4. Discovery (`src/discovery/`)

- [x] **P0 — Keep-alive retry-until-deadline semantics.** *(done)*
  `keep_alive_loop` now tracks `deadline = now + ttl`, resets it on every
  successful heartbeat, and retries failures at `min(ttl/4, 500ms)` until the
  deadline passes; only then (or on an authoritative "lease expired" reply)
  is the token cancelled. Connection loss no longer insta-cancels leases
  (`on_connection_lost` drops pending/watches only) — the deadline governs,
  mirroring the server-side TTL reaper. Full validation of the
  survive-a-transient-blip path needs client reconnect (P1 below); the
  TTL-expiry test still covers the lapse path.

- [x] **P1 — Discovery client reconnect.** *(done)*
  A manager thread owns the connection: reconnects with backoff (200ms→2s),
  re-registers watches and event subscriptions, and resyncs watches via the
  server's snapshot + sync-marker protocol (replayed puts are idempotent;
  deletes missed during the outage are synthesized by diffing the folded key
  set against the snapshot; stale sync markers are ignored). Leases ride
  through blips shorter than their TTL via the P0 keep-alive deadline.
  Tests: "[discovery][tcp][reconnect]" ×2 (resync correctness incl. a delete
  during the outage; lease survival through a blip).

- [x] **P2 — Watch revisions.** *(done)*
  Monotonic store revision in both backends; `KeyValue.mod_revision` carried
  in events and `get_prefix`; discoveryd keeps a bounded event log (4096) and
  supports `watch(prefix, from_rev)`: incremental replay when the log covers
  the gap, snapshot + diff fallback otherwise (sync markers now carry
  `mode: replay|snapshot`). Reconnecting clients resync from their last seen
  revision automatically. `prev_key` values are not retained (no user yet).
  Tests: `[discovery][kv]`, reconnect suite.

- [x] **P2 — `kv_put` / `kv_create_or_validate`.** *(done)*
  Both ops in the `Discovery` interface, the in-process store, discoveryd,
  and the TCP client. `kv_put` upserts and rebinds the lease (revocation then
  deletes the key); create_or_validate succeeds iff absent or identical.
  Tests: `[discovery][kv]` ×2 (both backends).

- [ ] **P3 — Real etcd/NATS backends.**
  `Discovery` is an interface precisely so an `EtcdDiscovery` (grpc + protobuf
  submodules are already declared in `.gitmodules`) can slot in later; same
  for a NATS-based control plane. Large dependency cost; only if deployment
  requires interop with an existing etcd/NATS cluster.

- [ ] **P3 — discoveryd HA/persistence.**
  Single-node, in-memory only. Snapshot/restore or raft is out of scope for
  now; document it as a deliberate limitation in docs/architecture.md.

## 5. Transports (`src/transports/`)

- [x] **P1 — Control-plane connection reuse.** *(done)*
  The server now serves multiple requests per connection; the client keeps a
  per-target idle pool (max 8), with stale pooled connections detected via a
  retry-once-with-fresh-connection path. Idle-timeout eviction left to P2
  (bounded pool size caps the cost).

- [x] **P1 — Timeouts on the request path.** *(done)*
  `CallOptions{dispatch_timeout (ack, default 5s), arrival_timeout (call-home,
  default 30s)}`; the arrival timeout injects a synthetic StreamArrival via a
  timer and deregisters the pending subject, failing generate() with a clear
  error. Test: "arrival timeout fails the call when a worker acks but never
  calls home". Remaining nuance for P2: the TCP connect() itself is still
  blocking without its own deadline (local networks make this a non-issue
  today).

- [x] **P1 — Configurable stream buffering (`StreamOptions`).** *(done)*
  `CallOptions::recv_buffer_count` (default 64) flows through
  `Client` → `register_response_stream` → the per-stream channel.
  `send_buffer_count` has no equivalent here (worker-side sends are direct
  framed writes with socket backpressure, no intermediate queue).

- [x] **P2 — Data-plane connection sharing.** *(resolved by measurement: keep as-is)*
  `dynamo_bench` (Release, loopback): unary calls with FULL per-call stream
  setup (dispatch + call-home connect + teardown) run at ~4.1k req/s with
  p50 239µs / p99 315µs; single-stream throughput ~159k items/s (40.7 MB/s
  at 256B items); codec ~366k msg/s. Stream setup is sub-millisecond and not
  a bottleneck for inference-shaped workloads — per-stream connections stay
  (simpler failure isolation). Revisit only if a workload needs >>4k new
  streams/s per node.

- [x] **P2 — Codec hardening.** *(done)*
  Inbound cap configurable via `Socket::set_max_frame_bytes` /
  `DYN_MAX_FRAME_MB` (default 256 MiB). `tests/codec_test.cpp` covers
  roundtrips (incl. binary payloads) and malformed input (truncated prelude/
  body, trailing garbage, corrupt checksum, corrupt length, oversize); an
  endpoint test feeds raw garbage to a live control plane and verifies the
  server keeps serving.

- [ ] **P3 — TLS/authentication.**
  Nothing on any socket (Dynamo delegates this to etcd/NATS deployment
  options). Would need a TLS wrapper around `Socket` and token auth on
  discoveryd/control plane.

- [ ] **P3 — ZMQ-style transport (`transports/zmq.rs`, 418 lines).**
  Alternative data-plane transport in Rust (used by some deployments). Our
  transport interfaces would host it as another `StreamSender`/receiver pair;
  skip unless a concrete need appears.

## 6. Eventing (`src/traits` equivalent — missing entirely)

- [x] **P1 — `EventPublisher`/`EventSubscriber`.** *(done)*
  `Discovery::publish/subscribe` (exact-subject transient events) implemented
  in both backends — discoveryd carries subscribe/unsubscribe/publish ops and
  fans out to subscribers; subscriptions survive reconnects (re-registered by
  the manager thread). `Namespace`/`Component` expose
  `publish(name, json)` / `subscribe(name)` on Dynamo's subject scheme
  (`{ns}.events.{name}` / `{ns}.{comp}.events.{name}`). Tests:
  `[discovery][events]` ×2, `[endpoint][events]`.

## 7. Tests, examples, tooling

- [x] **P1 — Multi-process integration test.** *(done)*
  `tests/integration/multi_process.sh` (ctest: `multi_process_integration`,
  label `integration`, serial): discoveryd + server + client as separate
  processes; asserts registration (via the new `instance_ls` tool), the
  streamed "hello world", then SIGKILLs the server and asserts TTL-based
  disappearance (~12s).

- [x] **P1 — Port Dynamo's `tests/soak.rs`.** *(done, CI-sized)*
  `tests/soak_test.cpp` ([soak]): 8 concurrent clients × 25 streaming
  requests round-robining over two stable instances while a third instance
  churns (serve → revoke ×6). Asserts zero payload corruption and full
  accounting (completions + clean routing errors). Runs in ~2s, so it stays
  in the default suite; scale the constants up for a real soak run.

- [x] **P2 — Graceful-shutdown timeout (911) subprocess test.** *(done)*
  `tests/helpers/hang_worker` + `integration/graceful_timeout.sh` (ctest:
  `graceful_timeout_911`): SIGTERM with a 1s window, asserts exit status 143
  (Unix truncates 911 to 8 bits — same observable as Rust's worker).

- [x] **P2 — Throughput/latency microbenchmarks.** *(done)*
  `dynamo_bench` example binary: codec encode+decode, unary req/s with
  latency percentiles, and single-stream item throughput. Baseline numbers
  recorded in the data-plane-sharing entry above.

- [x] **P2 — CI workflow.** *(done, unverified on Linux)*
  `.github/workflows/ci.yml`: {macos-14, ubuntu-24.04} × {default, asan,
  release} presets; fetches the non-gitlink deps explicitly (the dynamo
  reference repo is not needed to build). The code is POSIX-portable by
  construction but has only been executed on macOS here — expect the first
  Linux CI run to shake out minor include/flag nits.

## 8. LLM layer milestones (`src/llm/`)

Port of `third_party/dynamo/lib/llm` (~25k lines Rust) onto the v2 component
layer, in dependency order. Partial v1 ports (protocols, tokenizer, router,
backend, HTTP server) exist in git history (`lib/llm`, `lib/http`, deleted
2026-07-03) — mine them for reference, but they predate the v2 pipeline
model and are not drop-in.

### M1 — Protocols (`src/llm/protocols/`) — DONE

- [x] **Common LLM types.** *(done)* `llm/protocols/common.h` (FinishReason
  with Rust-serde-compatible JSON, StopConditions, SamplingOptions,
  OutputOptions) and `llm/protocols/llm_backend.h` (PreprocessedRequest =
  BackendInput, LLMEngineOutput, BackendOutput). Providers are plain
  functions (`extract_sampling_options`/`extract_stop_conditions`) rather
  than trait objects. Tests: `[llm][common]`.
- [x] **OpenAI chat/completions types.** *(done)* `llm/protocols/openai.{h,cpp}`:
  NvExt (+validation), chat + legacy-completions requests/responses
  (tolerant parsing: missing/null accepted, unknown fields ignored,
  string-or-array `stop`, multi-part content flattened), ChatDeltaGenerator /
  CompletionDeltaGenerator, ChatDeltaAggregator / CompletionDeltaAggregator.
  Deviations from Rust v0.1.0 (each commented at the site): completions
  `stop` is honored (Rust drops it), usage.total_tokens is computed (Rust
  leaves 0), assistant-role-on-every-chunk quirk preserved. Tests:
  `[llm][openai]` incl. ports of the Rust aggregator tests.
- [x] **SSE codec.** *(done)* `llm/protocols/sse.{h,cpp}`: incremental
  SseDecoder (feed/finish), comment/id/event/data fields, multi-line data,
  `[DONE]` dropped, CRLF tolerated; `encode_sse` for the M5 frontend;
  `annotated_from_sse<R>` (error events → error annotations). Tests:
  `[llm][sse]` incl. an abbreviated recorded OpenAI stream fixture.
- [x] **Token types.** *(done)* `llm/tokens.{h,cpp}`: XXH3-64 seed 1337 over
  LE token bytes (vendored xxHash, header-only), TokenBlock/TokenSequence
  with sequence-hash chaining. Golden hashes from the Rust test verified.
  Quirk preserved: first split block's seq hash = block hash. Deviations:
  empty input allowed (Rust panics); pushed blocks advance the partial
  block's parent hash (Rust chains every pushed block to a stale parent).
  Tests: `[llm][tokens]`.

### M2 — Tokenizers + preprocessor (`src/llm/tokenizers.*`, `src/llm/preprocessor*`) — DONE

- [x] **Tokenizer interface (+reference backend).** *(done; HF backend
  descoped)* `llm/tokenizers.{h,cpp}`: Encoding, abstract Tokenizer,
  Sequence (incremental decode, U+FFFD partials held, prefix/read offsets
  ported exactly), StopSequenceDecoder (visible/hidden stop tokens, hidden
  stop sequences with Held jail). Backend: ByteLevelTokenizer (byte ids +
  <s>/</s> specials, lossy UTF-8 decode) — self-contained and exercises the
  real partial-character paths; a HuggingFace tokenizer.json backend stays
  descoped until a real engine integration needs it (vendoring
  tokenizers-cpp pulls a Rust toolchain into the build). Tests:
  `[llm][tokenizer]`.
- [x] **Prompt formatting.** *(done)* Vendored google/minja (header-only,
  SYSTEM include). `llm/preprocessor/prompt.{h,cpp}`: PromptFormatter,
  HfChatTemplateFormatter (chat_template + bos/eos), `chat_template_input`
  from chat requests (add_generation_prompt iff last turn is user) and
  legacy completions (single user turn). Template-corpus validation still
  worthwhile when real models land. Tests: `[llm][prompt]`.
- [x] **Preprocessor operator.** *(done)* `llm/preprocessor.{h,cpp}`:
  OpenAIPreprocessor implements both pipeline Operators (chat + legacy
  completions): render (nvext.use_raw_prompt honored) → tokenize → extract
  sampling/stop options → merge model EOS ids into hidden stops →
  apply_ignore_eos → BackendInput with mdcsum; response side maps
  BackendOutput deltas through the M1 delta generators, prepends
  formatted_prompt/token_ids annotations, and on generator error emits an
  error annotation, stops the context, and closes the stream. Tests:
  `[llm][preprocessor]` incl. operator e2e via `link()` against an inline
  backend and the error path.

### M3 — Backend + engine interface (`src/llm/backend.*`, `src/llm/engines.*`) — DONE

- [x] **Backend operator.** *(done)* `llm/backend.{h,cpp}`: Decoder
  (incremental detokenization via Sequence, hidden stop tokens/sequences,
  min_tokens gate) + Backend Operator (BackendInput→BackendOutput over an
  ExecutionContext engine): pass-through for events and text-carrying
  engine deltas, decode+stop-check for token-only deltas, stop_generating()
  when the decoder stops a stream the engine did not finish. Deviations
  from Rust v0.1.0 (commented): partial stop-sequence text is jailed and
  never leaks (Rust streams it and only cuts after the fact — its own TODO),
  jailed text is flushed on finish, the engine's finish reason survives when
  the decoder found none (Rust overwrites it, dropping token-only Length),
  and the stream is cut locally after a stop trigger. Tests:
  `[llm][backend]`.
- [x] **Engine seam + echo engine.** *(done)* `llm/engines.{h,cpp}`:
  ExecutionContext/ExecutionOutputStream aliases (the engine seam real
  adapters plug into) and make_echo_engine_core() (dynamo-run echo_core:
  one token-only delta per request token + a stop delta; extended to honor
  max_tokens with a length finish and context stop). Full local pipeline
  test: preprocessor → backend → echo, aggregated to a unary chat response.

### M4 — Model deployment cards (`src/llm/model_card.*`) — DONE

- [x] **MDC model + create.** *(done)* `llm/model_card.{h,cpp}`:
  ModelDeploymentCard with serde-compatible JSON (externally-tagged
  artifacts, RFC3339 last_published, revision not serialized),
  `from_local_path` (config.json + tokenizer.json required,
  tokenizer_config.json optional), ModelInfo from HF config.json
  (int-or-array eos), HfTokenizerConfig (chat_template string/named-list,
  string-or-AddedToken bos/eos), slug + expiry (5 min). Deviations:
  slug/mdcsum hash with XXH3 instead of blake3 (checksums were never
  cross-implementation comparable); hf-hub downloading descoped (local
  paths only). Extension: `byte_level` tokenizer kind so full pipelines run
  without an HF tokenizer backend. Factories: OpenAIPreprocessor::from_mdc,
  Backend::from_mdc (+ tokenizer_from_mdc/formatter_from_mdc). Tests:
  `[llm][mdc]`.
- [x] **Publish/discover MDCs.** *(done)* `publish_model_card` (kv_put of
  mdc/{slug} bound to the worker lease, stamps last_published, bumps
  revision) and `list_model_cards` (prefix get, malformed entries skipped)
  over the existing Discovery interface; cards disappear with their lease
  (tested). The M5 frontend watches the same prefix; `instance_ls`-style
  listing tooling can come with M5's example.

### M5 — HTTP frontend (`src/llm/http/`) — DONE

- [x] **HTTP/1.1 + SSE server.** *(done)* `llm/http/http_server.{h,cpp}`:
  in-tree minimal HTTP/1.1 on the Socket/Listener stack (no new deps) —
  exact-path routing, keep-alive, Content-Length bodies (32 MiB cap),
  chunked streaming responses with a disconnect-aware StreamSink,
  thread-per-connection with tracked fds joined on stop.
- [x] **OpenAI routes.** *(done)* `llm/http/service.{h,cpp}`:
  /v1/chat/completions + /v1/completions (SSE streaming via the M1 codec, or
  unary via the M1 aggregators), /v1/models, /health; 400/404/500 error
  JSON; uuid request ids; client disconnect stops the request context.
  Deviations from Rust v0.1.0 (commented): nvext is preserved (Rust drops
  it, killing annotations over HTTP); model listing uses object:"model"
  (Rust emits the literal "object").
- [x] **Model watch + routing.** *(done)* `llm/http/model_watcher.{h,cpp}`:
  ModelEntry ({name, endpoint, model_type}) under "{prefix}{type}/{name}",
  `register_model_entry` (llmctl-style, lease-bound), `run_model_watcher`
  (prefix watch → component Client::as_engine registered in ModelManager;
  runtime-shutdown token closes the watch). e2e test: worker serve + entry
  publish → watcher wires it → HTTP request round-trips through the planes.
- [x] **Service metrics.** *(done)* Metrics with the Rust label scheme
  (requests_total{model,endpoint,request_type,status}, inflight gauge,
  duration histogram) + hand-rolled Prometheus text exposition on /metrics;
  RAII InflightGuard mirrors the Rust one. Tests: `[llm][http]` (4 cases,
  ASan-clean).

### M6 — KV-aware routing (`src/llm/kv_router/`) — DONE

- [x] **KV event protocols + publisher.** *(done)*
  `kv_router/protocols.h`: serde-compatible KvCacheEvent (externally-tagged
  stored/removed), ForwardPassMetrics, RouterEvent, KVHitRateEvent,
  `compute_block_hashes_for_seq` (same XXH3-1337 primitive as tokens.h,
  golden values verified). `kv_router/publisher.{h,cpp}`: KvEventPublisher
  (component pub/sub on kv_events) and KvMetricsPublisher (latest metrics +
  load_metrics endpoint whose stats handler feeds scrapes; per-worker lease
  support).
- [x] **Radix indexer.** *(done)* `kv_router/indexer.{h,cpp}`: RadixTree
  (per-worker attribution, engine-hash jump table, parent-based store,
  remove with children-clear, worker eviction, optional recent-use
  frequency tracking) + KvIndexer. Deviation: mutex-guarded tree instead of
  Rust's dedicated thread + channels (its tree is !Send). Tests:
  `[llm][kv][indexer]` (store/match/extend/unknown-parent/remove/evict).
- [x] **Scheduler + scoring.** *(done)* `kv_router/scheduler.{h,cpp}`: cost =
  alpha·load_deviation + (1-alpha)·normalized_new_tokens +
  gamma·request_load_ratio with balance mode; capacity exclusion;
  NoEndpoints/AllWorkersBusy errors. Deviations fixing Rust's own FIXMEs
  (commented): worker_ids aligned with endpoints (Rust's HashSet reorder
  misindexes), load stats over kv ratios (Rust mixes ratio and absolute
  units). `kv_router/router.cpp`: metrics aggregator loop on scrape_stats
  (custom stats under "custom"; worker id = instance id, no subject
  parsing).
- [x] **Router integration.** *(done)* `kv_router/router.{h,cpp}`: KvRouter
  owning indexer + scheduler with spawnable event-consumer and
  metrics-aggregator tasks (runtime-shutdown aware); `schedule(token_ids)`
  → hash → overlap → worker id, publishing kv-hit-rate events. The chosen
  id feeds `Client::direct`. e2e test: two live metrics endpoints, worker B
  publishes the prompt's blocks, router routes the matching prompt to B
  (5/5) and still schedules cold prompts; eviction clears affinity. Tests:
  `[llm][kv]` — all green in Debug, ASan, Release.

## 9. Out of scope (tracked, no milestone)

- **KV block manager + CUDA kernels** (`lib/llm/src/kv/`, 5.5k lines +
  `block_copy.cu`) — engine-side memory management; needs CUDA and a real
  engine to matter. Revisit only alongside a native engine integration.
- **Real engine adapters** (`engines/`: vllm, sglang, trtllm, llamacpp,
  mistralrs, python — 4.9k lines) — each drags in an SDK or Python interop;
  M3's engine interface is the seam they'd plug into.
- **Disaggregated prefill/decode router** (`disagg_router.rs`, 259 lines) —
  small, but only meaningful with real engines + KV transfer (NIXL).
- **Python/C bindings** (`lib/bindings`) — needs a stable C ABI over the
  component layer first; Python is Dynamo's main extensibility surface, so
  this is the first thing to reconsider if others build on dynamo-cpp.
- **Launch/deploy tooling** (`dynamo-run`, `llmctl`, `components/http`,
  `components/metrics`, `deploy/` SDK + k8s operator) — product packaging,
  not library parity. (`components/http` is the standalone frontend binary
  over `http/service`; our `src/examples` + M5 service cover the library
  surface it exercises.)

## 10. Re-audit findings (2026-07-09)

Fresh file-by-file sweep of the Rust tree against `src/`. Everything below is
either a newly identified open gap (10.1) or Rust code verified to be
unreachable in v0.1.0 and therefore deliberately not ported (10.2).

### 10.1 Open gaps

- [x] **P2 — Tool-calling chat-template plumbing** *(done)*
  (`preprocessor/prompt/template/oai.rs`, `formatters.rs`).
  `NvCreateChatCompletionRequest` now carries `tools`/`tool_choice` (raw
  JSON, shape-checked: mistyped fields are rejected with a 400 rather than
  fed to the template); `ChatMessage` carries `tool_calls`/`tool_call_id` so
  multi-turn tool conversations render faithfully; `chat_template_input()`
  forwards tools; `HfTokenizerConfig` keeps both `default` and `tool_use`
  entries of a named-template list; `HfChatTemplateFormatter` renders the
  `tool_use` template whenever the request carries tools (Rust's
  `env["tool_use"]` selection). Deviations (commented at the sites):
  (a) Rust's named-list registration is broken at v0.1.0 — it registers each
  map's raw k/v pairs, i.e. templates literally named "name"/"template", so
  `get_template("default")` can never resolve; we implement the HF
  convention it intended. (b) When tools are present but the config has no
  distinct `tool_use` template, Rust errors; we render `default` with tools
  in the context (minja's tool polyfills apply), matching the string-template
  case. Response-side tool-call *parsing* remains out of scope
  (`ToolCallingMatcher` is dead code — see 10.2). Tests: `[llm][openai]`
  round-trip, `[llm][prompt]` ×2, `[llm][mdc]` named-list e2e; 126/126 in
  Debug/ASan/Release.

- [x] **P3 — Prompt context mixins (`llama3_datetime`)** *(done)*
  (`preprocessor/prompt/template/context.rs`, `model_card/model.rs`).
  `HfChatTemplateFormatter` now takes the MDC's `prompt_context` mixin names
  (wired through `formatter_from_mdc`): `llama3_datetime` injects `datetime`
  (UTC now as "%d, %B, %Y", locale-independent month names, computed per
  render like Rust's `Utc::now()` in the mixin) via minja's `extra_context`;
  `oai_chat` is a recognized no-op (as in Rust, where only the datetime
  mixin is implemented); unknown names throw at formatter construction
  (Rust rejects them at MDC deserialization). Tests: `[llm][prompt]` mixin
  case, `[llm][mdc]` card→formatter e2e; 128/128 in Debug/ASan/Release.
  Not ported: `unk_token` in the context — minja's chat_template takes only
  bos/eos; no known template needs it (revisit with a real model corpus).

- **Behavioral difference (documented, no action): unmodeled OpenAI fields
  are dropped, not preserved.** Rust wraps the full `async-openai` 0.27
  request/response types, so fields its pipeline never reads (`logit_bias`,
  `response_format`, `functions`, …) still parse strictly and re-serialize.
  Our hand-modelled types (`protocols/openai.h`) parse tolerantly and drop
  unknown fields. Net behavior is identical for every field the Rust
  pipeline actually consumes (all modeled here); the differences are
  (a) type errors in unused fields are diagnosed by Rust but ignored here,
  and (b) middleware that expects unused fields echoed back won't get them.
  Revisit only if (a) or (b) bites. (`tools`/`tool_choice` were the one
  unused field pair with a real consumer — the template context — and are
  now modeled; see the completed P2 above.)

### 10.2 Dead code in Rust v0.1.0 — verified non-gaps, do not port

Each of these is defined but never referenced anywhere in the Rust workspace
(`lib/`, `components/`, `launch/`) at this commit:

- `preprocessor/tools/` (~200 lines): `ToolCallingMatcher` + tool-call
  request/response types — response-side tool-call extraction, unused.
- `protocols/common/postprocessor.rs`: `PostprocessedResponse` — unused.
- `common/versioned.rs`: `Versioned` trait (NATS atomic-update hook) — unused.
- `kv_router/worker.rs`: `KvRoutedIngress` — NATS-service ingress wrapper,
  unused (the kv_router pipeline goes through the normal component ingress).
- `pipeline/network/egress/queue.rs`: 14-line placeholder (already tracked
  as the §2 queue-groups P3).
