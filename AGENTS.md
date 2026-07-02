# Dynamo C++20 — Agent Instructions

## Build
```bash
mkdir build && cd build
cmake -G Ninja ..
cmake --build .
ctest --output-on-failure
```

## Code conventions
- C++20, `pragma once`, no header guards
- Namespace `dynamo` for runtime, `dynamo::llm` for LLM, `dynamo::http` for HTTP
- Headers in `include/dynamo/` matching include path
- Folly for async (CPUThreadPoolExecutor, coro::Task, SemiFuture)
- spdlog for logging (no cout/printf)
- Abseil for containers (flat_hash_map, strings)
- nlohmann/json for JSON
- gRPC+protobuf for RPC
- asio for low-level TCP

## Testing
- Catch2 v3 for all tests
- Tests in `tests/<module>_tests/` directory
- Test targets named `<module>_tests`
- Link against Catch2::Catch2WithMain

## Submodules
```bash
./setup.sh  # Initializes all third-party submodules
```

## Design deviations from NVIDIA Dynamo
1. gRPC replaces NATS as RPC transport
2. Folly replaces tokio (C++ equivalent)
3. In-memory etcd for development/testing
4. No Python/C bindings
5. Simplified config (env+TOML, no figment)
6. No NIXL support (proprietary NVIDIA library)
