# SPDX-License-Identifier: Apache-2.0
#
# The dynamo library: runtime core, pipeline, discovery, transports, component.

add_library(dynamo STATIC
  runtime/cancellation.cpp
  runtime/config.cpp
  runtime/executor.cpp
  runtime/logging.cpp
  runtime/runtime.cpp
  runtime/timer.cpp
  runtime/worker.cpp
  component/distributed_runtime.cpp
  discovery/in_process.cpp
  discovery/server.cpp
  discovery/tcp.cpp
  transports/codec.cpp
  transports/control_plane.cpp
  transports/data_plane.cpp
  transports/socket.cpp
)

target_include_directories(dynamo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dynamo
  PUBLIC spdlog::spdlog fmt::fmt nlohmann_json::nlohmann_json
  PRIVATE dynamo_warnings toml11::toml11)
target_compile_features(dynamo PUBLIC cxx_std_20)

find_package(Threads REQUIRED)
target_link_libraries(dynamo PUBLIC Threads::Threads)

if(DYNAMO_WITH_ETCD)
  target_sources(dynamo PRIVATE discovery/etcd.cpp)
  # PUBLIC define: consumers (backend selection, tests) compile etcd paths
  # conditionally on it. The etcd headers stay private to etcd.cpp (pimpl).
  target_compile_definitions(dynamo PUBLIC DYNAMO_HAVE_ETCD)
  target_link_libraries(dynamo PRIVATE etcd-cpp-api)
endif()

# The LLM layer (protocols, tokens; grows with M2+): mirrors Dynamo's
# lib/llm crate as a separate library on top of the runtime.
add_library(dynamo_llm STATIC
  llm/protocols/openai.cpp
  llm/protocols/sse.cpp
  llm/preprocessor/prompt.cpp
  llm/preprocessor.cpp
  llm/backend.cpp
  llm/engines.cpp
  llm/model_card.cpp
  llm/http/http_server.cpp
  llm/http/service.cpp
  llm/http/model_watcher.cpp
  llm/kv_router/indexer.cpp
  llm/kv_router/scheduler.cpp
  llm/kv_router/publisher.cpp
  llm/kv_router/router.cpp
  llm/tokenizers.cpp
  llm/tokens.cpp
)
target_link_libraries(dynamo_llm
  PUBLIC dynamo
  PRIVATE dynamo_warnings xxHash::xxhash minja::minja)
