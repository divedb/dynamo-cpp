add_library(dynamo_runtime
    src/config.cpp
    src/runtime.cpp
    src/component.cpp
    src/discovery.cpp
    src/pipeline_context.cpp
    src/pipeline_nodes.cpp
    src/engine.cpp
    src/transport_etcd.cpp
    src/transport_grpc.cpp
    src/transport_tcp.cpp
)

target_include_directories(dynamo_runtime
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(dynamo_runtime
    PUBLIC
        fmt::fmt
        spdlog::spdlog
        toml11::toml11
        nlohmann_json::nlohmann_json
        absl::flat_hash_map
        absl::strings
        Folly::folly
        asio::asio
        dynamo_protos
    PRIVATE
        grpc++
)

# Suppress some folly warnings
target_compile_options(dynamo_runtime PRIVATE
    -Wno-unused-parameter
    -Wno-deprecated
)
