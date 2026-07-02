cmake_minimum_required(VERSION 3.22)
project(dynamo VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- Options ----
option(DYNAMO_BUILD_EXAMPLES "Build examples" ON)
option(DYNAMO_BUILD_TESTS "Build tests" ON)
option(DYNAMO_BUILD_HTTP "Build HTTP frontend" ON)

# ---- Third-party dependencies ----
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# -------------------------------------------------------------------
# fmt
# -------------------------------------------------------------------
if(NOT TARGET fmt::fmt)
    add_subdirectory(third_party/fmt EXCLUDE_FROM_ALL)
endif()

# -------------------------------------------------------------------
# spdlog
# -------------------------------------------------------------------
if(NOT TARGET spdlog::spdlog)
    add_subdirectory(third_party/spdlog EXCLUDE_FROM_ALL)
endif()

# -------------------------------------------------------------------
# toml11 (header-only)
# -------------------------------------------------------------------
if(NOT TARGET toml11::toml11)
    add_library(toml11 INTERFACE)
    target_include_directories(toml11 INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/toml11)
endif()

# -------------------------------------------------------------------
# nlohmann_json (header-only)
# -------------------------------------------------------------------
if(NOT TARGET nlohmann_json::nlohmann_json)
    add_subdirectory(third_party/nlohmann_json EXCLUDE_FROM_ALL)
endif()

# -------------------------------------------------------------------
# Abseil
# -------------------------------------------------------------------
if(NOT TARGET absl::base)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(ABSL_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(ABSL_USE_EXTERNAL_GOOGLETEST OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/abseil-cpp EXCLUDE_FROM_ALL)
endif()

# -------------------------------------------------------------------
# Folly
# -------------------------------------------------------------------
if(NOT TARGET Folly::folly)
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/folly EXCLUDE_FROM_ALL)
endif()

# -------------------------------------------------------------------
# protobuf
# -------------------------------------------------------------------
if(NOT TARGET protobuf::libprotobuf)
    set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(protobuf_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/protobuf EXCLUDE_FROM_ALL)
endif()

# -------------------------------------------------------------------
# gRPC
# -------------------------------------------------------------------
if(NOT TARGET grpc++)
    set(gRPC_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(gRPC_BUILD_CSHARP_EXT OFF CACHE BOOL "" FORCE)
    set(gRPC_BUILD_GRPC_CSHARP_PLUGIN OFF CACHE BOOL "" FORCE)
    set(gRPC_BUILD_GRPC_NODE_PLUGIN OFF CACHE BOOL "" FORCE)
    set(gRPC_BUILD_GRPC_OBJECTIVE_C_PLUGIN OFF CACHE BOOL "" FORCE)
    set(gRPC_BUILD_GRPC_PHP_PLUGIN OFF CACHE BOOL "" FORCE)
    set(gRPC_BUILD_GRPC_PYTHON_PLUGIN OFF CACHE BOOL "" FORCE)
    set(gRPC_BUILD_GRPC_RUBY_PLUGIN OFF CACHE BOOL "" FORCE)
    set(gRPC_INSTALL OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/grpc EXCLUDE_FROM_ALL)
endif()

# -------------------------------------------------------------------
# asio (header-only, standalone)
# -------------------------------------------------------------------
if(NOT TARGET asio::asio)
    add_library(asio INTERFACE)
    target_include_directories(asio INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/asio/asio/include)
endif()

# -------------------------------------------------------------------
# Catch2 (for tests)
# -------------------------------------------------------------------
if(DYNAMO_BUILD_TESTS AND NOT TARGET Catch2::Catch2WithMain)
    set(CATCH_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(CATCH_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/Catch2 EXCLUDE_FROM_ALL)
endif()

# ---- Protobuf generated sources ----
set(PROTO_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/protos")
set(PROTO_BINARY_DIR "${CMAKE_BINARY_DIR}/protos-gen")

file(MAKE_DIRECTORY ${PROTO_BINARY_DIR})

find_package(Protobuf REQUIRED)
find_program(PROTOC protoc REQUIRED)
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)

set(PROTO_FILES
    ${PROTO_SRC_DIR}/dynamo/runtime.proto
    ${PROTO_SRC_DIR}/dynamo/llm.proto
)

# Generate protobuf and gRPC sources
set(PROTO_SRCS)
set(PROTO_HDRS)
set(GRPC_SRCS)
set(GRPC_HDRS)

foreach(PROTO_FILE ${PROTO_FILES})
    get_filename_component(PROTO_REL_DIR ${PROTO_FILE} DIRECTORY)
    file(RELATIVE_PATH PROTO_REL ${PROTO_SRC_DIR} ${PROTO_FILE})
    get_filename_component(PROTO_WE ${PROTO_FILE} NAME_WE)

    # protobuf sources
    list(APPEND PROTO_SRCS
        ${PROTO_BINARY_DIR}/${PROTO_REL_DIR}/${PROTO_WE}.pb.cc)
    list(APPEND PROTO_HDRS
        ${PROTO_BINARY_DIR}/${PROTO_REL_DIR}/${PROTO_WE}.pb.h)

    # gRPC sources
    list(APPEND GRPC_SRCS
        ${PROTO_BINARY_DIR}/${PROTO_REL_DIR}/${PROTO_WE}.grpc.pb.cc)
    list(APPEND GRPC_HDRS
        ${PROTO_BINARY_DIR}/${PROTO_REL_DIR}/${PROTO_WE}.grpc.pb.h)
endforeach()

add_custom_command(
    OUTPUT ${PROTO_SRCS} ${PROTO_HDRS} ${GRPC_SRCS} ${GRPC_HDRS}
    COMMAND ${PROTOC}
        ARGS --grpc_out=${PROTO_BINARY_DIR}
             --cpp_out=${PROTO_BINARY_DIR}
             --proto_path=${PROTO_SRC_DIR}
             --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
             ${PROTO_FILES}
    DEPENDS ${PROTO_FILES}
    COMMENT "Generating protobuf and gRPC sources"
)

add_library(dynamo_protos ${PROTO_SRCS} ${GRPC_SRCS})
target_include_directories(dynamo_protos
    PUBLIC ${PROTO_BINARY_DIR})
target_link_libraries(dynamo_protos
    PUBLIC protobuf::libprotobuf grpc++)
set_target_properties(dynamo_protos PROPERTIES
    POSITION_INDEPENDENT_CODE ON)

# ---- Libraries ----
add_subdirectory(lib/runtime)
add_subdirectory(lib/llm)

if(DYNAMO_BUILD_HTTP)
    add_subdirectory(lib/http)
endif()

# ---- Examples ----
if(DYNAMO_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ---- Tests ----
if(DYNAMO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
