# SPDX-License-Identifier: Apache-2.0
#
# dynamo-cpp: a C++20 reimplementation of a Dynamo-like distributed runtime.
# See docs/architecture.md for the design.

cmake_minimum_required(VERSION 3.20)

project(dynamo_cpp
  VERSION 0.1.0
  DESCRIPTION "C++20 reimplementation of a Dynamo-like distributed runtime"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(DYNAMO_IS_TOP_LEVEL OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(DYNAMO_IS_TOP_LEVEL ON)
endif()

option(DYNAMO_BUILD_TESTS "Build dynamo-cpp tests" ${DYNAMO_IS_TOP_LEVEL})
option(DYNAMO_BUILD_EXAMPLES "Build dynamo-cpp examples" ${DYNAMO_IS_TOP_LEVEL})
option(DYNAMO_WERROR "Treat warnings as errors for dynamo-cpp targets" OFF)
option(DYNAMO_SANITIZE "Enable ASan+UBSan for dynamo-cpp targets" OFF)
option(DYNAMO_WITH_ETCD "Build the etcd discovery backend (needs etcd-cpp-apiv3)" ON)

# The etcd backend is optional: it needs the system-installed etcd-cpp-apiv3
# package (gRPC + protobuf + cpprestsdk). Auto-disabled when absent so plain
# checkouts and CI keep building without it.
if(DYNAMO_WITH_ETCD)
  # Pre-load protobuf in CONFIG mode: etcd-cpp-api's config finds Protobuf via
  # the legacy module, then gRPC's config loads the CONFIG package — the two
  # define overlapping-but-different target sets and CMake errors out.
  # Resolving Protobuf first (CONFIG) makes both find_dependency calls no-ops.
  find_package(Protobuf CONFIG QUIET)
  find_package(etcd-cpp-api CONFIG QUIET)
  if(NOT etcd-cpp-api_FOUND)
    set(DYNAMO_WITH_ETCD OFF)
    message(STATUS "etcd-cpp-apiv3 not found; building without the etcd backend")
  endif()
endif()

# Warning/sanitizer flags applied only to our own targets via this interface lib.
add_library(dynamo_warnings INTERFACE)
if(MSVC)
  target_compile_options(dynamo_warnings INTERFACE /W4)
  if(DYNAMO_WERROR)
    target_compile_options(dynamo_warnings INTERFACE /WX)
  endif()
else()
  target_compile_options(dynamo_warnings INTERFACE -Wall -Wextra -Wpedantic)
  if(DYNAMO_WERROR)
    target_compile_options(dynamo_warnings INTERFACE -Werror)
  endif()
  if(DYNAMO_SANITIZE)
    target_compile_options(dynamo_warnings INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(dynamo_warnings INTERFACE -fsanitize=address,undefined)
  endif()
endif()

add_subdirectory(third_party)
add_subdirectory(src)

if(DYNAMO_BUILD_EXAMPLES)
  add_subdirectory(src/examples)
endif()

if(DYNAMO_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

message(STATUS "dynamo-cpp configured: tests=${DYNAMO_BUILD_TESTS} examples=${DYNAMO_BUILD_EXAMPLES} werror=${DYNAMO_WERROR} sanitize=${DYNAMO_SANITIZE} etcd=${DYNAMO_WITH_ETCD}")
