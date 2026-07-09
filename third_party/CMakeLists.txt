# SPDX-License-Identifier: Apache-2.0
#
# Aggregates the third-party submodules needed for the current milestone.
# Each dependency is added with its build knobs tuned so we only build what we use.
# Heavier deps (folly, grpc, protobuf, abseil, asio) are wired in from M3 onward.

include(FetchContent)

# Helper: fail early with an actionable message if a submodule is not checked out.
function(dynamo_require_submodule _name _sentinel)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_name}/${_sentinel}")
    message(FATAL_ERROR
      "Submodule third_party/${_name} is missing (${_sentinel} not found).\n"
      "Run: git submodule update --init --depth 1 third_party/${_name}")
  endif()
endfunction()

dynamo_require_submodule(fmt      CMakeLists.txt)
dynamo_require_submodule(spdlog   CMakeLists.txt)
dynamo_require_submodule(toml11   CMakeLists.txt)

# --- fmt -------------------------------------------------------------------
set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
set(FMT_TEST OFF CACHE BOOL "" FORCE)
set(FMT_DOC OFF CACHE BOOL "" FORCE)
add_subdirectory(fmt EXCLUDE_FROM_ALL)

# --- spdlog (use our external fmt) -----------------------------------------
set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)
set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(spdlog EXCLUDE_FROM_ALL)

# --- toml11 (header-only, INTERFACE target toml11::toml11) -----------------
set(TOML11_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(toml11 EXCLUDE_FROM_ALL)

# --- nlohmann/json (header-only) --------------------------------------------
dynamo_require_submodule(nlohmann_json CMakeLists.txt)
set(JSON_Install OFF CACHE BOOL "" FORCE)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
add_subdirectory(nlohmann_json EXCLUDE_FROM_ALL)

# --- xxHash (header-only via XXH_INLINE_ALL; used for KV block hashing) ----
dynamo_require_submodule(xxHash xxhash.h)
add_library(xxhash INTERFACE)
add_library(xxHash::xxhash ALIAS xxhash)
target_include_directories(xxhash SYSTEM INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/xxHash")
target_compile_definitions(xxhash INTERFACE XXH_INLINE_ALL)

# --- minja (header-only Jinja for HF chat templates) ------------------------
dynamo_require_submodule(minja include/minja/minja.hpp)
add_library(minja INTERFACE)
add_library(minja::minja ALIAS minja)
target_include_directories(minja SYSTEM INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/minja/include")
target_link_libraries(minja INTERFACE nlohmann_json::nlohmann_json)

# --- Catch2 (tests only) ---------------------------------------------------
if(DYNAMO_BUILD_TESTS)
  dynamo_require_submodule(Catch2 CMakeLists.txt)
  set(CATCH_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
  set(CATCH_INSTALL_EXTRAS ON CACHE BOOL "" FORCE) # provides Catch.cmake (catch_discover_tests)
  add_subdirectory(Catch2 EXCLUDE_FROM_ALL)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Catch2/extras")
  set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)
endif()
