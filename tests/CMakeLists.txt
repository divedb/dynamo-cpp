# SPDX-License-Identifier: Apache-2.0

add_executable(dynamo_tests
  coro_test.cpp
  runtime_test.cpp
  pipeline_test.cpp
  codec_test.cpp
  discovery_test.cpp
  endpoint_test.cpp
  soak_test.cpp
  llm_protocols_test.cpp
  llm_preprocessor_test.cpp
  llm_backend_test.cpp
  llm_model_card_test.cpp
  llm_http_test.cpp
  llm_kv_router_test.cpp
)
# etcd backend tests: compiled only with the backend; they SKIP themselves at
# runtime when no etcd server is reachable (DYN_TEST_ETCD, default :2379).
if(DYNAMO_WITH_ETCD)
  target_sources(dynamo_tests PRIVATE discovery_etcd_test.cpp)
endif()

target_link_libraries(dynamo_tests PRIVATE dynamo dynamo_llm dynamo_warnings Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(dynamo_tests PROPERTIES TIMEOUT 120)

# Multi-process integration test: discoveryd + server + client as separate
# processes, including SIGKILL + lease-TTL disappearance.
if(DYNAMO_BUILD_EXAMPLES AND UNIX)
  add_test(NAME multi_process_integration
    COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/integration/multi_process.sh
            $<TARGET_FILE:discoveryd>
            $<TARGET_FILE:hello_world_server>
            $<TARGET_FILE:hello_world_client>
            $<TARGET_FILE:instance_ls>)
  set_tests_properties(multi_process_integration PROPERTIES
    TIMEOUT 90
    LABELS "integration"
    RUN_SERIAL TRUE)
endif()

# Graceful-shutdown hard-exit (911) path, tested via a subprocess.
if(UNIX)
  add_executable(hang_worker helpers/hang_worker.cpp)
  target_link_libraries(hang_worker PRIVATE dynamo dynamo_warnings)
  add_test(NAME graceful_timeout_911
    COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/integration/graceful_timeout.sh
            $<TARGET_FILE:hang_worker>)
  set_tests_properties(graceful_timeout_911 PROPERTIES TIMEOUT 30 LABELS "integration")
endif()
