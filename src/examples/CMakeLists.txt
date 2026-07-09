# SPDX-License-Identifier: Apache-2.0

function(dynamo_example name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE dynamo dynamo_warnings)
endfunction()

dynamo_example(hello_world_server hello_world/server.cpp)
dynamo_example(hello_world_client hello_world/client.cpp)
dynamo_example(discoveryd discoveryd/main.cpp)
dynamo_example(multi_instance multi_instance/main.cpp)
dynamo_example(service_metrics service_metrics/main.cpp)
dynamo_example(echo_pipeline echo_pipeline/main.cpp)
dynamo_example(instance_ls tools/instance_ls.cpp)
dynamo_example(dynamo_bench bench/main.cpp)
