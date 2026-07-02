add_executable(pipeline_tests
    context_test.cpp
    engine_test.cpp
    pipeline_node_test.cpp
)
target_link_libraries(pipeline_tests PRIVATE dynamo_runtime Catch2::Catch2WithMain)
