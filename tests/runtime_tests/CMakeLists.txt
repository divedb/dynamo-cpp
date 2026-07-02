add_executable(runtime_tests
    config_test.cpp
    runtime_test.cpp
    cancellation_test.cpp
)
target_link_libraries(runtime_tests PRIVATE dynamo_runtime Catch2::Catch2WithMain)
