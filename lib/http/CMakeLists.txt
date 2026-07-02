add_library(dynamo_http
    src/server.cpp
)

target_include_directories(dynamo_http
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(dynamo_http
    PUBLIC
        dynamo_runtime
        dynamo_llm
        nlohmann_json::nlohmann_json
        Folly::folly
        asio::asio
)
